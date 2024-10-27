import json
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from loguru import logger


class SFTDataset(Dataset):
    system: str = "You're a great assistant"

    def __init__(self, file: Path, tokenizer, max_seq_length: int, template):
        self.tokenizer = tokenizer
        self.system_format = template["system_format"]
        self.user_format = template["user_format"]
        self.assistant_format = template["assistant_format"]

        self.max_seq_length = max_seq_length
        logger.info(f"Loading data: {file}")
        with open(file, encoding="utf8") as f:
            data_list = f.readlines()
        logger.info(f"There are {len(data_list)} data in dataset")
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        data = self.data_list[index]
        data = json.loads(data)
        input_ids, target_mask = [], []

        # setting system information
        if self.system_format is not None:
            system = data["system"].strip() if "system" in data.keys() else self.system

            if system is not None:
                system_text = self.system_format.format(content=system)
                input_ids = self.tokenizer.encode(system_text, add_special_tokens=False)
                target_mask = [0] * len(input_ids)

        conversations = data["conversations"]

        for user_msg, assistant_msg in zip(
            conversations[::2], conversations[1::2], strict=False
        ):
            if user_msg["role"] != "user" or assistant_msg["role"] != "assistant":
                raise ValueError("The role order of the conversation is not correct")

            human = user_msg["content"].strip()
            assistant = assistant_msg["content"].strip()

            human = self.user_format.format(
                content=human, stop_token=self.tokenizer.eos_token
            )
            assistant = self.assistant_format.format(
                content=assistant, stop_token=self.tokenizer.eos_token
            )

            input_tokens = self.tokenizer.encode(human, add_special_tokens=False)
            output_tokens = self.tokenizer.encode(assistant, add_special_tokens=False)

            input_ids += input_tokens + output_tokens
            target_mask += [0] * len(input_tokens) + [1] * len(output_tokens)

        assert len(input_ids) == len(target_mask)

        input_ids = input_ids[: self.max_seq_length]
        target_mask = target_mask[: self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "target_mask": target_mask,
        }
        return inputs


class SFTDataCollator:
    def __init__(self, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        # Find the maximum length in the batch
        lengths = [len(x["input_ids"]) for x in batch if x["input_ids"] is not None]
        # Take the maximum length in the batch
        # if it exceeds max_seq_length, take max_seq_length
        batch_max_len = min(max(lengths), self.max_seq_length)

        input_ids_batch, attention_mask_batch, target_mask_batch = (
            self._get_batch_values(batch, batch_max_len)
        )

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }
        return inputs

    def _get_batch_values(
        self, batch: list[dict[str, Any]], batch_max_len: int
    ) -> list[torch.Tensor]:
        input_ids_batch, attention_mask_batch, target_mask_batch = [], [], []
        # Truncate and pad
        for x in batch:
            input_ids = x["input_ids"]
            attention_mask = x["attention_mask"]
            target_mask = x["target_mask"]
            if input_ids is None:
                logger.info("some input_ids is None")
                continue
            padding_len = batch_max_len - len(input_ids)
            # Pad
            input_ids = input_ids + [self.pad_token_id] * padding_len
            attention_mask = attention_mask + [0] * padding_len
            target_mask = target_mask + [0] * padding_len
            # Truncate
            input_ids = input_ids[: self.max_seq_length]
            attention_mask = attention_mask[: self.max_seq_length]
            target_mask = target_mask[: self.max_seq_length]

            input_ids_batch.append(input_ids)
            attention_mask_batch.append(attention_mask)
            target_mask_batch.append(target_mask)

            # Convert lists to tensors to get the final model input
        input_ids_batch = torch.tensor(input_ids_batch, dtype=torch.long)
        attention_mask_batch = torch.tensor(attention_mask_batch, dtype=torch.long)
        target_mask_batch = torch.tensor(target_mask_batch, dtype=torch.long)

        return [input_ids_batch, attention_mask_batch, target_mask_batch]
