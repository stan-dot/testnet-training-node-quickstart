import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

import fastapi
import requests
import torch
import yaml
from datasets import Dataset
from huggingface_hub import HfApi
from loguru import logger
from peft.peft_model import PeftModel
from peft.tuners.lora import LoraConfig
from torch.cuda import get_device_name
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from server.models import ModelConfig, model_configs

FLOCK_API_KEY = os.environ.get("FLOCK_API_KEY", "empty")
FED_LEDGER_BASE_URL = "https://fed-ledger-prod.flock.io/api/v1"

HF_USERNAME = os.environ.get("HF_USERNAME", "empty")


def get_gpu_type():
    try:
        gpu_name = get_device_name(0)
        return gpu_name
    except Exception as e:
        return f"Error retrieving GPU type: {e}"


gpu_type = get_gpu_type()
tasks = {}


@asynccontextmanager
async def lifespan(app: fastapi.FastAPI):
    # Load the ML model
    yield
    os.system("rm -rf merged_model")
    os.system("rm -rf outputs")


app = fastapi.FastAPI()


@app.get("/tasks")
async def list_tasks():
    """List all tasks with their statuses."""
    return tasks


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/models", response_model=list[dict])  # noqa: F821
def list_models():
    """Route to list all available model configurations."""
    return [
        {
            "model_name": model_name,
            "size": config.size,
            "base_model": config.base_model,
            "template": {
                "system_format": config.template.system_format,
                "user_format": config.template.user_format,
                "assistant_format": config.template.assistant_format,
                "system": config.template.system,
            },
        }
        for model_name, config in model_configs.items()
    ]


def get_model_configs_below_size(size: int) -> dict[str, ModelConfig]:
    """
    Helper function to get all model configurations
    with a size less than the specified value.
    """
    return {k: v for k, v in model_configs.items() if v.size < size}


@dataclass
class LoraTrainingArguments:
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: int
    lora_rank: int
    lora_alpha: int
    lora_dropout: int


def train_lora(
    raw_args: LoraTrainingArguments,
    model_id: str = list(model_configs)[0],
    context_length: int = 1024,
    model: ModelConfig = model_configs[list(model_configs)[0]],
):
    logger.info(f"Start to train the model {model_id}...")
    lora_config = LoraConfig(
        r=raw_args.lora_rank,
        target_modules=[
            "q_proj",
            "v_proj",
        ],
        lora_alpha=raw_args.lora_alpha,
        lora_dropout=raw_args.lora_dropout,
        task_type="CAUSAL_LM",
    )

    # Load model in 4-bit to do qLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    training_args = SFTConfig(
        per_device_train_batch_size=raw_args.per_device_train_batch_size,
        gradient_accumulation_steps=raw_args.gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=20,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        num_train_epochs=raw_args.num_train_epochs,
        max_seq_length=context_length,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=os.environ["HF_TOKEN"],
    )

    # Load dataset
    dataset = SFTDataset(
        file="demo_data.jsonl",
        tokenizer=tokenizer,
        max_seq_length=context_length,
        template=model.template,
    )

    # Define trainer
    trainer = SFTTrainer(
        model=model_id,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        data_collator=SFTDataCollator(tokenizer, max_seq_length=context_length),
    )

    # Train model
    trainer.train(self=trainer)

    # save model
    trainer.save_model("outputs")

    # remove checkpoint folder
    os.system("rm -rf outputs/checkpoint-*")

    # upload lora weights and tokenizer
    print("Training Completed.")


class SFTDataset(Dataset):
    system: str = "You are a helpful assistant."

    def __init__(self, file, tokenizer, max_seq_length, template):
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

        for i in range(0, len(conversations) - 1, 2):
            if (
                conversations[i]["role"] != "user"
                or conversations[i + 1]["role"] != "assistant"
            ):
                raise ValueError("The role order of the conversation is not correct")
            human = conversations[i]["content"].strip()
            assistant = conversations[i + 1]["content"].strip()

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

        labels = torch.where(target_mask_batch == 1, input_ids_batch, -100)
        inputs = {
            "input_ids": input_ids_batch,
            "attention_mask": attention_mask_batch,
            "labels": labels,
        }
        return inputs


def get_active_decentralized_network_task(task_id: int):
    response = requests.request(
        "GET", f"{FED_LEDGER_BASE_URL}/tasks/get?task_id={task_id}"
    )
    return response.json()


# Dataclass for Task Submission Parameters
@dataclass
class TaskSubmission:
    task_id: int
    hg_repo_id: str
    base_model: str
    gpu_type: str
    commit_hash: str
    revision: int


def submit_task_result_to_decentralized_network(submission: TaskSubmission):
    payload = json.dumps(
        {
            "task_id": submission.task_id,
            "data": {
                "hg_repo_id": submission.hg_repo_id,
                "base_model": submission.base_model,
                "gpu_type": submission.gpu_type,
                "revision": submission.revision,
            },
        }
    )
    headers = {
        "flock-api-key": FLOCK_API_KEY,
        "Content-Type": "application/json",
    }

    try:
        response = requests.post(
            f"{FED_LEDGER_BASE_URL}/tasks/submit-result",
            headers=headers,
            data=payload,
        )
        response.raise_for_status()  # Raise an error for 4xx/5xx status codes
        return response.json()

    except requests.HTTPError as e:
        error_msg = f"Failed to submit task (HTTP error): {e.response.text}"
        print(error_msg)
        raise Exception(error_msg) from e
    except requests.RequestException as e:
        error_msg = f"Request error while submitting task: {e}"
        print(error_msg)
        raise Exception(error_msg) from e


def merge_lora_to_base_model(
    model_name_or_path: str, adapter_name_or_path: str, save_path: str
):
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_name_or_path,
        use_fast=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )
    model = PeftModel.from_pretrained(
        model, adapter_name_or_path, device_map={"": "cpu"}
    )
    model = model.merge_and_unload()

    tokenizer.save_pretrained(save_path)
    model.save_pretrained(save_path)


@app.route("/train", methods=["POST"])
def train():
    # task_id = os.environ["TASK_ID"]
    task_id = 55
    # load trainin args
    # define the path of the current file
    current_folder = os.path.dirname(os.path.realpath(__file__))
    with open(f"{current_folder}/training_args.yaml") as f:
        all_training_args = yaml.safe_load(f)

    task = get_active_decentralized_network_task(task_id)
    # log the task info
    logger.info(json.dumps(task, indent=4))
    # download data from a presigned url
    data_url = task["data"]["training_set_url"]
    context_length = task["data"]["context_length"]
    max_params = task["data"]["max_params"]

    valid_models = get_model_configs_below_size(max_params)
    logger.info(f"Models within the max_params: {valid_models}")
    # download in chunks
    response = requests.get(data_url, stream=True)
    with open("demo_data.jsonl", "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # for testing now
    model_id = list(valid_models.keys())[0]
    model_config = valid_models[model_id]

    # train all feasible models and merge
    # if OOM, proceed to the next model
    try:
        train_lora(
            context_length=context_length,
            raw_args=LoraTrainingArguments(**all_training_args[model_id]),
        )
    except RuntimeError as e:
        logger.error(f"Error: {e}")
        logger.info("Proceed to the next model...")

    # generate a random repo id based on timestamp

    try:
        logger.info("Start to push the lora weight to the hub...")
        api = HfApi(token=os.environ["HF_TOKEN"])
        repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
        # check whether the repo exists
        try:
            api.create_repo(
                repo_name,
                exist_ok=False,
                repo_type="model",
            )
        except Exception as e:
            logger.info(
                f"Repo {repo_name} already exists. Will commit the new version."
            )

        commit_message = api.upload_folder(
            folder_path="outputs",
            repo_id=repo_name,
            repo_type="model",
        )
        # get commit hash
        commit_hash = commit_message.oid
        logger.info(f"Commit hash: {commit_hash}")
        logger.info(f"Repo name: {repo_name}")
        # submit
        submission = TaskSubmission(
            int(task_id),
            repo_name,
            model_config.base_model,
            gpu_type,
            commit_hash,
            revision=4,
        )
        submit_task_result_to_decentralized_network(submission)
        logger.info("Task submitted successfully")
    except Exception as e:
        logger.error(f"Error: {e}")
