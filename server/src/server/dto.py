from pydantic import BaseModel, Field


class LoraTrainingArguments(BaseModel):
    per_device_train_batch_size: int = Field(..., description="Batch size per device")
    gradient_accumulation_steps: int = Field(
        ..., description="Gradient accumulation steps"
    )
    num_train_epochs: int = Field(..., description="Number of training epochs")
    lora_rank: int = Field(..., description="LoRA rank")
    lora_alpha: int = Field(..., description="LoRA alpha parameter")
    lora_dropout: float = Field(..., description="LoRA dropout rate")


class FineTuningArguments(BaseModel):
    warmup_steps: int = Field(100, description="Number of warmup steps")
    learning_rate: float = Field(2e-4, description="Learning rate")
    bf16: bool = Field(True, description="Use BF16 precision")
    logging_steps: int = Field(20, description="Logging steps interval")


class TaskState(BaseModel):
    total_steps: int = Field(..., description="Total number of steps for training")
    current_progress: int = Field(0, description="Current progress in steps")


class TrainingParams(BaseModel):
    model_name: str = Field(..., description="The name of the model to train")
    lora_params: LoraTrainingArguments
    fine_tuning_params: FineTuningArguments
