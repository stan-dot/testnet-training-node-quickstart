from dataclasses import dataclass


# Define dataclass for the template structure
@dataclass
class ModelTemplate:
    system_format: str
    user_format: str
    assistant_format: str
    system: str | None = None


# Initialize template data
qwen_template = ModelTemplate(
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system="You are a helpful assistant.",
)

gemma_template = ModelTemplate(
    system_format="<bos>",
    user_format="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    assistant_format="{content}<eos>\n",
    system=None,
)

# Mapping of models to templates
model2template: dict[str, ModelTemplate] = {
    "Qwen/Qwen1.5-0.5B": qwen_template,
    "Qwen/Qwen1.5-1.8B": qwen_template,
    "Qwen/Qwen1.5-7B": qwen_template,
    "google/gemma-2b": gemma_template,
    "google/gemma-7b": gemma_template,
}


# Dataclass for model size information
@dataclass
class ModelSize:
    size: int


# Model size mappings
model2size: dict[str, ModelSize] = {
    "Qwen/Qwen1.5-0.5B": ModelSize(size=620_000_000),
    "Qwen/Qwen1.5-1.8B": ModelSize(size=1_840_000_000),
    "Qwen/Qwen1.5-7B": ModelSize(size=7_720_000_000),
    "google/gemma-2b": ModelSize(size=2_510_000_000),
    "google/gemma-7b": ModelSize(size=8_540_000_000),
}


# Dataclass for base model information
@dataclass
class BaseModel:
    name: str


# Model base mappings
model2base_model: dict[str, BaseModel] = {
    "Qwen/Qwen1.5-0.5B": BaseModel(name="qwen1.5"),
    "Qwen/Qwen1.5-1.8B": BaseModel(name="qwen1.5"),
    "Qwen/Qwen1.5-7B": BaseModel(name="qwen1.5"),
    "google/gemma-2b": BaseModel(name="gemma"),
    "google/gemma-7b": BaseModel(name="gemma"),
}
