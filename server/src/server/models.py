from dataclasses import dataclass


# Dataclass for Template Configuration
@dataclass
class TemplateConfig:
    system_format: str
    user_format: str
    assistant_format: str
    system: str


# Dataclass for Model Configuration
@dataclass
class ModelConfig:
    template: TemplateConfig
    size: int
    base_model: str


# Template configurations
qwen_template = TemplateConfig(
    system_format="<|im_start|>system\n{content}<|im_end|>\n",
    user_format="<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n",
    assistant_format="{content}<|im_end|>\n",
    system="You are a helpful assistant.",
)

gemma_template = TemplateConfig(
    system_format="<bos>",
    user_format="<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n",
    assistant_format="{content}<eos>\n",
    system="The model is a helpful assistant.",
)

# Consolidated model configurations
model_configs: dict[str, ModelConfig] = {
    "Qwen/Qwen1.5-0.5B": ModelConfig(
        template=qwen_template, size=620_000_000, base_model="qwen1.5"
    ),
    "Qwen/Qwen1.5-1.8B": ModelConfig(
        template=qwen_template, size=1_840_000_000, base_model="qwen1.5"
    ),
    "Qwen/Qwen1.5-7B": ModelConfig(
        template=qwen_template, size=7_720_000_000, base_model="qwen1.5"
    ),
    "google/gemma-2b": ModelConfig(
        template=gemma_template, size=2_510_000_000, base_model="gemma"
    ),
    "google/gemma-7b": ModelConfig(
        template=gemma_template, size=8_540_000_000, base_model="gemma"
    ),
}
