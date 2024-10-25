import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import requests
import yaml
from demo import LoraTrainingArguments, train_lora
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi
from loguru import logger
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

# Configurations and initial setup
HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")
DATA_PATH = "../../data/demo_data.jsonl"
# Server config model
@dataclass
class MyServerConfig:
    hf_username: str = HF_USERNAME


@asynccontextmanager
async def lifespan(app: FastAPI):
    # todo here starting tasks
    yield
    """Perform any cleanup tasks on shutdown."""
    os.system("rm -rf merged_model outputs")
    print("Cleanup completed successfully.")

app = FastAPI(lifespan=lifespan)
# Routes
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/tasks/{task_id}")
async def run_task(task_id: str):
    """Fetches the task and processes training based on task configuration."""
    try:
        # Load training arguments
        current_folder = os.path.dirname(os.path.realpath(__file__))
        # todo parametrize training args
        with open(f"{current_folder}/training_args.yaml") as f:
            all_training_args = yaml.safe_load(f)

        # Fetch task details
        task = get_task(int(task_id))
        logger.info(json.dumps(task, indent=4))
        data_url = task["data"]["training_set_url"]
        context_length = task["data"]["context_length"]
        max_params = task["data"]["max_params"]

        # Filter models by max_params
        # todo this relates to theconstants import
        filtered_models = {k: v for k, v in model2size.items() if v <= max_params}
        filtered_args = {k: v for k, v in all_training_args.items() if k in filtered_models}

        logger.info(f"Models within the max_params: {filtered_args.keys()}")
        # Download the data file
        # todo add a try block
        response = requests.get(data_url, stream=True)
        CHUNK_SIZE = 8192
        with open(DATA_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)

        # Train and upload models
        for model_id in filtered_args.keys():
            logger.info(f"Start to train the model {model_id}...")
            try:
                train_lora(
                    model_id=model_id,
                    context_length=context_length,
                    training_args=LoraTrainingArguments(**filtered_args[model_id]),
                )
                # Upload to Hugging Face Hub
                repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
                api = HfApi(token=HF_TOKEN)
                try:
                    logger.info(f"Start to train the model {model_id}...")
                    api.create_repo(repo_name, exist_ok=True, repo_type="model")
                except Exception as e:
                    logger.error(f"Repo {repo_name} already exists, adding new commit.")

                commit_message = api.upload_folder(
                    folder_path="outputs",
                    repo_id=repo_name,
                    repo_type="model",
                )

                commit_hash = commit_message.oid
                gpu_type = get_gpu_type()
                submit_task(int(task_id), repo_name, model2base_model[model_id], gpu_type, commit_hash)
            except RuntimeError as e:
                logger.error(f"Training error for {model_id}: {e}")
                continue
            finally:
                # Cleanup
                os.system("rm -rf merged_model outputs")
        return {"status": "completed", "task_id": task_id}

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
