import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import requests
import yaml
from demo import LoraTrainingArguments, create_learning_task
from fastapi import FastAPI, HTTPException
from huggingface_hub import HfApi
from loguru import logger
from utils.constants import model2base_model, model2size
from utils.flock_api import get_task, submit_task
from utils.gpu_utils import get_gpu_type

from server.facades.hf import HuggingFaceFacade

# Configurations and initial setup
DATA_PATH = "../../data/demo_data.jsonl"


# Server config model
@dataclass
class MyServerConfig:
    gpu_type = get_gpu_type()


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    """Perform any cleanup tasks on shutdown."""
    os.system("rm -rf merged_model outputs")
    print("Cleanup completed successfully.")


app = FastAPI(lifespan=lifespan)

hf_client = HuggingFaceFacade()


# Routes
@app.get("/")
async def root():
    return {"message": "Hello World"}

tasks = {}

@app.get("/tasks/{task_id}")
async def create_training_run(all_training_args: LoraTrainingArguments):
    """Fetches the task and processes training based on task configuration."""
    try:
        # Fetch task details
        tasks["task1"] = {}

        data_url = task["data"]["training_set_url"]
        context_length = task["data"]["context_length"]
        max_params = task["data"]["max_params"]

        # Filter models by max_params
        # todo this relates to theconstants import
        filtered_args = filter_models(all_training_args, max_params)
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
                create_learning_task(
                    model_id=model_id,
                    context_length=context_length,
                    training_args=LoraTrainingArguments(**filtered_args[model_id]),
                )
                commit_hash = commit_message.oid
                submit_task(
                    int(task_id),
                    hf_client.repo_name,
                    model2base_model[model_id].name,
                    gpu_type=get_gpu_type(),
                    commit_hash,
                    revision=4
                )
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


def filter_models(all_training_args, max_params):
    filtered_models = {k: v for k, v in model2size.items() if v <= max_params}
    filtered_args = {k: v for k, v in all_training_args.items() if k in filtered_models}

    logger.info(f"Models within the max_params: {filtered_args.keys()}")
    return filtered_args


@app.post("/run_task")
def run_defined_task(task_id:str):
    try:
        # Fetch task details
        task = get_task(int(task_id))
        logger.info(json.dumps(task, indent=4))

    except Exception as e:
        logger.error(f"Error processing task {task_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
