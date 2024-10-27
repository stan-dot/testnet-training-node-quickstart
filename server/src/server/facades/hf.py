import os

from huggingface_hub import CommitInfo, HfApi
from loguru import logger

HF_USERNAME = os.environ["HF_USERNAME"]
HF_TOKEN = os.environ.get("HF_TOKEN", "")


class HuggingFaceFacade:
    api = HfApi(token=HF_TOKEN)

    def upload(self) -> CommitInfo:
        assert self.repo_name is not None, "You need to create repo first!"

        commit_message = self.api.upload_folder(
            folder_path="outputs",
            repo_id=self.repo_name,
            repo_type="model",
        )

        return commit_message

    def create_repo(self, model_id: str, task_id: str):
        print("todo")

        # Upload to Hugging Face Hub
        repo_name = f"{HF_USERNAME}/task-{task_id}-{model_id.replace('/', '-')}"
        self.repo_name = repo_name
        try:
            logger.info(f"Start to train the model {model_id}...")
            self.api.create_repo(repo_name, exist_ok=True, repo_type="model")
        except Exception:
            logger.error(f"Repo {repo_name} already exists, adding new commit.")
