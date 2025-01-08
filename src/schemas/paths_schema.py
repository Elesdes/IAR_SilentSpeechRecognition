from dataclasses import dataclass
import os
from datetime import datetime


@dataclass
class PathsSchema:
    root: str = os.path.abspath(".")
    data: str = os.path.join(root, "data")
    tensorboard: str = os.path.join(root, "tensorboard")
    models: str = os.path.join(root, "models")

    model_name: str = ""

    model_path: str = ""
    tensorboard_log_model: str = ""

    def __post_init__(self):
        assert self.model_name, "Please provide a model name."

        self.model_path = (
            os.path.join(self.models, self.model_name) if self.model_name else ""
        )
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)

        self.tensorboard_log_model = self._rename_directory_if_exists(
            os.path.join(self.tensorboard, self.model_name) if self.model_name else ""
        )

    def _rename_directory_if_exists(self, directory_path: str) -> str:
        if os.path.exists(directory_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_directory_path = f"{directory_path}_{timestamp}"
            os.rename(directory_path, new_directory_path)
            print(f"Renamed directory: {directory_path} -> {new_directory_path}")
            return new_directory_path
        return directory_path
