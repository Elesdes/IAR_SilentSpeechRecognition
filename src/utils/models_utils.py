import os
import torch
from typing import Any
import re


def load_model(model_path: str) -> Any:
    """
    Load a PyTorch model from a .pth or .pt file.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Any: Loaded PyTorch model object.

    Raises:
        ValueError: If the file doesn't exist or the format is unsupported.
    """
    if not os.path.exists(model_path):
        raise ValueError(f"Model file not found: {model_path}")

    file_extension = os.path.splitext(model_path)[1].lower()

    if file_extension in {".pth", ".pt"}:
        try:
            model = torch.load(model_path)
            print(f"Loaded PyTorch model from {model_path}")
            return model
        except Exception as e:
            raise ValueError(f"Error loading PyTorch model: {str(e)}")
    else:
        raise ValueError(f"Unsupported model format: {file_extension}")
