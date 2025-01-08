import numpy as np
import torch
import random


def set_global_seeds(seed: int = 42) -> None:
    """
    Set global random seeds for reproducibility.

    Args:
    seed (int): The seed value to use.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
