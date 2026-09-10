"""
Seed module for reproducible random seed initialization across PyTorch, NumPy, Python random, and Librosa.
"""

import os
import random
import numpy as np
import torch
from utils.logger import setup_logger

logger = setup_logger(__name__)


def set_seed(seed: int = 42) -> None:
    """Sets random seeds for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # PyTorch deterministic operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Global random seed initialized to: {seed}")
