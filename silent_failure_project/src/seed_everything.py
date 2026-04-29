"""
src/seed_everything.py
Call seed_everything() at the top of any script or notebook cell
that does data splitting, model init, or stochastic inference.
"""

import random
import numpy as np


def seed_everything(seed: int = 42) -> None:
    """Set all random seeds for full reproducibility.

    Args:
        seed: Integer seed value. Default 42.
    """
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    except ImportError:
        pass
