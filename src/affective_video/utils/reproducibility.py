"""Seed repeatable experiments without promising cross-platform bitwise identity."""

import random

import numpy as np
import torch


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch; deterministic mode raises on unsupported ops.

    Independently constructed NumPy generators must receive their own explicit seed.
    PYTHONHASHSEED and CUDA CUBLAS_WORKSPACE_CONFIG must be set before process/CUDA
    initialization when needed; this function cannot retroactively guarantee them.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32 - 1]")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
