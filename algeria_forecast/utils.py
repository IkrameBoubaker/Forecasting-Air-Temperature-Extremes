"""Small shared utilities: global seeding, safe filenames, time formatting."""

import os
import random

import numpy as np


def set_global_seeds(seed: int) -> None:
    """Fix all random seeds (Python, NumPy, TensorFlow) for reproducibility."""
    import tensorflow as tf  # imported lazily so utils.py has no hard TF dep at import time

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def safe_name(name: str) -> str:
    """Turn a station name into a filesystem-safe string for filenames."""
    for ch in r'/\:*?"<>|':
        name = name.replace(ch, "_")
    return name.replace(" ", "_")


def fmt_time(seconds: float) -> str:
    """Human-readable duration (ms / s / min) for logging."""
    if seconds is None or (isinstance(seconds, float) and np.isnan(seconds)):
        return "—"
    if seconds < 1.0:
        return f"{seconds * 1e3:.1f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.2f} min"
