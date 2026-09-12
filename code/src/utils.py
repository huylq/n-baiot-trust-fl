from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_directories(config: dict) -> None:
    for key, path_str in config["paths"].items():
        if key.endswith("_dir") or key.endswith("_root"):
            Path(path_str).mkdir(parents=True, exist_ok=True)


def timestamp_string() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_shifted = np.exp(shifted)
    return exp_shifted / exp_shifted.sum(axis=1, keepdims=True)


def l2_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.sum(np.square(values))))
