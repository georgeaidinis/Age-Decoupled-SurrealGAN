from __future__ import annotations

import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cycle(iterable):
    while True:
        for item in iterable:
            yield item
