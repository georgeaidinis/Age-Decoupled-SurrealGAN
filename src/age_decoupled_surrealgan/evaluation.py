from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .metrics import compute_age_metrics, compute_repetition_agreement
from .utils import save_json


def save_prediction_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def evaluate_prediction_frame(frame: pd.DataFrame, n_processes: int) -> dict[str, Any]:
    return compute_age_metrics(frame, n_processes)


def aggregate_repetition_predictions(
    prediction_paths: list[Path],
    n_processes: int,
) -> dict[str, Any]:
    frames = [pd.read_csv(path) for path in prediction_paths]
    return compute_repetition_agreement(frames, n_processes)


def save_metrics(metrics: dict[str, Any], path: Path) -> None:
    save_json(path, metrics)
