from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def fit_normalization_stats(
    frame: pd.DataFrame,
    feature_columns: list[str],
    *,
    method: str = "zscore",
    epsilon: float = 1.0e-6,
    clip: float | None = None,
    std_scale: float = 1.0,
) -> pd.DataFrame:
    if method not in {"none", "zscore", "surreal"}:
        raise ValueError(f"Unsupported normalization method: {method}")
    values = frame[feature_columns].apply(pd.to_numeric, errors="coerce")
    mean = values.mean(axis=0)
    std = values.std(axis=0).replace(0.0, np.nan).fillna(1.0)
    scale = std * float(std_scale)
    scale = scale.clip(lower=float(epsilon))
    stats = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "mean": mean.reindex(feature_columns).to_numpy(dtype=float),
            "std": std.reindex(feature_columns).to_numpy(dtype=float),
            "scale": scale.reindex(feature_columns).to_numpy(dtype=float),
            "method": method,
            "epsilon": float(epsilon),
            "clip": float(clip) if clip is not None else np.nan,
            "std_scale": float(std_scale),
        }
    )
    return stats


def save_normalization_stats(path: Path, stats: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats.to_csv(path, index=False)


def load_normalization_stats(path: str | Path) -> pd.DataFrame:
    stats = pd.read_csv(path)
    if "feature_name" not in stats.columns:
        raise ValueError("Normalization stats missing feature_name column.")
    return stats


def normalization_payload_from_stats(stats: pd.DataFrame) -> dict[str, Any]:
    feature_names = stats["feature_name"].astype(str).tolist()
    return {
        "feature_columns": feature_names,
        "mean": stats.set_index("feature_name")["mean"].reindex(feature_names).to_dict(),
        "std": stats.set_index("feature_name")["std"].reindex(feature_names).to_dict(),
        "scale": stats.set_index("feature_name")["scale"].reindex(feature_names).to_dict(),
        "method": str(stats["method"].iloc[0]) if "method" in stats.columns and not stats.empty else "none",
        "epsilon": float(stats["epsilon"].iloc[0]) if "epsilon" in stats.columns and not stats.empty else 1.0e-6,
        "clip": (
            None
            if "clip" not in stats.columns or stats.empty or pd.isna(stats["clip"].iloc[0])
            else float(stats["clip"].iloc[0])
        ),
        "std_scale": float(stats["std_scale"].iloc[0]) if "std_scale" in stats.columns and not stats.empty else 1.0,
    }


def _ordered_stats(payload: dict[str, Any], feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray, str, float | None]:
    mean_map = payload.get("mean", {})
    scale_map = payload.get("scale", {})
    method = str(payload.get("method", "none"))
    clip = payload.get("clip")
    clip_value = None if clip is None else float(clip)
    means = np.asarray([float(mean_map.get(name, 0.0)) for name in feature_columns], dtype=np.float32)
    scales = np.asarray([max(float(scale_map.get(name, 1.0)), 1.0e-6) for name in feature_columns], dtype=np.float32)
    return means, scales, method, clip_value


def apply_feature_normalization(
    values: np.ndarray | pd.DataFrame | pd.Series,
    payload: dict[str, Any],
    feature_columns: list[str],
) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        raw = values.reindex(columns=feature_columns).to_numpy(dtype=np.float32)
    elif isinstance(values, pd.Series):
        raw = values.reindex(feature_columns).to_numpy(dtype=np.float32).reshape(1, -1)
    else:
        raw = np.asarray(values, dtype=np.float32)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
    means, scales, method, clip = _ordered_stats(payload, feature_columns)
    if method == "none":
        normalized = raw.copy()
    elif method in {"zscore", "surreal"}:
        normalized = (raw - means) / scales
        if method == "surreal":
            normalized = 1.0 + normalized
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    if clip is not None:
        normalized = np.clip(normalized, -clip, clip)
    return normalized.astype(np.float32, copy=False)


def invert_feature_normalization(
    normalized_values: np.ndarray,
    payload: dict[str, Any],
    feature_columns: list[str],
) -> np.ndarray:
    normalized = np.asarray(normalized_values, dtype=np.float32)
    if normalized.ndim == 1:
        normalized = normalized.reshape(1, -1)
    means, scales, method, _ = _ordered_stats(payload, feature_columns)
    if method == "none":
        raw = normalized
    elif method == "zscore":
        raw = normalized * scales + means
    elif method == "surreal":
        raw = (normalized - 1.0) * scales + means
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return raw.astype(np.float32, copy=False)


def scale_delta_to_raw(
    normalized_delta: np.ndarray,
    payload: dict[str, Any],
    feature_columns: list[str],
) -> np.ndarray:
    delta = np.asarray(normalized_delta, dtype=np.float32)
    if delta.ndim == 1:
        delta = delta.reshape(1, -1)
    _, scales, method, _ = _ordered_stats(payload, feature_columns)
    if method == "none":
        raw = delta
    elif method == "zscore":
        raw = delta * scales
    elif method == "surreal":
        raw = delta * scales
    else:
        raise ValueError(f"Unsupported normalization method: {method}")
    return raw.astype(np.float32, copy=False)
