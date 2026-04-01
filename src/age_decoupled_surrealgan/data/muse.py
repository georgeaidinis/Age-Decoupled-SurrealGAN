from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


ROI_ID_PATTERN = re.compile(r"(\d+)$")


def load_muse_roi_map(json_path: str | Path) -> dict[str, dict[str, Any]]:
    path = Path(json_path).expanduser().resolve()
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected MUSE ROI mapping to be a dict, got {type(payload).__name__}")
    return payload


def extract_roi_id(feature_name: str) -> str:
    match = ROI_ID_PATTERN.search(feature_name)
    if not match:
        raise ValueError(f"Could not extract ROI ID from feature name: {feature_name}")
    return match.group(1)


def build_roi_metadata(feature_columns: list[str], roi_map: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for feature_name in feature_columns:
        roi_id = extract_roi_id(feature_name)
        info = roi_map.get(roi_id, {})
        rows.append(
            {
                "feature_name": feature_name,
                "roi_id": roi_id,
                "roi_name": info.get("Name", roi_id),
                "roi_full_name": info.get("Full_Name", info.get("Name", roi_id)),
                "muse_roi_name": info.get("MUSE_ROI_Name", ""),
                "available": info.get("Available", ""),
            }
        )
    return pd.DataFrame(rows)
