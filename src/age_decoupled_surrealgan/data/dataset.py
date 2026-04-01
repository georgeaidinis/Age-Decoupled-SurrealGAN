from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd
import torch
from torch.utils.data import Dataset

from ..config import ProjectConfig


@dataclass
class SplitDataBundle:
    frame: pd.DataFrame
    feature_columns: list[str]
    age_min: float
    age_max: float


def normalize_age(age: torch.Tensor, age_min: float, age_max: float) -> torch.Tensor:
    scale = max(age_max - age_min, 1.0)
    return (age - age_min) / scale


class CohortDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, feature_columns: list[str], age_min: float, age_max: float):
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.age_min = age_min
        self.age_max = age_max
        self.features = torch.tensor(self.frame[self.feature_columns].values, dtype=torch.float32)
        self.ages = torch.tensor(self.frame["age"].fillna(age_min).values, dtype=torch.float32)
        self.subject_ids = self.frame["subject_id"].astype(str).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        age_value = self.ages[index]
        return {
            "features": self.features[index],
            "age": age_value,
            "age_norm": normalize_age(age_value.unsqueeze(0), self.age_min, self.age_max).squeeze(0),
            "subject_id": self.subject_ids[index],
        }


def load_split_frame(config: ProjectConfig, split_name: str) -> pd.DataFrame:
    processed_dir = Path(config.paths.processed_dir)
    return pd.read_csv(processed_dir / f"{split_name}.csv")


def load_feature_columns(config: ProjectConfig) -> list[str]:
    manifest_path = Path(config.paths.processed_dir) / "split_manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    feature_columns = manifest.get("feature_columns")
    if not isinstance(feature_columns, list):
        raise ValueError("split_manifest.json does not contain feature_columns")
    return feature_columns
