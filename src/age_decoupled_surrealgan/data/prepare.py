from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..config import ProjectConfig
from ..utils import ensure_dir, save_json
from .normalization import (
    apply_feature_normalization,
    fit_normalization_stats,
    normalization_payload_from_stats,
    save_normalization_stats,
)
from .muse import build_roi_metadata, extract_roi_id, load_muse_roi_map


NORMAL_DIAGNOSES = {
    "CN",
    "CN assumed by study criteria",
    "Cognitively normal",
    "Cognitively Normal",
    "Cognitively Unimpaired",
    "Memory Complainer (Healthy control)",
    "Memory Complainer (Healthy control) MMSE < 28",
    "Non-Memory Complainer (Healthy control)",
    "Normal",
    "Normal Cognition",
}


def coarse_diagnosis_group(value: str) -> str:
    raw = (value or "").strip()
    if not raw:
        return "unknown"
    if raw in NORMAL_DIAGNOSES:
        return "normal"
    lowered = raw.lower()
    if "mci" in lowered or "mild cognitive impairment" in lowered:
        return "mci"
    if "ad" in lowered or "alzheimer" in lowered:
        return "ad"
    if "dementia" in lowered or "dement" in lowered:
        return "dementia_other"
    if "parkinson" in lowered:
        return "parkinsonian"
    if "vascular" in lowered:
        return "vascular"
    if "frontotemporal" in lowered or "aphasia" in lowered or "cortical basal" in lowered:
        return "ftd_related"
    return "other"


def is_reference_subject(diagnosis: str, age: float, cfg: ProjectConfig) -> bool:
    return diagnosis in NORMAL_DIAGNOSES and cfg.data.ref_min_age <= age <= cfg.data.ref_max_age


def is_target_subject(age: float, cfg: ProjectConfig) -> bool:
    return cfg.data.tar_min_age <= age <= cfg.data.tar_max_age


def cohort_bucket(diagnosis: str, age: float, cfg: ProjectConfig) -> str:
    if is_reference_subject(diagnosis, age, cfg):
        return "ref"
    if is_target_subject(age, cfg):
        return "tar"
    return "excluded"


def canonicalize_dataframe(df: pd.DataFrame, cfg: ProjectConfig) -> tuple[pd.DataFrame, list[str]]:
    feature_columns = [
        col
        for col in df.columns
        if col.startswith(cfg.data.roi_prefix) and int(extract_roi_id(col)) <= cfg.data.max_atomic_roi_id
    ]
    frame = df.copy()
    frame["subject_id"] = frame[cfg.data.subject_id_column].astype(str)
    frame["study"] = frame[cfg.data.study_column].fillna("UNKNOWN").astype(str)
    frame["age"] = pd.to_numeric(frame[cfg.data.age_column], errors="coerce")
    frame["sex"] = frame[cfg.data.sex_column].fillna("U").astype(str)
    frame["diagnosis_raw"] = frame[cfg.data.diagnosis_column].fillna("").astype(str)
    frame["diagnosis_group"] = frame["diagnosis_raw"].map(coarse_diagnosis_group)
    frame["cohort_bucket"] = [
        cohort_bucket(diag, age, cfg) if not np.isnan(age) else "excluded"
        for diag, age in zip(frame["diagnosis_raw"], frame["age"])
    ]
    frame["eligible"] = frame["cohort_bucket"].isin(["ref", "tar"])
    frame["is_holdout_study"] = frame["study"] == cfg.data.holdout_study
    keep_columns = [
        "subject_id",
        "study",
        "age",
        "sex",
        "diagnosis_raw",
        "diagnosis_group",
        "cohort_bucket",
        "eligible",
        "is_holdout_study",
    ] + feature_columns
    return frame[keep_columns], feature_columns


def _subject_level_split_labels(frame: pd.DataFrame) -> pd.DataFrame:
    sort_cols = ["subject_id"]
    if "age" in frame.columns:
        sort_cols.append("age")
    sorted_frame = frame.sort_values(sort_cols, ascending=True)
    grouped = sorted_frame.groupby("subject_id", as_index=False).first()
    grouped["stratify_label"] = grouped["cohort_bucket"] + "__" + grouped["diagnosis_group"]
    return grouped[["subject_id", "stratify_label"]]


def _safe_train_test_split(
    subject_ids: pd.Series,
    stratify_labels: pd.Series,
    test_size: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    label_counts = stratify_labels.value_counts()
    can_stratify = bool(len(label_counts) > 1 and (label_counts >= 2).all())
    stratify = stratify_labels if can_stratify else None
    train_ids, test_ids = train_test_split(
        subject_ids,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return train_ids.tolist(), test_ids.tolist()


def create_split_manifest(frame: pd.DataFrame, cfg: ProjectConfig) -> dict[str, list[str]]:
    in_distribution = frame.loc[~frame["is_holdout_study"] & frame["eligible"]].copy()
    subject_table = _subject_level_split_labels(in_distribution)
    train_ids, temp_ids = _safe_train_test_split(
        subject_table["subject_id"],
        subject_table["stratify_label"],
        test_size=cfg.data.val_fraction + cfg.data.id_test_fraction,
        seed=cfg.data.split_seed,
    )
    temp_table = subject_table.loc[subject_table["subject_id"].isin(temp_ids)].copy()
    val_fraction_adjusted = cfg.data.val_fraction / (cfg.data.val_fraction + cfg.data.id_test_fraction)
    val_ids, id_test_ids = _safe_train_test_split(
        temp_table["subject_id"],
        temp_table["stratify_label"],
        test_size=1 - val_fraction_adjusted,
        seed=cfg.data.split_seed + 1,
    )
    ood_ids = sorted(frame.loc[frame["is_holdout_study"], "subject_id"].astype(str).unique().tolist())
    return {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "id_test": sorted(id_test_ids),
        "ood_test": ood_ids,
        "application": ood_ids,
    }


def _frame_for_subject_ids(frame: pd.DataFrame, subject_ids: list[str]) -> pd.DataFrame:
    return frame.loc[frame["subject_id"].isin(subject_ids)].copy()


def _split_summary(frame: pd.DataFrame) -> dict[str, Any]:
    eligible = frame.loc[frame["eligible"]]
    return {
        "rows": int(len(frame)),
        "subjects": int(frame["subject_id"].nunique()),
        "eligible_rows": int(len(eligible)),
        "eligible_subjects": int(eligible["subject_id"].nunique()),
        "cohort_counts": frame["cohort_bucket"].value_counts(dropna=False).to_dict(),
        "diagnosis_group_counts": frame["diagnosis_group"].value_counts(dropna=False).to_dict(),
        "study_counts_top10": frame["study"].value_counts(dropna=False).head(10).to_dict(),
    }


def prepare_dataset(config: ProjectConfig) -> dict[str, Any]:
    raw_path = Path(config.paths.raw_dataset)
    processed_dir = ensure_dir(Path(config.paths.processed_dir))
    df = pd.read_csv(raw_path, low_memory=False)
    canonical, feature_columns = canonicalize_dataframe(df, config)

    roi_map = load_muse_roi_map(config.paths.muse_roi_json)
    roi_metadata = build_roi_metadata(feature_columns, roi_map)
    roi_metadata.to_csv(processed_dir / "roi_metadata.csv", index=False)

    split_manifest = create_split_manifest(canonical, config)
    split_frames: dict[str, pd.DataFrame] = {}
    for split_name, subject_ids in split_manifest.items():
        split_frame = _frame_for_subject_ids(canonical, subject_ids)
        split_frames[split_name] = split_frame
        split_frame.to_csv(processed_dir / f"{split_name}.csv", index=False)

    canonical.to_csv(processed_dir / "all_rows.csv", index=False)
    canonical.loc[canonical["eligible"]].to_csv(processed_dir / "eligible_rows.csv", index=False)

    train_reference = split_frames["train"].loc[split_frames["train"]["cohort_bucket"] == "ref", feature_columns]
    normalization_stats = fit_normalization_stats(
        train_reference,
        feature_columns,
        method=config.data.roi_normalization,
        epsilon=config.data.roi_normalization_epsilon,
        clip=config.data.roi_normalization_clip,
        std_scale=config.data.roi_normalization_std_scale,
    )
    save_normalization_stats(processed_dir / "normalization_stats.csv", normalization_stats)
    normalization_payload = normalization_payload_from_stats(normalization_stats)

    reference_template_raw = train_reference.mean(axis=0).rename("value").to_frame()
    reference_template_raw.index.name = "feature_name"
    reference_template_raw.reset_index().to_csv(processed_dir / "reference_template.csv", index=False)
    reference_template_normalized_values = apply_feature_normalization(
        reference_template_raw["value"].reindex(feature_columns),
        normalization_payload,
        feature_columns,
    ).reshape(-1)
    reference_template_normalized = pd.DataFrame(
        {
            "feature_name": feature_columns,
            "value": reference_template_normalized_values,
        }
    )
    reference_template_normalized.to_csv(processed_dir / "reference_template_normalized.csv", index=False)

    manifest = {
        "config": config.as_dict(),
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "holdout_study": config.data.holdout_study,
        "roi_normalization": normalization_payload,
        "splits": {name: _split_summary(frame) for name, frame in split_frames.items()},
        "all_rows": _split_summary(canonical),
    }
    save_json(processed_dir / "split_manifest.json", manifest)
    save_json(processed_dir / "split_subject_ids.json", split_manifest)
    return manifest
