from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .config import ProjectConfig, load_project_config
from .model import AgeDecoupledSurrealGAN


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def load_checkpoint(checkpoint_path: str | Path, device: str = "cpu") -> tuple[AgeDecoupledSurrealGAN, dict[str, Any]]:
    checkpoint = torch.load(Path(checkpoint_path), map_location=device)
    config_payload = checkpoint.get("config")
    if isinstance(config_payload, dict):
        config = ProjectConfig(
            experiment_name=config_payload.get("experiment_name", "age-decoupled-surrealgan"),
        )
        for section_name in ["paths", "data", "model", "training", "losses", "tuning", "app"]:
            section = config_payload.get(section_name, {})
            if isinstance(section, dict):
                current = getattr(config, section_name)
                for key, value in section.items():
                    setattr(current, key, value)
    else:
        config = load_project_config()
    config.model.n_processes = int(checkpoint["n_processes"])
    model = AgeDecoupledSurrealGAN(len(checkpoint["feature_columns"]), config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint


def load_run_summary(run_dir: str | Path) -> dict[str, Any]:
    path = Path(run_dir) / "run_summary.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_dataframe(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    reference_template: pd.Series | torch.Tensor,
    device: str = "cpu",
) -> pd.DataFrame:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    features = torch.tensor(frame[feature_columns].values, dtype=torch.float32, device=run_device)
    if isinstance(reference_template, pd.Series):
        ref_values = torch.tensor(reference_template[feature_columns].values, dtype=torch.float32, device=run_device)
    else:
        ref_values = reference_template.to(run_device)
    reference = ref_values.unsqueeze(0).repeat(features.shape[0], 1)
    outputs = model.infer(features, reference)
    result = frame[["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]].copy()
    result["age_latent"] = outputs["age_latent"].cpu().numpy().reshape(-1)
    for idx in range(checkpoint["n_processes"]):
        result[f"r{idx + 1}"] = outputs["process_latents"].cpu().numpy()[:, idx]
    return result


def infer_single_row(
    checkpoint_path: str | Path,
    row: pd.Series,
    feature_columns: list[str],
    reference_template: pd.Series,
    device: str = "cpu",
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    feature_tensor = torch.tensor(row[feature_columns].values, dtype=torch.float32, device=run_device).unsqueeze(0)
    ref_tensor = torch.tensor(
        reference_template[feature_columns].values,
        dtype=torch.float32,
        device=run_device,
    ).unsqueeze(0)
    outputs = model.infer(feature_tensor, ref_tensor)
    roi_delta = outputs["synthetic_delta"].detach().cpu().numpy().reshape(-1)
    age_delta = outputs["age_delta"].detach().cpu().numpy().reshape(-1)
    process_deltas = outputs["process_deltas"].detach().cpu().numpy().reshape(model.n_processes, -1)
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "age_latent": float(outputs["age_latent"].detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": outputs["process_latents"].detach().cpu().numpy().reshape(-1).tolist(),
        "synthetic_delta": roi_delta.tolist(),
        "age_delta": age_delta.tolist(),
        "process_deltas": process_deltas.tolist(),
    }
