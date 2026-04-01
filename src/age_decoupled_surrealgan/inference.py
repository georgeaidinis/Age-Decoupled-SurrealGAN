from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
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
    run_device = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=run_device)
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
    model.to(run_device)
    model.eval()
    return model, checkpoint


def _numeric_feature_series(row: pd.Series, feature_columns: list[str]) -> pd.Series:
    feature_values = pd.to_numeric(row.reindex(feature_columns), errors="coerce")
    if feature_values.isna().any():
        missing_columns = feature_values.index[feature_values.isna()].tolist()
        preview = ", ".join(missing_columns[:10])
        suffix = "..." if len(missing_columns) > 10 else ""
        raise ValueError(f"Non-numeric or missing ROI values for columns: {preview}{suffix}")
    return feature_values


def normalize_age_years(age_years: float, config: ProjectConfig) -> float:
    age_min = float(config.data.age_latent_normalization_min)
    age_max = float(config.data.age_latent_normalization_max)
    clipped = min(max(float(age_years), age_min), max(age_min, age_max))
    if age_max <= age_min:
        return 0.0
    return float((clipped - age_min) / (age_max - age_min))


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
    feature_values = _numeric_feature_series(row, feature_columns)
    feature_tensor = torch.tensor(feature_values.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    ref_tensor = torch.tensor(
        reference_template.reindex(feature_columns).to_numpy(dtype="float32"),
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


def infer_subject_defaults(
    checkpoint_path: str | Path,
    row: pd.Series,
    feature_columns: list[str],
    device: str = "cpu",
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    feature_values = _numeric_feature_series(row, feature_columns)
    feature_tensor = torch.tensor(feature_values.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    age_latent, process_latents = model.encode(feature_tensor)
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "age_latent": float(age_latent.detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": process_latents.detach().cpu().numpy().reshape(-1).tolist(),
    }


def generate_single_row(
    checkpoint_path: str | Path,
    row: pd.Series,
    feature_columns: list[str],
    age_years: float | None,
    process_latents: list[float] | None,
    device: str = "cpu",
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    feature_values = _numeric_feature_series(row, feature_columns)
    baseline = torch.tensor(feature_values.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    reference_template = pd.Series(checkpoint["reference_template"]).reindex(feature_columns)
    reference_tensor = torch.tensor(reference_template.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    config_payload = checkpoint.get("config", {})
    config = ProjectConfig()
    if isinstance(config_payload, dict):
        for section_name in ["paths", "data", "model", "training", "losses", "tuning", "app"]:
            section = config_payload.get(section_name, {})
            if isinstance(section, dict):
                current = getattr(config, section_name)
                for key, value in section.items():
                    setattr(current, key, value)

    default_age_latent, default_process_latents = model.encode(baseline)
    if age_years is None:
        requested_age_years = float(row.get("age", config.data.age_latent_normalization_min))
        age_latent_tensor = default_age_latent
    else:
        requested_age_years = float(age_years)
        age_latent_value = normalize_age_years(requested_age_years, config)
        age_latent_tensor = torch.full_like(default_age_latent, float(age_latent_value))

    if process_latents is None:
        process_latent_tensor = default_process_latents
    else:
        if len(process_latents) != int(checkpoint["n_processes"]):
            raise ValueError(
                f"Expected {int(checkpoint['n_processes'])} process latents, received {len(process_latents)}."
            )
        process_latent_tensor = torch.tensor([process_latents], dtype=torch.float32, device=run_device)

    with torch.no_grad():
        # Direct generator response is preserved for debugging, but the interactive control path
        # is anchored at the selected subject so default slider values imply zero change.
        direct_target, direct_delta = model.synthesize(baseline, age_latent_tensor, process_latent_tensor)
        _, subject_fake_delta = model.synthesize(reference_tensor, default_age_latent, default_process_latents)
        age_basis, process_basis = model.decompose(subject_fake_delta)
        latent_age_shift = age_latent_tensor - default_age_latent
        latent_process_shift = process_latent_tensor - default_process_latents
        anchored_delta = latent_age_shift * age_basis + (
            latent_process_shift.unsqueeze(-1) * process_basis
        ).sum(dim=1)
        synthetic_target = baseline + anchored_delta

    baseline_values = feature_values.to_numpy(dtype="float32")
    target_values = synthetic_target.detach().cpu().numpy().reshape(-1)
    raw_delta = target_values - baseline_values
    baseline_abs = np.abs(baseline_values)
    percent_change = np.zeros_like(raw_delta, dtype="float32")
    valid_mask = baseline_abs > 0
    percent_change[valid_mask] = (raw_delta[valid_mask] / baseline_abs[valid_mask]) * 100.0
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "requested_age_years": requested_age_years,
        "age_latent": float(age_latent_tensor.detach().cpu().numpy().reshape(-1)[0]),
        "default_age_latent": float(default_age_latent.detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": process_latent_tensor.detach().cpu().numpy().reshape(-1).tolist(),
        "default_process_latents": default_process_latents.detach().cpu().numpy().reshape(-1).tolist(),
        "baseline_target": baseline_values.tolist(),
        "synthetic_target": target_values.tolist(),
        "synthetic_delta": raw_delta.tolist(),
        "percent_change": percent_change.tolist(),
        "age_delta": age_basis.detach().cpu().numpy().reshape(-1).tolist(),
        "process_deltas": process_basis.detach().cpu().numpy().reshape(model.n_processes, -1).tolist(),
        "debug": {
            "generation_mode": "anchored_component_linear",
            "direct_delta_abs_max": float(direct_delta.detach().abs().max().cpu()),
            "direct_delta_abs_mean": float(direct_delta.detach().abs().mean().cpu()),
            "anchored_delta_abs_max": float(anchored_delta.detach().abs().max().cpu()),
            "anchored_delta_abs_mean": float(anchored_delta.detach().abs().mean().cpu()),
        },
    }
