from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from .config import ProjectConfig, load_project_config
from .data.normalization import apply_feature_normalization, invert_feature_normalization, scale_delta_to_raw
from .legacy_model import LegacyAgeDecoupledSurrealGAN
from .model import AgeDecoupledSurrealGAN


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_name)


def _build_config_from_payload(config_payload: dict[str, Any] | None) -> ProjectConfig:
    if not isinstance(config_payload, dict):
        return load_project_config()
    config = ProjectConfig(
        experiment_name=config_payload.get("experiment_name", "age-decoupled-surrealgan"),
    )
    for section_name in ["paths", "data", "model", "training", "losses", "tuning", "app"]:
        section = config_payload.get(section_name, {})
        if isinstance(section, dict):
            current = getattr(config, section_name)
            for key, value in section.items():
                setattr(current, key, value)
    return config


def checkpoint_model_version(checkpoint: dict[str, Any]) -> str:
    return str(checkpoint.get("model_version", "v1_legacy"))


def checkpoint_normalization_payload(checkpoint: dict[str, Any]) -> dict[str, Any]:
    payload = checkpoint.get("normalization")
    feature_columns = checkpoint.get("feature_columns", [])
    if isinstance(payload, dict):
        return payload
    return {
        "feature_columns": feature_columns,
        "mean": {name: 0.0 for name in feature_columns},
        "std": {name: 1.0 for name in feature_columns},
        "scale": {name: 1.0 for name in feature_columns},
        "method": "none",
        "epsilon": 1.0e-6,
        "clip": None,
        "std_scale": 1.0,
    }


def checkpoint_reference_template(checkpoint: dict[str, Any], *, raw: bool) -> pd.Series:
    key = "reference_template_raw" if raw else "reference_template"
    payload = checkpoint.get(key)
    if payload is None and raw:
        payload = checkpoint.get("reference_template")
    if payload is None and not raw:
        payload = checkpoint.get("reference_template_raw")
    return pd.Series(payload or {})


def load_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    run_device = resolve_device(device)
    checkpoint = torch.load(Path(checkpoint_path), map_location=run_device)
    config = _build_config_from_payload(checkpoint.get("config"))
    config.model.n_processes = int(checkpoint["n_processes"])
    version = checkpoint_model_version(checkpoint)
    if version == AgeDecoupledSurrealGAN.model_version:
        model: torch.nn.Module = AgeDecoupledSurrealGAN(len(checkpoint["feature_columns"]), config)
    else:
        model = LegacyAgeDecoupledSurrealGAN(len(checkpoint["feature_columns"]), config)
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
        return __import__("json").load(handle)


def infer_dataframe(
    checkpoint_path: str | Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    reference_template: pd.Series | torch.Tensor,
    device: str = "cpu",
) -> pd.DataFrame:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    normalization = checkpoint_normalization_payload(checkpoint)
    features_norm = apply_feature_normalization(frame[feature_columns], normalization, feature_columns)
    features = torch.tensor(features_norm, dtype=torch.float32, device=run_device)
    if isinstance(reference_template, pd.Series):
        ref_norm = apply_feature_normalization(reference_template.reindex(feature_columns), normalization, feature_columns)
        ref_values = torch.tensor(ref_norm.reshape(-1), dtype=torch.float32, device=run_device)
    else:
        ref_values = reference_template.to(run_device)
    reference = ref_values.unsqueeze(0).repeat(features.shape[0], 1)
    outputs = model.infer(features, reference)  # type: ignore[attr-defined]
    result = frame[["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]].copy()
    result["age_latent"] = outputs["age_latent"].detach().cpu().numpy().reshape(-1)
    process = outputs["process_latents"].detach().cpu().numpy()
    for idx in range(checkpoint["n_processes"]):
        result[f"r{idx + 1}"] = process[:, idx]
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
    normalization = checkpoint_normalization_payload(checkpoint)
    feature_values = _numeric_feature_series(row, feature_columns)
    feature_norm = apply_feature_normalization(feature_values, normalization, feature_columns)
    feature_tensor = torch.tensor(feature_norm.reshape(1, -1), dtype=torch.float32, device=run_device)
    ref_norm = apply_feature_normalization(reference_template.reindex(feature_columns), normalization, feature_columns)
    ref_tensor = torch.tensor(ref_norm.reshape(1, -1), dtype=torch.float32, device=run_device)
    outputs = model.infer(feature_tensor, ref_tensor)  # type: ignore[attr-defined]
    roi_delta_raw = scale_delta_to_raw(outputs["synthetic_delta"].detach().cpu().numpy(), normalization, feature_columns).reshape(-1)
    age_delta_raw = scale_delta_to_raw(outputs["age_delta"].detach().cpu().numpy(), normalization, feature_columns).reshape(-1)
    process_deltas_raw = scale_delta_to_raw(
        outputs["process_deltas"].detach().cpu().numpy().reshape(checkpoint["n_processes"], -1),
        normalization,
        feature_columns,
    ).reshape(checkpoint["n_processes"], -1)
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "age_latent": float(outputs["age_latent"].detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": outputs["process_latents"].detach().cpu().numpy().reshape(-1).tolist(),
        "synthetic_delta": roi_delta_raw.tolist(),
        "age_delta": age_delta_raw.tolist(),
        "process_deltas": process_deltas_raw.tolist(),
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
    normalization = checkpoint_normalization_payload(checkpoint)
    feature_norm = apply_feature_normalization(feature_values, normalization, feature_columns)
    feature_tensor = torch.tensor(feature_norm.reshape(1, -1), dtype=torch.float32, device=run_device)
    if hasattr(model, "infer_latents"):
        age_latent, process_latents, _, _ = model.infer_latents(feature_tensor)  # type: ignore[attr-defined]
    else:
        age_latent, process_latents = model.encode(feature_tensor)  # type: ignore[attr-defined]
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "age_latent": float(age_latent.detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": process_latents.detach().cpu().numpy().reshape(-1).tolist(),
    }


def _generate_single_row_legacy(
    *,
    model: LegacyAgeDecoupledSurrealGAN,
    checkpoint: dict[str, Any],
    row: pd.Series,
    feature_columns: list[str],
    age_years: float | None,
    process_latents: list[float] | None,
    device: str,
) -> dict[str, Any]:
    run_device = resolve_device(device)
    feature_values = _numeric_feature_series(row, feature_columns)
    baseline = torch.tensor(feature_values.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    reference_template = checkpoint_reference_template(checkpoint, raw=False).reindex(feature_columns)
    reference_tensor = torch.tensor(reference_template.to_numpy(dtype="float32"), device=run_device).unsqueeze(0)
    config = _build_config_from_payload(checkpoint.get("config"))

    default_age_latent, default_process_latents = model.encode(baseline)
    if age_years is None:
        requested_age_years = float(row.get("age", config.data.age_latent_normalization_min))
        age_latent_tensor = default_age_latent
    else:
        requested_age_years = float(age_years)
        age_latent_tensor = torch.full_like(default_age_latent, normalize_age_years(requested_age_years, config))

    if process_latents is None:
        process_latent_tensor = default_process_latents
    else:
        if len(process_latents) != int(checkpoint["n_processes"]):
            raise ValueError(f"Expected {int(checkpoint['n_processes'])} process latents, received {len(process_latents)}.")
        process_latent_tensor = torch.tensor([process_latents], dtype=torch.float32, device=run_device)

    with torch.no_grad():
        direct_target, direct_delta = model.synthesize(baseline, age_latent_tensor, process_latent_tensor)
        _, subject_fake_delta = model.synthesize(reference_tensor, default_age_latent, default_process_latents)
        age_basis, process_basis = model.decompose(subject_fake_delta)
        latent_age_shift = age_latent_tensor - default_age_latent
        latent_process_shift = process_latent_tensor - default_process_latents
        anchored_delta = latent_age_shift * age_basis + (latent_process_shift.unsqueeze(-1) * process_basis).sum(dim=1)
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
            "generation_mode": "anchored_component_linear_legacy",
            "direct_delta_abs_max": float(direct_delta.detach().abs().max().cpu()),
            "direct_delta_abs_mean": float(direct_delta.detach().abs().mean().cpu()),
            "anchored_delta_abs_max": float(anchored_delta.detach().abs().max().cpu()),
            "anchored_delta_abs_mean": float(anchored_delta.detach().abs().mean().cpu()),
        },
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
    if checkpoint_model_version(checkpoint) != AgeDecoupledSurrealGAN.model_version:
        return _generate_single_row_legacy(
            model=model,  # type: ignore[arg-type]
            checkpoint=checkpoint,
            row=row,
            feature_columns=feature_columns,
            age_years=age_years,
            process_latents=process_latents,
            device=device,
        )

    run_device = resolve_device(device)
    normalization = checkpoint_normalization_payload(checkpoint)
    config = _build_config_from_payload(checkpoint.get("config"))
    feature_values = _numeric_feature_series(row, feature_columns)
    baseline_raw = feature_values.to_numpy(dtype="float32")
    baseline_norm = apply_feature_normalization(feature_values, normalization, feature_columns).reshape(1, -1)
    baseline_tensor = torch.tensor(baseline_norm, dtype=torch.float32, device=run_device)

    age_default, process_defaults, _, _ = model.infer_latents(baseline_tensor)  # type: ignore[attr-defined]
    if age_years is None:
        requested_age_years = float(row.get("age", config.data.age_latent_normalization_min))
        age_latent_tensor = age_default
    else:
        requested_age_years = float(age_years)
        age_latent_tensor = torch.full_like(age_default, normalize_age_years(requested_age_years, config))
    if process_latents is None:
        process_latent_tensor = process_defaults
    else:
        if len(process_latents) != int(checkpoint["n_processes"]):
            raise ValueError(f"Expected {int(checkpoint['n_processes'])} process latents, received {len(process_latents)}.")
        process_latent_tensor = torch.tensor([process_latents], dtype=torch.float32, device=run_device)

    with torch.no_grad():
        synth = model.synthesize_full(baseline_tensor, age_latent_tensor, process_latent_tensor)  # type: ignore[attr-defined]

    baseline_target = invert_feature_normalization(baseline_norm, normalization, feature_columns).reshape(-1)
    synthetic_target = invert_feature_normalization(
        synth.fake_target.detach().cpu().numpy(),
        normalization,
        feature_columns,
    ).reshape(-1)
    raw_delta = scale_delta_to_raw(synth.total_delta.detach().cpu().numpy(), normalization, feature_columns).reshape(-1)
    age_delta = scale_delta_to_raw(synth.age_component.detach().cpu().numpy(), normalization, feature_columns).reshape(-1)
    process_deltas = scale_delta_to_raw(
        synth.process_components.detach().cpu().numpy().reshape(checkpoint["n_processes"], -1),
        normalization,
        feature_columns,
    ).reshape(checkpoint["n_processes"], -1)

    baseline_abs = np.abs(baseline_raw)
    percent_change = np.zeros_like(raw_delta, dtype=np.float32)
    valid_mask = baseline_abs > 0
    percent_change[valid_mask] = (raw_delta[valid_mask] / baseline_abs[valid_mask]) * 100.0
    return {
        "n_processes": int(checkpoint["n_processes"]),
        "requested_age_years": requested_age_years,
        "age_latent": float(age_latent_tensor.detach().cpu().numpy().reshape(-1)[0]),
        "default_age_latent": float(age_default.detach().cpu().numpy().reshape(-1)[0]),
        "process_latents": process_latent_tensor.detach().cpu().numpy().reshape(-1).tolist(),
        "default_process_latents": process_defaults.detach().cpu().numpy().reshape(-1).tolist(),
        "baseline_target": baseline_target.tolist(),
        "synthetic_target": synthetic_target.tolist(),
        "synthetic_delta": raw_delta.tolist(),
        "percent_change": percent_change.tolist(),
        "age_delta": age_delta.tolist(),
        "process_deltas": process_deltas.tolist(),
        "debug": {
            "generation_mode": "direct_sampled_additive",
            "delta_abs_max": float(np.abs(raw_delta).max()),
            "delta_abs_mean": float(np.abs(raw_delta).mean()),
        },
    }
