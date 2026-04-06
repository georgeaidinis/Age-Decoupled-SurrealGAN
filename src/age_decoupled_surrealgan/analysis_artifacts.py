from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from nibabel import Nifti1Image, load as load_nifti

from .config import ProjectConfig
from .inference import load_checkpoint, normalize_age_years, resolve_device
from .metrics import align_process_latents, compute_repetition_agreement
from .utils import ensure_dir, save_json


def _load_roi_metadata(processed_dir: Path) -> pd.DataFrame:
    return pd.read_csv(processed_dir / "roi_metadata.csv")


def _load_split_frame(processed_dir: Path, split_name: str) -> pd.DataFrame:
    return pd.read_csv(processed_dir / f"{split_name}.csv", low_memory=False)


def _save_overlay(segmentation_img: Any, segmentation_data: np.ndarray, roi_df: pd.DataFrame, output_path: Path) -> None:
    overlay_data = np.zeros(segmentation_data.shape, dtype=np.float32)
    for row in roi_df[["roi_id", "percent_change"]].itertuples(index=False):
        roi_id = int(row.roi_id)
        overlay_data[segmentation_data == roi_id] = float(row.percent_change)
    overlay_img = Nifti1Image(overlay_data, segmentation_img.affine, segmentation_img.header)
    overlay_img.to_filename(str(output_path))


def _top_changes(frame: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    return frame.assign(abs_percent_change=frame["percent_change"].abs()).sort_values(
        ["abs_percent_change", "percent_change"], ascending=[False, False]
    ).head(limit)


def _sign_summary(pattern_key: str, frame: pd.DataFrame) -> dict[str, Any]:
    positive_mask = frame["delta"] > 0
    negative_mask = frame["delta"] < 0
    return {
        "pattern_key": pattern_key,
        "positive_roi_count": int(positive_mask.sum()),
        "negative_roi_count": int(negative_mask.sum()),
        "zero_roi_count": int((~positive_mask & ~negative_mask).sum()),
        "positive_roi_fraction": float(positive_mask.mean()),
        "negative_roi_fraction": float(negative_mask.mean()),
        "mean_delta": float(frame["delta"].mean()),
        "mean_absolute_delta": float(frame["delta"].abs().mean()),
        "mean_percent_change": float(frame["percent_change"].mean()),
        "mean_absolute_percent_change": float(frame["percent_change"].abs().mean()),
    }


def _plot_heatmap(corr: pd.DataFrame, output_path: Path, title: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return
    size = max(8, min(24, 0.42 * len(corr.columns) + 6))
    fig, ax = plt.subplots(figsize=(size, size))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(np.arange(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _prediction_frame_for_split(
    checkpoint_path: Path,
    frame: pd.DataFrame,
    feature_columns: list[str],
    reference_template: pd.Series,
    n_processes: int,
    device: str,
    batch_size: int = 1024,
) -> pd.DataFrame:
    model, _ = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    ref_tensor = torch.tensor(reference_template.reindex(feature_columns).to_numpy(dtype="float32"), device=run_device)
    outputs: list[pd.DataFrame] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            batch = frame.iloc[start : start + batch_size].copy()
            features = torch.tensor(batch[feature_columns].to_numpy(dtype="float32"), device=run_device)
            reference = ref_tensor.unsqueeze(0).repeat(features.shape[0], 1)
            prediction = model.infer(features, reference)
            block = batch[
                ["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
            ].copy()
            block["age_latent"] = prediction["age_latent"].detach().cpu().numpy().reshape(-1)
            process = prediction["process_latents"].detach().cpu().numpy()
            for idx in range(n_processes):
                block[f"r{idx + 1}"] = process[:, idx]
            outputs.append(block)
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()


def _compute_population_patterns(
    *,
    checkpoint_path: Path,
    config: ProjectConfig,
    processed_dir: Path,
    run_dir: Path,
    device: str,
) -> dict[str, Any]:
    model, checkpoint = load_checkpoint(checkpoint_path, device=device)
    run_device = resolve_device(device)
    feature_columns = checkpoint["feature_columns"]
    checkpoint_config = ProjectConfig()
    config_payload = checkpoint.get("config", {})
    if isinstance(config_payload, dict):
        for section_name in ["paths", "data", "model", "training", "losses", "tuning", "app"]:
            section = config_payload.get(section_name, {})
            if isinstance(section, dict):
                current = getattr(checkpoint_config, section_name)
                for key, value in section.items():
                    setattr(current, key, value)
    roi_df = _load_roi_metadata(processed_dir)
    reference_frame = _load_split_frame(processed_dir, "train")
    reference_frame = reference_frame.loc[reference_frame["cohort_bucket"] == "ref"].dropna(subset=feature_columns)
    features = torch.tensor(reference_frame[feature_columns].to_numpy(dtype="float32"), device=run_device)
    baseline_abs = features.abs().clamp_min(1.0)
    zero_process = torch.zeros(features.shape[0], int(checkpoint["n_processes"]), device=run_device)
    age_low = torch.full(
        (features.shape[0], checkpoint_config.model.age_latent_dim),
        normalize_age_years(checkpoint_config.data.age_latent_normalization_min, checkpoint_config),
        device=run_device,
        dtype=torch.float32,
    )
    age_high = torch.full(
        (features.shape[0], checkpoint_config.model.age_latent_dim),
        normalize_age_years(checkpoint_config.data.age_latent_normalization_max, checkpoint_config),
        device=run_device,
        dtype=torch.float32,
    )
    anchor_age = torch.full(
        (features.shape[0], checkpoint_config.model.age_latent_dim),
        normalize_age_years(checkpoint_config.training.sensitivity_process_anchor_age, checkpoint_config),
        device=run_device,
        dtype=torch.float32,
    )

    analysis_dir = ensure_dir(run_dir / "analysis" / "population_patterns")
    segmentation_img = load_nifti(config.paths.atlas_segmentation)
    segmentation_data = np.asanyarray(segmentation_img.dataobj)

    manifest_patterns: list[dict[str, Any]] = []
    sign_summaries: list[dict[str, Any]] = []
    top_rows: list[pd.DataFrame] = []

    with torch.no_grad():
        age_low_target, _ = model.synthesize(features, age_low, zero_process)
        age_high_target, _ = model.synthesize(features, age_high, zero_process)
        age_delta = age_high_target - age_low_target
        age_percent = ((age_delta / baseline_abs) * 100.0).mean(dim=0).detach().cpu().numpy()
        age_mean_delta = age_delta.mean(dim=0).detach().cpu().numpy()
        age_baseline = features.mean(dim=0).detach().cpu().numpy()
        age_predicted = age_high_target.mean(dim=0).detach().cpu().numpy()
        age_table = roi_df.copy()
        age_table["baseline_value"] = age_baseline
        age_table["predicted_value"] = age_predicted
        age_table["delta"] = age_mean_delta
        age_table["percent_change"] = age_percent
        age_csv = analysis_dir / "age.csv"
        age_json = analysis_dir / "age.json"
        age_overlay = analysis_dir / "age_overlay.nii.gz"
        age_top = _top_changes(age_table, limit=10)
        age_table.to_csv(age_csv, index=False)
        age_payload = {
            "pattern_key": "age",
            "label": "Chronological age",
            "description": "Average isolated generator response when age moves from the configured minimum to maximum with all process latents fixed at zero.",
            "roi_table": age_table.to_dict(orient="records"),
            "top_changes": age_top.to_dict(orient="records"),
            "sign_summary": _sign_summary("age", age_table),
            "overlay_filename": age_overlay.name,
        }
        save_json(age_json, age_payload)
        _save_overlay(segmentation_img, segmentation_data, age_table, age_overlay)
        manifest_patterns.append(
            {
                "key": "age",
                "label": "Chronological age",
                "description": age_payload["description"],
                "json_filename": age_json.name,
                "overlay_filename": age_overlay.name,
                "csv_filename": age_csv.name,
                "sign_summary": age_payload["sign_summary"],
            }
        )
        sign_summaries.append(age_payload["sign_summary"])
        top_rows.append(age_top.assign(pattern_key="age"))

        anchor_target, _ = model.synthesize(features, anchor_age, zero_process)
        for idx in range(int(checkpoint["n_processes"])):
            process_latents = torch.zeros(features.shape[0], int(checkpoint["n_processes"]), device=run_device)
            process_latents[:, idx] = 1.0
            process_target, _ = model.synthesize(features, anchor_age, process_latents)
            process_delta = process_target - anchor_target
            process_percent = ((process_delta / baseline_abs) * 100.0).mean(dim=0).detach().cpu().numpy()
            process_mean_delta = process_delta.mean(dim=0).detach().cpu().numpy()
            process_baseline = anchor_target.mean(dim=0).detach().cpu().numpy()
            process_predicted = process_target.mean(dim=0).detach().cpu().numpy()
            process_table = roi_df.copy()
            process_table["baseline_value"] = process_baseline
            process_table["predicted_value"] = process_predicted
            process_table["delta"] = process_mean_delta
            process_table["percent_change"] = process_percent
            pattern_key = f"r{idx + 1}"
            pattern_csv = analysis_dir / f"{pattern_key}.csv"
            pattern_json = analysis_dir / f"{pattern_key}.json"
            pattern_overlay = analysis_dir / f"{pattern_key}_overlay.nii.gz"
            pattern_top = _top_changes(process_table, limit=10)
            process_table.to_csv(pattern_csv, index=False)
            payload = {
                "pattern_key": pattern_key,
                "label": f"Process {idx + 1}",
                "description": "Average isolated generator response when this process latent is moved from 0 to 1 while age is held at the configured anchor age and all other process latents remain zero.",
                "roi_table": process_table.to_dict(orient="records"),
                "top_changes": pattern_top.to_dict(orient="records"),
                "sign_summary": _sign_summary(pattern_key, process_table),
                "overlay_filename": pattern_overlay.name,
            }
            save_json(pattern_json, payload)
            _save_overlay(segmentation_img, segmentation_data, process_table, pattern_overlay)
            manifest_patterns.append(
                {
                    "key": pattern_key,
                    "label": f"Process {idx + 1}",
                    "description": payload["description"],
                    "json_filename": pattern_json.name,
                    "overlay_filename": pattern_overlay.name,
                    "csv_filename": pattern_csv.name,
                    "sign_summary": payload["sign_summary"],
                }
            )
            sign_summaries.append(payload["sign_summary"])
            top_rows.append(pattern_top.assign(pattern_key=pattern_key))

    sign_summary_path = analysis_dir / "sign_summary.csv"
    top_summary_path = analysis_dir / "top10_per_factor.csv"
    pd.DataFrame(sign_summaries).to_csv(sign_summary_path, index=False)
    pd.concat(top_rows, ignore_index=True).to_csv(top_summary_path, index=False)
    manifest = {
        "reference_split": "train",
        "reference_cohort_bucket": "ref",
        "age_pattern": {
            "age_min_years": checkpoint_config.data.age_latent_normalization_min,
            "age_max_years": checkpoint_config.data.age_latent_normalization_max,
        },
        "process_anchor_age_years": checkpoint_config.training.sensitivity_process_anchor_age,
        "patterns": manifest_patterns,
        "sign_summary_csv": sign_summary_path.name,
        "top10_csv": top_summary_path.name,
    }
    save_json(analysis_dir / "manifest.json", manifest)
    return manifest


def _compute_repetition_stability(run_dir: Path, n_processes: int) -> dict[str, Any]:
    prediction_paths = sorted((run_dir / "predictions").glob("repetition_*_val.csv"))
    output_dir = ensure_dir(run_dir / "analysis" / "repetition_stability")
    if not prediction_paths:
        payload = {
            "available": False,
            "reason": "No repetition val prediction files found.",
        }
        save_json(output_dir / "summary.json", payload)
        return payload

    frames = [pd.read_csv(path) for path in prediction_paths]
    process_columns = [f"r{i + 1}" for i in range(n_processes)]
    dim_matrix = np.eye(len(frames), dtype=float)
    diff_matrix = np.eye(len(frames), dtype=float)
    overall_matrix = np.eye(len(frames), dtype=float)
    for i in range(len(frames)):
        lhs = frames[i][process_columns].to_numpy()
        for j in range(i + 1, len(frames)):
            rhs = frames[j][process_columns].to_numpy()
            _, dim_score, diff_score = align_process_latents(lhs, rhs)
            dim_matrix[i, j] = dim_matrix[j, i] = dim_score
            diff_matrix[i, j] = diff_matrix[j, i] = diff_score
            overall_matrix[i, j] = overall_matrix[j, i] = 0.5 * (dim_score + diff_score)
    labels = [path.stem for path in prediction_paths]
    pd.DataFrame(dim_matrix, index=labels, columns=labels).to_csv(output_dir / "dimension_correlation.csv")
    pd.DataFrame(diff_matrix, index=labels, columns=labels).to_csv(output_dir / "difference_correlation.csv")
    pd.DataFrame(overall_matrix, index=labels, columns=labels).to_csv(output_dir / "overall_correlation.csv")
    agreement = compute_repetition_agreement(frames, n_processes)
    payload = {
        "available": True,
        **agreement,
        "repetitions": labels,
    }
    save_json(output_dir / "summary.json", payload)
    return payload


def _compute_correlation_artifacts(run_dir: Path, n_processes: int) -> dict[str, Any]:
    output_dir = ensure_dir(run_dir / "analysis" / "correlations")
    manifests: dict[str, Any] = {}
    for prediction_path in sorted((run_dir / "predictions").glob("*.csv")):
        if prediction_path.name.startswith("repetition_"):
            continue
        split_name = prediction_path.stem
        frame = pd.read_csv(prediction_path)
        numeric = frame[["age", "age_latent"] + [f"r{i + 1}" for i in range(n_processes)]].copy()
        diagnosis_dummies = pd.get_dummies(frame["diagnosis_group"], prefix="dx", dtype=float)
        study_dummies = pd.get_dummies(frame["study"], prefix="study", dtype=float)
        corr_frame = pd.concat([numeric, diagnosis_dummies, study_dummies], axis=1)
        corr = corr_frame.corr(numeric_only=True).fillna(0.0)
        corr_csv = output_dir / f"{split_name}_matrix.csv"
        corr_png = output_dir / f"{split_name}_heatmap.png"
        corr.to_csv(corr_csv)
        _plot_heatmap(corr, corr_png, title=f"{split_name} latent / metadata correlation map")
        manifests[split_name] = {
            "matrix_csv": corr_csv.name,
            "heatmap_png": corr_png.name,
            "n_variables": int(corr.shape[0]),
        }
    save_json(output_dir / "manifest.json", manifests)
    return manifests


def ensure_prediction_splits(
    *,
    run_dir: Path,
    checkpoint_path: Path,
    processed_dir: Path,
    n_processes: int,
    feature_columns: list[str],
    reference_template: pd.Series,
    device: str,
) -> None:
    prediction_dir = ensure_dir(run_dir / "predictions")
    for split_name in ["train", "val", "id_test", "ood_test", "application"]:
        target_path = prediction_dir / f"{split_name}.csv"
        if target_path.exists():
            continue
        frame = _load_split_frame(processed_dir, split_name)
        prediction = _prediction_frame_for_split(
            checkpoint_path=checkpoint_path,
            frame=frame,
            feature_columns=feature_columns,
            reference_template=reference_template,
            n_processes=n_processes,
            device=device,
        )
        prediction.to_csv(target_path, index=False)


def build_run_analysis_artifacts(
    run_dir: Path,
    config: ProjectConfig,
    *,
    force: bool = False,
    device: str | None = None,
) -> dict[str, Any]:
    run_summary_path = run_dir / "run_summary.json"
    if not run_summary_path.exists():
        raise FileNotFoundError(f"run_summary.json not found in {run_dir}")
    summary = json.loads(run_summary_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(summary["selected_checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    processed_dir = Path(config.paths.processed_dir)
    analysis_dir = ensure_dir(run_dir / "analysis")
    reference_template = pd.Series(checkpoint["reference_template"])
    feature_columns = checkpoint["feature_columns"]
    n_processes = int(checkpoint["n_processes"])
    ensure_prediction_splits(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        processed_dir=processed_dir,
        n_processes=n_processes,
        feature_columns=feature_columns,
        reference_template=reference_template,
        device=device or config.training.device,
    )

    manifest_path = analysis_dir / "manifest.json"
    if manifest_path.exists() and not force:
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    population_manifest = _compute_population_patterns(
        checkpoint_path=checkpoint_path,
        config=config,
        processed_dir=processed_dir,
        run_dir=run_dir,
        device=device or config.training.device,
    )
    repetition_manifest = _compute_repetition_stability(run_dir, n_processes)
    correlation_manifest = _compute_correlation_artifacts(run_dir, n_processes)
    analysis_manifest = {
        "population_patterns": population_manifest,
        "repetition_stability": repetition_manifest,
        "correlations": correlation_manifest,
    }
    save_json(manifest_path, analysis_manifest)
    return analysis_manifest


def backfill_analysis_artifacts(config: ProjectConfig, run_dir: str | None = None, force: bool = False) -> list[dict[str, Any]]:
    runs_root = Path(config.paths.runs_dir)
    target_dirs = [Path(run_dir)] if run_dir else sorted(
        [path for path in runs_root.glob("*") if (path / "run_summary.json").exists()],
        reverse=True,
    )
    results: list[dict[str, Any]] = []
    for path in target_dirs:
        manifest = build_run_analysis_artifacts(path, config, force=force, device=config.training.device)
        results.append({"run_dir": str(path), "analysis_manifest": manifest})
    return results
