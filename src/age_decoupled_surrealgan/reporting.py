from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .utils import ensure_dir


TRAIN_TENSORBOARD_TAGS = {
    "discriminator": "loss/discriminator/total",
    "generator_total": "loss/generator/total",
    "adv": "loss/generator/adversarial",
    "age_sup": "loss/supervision/age_target",
    "ref_age_sup": "loss/supervision/age_reference",
    "age_adv": "loss/disentanglement/age_adversary",
    "latent_recon": "loss/reconstruction/latents",
    "decompose": "loss/reconstruction/decomposition",
    "identity": "loss/identity/exact_zero",
    "monotonicity": "loss/regularization/monotonicity",
    "orthogonality": "loss/regularization/orthogonality",
    "covariance": "loss/regularization/age_process_covariance",
    "reference_process": "loss/regularization/reference_process_sparsity",
    "change_mag": "loss/ablation/change_magnitude",
    "low_identity": "loss/ablation/low_activation_identity",
    "process_age_corr": "loss/ablation/process_age_correlation",
    "process_sparse": "loss/ablation/process_latent_sparsity",
    "age_sensitivity": "loss/generator/age_sensitivity_margin",
    "process_sensitivity": "loss/generator/process_sensitivity_margin",
    "age_shrinkage": "loss/generator/age_shrinkage_bias",
    "process_shrinkage": "loss/generator/process_shrinkage_bias",
    "state_age_latent_mean": "state/latents/age_mean",
    "state_process_latent_abs_mean": "state/latents/process_abs_mean",
    "state_fake_change_abs_mean": "state/change/fake_abs_mean",
    "state_age_sensitivity_pct": "state/sensitivity/age_pct",
    "state_process_sensitivity_pct": "state/sensitivity/process_pct",
    "state_process_separation_pct": "state/sensitivity/process_separation_pct",
    "state_age_growth_penalty": "state/sensitivity/age_growth_penalty",
    "state_process_growth_penalty": "state/sensitivity/process_growth_penalty",
}


VAL_TENSORBOARD_TAGS = {
    "age_latent_age_correlation": "metric/validation/age_latent_age_correlation",
    "mean_absolute_process_age_correlation": "metric/validation/process_age_abs_mean",
    "mean_absolute_residual_process_age_correlation": "metric/validation/process_age_residual_abs_mean",
    "age_sensitivity_pct_mean": "metric/validation/age_sensitivity_pct_mean",
    "mean_process_sensitivity_pct_mean": "metric/validation/process_sensitivity_pct_mean",
    "process_separation_pct_mean": "metric/validation/process_separation_pct_mean",
    "latent_sensitivity_score": "metric/validation/latent_sensitivity_score",
    "composite_score": "selection/validation/composite_score",
    "quality_score": "selection/validation/quality_score",
}


def flatten_metrics(metrics: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in metrics.items():
        compound = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_metrics(value, prefix=f"{compound}."))
        else:
            flat[compound] = value
    return flat


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True))
            handle.write("\n")


def save_records_csv(path: Path, records: list[dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    pd.DataFrame(records).to_csv(path, index=False)


def epoch_log_line(
    repetition_index: int,
    repetitions: int,
    epoch: int,
    total_epochs: int,
    train_metrics: dict[str, float],
    val_metrics: dict[str, Any],
) -> str:
    return (
        f"[rep {repetition_index + 1}/{repetitions}] "
        f"[epoch {epoch}/{total_epochs}] "
        f"train: G={train_metrics['generator_total']:.4f} "
        f"D={train_metrics['discriminator']:.4f} "
        f"adv={train_metrics['adv']:.4f} "
        f"age={train_metrics['age_sup']:.4f} "
        f"decomp={train_metrics['decompose']:.4f} "
        f"val: comp={val_metrics['composite_score']:.4f} "
        f"qual={val_metrics.get('quality_score', val_metrics['composite_score']):.4f} "
        f"sens={val_metrics.get('latent_sensitivity_score', 0.0):.4f} "
        f"age_corr={val_metrics['age_latent_age_correlation']:.4f} "
        f"resid_age={val_metrics['mean_absolute_residual_process_age_correlation']:.4f}"
    )


def append_log_line(path: Path, line: str) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.write("\n")


def startup_summary_lines(
    *,
    experiment_name: str,
    config_path: str,
    device: str,
    n_features: int,
    n_processes: int,
    repetitions: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    discriminator_learning_rate: float,
    checkpoint_epochs: list[int],
    num_workers: int,
    persistent_workers: bool,
    use_amp: bool,
    compile_model: bool,
    encoder_hidden_dims: list[int],
    generator_hidden_dims: list[int],
    discriminator_hidden_dims: list[int],
    decomposer_hidden_dims: list[int],
) -> list[str]:
    return [
        f"Experiment: {experiment_name}",
        f"Config: {config_path}",
        f"Device: {device}",
        (
            f"Run setup: repetitions={repetitions}, epochs={epochs}, batch_size={batch_size}, "
            f"lr={learning_rate:.2e}, d_lr={discriminator_learning_rate:.2e}"
        ),
        (
            f"Runtime: num_workers={num_workers}, persistent_workers={persistent_workers}, "
            f"use_amp={use_amp}, compile_model={compile_model}"
        ),
        (
            f"Architecture: n_features={n_features}, n_processes={n_processes}, "
            f"encoder={encoder_hidden_dims}, generator={generator_hidden_dims}, "
            f"discriminator={discriminator_hidden_dims}, decomposer={decomposer_hidden_dims}"
        ),
        f"Checkpoint epochs: {checkpoint_epochs} plus best-per-repetition checkpoint",
    ]


def save_split_metrics_tables(split_metrics: dict[str, dict[str, Any]], output_dir: Path) -> None:
    records = []
    for split_name, metrics in split_metrics.items():
        flat = flatten_metrics(metrics)
        flat["split"] = split_name
        records.append(flat)
    save_records_csv(output_dir / "split_metrics.csv", records)


def save_run_markdown_summary(summary: dict[str, Any], output_path: Path) -> None:
    split_metrics = summary.get("split_metrics", {})
    lines = [
        "# Run Summary",
        "",
        f"- Selected repetition: `{summary.get('selected_repetition')}`",
        f"- Selected checkpoint: `{summary.get('selected_checkpoint')}`",
        "",
        "## Agreement",
        "",
        f"- Mean dimension correlation: `{summary['agreement']['mean_dimension_correlation']:.4f}`",
        f"- Mean difference correlation: `{summary['agreement']['mean_difference_correlation']:.4f}`",
        "",
        "## Split Metrics",
        "",
    ]
    for split_name, metrics in split_metrics.items():
        lines.extend(
            [
                f"### {split_name}",
                "",
                f"- Composite score: `{metrics['composite_score']:.4f}`",
                f"- Quality score: `{metrics.get('quality_score', metrics['composite_score']):.4f}`",
                f"- Age latent vs age correlation: `{metrics['age_latent_age_correlation']:.4f}`",
                f"- Mean absolute process-age correlation: `{metrics['mean_absolute_process_age_correlation']:.4f}`",
                f"- Mean absolute residual process-age correlation: `{metrics['mean_absolute_residual_process_age_correlation']:.4f}`",
                f"- Age sensitivity (% mean): `{metrics.get('age_sensitivity_pct_mean', 0.0):.4f}`",
                f"- Mean process sensitivity (% mean): `{metrics.get('mean_process_sensitivity_pct_mean', 0.0):.4f}`",
                "",
            ]
        )
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def load_doc_text(repo_root: Path, filename: str) -> str | None:
    path = repo_root / "docs" / filename
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")
