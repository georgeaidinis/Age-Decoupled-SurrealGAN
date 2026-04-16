from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import ProjectConfig
from .utils import ensure_dir, save_json


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _val_metrics(summary: dict[str, Any]) -> dict[str, float]:
    val = summary.get("split_metrics", {}).get("val", {})
    return {key: float(value) for key, value in val.items() if isinstance(value, (int, float))}


def _agreement_score(summary: dict[str, Any]) -> float:
    agreement = summary.get("agreement", {})
    return 0.5 * float(agreement.get("mean_dimension_correlation", 0.0)) + 0.5 * float(
        agreement.get("mean_difference_correlation", 0.0)
    )


def _k_from_run(run_dir: Path) -> int:
    resolved_path = run_dir / "resolved_config.json"
    if resolved_path.exists():
        resolved = _load_json(resolved_path)
        return int(resolved.get("model", {}).get("n_processes", 0))
    summary = _load_json(run_dir / "run_summary.json")
    return int(summary.get("summary", {}).get("n_processes", 0))


def evaluate_run_admissibility(run_dir: Path, config: ProjectConfig) -> dict[str, Any]:
    summary = _load_json(run_dir / "run_summary.json")
    val = _val_metrics(summary)
    thresholds = config.selection
    checks = {
        "age_corr": val.get("age_latent_age_correlation", 0.0) >= thresholds.min_age_latent_age_correlation,
        "composite": val.get("composite_score", 0.0) >= thresholds.min_composite_score,
        "process_sensitivity": val.get("mean_process_sensitivity_pct_mean", 0.0)
        >= thresholds.min_mean_process_sensitivity_pct_mean,
        "process_separation": val.get("process_separation_pct_mean", 0.0) >= thresholds.min_process_separation_pct_mean,
        "pattern_collapse": val.get("process_pattern_correlation_abs_mean", 1.0)
        <= thresholds.max_process_pattern_correlation_abs_mean,
        "latent_collapse": val.get("process_latent_pairwise_correlation_abs_mean", 1.0)
        <= thresholds.max_process_latent_pairwise_correlation_abs_mean,
        "residual_age_leakage": val.get("mean_absolute_residual_process_age_correlation", 1.0)
        <= thresholds.max_mean_absolute_residual_process_age_correlation,
    }
    pass_count = sum(int(value) for value in checks.values())
    agreement_score = _agreement_score(summary)
    selection_score = float(
        val.get("collapse_aware_selection_score", val.get("collapse_aware_quality_score", val.get("selection_score", 0.0)))
    )
    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "k": _k_from_run(run_dir),
        "checks": checks,
        "screen_pass": all(checks.values()),
        "screen_pass_count": pass_count,
        "agreement_score": agreement_score,
        "selection_score": selection_score,
        "val_metrics": val,
        "selected_checkpoint": summary.get("selected_checkpoint"),
    }


def select_best_runs(
    *,
    config: ProjectConfig,
    record_dir: str | None = None,
    output_path: str | None = None,
) -> dict[str, Any]:
    if record_dir is not None:
        record_root = Path(record_dir)
        run_dirs = []
        for path in sorted(record_root.glob("*.json")):
            payload = _load_json(path)
            run_dir = payload.get("run_dir")
            if run_dir:
                run_dirs.append(Path(run_dir))
    else:
        run_dirs = sorted(path.parent for path in Path(config.paths.runs_dir).glob("*/run_summary.json"))

    records = [evaluate_run_admissibility(run_dir, config) for run_dir in run_dirs if (run_dir / "run_summary.json").exists()]
    ranked = sorted(
        records,
        key=lambda row: (
            int(row["screen_pass"]),
            int(row["screen_pass_count"]),
            float(row["agreement_score"]),
            float(row["selection_score"]),
            float(row["val_metrics"].get("composite_score", 0.0)),
        ),
        reverse=True,
    )
    by_k: dict[int, dict[str, Any]] = {}
    for record in ranked:
        by_k.setdefault(int(record["k"]), record)
    payload = {
        "best_overall": ranked[0] if ranked else None,
        "best_by_k": by_k,
        "records": ranked,
    }
    if output_path is not None:
        output = Path(output_path)
        ensure_dir(output.parent)
        save_json(output, payload)
    return payload
