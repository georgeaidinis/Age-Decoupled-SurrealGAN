from __future__ import annotations

import itertools
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression


def safe_pearsonr(lhs: np.ndarray, rhs: np.ndarray) -> float:
    if lhs.ndim != 1 or rhs.ndim != 1 or len(lhs) < 2:
        return 0.0
    if np.std(lhs) < 1e-8 or np.std(rhs) < 1e-8:
        return 0.0
    return float(pearsonr(lhs, rhs)[0])


def align_process_latents(lhs: np.ndarray, rhs: np.ndarray) -> tuple[list[int], float, float]:
    n_processes = lhs.shape[1]
    best_order = list(range(n_processes))
    best_dim_score = -np.inf
    best_diff_score = -np.inf
    for order in itertools.permutations(range(n_processes)):
        dim_scores = [safe_pearsonr(lhs[:, idx], rhs[:, order[idx]]) for idx in range(n_processes)]
        dim_mean = float(np.mean(dim_scores))
        if n_processes == 1:
            diff_mean = dim_mean
        else:
            pair_scores = []
            for i in range(n_processes):
                for j in range(i + 1, n_processes):
                    pair_scores.append(
                        safe_pearsonr(
                            lhs[:, i] - lhs[:, j],
                            rhs[:, order[i]] - rhs[:, order[j]],
                        )
                    )
            diff_mean = float(np.mean(pair_scores)) if pair_scores else dim_mean
        score = 0.5 * (dim_mean + diff_mean)
        if score > 0.5 * (best_dim_score + best_diff_score):
            best_order = list(order)
            best_dim_score = dim_mean
            best_diff_score = diff_mean
    return best_order, best_dim_score, best_diff_score


def compute_repetition_agreement(prediction_frames: list[pd.DataFrame], n_processes: int) -> dict[str, Any]:
    if len(prediction_frames) <= 1:
        return {
            "mean_dimension_correlation": 1.0,
            "mean_difference_correlation": 1.0,
            "per_repetition_score": [1.0],
            "best_repetition_index": 0,
        }

    process_columns = [f"r{i + 1}" for i in range(n_processes)]
    aligned_arrays = [frame[process_columns].to_numpy() for frame in prediction_frames]
    repetition_scores = np.zeros((len(prediction_frames), len(prediction_frames), 2), dtype=float)

    for i in range(len(aligned_arrays)):
        for j in range(i + 1, len(aligned_arrays)):
            _, dim_score, diff_score = align_process_latents(aligned_arrays[i], aligned_arrays[j])
            repetition_scores[i, j, 0] = dim_score
            repetition_scores[j, i, 0] = dim_score
            repetition_scores[i, j, 1] = diff_score
            repetition_scores[j, i, 1] = diff_score

    denominator = max(len(prediction_frames) * (len(prediction_frames) - 1), 1)
    mean_dim = repetition_scores[:, :, 0].sum() / denominator
    mean_diff = repetition_scores[:, :, 1].sum() / denominator

    per_rep_scores = []
    for idx in range(len(prediction_frames)):
        mask = [j for j in range(len(prediction_frames)) if j != idx]
        per_rep_scores.append(float(repetition_scores[idx, mask, :].mean()))
    return {
        "mean_dimension_correlation": float(mean_dim),
        "mean_difference_correlation": float(mean_diff),
        "per_repetition_score": per_rep_scores,
        "best_repetition_index": int(np.argmax(per_rep_scores)),
    }


def compute_age_metrics(prediction_frame: pd.DataFrame, n_processes: int) -> dict[str, Any]:
    frame = prediction_frame.loc[prediction_frame["age"].notna()].copy()
    if frame.empty:
        return {
            "age_latent_age_correlation": 0.0,
            "process_age_correlations": {},
            "residual_process_age_correlations": {},
            "mean_absolute_process_age_correlation": 0.0,
            "mean_absolute_residual_process_age_correlation": 0.0,
            "composite_score": 0.0,
        }

    ages = frame["age"].to_numpy(dtype=float)
    age_latent = frame["age_latent"].to_numpy(dtype=float)
    age_corr = safe_pearsonr(ages, age_latent)

    process_corrs: dict[str, float] = {}
    residual_corrs: dict[str, float] = {}
    lr = LinearRegression()

    for idx in range(n_processes):
        column = f"r{idx + 1}"
        values = frame[column].to_numpy(dtype=float)
        process_corrs[column] = safe_pearsonr(ages, values)
        lr.fit(age_latent.reshape(-1, 1), values)
        residuals = values - lr.predict(age_latent.reshape(-1, 1))
        residual_corrs[column] = safe_pearsonr(ages, residuals)

    mean_abs_corr = float(np.mean(np.abs(list(process_corrs.values()))))
    mean_abs_residual = float(np.mean(np.abs(list(residual_corrs.values()))))
    composite = float(age_corr - mean_abs_residual - 0.5 * mean_abs_corr)
    return {
        "age_latent_age_correlation": age_corr,
        "process_age_correlations": process_corrs,
        "residual_process_age_correlations": residual_corrs,
        "mean_absolute_process_age_correlation": mean_abs_corr,
        "mean_absolute_residual_process_age_correlation": mean_abs_residual,
        "composite_score": composite,
    }


def summarize_latent_sensitivity(
    *,
    age_sensitivity_pct_mean: float,
    process_sensitivity_pct_means: dict[str, float],
    process_separation_pct_mean: float,
) -> dict[str, Any]:
    mean_process_sensitivity = float(np.mean(list(process_sensitivity_pct_means.values()))) if process_sensitivity_pct_means else 0.0
    quality = float(
        np.log1p(max(age_sensitivity_pct_mean, 0.0))
        + np.log1p(max(mean_process_sensitivity, 0.0))
        + 0.5 * np.log1p(max(process_separation_pct_mean, 0.0))
    )
    return {
        "age_sensitivity_pct_mean": float(age_sensitivity_pct_mean),
        "process_sensitivity_pct_means": process_sensitivity_pct_means,
        "mean_process_sensitivity_pct_mean": mean_process_sensitivity,
        "process_separation_pct_mean": float(process_separation_pct_mean),
        "latent_sensitivity_score": quality,
    }
