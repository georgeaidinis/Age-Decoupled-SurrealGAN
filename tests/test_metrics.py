from __future__ import annotations

import pandas as pd

from age_decoupled_surrealgan.metrics import compute_age_metrics, compute_repetition_agreement


def test_compute_repetition_agreement_single_frame_returns_identity_scores():
    frame = pd.DataFrame({"r1": [0.1, 0.2], "r2": [0.3, 0.4]})
    metrics = compute_repetition_agreement([frame], n_processes=2)
    assert metrics["best_repetition_index"] == 0
    assert metrics["mean_dimension_correlation"] == 1.0


def test_compute_age_metrics_returns_expected_keys():
    frame = pd.DataFrame(
        {
            "age": [20, 30, 40, 50],
            "age_latent": [0.1, 0.3, 0.5, 0.7],
            "r1": [0.2, 0.1, 0.3, 0.4],
            "r2": [0.4, 0.3, 0.2, 0.1],
        }
    )
    metrics = compute_age_metrics(frame, n_processes=2)
    assert "composite_score" in metrics
    assert set(metrics["process_age_correlations"]) == {"r1", "r2"}
