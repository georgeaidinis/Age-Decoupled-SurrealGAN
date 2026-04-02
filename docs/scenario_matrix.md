# Scenario Matrix

This document describes the current train and tuning scenarios for large-scale SLURM sweeps.

## Train Scenarios

- `balanced_4`
  - Four process latents, balanced reconstruction/disentanglement/generator sensitivity.
- `balanced_5`
  - Same as `balanced_4`, but with five process latents to stay closer to the original SurrealGAN paper.
- `faithful_surreal_5`
  - Stronger identity, change-magnitude, low-activation identity, and monotonicity terms.
  - This is the closest scenario to the original SurrealGAN regularization style.
- `sensitivity_push_4`
  - Lower identity pressure and stronger adversarial/sensitivity pressure.
  - Intended to test whether the direct generator path can be made more responsive.
- `shrinkage_directional_4`
  - Four processes with explicit shrinkage bias and `directional_quality_score` model selection.
- `shrinkage_directional_5`
  - Five-process version of the directional shrinkage scenario.
- `sparse_disentangled_4`
  - Stronger age-adversary, covariance, process-age-correlation, and sparsity penalties.
  - Intended to test whether clearer process axes emerge with stronger disentanglement.
- `generator_freedom_5`
  - Lower reconstruction and identity pressure, higher adversarial/sensitivity pressure, wider networks.
  - Intended to test whether the generator is currently overconstrained.
- `change_regularized_4`
  - Adds explicit change-magnitude regularization and stronger low-activation identity.
  - Intended to keep transformations small and smooth while preserving latent responsiveness.

## Tune Scenarios

- `balanced_4_5`
  - Broad tuning around the current balanced regime with `n_processes in {4, 5}`.
- `directional_4_5`
  - Tunes toward `directional_quality_score`, penalizing positive age/process growth.
- `faithful_surreal_5`
  - Five-process tuning with stronger original-SurrealGAN-style regularization ranges.
- `sensitivity_push_4_5`
  - Emphasizes adversarial strength, generator sensitivity, and reduced identity pressure.
- `sparse_disentangle_4_5`
  - Searches stronger disentanglement and sparsity settings.
- `shrinkage_5`
  - Five-process directional shrinkage tuning with explicit growth penalties.

## Selection Metrics

- `composite_score`
  - Age capture minus age leakage into process latents.
- `quality_score`
  - `composite_score` plus generator sensitivity.
- `directional_quality_score`
  - `composite_score` plus direction-aware generator sensitivity.
  - Penalizes positive ROI changes under age/process pushes.

## SLURM Sweep Files

- Train config list:
  - `scripts/slurm/train_scenarios.txt`
- Tune config list:
  - `scripts/slurm/tune_scenarios.txt`
- Train array job:
  - `scripts/slurm/train_array.sh`
- Tune array job:
  - `scripts/slurm/tune_array.sh`
- Submit train array:
  - `scripts/slurm/submit_train_scenarios.sh`
- Submit tune array:
  - `scripts/slurm/submit_tune_scenarios.sh`
- Submit both arrays:
  - `scripts/slurm/submit_all_scenarios.sh`
