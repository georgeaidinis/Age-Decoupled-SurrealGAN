# Metrics And Logging

This document explains the metrics written by the age-decoupled pipeline and where they appear in TensorBoard, terminal logs, and text-readable artifacts.

## Validation Metrics

### Age latent to age correlation

$$
\rho_a = \rho(a, \mathrm{age})
$$

Interpretation:
- Higher is better.
- If this is near zero, the age branch is not actually capturing chronological age.

### Process-age correlations

For each process latent:

$$
\rho_i = \rho(r_i, \mathrm{age})
$$

Interpretation:
- Large absolute values mean that process latents still carry age information.

### Residual process-age correlations

For each process latent, regress out the age latent:

$$
r_i^{res} = r_i - \hat r_i(a)
$$

and compute:

$$
\rho_i^{res} = \rho(r_i^{res}, \mathrm{age})
$$

Interpretation:
- This is the most useful “age leakage” summary in the current extension.
- Smaller absolute values are better.

### Composite score

The current checkpoint selection metric is:

$$
\mathrm{composite} =
\rho(a, \mathrm{age})
- \mathrm{mean}_i |\rho_i^{res}|
- 0.5 \cdot \mathrm{mean}_i |\rho_i|
$$

Interpretation:
- Higher is better.
- It rewards age capture in the explicit age latent and penalizes age leakage into process latents.

### Latent sensitivity metrics

These metrics probe whether the direct generator path actually responds when age or process latents are changed.

Age sensitivity:

$$
S_a = \frac{100}{nd}\sum_{j=1}^{n}\sum_{k=1}^{d}
\frac{\left|G(x_j, a_{\max}, 0)_k - G(x_j, a_{\min}, 0)_k\right|}{|x_{jk}| + \epsilon}
$$

Process sensitivity for each latent:

$$
S_{r_i} = \frac{100}{nd}\sum_{j=1}^{n}\sum_{k=1}^{d}
\frac{\left|G(x_j, a_{anchor}, e_i)_k - G(x_j, a_{anchor}, 0)_k\right|}{|x_{jk}| + \epsilon}
$$

Process separation:

$$
S_{sep} = \mathrm{mean}_{i<j}\;
\frac{100}{nd}\sum_{m=1}^{n}\sum_{k=1}^{d}
\frac{\left|G(x_m, a_{anchor}, e_i)_k - G(x_m, a_{anchor}, e_j)_k\right|}{|x_{mk}| + \epsilon}
$$

Interpretation:
- Higher means the generator is more responsive to latent controls.
- Near-zero values indicate a collapsed or weakly conditioned generator branch even if the encoder/decomposer metrics look good.

### Quality score

The new validation score used for overnight runs is:

$$
\mathrm{quality} =
\mathrm{composite}
+ 0.25 \cdot
\left[
\log(1 + S_a) +
\log(1 + \overline{S_r}) +
0.5 \log(1 + S_{sep})
\right]
$$

where

$$
\overline{S_r} = \mathrm{mean}_i\; S_{r_i}
$$

Interpretation:
- Higher is better.
- It keeps the age-decoupling goal of `composite_score` but adds pressure toward a generator that actually responds to latent manipulation.

## Agreement Metrics

Agreement is still computed across repetitions using permutation alignment of process dimensions:

- dimension-correlation
- difference-correlation

These are conceptually inherited from the original SurrealGAN code.

## TensorBoard Layout

The trainer now groups TensorBoard scalars under readable namespaces:

- `loss/discriminator/*`
- `loss/generator/*`
- `loss/supervision/*`
- `loss/reconstruction/*`
- `loss/disentanglement/*`
- `loss/regularization/*`
- `loss/ablation/*`
- `state/latents/*`
- `state/change/*`
- `metric/validation/*`
- `selection/validation/*`
- `docs/*`

The `docs/*` text tabs embed the objective and metric explanation documents directly into TensorBoard.

For repeated runs, TensorBoard now writes one event stream per repetition:

- `tensorboard/repetition_00`
- `tensorboard/repetition_01`
- `tensorboard/repetition_02`

Each repetition contains the full metric set under the same tag names. In practice this gives:

- one TensorBoard run per repetition
- one plot per metric tag
- consistent colors for a repetition across all plots
- much less left-panel clutter than one subrun per metric

This is the intended default because the repetition is the natural comparison unit inside a training run.

## Terminal Logging

Every run now begins with a concise startup summary:

```text
Experiment: ...
Config: ...
Device: ...
Run setup: repetitions=..., epochs=..., batch_size=..., lr=..., d_lr=...
Runtime: num_workers=..., persistent_workers=..., use_amp=..., compile_model=...
Architecture: n_features=..., n_processes=..., encoder=[...], generator=[...], ...
Checkpoint epochs: [...] plus best-per-repetition checkpoint
```

Every epoch then prints a condensed summary line:

```text
[rep 1/3] [epoch 5/50] train: G=... D=... adv=... age=... decomp=... val: comp=... age_corr=... resid_age=...
```

This is also written to:

- `runs/<run>/logs/train.log`

## Checkpoint Cadence

By default, the trainer no longer saves a checkpoint at every epoch.

If `save_every > 0`, checkpoints are written every `save_every` epochs.

If `save_every = 0`, the trainer computes an interval automatically so that the total number of regular checkpoints is approximately:

$$
\text{target\_regular\_checkpoints}
$$

with the final epoch always included. A separate best checkpoint is still saved for each repetition.

## Resume Support

Training can be resumed from an interrupted run directory:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config path/to/config.toml \
  train --resume-run-dir runs/<existing_run_dir>
```

The trainer will:
- skip finished repetitions
- restore the latest regular checkpoint for the active repetition
- restore optimizer and scaler state when present
- continue appending metrics and logs into the same run directory

Optuna tuning resumes automatically when the tuning config uses a persistent SQLite storage and `resume_if_exists = true`.

## Text-Readable Artifacts

Each run now writes:

- `runs/<run>/run_summary.json`
- `runs/<run>/metrics/epoch_history.csv`
- `runs/<run>/logs/epoch_history.jsonl`
- `runs/<run>/metrics/repetition_summary.csv`
- `runs/<run>/metrics/split_metrics.csv`
- per-split metric JSONs in `runs/<run>/metrics/*.json`

These are intended to be readable in a text editor without opening TensorBoard.

## Losses Logged But Disabled By Default

The following losses are always computed and logged, but default to zero weight:

- `change_mag`
- `low_identity`
- `process_age_corr`
- `process_sparse`

This is deliberate so that ablation runs can be turned on by config without needing code changes.
