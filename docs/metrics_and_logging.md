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

## Terminal Logging

Every epoch now prints a condensed summary line:

```text
[rep 1/3] [epoch 5/50] train: G=... D=... adv=... age=... decomp=... val: comp=... age_corr=... resid_age=...
```

This is also written to:

- `runs/<run>/logs/train.log`

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
