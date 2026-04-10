# Metrics And Logging

This document explains the metrics emitted by the **redesigned normalized additive sampled-latent model**.

## Core Validation Metrics

### Age capture

$$
\rho_{age} = \rho(\hat a, \mathrm{age})
$$

Interpretation:

- high is good,
- this checks whether the explicit age latent is doing real work.

### Process-age leakage

For each process latent:

$$
\rho_k = \rho(\hat r_k, \mathrm{age})
$$

Residual age leakage is measured after regressing \(\hat r_k\) on \(\hat a\):

$$
\hat r_k^{res} = \hat r_k - \widehat{\mathbb{E}}[\hat r_k \mid \hat a]
$$

$$
\rho_k^{res} = \rho(\hat r_k^{res}, \mathrm{age})
$$

Aggregates:

$$
\overline{\rho}_{proc} = \mathrm{mean}_k |\rho_k|,
\qquad
\overline{\rho}_{proc}^{res} = \mathrm{mean}_k |\rho_k^{res}|.
$$

Smaller is better.

### Subject-level process collapse

Process collapse in latent space is tracked by pairwise correlations:

$$
C_{latent} = \mathrm{mean}_{i < j} |\rho(\hat r_i, \hat r_j)|.
$$

This is important because several earlier run families looked “good” only because every \(r_k\) encoded the same axis.

### Composite score

$$
\mathrm{composite}
= \rho_{age} - \overline{\rho}_{proc}^{res} - 0.5\,\overline{\rho}_{proc}.
$$

This measures age capture minus age leakage, but it does **not** reward generator expressivity by itself.

## Generator-Responsiveness Metrics

These are computed on saved `ref` samples and are what finally separated the redesigned model from the earlier nearly silent generator family.

### Age sensitivity

Let \(a_{min}\) and \(a_{max}\) be the configured minimum and maximum normalized ages. Then:

$$
S_{age}
= \frac{100}{nd}\sum_{j=1}^{n}\sum_{\ell=1}^{d}
\frac{|G(x_j, a_{max}, 0)_\ell - G(x_j, a_{min}, 0)_\ell|}{|x^{raw}_{j\ell}| + \epsilon}.
$$

### Process sensitivity

For process \(k\), with anchor age \(a_0\):

$$
S_{proc,k}
= \frac{100}{nd}\sum_{j=1}^{n}\sum_{\ell=1}^{d}
\frac{|G(x_j, a_0, e_k)_\ell - G(x_j, a_0, 0)_\ell|}{|x^{raw}_{j\ell}| + \epsilon}.
$$

and

$$
\overline{S}_{proc} = \mathrm{mean}_k S_{proc,k}.
$$

### Process-pattern separation

Define the raw process response:

$$
g_k(x) = G(x, a_0, e_k) - G(x, a_0, 0).
$$

Then the mean absolute pairwise separation is:

$$
S_{sep} = \mathrm{mean}_{i < j} \frac{100}{nd} \sum_{j,\ell}
\frac{|g_i(x_j)_\ell - g_j(x_j)_\ell|}{|x^{raw}_{j\ell}| + \epsilon}.
$$

### Process-pattern correlation

The generator can still collapse even when amplitudes are large, so we also measure:

$$
C_{pattern} = \mathrm{mean}_{i<j} |\rho(\bar g_i, \bar g_j)|
$$

where \(\bar g_k\) is the mean process response vector across sampled `ref` subjects.

Smaller is better.

### Directional growth summaries

To detect implausible global growth, we track the positive-change mass under age and process pushes:

$$
P_{age}^{+} = \mathrm{mean}\,\max(g_{age}^{raw}, 0),
\qquad
P_{proc}^{+} = \mathrm{mean}_k \mathrm{mean}\,\max(g_k^{raw}, 0).
$$

These are diagnostics; they matter most when running directional/shrinkage scenarios.

## Selection Metrics

### Quality score

$$
\mathrm{quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{latent\_sensitivity\_score}
 - 0.5 \cdot C_{latent}.
$$

### Directional quality score

$$
\mathrm{directional\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{directional\_latent\_sensitivity\_score}
 - 0.5 \cdot C_{latent}.
$$

### Collapse-aware quality score

This is the main redesigned selection metric:

$$
\mathrm{collapse\_aware\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{collapse\_aware\_latent\_sensitivity\_score}
 - 0.75 \cdot C_{latent}.
$$

Interpretation:

- reward age capture,
- reward non-silent generator behavior,
- penalize subject-level process collapse,
- penalize generator-pattern redundancy.

This is why the redesigned run is more meaningful than the earlier families, even when some older runs looked superficially “clean” on age-only metrics.

## Repetition Agreement

Across repetitions, latent alignment is measured by permutation matching of \(r_1,\dots,r_K\).

Two agreement summaries are saved:

$$
A_{dim} = \text{mean matched dimension correlation}
$$

$$
A_{diff} = \text{mean matched pair-difference correlation}
$$

These matter because the process axes are only meaningful if they are reproducible across repeated fits.

## Population-Level Analysis Artifacts

For the selected checkpoint, the pipeline precomputes isolated factor patterns:

### Age pattern

$$
\Delta^{age}_{pop}
= \mathbb{E}_{x \sim \text{ref}}
\left[G(x, a_{max}, 0) - G(x, a_{min}, 0)\right].
$$

### Process pattern

$$
\Delta^{(k)}_{pop}
= \mathbb{E}_{x \sim \text{ref}}
\left[G(x, a_0, e_k) - G(x, a_0, 0)\right].
$$

Saved artifacts per run:

- CSV ROI tables
- JSON summaries
- prebuilt NIfTI overlays
- top-10 ROI tables
- sign summaries
- repetition stability summaries
- split-wise correlation heatmaps

These are what power the GUI population mode.

## Logging Outputs

### Terminal and log files

Training now prints:

- run start summary,
- device and config path,
- model dimensions,
- checkpoint schedule,
- per-epoch train/validation metrics,
- train time, validation time, total epoch time, and repetition elapsed time.

Example:

```text
[rep 1/5] [epoch 12/100] [t_train=00:01:44 t_val=00:00:18 t_epoch=00:02:02 t_rep=00:25:31] train: G=1.4821 D=0.6324 ...
```

Text-readable outputs include:

- `logs/train.log`
- `logs/epoch_history.jsonl`
- `metrics/epoch_history.csv`
- `metrics/split_metrics.csv`
- `metrics/repetition_summary.csv`
- `metrics/run_summary.md`

### TensorBoard

Each repetition is its own TensorBoard run:

- `tensorboard/repetition_00`
- `tensorboard/repetition_01`
- ...

This keeps one consistent color per repetition across plots.

Important tag groups:

- `loss/*`
- `state/*`
- `metric/validation/*`
- `selection/validation/*`

The most important plots to inspect for the redesigned model are:

1. `metric/validation/age_latent_age_correlation`
2. `metric/validation/mean_process_sensitivity_pct_mean`
3. `metric/validation/process_pattern_correlation_abs_mean`
4. `metric/validation/process_latent_pairwise_correlation_abs_mean`
5. `selection/validation/collapse_aware_quality_score`
