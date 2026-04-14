# Metrics And Logging

This document explains the metrics emitted by the **current redesigned normalized additive sampled-latent model** and how they are used during training, tuning, and model selection.

The main point to keep in mind is this:

- some metrics measure **age disentanglement**,
- some metrics measure **generator responsiveness**,
- some metrics measure **process collapse / redundancy**,
- and **agreement** is a separate reproducibility measure across repeated trainings.

Those are not interchangeable.

## 1. What Is Computed Where?

There are three different metric stages.

### Stage A: per-epoch validation metrics

During each repetition, after every epoch, the trainer evaluates the current model on the validation split:

$$
\texttt{trainer.\_evaluate\_model(model, "val")}
$$

This produces:

- age/disentanglement metrics,
- generator sensitivity metrics,
- derived selection scores such as `quality_score` and `collapse_aware_quality_score`.

These metrics are used to decide the **best epoch within a repetition**.

### Stage B: repetition agreement

After all repetitions of a single run finish, the trainer compares the saved validation predictions from the best epoch of each repetition:

$$
\texttt{aggregate\_repetition\_predictions(repetition\_val\_paths, K)}.
$$

This produces the agreement summary and selects the **representative repetition** for that run.

### Stage C: final split metrics

After the representative repetition is selected, its chosen checkpoint is loaded and evaluated on:

- `train`
- `val`
- `id_test`
- `ood_test`
- `application`

Those split-level metrics are saved into:

- `run_summary.json`
- `metrics/split_metrics.csv`
- `metrics/<split>.json`

These are the numbers you compare across completed runs.

## 2. Age / Disentanglement Metrics

These come from [`src/age_decoupled_surrealgan/metrics.py`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/metrics.py).

### Age latent to chronological age correlation

$$
\rho_{age} = \rho(\hat a, \mathrm{age})
$$

Interpretation:

- high is good,
- this asks whether the explicit age latent is actually tracking chronological age.

### Process-age correlation

For each process latent:

$$
\rho_k = \rho(\hat r_k, \mathrm{age})
$$

We summarize:

$$
\overline{\rho}_{proc} = \mathrm{mean}_{k} |\rho_k|.
$$

Interpretation:

- low is good,
- this measures how much raw age leakage remains in the process latents.

### Residual process-age correlation

For each process latent, regress it on the age latent and compute correlation of the residual with chronological age:

$$
\hat r_k^{res} = \hat r_k - \widehat{\mathbb{E}}[\hat r_k \mid \hat a]
$$

$$
\rho_k^{res} = \rho(\hat r_k^{res}, \mathrm{age})
$$

with summary:

$$
\overline{\rho}_{proc}^{res} = \mathrm{mean}_{k} |\rho_k^{res}|.
$$

Interpretation:

- low is good,
- this is the stricter age-leakage metric because it asks whether process latents still track age **after accounting for the explicit age latent**.

### Subject-level process collapse

Pairwise process-latent correlation:

$$
C_{latent} = \mathrm{mean}_{i<j} |\rho(\hat r_i, \hat r_j)|.
$$

Interpretation:

- low is good,
- if this is large, the model is turning multiple process latents into the same scalar.

This metric is especially important because several earlier runs looked superficially clean on age metrics while all \(r_k\) were effectively duplicates.

### Composite score

The base age-decoupling score is:

$$
\mathrm{composite}
= \rho_{age} - \overline{\rho}_{proc}^{res} - 0.5\,\overline{\rho}_{proc}.
$$

Interpretation:

- high is better,
- this rewards age capture,
- penalizes age leakage into process latents,
- but says **nothing** about whether the generator is actually producing meaningful process effects.

That last point is why `composite_score` alone is not enough for this project.

## 3. Generator-Responsiveness Metrics

These are computed on sampled `ref` subjects in normalized training space but reported in **raw ROI-relative percent units**.

They are built in [`src/age_decoupled_surrealgan/trainer.py`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/trainer.py), mainly in:

- `_latent_sensitivity_batch_metrics`
- `_compute_latent_sensitivity_metrics`

### Age sensitivity

Let \(a_{min}\) and \(a_{max}\) be the configured minimum and maximum normalized ages. Then

$$
S_{age}
= \frac{100}{nd}\sum_{j=1}^{n}\sum_{\ell=1}^{d}
\frac{|G(x_j, a_{max}, 0)_\ell - G(x_j, a_{min}, 0)_\ell|}{|x^{raw}_{j\ell}| + \epsilon}.
$$

Interpretation:

- high is better,
- this asks whether the generator actually changes anatomy when age changes.

### Process sensitivity

For process \(k\), at anchor age \(a_0\):

$$
S_{proc,k}
= \frac{100}{nd}\sum_{j=1}^{n}\sum_{\ell=1}^{d}
\frac{|G(x_j, a_0, e_k)_\ell - G(x_j, a_0, 0)_\ell|}{|x^{raw}_{j\ell}| + \epsilon}.
$$

and the mean process sensitivity is:

$$
\overline{S}_{proc} = \mathrm{mean}_k S_{proc,k}.
$$

Interpretation:

- high is better,
- if this is near zero, the process latents are effectively silent.

### Process separation

Define the isolated process response:

$$
g_k(x) = G(x, a_0, e_k) - G(x, a_0, 0).
$$

Then:

$$
S_{sep}
= \mathrm{mean}_{i<j}
\frac{100}{nd}
\sum_{x,\ell}
\frac{|g_i(x)_\ell - g_j(x)_\ell|}{|x^{raw}_{\ell}| + \epsilon}.
$$

Interpretation:

- high is better,
- this asks whether different process sliders do different things.

### Process-pattern correlation

Even when amplitudes are nonzero, two process bases can still encode the same pattern. We therefore compute:

$$
C_{pattern} = \mathrm{mean}_{i<j} |\rho(\bar g_i, \bar g_j)|
$$

where \(\bar g_k\) is the average isolated process response vector across sampled `ref` subjects.

Interpretation:

- low is better,
- this is a generator-space redundancy metric.

### Positive-change summaries

For directional diagnostics:

$$
P_{age}^{+} = \mathrm{mean}\,\max(g_{age}^{raw}, 0)
$$

$$
P_{proc}^{+} = \mathrm{mean}_k \mathrm{mean}\,\max(g_k^{raw}, 0)
$$

These are not always “bad” biologically, because ventricles and CSF can enlarge, but they are useful diagnostics when applying shrinkage-aware priors.

## 4. Derived Selection Scores

These are computed in [`src/age_decoupled_surrealgan/trainer.py`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/trainer.py) after combining age metrics with sensitivity metrics.

### Latent sensitivity score

Internally this is a compressed generator-quality summary built from:

- age sensitivity,
- mean process sensitivity,
- process separation,
- process-pattern correlation.

Conceptually:

$$
\mathrm{latent\_sensitivity\_score}
\uparrow
\quad \text{when} \quad
S_{age}, \overline{S}_{proc}, S_{sep} \uparrow
\quad \text{and} \quad
C_{pattern} \downarrow.
$$

### Quality score

$$
\mathrm{quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{latent\_sensitivity\_score}
 - 0.5 \cdot C_{latent}.
$$

Interpretation:

- adds generator responsiveness to the disentanglement objective,
- also penalizes subject-level latent collapse.

### Directional quality score

This is the same idea, but uses a direction-aware sensitivity summary:

$$
\mathrm{directional\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{directional\_latent\_sensitivity\_score}
 - 0.5 \cdot C_{latent}.
$$

Use this when testing shrinkage-biased scenarios.

### Collapse-aware quality score

This is the main redesigned selection score:

$$
\mathrm{collapse\_aware\_quality}
= \mathrm{composite}
 + 0.25 \cdot \mathrm{collapse\_aware\_latent\_sensitivity\_score}
 - 0.75 \cdot C_{latent}.
$$

Interpretation:

- reward age capture,
- reward generator responsiveness,
- penalize process collapse more strongly.

This is currently the best default because it addresses the actual failure mode of the earlier model family.

## 5. Agreement Metrics

Agreement is computed **after** the best epoch of each repetition has already been selected.

It is implemented in:

- [`src/age_decoupled_surrealgan/evaluation.py`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/evaluation.py)
- [`src/age_decoupled_surrealgan/metrics.py`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/metrics.py)

The code permutes process orderings across repetitions to find the best alignment, then computes:

### Dimension agreement

$$
A_{dim} = \text{mean matched-dimension correlation}
$$

### Difference agreement

$$
A_{diff} = \text{mean matched pair-difference correlation}
$$

### Best repetition index

For each repetition, the code averages how well it agrees with the others and picks:

$$
r^* = \arg\max_r \text{mean agreement of repetition } r.
$$

This selected repetition becomes the representative checkpoint for that run.

## 6. Important Clarification: Agreement vs Selection Metric

These answer different questions.

- `collapse_aware_quality_score`: how good is a given epoch at age disentanglement + responsive, non-collapsed generation?
- agreement: how reproducible are the discovered process axes across repeated trainings?

The pipeline currently uses **both**:

1. select best epoch within each repetition by `monitor_metric`,
2. select best repetition across repetitions by agreement.

So the final chosen model is **not** “the highest agreement epoch”. It is:

- the checkpoint from the repetition with the highest cross-repetition agreement,
- where each repetition contributes its own internally best epoch.

## 7. Logging Outputs

### Terminal and log files

Training prints:

- startup summary,
- config path,
- device,
- architecture,
- checkpoint schedule,
- per-epoch train/validation summaries,
- train time, validation time, total epoch time, and elapsed repetition time.

Example:

```text
[rep 2/5] [epoch 37/100] [t_train=00:01:44 t_val=00:00:18 t_epoch=00:02:02 t_rep=01:15:31] train: G=1.4821 D=0.6324 adv=0.5510 age=0.0422 decomp=0.1774 val: comp=0.6112 qual=1.7344 sens=4.3001 age_corr=0.8920 resid_age=0.0715
```

Saved text-readable outputs:

- `logs/train.log`
- `logs/epoch_history.jsonl`
- `metrics/epoch_history.csv`
- `metrics/repetition_summary.csv`
- `metrics/split_metrics.csv`
- `metrics/run_summary.md`

### TensorBoard

Each repetition is logged as a separate TensorBoard run:

- `tensorboard/repetition_00`
- `tensorboard/repetition_01`
- ...

This means:

- one consistent color per repetition across all plots,
- one plot per metric,
- no metric-by-metric run explosion.

Important TensorBoard families:

- `loss/*`
- `state/*`
- `metric/validation/*`
- `selection/validation/*`

## 8. What Metrics Matter Most in Practice?

For this project, the most informative plots and tables are usually:

1. `age_latent_age_correlation`
2. `mean_absolute_residual_process_age_correlation`
3. `mean_process_sensitivity_pct_mean`
4. `process_separation_pct_mean`
5. `process_pattern_correlation_abs_mean`
6. `process_latent_pairwise_correlation_abs_mean`
7. `collapse_aware_quality_score`
8. agreement (`mean_dimension_correlation`, `mean_difference_correlation`)

That set gives you:

- age quality,
- age leakage,
- whether the generator is alive,
- whether the process axes are distinct,
- whether they are reproducible.
