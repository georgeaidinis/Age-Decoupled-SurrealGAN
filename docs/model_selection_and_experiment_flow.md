# Model Selection And Experiment Flow

This document gives the end-to-end algorithm for how experiments are run and how a final model is selected.

## 1. What This Document Outlines

- data preparation,
- normalization,
- training repetitions,
- within-run selection,
- across-run comparison,
- tuning,
- post-hoc analysis,
- GUI-based qualitative review.

## 2. End-to-End Timeline

The intended order is:

### Step 1. Prepare processed data

Run:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli prepare-data
```

This:

1. canonicalizes the dataset,
2. creates the split files,
3. fits REF normalization on `train ref`,
4. writes the normalized reference template,
5. writes ROI metadata.

### Step 2. Choose a model family / scenario

Examples:

- `redesign_balanced_5`
- `redesign_generator_push_5`
- `redesign_disentangle_5`

At this stage, you are choosing a **scientific hypothesis** about the loss balance and architectural regime.

### Step 3. Train one run of that scenario

Each run trains:

$$
R
$$

repetitions.

For each repetition:

1. train epoch by epoch on `train`,
2. evaluate each epoch on `val`,
3. choose the best epoch by `training.monitor_metric`.

So each repetition yields **one best checkpoint**.

### Step 4. Compute agreement across repetitions

Take the saved validation predictions from each repetition’s best checkpoint.

Then:

1. align process axes by permutation,
2. compute matched-dimension agreement,
3. compute pair-difference agreement,
4. choose the repetition that agrees best with the others.

This selected repetition becomes the **representative checkpoint** for the run.

### Step 5. Evaluate the selected checkpoint on all splits

The representative checkpoint is then applied to:

- `train`
- `val`
- `id_test`
- `ood_test`
- `application`

This writes:

- prediction CSVs,
- split-level metrics,
- run summary,
- population patterns,
- analysis artifacts.

### Step 6. Compare completed runs

Across many runs, compare:

1. validation selection score,
2. agreement,
3. process-pattern distinctness,
4. population patterns,
5. qualitative plausibility,
6. `id_test` / `ood_test` behavior.

This is where scientific judgment enters. There is no single scalar that replaces all interpretation.

### Step 7. Tune inside the most promising family

Optuna is used **after** a family looks plausible.

It searches:

- widths,
- dropout,
- learning rates,
- loss weights,
- usually \(K \in \{4,5\}\) right now.

Each Optuna trial is just another training run following the same selection pipeline above.

### Step 8. Post-hoc analysis

Once a family and tuned configuration look promising:

1. inspect population patterns,
2. inspect latent-space structure,
3. inspect top ROIs and sign summaries,
4. inspect diagnosis/study associations,
5. study how `r`-indices correlate with age, diagnosis, MMSE, etc.

This is where the scientific interpretation is built.

## 3. The Two-Stage Selection Rule Inside One Run

This is the most important detail.

### Stage A: choose the best epoch

Inside each repetition:

$$
e_r^* = \arg\max_e \mathrm{monitor\_metric}(r,e,\text{val})
$$

where `monitor_metric` is usually:

- `collapse_aware_quality_score`
- or `directional_quality_score`

### Stage B: choose the representative repetition

Across repetitions:

$$
r^* = \arg\max_r \mathrm{agreement}(r; e_1^*, \dots, e_R^*)
$$

The final selected checkpoint is:

$$
\text{checkpoint}^* = (r^*, e_{r^*}^*).
$$

So:

- the best epoch is chosen by a validation metric,
- the best repetition is chosen by agreement.

This is the core selection algorithm.

## 4. What Tuning Actually Optimizes

Optuna does **not** choose the final model directly across all scientific criteria.

It optimizes one scalar:

$$
\texttt{tuning.objective\_metric}
$$

which is currently usually:

$$
\mathrm{collapse\_aware\_quality\_score}.
$$

That means:

- Optuna is useful for narrowing the search,
- but the final scientific decision should still look at agreement and qualitative interpretability too.

## 5. Are We Done Once Optuna Finds a Best Trial?

No.

A good Optuna score means:

- within that defined objective, the trial looked strong on validation.

It does **not** automatically mean:

- the process factors are reproducible enough,
- the population maps are anatomically sensible,
- the OOD behavior is acceptable,
- the post-hoc scientific story is good.

Those still need to be checked explicitly.

## 6. When Should a Broader K-Sweep Happen?

Only after the family is trustworthy.

Recommended order:

1. settle architecture + loss family,
2. tune hyperparameters,
3. then run a broader \(K\)-sweep.

For example:

$$
K \in \{3,4,5,6,7\}.
$$

At that stage, compare:

- agreement,
- collapse-aware validation score,
- population-pattern interpretability,
- downstream post-hoc usefulness.

That is much more defensible than changing \(K\) while also changing everything else.

## 7. What Counts as a Good Intermediate Result?

Without being overly prescriptive, a strong run should usually show:

- age latent strongly correlated with age,
- low residual age leakage in process latents,
- nontrivial process sensitivity,
- low process-pattern correlation,
- low subject-level process collapse,
- reasonable agreement across repetitions,
- population patterns that are not trivial or nonsensical.

If one of those fails badly, the run is usually not scientifically satisfying even if one summary score looks good.

## 8. Practical Workflow For Notes / Slides

When documenting experiments, the cleanest structure is:

1. data split and normalization setup,
2. architecture used,
3. losses used,
4. selection metric used,
5. number of repetitions,
6. how best epoch was chosen,
7. how best repetition was chosen,
8. held-out evaluation splits,
9. post-hoc analysis outputs,
10. interpretation.

That is the most defensible way to explain what was done and why.
