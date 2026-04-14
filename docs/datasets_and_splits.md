# Datasets And Splits

## 1. Cohort Logic

The preprocessing step starts from [`datasets/cleaned_istaging.csv`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/datasets/cleaned_istaging.csv) and creates a canonical table with:

- `subject_id`
- `study`
- `age`
- `sex`
- `diagnosis_raw`
- `diagnosis_group`
- `cohort_bucket`
- ROI columns

Default cohort rules:

- `ref`: cognitively normal subjects with ages between `ref_min_age` and `ref_max_age`
- `tar`: all subjects with ages between `tar_min_age` and `tar_max_age`
- `excluded`: anyone outside those age / cohort rules

By default:

- `ref = normal, age 20–49`
- `tar = age 50–97`

## 2. Holdout Study

One entire study is held out as a domain-shift / OOD set:

```toml
holdout_study = "HANDLS"
```

This is why `HANDLS` appears in:

- `ood_test.csv`
- `application.csv`

Those splits are not meant to drive hyperparameter selection.

## 3. Processed Splits

### `train.csv`

Use:

- fitting the model,
- fitting REF-based normalization,
- estimating the reference template,
- training losses.

Do **not** use it to decide whether a model generalizes.

### `val.csv`

Use:

- choosing the best epoch within a repetition,
- comparing configs / trials,
- Optuna objective evaluation.

This is the primary model-selection split.

### `id_test.csv`

`id_test` means **in-distribution test**.

Use:

- final evaluation on held-out subjects from the same general training distribution.

Interpretation:

- this checks whether the chosen model generalizes to unseen but in-distribution subjects.

Do **not** tune on it.

### `ood_test.csv`

`ood_test` means **out-of-distribution test**.

Use:

- evaluating robustness to a held-out study.

Interpretation:

- this is the most important split for “does this survive study shift?”

Do **not** tune on it.

### `application.csv`

This currently mirrors `ood_test` by design.

Purpose:

- provide a stable inference-only set for GUI exploration and repeated application-style use.

It exists so you can keep a consistent “analysis/demo” set separate from the training loop.

## 4. Other Processed Files

### `all_rows.csv`

All canonicalized rows, including excluded ones.

### `eligible_rows.csv`

Only rows belonging to `ref` or `tar`.

### `reference_template.csv`

Mean raw ROI vector of the training `ref` cohort.

### `reference_template_normalized.csv`

The same template in normalized space.

### `normalization_stats.csv`

Per-ROI REF normalization parameters:

- mean
- scale
- clipping setup

### `roi_metadata.csv`

Mapping between ROI feature columns and human-readable ROI metadata.

## 5. Which Splits Are Used at Which Stage?

### Data preparation

- normalization is fit on `train ref`
- reference template is estimated on `train ref`

### Training

- sampled `ref` and `tar` batches come from `train.csv`

### Epoch selection

- current epoch is evaluated on `val.csv`
- best epoch within each repetition is chosen from `val`

### Repetition agreement

- agreement is computed from the saved **validation** predictions of each repetition

### Final reporting

After the representative repetition is selected, the checkpoint is evaluated on:

- `train`
- `val`
- `id_test`
- `ood_test`
- `application`

## 6. Practical Rule

When doing science with this project:

- use `val` to compare and select,
- use `id_test` to check held-out in-distribution generalization,
- use `ood_test` to check robustness to study shift,
- use `application` for stable qualitative exploration and demos.

If you keep that rule, you will avoid the most common leakage mistakes.
