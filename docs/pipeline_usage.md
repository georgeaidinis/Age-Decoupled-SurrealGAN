# Pipeline Usage

This page reflects the **redesigned normalized additive sampled-latent pipeline**.

## 1. Prepare Data

Whenever you change ROI filtering, cohort definitions, or normalization settings, regenerate the processed artifacts:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli prepare-data
```

Important outputs under [`artifacts/data/processed`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/artifacts/data/processed):

- `train.csv`, `val.csv`, `id_test.csv`, `ood_test.csv`, `application.csv`
- `reference_template.csv`
- `reference_template_normalized.csv`
- `normalization_stats.csv`
- `roi_metadata.csv`
- `split_manifest.json`

The redesigned model assumes these normalization artifacts exist.

## 2. Baseline Training

Default training:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli train
```

Redesign reference config:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/redesign_train.toml \
  train
```

Quick smoke test:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/quickstart.toml \
  train
```

Resume an interrupted run:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/redesign_train.toml \
  train --resume-run-dir runs/<existing_run_dir>
```

## 3. Hyperparameter Tuning

Baseline redesign tuner:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/redesign_tune.toml \
  tune
```

Tuning resumes automatically when the same Optuna study/storage path is reused.

The redesign-era large sweeps use the scenario configs under:

- [`src/age_decoupled_surrealgan/configs/scenarios/train`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/configs/scenarios/train)
- [`src/age_decoupled_surrealgan/configs/scenarios/tune`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/src/age_decoupled_surrealgan/configs/scenarios/tune)

## 4. Backfill Analysis Artifacts

If a run was trained before a new analysis feature existed, backfill its population patterns and summaries:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli backfill-run-artifacts --force
```

For a single run:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  backfill-run-artifacts --run-dir runs/<run_dir> --force
```

The backfill command now prints progress in the form:

```text
[backfill 3/18] processing runs/...
```

## 5. Serve the API and GUI

Start the backend:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli serve
```

Enable verbose API debugging:

```bash
AGE_DECOUPLED_SURREALGAN_DEBUG=1 PYTHONPATH=src python -m age_decoupled_surrealgan.cli serve
```

Frontend:

```bash
cd webui
npm install
npm run dev
```

The GUI now supports:

- subject mode
- precomputed population-factor mode
- signed ROI overlays
- crosshair ROI readout
- latent-space exploration from saved prediction CSVs

## 6. What a Run Contains

Each run directory contains:

- `resolved_config.json`
- `run_summary.json`
- `metrics/*.csv`
- `logs/train.log`
- `predictions/<split>.csv`
- `analysis/`
- `tensorboard/repetition_*`
- repetition checkpoints and a selected best checkpoint

The GUI loads the checkpoint stored in:

```text
run_summary.json -> selected_checkpoint
```

## 7. Practical Workflow

Recommended loop:

1. `prepare-data`
2. run one baseline redesign train
3. inspect TensorBoard
4. inspect population patterns in the GUI
5. submit redesign train scenario sweep
6. submit redesign tune scenario sweep around the most promising families
7. backfill older runs if analysis features changed

## 8. Most Important Config Knobs

### Data

- `data.roi_normalization`
- `data.roi_normalization_clip`
- `data.ref_min_age`, `data.ref_max_age`
- `data.tar_min_age`, `data.tar_max_age`
- `data.holdout_study`

### Model

- `model.n_processes`
- `model.encoder_hidden_dims`
- `model.generator_hidden_dims`
- `model.decomposer_hidden_dims`
- `model.process_separation_margin`

### Training

- `training.repetitions`
- `training.epochs`
- `training.batch_size`
- `training.learning_rate`
- `training.discriminator_learning_rate`
- `training.monitor_metric`
- `training.sampled_process_one_hot_only`

### Losses

- `losses.age_supervision`
- `losses.age_adversary`
- `losses.latent_reconstruction`
- `losses.decomposition`
- `losses.identity`
- `losses.process_sensitivity`
- `losses.generator_process_separation`
- `losses.generator_process_redundancy`
- `losses.process_latent_pairwise_correlation`

### Tuning

- `tuning.objective_metric`
- `tuning.n_processes_options`
- `tuning.width_options`
- all `*_min` / `*_max` ranges for the loss families above
