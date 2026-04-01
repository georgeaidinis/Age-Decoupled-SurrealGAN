# Pipeline Usage

## Core Commands

Prepare data:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli prepare-data
```

Train:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli train
```

Train with override config:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli --config path/to/config.toml train
```

Resume an interrupted run:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config path/to/config.toml \
  train --resume-run-dir runs/<existing_run_dir>
```

Tune:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli tune
```

Resume tuning:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/overnight_tune.toml \
  tune
```

If the tuning config uses SQLite storage with `resume_if_exists = true`, rerunning the same command continues the Optuna study.

Serve the API:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli serve
```

Run the frontend:

```bash
cd webui
npm install
npm run dev
```

During local development, the Vite server proxies API requests to `127.0.0.1:8000`. If your API runs elsewhere, set `VITE_API_BASE_URL` or edit [`webui/vite.config.ts`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/webui/vite.config.ts).

## Important Files

- `src/age_decoupled_surrealgan/data/prepare.py`: preprocessing and split generation
- `src/age_decoupled_surrealgan/model.py`: model definition
- `src/age_decoupled_surrealgan/trainer.py`: training loop
- `src/age_decoupled_surrealgan/losses.py`: implemented objectives
- `src/age_decoupled_surrealgan/metrics.py`: selection and analysis metrics

## Recommended Experiment Pattern

1. Regenerate processed data if cohort or feature rules changed.
2. Run a short quickstart config for smoke testing.
3. Run the default config for a baseline.
4. Run ablations by changing only one or two loss weights at a time.
5. Compare:
   - validation composite score
   - validation quality score
   - latent sensitivity metrics
   - agreement across repetitions
   - ID vs OOD behavior

## Runtime Notes

- On Apple Silicon with MPS, GPU utilization is usually the main limiter for this ROI model rather than CPU saturation.
- `num_workers > 0` and `persistent_workers = true` can help only if data loading becomes a bottleneck.
- The default checkpoint cadence now uses `save_every = 0` with `target_regular_checkpoints = 5`, which keeps artifact volume manageable while preserving training history.

## Ready-Made Ablation Configs

The repository includes minimal override configs in `src/age_decoupled_surrealgan/configs/ablations/`:

- `change_magnitude.toml`
- `low_activation_identity.toml`
- `process_age_correlation.toml`

Example:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/ablations/change_magnitude.toml \
  train
```

## Overnight Configs

For longer laptop runs:

- `src/age_decoupled_surrealgan/configs/overnight_train.toml`
- `src/age_decoupled_surrealgan/configs/overnight_tune.toml`
- `scripts/run_overnight_laptop.sh`
