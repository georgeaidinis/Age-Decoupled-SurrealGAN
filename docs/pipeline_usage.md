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

Tune:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli tune
```

Serve the API:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli serve
```

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
   - agreement across repetitions
   - ID vs OOD behavior

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
