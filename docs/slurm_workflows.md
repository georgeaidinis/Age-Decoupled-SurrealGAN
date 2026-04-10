# SLURM Workflows

This page explains the cluster scripts in plain language.

## What SLURM Is Doing

SLURM is the cluster scheduler. You do not run training directly on the login node. Instead, you submit a **job** that says:

- which script to run,
- how many CPUs / how much RAM to reserve,
- which GPU type to reserve,
- how long the job may run.

The scheduler places the job on a suitable compute node when resources become available.

## The Core Commands

### `sbatch`

`sbatch` submits a job script to the scheduler.

Example:

```bash
sbatch scripts/slurm/train_array.sh
```

That does **not** run training immediately in your current shell. It asks SLURM to queue it.

### `squeue`

`squeue` shows queued and running jobs.

Example:

```bash
squeue -u aidinisg
```

## Why We Use Array Jobs

If you have many configs, do **not** loop over `sbatch` hundreds of times from the shell. That creates many separate top-level jobs.

Instead, use one **array job**:

```bash
#SBATCH --array=0-7
```

Then each array task picks one config line from a scenario list file using:

```bash
SLURM_ARRAY_TASK_ID
```

In this project:

- [`scripts/slurm/train_array.sh`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/train_array.sh)
- [`scripts/slurm/tune_array.sh`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/tune_array.sh)

read one config path from:

- [`scripts/slurm/train_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/train_scenarios.txt)
- [`scripts/slurm/tune_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/tune_scenarios.txt)

## Current Cluster Defaults

The helper scripts now default to:

- partition: `ai`
- GPU: `a40:1`
- CPUs per task: `8`
- memory: `32G`
- time limit: `1-00:00:00`

This matches the current ROI-based redesign sweeps better than the older `v100` defaults.

## Helper Submit Scripts

You usually do not need to call `sbatch` manually. The helper scripts construct the array submission command for you.

### Submit training scenarios

```bash
PARTITION=ai GPU_GRES=gpu:a40:1 TIME_LIMIT=1-00:00:00 bash scripts/slurm/submit_train_scenarios.sh
```

### Submit tuning scenarios

```bash
PARTITION=ai GPU_GRES=gpu:a40:1 TIME_LIMIT=1-00:00:00 bash scripts/slurm/submit_tune_scenarios.sh
```

### Submit both sweeps

```bash
PARTITION=ai GPU_GRES=gpu:a40:1 TRAIN_TIME_LIMIT=1-00:00:00 TUNE_TIME_LIMIT=1-00:00:00 bash scripts/slurm/submit_all_scenarios.sh
```

The helpers print the exact `sbatch` command before submitting it.

## What Each Script Does

### `train_array.sh`

For each array index:

1. loads modules,
2. activates the Conda environment,
3. reads the config path from `train_scenarios.txt`,
4. launches:

```bash
python -m age_decoupled_surrealgan.cli --config <config> train
```

### `tune_array.sh`

Same idea, but launches:

```bash
python -m age_decoupled_surrealgan.cli --config <config> tune
```

## Environment Variables You Can Override

The submit helpers accept:

- `PROJECT_ROOT`
- `CONDA_ENV_NAME`
- `CONFIG_LIST_FILE`
- `PARTITION`
- `GPU_GRES`
- `CPUS_PER_TASK`
- `MEMORY`
- `TIME_LIMIT`

For `submit_all_scenarios.sh`, use:

- `TRAIN_TIME_LIMIT`
- `TUNE_TIME_LIMIT`

Example:

```bash
PROJECT_ROOT=/gpfs/fs001/cbica/home/aidinisg/Projects/Age-Decoupled-SurrealGAN \
CONDA_ENV_NAME=age-decoupled-surrealgan \
PARTITION=ai \
GPU_GRES=gpu:a100:1 \
TIME_LIMIT=1-00:00:00 \
bash scripts/slurm/submit_tune_scenarios.sh
```

## Resume Behavior

### Training

Training can resume from an interrupted run directory:

```bash
PYTHONPATH=src python -m age_decoupled_surrealgan.cli \
  --config src/age_decoupled_surrealgan/configs/redesign_train.toml \
  train --resume-run-dir runs/<run_dir>
```

If you want to route that through SLURM, export `RESUME_RUN_DIR` to the train array script.

### Tuning

Optuna tuning resumes automatically as long as:

- the same `study_name` is used,
- the same SQLite `storage` path is used,
- `resume_if_exists = true`.

## Interpreting Pending Jobs

Typical reasons:

- `(Resources)`: no matching node/GPU is currently free
- `(PartitionTimeLimit)`: requested wall time exceeds what that partition allows

If you request `ai` and `1-00:00:00`, that is usually acceptable for the redesigned ROI sweeps.

## Practical Recommendation

For the redesigned model:

1. run `prepare-data` once after any normalization/data change,
2. submit the redesign train scenario array,
3. inspect the best families,
4. submit the redesign tuning array around those families.

That is the current intended workflow.
