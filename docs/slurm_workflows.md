# SLURM Workflows

The cluster notes in [`scripts/cluster_notes.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/cluster_notes.txt) indicate:

- ML jobs should target the `ai` family of partitions
- GPU jobs must request a specific GPU type
- jobs should usually request at least one node and one GPU
- a rule of thumb is to request system memory around `2x` GPU memory

The provided scripts under [`scripts/slurm/`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm) assume:

- `module load slurm/current`
- a Conda environment named `age-decoupled-surrealgan`
- the repo is already cloned on the cluster filesystem

## Typical Usage

Prepare processed data:

```bash
sbatch scripts/slurm/prepare_data.sh
```

Train:

```bash
sbatch scripts/slurm/train.sh
```

Tune:

```bash
sbatch scripts/slurm/tune.sh
```

Array training over multiple config files:

```bash
sbatch --array=0-8 scripts/slurm/train_array.sh
```

Array tuning over multiple config files:

```bash
sbatch --array=0-5 scripts/slurm/tune_array.sh
```

Or use the helper submitters:

```bash
scripts/slurm/submit_train_scenarios.sh
scripts/slurm/submit_tune_scenarios.sh
scripts/slurm/submit_all_scenarios.sh
```

The default config lists for the scenario arrays are:

- [`scripts/slurm/train_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/train_scenarios.txt)
- [`scripts/slurm/tune_scenarios.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/tune_scenarios.txt)

## Resource Defaults

The scripts now default to a lighter request:

- partition: `all`
- GPU: `v100:1`
- memory: `32G`

This is intentionally conservative for ROI-based training. If a particular run needs more memory or a newer GPU, edit the `#SBATCH` lines or override the script you submit.

## Notes

- Edit `PROJECT_ROOT`, `CONDA_ENV_NAME`, and the mail address lines before first use.
- For longer jobs, consider `long`.
- For shorter smoke runs, consider `short`.
- If your environment lives outside the default Conda path, update the activation lines accordingly.
