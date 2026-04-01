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
sbatch scripts/slurm/prepare_data.slurm
```

Train:

```bash
sbatch scripts/slurm/train.slurm
```

Tune:

```bash
sbatch scripts/slurm/tune.slurm
```

Array training over multiple config files:

```bash
sbatch --array=0-2 scripts/slurm/train_array.slurm
```

The default config list for array jobs is:

- [`scripts/slurm/config_list.txt`](/Users/georgeaidinis/Desktop/PhD/Experiments/Age-Decoupled-SurrealGAN/scripts/slurm/config_list.txt)

## Resource Defaults

The scripts now default to a lighter request:

- partition: `all`
- GPU: `v100:1`
- memory: `32G`

This is intentionally conservative for ROI-based training. If a particular run needs more memory or a newer GPU, edit the `#SBATCH` lines or override the script you submit.

## Notes

- Edit `PROJECT_ROOT`, `CONDA_ENV_NAME`, `CONFIG_PATH`, and GPU request lines before first use.
- For longer jobs, consider `long`.
- For shorter smoke runs, consider `short`.
- If your environment lives outside the default Conda path, update the activation lines accordingly.
