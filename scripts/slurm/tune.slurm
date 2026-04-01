#!/bin/bash
#SBATCH --job-name=age_surrealgan_tune
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out

set -euo pipefail

module load slurm/current

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/YOUR_PROJECT/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_PATH="${CONFIG_PATH:-src/age_decoupled_surrealgan/configs/default.toml}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
export PYTHONPATH=src

srun python -m age_decoupled_surrealgan.cli --config "${CONFIG_PATH}" tune
