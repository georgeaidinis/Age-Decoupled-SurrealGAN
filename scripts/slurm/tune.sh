#!/bin/bash
#SBATCH --job-name=age_surrealgan_tune
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --mail-type=ALL
#SBATCH mail-user=aidinisg@pennmedicine.upenn.edu

set -euo pipefail

nvidia-smi

module load slurm/current
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/home/aidinisg/Projects/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_PATH="${CONFIG_PATH:-src/age_decoupled_surrealgan/configs/extended_tune.toml}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
export PYTHONPATH=src

srun python -m age_decoupled_surrealgan.cli --config "${CONFIG_PATH}" tune
