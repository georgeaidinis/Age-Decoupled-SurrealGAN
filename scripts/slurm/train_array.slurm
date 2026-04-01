#!/bin/bash
#SBATCH --job-name=age_surrealgan_array
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=160G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out

set -euo pipefail

module load slurm/current

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/projects/YOUR_PROJECT/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-scripts/slurm/config_list.txt}"

source "${HOME}/miniconda3/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
export PYTHONPATH=src

CONFIG_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "No config found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Using config: ${CONFIG_PATH}"
srun python -m age_decoupled_surrealgan.cli --config "${CONFIG_PATH}" train
