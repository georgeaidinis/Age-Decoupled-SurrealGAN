#!/bin/bash
#SBATCH --job-name=age_surrealgan_tune_array
#SBATCH --array=0-0
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aidinis@seas.upenn.edu

set -euo pipefail

nvidia-smi

module load slurm/current
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/home/aidinisg/Projects/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-scripts/slurm/tune_scenarios.txt}"

source activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
export PYTHONPATH=src

CONFIG_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "No config found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

echo "Using tuning config: ${CONFIG_PATH}"
srun python -m age_decoupled_surrealgan.cli --config "${CONFIG_PATH}" tune
