#!/bin/bash
#SBATCH --job-name=age_k_repeat
#SBATCH --array=0-0
#SBATCH --partition=ai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=48G
#SBATCH --time=1-12:00:00
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aidinis@seas.upenn.edu
#SBATCH --propagate=NONE

set -euo pipefail

nvidia-smi

module load slurm/current
module load cuda/12.2

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/home/aidinisg/Projects/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-scripts/slurm/model_selection_configs.txt}"
REPETITIONS="${REPETITIONS:-5}"
EPOCHS="${EPOCHS:-180}"
RECORD_DIR="${RECORD_DIR:-runs/model_selection/records}"

source activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs "${RECORD_DIR}"
export PYTHONPATH=src

CONFIG_PATH="$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${CONFIG_LIST_FILE}")"
if [[ -z "${CONFIG_PATH}" ]]; then
  echo "No config found for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
  exit 1
fi

CONFIG_BASENAME="$(basename "${CONFIG_PATH}" .toml)"
EXPERIMENT_NAME="${CONFIG_BASENAME}_repeat"
RECORD_PATH="${RECORD_DIR}/${EXPERIMENT_NAME}.json"

echo "Training best study trial for ${CONFIG_PATH}"
echo "Experiment: ${EXPERIMENT_NAME}"
echo "Record path: ${RECORD_PATH}"
srun python -m age_decoupled_surrealgan.cli \
  --config "${CONFIG_PATH}" \
  train-best-from-study \
  --repetitions "${REPETITIONS}" \
  --epochs "${EPOCHS}" \
  --record-path "${RECORD_PATH}" \
  --experiment-name "${EXPERIMENT_NAME}"
