#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-scripts/slurm/train_scenarios.txt}"
PARTITION="${PARTITION:-ai}"
GPU_GRES="${GPU_GRES:-gpu:v100:1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEMORY="${MEMORY:-32G}"
TIME_LIMIT="${TIME_LIMIT:-1-00:00:00}"

count=$(grep -cv '^\s*$' "${CONFIG_LIST_FILE}")
if [[ "${count}" -le 0 ]]; then
  echo "No configs found in ${CONFIG_LIST_FILE}" >&2
  exit 1
fi

sbatch \
  --array="0-$((count - 1))" \
  --partition="${PARTITION}" \
  --gres="${GPU_GRES}" \
  --cpus-per-task="${CPUS_PER_TASK}" \
  --mem="${MEMORY}" \
  --time="${TIME_LIMIT}" \
  --export=ALL,PROJECT_ROOT="${PROJECT_ROOT}",CONDA_ENV_NAME="${CONDA_ENV_NAME}",CONFIG_LIST_FILE="${CONFIG_LIST_FILE}" \
  scripts/slurm/train_array.sh
