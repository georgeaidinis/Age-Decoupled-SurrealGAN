#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
CONFIG_LIST_FILE="${CONFIG_LIST_FILE:-scripts/slurm/model_selection_configs.txt}"
TUNE_PARTITION="${TUNE_PARTITION:-ai}"
TRAIN_PARTITION="${TRAIN_PARTITION:-ai}"
SELECT_PARTITION="${SELECT_PARTITION:-all}"
GPU_GRES="${GPU_GRES:-gpu:a40:1}"
TUNE_CPUS_PER_TASK="${TUNE_CPUS_PER_TASK:-8}"
TRAIN_CPUS_PER_TASK="${TRAIN_CPUS_PER_TASK:-8}"
SELECT_CPUS_PER_TASK="${SELECT_CPUS_PER_TASK:-4}"
TUNE_MEMORY="${TUNE_MEMORY:-48G}"
TRAIN_MEMORY="${TRAIN_MEMORY:-48G}"
SELECT_MEMORY="${SELECT_MEMORY:-8G}"
TUNE_TIME_LIMIT="${TUNE_TIME_LIMIT:-2-00:00:00}"
TRAIN_TIME_LIMIT="${TRAIN_TIME_LIMIT:-1-12:00:00}"
SELECT_TIME_LIMIT="${SELECT_TIME_LIMIT:-02:00:00}"
REPETITIONS="${REPETITIONS:-5}"
EPOCHS="${EPOCHS:-180}"
RECORD_DIR="${RECORD_DIR:-runs/model_selection/records}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/model_selection/best_model_summary.json}"
SELECTION_CONFIG="${SELECTION_CONFIG:-src/age_decoupled_surrealgan/configs/model_selection/k3.toml}"

count=$(grep -cv '^\s*$' "${CONFIG_LIST_FILE}")
if [[ "${count}" -le 0 ]]; then
  echo "No configs found in ${CONFIG_LIST_FILE}" >&2
  exit 1
fi

tune_output=$(
  sbatch \
    --array="0-$((count - 1))" \
    --partition="${TUNE_PARTITION}" \
    --gres="${GPU_GRES}" \
    --cpus-per-task="${TUNE_CPUS_PER_TASK}" \
    --mem="${TUNE_MEMORY}" \
    --time="${TUNE_TIME_LIMIT}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_LIST_FILE=${CONFIG_LIST_FILE}" \
    --propagate=NONE \
    scripts/slurm/model_selection_tune_array.sh
)
tune_job_id=$(echo "${tune_output}" | awk '{print $4}')
echo "Submitted tuning array: ${tune_output}"

train_output=$(
  sbatch \
    --dependency="afterok:${tune_job_id}" \
    --array="0-$((count - 1))" \
    --partition="${TRAIN_PARTITION}" \
    --gres="${GPU_GRES}" \
    --cpus-per-task="${TRAIN_CPUS_PER_TASK}" \
    --mem="${TRAIN_MEMORY}" \
    --time="${TRAIN_TIME_LIMIT}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},CONFIG_LIST_FILE=${CONFIG_LIST_FILE},REPETITIONS=${REPETITIONS},EPOCHS=${EPOCHS},RECORD_DIR=${RECORD_DIR}" \
    --propagate=NONE \
    scripts/slurm/model_selection_train_best_array.sh
)
train_job_id=$(echo "${train_output}" | awk '{print $4}')
echo "Submitted repeated-train array: ${train_output}"

select_output=$(
  sbatch \
    --dependency="afterok:${train_job_id}" \
    --partition="${SELECT_PARTITION}" \
    --cpus-per-task="${SELECT_CPUS_PER_TASK}" \
    --mem="${SELECT_MEMORY}" \
    --time="${SELECT_TIME_LIMIT}" \
    --export="ALL,PROJECT_ROOT=${PROJECT_ROOT},CONDA_ENV_NAME=${CONDA_ENV_NAME},SELECTION_CONFIG=${SELECTION_CONFIG},RECORD_DIR=${RECORD_DIR},OUTPUT_PATH=${OUTPUT_PATH}" \
    --propagate=NONE \
    scripts/slurm/model_selection_select.sh
)
select_job_id=$(echo "${select_output}" | awk '{print $4}')
echo "Submitted selection job: ${select_output}"

printf 'Pipeline queued.\nTune job: %s\nRepeated-train job: %s\nSelection job: %s\n' "${tune_job_id}" "${train_job_id}" "${select_job_id}"
