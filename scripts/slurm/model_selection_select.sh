#!/bin/bash
#SBATCH --job-name=age_k_select
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm_logs/%x-%A.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=aidinis@seas.upenn.edu
#SBATCH --propagate=NONE

set -euo pipefail

module load slurm/current

PROJECT_ROOT="${PROJECT_ROOT:-/cbica/home/aidinisg/Projects/Age-Decoupled-SurrealGAN}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-age-decoupled-surrealgan}"
SELECTION_CONFIG="${SELECTION_CONFIG:-src/age_decoupled_surrealgan/configs/model_selection/k3.toml}"
RECORD_DIR="${RECORD_DIR:-runs/model_selection/records}"
OUTPUT_PATH="${OUTPUT_PATH:-runs/model_selection/best_model_summary.json}"

source activate "${CONDA_ENV_NAME}"

cd "${PROJECT_ROOT}"
mkdir -p slurm_logs
export PYTHONPATH=src

python -m age_decoupled_surrealgan.cli \
  --config "${SELECTION_CONFIG}" \
  select-best-model \
  --record-dir "${RECORD_DIR}" \
  --output-path "${OUTPUT_PATH}"
