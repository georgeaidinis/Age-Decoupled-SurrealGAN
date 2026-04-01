#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src

TRAIN_CONFIG="src/age_decoupled_surrealgan/configs/extended_train.toml"
TUNE_CONFIG="src/age_decoupled_surrealgan/configs/extended_tune.toml"

python -m age_decoupled_surrealgan.cli --config "${TRAIN_CONFIG}" train
python -m age_decoupled_surrealgan.cli --config "${TUNE_CONFIG}" tune
