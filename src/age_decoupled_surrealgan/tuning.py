from __future__ import annotations

import copy
from typing import Any

from .config import ProjectConfig
from .trainer import AgeDecoupledTrainer


def run_optuna_search(config: ProjectConfig) -> dict[str, Any]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Optuna is not installed in the active environment.") from exc

    def objective(trial: optuna.Trial) -> float:
        trial_config = copy.deepcopy(config)
        trial_config.model.n_processes = trial.suggest_int("n_processes", 2, 6)
        trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1e-4, 5e-4, log=True)
        trial_config.training.discriminator_learning_rate = trial.suggest_float(
            "discriminator_learning_rate", 5e-5, 2e-4, log=True
        )
        trial_config.training.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        trial_config.losses.age_supervision = trial.suggest_float("age_supervision", 2.0, 8.0)
        trial_config.losses.process_orthogonality = trial.suggest_float("process_orthogonality", 0.05, 0.4)
        trial_config.losses.age_process_covariance = trial.suggest_float("age_process_covariance", 0.1, 1.0)
        trial_config.losses.change_magnitude = trial.suggest_categorical("change_magnitude", [0.0, 0.02, 0.05])
        trial_config.losses.process_age_correlation = trial.suggest_categorical(
            "process_age_correlation", [0.0, 0.1, 0.25]
        )
        trial_config.experiment_name = f"{config.experiment_name}-trial-{trial.number:03d}"

        trainer = AgeDecoupledTrainer(trial_config)
        summary = trainer.train(trial_name=f"trial_{trial.number:03d}")
        return float(summary["split_metrics"]["val"]["composite_score"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.tuning.trials, timeout=config.tuning.timeout_seconds)
    return {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }
