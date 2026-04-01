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
        width = trial.suggest_categorical("model_width", [384, 512])
        trial_config.model.n_processes = trial.suggest_int("n_processes", 4, 5)
        trial_config.model.encoder_hidden_dims = [width, width // 2]
        trial_config.model.generator_hidden_dims = [width, width, width // 2]
        trial_config.model.discriminator_hidden_dims = [width, width // 2]
        trial_config.model.decomposer_hidden_dims = [width, width // 2]
        trial_config.model.dropout = trial.suggest_float("dropout", 0.02, 0.12)
        trial_config.training.learning_rate = trial.suggest_float("learning_rate", 1.5e-4, 3.5e-4, log=True)
        trial_config.training.discriminator_learning_rate = trial.suggest_float(
            "discriminator_learning_rate", 2.5e-5, 8e-5, log=True
        )
        trial_config.training.batch_size = trial.suggest_categorical("batch_size", [96, 128])
        trial_config.losses.adversarial = trial.suggest_float("adversarial", 1.0, 2.0)
        trial_config.losses.age_supervision = trial.suggest_float("age_supervision", 4.2, 6.3)
        trial_config.losses.reference_age_supervision = trial.suggest_float("reference_age_supervision", 1.2, 2.4)
        trial_config.losses.age_adversary = trial.suggest_float("age_adversary", 0.75, 1.5)
        trial_config.losses.latent_reconstruction = trial.suggest_float("latent_reconstruction", 1.2, 1.8)
        trial_config.losses.decomposition = trial.suggest_float("decomposition", 1.2, 2.0)
        trial_config.losses.identity = trial.suggest_float("identity", 2.2, 4.0)
        trial_config.losses.monotonicity = trial.suggest_float("monotonicity", 0.15, 0.45)
        trial_config.losses.process_orthogonality = trial.suggest_float("process_orthogonality", 0.18, 0.4)
        trial_config.losses.age_process_covariance = trial.suggest_float("age_process_covariance", 0.2, 0.5)
        trial_config.losses.reference_process_sparsity = trial.suggest_float("reference_process_sparsity", 0.03, 0.12)
        trial_config.losses.age_sensitivity = trial.suggest_float("age_sensitivity", 1.0, 3.0)
        trial_config.losses.process_sensitivity = trial.suggest_float("process_sensitivity", 0.4, 1.5)
        trial_config.losses.age_sensitivity_target_pct = trial.suggest_float("age_sensitivity_target_pct", 0.15, 0.5)
        trial_config.losses.process_sensitivity_target_pct = trial.suggest_float(
            "process_sensitivity_target_pct", 0.08, 0.35
        )
        trial_config.training.monitor_metric = trial_config.tuning.objective_metric
        trial_config.experiment_name = config.experiment_name

        trainer = AgeDecoupledTrainer(trial_config)
        summary = trainer.train(trial_name=f"trial-{trial.number:03d}")
        return float(summary["split_metrics"]["val"][trial_config.tuning.objective_metric])

    study = optuna.create_study(
        study_name=config.tuning.study_name,
        storage=config.tuning.storage,
        load_if_exists=config.tuning.resume_if_exists,
        direction="maximize",
    )
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
