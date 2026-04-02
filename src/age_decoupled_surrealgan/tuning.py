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
        width = trial.suggest_categorical("model_width", config.tuning.width_options)
        trial_config.model.n_processes = trial.suggest_categorical("n_processes", config.tuning.n_processes_options)
        trial_config.model.encoder_hidden_dims = [width, width // 2]
        trial_config.model.generator_hidden_dims = [width, width, width // 2]
        trial_config.model.discriminator_hidden_dims = [width, width // 2]
        trial_config.model.decomposer_hidden_dims = [width, width // 2]
        trial_config.model.dropout = trial.suggest_float("dropout", config.tuning.dropout_min, config.tuning.dropout_max)
        trial_config.training.learning_rate = trial.suggest_float(
            "learning_rate",
            config.tuning.learning_rate_min,
            config.tuning.learning_rate_max,
            log=True,
        )
        trial_config.training.discriminator_learning_rate = trial.suggest_float(
            "discriminator_learning_rate",
            config.tuning.discriminator_learning_rate_min,
            config.tuning.discriminator_learning_rate_max,
            log=True,
        )
        trial_config.training.batch_size = trial.suggest_categorical("batch_size", config.tuning.batch_size_options)
        trial_config.losses.adversarial = trial.suggest_float("adversarial", config.tuning.adversarial_min, config.tuning.adversarial_max)
        trial_config.losses.age_supervision = trial.suggest_float(
            "age_supervision", config.tuning.age_supervision_min, config.tuning.age_supervision_max
        )
        trial_config.losses.reference_age_supervision = trial.suggest_float(
            "reference_age_supervision",
            config.tuning.reference_age_supervision_min,
            config.tuning.reference_age_supervision_max,
        )
        trial_config.losses.age_adversary = trial.suggest_float(
            "age_adversary", config.tuning.age_adversary_min, config.tuning.age_adversary_max
        )
        trial_config.losses.latent_reconstruction = trial.suggest_float(
            "latent_reconstruction", config.tuning.latent_reconstruction_min, config.tuning.latent_reconstruction_max
        )
        trial_config.losses.decomposition = trial.suggest_float(
            "decomposition", config.tuning.decomposition_min, config.tuning.decomposition_max
        )
        trial_config.losses.identity = trial.suggest_float("identity", config.tuning.identity_min, config.tuning.identity_max)
        trial_config.losses.monotonicity = trial.suggest_float(
            "monotonicity", config.tuning.monotonicity_min, config.tuning.monotonicity_max
        )
        trial_config.losses.process_orthogonality = trial.suggest_float(
            "process_orthogonality", config.tuning.process_orthogonality_min, config.tuning.process_orthogonality_max
        )
        trial_config.losses.age_process_covariance = trial.suggest_float(
            "age_process_covariance",
            config.tuning.age_process_covariance_min,
            config.tuning.age_process_covariance_max,
        )
        trial_config.losses.reference_process_sparsity = trial.suggest_float(
            "reference_process_sparsity",
            config.tuning.reference_process_sparsity_min,
            config.tuning.reference_process_sparsity_max,
        )
        trial_config.losses.change_magnitude = trial.suggest_float(
            "change_magnitude", config.tuning.change_magnitude_min, config.tuning.change_magnitude_max
        )
        trial_config.losses.low_activation_identity = trial.suggest_float(
            "low_activation_identity",
            config.tuning.low_activation_identity_min,
            config.tuning.low_activation_identity_max,
        )
        trial_config.losses.process_age_correlation = trial.suggest_float(
            "process_age_correlation",
            config.tuning.process_age_correlation_min,
            config.tuning.process_age_correlation_max,
        )
        trial_config.losses.process_latent_sparsity = trial.suggest_float(
            "process_latent_sparsity",
            config.tuning.process_latent_sparsity_min,
            config.tuning.process_latent_sparsity_max,
        )
        trial_config.losses.age_sensitivity = trial.suggest_float(
            "age_sensitivity", config.tuning.age_sensitivity_min, config.tuning.age_sensitivity_max
        )
        trial_config.losses.process_sensitivity = trial.suggest_float(
            "process_sensitivity", config.tuning.process_sensitivity_min, config.tuning.process_sensitivity_max
        )
        trial_config.losses.age_sensitivity_target_pct = trial.suggest_float(
            "age_sensitivity_target_pct",
            config.tuning.age_sensitivity_target_pct_min,
            config.tuning.age_sensitivity_target_pct_max,
        )
        trial_config.losses.process_sensitivity_target_pct = trial.suggest_float(
            "process_sensitivity_target_pct",
            config.tuning.process_sensitivity_target_pct_min,
            config.tuning.process_sensitivity_target_pct_max,
        )
        trial_config.losses.age_shrinkage = trial.suggest_float(
            "age_shrinkage", config.tuning.age_shrinkage_min, config.tuning.age_shrinkage_max
        )
        trial_config.losses.process_shrinkage = trial.suggest_float(
            "process_shrinkage", config.tuning.process_shrinkage_min, config.tuning.process_shrinkage_max
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
