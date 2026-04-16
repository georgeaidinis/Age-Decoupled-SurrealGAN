from __future__ import annotations

import copy
import time
from datetime import datetime
from typing import Any

from .config import ProjectConfig
from .reporting import format_duration
from .trainer import AgeDecoupledTrainer


def apply_trial_params(trial_config: ProjectConfig, params: dict[str, Any]) -> ProjectConfig:
    width = int(params.get("model_width", trial_config.model.generator_hidden_dims[0]))
    trial_config.model.n_processes = int(params.get("n_processes", trial_config.model.n_processes))
    trial_config.model.encoder_hidden_dims = [width, width // 2]
    trial_config.model.generator_hidden_dims = [width, width, width // 2]
    trial_config.model.discriminator_hidden_dims = [width, width // 2]
    trial_config.model.decomposer_hidden_dims = [width, width // 2]
    for section_name, key in [
        ("model", "dropout"),
        ("training", "learning_rate"),
        ("training", "discriminator_learning_rate"),
        ("training", "batch_size"),
        ("losses", "adversarial"),
        ("losses", "age_supervision"),
        ("losses", "reference_age_supervision"),
        ("losses", "age_adversary"),
        ("losses", "latent_reconstruction"),
        ("losses", "decomposition"),
        ("losses", "identity"),
        ("losses", "monotonicity"),
        ("losses", "process_orthogonality"),
        ("losses", "age_process_covariance"),
        ("losses", "reference_process_sparsity"),
        ("losses", "change_magnitude"),
        ("losses", "low_activation_identity"),
        ("losses", "process_age_correlation"),
        ("losses", "process_latent_sparsity"),
        ("losses", "age_sensitivity"),
        ("losses", "process_sensitivity"),
        ("losses", "age_sensitivity_target_pct"),
        ("losses", "process_sensitivity_target_pct"),
        ("losses", "age_shrinkage"),
        ("losses", "process_shrinkage"),
        ("losses", "generator_process_separation"),
        ("losses", "generator_process_redundancy"),
        ("losses", "process_latent_pairwise_correlation"),
    ]:
        if key in params:
            setattr(getattr(trial_config, section_name), key, params[key])
    return trial_config


def best_trial_config(config: ProjectConfig) -> tuple[ProjectConfig, dict[str, Any]]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Optuna is not installed in the active environment.") from exc

    study = optuna.load_study(study_name=config.tuning.study_name, storage=config.tuning.storage)
    trial_config = copy.deepcopy(config)
    apply_trial_params(trial_config, study.best_trial.params)
    trial_config.training.monitor_metric = config.tuning.objective_metric
    return trial_config, {
        "number": study.best_trial.number,
        "value": study.best_value,
        "params": dict(study.best_trial.params),
        "study_name": config.tuning.study_name,
        "storage": config.tuning.storage,
    }


def run_optuna_search(config: ProjectConfig) -> dict[str, Any]:
    try:
        import optuna
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("Optuna is not installed in the active environment.") from exc

    tuning_started_at = datetime.now()
    tuning_timer_start = time.perf_counter()
    print(
        f"Starting Optuna study '{config.tuning.study_name}' at "
        f"{tuning_started_at.strftime('%Y-%m-%d %H:%M:%S')} "
        f"(trials={config.tuning.trials}, timeout_seconds={config.tuning.timeout_seconds})"
    )

    def objective(trial: optuna.Trial) -> float:
        trial_started_at = datetime.now()
        trial_timer_start = time.perf_counter()
        print(f"[optuna] starting trial {trial.number:03d} at {trial_started_at.strftime('%Y-%m-%d %H:%M:%S')}")
        trial_config = copy.deepcopy(config)
        params = {
            "model_width": trial.suggest_categorical("model_width", config.tuning.width_options),
            "n_processes": trial.suggest_categorical("n_processes", config.tuning.n_processes_options),
            "dropout": trial.suggest_float("dropout", config.tuning.dropout_min, config.tuning.dropout_max),
            "learning_rate": trial.suggest_float(
                "learning_rate",
                config.tuning.learning_rate_min,
                config.tuning.learning_rate_max,
                log=True,
            ),
            "discriminator_learning_rate": trial.suggest_float(
                "discriminator_learning_rate",
                config.tuning.discriminator_learning_rate_min,
                config.tuning.discriminator_learning_rate_max,
                log=True,
            ),
            "batch_size": trial.suggest_categorical("batch_size", config.tuning.batch_size_options),
            "adversarial": trial.suggest_float("adversarial", config.tuning.adversarial_min, config.tuning.adversarial_max),
            "age_supervision": trial.suggest_float(
                "age_supervision", config.tuning.age_supervision_min, config.tuning.age_supervision_max
            ),
            "reference_age_supervision": trial.suggest_float(
                "reference_age_supervision",
                config.tuning.reference_age_supervision_min,
                config.tuning.reference_age_supervision_max,
            ),
            "age_adversary": trial.suggest_float(
                "age_adversary", config.tuning.age_adversary_min, config.tuning.age_adversary_max
            ),
            "latent_reconstruction": trial.suggest_float(
                "latent_reconstruction", config.tuning.latent_reconstruction_min, config.tuning.latent_reconstruction_max
            ),
            "decomposition": trial.suggest_float(
                "decomposition", config.tuning.decomposition_min, config.tuning.decomposition_max
            ),
            "identity": trial.suggest_float("identity", config.tuning.identity_min, config.tuning.identity_max),
            "monotonicity": trial.suggest_float(
                "monotonicity", config.tuning.monotonicity_min, config.tuning.monotonicity_max
            ),
            "process_orthogonality": trial.suggest_float(
                "process_orthogonality", config.tuning.process_orthogonality_min, config.tuning.process_orthogonality_max
            ),
            "age_process_covariance": trial.suggest_float(
                "age_process_covariance",
                config.tuning.age_process_covariance_min,
                config.tuning.age_process_covariance_max,
            ),
            "reference_process_sparsity": trial.suggest_float(
                "reference_process_sparsity",
                config.tuning.reference_process_sparsity_min,
                config.tuning.reference_process_sparsity_max,
            ),
            "change_magnitude": trial.suggest_float(
                "change_magnitude", config.tuning.change_magnitude_min, config.tuning.change_magnitude_max
            ),
            "low_activation_identity": trial.suggest_float(
                "low_activation_identity",
                config.tuning.low_activation_identity_min,
                config.tuning.low_activation_identity_max,
            ),
            "process_age_correlation": trial.suggest_float(
                "process_age_correlation",
                config.tuning.process_age_correlation_min,
                config.tuning.process_age_correlation_max,
            ),
            "process_latent_sparsity": trial.suggest_float(
                "process_latent_sparsity",
                config.tuning.process_latent_sparsity_min,
                config.tuning.process_latent_sparsity_max,
            ),
            "age_sensitivity": trial.suggest_float(
                "age_sensitivity", config.tuning.age_sensitivity_min, config.tuning.age_sensitivity_max
            ),
            "process_sensitivity": trial.suggest_float(
                "process_sensitivity", config.tuning.process_sensitivity_min, config.tuning.process_sensitivity_max
            ),
            "age_sensitivity_target_pct": trial.suggest_float(
                "age_sensitivity_target_pct",
                config.tuning.age_sensitivity_target_pct_min,
                config.tuning.age_sensitivity_target_pct_max,
            ),
            "process_sensitivity_target_pct": trial.suggest_float(
                "process_sensitivity_target_pct",
                config.tuning.process_sensitivity_target_pct_min,
                config.tuning.process_sensitivity_target_pct_max,
            ),
            "age_shrinkage": trial.suggest_float(
                "age_shrinkage", config.tuning.age_shrinkage_min, config.tuning.age_shrinkage_max
            ),
            "process_shrinkage": trial.suggest_float(
                "process_shrinkage", config.tuning.process_shrinkage_min, config.tuning.process_shrinkage_max
            ),
            "generator_process_separation": trial.suggest_float(
                "generator_process_separation",
                config.tuning.generator_process_separation_min,
                config.tuning.generator_process_separation_max,
            ),
            "generator_process_redundancy": trial.suggest_float(
                "generator_process_redundancy",
                config.tuning.generator_process_redundancy_min,
                config.tuning.generator_process_redundancy_max,
            ),
            "process_latent_pairwise_correlation": trial.suggest_float(
                "process_latent_pairwise_correlation",
                config.tuning.process_latent_pairwise_correlation_min,
                config.tuning.process_latent_pairwise_correlation_max,
            ),
        }
        apply_trial_params(trial_config, params)
        trial_config.training.monitor_metric = trial_config.tuning.objective_metric
        trial_config.experiment_name = config.experiment_name

        trainer = AgeDecoupledTrainer(trial_config)
        summary = trainer.train(trial_name=f"trial-{trial.number:03d}")
        value = float(summary["split_metrics"]["val"][trial_config.tuning.objective_metric])
        print(
            f"[optuna] completed trial {trial.number:03d} "
            f"in {format_duration(time.perf_counter() - trial_timer_start)} "
            f"with {trial_config.tuning.objective_metric}={value:.4f}"
        )
        return value

    study = optuna.create_study(
        study_name=config.tuning.study_name,
        storage=config.tuning.storage,
        load_if_exists=config.tuning.resume_if_exists,
        direction="maximize",
    )
    study.optimize(objective, n_trials=config.tuning.trials, timeout=config.tuning.timeout_seconds)
    total_seconds = time.perf_counter() - tuning_timer_start
    print(
        f"Completed Optuna study '{config.tuning.study_name}' "
        f"in {format_duration(total_seconds)} "
        f"(best_value={study.best_value:.4f})"
    )
    return {
        "started_at": tuning_started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "total_seconds": total_seconds,
        "best_value": study.best_value,
        "best_params": study.best_params,
        "trials": [
            {
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": str(trial.state),
                "duration_seconds": trial.duration.total_seconds() if trial.duration is not None else None,
            }
            for trial in study.trials
        ],
    }
