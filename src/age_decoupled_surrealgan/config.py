from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11 fallback
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:  # pragma: no cover
        tomllib = None  # type: ignore[assignment]

from .utils import to_serializable


@dataclass
class PathConfig:
    raw_dataset: str = "datasets/cleaned_istaging.csv"
    muse_roi_json: str = "datasets/MUSE_ROI_complete_list.json"
    atlas_image: str = "datasets/atlas.nii.gz"
    atlas_segmentation: str = "datasets/atlas_segmentation.nii.gz"
    processed_dir: str = "artifacts/data/processed"
    runs_dir: str = "runs"
    notebook_dir: str = "output/jupyter-notebook"


@dataclass
class DataConfig:
    subject_id_column: str = "PTID"
    study_column: str = "Study"
    age_column: str = "Age"
    diagnosis_column: str = "Diagnosis"
    sex_column: str = "Sex"
    roi_prefix: str = "MUSE_Volume_"
    max_atomic_roi_id: int = 299
    ref_min_age: int = 20
    ref_max_age: int = 49
    tar_min_age: int = 50
    tar_max_age: int = 97
    holdout_study: str = "HANDLS"
    val_fraction: float = 0.1
    id_test_fraction: float = 0.1
    split_seed: int = 17
    age_latent_normalization_min: int = 20
    age_latent_normalization_max: int = 97


@dataclass
class ModelConfig:
    n_processes: int = 4
    age_latent_dim: int = 1
    encoder_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    generator_hidden_dims: list[int] = field(default_factory=lambda: [512, 512, 256])
    discriminator_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    decomposer_hidden_dims: list[int] = field(default_factory=lambda: [512, 256])
    dropout: float = 0.1


@dataclass
class TrainingConfig:
    repetitions: int = 3
    epochs: int = 50
    batch_size: int = 128
    learning_rate: float = 2.0e-4
    discriminator_learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-4
    gradient_clip_norm: float = 5.0
    num_workers: int = 0
    persistent_workers: bool = False
    use_amp: bool = False
    compile_model: bool = False
    device: str = "auto"
    save_every: int = 0
    target_regular_checkpoints: int = 3
    log_every: int = 10
    console_log_every: int = 1
    monitor_split: str = "val"
    monitor_metric: str = "composite_score"
    sensitivity_eval_subjects: int = 64
    sensitivity_process_anchor_age: float = 20.0
    resume_run_dir: str | None = None


@dataclass
class LossConfig:
    adversarial: float = 1.0
    age_supervision: float = 4.0
    reference_age_supervision: float = 2.0
    age_adversary: float = 1.0
    latent_reconstruction: float = 1.0
    decomposition: float = 1.0
    identity: float = 4.0
    monotonicity: float = 0.5
    process_orthogonality: float = 0.2
    age_process_covariance: float = 0.4
    reference_process_sparsity: float = 0.2
    change_magnitude: float = 0.0
    low_activation_identity: float = 0.0
    process_age_correlation: float = 0.0
    process_latent_sparsity: float = 0.0
    low_activation_max: float = 0.05
    age_sensitivity: float = 0.0
    process_sensitivity: float = 0.0
    age_sensitivity_target_pct: float = 0.25
    process_sensitivity_target_pct: float = 0.10
    age_shrinkage: float = 0.0
    process_shrinkage: float = 0.0


@dataclass
class TuningConfig:
    enabled: bool = True
    trials: int = 20
    timeout_seconds: int | None = None
    study_name: str = "age_decoupled_surrealgan"
    storage: str = "sqlite:///runs/optuna_age_decoupled_surrealgan.db"
    resume_if_exists: bool = True
    objective_metric: str = "quality_score"
    width_options: list[int] = field(default_factory=lambda: [384, 512, 640])
    n_processes_options: list[int] = field(default_factory=lambda: [4, 5])
    batch_size_options: list[int] = field(default_factory=lambda: [96, 128])
    dropout_min: float = 0.02
    dropout_max: float = 0.12
    learning_rate_min: float = 1.5e-4
    learning_rate_max: float = 3.5e-4
    discriminator_learning_rate_min: float = 2.5e-5
    discriminator_learning_rate_max: float = 8e-5
    adversarial_min: float = 1.0
    adversarial_max: float = 2.0
    age_supervision_min: float = 4.0
    age_supervision_max: float = 6.5
    reference_age_supervision_min: float = 1.2
    reference_age_supervision_max: float = 2.5
    age_adversary_min: float = 0.75
    age_adversary_max: float = 1.5
    latent_reconstruction_min: float = 1.0
    latent_reconstruction_max: float = 2.0
    decomposition_min: float = 1.0
    decomposition_max: float = 2.0
    identity_min: float = 1.5
    identity_max: float = 4.0
    monotonicity_min: float = 0.1
    monotonicity_max: float = 0.6
    process_orthogonality_min: float = 0.1
    process_orthogonality_max: float = 0.45
    age_process_covariance_min: float = 0.15
    age_process_covariance_max: float = 0.6
    reference_process_sparsity_min: float = 0.02
    reference_process_sparsity_max: float = 0.2
    change_magnitude_min: float = 0.0
    change_magnitude_max: float = 0.4
    low_activation_identity_min: float = 0.0
    low_activation_identity_max: float = 0.2
    process_age_correlation_min: float = 0.0
    process_age_correlation_max: float = 0.3
    process_latent_sparsity_min: float = 0.0
    process_latent_sparsity_max: float = 0.2
    age_sensitivity_min: float = 0.5
    age_sensitivity_max: float = 3.0
    process_sensitivity_min: float = 0.3
    process_sensitivity_max: float = 1.8
    age_sensitivity_target_pct_min: float = 0.1
    age_sensitivity_target_pct_max: float = 0.5
    process_sensitivity_target_pct_min: float = 0.05
    process_sensitivity_target_pct_max: float = 0.35
    age_shrinkage_min: float = 0.0
    age_shrinkage_max: float = 0.8
    process_shrinkage_min: float = 0.0
    process_shrinkage_max: float = 0.6


@dataclass
class AppConfig:
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    allow_origin: str = "*"


@dataclass
class ProjectConfig:
    experiment_name: str = "age-decoupled-surrealgan"
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    losses: LossConfig = field(default_factory=LossConfig)
    tuning: TuningConfig = field(default_factory=TuningConfig)
    app: AppConfig = field(default_factory=AppConfig)

    def as_dict(self) -> dict[str, Any]:
        return to_serializable(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_config(payload: dict[str, Any]) -> ProjectConfig:
    return ProjectConfig(
        experiment_name=payload.get("experiment_name", ProjectConfig().experiment_name),
        paths=PathConfig(**payload.get("paths", {})),
        data=DataConfig(**payload.get("data", {})),
        model=ModelConfig(**payload.get("model", {})),
        training=TrainingConfig(**payload.get("training", {})),
        losses=LossConfig(**payload.get("losses", {})),
        tuning=TuningConfig(**payload.get("tuning", {})),
        app=AppConfig(**payload.get("app", {})),
    )


def default_config_path() -> Path:
    return Path(__file__).resolve().parent / "configs" / "default.toml"


def load_project_config(config_path: str | Path | None = None) -> ProjectConfig:
    base_path = default_config_path()
    if tomllib is None:
        payload = ProjectConfig().as_dict()
    else:
        with base_path.open("rb") as handle:
            payload = tomllib.load(handle)
    if config_path is not None:
        if tomllib is None:
            raise RuntimeError("Loading TOML overrides requires tomli or Python 3.11+.")
        with Path(config_path).expanduser().resolve().open("rb") as handle:
            override = tomllib.load(handle)
        payload = _merge_dict(payload, override)
    return _build_config(payload)
