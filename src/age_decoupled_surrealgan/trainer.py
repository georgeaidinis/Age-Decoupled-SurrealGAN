from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ProjectConfig, default_config_path
from .data.dataset import CohortDataset
from .evaluation import aggregate_repetition_predictions, evaluate_prediction_frame, save_metrics, save_prediction_frame
from .inference import normalize_age_years, resolve_device
from .losses import (
    change_magnitude_loss,
    correlation_penalty,
    covariance_penalty,
    discriminator_loss,
    generator_adversarial_loss,
    latent_sparsity_loss,
    monotonicity_loss,
    nonpositive_change_loss,
    orthogonality_loss,
)
from .metrics import summarize_latent_sensitivity
from .model import AgeDecoupledSurrealGAN
from .reporting import (
    TRAIN_TENSORBOARD_TAGS,
    VAL_TENSORBOARD_TAGS,
    append_log_line,
    epoch_log_line,
    flatten_metrics,
    load_doc_text,
    save_run_markdown_summary,
    save_jsonl,
    save_records_csv,
    save_split_metrics_tables,
    startup_summary_lines,
)
from .utils import cycle, ensure_dir, save_json, seed_everything

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional at runtime
    SummaryWriter = None


class AgeDecoupledTrainer:
    def __init__(self, config: ProjectConfig, config_path: str | None = None):
        self.config = config
        self.config_path = config_path
        self.processed_dir = Path(config.paths.processed_dir)
        with (self.processed_dir / "split_manifest.json").open("r", encoding="utf-8") as handle:
            self.manifest = json.load(handle)
        self.feature_columns = self.manifest["feature_columns"]
        self.n_features = len(self.feature_columns)
        self.reference_template = pd.read_csv(self.processed_dir / "reference_template.csv").set_index("feature_name")[
            "value"
        ]
        self.device = resolve_device(config.training.device)
        self.reference_template_abs_max = float(self.reference_template.abs().max())
        self.reference_template_abs_mean = float(self.reference_template.abs().mean())

    def _check_finite_tensor(self, name: str, value: torch.Tensor) -> None:
        if torch.isfinite(value).all():
            return
        finite_mask = torch.isfinite(value)
        sanitized = torch.nan_to_num(value.detach(), nan=0.0, posinf=0.0, neginf=0.0)
        payload = {
            "name": name,
            "shape": list(value.shape),
            "finite_fraction": float(finite_mask.float().mean().cpu()),
            "abs_max": float(sanitized.abs().max().cpu()),
            "device": str(value.device),
            "dtype": str(value.dtype),
            "amp_enabled": bool(self.config.training.use_amp and self.device.type == "cuda"),
            "reference_template_abs_mean": self.reference_template_abs_mean,
            "reference_template_abs_max": self.reference_template_abs_max,
        }
        raise RuntimeError(
            "Non-finite tensor encountered during training. "
            f"{json.dumps(payload, sort_keys=True)}. "
            "This is commonly caused by fp16 autocast on raw ROI-scale inputs. Disable AMP or normalize the inputs."
        )

    def _combine_weighted_losses(self, losses: dict[str, torch.Tensor], weights: dict[str, float]) -> torch.Tensor:
        device = next(iter(losses.values())).device
        total: torch.Tensor | None = None
        for key, loss in losses.items():
            self._check_finite_tensor(f"loss::{key}", loss)
            weight = float(weights.get(key, 0.0))
            if weight == 0.0:
                continue
            contribution = loss * weight
            self._check_finite_tensor(f"weighted_loss::{key}", contribution)
            total = contribution if total is None else total + contribution
        if total is None:
            total = torch.zeros((), device=device)
        self._check_finite_tensor("loss::generator_total", total)
        return total

    def _reference_frame_for_sensitivity(self, split_name: str) -> pd.DataFrame:
        frame = self._load_split(split_name)
        frame = frame.loc[frame["cohort_bucket"] == "ref"].dropna(subset=self.feature_columns)
        if frame.empty and split_name != "train":
            frame = self._load_split("train")
            frame = frame.loc[frame["cohort_bucket"] == "ref"].dropna(subset=self.feature_columns)
        return frame

    def _age_tensor(self, value_years: float, batch_size: int, device: torch.device) -> torch.Tensor:
        normalized = normalize_age_years(value_years, self.config)
        return torch.full((batch_size, self.config.model.age_latent_dim), normalized, device=device, dtype=torch.float32)

    def _latent_sensitivity_batch_metrics(
        self,
        model: AgeDecoupledSurrealGAN,
        reference_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = reference_features.shape[0]
        baseline_abs = reference_features.abs().clamp_min(1.0)
        zero_process = torch.zeros(batch, self.config.model.n_processes, device=reference_features.device)
        age_low = self._age_tensor(self.config.data.age_latent_normalization_min, batch, reference_features.device)
        age_high = self._age_tensor(self.config.data.age_latent_normalization_max, batch, reference_features.device)
        process_anchor_age = self._age_tensor(
            self.config.training.sensitivity_process_anchor_age,
            batch,
            reference_features.device,
        )

        fake_low_age, _ = model.synthesize(reference_features, age_low, zero_process)
        fake_high_age, _ = model.synthesize(reference_features, age_high, zero_process)
        age_sensitivity_pct = ((fake_high_age - fake_low_age).abs() / baseline_abs).mean() * 100.0
        age_shrinkage_penalty = nonpositive_change_loss(fake_high_age - fake_low_age)

        process_responses: list[torch.Tensor] = []
        process_targets: list[torch.Tensor] = []
        process_shrinkage_penalties: list[torch.Tensor] = []
        fake_anchor_process, _ = model.synthesize(reference_features, process_anchor_age, zero_process)
        for idx in range(self.config.model.n_processes):
            process_latents = torch.zeros(batch, self.config.model.n_processes, device=reference_features.device)
            process_latents[:, idx] = 1.0
            fake_process, _ = model.synthesize(reference_features, process_anchor_age, process_latents)
            process_targets.append(fake_process)
            process_responses.append(((fake_process - fake_anchor_process).abs() / baseline_abs).mean() * 100.0)
            process_shrinkage_penalties.append(nonpositive_change_loss(fake_process - fake_anchor_process))

        mean_process_sensitivity_pct = (
            torch.stack(process_responses).mean() if process_responses else torch.tensor(0.0, device=reference_features.device)
        )
        pairwise_distances: list[torch.Tensor] = []
        for i in range(len(process_targets)):
            for j in range(i + 1, len(process_targets)):
                pairwise_distances.append(((process_targets[i] - process_targets[j]).abs() / baseline_abs).mean() * 100.0)
        process_separation_pct = (
            torch.stack(pairwise_distances).mean()
            if pairwise_distances
            else torch.tensor(0.0, device=reference_features.device)
        )
        mean_process_shrinkage_penalty = (
            torch.stack(process_shrinkage_penalties).mean()
            if process_shrinkage_penalties
            else torch.tensor(0.0, device=reference_features.device)
        )
        return (
            age_sensitivity_pct,
            mean_process_sensitivity_pct,
            process_separation_pct,
            age_shrinkage_penalty,
            mean_process_shrinkage_penalty,
        )

    @torch.no_grad()
    def _compute_latent_sensitivity_metrics(self, model: AgeDecoupledSurrealGAN, split_name: str) -> dict[str, Any]:
        frame = self._reference_frame_for_sensitivity(split_name)
        if frame.empty:
            return summarize_latent_sensitivity(
                age_sensitivity_pct_mean=0.0,
                process_sensitivity_pct_means={f"r{i + 1}": 0.0 for i in range(self.config.model.n_processes)},
                process_separation_pct_mean=0.0,
                age_positive_change_pct_mean=0.0,
                process_positive_change_pct_means={f"r{i + 1}": 0.0 for i in range(self.config.model.n_processes)},
            )

        sample_n = min(len(frame), self.config.training.sensitivity_eval_subjects)
        if sample_n < len(frame):
            frame = frame.sample(sample_n, random_state=self.config.data.split_seed)
        reference_features = torch.tensor(frame[self.feature_columns].to_numpy(dtype="float32"), device=self.device)
        batch = reference_features.shape[0]
        baseline_abs = reference_features.abs().clamp_min(1.0)
        zero_process = torch.zeros(batch, self.config.model.n_processes, device=self.device)
        age_low = self._age_tensor(self.config.data.age_latent_normalization_min, batch, self.device)
        age_high = self._age_tensor(self.config.data.age_latent_normalization_max, batch, self.device)
        anchor_age = self._age_tensor(self.config.training.sensitivity_process_anchor_age, batch, self.device)

        fake_low_age, _ = model.synthesize(reference_features, age_low, zero_process)
        fake_high_age, _ = model.synthesize(reference_features, age_high, zero_process)
        age_delta = fake_high_age - fake_low_age
        age_sensitivity = float(((age_delta.abs() / baseline_abs).mean() * 100.0).cpu())
        age_positive_change = float((((age_delta.clamp_min(0.0)) / baseline_abs).mean() * 100.0).cpu())

        process_metrics: dict[str, float] = {}
        process_positive_metrics: dict[str, float] = {}
        process_targets: list[torch.Tensor] = []
        fake_anchor_process, _ = model.synthesize(reference_features, anchor_age, zero_process)
        for idx in range(self.config.model.n_processes):
            process_latents = torch.zeros(batch, self.config.model.n_processes, device=self.device)
            process_latents[:, idx] = 1.0
            fake_process, _ = model.synthesize(reference_features, anchor_age, process_latents)
            process_targets.append(fake_process)
            process_delta = fake_process - fake_anchor_process
            process_metrics[f"r{idx + 1}"] = float(((process_delta.abs() / baseline_abs).mean() * 100.0).cpu())
            process_positive_metrics[f"r{idx + 1}"] = float(
                (((process_delta.clamp_min(0.0)) / baseline_abs).mean() * 100.0).cpu()
            )

        pairwise_distances: list[float] = []
        for i in range(len(process_targets)):
            for j in range(i + 1, len(process_targets)):
                value = (((process_targets[i] - process_targets[j]).abs() / baseline_abs).mean() * 100.0).cpu()
                pairwise_distances.append(float(value))
        process_separation = float(np.mean(pairwise_distances)) if pairwise_distances else 0.0
        return summarize_latent_sensitivity(
            age_sensitivity_pct_mean=age_sensitivity,
            process_sensitivity_pct_means=process_metrics,
            process_separation_pct_mean=process_separation,
            age_positive_change_pct_mean=age_positive_change,
            process_positive_change_pct_means=process_positive_metrics,
        )

    def _checkpoint_epochs(self) -> list[int]:
        if self.config.training.save_every and self.config.training.save_every > 0:
            interval = self.config.training.save_every
        else:
            target = max(1, self.config.training.target_regular_checkpoints)
            interval = max(1, -(-self.config.training.epochs // target))
        epochs = sorted(set(range(interval, self.config.training.epochs + 1, interval)))
        if self.config.training.epochs not in epochs:
            epochs.append(self.config.training.epochs)
        return epochs

    def _load_split(self, split_name: str) -> pd.DataFrame:
        return pd.read_csv(self.processed_dir / f"{split_name}.csv", low_memory=False)

    def _cohort_loader(self, split_name: str, cohort_bucket: str, shuffle: bool) -> DataLoader:
        frame = self._load_split(split_name)
        frame = frame.loc[frame["cohort_bucket"] == cohort_bucket].dropna(subset=["age"] + self.feature_columns)
        dataset = CohortDataset(
            frame=frame,
            feature_columns=self.feature_columns,
            age_min=self.config.data.age_latent_normalization_min,
            age_max=self.config.data.age_latent_normalization_max,
        )
        return DataLoader(
            dataset,
            batch_size=self.config.training.batch_size,
            shuffle=shuffle,
            num_workers=self.config.training.num_workers,
            persistent_workers=self.config.training.persistent_workers and self.config.training.num_workers > 0,
            drop_last=shuffle and len(dataset) > self.config.training.batch_size,
        )

    def _run_dir(self, trial_name: str | None = None) -> Path:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = self.config.experiment_name if trial_name is None else f"{self.config.experiment_name}_{trial_name}"
        return ensure_dir(Path(self.config.paths.runs_dir) / f"{stamp}_{name}")

    def _save_checkpoint(
        self,
        model: AgeDecoupledSurrealGAN,
        run_dir: Path,
        repetition_index: int,
        epoch: int,
        optimizer_g: AdamW | None = None,
        optimizer_d: AdamW | None = None,
        scaler: GradScaler | None = None,
        best: bool = False,
    ) -> Path:
        payload = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "n_processes": self.config.model.n_processes,
            "feature_columns": self.feature_columns,
            "reference_template": self.reference_template.to_dict(),
            "config": self.config.as_dict(),
            "optimizer_g_state": optimizer_g.state_dict() if optimizer_g is not None else None,
            "optimizer_d_state": optimizer_d.state_dict() if optimizer_d is not None else None,
            "scaler_state": scaler.state_dict() if scaler is not None else None,
        }
        if best:
            best_path = run_dir / f"repetition_{repetition_index:02d}_best.pt"
            torch.save(payload, best_path)
            return best_path
        checkpoint_path = run_dir / f"repetition_{repetition_index:02d}_epoch_{epoch:03d}.pt"
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def _load_checkpoint_state(
        self,
        checkpoint_path: Path,
        model: AgeDecoupledSurrealGAN,
        optimizer_g: AdamW | None = None,
        optimizer_d: AdamW | None = None,
        scaler: GradScaler | None = None,
    ) -> int:
        payload = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(payload["model_state"])
        if optimizer_g is not None and payload.get("optimizer_g_state") is not None:
            optimizer_g.load_state_dict(payload["optimizer_g_state"])
        if optimizer_d is not None and payload.get("optimizer_d_state") is not None:
            optimizer_d.load_state_dict(payload["optimizer_d_state"])
        if scaler is not None and payload.get("scaler_state") is not None:
            scaler.load_state_dict(payload["scaler_state"])
        return int(payload.get("epoch", 0))

    def _load_existing_records(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        frame = pd.read_csv(path)
        return frame.to_dict(orient="records")

    def _predict_split(
        self,
        model: AgeDecoupledSurrealGAN,
        split_name: str,
    ) -> pd.DataFrame:
        frame = self._load_split(split_name)
        features = torch.tensor(frame[self.feature_columns].values, dtype=torch.float32, device=self.device)
        reference_values = torch.tensor(
            self.reference_template[self.feature_columns].values,
            dtype=torch.float32,
            device=self.device,
        )
        reference = reference_values.unsqueeze(0).repeat(features.shape[0], 1)
        outputs = model.infer(features, reference)
        result = frame[
            ["subject_id", "study", "age", "sex", "diagnosis_raw", "diagnosis_group", "cohort_bucket"]
        ].copy()
        result["age_latent"] = outputs["age_latent"].detach().cpu().numpy().reshape(-1)
        process = outputs["process_latents"].detach().cpu().numpy()
        for idx in range(self.config.model.n_processes):
            result[f"r{idx + 1}"] = process[:, idx]
        return result

    def _evaluate_model(self, model: AgeDecoupledSurrealGAN, split_name: str) -> tuple[dict[str, Any], pd.DataFrame]:
        prediction_frame = self._predict_split(model, split_name)
        metrics = evaluate_prediction_frame(prediction_frame, self.config.model.n_processes)
        sensitivity_metrics = self._compute_latent_sensitivity_metrics(model, split_name)
        metrics = {
            **metrics,
            **sensitivity_metrics,
            "quality_score": float(metrics["composite_score"] + 0.25 * sensitivity_metrics["latent_sensitivity_score"]),
            "directional_quality_score": float(
                metrics["composite_score"] + 0.25 * sensitivity_metrics["directional_latent_sensitivity_score"]
            ),
        }
        return metrics, prediction_frame

    def _train_epoch(
        self,
        model: AgeDecoupledSurrealGAN,
        reference_loader: DataLoader,
        target_loader: DataLoader,
        optimizer_g: AdamW,
        optimizer_d: AdamW,
        scaler: GradScaler,
    ) -> dict[str, float]:
        model.train()
        amp_enabled = self.config.training.use_amp and self.device.type == "cuda"
        target_iterator = cycle(target_loader)
        metrics: dict[str, list[float]] = {}

        for reference_batch in reference_loader:
            target_batch = next(target_iterator)
            ref_x = reference_batch["features"].to(self.device)
            ref_age = reference_batch["age_norm"].to(self.device).unsqueeze(1)
            tar_x = target_batch["features"].to(self.device)
            tar_age = target_batch["age_norm"].to(self.device).unsqueeze(1)
            self._check_finite_tensor("batch::ref_x", ref_x)
            self._check_finite_tensor("batch::tar_x", tar_x)

            with autocast('cuda', enabled=amp_enabled):
                outputs = model(ref_x, tar_x)
                self._check_finite_tensor("output::fake_target_detached", outputs.fake_target.detach())
                d_loss = discriminator_loss(
                    model.discriminate(tar_x),
                    model.discriminate(outputs.fake_target.detach()),
                )
                self._check_finite_tensor("loss::discriminator", d_loss)
            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)

            with autocast('cuda', enabled=amp_enabled):
                outputs = model(ref_x, tar_x)
                self._check_finite_tensor("output::age_latent", outputs.age_latent)
                self._check_finite_tensor("output::process_latents", outputs.process_latents)
                self._check_finite_tensor("output::fake_target", outputs.fake_target)
                self._check_finite_tensor("output::fake_delta", outputs.fake_delta)
                fake_logits = model.discriminate(outputs.fake_target)
                self._check_finite_tensor("output::fake_logits", fake_logits)
                _, low_delta = model.synthesize(
                    ref_x,
                    outputs.age_latent,
                    outputs.process_latents * 0.5,
                )
                low_activation_age = torch.rand_like(outputs.age_latent) * self.config.losses.low_activation_max
                low_activation_process = (
                    torch.rand_like(outputs.process_latents) * self.config.losses.low_activation_max
                )
                low_identity_target, _ = model.synthesize(ref_x, low_activation_age, low_activation_process)
                ref_age_latent, ref_process_latents = model.encode(ref_x)
                total_delta = outputs.age_delta + outputs.process_deltas.sum(dim=1)
                losses = {
                    "adv": generator_adversarial_loss(fake_logits),
                    "age_sup": F.mse_loss(outputs.age_latent, tar_age),
                    "ref_age_sup": F.mse_loss(ref_age_latent, ref_age),
                    "age_adv": F.mse_loss(outputs.age_adversary_pred, tar_age),
                    "latent_recon": F.mse_loss(outputs.age_latent_recon, outputs.age_latent.detach())
                    + F.mse_loss(outputs.process_latent_recon, outputs.process_latents.detach()),
                    "decompose": F.mse_loss(total_delta, outputs.fake_delta),
                    "identity": F.l1_loss(outputs.identity_target, ref_x),
                    "monotonicity": monotonicity_loss(low_delta, outputs.fake_delta),
                    "orthogonality": orthogonality_loss(outputs.process_deltas),
                    "covariance": covariance_penalty(outputs.age_latent, outputs.process_latents),
                    "reference_process": ref_process_latents.abs().mean(),
                    "change_mag": change_magnitude_loss(outputs.fake_delta),
                    "low_identity": F.l1_loss(low_identity_target, ref_x),
                    "process_age_corr": correlation_penalty(outputs.process_latents, tar_age),
                    "process_sparse": latent_sparsity_loss(outputs.process_latents),
                }
                sensitivity_reference = ref_x[: min(ref_x.shape[0], 16)]
                (
                    age_sensitivity_pct,
                    process_sensitivity_pct,
                    process_separation_pct,
                    age_shrinkage_penalty,
                    process_shrinkage_penalty,
                ) = self._latent_sensitivity_batch_metrics(model, sensitivity_reference)
                losses["age_sensitivity"] = torch.relu(
                    torch.tensor(self.config.losses.age_sensitivity_target_pct, device=ref_x.device) - age_sensitivity_pct
                )
                losses["process_sensitivity"] = torch.relu(
                    torch.tensor(self.config.losses.process_sensitivity_target_pct, device=ref_x.device)
                    - process_sensitivity_pct
                )
                losses["age_shrinkage"] = age_shrinkage_penalty
                losses["process_shrinkage"] = process_shrinkage_penalty
                total_loss = self._combine_weighted_losses(
                    losses,
                    {
                        "adv": self.config.losses.adversarial,
                        "age_sup": self.config.losses.age_supervision,
                        "ref_age_sup": self.config.losses.reference_age_supervision,
                        "age_adv": self.config.losses.age_adversary,
                        "latent_recon": self.config.losses.latent_reconstruction,
                        "decompose": self.config.losses.decomposition,
                        "identity": self.config.losses.identity,
                        "monotonicity": self.config.losses.monotonicity,
                        "orthogonality": self.config.losses.process_orthogonality,
                        "covariance": self.config.losses.age_process_covariance,
                        "reference_process": self.config.losses.reference_process_sparsity,
                        "change_mag": self.config.losses.change_magnitude,
                        "low_identity": self.config.losses.low_activation_identity,
                        "process_age_corr": self.config.losses.process_age_correlation,
                        "process_sparse": self.config.losses.process_latent_sparsity,
                        "age_sensitivity": self.config.losses.age_sensitivity,
                        "process_sensitivity": self.config.losses.process_sensitivity,
                        "age_shrinkage": self.config.losses.age_shrinkage,
                        "process_shrinkage": self.config.losses.process_shrinkage,
                    },
                )
            optimizer_g.zero_grad(set_to_none=True)
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer_g)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.training.gradient_clip_norm)
            scaler.step(optimizer_g)
            scaler.update()

            state_metrics = {
                "state_age_latent_mean": outputs.age_latent.mean(),
                "state_process_latent_abs_mean": outputs.process_latents.abs().mean(),
                "state_fake_change_abs_mean": outputs.fake_delta.abs().mean(),
                "state_age_sensitivity_pct": age_sensitivity_pct.detach(),
                "state_process_sensitivity_pct": process_sensitivity_pct.detach(),
                "state_process_separation_pct": process_separation_pct.detach(),
                "state_age_growth_penalty": age_shrinkage_penalty.detach(),
                "state_process_growth_penalty": process_shrinkage_penalty.detach(),
            }
            step_metrics = {"discriminator": d_loss, "generator_total": total_loss, **losses, **state_metrics}
            for key, value in step_metrics.items():
                metrics.setdefault(key, []).append(float(value.detach().cpu()))

        return {key: float(sum(values) / max(len(values), 1)) for key, values in metrics.items()}

    def train(self, trial_name: str | None = None, resume_run_dir: str | None = None) -> dict[str, Any]:
        configured_resume_dir = (
            self.config.training.resume_run_dir if getattr(self.config.training, "resume_run_dir", None) else None
        )
        active_resume_dir = resume_run_dir or configured_resume_dir or None
        run_dir = ensure_dir(Path(active_resume_dir)) if active_resume_dir else self._run_dir(trial_name=trial_name)
        ensure_dir(run_dir / "predictions")
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "tensorboard")
        ensure_dir(run_dir / "logs")

        save_json(run_dir / "resolved_config.json", self.config.as_dict())
        checkpoint_epochs = self._checkpoint_epochs()

        writer_cls = SummaryWriter if SummaryWriter is not None else None
        repo_root = Path(__file__).resolve().parents[2]
        doc_payloads: list[tuple[str, str]] = []
        for filename, tag in [
            ("objectives_and_losses.md", "docs/objectives_and_losses"),
            ("metrics_and_logging.md", "docs/metrics_and_logging"),
        ]:
            text = load_doc_text(repo_root, filename)
            if text is not None:
                doc_payloads.append((tag, text))

        reference_loader = self._cohort_loader("train", "ref", shuffle=True)
        target_loader = self._cohort_loader("train", "tar", shuffle=True)

        repetition_summaries = self._load_existing_records(run_dir / "metrics" / "repetition_summary.csv")
        epoch_history_records = self._load_existing_records(run_dir / "metrics" / "epoch_history.csv")
        repetition_summaries = [
            {
                "repetition_index": int(row["repetition_index"]),
                "best_checkpoint": row["best_checkpoint"],
                "best_score": float(row["best_score"]),
            }
            for row in repetition_summaries
        ]
        repetition_val_paths: list[Path] = []
        for row in repetition_summaries:
            val_path = run_dir / "predictions" / f"repetition_{int(row['repetition_index']):02d}_val.csv"
            if val_path.exists():
                repetition_val_paths.append(val_path)
        log_path = run_dir / "logs" / "train.log"
        startup_lines = startup_summary_lines(
            experiment_name=self.config.experiment_name if trial_name is None else trial_name,
            config_path=str(
                Path(self.config_path).expanduser().resolve() if self.config_path else default_config_path().resolve()
            ),
            device=str(self.device),
            n_features=self.n_features,
            n_processes=self.config.model.n_processes,
            repetitions=self.config.training.repetitions,
            epochs=self.config.training.epochs,
            batch_size=self.config.training.batch_size,
            learning_rate=self.config.training.learning_rate,
            discriminator_learning_rate=self.config.training.discriminator_learning_rate,
            checkpoint_epochs=checkpoint_epochs,
            num_workers=self.config.training.num_workers,
            persistent_workers=self.config.training.persistent_workers,
            use_amp=self.config.training.use_amp,
            compile_model=self.config.training.compile_model,
            encoder_hidden_dims=self.config.model.encoder_hidden_dims,
            generator_hidden_dims=self.config.model.generator_hidden_dims,
            discriminator_hidden_dims=self.config.model.discriminator_hidden_dims,
            decomposer_hidden_dims=self.config.model.decomposer_hidden_dims,
        )
        startup_lines.extend(
            [
                (
                    "Reference scale: "
                    f"abs_mean={self.reference_template_abs_mean:.2f}, abs_max={self.reference_template_abs_max:.2f}"
                ),
                (
                    "AMP guidance: raw ROI inputs are not normalized; fp16 autocast on CUDA can be unstable. "
                    f"use_amp={self.config.training.use_amp}"
                ),
            ]
        )
        if active_resume_dir:
            resume_line = f"Resume mode: {Path(active_resume_dir).resolve()}"
            startup_lines.append(resume_line)
        for line in startup_lines:
            print(line)
            append_log_line(log_path, line)

        completed_repetitions = {int(row["repetition_index"]) for row in repetition_summaries}
        for repetition_index in range(self.config.training.repetitions):
            if repetition_index in completed_repetitions:
                skip_line = f"Skipping completed repetition {repetition_index + 1}/{self.config.training.repetitions}"
                print(skip_line)
                append_log_line(log_path, skip_line)
                continue
            seed_everything(self.config.data.split_seed + repetition_index)
            model = AgeDecoupledSurrealGAN(self.n_features, self.config).to(self.device)
            writer = (
                writer_cls(run_dir / "tensorboard" / f"repetition_{repetition_index:02d}")
                if writer_cls is not None
                else None
            )
            if writer is not None:
                writer.add_text("run/summary", "\n".join(startup_lines), 0)
                for tag, text in doc_payloads:
                    writer.add_text(tag, text, 0)
            rep_line = f"Starting repetition {repetition_index + 1}/{self.config.training.repetitions}"
            print(rep_line)
            append_log_line(log_path, rep_line)

            optimizer_g = AdamW(
                list(model.encoder.parameters())
                + list(model.generator.parameters())
                + list(model.decomposer.parameters())
                + list(model.age_reconstructor.parameters())
                + list(model.process_reconstructors.parameters())
                + list(model.age_adversary.parameters()),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            optimizer_d = AdamW(
                model.discriminator.parameters(),
                lr=self.config.training.discriminator_learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
            scaler = GradScaler('cuda', enabled=self.config.training.use_amp and self.device.type == "cuda")
            start_epoch = 1
            existing_checkpoints = sorted(run_dir.glob(f"repetition_{repetition_index:02d}_epoch_*.pt"))
            best_checkpoint = run_dir / f"repetition_{repetition_index:02d}_best.pt"
            existing_rows = [
                row for row in epoch_history_records if int(row.get("repetition_index", -1)) == repetition_index
            ]
            best_score = max((float(row["selection_score"]) for row in existing_rows), default=float("-inf"))
            best_val_frame: pd.DataFrame | None = None
            best_val_metrics: dict[str, Any] | None = None
            if existing_checkpoints:
                latest_checkpoint = max(existing_checkpoints, key=lambda path: int(path.stem.split("_")[-1]))
                start_epoch = self._load_checkpoint_state(latest_checkpoint, model, optimizer_g, optimizer_d, scaler) + 1
                epoch_history_records = [
                    row
                    for row in epoch_history_records
                    if not (
                        int(row.get("repetition_index", -1)) == repetition_index
                        and int(row.get("epoch", -1)) >= start_epoch
                    )
                ]
                resume_rep_line = (
                    f"Resuming repetition {repetition_index + 1}/{self.config.training.repetitions} "
                    f"from {latest_checkpoint.name} at epoch {start_epoch}"
                )
                print(resume_rep_line)
                append_log_line(log_path, resume_rep_line)
            if self.config.training.compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)  # type: ignore[assignment]

            for epoch in range(start_epoch, self.config.training.epochs + 1):
                train_metrics = self._train_epoch(model, reference_loader, target_loader, optimizer_g, optimizer_d, scaler)
                val_metrics, val_frame = self._evaluate_model(model, "val")
                score = float(val_metrics[self.config.training.monitor_metric])

                epoch_record = {
                    "repetition_index": repetition_index,
                    "epoch": epoch,
                    **flatten_metrics({f"train_{key}": value for key, value in train_metrics.items()}),
                    **flatten_metrics({f"val_{key}": value for key, value in val_metrics.items()}),
                    "selection_score": score,
                }
                epoch_history_records.append(epoch_record)

                if self.config.training.console_log_every > 0 and epoch % self.config.training.console_log_every == 0:
                    line = epoch_log_line(
                        repetition_index,
                        self.config.training.repetitions,
                        epoch,
                        self.config.training.epochs,
                        train_metrics,
                        val_metrics,
                    )
                    print(line)
                    append_log_line(log_path, line)

                if writer is not None:
                    for key, value in train_metrics.items():
                        tag = TRAIN_TENSORBOARD_TAGS.get(key, f"train/{key}")
                        writer.add_scalar(tag, value, epoch)
                    for key, value in val_metrics.items():
                        if isinstance(value, (float, int)):
                            tag = VAL_TENSORBOARD_TAGS.get(key, f"metric/validation/{key}")
                            writer.add_scalar(tag, float(value), epoch)

                if epoch in checkpoint_epochs:
                    self._save_checkpoint(
                        model,
                        run_dir,
                        repetition_index,
                        epoch,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        scaler=scaler,
                    )

                if score > best_score:
                    best_score = score
                    best_checkpoint = self._save_checkpoint(
                        model,
                        run_dir,
                        repetition_index,
                        epoch,
                        optimizer_g=optimizer_g,
                        optimizer_d=optimizer_d,
                        scaler=scaler,
                        best=True,
                    )
                    best_val_frame = val_frame
                    best_val_metrics = val_metrics

            if best_checkpoint.exists() and (best_val_frame is None or best_val_metrics is None):
                best_model = AgeDecoupledSurrealGAN(self.n_features, self.config).to(self.device)
                self._load_checkpoint_state(best_checkpoint, best_model)
                if self.config.training.compile_model and hasattr(torch, "compile"):
                    best_model = torch.compile(best_model)  # type: ignore[assignment]
                best_model.eval()
                best_val_metrics, best_val_frame = self._evaluate_model(best_model, "val")

            assert best_checkpoint is not None
            assert best_val_frame is not None
            assert best_val_metrics is not None
            val_prediction_path = run_dir / "predictions" / f"repetition_{repetition_index:02d}_val.csv"
            save_prediction_frame(best_val_frame, val_prediction_path)
            save_metrics(best_val_metrics, run_dir / "metrics" / f"repetition_{repetition_index:02d}_val.json")
            repetition_val_paths = [
                path for path in repetition_val_paths if path.name != val_prediction_path.name
            ] + [val_prediction_path]
            repetition_summaries = [
                row for row in repetition_summaries if int(row["repetition_index"]) != repetition_index
            ]
            repetition_summaries.append(
                {
                    "repetition_index": repetition_index,
                    "best_checkpoint": str(best_checkpoint),
                    "best_score": best_score,
                }
            )
            if writer is not None:
                writer.close()

        repetition_summaries = sorted(repetition_summaries, key=lambda row: int(row["repetition_index"]))
        repetition_val_paths = sorted(repetition_val_paths, key=lambda path: path.name)
        save_records_csv(run_dir / "metrics" / "epoch_history.csv", epoch_history_records)
        save_jsonl(run_dir / "logs" / "epoch_history.jsonl", epoch_history_records)
        save_records_csv(run_dir / "metrics" / "repetition_summary.csv", repetition_summaries)

        agreement = aggregate_repetition_predictions(repetition_val_paths, self.config.model.n_processes)
        selected_repetition = agreement["best_repetition_index"]
        selected_checkpoint = Path(repetition_summaries[selected_repetition]["best_checkpoint"])

        best_model = AgeDecoupledSurrealGAN(self.n_features, self.config).to(self.device)
        checkpoint = torch.load(selected_checkpoint, map_location=self.device)
        best_model.load_state_dict(checkpoint["model_state"])
        best_model.eval()

        split_metrics: dict[str, Any] = {}
        for split_name in ["train", "val", "id_test", "ood_test", "application"]:
            prediction_frame = self._predict_split(best_model, split_name)
            prediction_path = run_dir / "predictions" / f"{split_name}.csv"
            save_prediction_frame(prediction_frame, prediction_path)
            metrics = evaluate_prediction_frame(prediction_frame, self.config.model.n_processes)
            sensitivity_metrics = self._compute_latent_sensitivity_metrics(best_model, split_name)
            metrics = {
                **metrics,
                **sensitivity_metrics,
                "quality_score": float(metrics["composite_score"] + 0.25 * sensitivity_metrics["latent_sensitivity_score"]),
                "directional_quality_score": float(
                    metrics["composite_score"] + 0.25 * sensitivity_metrics["directional_latent_sensitivity_score"]
                ),
            }
            split_metrics[split_name] = metrics
            save_metrics(metrics, run_dir / "metrics" / f"{split_name}.json")

        summary = {
            "run_dir": str(run_dir),
            "selected_repetition": selected_repetition,
            "selected_checkpoint": str(selected_checkpoint),
            "agreement": agreement,
            "repetitions": repetition_summaries,
            "split_metrics": split_metrics,
        }
        save_json(run_dir / "run_summary.json", summary)
        save_split_metrics_tables(split_metrics, run_dir / "metrics")
        save_run_markdown_summary(summary, run_dir / "metrics" / "run_summary.md")
        return summary
