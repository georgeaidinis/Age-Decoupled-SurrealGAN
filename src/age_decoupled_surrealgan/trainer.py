from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import ProjectConfig
from .data.dataset import CohortDataset
from .evaluation import aggregate_repetition_predictions, evaluate_prediction_frame, save_metrics, save_prediction_frame
from .inference import resolve_device
from .losses import (
    change_magnitude_loss,
    correlation_penalty,
    covariance_penalty,
    discriminator_loss,
    generator_adversarial_loss,
    latent_sparsity_loss,
    monotonicity_loss,
    orthogonality_loss,
)
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
        best: bool = False,
    ) -> Path:
        checkpoint_path = run_dir / f"repetition_{repetition_index:02d}_epoch_{epoch:03d}.pt"
        payload = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "n_processes": self.config.model.n_processes,
            "feature_columns": self.feature_columns,
            "reference_template": self.reference_template.to_dict(),
            "config": self.config.as_dict(),
        }
        torch.save(payload, checkpoint_path)
        if best:
            best_path = run_dir / f"repetition_{repetition_index:02d}_best.pt"
            torch.save(payload, best_path)
            return best_path
        return checkpoint_path

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

            with autocast('cuda', enabled=amp_enabled):
                outputs = model(ref_x, tar_x)
                d_loss = discriminator_loss(
                    model.discriminate(tar_x),
                    model.discriminate(outputs.fake_target.detach()),
                )
            optimizer_d.zero_grad(set_to_none=True)
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)

            with autocast('cuda', enabled=amp_enabled):
                outputs = model(ref_x, tar_x)
                fake_logits = model.discriminate(outputs.fake_target)
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
                total_loss = (
                    self.config.losses.adversarial * losses["adv"]
                    + self.config.losses.age_supervision * losses["age_sup"]
                    + self.config.losses.reference_age_supervision * losses["ref_age_sup"]
                    + self.config.losses.age_adversary * losses["age_adv"]
                    + self.config.losses.latent_reconstruction * losses["latent_recon"]
                    + self.config.losses.decomposition * losses["decompose"]
                    + self.config.losses.identity * losses["identity"]
                    + self.config.losses.monotonicity * losses["monotonicity"]
                    + self.config.losses.process_orthogonality * losses["orthogonality"]
                    + self.config.losses.age_process_covariance * losses["covariance"]
                    + self.config.losses.reference_process_sparsity * losses["reference_process"]
                    + self.config.losses.change_magnitude * losses["change_mag"]
                    + self.config.losses.low_activation_identity * losses["low_identity"]
                    + self.config.losses.process_age_correlation * losses["process_age_corr"]
                    + self.config.losses.process_latent_sparsity * losses["process_sparse"]
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
            }
            step_metrics = {"discriminator": d_loss, "generator_total": total_loss, **losses, **state_metrics}
            for key, value in step_metrics.items():
                metrics.setdefault(key, []).append(float(value.detach().cpu()))

        return {key: float(sum(values) / max(len(values), 1)) for key, values in metrics.items()}

    def train(self, trial_name: str | None = None) -> dict[str, Any]:
        run_dir = self._run_dir(trial_name=trial_name)
        ensure_dir(run_dir / "predictions")
        ensure_dir(run_dir / "metrics")
        ensure_dir(run_dir / "tensorboard")
        ensure_dir(run_dir / "logs")
        ensure_dir(run_dir / "docs")

        save_json(run_dir / "resolved_config.json", self.config.as_dict())

        writer = SummaryWriter(run_dir / "tensorboard") if SummaryWriter is not None else None
        repo_root = Path(__file__).resolve().parents[2]
        if writer is not None:
            for filename, tag in [
                ("objectives_and_losses.md", "docs/objectives_and_losses"),
                ("metrics_and_logging.md", "docs/metrics_and_logging"),
            ]:
                text = load_doc_text(repo_root, filename)
                if text is not None:
                    writer.add_text(tag, text, 0)
                    (run_dir / "docs" / filename).write_text(text, encoding="utf-8")

        reference_loader = self._cohort_loader("train", "ref", shuffle=True)
        target_loader = self._cohort_loader("train", "tar", shuffle=True)

        repetition_val_paths: list[Path] = []
        repetition_summaries: list[dict[str, Any]] = []
        epoch_history_records: list[dict[str, Any]] = []
        log_path = run_dir / "logs" / "train.log"

        for repetition_index in range(self.config.training.repetitions):
            seed_everything(self.config.data.split_seed + repetition_index)
            model = AgeDecoupledSurrealGAN(self.n_features, self.config).to(self.device)
            if self.config.training.compile_model and hasattr(torch, "compile"):
                model = torch.compile(model)  # type: ignore[assignment]

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

            best_score = float("-inf")
            best_checkpoint: Path | None = None
            best_val_frame: pd.DataFrame | None = None
            best_val_metrics: dict[str, Any] | None = None

            for epoch in range(1, self.config.training.epochs + 1):
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
                        tag = TRAIN_TENSORBOARD_TAGS.get(key, f"train/repetition_{repetition_index}/{key}")
                        writer.add_scalar(f"{tag}/repetition_{repetition_index}", value, epoch)
                    for key, value in val_metrics.items():
                        if isinstance(value, (float, int)):
                            tag = VAL_TENSORBOARD_TAGS.get(key, f"metric/validation/{key}")
                            writer.add_scalar(f"{tag}/repetition_{repetition_index}", value, epoch)

                if epoch % self.config.training.save_every == 0:
                    self._save_checkpoint(model, run_dir, repetition_index, epoch)

                if score > best_score:
                    best_score = score
                    best_checkpoint = self._save_checkpoint(model, run_dir, repetition_index, epoch, best=True)
                    best_val_frame = val_frame
                    best_val_metrics = val_metrics

            assert best_checkpoint is not None
            assert best_val_frame is not None
            assert best_val_metrics is not None
            val_prediction_path = run_dir / "predictions" / f"repetition_{repetition_index:02d}_val.csv"
            save_prediction_frame(best_val_frame, val_prediction_path)
            save_metrics(best_val_metrics, run_dir / "metrics" / f"repetition_{repetition_index:02d}_val.json")
            repetition_val_paths.append(val_prediction_path)
            repetition_summaries.append(
                {
                    "repetition_index": repetition_index,
                    "best_checkpoint": str(best_checkpoint),
                    "best_score": best_score,
                }
            )

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
        if writer is not None:
            writer.close()
        return summary
