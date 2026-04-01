from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return 0.5 * (real_loss + fake_loss)


def generator_adversarial_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))


def change_magnitude_loss(fake_delta: torch.Tensor) -> torch.Tensor:
    return fake_delta.abs().mean()


def covariance_penalty(age_latent: torch.Tensor, process_latents: torch.Tensor) -> torch.Tensor:
    centered_age = age_latent - age_latent.mean(dim=0, keepdim=True)
    centered_process = process_latents - process_latents.mean(dim=0, keepdim=True)
    cov = centered_age.transpose(0, 1) @ centered_process / max(age_latent.shape[0] - 1, 1)
    return (cov**2).mean()


def orthogonality_loss(process_deltas: torch.Tensor) -> torch.Tensor:
    if process_deltas.ndim != 3:
        raise ValueError("process_deltas must have shape [batch, n_processes, n_features]")
    mean_vectors = process_deltas.mean(dim=0)
    norms = mean_vectors.norm(dim=1, keepdim=True).clamp_min(1e-6)
    normalized = mean_vectors / norms
    gram = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    return F.mse_loss(gram, eye)


def monotonicity_loss(low_delta: torch.Tensor, high_delta: torch.Tensor) -> torch.Tensor:
    low_mag = low_delta.abs().mean(dim=1)
    high_mag = high_delta.abs().mean(dim=1)
    return F.relu(low_mag - high_mag).mean()


def correlation_penalty(process_latents: torch.Tensor, age_targets: torch.Tensor) -> torch.Tensor:
    centered_process = process_latents - process_latents.mean(dim=0, keepdim=True)
    centered_age = age_targets - age_targets.mean(dim=0, keepdim=True)
    numerator = (centered_process * centered_age).mean(dim=0)
    denominator = centered_process.std(dim=0).clamp_min(1e-6) * centered_age.std(dim=0).clamp_min(1e-6)
    corr = numerator / denominator
    return (corr**2).mean()


def latent_sparsity_loss(latents: torch.Tensor) -> torch.Tensor:
    return latents.abs().mean()
