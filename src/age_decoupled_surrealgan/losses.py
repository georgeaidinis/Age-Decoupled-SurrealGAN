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


def generator_redundancy_loss(process_components: torch.Tensor) -> torch.Tensor:
    if process_components.ndim != 3:
        raise ValueError("process_components must have shape [batch, n_processes, n_features]")
    mean_vectors = process_components.mean(dim=0)
    normalized = mean_vectors / mean_vectors.norm(dim=1, keepdim=True).clamp_min(1e-6)
    gram = normalized @ normalized.transpose(0, 1)
    eye = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
    off_diagonal = (gram - eye).abs()
    mask = ~eye.bool()
    return off_diagonal[mask].mean() if mask.any() else torch.tensor(0.0, device=gram.device, dtype=gram.dtype)


def generator_separation_margin_loss(process_components: torch.Tensor, margin: float) -> torch.Tensor:
    if process_components.ndim != 3:
        raise ValueError("process_components must have shape [batch, n_processes, n_features]")
    mean_vectors = process_components.mean(dim=0)
    pairwise_distances: list[torch.Tensor] = []
    for i in range(mean_vectors.shape[0]):
        for j in range(i + 1, mean_vectors.shape[0]):
            pairwise_distances.append((mean_vectors[i] - mean_vectors[j]).abs().mean())
    if not pairwise_distances:
        return torch.tensor(0.0, device=process_components.device, dtype=process_components.dtype)
    distances = torch.stack(pairwise_distances)
    return F.relu(torch.tensor(float(margin), device=distances.device, dtype=distances.dtype) - distances).mean()


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


def pairwise_latent_correlation_penalty(process_latents: torch.Tensor) -> torch.Tensor:
    if process_latents.ndim != 2 or process_latents.shape[1] <= 1:
        return torch.tensor(0.0, device=process_latents.device, dtype=process_latents.dtype)
    centered = process_latents - process_latents.mean(dim=0, keepdim=True)
    std = centered.std(dim=0, keepdim=True).clamp_min(1e-6)
    standardized = centered / std
    corr = standardized.transpose(0, 1) @ standardized / max(process_latents.shape[0] - 1, 1)
    eye = torch.eye(corr.shape[0], device=corr.device, dtype=corr.dtype)
    mask = ~eye.bool()
    return (corr[mask] ** 2).mean() if mask.any() else torch.tensor(0.0, device=corr.device, dtype=corr.dtype)


def nonpositive_change_loss(delta: torch.Tensor) -> torch.Tensor:
    return F.relu(delta).mean()
