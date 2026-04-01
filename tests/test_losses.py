from __future__ import annotations

import torch

from age_decoupled_surrealgan.losses import (
    change_magnitude_loss,
    correlation_penalty,
    latent_sparsity_loss,
)


def test_change_magnitude_loss_is_nonnegative():
    tensor = torch.tensor([[1.0, -2.0], [0.5, -0.5]])
    assert float(change_magnitude_loss(tensor)) >= 0.0


def test_correlation_penalty_is_nonnegative():
    process = torch.tensor([[0.1, 0.2], [0.2, 0.4], [0.3, 0.6]])
    age = torch.tensor([[0.1], [0.2], [0.3]])
    assert float(correlation_penalty(process, age)) >= 0.0


def test_latent_sparsity_loss_is_nonnegative():
    latents = torch.tensor([[1.0, -1.0], [0.0, 0.5]])
    assert float(latent_sparsity_loss(latents)) >= 0.0
