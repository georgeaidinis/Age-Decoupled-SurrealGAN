from __future__ import annotations

import torch

from age_decoupled_surrealgan.config import ProjectConfig
from age_decoupled_surrealgan.model import AgeDecoupledSurrealGAN


def test_model_forward_shapes():
    cfg = ProjectConfig()
    cfg.model.n_processes = 3
    model = AgeDecoupledSurrealGAN(n_features=5, config=cfg)
    reference = torch.randn(4, 5)
    target = torch.randn(4, 5)
    outputs = model(reference, target)
    assert outputs.age_latent.shape == (4, 1)
    assert outputs.process_latents.shape == (4, 3)
    assert outputs.process_deltas.shape == (4, 3, 5)
