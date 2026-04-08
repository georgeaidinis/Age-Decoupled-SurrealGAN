from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.autograd import Function

from .config import ProjectConfig


class GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, coeff: float):
        ctx.coeff = coeff
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.coeff * grad_output, None


def grad_reverse(inputs: torch.Tensor, coeff: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(inputs, coeff)


def build_mlp(
    input_dim: int,
    hidden_dims: list[int],
    output_dim: int,
    dropout: float,
    final_activation: nn.Module | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous_dim = input_dim
    for hidden_dim in hidden_dims:
        layers.extend(
            [
                nn.Linear(previous_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout),
            ]
        )
        previous_dim = hidden_dim
    layers.append(nn.Linear(previous_dim, output_dim))
    if final_activation is not None:
        layers.append(final_activation)
    return nn.Sequential(*layers)


@dataclass
class LegacyModelOutputs:
    age_latent: torch.Tensor
    process_latents: torch.Tensor
    fake_target: torch.Tensor
    fake_delta: torch.Tensor
    age_delta: torch.Tensor
    process_deltas: torch.Tensor
    age_latent_recon: torch.Tensor
    process_latent_recon: torch.Tensor
    age_adversary_pred: torch.Tensor
    identity_target: torch.Tensor


class LegacyAgeDecoupledSurrealGAN(nn.Module):
    def __init__(self, n_features: int, config: ProjectConfig):
        super().__init__()
        self.n_features = n_features
        self.n_processes = config.model.n_processes
        self.age_latent_dim = config.model.age_latent_dim

        latent_dim = self.age_latent_dim + self.n_processes

        self.encoder = build_mlp(
            n_features,
            config.model.encoder_hidden_dims,
            latent_dim,
            config.model.dropout,
        )
        self.generator = build_mlp(
            n_features + latent_dim,
            config.model.generator_hidden_dims,
            n_features,
            config.model.dropout,
        )
        self.decomposer = build_mlp(
            n_features,
            config.model.decomposer_hidden_dims,
            n_features * (self.n_processes + 1),
            config.model.dropout,
        )
        self.age_reconstructor = build_mlp(
            n_features,
            [max(32, config.model.decomposer_hidden_dims[-1] // 2)],
            self.age_latent_dim,
            config.model.dropout,
        )
        self.process_reconstructors = nn.ModuleList(
            [
                build_mlp(
                    n_features,
                    [max(32, config.model.decomposer_hidden_dims[-1] // 2)],
                    1,
                    config.model.dropout,
                )
                for _ in range(self.n_processes)
            ]
        )
        self.discriminator = build_mlp(
            n_features,
            config.model.discriminator_hidden_dims,
            1,
            config.model.dropout,
        )
        self.age_adversary = build_mlp(
            self.n_processes,
            [max(16, self.n_processes * 4)],
            self.age_latent_dim,
            config.model.dropout,
        )

    def encode(self, target_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.sigmoid(self.encoder(target_features))
        age_latent = latent[:, : self.age_latent_dim]
        process_latents = latent[:, self.age_latent_dim :]
        return age_latent, process_latents

    def synthesize(
        self,
        reference_features: torch.Tensor,
        age_latent: torch.Tensor,
        process_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        latent = torch.cat([age_latent, process_latents], dim=1)
        delta = self.generator(torch.cat([reference_features, latent], dim=1))
        fake_target = reference_features + delta
        return fake_target, delta

    def decompose(self, delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decomposition = self.decomposer(delta).view(delta.shape[0], self.n_processes + 1, self.n_features)
        age_delta = decomposition[:, 0, :]
        process_deltas = decomposition[:, 1:, :]
        return age_delta, process_deltas

    def reconstruct_latents(
        self,
        age_delta: torch.Tensor,
        process_deltas: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        age_recon = torch.sigmoid(self.age_reconstructor(age_delta))
        process_recon = torch.cat(
            [torch.sigmoid(head(process_deltas[:, idx, :])) for idx, head in enumerate(self.process_reconstructors)],
            dim=1,
        )
        return age_recon, process_recon

    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        return self.discriminator(features)

    def predict_age_from_process(self, process_latents: torch.Tensor, reverse: bool = True) -> torch.Tensor:
        adversary_inputs = grad_reverse(process_latents) if reverse else process_latents
        return torch.sigmoid(self.age_adversary(adversary_inputs))

    def forward(self, reference_features: torch.Tensor, target_features: torch.Tensor) -> LegacyModelOutputs:
        age_latent, process_latents = self.encode(target_features)
        fake_target, fake_delta = self.synthesize(reference_features, age_latent, process_latents)
        age_delta, process_deltas = self.decompose(fake_delta)
        age_latent_recon, process_latent_recon = self.reconstruct_latents(age_delta, process_deltas)
        age_adversary_pred = self.predict_age_from_process(process_latents, reverse=True)

        zeros_age = torch.zeros(reference_features.shape[0], self.age_latent_dim, device=reference_features.device)
        zeros_process = torch.zeros(reference_features.shape[0], self.n_processes, device=reference_features.device)
        identity_target, _ = self.synthesize(reference_features, zeros_age, zeros_process)
        return LegacyModelOutputs(
            age_latent=age_latent,
            process_latents=process_latents,
            fake_target=fake_target,
            fake_delta=fake_delta,
            age_delta=age_delta,
            process_deltas=process_deltas,
            age_latent_recon=age_latent_recon,
            process_latent_recon=process_latent_recon,
            age_adversary_pred=age_adversary_pred,
            identity_target=identity_target,
        )

    @torch.no_grad()
    def infer(
        self,
        target_features: torch.Tensor,
        reference_template: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        age_latent, process_latents = self.encode(target_features)
        fake_target, fake_delta = self.synthesize(reference_template, age_latent, process_latents)
        age_delta, process_deltas = self.decompose(fake_delta)
        return {
            "age_latent": age_latent,
            "process_latents": process_latents,
            "synthetic_target": fake_target,
            "synthetic_delta": fake_delta,
            "age_delta": age_delta,
            "process_deltas": process_deltas,
        }
