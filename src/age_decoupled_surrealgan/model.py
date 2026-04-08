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
class GeneratorOutputs:
    fake_target: torch.Tensor
    total_delta: torch.Tensor
    age_component: torch.Tensor
    process_components: torch.Tensor
    age_basis: torch.Tensor
    process_bases: torch.Tensor


class AgeDecoupledSurrealGAN(nn.Module):
    model_version = "v2_sampled_additive"

    def __init__(self, n_features: int, config: ProjectConfig):
        super().__init__()
        self.n_features = n_features
        self.n_processes = config.model.n_processes
        self.age_latent_dim = config.model.age_latent_dim

        trunk_dim = config.model.generator_hidden_dims[-1]
        self.generator_trunk = build_mlp(
            n_features,
            config.model.generator_hidden_dims[:-1],
            trunk_dim,
            config.model.dropout,
            final_activation=nn.LeakyReLU(0.2, inplace=True),
        )
        self.age_basis_head = nn.Linear(trunk_dim, n_features)
        self.process_basis_head = nn.Linear(trunk_dim, n_features * self.n_processes)

        self.decomposer = build_mlp(
            n_features,
            config.model.decomposer_hidden_dims,
            n_features * (self.n_processes + 1),
            config.model.dropout,
        )
        self.age_reconstructor = build_mlp(
            n_features,
            [max(32, config.model.encoder_hidden_dims[-1] // 2)],
            self.age_latent_dim,
            config.model.dropout,
        )
        self.process_reconstructors = nn.ModuleList(
            [
                build_mlp(
                    n_features,
                    [max(32, config.model.encoder_hidden_dims[-1] // 2)],
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

    def generator_bases(self, reference_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        context = self.generator_trunk(reference_features)
        age_basis = self.age_basis_head(context)
        process_bases = self.process_basis_head(context).view(reference_features.shape[0], self.n_processes, self.n_features)
        return age_basis, process_bases

    def synthesize_full(
        self,
        reference_features: torch.Tensor,
        age_latent: torch.Tensor,
        process_latents: torch.Tensor,
    ) -> GeneratorOutputs:
        age_basis, process_bases = self.generator_bases(reference_features)
        age_component = age_latent * age_basis
        process_components = process_latents.unsqueeze(-1) * process_bases
        total_delta = age_component + process_components.sum(dim=1)
        fake_target = reference_features + total_delta
        return GeneratorOutputs(
            fake_target=fake_target,
            total_delta=total_delta,
            age_component=age_component,
            process_components=process_components,
            age_basis=age_basis,
            process_bases=process_bases,
        )

    def synthesize(
        self,
        reference_features: torch.Tensor,
        age_latent: torch.Tensor,
        process_latents: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.synthesize_full(reference_features, age_latent, process_latents)
        return outputs.fake_target, outputs.total_delta

    def decompose(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        decomposition = self.decomposer(features).view(features.shape[0], self.n_processes + 1, self.n_features)
        age_component = decomposition[:, 0, :]
        process_components = decomposition[:, 1:, :]
        return age_component, process_components

    def reconstruct_latents(
        self,
        age_component: torch.Tensor,
        process_components: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        age_latent = torch.sigmoid(self.age_reconstructor(age_component))
        process_latents = torch.cat(
            [torch.sigmoid(head(process_components[:, idx, :])) for idx, head in enumerate(self.process_reconstructors)],
            dim=1,
        )
        return age_latent, process_latents

    def infer_latents(self, target_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        age_component, process_components = self.decompose(target_features)
        age_latent, process_latents = self.reconstruct_latents(age_component, process_components)
        return age_latent, process_latents, age_component, process_components

    def discriminate(self, features: torch.Tensor) -> torch.Tensor:
        return self.discriminator(features)

    def predict_age_from_process(self, process_latents: torch.Tensor, reverse: bool = True) -> torch.Tensor:
        inputs = grad_reverse(process_latents) if reverse else process_latents
        return torch.sigmoid(self.age_adversary(inputs))

    @torch.no_grad()
    def infer(self, target_features: torch.Tensor, reference_template: torch.Tensor) -> dict[str, torch.Tensor]:
        age_latent, process_latents, age_component, process_components = self.infer_latents(target_features)
        fake_target, fake_delta = self.synthesize(reference_template, age_latent, process_latents)
        return {
            "age_latent": age_latent,
            "process_latents": process_latents,
            "synthetic_target": fake_target,
            "synthetic_delta": fake_delta,
            "age_delta": age_component,
            "process_deltas": process_components,
        }
