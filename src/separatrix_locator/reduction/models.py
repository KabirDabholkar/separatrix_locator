from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


class MLP(nn.Module):
    """Small configurable MLP used for encoder/dynamics/eigenfunction."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        activation: Optional[nn.Module] = None,
    ):
        super().__init__()
        if activation is None:
            activation = nn.Tanh()

        layers: list[nn.Module] = []
        dims = [input_dim] + [hidden_dim] * max(0, num_layers - 1) + [output_dim]
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(activation)
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderNN(nn.Module):
    """Learned encoder: x (D) -> z (d_latent)."""

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.mlp = MLP(input_dim, latent_dim, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., input_dim) -> z: (..., latent_dim)
        return self.mlp(x)


class LatentDynamicsNN(nn.Module):
    """Discrete latent dynamics: z_{t+1} = g(z_t, u_t)."""

    def __init__(
        self,
        latent_dim: int,
        u_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.u_dim = u_dim
        self.mlp = MLP(
            input_dim=latent_dim + u_dim,
            output_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
        )

    def forward(self, z_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        # z_t: (..., latent_dim), u_t: (..., u_dim) -> z_next: (..., latent_dim)
        if u_t is None:
            u_t = torch.zeros(*z_t.shape[:-1], self.u_dim, device=z_t.device, dtype=z_t.dtype)
        return self.mlp(torch.cat([z_t, u_t], dim=-1))


class LatentEigenfunctionNN(nn.Module):
    """Learned eigenfunction: psi(z) (scalar)."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.mlp = MLP(input_dim=latent_dim, output_dim=1, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: (..., latent_dim) -> psi: (..., 1)
        return self.mlp(z)


@dataclass
class ReductionModels:
    encoder: EncoderNN
    latent_dynamics: LatentDynamicsNN
    eigenfunction: LatentEigenfunctionNN

