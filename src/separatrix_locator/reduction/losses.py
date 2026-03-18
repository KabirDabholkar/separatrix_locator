from __future__ import annotations

import inspect
from typing import Callable

import torch
from torch import nn
import torch.nn.functional as F


def call_rhs_function(
    rhs_function: Callable,
    psi: torch.Tensor,
    u_t: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Call RHS function with either signature:
      - rhs_function(psi)
      - rhs_function(psi, u_t)
    """
    if u_t is None:
        return rhs_function(psi)

    # Try signature-based dispatch first to avoid catching unrelated TypeErrors.
    try:
        sig = inspect.signature(rhs_function)
        if len(sig.parameters) >= 2:
            return rhs_function(psi, u_t)
        return rhs_function(psi)
    except (TypeError, ValueError):
        # Fallback: try two-arg call then one-arg call.
        try:
            return rhs_function(psi, u_t)
        except TypeError:
            return rhs_function(psi)


def pairwise_stress_loss(
    x_points: torch.Tensor,
    z_points: torch.Tensor,
    normalize: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Distance-preservation loss (stress):
      D_high(i,j) ~ D_low(i,j)

    Args:
      x_points: (N, D)
      z_points: (N, d)
    """
    if x_points.ndim != 2 or z_points.ndim != 2:
        raise ValueError(f"Expected x_points, z_points to be 2D; got {x_points.shape}, {z_points.shape}")
    if x_points.shape[0] != z_points.shape[0]:
        raise ValueError("x_points and z_points must contain the same number of points")

    # Upper-triangular pairs only (avoid diagonal and redundancy).
    n = x_points.shape[0]
    if n < 2:
        # No pairs => no stress signal.
        return torch.zeros((), device=x_points.device, dtype=x_points.dtype)

    dist_x = torch.cdist(x_points, x_points)  # (N, N)
    dist_z = torch.cdist(z_points, z_points)  # (N, N)

    triu_mask = torch.triu(torch.ones(n, n, device=x_points.device, dtype=torch.bool), diagonal=1)
    dx = dist_x[triu_mask]  # (N*(N-1)/2,)
    dz = dist_z[triu_mask]

    if normalize:
        dx_mean = dx.mean().detach().clamp_min(eps)
        dz_mean = dz.mean().detach().clamp_min(eps)
        dx = dx / dx_mean
        dz = dz / dz_mean

    return torch.mean((dz - dx) ** 2)


def discrete_one_step_state_loss(
    encoder: nn.Module,
    latent_dynamics: nn.Module,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    u_t: torch.Tensor,
) -> torch.Tensor:
    """
    One-step consistency between predicted latent state and encoded true next state.
    """
    z_t = encoder(x_t)
    z_true_next = encoder(x_next)
    z_pred_next = latent_dynamics(z_t, u_t)
    return F.mse_loss(z_pred_next, z_true_next)


def discrete_one_step_koopman_loss(
    encoder: nn.Module,
    latent_dynamics: nn.Module,
    eigenfunction: nn.Module,
    rhs_function: Callable,
    x_t: torch.Tensor,
    x_next: torch.Tensor,
    u_t: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    """
    Discrete-time Koopman/eigenfunction consistency:

      psi(z_{t+1}) = kappa * RHS(psi(z_t), u_t)

    We enforce both:
      - psi(Encoded true next state) matches RHS
      - psi(Predicted next latent state) matches RHS
    """
    z_t = encoder(x_t)
    z_true_next = encoder(x_next)
    z_pred_next = latent_dynamics(z_t, u_t)

    psi_t = eigenfunction(z_t)  # (..., 1)
    rhs = call_rhs_function(rhs_function, psi_t, u_t=u_t)
    rhs = kappa * rhs

    psi_true_next = eigenfunction(z_true_next)
    psi_pred_next = eigenfunction(z_pred_next)

    return F.mse_loss(psi_true_next, rhs) + F.mse_loss(psi_pred_next, rhs)


def pca_supervised_geometry_loss(
    encoder: nn.Module,
    x_points: torch.Tensor,
    pca_components: torch.Tensor,
    pca_mean: torch.Tensor,
) -> torch.Tensor:
    """
    Supervised geometry loss:
    - compute PCA coordinates of x_points in the original (high-D) space
    - regress encoder(x_points) to those PCA coordinates

    Args:
      encoder: maps x -> z (shape (..., d_latent))
      x_points: (N, D)
      pca_components: (d_latent, D) (sklearn PCA .components_)
      pca_mean: (D,) (sklearn PCA .mean_)
    """
    if x_points.ndim != 2:
        raise ValueError(f"Expected x_points to be 2D (N,D); got {x_points.shape}")
    if pca_components.ndim != 2:
        raise ValueError("Expected pca_components to be 2D (d_latent,D)")
    if pca_mean.ndim != 1:
        raise ValueError("Expected pca_mean to be 1D (D,)")

    # Target PCA coordinates: (x - mean) @ components^T
    x_centered = x_points - pca_mean.to(device=x_points.device, dtype=x_points.dtype)
    z_target = x_centered @ pca_components.to(device=x_points.device, dtype=x_points.dtype).T

    z_pred = encoder(x_points)
    return F.mse_loss(z_pred, z_target)

