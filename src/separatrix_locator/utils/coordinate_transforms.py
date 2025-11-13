"""
Coordinate transform helpers.
"""

from __future__ import annotations

import torch

from typing import Callable


def radial_to_cartesian(
    radial_dynamics: Callable[[torch.Tensor], torch.Tensor]
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Wrap a dynamics function expressed in polar coordinates and expose it in Cartesian space.

    Parameters
    ----------
    radial_dynamics : callable
        Function that accepts a tensor whose last dimension is 2 with entries (r, theta)
        and returns the time derivatives (dr/dt, dtheta/dt).

    Returns
    -------
    callable
        Function that accepts Cartesian coordinates (x, y) and returns the corresponding
        derivatives (dx/dt, dy/dt).
    """

    def cartesian_dynamics(z: torch.Tensor) -> torch.Tensor:
        x, y = z[..., 0], z[..., 1]
        r = torch.sqrt(x ** 2 + y ** 2)
        theta = torch.atan2(y, x)

        dr_dt, dtheta_dt = radial_dynamics(
            torch.stack([r, theta], dim=-1)
        ).unbind(dim=-1)

        dx_dt = dr_dt * torch.cos(theta) - r * dtheta_dt * torch.sin(theta)
        dy_dt = dr_dt * torch.sin(theta) + r * dtheta_dt * torch.cos(theta)
        return torch.stack([dx_dt, dy_dt], dim=-1)

    return cartesian_dynamics


__all__ = ["radial_to_cartesian"]

