from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ..core.separatrix_locator import SeparatrixLocator


def remove_frame(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def evaluate_on_grid(func: Callable[[torch.Tensor], torch.Tensor],
                     x_limits: Tuple[float, float],
                     y_limits: Tuple[float, float],
                     resolution: int = 200):
    x = torch.linspace(x_limits[0], x_limits[1], resolution)
    y = torch.linspace(y_limits[0], y_limits[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        Z = func(XY).reshape(resolution, resolution)
    return X.numpy(), Y.numpy(), Z.detach().cpu().numpy()


def dynamics_to_kinetic_energy(dynamics_function: Callable[[torch.Tensor], torch.Tensor]):
    def kinetic_energy(points: torch.Tensor):
        dx = dynamics_function(points)
        if dx.dim() == 1:
            dx = dx.unsqueeze(-1)
        return torch.sum(dx ** 2, dim=-1)
    return kinetic_energy


def plot_flow_streamlines(dynamics_function: Callable[[torch.Tensor], torch.Tensor],
                          axes,
                          x_limits: Tuple[float, float],
                          y_limits: Tuple[float, float],
                          resolution: int = 200,
                          density: float = 0.7,
                          color: str = 'k',
                          linewidth: float = 0.5,
                          alpha: float = 0.5):
    if not isinstance(axes, (list, tuple, np.ndarray)):
        axes = [axes]

    x = torch.linspace(x_limits[0], x_limits[1], resolution)
    y = torch.linspace(y_limits[0], y_limits[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='xy')
    XY = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
    with torch.no_grad():
        dXY = dynamics_function(XY)
    if dXY.dim() == 1:
        dXY = dXY.unsqueeze(-1)
    U = dXY[:, 0].reshape(resolution, resolution).detach().cpu().numpy()
    V = dXY[:, 1].reshape(resolution, resolution).detach().cpu().numpy()
    Xn = X.detach().cpu().numpy()
    Yn = Y.detach().cpu().numpy()

    for ax in axes:
        ax.streamplot(Xn, Yn, U, V, density=density, color=color, linewidth=linewidth, arrowsize=0.8, alpha=alpha)


def plot_dynamics_1D(dynamics_function: Callable[[torch.Tensor], torch.Tensor],
                     x_limits: Tuple[float, float],
                     save_dir: Optional[str] = None,
                     show: bool = True):
    x = torch.linspace(x_limits[0], x_limits[1], 200).reshape(-1, 1)
    with torch.no_grad():
        dx = dynamics_function(x)

    x_np = x.detach().cpu().numpy()
    dx_np = dx.detach().cpu().numpy()

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(7, 2.2))

    # Dynamics
    ax1.plot(x_np, dx_np, 'b-', linewidth=2)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title(r'$dx/dt$')
    ax1.spines['left'].set_bounds(-1, 1)
    ax1.set_yticks([-1, 0, 1])

    # Kinetic energy q(x) = |f(x)|^2
    kinetic_energy = (dx ** 2).detach().cpu().numpy()
    ax2.plot(x_np, kinetic_energy, 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2.set_title(r'$q(x)$')
    ax2.set_ylim(-0.1, 1.1)
    ax2.spines['left'].set_bounds(0, 1)
    ax2.set_yticks([0, 1])

    # Phase portrait surrogate (sign of dx)
    ax3.plot(x_np, np.sign(dx_np), 'g-', linewidth=2)
    ax3.set_title('sign(dx)')
    ax3.set_ylim(-1.1, 1.1)

    for ax in (ax1, ax2, ax3):
        remove_frame(ax)

    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / 'results1D.png', dpi=200)
        fig.savefig(Path(save_dir) / 'results1D.pdf')
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_dynamics_2D(dynamics_function: Callable[[torch.Tensor], torch.Tensor],
                     locator: SeparatrixLocator,
                     distribution,
                     x_limits: Tuple[float, float],
                     y_limits: Tuple[float, float],
                     save_dir: Optional[str] = None,
                     plot_trajectories: bool = False,
                     num_trajectories: int = 20,
                     resolution: int = 200,
                     show: bool = True):
    # Prepare KEF function from the first normalized model function
    kef_funcs = locator.prepare_models_for_gradient_descent(distribution)
    KEF_func = kef_funcs[0]

    fig, axs = plt.subplots(1, 2, figsize=(5, 3))

    # Left: log10 kinetic energy heatmap
    ax = axs[0]
    kinetic_energy_function = dynamics_to_kinetic_energy(dynamics_function)
    X, Y, KE = evaluate_on_grid(kinetic_energy_function, x_limits, y_limits, resolution)
    im = ax.contourf(X, Y, np.log10(KE + 1e-12), levels=np.linspace(-4, 2, 14), cmap='Blues_r')
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, location='bottom', label=r'$\log_{10}(q(x))$', ticks=np.linspace(-4, 2, 3))
    cbar.ax.set_xlabel(cbar.ax.get_xlabel(), size=12)
    cbar.outline.set_edgecolor('grey')

    # Right: |psi| with zero-contour and streamlines
    ax = axs[1]
    X, Y, KEF_vals = evaluate_on_grid(lambda p: locator.predict(p)[..., 0], x_limits, y_limits, resolution)
    im = ax.contourf(X, Y, np.abs(KEF_vals), levels=np.arange(0, 1.1, 0.1), cmap='Blues_r')
    CS = ax.contour(X, Y, KEF_vals, levels=[0], colors='lightgreen')
    ax.clabel(CS, fontsize=10)
    plot_flow_streamlines(dynamics_function, axs.flatten(), x_limits, y_limits, resolution=resolution, density=0.7, color='red', linewidth=0.5, alpha=0.4)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, location='bottom', ticks=[0, 0.5, 1], label=r'$|\psi(x)|$')
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_xlabel(cbar.ax.get_xlabel(), size=12)
    cbar.outline.set_edgecolor('grey')

    # Optional: integrate and plot sample trajectories
    if plot_trajectories:
        try:
            from torchdiffeq import odeint
            t_span = torch.linspace(0, 10, 100)
            initial_conditions = torch.FloatTensor(num_trajectories, 2)
            initial_conditions[:, 0] = torch.linspace(x_limits[0], x_limits[1], num_trajectories)
            initial_conditions[:, 1] = torch.linspace(y_limits[0], y_limits[1], num_trajectories)
            trajectories = odeint(lambda t, x: dynamics_function(x), initial_conditions, t_span)
            trajectories_np = trajectories.detach().cpu().numpy()
            for ax in axs:
                for traj in trajectories_np.transpose(1, 0, 2):
                    ax.plot(traj[:, 0], traj[:, 1], 'k-', alpha=0.3, linewidth=0.5)
        except Exception:
            pass

    for ax in axs.flatten():
        ax.set_xlim(*x_limits)
        ax.set_ylim(*y_limits)
        remove_frame(ax)
        ax.set_aspect('equal')

    fig.tight_layout()
    if save_dir is not None:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(save_dir) / 'results2D.png', dpi=300)
        fig.savefig(Path(save_dir) / 'results2D.pdf')
    if show:
        plt.show()
    else:
        plt.close(fig)


