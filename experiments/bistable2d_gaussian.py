"""
Experiment: Bistable2D with Multivariate Gaussian initial conditions.

This script constructs a 2D bistable dynamical system, a multivariate Gaussian
distribution for initial conditions, and runs the SeparatrixLocator pipeline.

Run as a module (recommended):
    python -m separatrix_locator.experiments.bistable2d_gaussian

Or from within the separatrix_locator directory:
    python -m experiments.bistable2d_gaussian
"""

import sys
import os
from typing import Optional, Dict, Any
from pathlib import Path

import torch

from separatrix_locator.core.separatrix_locator import SeparatrixLocator
from separatrix_locator.core.models import ResNet
from separatrix_locator.dynamics import Bistable2D
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.plotting.plots import plot_dynamics_2D


def run_bistable2d_experiment(
    num_models: int = 1,
    epochs: int = 1000, #1000, #00,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    gaussian_scale: float = 2.0,
    device: str = "cpu",
    verbose: bool = True,
    extra_train_kwargs: Optional[Dict[str, Any]] = None,
):
    """
    Run a quick experiment on the 2D bistable system using a multivariate Gaussian.

    Returns the trained locator and the separatrix descent results.
    """
    # Set up system and distribution
    dynamics = Bistable2D()
    dim = dynamics.dim

    mean = torch.zeros(dim)
    cov = gaussian_scale * torch.eye(dim)
    dist = MultivariateGaussian(dim=dim, mean=mean, covariance_matrix=cov)
    dists = multiscaler(dist, [0.1,0.5,1.0])

    # Build locator
    locator = SeparatrixLocator(
        num_models=num_models,
        dynamics_dim=dim,
        model_class=ResNet,
        lr=learning_rate,
        epochs=epochs,
        use_multiprocessing=False,
        verbose=verbose,
        device=device,
    )
    save_dir = Path(f"results/{dynamics.name}_{dist.name}/{epochs}_epochs_{learning_rate}_learning_rate_{batch_size}_batch_size_{gaussian_scale}_gaussian_scale/")

    # Train
    locator.init_models()

    # locator.fit(dynamics.function, dists, batch_size=batch_size, eigenvalue=1.0, balance_loss_lambda=1e-1, RHS_function='lambda psi: psi-psi**3')
    # locator.save_models(save_dir)

    locator.load_models(save_dir)

    plot_dynamics_2D(dynamics.function, locator.predict, x_limits=(-2, 2), y_limits=(-2, 2), show=False, save_dir=save_dir)

    # Prepare and run separatrix search
    locator.prepare_models_for_gradient_descent(dist)
    traj, below = locator.find_separatrix(dist)


    if verbose:
        print("Experiment completed. Collected trajectories and below-threshold points.")

    return locator, (traj, below)


if __name__ == "__main__":
    run_bistable2d_experiment()


