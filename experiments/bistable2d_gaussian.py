"""
Experiment: Bistable2D with Multivariate Gaussian initial conditions.

This script constructs a 2D bistable dynamical system, a multivariate Gaussian
distribution for initial conditions, and runs the SeparatrixLocator pipeline.

Run as a module (recommended):
    python -m separatrix_locator.experiments.bistable2d_gaussian
"""

from typing import Optional, Dict, Any

import torch

from separatrix_locator.core.separatrix_locator import SeparatrixLocator
from separatrix_locator.core.models import ResNet
from separatrix_locator.dynamics import Bistable2D
from separatrix_locator.distributions import MultivariateGaussian


def run_bistable2d_experiment(
    num_models: int = 1,
    epochs: int = 600,
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

    # Train
    locator.init_models()
    locator.fit(dynamics.function, [dist], batch_size=batch_size, eigenvalue=1.0, balance_loss_lambda=1e-1, RHS_function='lambda psi: psi-psi**3')

    # Prepare and run separatrix search
    locator.prepare_models_for_gradient_descent(dist)
    traj, below = locator.find_separatrix(dist)

    if verbose:
        print("Experiment completed. Collected trajectories and below-threshold points.")

    return locator, (traj, below)


if __name__ == "__main__":
    run_bistable2d_experiment()


