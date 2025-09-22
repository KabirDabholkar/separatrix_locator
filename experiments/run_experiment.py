"""
Modular experiment runner for separatrix locator experiments.

This script provides a flexible framework for running separatrix locator experiments
using configuration files and pre-built models.
"""

import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path
import argparse

import torch
import torch.nn as nn

# Add the src directory to the path so we can import separatrix_locator
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from separatrix_locator.core.separatrix_locator import SeparatrixLocator
from separatrix_locator.core.models import ResNet, KoopmanEigenfunctionModel, LinearModel, DeepKoopmanModel
from separatrix_locator.dynamics import Bistable2D
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.plotting.plots import plot_dynamics_2D

from config import (
    ExperimentConfig, 
    DynamicsConfig, 
    ModelConfig, 
    DistributionConfig, 
    TrainingConfig,
    get_bistable2d_config,
    get_quick_test_config
)


def create_dynamics(config: DynamicsConfig):
    """Create a dynamics object from configuration."""
    if config.system_type == "Bistable2D":
        from separatrix_locator.dynamics import Bistable2D
        return Bistable2D()
    elif config.system_type == "BistableND":
        from separatrix_locator.dynamics import BistableND
        return BistableND(dim=config.dim, bistable_axis=config.bistable_axis)
    else:
        raise ValueError(f"Unknown system type: {config.system_type}")


def create_model(config: ModelConfig, input_dim: int):
    """Create a model from configuration."""
    kwargs = config.get_model_kwargs(input_dim)
    
    if config.model_type == "ResNet":
        return ResNet(**kwargs)
    elif config.model_type == "KoopmanEigenfunctionModel":
        return KoopmanEigenfunctionModel(**kwargs)
    elif config.model_type == "LinearModel":
        return LinearModel(**kwargs)
    elif config.model_type == "DeepKoopmanModel":
        return DeepKoopmanModel(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")


def create_distribution(config: DistributionConfig, dim: int):
    """Create distribution(s) from configuration."""
    kwargs = config.get_distribution_kwargs(dim)
    
    if config.dist_type == "MultivariateGaussian":
        dist = MultivariateGaussian(**kwargs)
        
        if config.use_multiscaler:
            dists = multiscaler(dist, config.scales)
            return dist, dists
        else:
            return dist, [dist]
    else:
        raise ValueError(f"Unknown distribution type: {config.dist_type}")


def create_models(config: TrainingConfig, model_config: ModelConfig, dynamics_dim: int, device: str):
    """Create a list of models for the SeparatrixLocator."""
    models = []
    for i in range(config.num_models):
        model = create_model(model_config, dynamics_dim)
        model.to(device)
        models.append(model)
    return models


def run_experiment(
    config: ExperimentConfig,
    save_dir: Optional[str] = None
) -> tuple:
    """
    Run a complete separatrix locator experiment.
    
    Parameters:
    -----------
    config : ExperimentConfig
        Complete experiment configuration
    save_dir : Optional[str]
        Override save directory from config
        
    Returns:
    --------
    tuple
        (locator, trajectories_and_below_threshold)
    """
    # Use provided save_dir or config save_dir
    if save_dir is None:
        save_dir = config.save_dir
    
    # Create dynamics system
    dynamics = create_dynamics(config.dynamics)
    if config.training.verbose:
        print(f"Created dynamics: {dynamics.name}")
    
    # Create distribution(s)
    dist, dists = create_distribution(config.distribution, dynamics.dim)
    if config.training.verbose:
        print(f"Created distribution: {dist.name}")
        print(f"Using {len(dists)} distribution scales: {config.distribution.scales}")
    
    # Create models
    models = create_models(config.training, config.model, dynamics.dim, config.training.device)
    if config.training.verbose:
        print(f"Created {len(models)} models of type: {config.model.model_type}")
    
    # Build locator with pre-built models
    locator = SeparatrixLocator(
        num_models=config.training.num_models,
        dynamics_dim=dynamics.dim,
        model_class=None,  # We're providing models directly
        models=models,
        lr=config.training.learning_rate,
        epochs=config.training.epochs,
        use_multiprocessing=config.training.use_multiprocessing,
        verbose=config.training.verbose,
        device=config.training.device,
    )
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if config.training.verbose:
        print(f"Save directory: {save_dir}")
    
    # Train models (uncomment to actually train)
    # locator.fit(
    #     dynamics.function, 
    #     dists, 
    #     batch_size=config.training.batch_size, 
    #     eigenvalue=config.training.eigenvalue, 
    #     balance_loss_lambda=config.training.balance_loss_lambda, 
    #     RHS_function=config.training.RHS_function
    # )
    # locator.save_models(save_dir)
    
    # Load pre-trained models
    locator.load_models(save_dir)
    
    # Generate plots
    plot_dynamics_2D(
        dynamics.function, 
        locator.predict, 
        x_limits=config.plot_limits["x_limits"], 
        y_limits=config.plot_limits["y_limits"], 
        show=config.show_plots, 
        save_dir=save_dir
    )
    
    # Prepare and run separatrix search
    locator.prepare_models_for_gradient_descent(dist)
    traj, below = locator.find_separatrix(dist)
    
    if config.training.verbose:
        print("Experiment completed. Collected trajectories and below-threshold points.")
    
    return locator, (traj, below)


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run separatrix locator experiments")
    parser.add_argument(
        "--config", 
        type=str, 
        choices=["bistable2d", "quick_test"], 
        default="bistable2d",
        help="Configuration preset to use"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        help="Override save directory"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        help="Override number of epochs"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        help="Override learning rate"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        help="Override batch size"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"], 
        help="Override device"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Get base configuration
    if args.config == "bistable2d":
        config = get_bistable2d_config()
    elif args.config == "quick_test":
        config = get_quick_test_config()
    else:
        raise ValueError(f"Unknown config: {args.config}")
    
    # Apply command line overrides
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    if args.device is not None:
        config.training.device = args.device
    if args.verbose:
        config.training.verbose = True
    
    # Run experiment
    try:
        locator, results = run_experiment(config, args.save_dir)
        print("Experiment completed successfully!")
        return 0
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
