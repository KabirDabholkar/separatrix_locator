"""
Simple experiment runner for separatrix locator experiments.

This script provides a simplified framework for running separatrix locator experiments
using configuration files that define dynamics, model, and distributions directly.
"""

import sys
import os
from typing import Optional, Callable, Iterable, Union
from pathlib import Path
import argparse

import torch
import torch.nn as nn

# Add the src directory to the path so we can import separatrix_locator
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.separatrix_locator import SeparatrixLocator
from src.plotting.plots import plot_dynamics_2D


def load_config(config_path: str):
    """Load configuration from a Python file."""
    config_path = Path(config_path)
    
    # Check if file exists
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Add the config file's directory to the path
    config_dir = config_path.parent
    sys.path.insert(0, str(config_dir))
    
    # Import the config module using importlib
    import importlib.util
    spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    # Extract the required variables
    if not hasattr(config_module, 'dynamics'):
        raise ValueError(f"Config file {config_path} must define 'dynamics'")
    if not hasattr(config_module, 'dists'):
        raise ValueError(f"Config file {config_path} must define 'dists'")
    
    # Handle both 'model' and 'models' - prefer 'models' if available
    if hasattr(config_module, 'models'):
        model_or_models = config_module.models
    elif hasattr(config_module, 'model'):
        model_or_models = config_module.model
    else:
        raise ValueError(f"Config file {config_path} must define either 'model' or 'models'")
    
    # Get save_dir if defined in config, otherwise None
    save_dir = getattr(config_module, 'save_dir', None)
    
    return config_module.dynamics, model_or_models, config_module.dists, save_dir


def run_simple_experiment(
    dynamics,
    model_or_models: Union[nn.Module, Iterable[nn.Module]],
    dists,
    save_dir: str = "results",
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 1000,
    device: str = "cpu",
    verbose: bool = True,
    train_models: bool = False,
    show_plots: bool = True,
    x_limits: tuple = (-2, 2),
    y_limits: tuple = (-2, 2)
) -> tuple:
    """
    Run a complete separatrix locator experiment.
    
    Parameters:
    -----------
    dynamics : object
        Dynamics system with .function attribute
    model_or_models : nn.Module or list
        Either a single model or list of models to use
    dists : list
        List of distributions for training
    save_dir : str
        Directory to save results
    epochs : int
        Number of training epochs
    learning_rate : float
        Learning rate for training
    batch_size : int
        Batch size for training
    device : str
        Device to use ("cpu" or "cuda")
    verbose : bool
        Whether to print progress
    train_models : bool
        Whether to train models (if False, loads pre-trained)
    show_plots : bool
        Whether to show plots
    x_limits : tuple
        X-axis limits for plotting
    y_limits : tuple
        Y-axis limits for plotting
        
    Returns:
    --------
    tuple
        (locator, trajectories_and_below_threshold)
    """
    
    # Convert single model to list if needed, then infer num_models
    if isinstance(model_or_models, list):
        models = model_or_models
    else:
        models = [model_or_models]
    
    # Move all models to device and get number of models
    for model in models:
        model.to(device)
    num_models = len(models)
    
    if verbose:
        print(f"Running experiment with {num_models} models")
        print(f"Dynamics: {dynamics.name if hasattr(dynamics, 'name') else type(dynamics).__name__}")
        print(f"Model: {models[0].name if hasattr(models[0], 'name') else type(models[0]).__name__}")
        print(f"Distributions: {len(dists)} scales")
        print(f"Device: {device}")
    
    # Build locator with pre-built models
    locator = SeparatrixLocator(
        num_models=num_models,
        dynamics_dim=dynamics.dim if hasattr(dynamics, 'dim') else 2,
        models=models,
        lr=learning_rate,
        epochs=epochs,
        use_multiprocessing=False,  # Simplified for now
        verbose=verbose,
        device=device,
    )
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    if train_models:
        if verbose:
            print("Training models...")
        # Train models
        locator.fit(
            dynamics.function, 
            dists, 
            batch_size=batch_size, 
            eigenvalue=1.0,  # Default eigenvalue
            balance_loss_lambda=0.0,  # Default balance loss
            RHS_function="lambda phi: phi-phi**3"
        )
        locator.save_models(save_dir)
        if verbose:
            print("Models trained and saved")
    else:
        if verbose:
            print("Loading pre-trained models...")
        # Load pre-trained models
        try:
            locator.load_models(save_dir)
            if verbose:
                print("Pre-trained models loaded")
        except FileNotFoundError:
            if verbose:
                print("No pre-trained models found. Training new models...")
            locator.fit(
                dynamics.function, 
                dists, 
                batch_size=batch_size, 
                eigenvalue=1.0,
                balance_loss_lambda=0.0,
                RHS_function="lambda phi: phi-phi**3"
            )
            locator.save_models(save_dir)
            if verbose:
                print("Models trained and saved")
    
    # Generate plots
    if verbose:
        print("Generating plots...")
    
    # Create a simple prediction function that averages across models
    def predict_function(x):
        with torch.no_grad():
            predictions = []
            for model in locator.models:
                pred = model(x.to(device))
                predictions.append(pred.cpu())
            return torch.stack(predictions, dim=-1).mean(dim=-1)
    
    plot_dynamics_2D(
        dynamics.function, 
        predict_function, 
        x_limits=x_limits, 
        y_limits=y_limits, 
        show=show_plots, 
        save_dir=save_dir
    )
    
    if verbose:
        print("Plots generated")
    
    # Prepare and run separatrix search
    if verbose:
        print("Preparing models for separatrix search...")
    
    # Use the last distribution for separatrix search
    base_dist = dists[-1] if isinstance(dists, Iterable) else dists
    locator.prepare_models_for_gradient_descent(base_dist)
    
    if verbose:
        print("Finding separatrix...")
    
    traj, below = locator.find_separatrix(base_dist)
    
    if verbose:
        print("Experiment completed successfully!")
        print(f"Found {len(traj)} trajectories and {len(below)} below-threshold points")
    
    return locator, (traj, below)


def main():
    """Main entry point for running experiments."""
    parser = argparse.ArgumentParser(description="Run simple separatrix locator experiments")
    parser.add_argument(
        "config", 
        type=str, 
        help="Path to configuration file (e.g., configs/bistable2d.py)"
    )
    parser.add_argument(
        "--save-dir", 
        type=str, 
        default="results",
        help="Directory to save results (default: results)"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=1000,
        help="Number of training epochs (default: 1000)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=1e-3,
        help="Learning rate (default: 1e-3)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=128,
        help="Batch size (default: 128)"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["cpu", "cuda"], 
        default="cpu",
        help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "--train", 
        action="store_true", 
        help="Force training of new models (default: load pre-trained if available)"
    )
    parser.add_argument(
        "--no-plots", 
        action="store_true", 
        help="Don't show plots"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true", 
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        dynamics, model_or_models, dists, config_save_dir = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        return 1
    
    # Use config save_dir if provided, otherwise use command line argument
    save_dir = config_save_dir if config_save_dir is not None else args.save_dir
    
    # Run experiment
    try:
        locator, results = run_simple_experiment(
            dynamics=dynamics,
            model_or_models=model_or_models,
            dists=dists,
            save_dir=save_dir,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            device=args.device,
            verbose=args.verbose,
            train_models=args.train,
            show_plots=not args.no_plots,
            x_limits=(-2, 2),  # Default limits
            y_limits=(-2, 2)   # Default limits
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {save_dir}")
        return 0
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
