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
        
from functools import partial
import torch.nn as nn

# Import from the separatrix_locator package
from separatrix_locator.core.separatrix_locator import SeparatrixLocator
from separatrix_locator.plotting.plots import plot_dynamics_2D, plot_model_output_histograms


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

    # Optional default hyperparameters from config (which can import configs.base)
    defaults = {
        'epochs': getattr(config_module, 'epochs', None),
        'batch_size': getattr(config_module, 'batch_size', None),
        'learning_rate': getattr(config_module, 'learning_rate', None),
        'optimizer': getattr(config_module, 'optimizer', None),
        'RHS_function': getattr(config_module, 'RHS_function', None),
        'balance_loss_lambda': getattr(config_module, 'balance_loss_lambda', None),
        'x_limits': getattr(config_module, 'x_limits', None),
        'y_limits': getattr(config_module, 'y_limits', None),
    }
    
    return config_module.dynamics, model_or_models, config_module.dists, save_dir, defaults


def run_simple_experiment(
    dynamics,
    model_or_models: Union[nn.Module, Iterable[nn.Module]],
    dists,
    save_dir: str = "results",
    epochs: int = 100,
    learning_rate: float = 1e-3,
    batch_size: int = 1000,
    optimizer: Optional[Callable] = None,
    rhs_function: str = "lambda phi: phi-phi**3",
    balance_loss_lambda: float = 0.0,
    device: str = "cpu",
    verbose: bool = True,
    train_models: bool = False,
    show_plots: bool = True,
    x_limits: tuple = (-2, 2),
    y_limits: tuple = (-2, 2),
    plot_histograms_before_training: bool = False,
    plot_histograms_after_training: bool = True,
    strict_no_fit: bool = False
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
    plot_histograms_before_training : bool
        Whether to plot model output histograms before training
    plot_histograms_after_training : bool
        Whether to plot model output histograms after training
        
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
    
    # Plot histograms before training if requested
    if plot_histograms_before_training:
        if verbose:
            print("Plotting model output histograms before training...")
        plot_model_output_histograms(
            models=locator.models,
            distributions=dists,
            device=device,
            save_dir=save_dir,
            show=show_plots,
            title_prefix="Model Output Histograms (Before Training)"
        )
        if verbose:
            print("Pre-training histograms generated")
    
    if strict_no_fit:
        if verbose:
            print("No-fit mode: skipping training and not calling .fit()")
            print("Attempting to load pre-trained models (if available)...")
        try:
            locator.load_models(save_dir)
            if verbose:
                print("Pre-trained models loaded")
        except FileNotFoundError:
            if verbose:
                print("No pre-trained models found. Proceeding without training.")
    elif train_models:
        if verbose:
            print("Training models...")
        # Train models
        locator.fit(
            dynamics.function, 
            dists, 
            batch_size=batch_size, 
            eigenvalue=1.0,  # Default eigenvalue
            balance_loss_lambda=balance_loss_lambda,
            RHS_function=rhs_function,
            **({ 'optimizer': optimizer } if optimizer is not None else {})
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
                balance_loss_lambda=balance_loss_lambda,
                RHS_function=rhs_function,
                **({ 'optimizer': optimizer } if optimizer is not None else {})
            )
            locator.save_models(save_dir)
            if verbose:
                print("Models trained and saved")
    
    # Plot histograms after training if requested
    if plot_histograms_after_training:
        if verbose:
            print("Plotting model output histograms after training...")
        plot_model_output_histograms(
            models=locator.models,
            distributions=dists,
            device=device,
            save_dir=save_dir,
            show=show_plots,
            title_prefix="Model Output Histograms (After Training)"
        )
        if verbose:
            print("Post-training histograms generated")
    
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
        default=None,
        help="Number of training epochs (default: from config or 1000)"
    )
    parser.add_argument(
        "--learning-rate", 
        type=float, 
        default=None,
        help="Learning rate (default: from config or 1e-3)"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        choices=["adam", "sgd", "adamw"],
        default=None,
        help="Optimizer to use (default: from config; options: adam, sgd, adamw)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Batch size (default: from config or 128)"
    )
    parser.add_argument(
        "--rhs-function",
        type=str,
        default=None,
        help="Right-hand side function of phi as a Python lambda string (default from config)"
    )
    parser.add_argument(
        "--balance-loss-lambda",
        type=float,
        default=None,
        help="Coefficient for balance loss regularization (default from config)"
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
    parser.add_argument(
        "--plot-histograms-before", 
        action="store_true", 
        help="Plot model output histograms before training"
    )
    parser.add_argument(
        "--no-histograms-after", 
        action="store_true", 
        help="Don't plot model output histograms after training (default: plot after training)"
    )
    parser.add_argument(
        "--no-fit",
        action="store_true",
        help="Skip training entirely; never call .fit() even if no pre-trained models exist"
    )
    parser.add_argument(
        "--x-limits", 
        nargs=2, 
        type=float, 
        default=None,
        help="X-axis limits for plotting (default: from config or (-2, 2))"
    )
    parser.add_argument(
        "--y-limits", 
        nargs=2, 
        type=float, 
        default=None,
        help="Y-axis limits for plotting (default: from config or (-2, 2))"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        dynamics, model_or_models, dists, config_save_dir, defaults = load_config(args.config)
    except Exception as e:
        print(f"Error loading config file {args.config}: {e}")
        return 1
    
    # Use config save_dir if provided, otherwise use command line argument
    save_dir = config_save_dir if config_save_dir is not None else args.save_dir
    
    # Determine hyperparameters with precedence: CLI > config defaults > built-in defaults
    resolved_epochs = args.epochs if args.epochs is not None else (defaults.get('epochs') if defaults.get('epochs') is not None else 1000)
    resolved_batch_size = args.batch_size if args.batch_size is not None else (defaults.get('batch_size') if defaults.get('batch_size') is not None else 128)
    resolved_lr = args.learning_rate if args.learning_rate is not None else (defaults.get('learning_rate') if defaults.get('learning_rate') is not None else 1e-3)
    resolved_rhs = args.rhs_function if args.rhs_function is not None else (defaults.get('RHS_function') if defaults.get('RHS_function') is not None else "lambda phi: phi-phi**3")
    resolved_balance_lambda = args.balance_loss_lambda if args.balance_loss_lambda is not None else (defaults.get('balance_loss_lambda') if defaults.get('balance_loss_lambda') is not None else 0.0)
    # Resolve plot limits: CLI > config defaults > built-in defaults
    resolved_x_limits = tuple(args.x_limits) if args.x_limits is not None else (defaults.get('x_limits') if defaults.get('x_limits') is not None else (-2, 2))
    resolved_y_limits = tuple(args.y_limits) if args.y_limits is not None else (defaults.get('y_limits') if defaults.get('y_limits') is not None else (-2, 2))

    # Resolve optimizer
    optimizer = None
    if args.optimizer is not None:
        if args.optimizer == "adam":
            optimizer = partial(torch.optim.Adam, lr=resolved_lr)
        elif args.optimizer == "sgd":
            optimizer = partial(torch.optim.SGD, lr=resolved_lr)
        elif args.optimizer == "adamw":
            optimizer = partial(torch.optim.AdamW, lr=resolved_lr)
    else:
        optimizer = defaults.get('optimizer')


    locator, results = run_simple_experiment(
        dynamics=dynamics,
        model_or_models=model_or_models,
        dists=dists,
        save_dir=save_dir,
        epochs=resolved_epochs,
        learning_rate=resolved_lr,
        batch_size=resolved_batch_size,
        optimizer=optimizer,
        rhs_function=resolved_rhs,
        balance_loss_lambda=resolved_balance_lambda,
        device=args.device,
        verbose=args.verbose,
        train_models=args.train,
        show_plots=not args.no_plots,
        x_limits=resolved_x_limits,
        y_limits=resolved_y_limits,
        plot_histograms_before_training=args.plot_histograms_before,
        plot_histograms_after_training=not args.no_histograms_after,
        strict_no_fit=args.no_fit
    )
    
    print(f"\nExperiment completed successfully!")
    print(f"Results saved to: {save_dir}")
    return 0



if __name__ == "__main__":
    exit(main())
