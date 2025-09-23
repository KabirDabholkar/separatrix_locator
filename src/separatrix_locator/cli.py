"""
Command-line interface for the separatrix_locator package.
"""

import argparse
import sys
from pathlib import Path

from .core import SeparatrixLocator
from .dynamics import Bistable1D, DuffingOscillator
from .distributions import get_default_distribution
from .core import KoopmanEigenfunctionModel


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Separatrix Locator - Find separatrices in dynamical systems"
    )
    
    parser.add_argument(
        "--system",
        choices=["bistable1d", "duffing"],
        default="bistable1d",
        help="Dynamical system to analyze"
    )
    
    parser.add_argument(
        "--num-models",
        type=int,
        default=5,
        help="Number of models in the ensemble"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Output directory for results"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cpu, cuda, or auto)"
    )
    
    args = parser.parse_args()
    
    # Set up device
    if args.device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    print(f"Analyzing system: {args.system}")
    
    # Create the dynamical system
    if args.system == "bistable1d":
        dynamics = Bistable1D()
    elif args.system == "duffing":
        dynamics = DuffingOscillator()
    else:
        raise ValueError(f"Unknown system: {args.system}")
    
    print(f"System: {dynamics}")
    print(f"Attractors: {dynamics.get_attractors()}")
    print(f"Separatrix: {dynamics.get_separatrix()}")
    
    # Create the separatrix locator
    locator = SeparatrixLocator(
        num_models=args.num_models,
        dynamics_dim=dynamics.dim,
        model_class=KoopmanEigenfunctionModel,
        epochs=args.epochs,
        use_multiprocessing=False,
        verbose=True,
        device=device
    )
    
    print(f"\nTraining {args.num_models} models for {args.epochs} epochs...")
    
    # Train the models
    dist = get_default_distribution(dynamics)
    locator.fit(
        func=dynamics.function,
        distribution=dist,
        batch_size=1000
    )
    
    print("Training completed!")
    
    # Prepare for gradient descent
    print("Preparing models for gradient descent...")
    locator.prepare_models_for_gradient_descent(
        distribution=dist,
        dist_needs_dim=False
    )
    
    # Find the separatrix
    print("Finding separatrix...")
    trajectories, separatrix_points = locator.find_separatrix(
        distribution=dist,
        dist_needs_dim=False
    )
    
    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print(f"Saving results to {output_dir}")
    locator.save_models(str(output_dir))
    
    # Print summary
    if separatrix_points and len(separatrix_points) > 0:
        all_points = torch.cat(separatrix_points, dim=0)
        print(f"\nResults:")
        print(f"Found {len(separatrix_points)} gradient descent trajectories")
        print(f"Total separatrix points: {all_points.shape[0]}")
        
        if dynamics.dim == 1:
            mean_x = all_points[:, 0].mean().item()
            std_x = all_points[:, 0].std().item()
            print(f"Mean x: {mean_x:.4f}")
            print(f"Std x: {std_x:.4f}")
        elif dynamics.dim == 2:
            mean_x = all_points[:, 0].mean().item()
            mean_y = all_points[:, 1].mean().item()
            std_x = all_points[:, 0].std().item()
            std_y = all_points[:, 1].std().item()
            print(f"Mean (x, y): ({mean_x:.4f}, {mean_y:.4f})")
            print(f"Std (x, y): ({std_x:.4f}, {std_y:.4f})")
    
    print("Analysis completed!")


if __name__ == "__main__":
    main()
