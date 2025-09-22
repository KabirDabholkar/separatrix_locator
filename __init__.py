"""
Separatrix Locator

A tool for locating separatrices in black-box dynamical systems using Koopman eigenfunctions.

This package provides:
- Core separatrix location algorithms
- Pre-defined dynamical systems
- Training and validation utilities
- Example notebooks and tutorials

Example usage:
    from separatrix_locator import SeparatrixLocator
    from separatrix_locator.dynamics import Bistable1D
    from separatrix_locator.core import ResRBF
    
    # Create a simple bistable system
    dynamics = Bistable1D()
    
    # Set up the model and locator
    model = ResRBF(input_dim=1, hidden_size=100)
    locator = SeparatrixLocator(model_class=model, num_models=5)
    
    # Train and find separatrix
    from separatrix_locator.distributions import MultivariateGaussian
    dist = MultivariateGaussian(dim=dynamics.dim)
    locator.fit(dynamics.function, dist)
    separatrix_points = locator.find_separatrix(dist)
"""

__version__ = "0.1.0"
__author__ = "Your Name"

# Core classes
from .core.separatrix_locator import SeparatrixLocator
from .core.models import KoopmanEigenfunctionModel, ResNet, ResRBF, LinearModel

# Dynamics systems
from .dynamics import Bistable1D, Bistable2D, Bistable3D, BistableND
from .dynamics.duffing import DuffingOscillator
from .dynamics.flipflop import FlipFlop1Bit2D, FlipFlop2Bit2D
from .dynamics.rnn_systems import RNNDecisionMaking

# Utilities
from .utils.plotting import plot_trajectories, plot_separatrix
from .distributions import BaseDistribution, MultivariateGaussian

# Experiments (optional)
try:
    from .experiments import run_bistable2d_experiment  # type: ignore
    _has_experiments = True
except Exception:
    run_bistable2d_experiment = None  # type: ignore
    _has_experiments = False

__all__ = [
    # Core
    "SeparatrixLocator",
    "KoopmanEigenfunctionModel", 
    "ResNet",
    "ResRBF",
    "LinearModel",
    
    # Dynamics
    "BistableND",
    "Bistable1D",
    "Bistable2D", 
    "Bistable3D",
    "DuffingOscillator",
    "FlipFlop1Bit2D",
    "FlipFlop2Bit2D", 
    "RNNDecisionMaking",
    
    # Utils
    "plot_trajectories",
    "plot_separatrix",
    "BaseDistribution",
    "MultivariateGaussian",
    
    # Experiments
    # populated below if available
]

if _has_experiments:
    __all__.append("run_bistable2d_experiment")
