"""
Separatrix Locator

A tool for locating separatrices in black-box dynamical systems using Koopman eigenfunctions.

This package provides:
- Core separatrix location algorithms
- Pre-defined dynamical systems
- Training and validation utilities
- Example notebooks and tutorials

Example usage:
    from .core.separatrix_locator import SeparatrixLocator
    from .dynamics.bistableND import Bistable1D
    from .core.models import ResRBF
    
    # Create a simple bistable system
    dynamics = Bistable1D()
    
    # Set up the model and locator
    model = ResRBF(input_dim=1, hidden_size=100)
    locator = SeparatrixLocator(model_class=model, num_models=5)
    
    # Train and find separatrix
    from .distributions.gaussian import MultivariateGaussian
    dist = MultivariateGaussian(dim=dynamics.dim)
    locator.fit(dynamics.function, dist)
    separatrix_points = locator.find_separatrix(dist)
"""

__version__ = "0.1.0"
__author__ = "Kabir Dabholkar"

# Core classes
from .core.separatrix_locator import SeparatrixLocator
from .core.models import ResNet

# Dynamics systems
from .dynamics import Bistable1D, Bistable2D, Bistable3D, BistableND
from .dynamics.duffing import DuffingOscillator
from .dynamics.flipflop import FlipFlop1Bit2D, FlipFlop2Bit2D
from .dynamics.rnn_systems import RNNDecisionMaking

# Utilities
from .utils.plotting import plot_trajectories, plot_separatrix
from .distributions import BaseDistribution, MultivariateGaussian


__all__ = [
    # Core
    "SeparatrixLocator",
    "ResNet",
    
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
]

