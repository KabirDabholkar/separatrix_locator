# Separatrix Locator

A Python package for locating separatrices in black-box dynamical systems using Koopman eigenfunctions.

## Overview

The Separatrix Locator is a tool that learns to identify separatrices (boundaries between different basins of attraction) in dynamical systems without requiring analytical knowledge of the system. It uses neural networks to learn Koopman eigenfunctions and gradient descent to find separatrix points.

## Key Features

- **Easy-to-use API**: Simple interface for setting up and running separatrix location experiments
- **Multiple dynamical systems**: Pre-built systems including bistable systems, Duffing oscillator, flip-flop circuits, and RNN decision-making systems
- **Flexible model architectures**: Support for various neural network architectures including RBF networks and deep networks
- **Comprehensive tutorials**: Step-by-step notebooks demonstrating usage with different systems
- **Research integration**: Compatible with existing research workflows and hydra configurations

## Installation

### From source (recommended for development)

```bash
git clone <repository-url>
cd separatrix_locator
pip install -e .
```

### Dependencies

The package requires:
- Python >= 3.8
- PyTorch >= 1.9.0
- NumPy, Matplotlib, Scikit-learn
- torchdiffeq (for ODE integration)

## Quick Start

```python
from separatrix_locator import SeparatrixLocator
from separatrix_locator.dynamics import Bistable1D
from separatrix_locator.core import KoopmanEigenfunctionModel

# Create a simple bistable system
dynamics = Bistable1D()

# Set up the separatrix locator
locator = SeparatrixLocator(
    num_models=5,
    dynamics_dim=dynamics.dim,
    model_class=KopmanEigenfunctionModel,
    epochs=500
)

# Train the models
locator.fit(dynamics.function, dynamics.distribution)

# Find the separatrix
locator.prepare_models_for_gradient_descent(dynamics.distribution)
trajectories, separatrix_points = locator.find_separatrix(dynamics.distribution)

print(f"Found separatrix points: {separatrix_points}")
```

## Tutorials

The package includes comprehensive tutorials in the `examples/` directory:

1. **01_bistable_1d.ipynb**: Introduction to separatrix location with a simple 1D system
2. **02_duffing_oscillator.ipynb**: Working with 2D systems and phase portraits
3. **03_flipflop_systems.ipynb**: Discrete-state systems and decision-making circuits
4. **04_rnn_decision_making.ipynb**: Neural circuit modeling and complex dynamics

## Supported Dynamical Systems

### Built-in Systems

- **Bistable Systems**: 1D, 2D, and 3D systems with two stable fixed points
- **Duffing Oscillator**: Classic nonlinear oscillator with rich dynamics
- **Flip-flop Circuits**: Binary decision-making systems
- **RNN Systems**: Recurrent neural networks for decision-making tasks

### Custom Systems

You can easily define your own dynamical systems by inheriting from the `DynamicalSystem` base class:

```python
from separatrix_locator.dynamics import DynamicalSystem

class MySystem(DynamicalSystem):
    def __init__(self):
        super().__init__(dim=2, name="MySystem")
        # Set up your system parameters
        
    def function(self, x):
        # Define dx/dt = f(x)
        return your_dynamics_function(x)
    
    @property
    def distribution(self):
        # Return initial condition distribution
        return your_distribution
    
    def get_attractors(self):
        # Return fixed points/attractors
        return your_attractors
```

## Model Architectures

The package supports several neural network architectures:

- **KoopmanEigenfunctionModel**: Basic feedforward network
- **ResRBF**: Residual RBF network with radial basis functions
- **LinearModel**: Simple linear model for baseline comparisons
- **DeepKoopmanModel**: Deep network for complex systems

## Research Integration

The package is designed to work alongside existing research workflows:

- **Hydra compatibility**: Can be used with existing hydra configurations
- **Experiment tracking**: Supports saving/loading trained models
- **Custom training**: Flexible training procedures for different systems

## Contributing

We welcome contributions! Please see the contributing guidelines for details on:

- Setting up a development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## Citation

If you use this package in your research, please cite:

```bibtex
@software{separatrix_locator,
  title={Separatrix Locator: A tool for locating separatrices in dynamical systems},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/separatrix_locator}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on PyTorch and torchdiffeq
- Inspired by Koopman operator theory and dynamical systems research
- Developed for research in computational neuroscience and dynamical systems
