# Separatrix Locator

A Python package for locating separatrices in black-box dynamical systems using Koopman eigenfunctions.

## Installation

```bash
git clone https://github.com/KabirDabholkar/separatrix_locator.git
cd separatrix_locator
pip install -e .
```

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
    model_class=KoopmanEigenfunctionModel,
    epochs=500
)

# Train the models
locator.fit(dynamics.function, dynamics.distribution)

# Find the separatrix
locator.prepare_models_for_gradient_descent(dynamics.distribution)
trajectories, separatrix_points = locator.find_separatrix(dynamics.distribution)
```

## Dynamical Systems

- **Bistable Systems**: 1D, 2D, and 3D systems with two stable fixed points
- **Duffing Oscillator**: Classic nonlinear oscillator
- **Custom Systems**: Inherit from `DynamicalSystem` base class

## Model Architectures

- **KoopmanEigenfunctionModel**: Basic feedforward network
- **ResRBF**: Residual network with skip connections
- **LinearModel**: Simple linear baseline
- **DeepKoopmanModel**: Deep network for complex systems

## Examples

See `examples/` directory for tutorials:
- `01_bistable_1d.ipynb`: Introduction with 1D system
- `02_duffing_oscillator.ipynb`: 2D systems and phase portraits
