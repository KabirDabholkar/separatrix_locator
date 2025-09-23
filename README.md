# Separatrix Locator

A Python package for locating separatrices in black-box dynamical systems using Koopman eigenfunctions.

## Installation

```bash
git clone https://github.com/KabirDabholkar/separatrix_locator.git
cd separatrix_locator
```

### CPU-only (simple)
```bash
pip install -e .
```

### CUDA (GPU) users
Install [the correct CUDA-enabled PyTorch](https://pytorch.org/get-started/locally/) first using the PyTorch wheel index, then install this package.

Examples:
```bash
# CUDA 11.8, PyTorch 2.4.* (adjust versions as needed)
pip install --index-url https://download.pytorch.org/whl/cu118 torch==2.4.*

# Then install this package (keeps your installed torch)
pip install -e .
```

Tip: you can set `PIP_INDEX_URL` to the PyTorch CUDA index in your environment or `pip.conf` to make CUDA selection persistent.

## Run baselines

```bash
python -m experiments.simple_experiment configs/bistable2d.py
```

```bash
python -m experiments.simple_experiment configs/duffing.py
```

## Quick Start

```python
from src.core.separatrix_locator import SeparatrixLocator
from src.dynamics import Bistable1D
from src.core.models import ResNet

# Create a simple bistable system
dynamics = Bistable1D()

# Set up the separatrix locator
locator = SeparatrixLocator(
    num_models=5,
    dynamics_dim=dynamics.dim,
    model_class=ResNet,
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

- **ResNet**: Residual network with skip connections
- **RBF**: Radial basis functions


## Implement a custom dynamical system

Example: 1D system with \(\dot x = x - x^3\), implemented by inheriting from `src/dynamics/base.py`'s `DynamicalSystem`:

```python
import torch
from src.dynamics.base import DynamicalSystem

class CubicBistable1D(DynamicalSystem):
    def __init__(self):
        super().__init__(dim=1, name="CubicBistable1D")

    def function(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1)
        return x - x ** 3

    def get_attractors(self) -> torch.Tensor:
        # Stable fixed points at x = -1 and x = 1
        return torch.tensor([[-1.0], [1.0]], dtype=torch.float32)


# Usage with the SeparatrixLocator
if __name__ == "__main__":
    from src.core.separatrix_locator import SeparatrixLocator
    from src.core.models import ResNet

    dynamics = CubicBistable1D()

    locator = SeparatrixLocator(
        num_models=3,
        dynamics_dim=dynamics.dim,
        model_class=ResNet,
        epochs=200,
    )

    # Choose a sampling distribution for initial conditions (mean 0, std 1)
    init_dist = torch.distributions.Normal(loc=torch.tensor([0.0]), scale=torch.tensor([1.0]))

    # Train models to learn Koopman eigenfunctions
    locator.fit(dynamics.function, init_dist)

    # Prepare for gradient-based refinement and find separatrix
    locator.prepare_models_for_gradient_descent(init_dist)
    trajectories, separatrix_points = locator.find_separatrix(init_dist)
```


