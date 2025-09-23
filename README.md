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
Install the correct CUDA-enabled PyTorch first using the PyTorch wheel index, then install this package.

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


