# Separatrix Locator

A Python package for locating separatrices in black-box dynamical systems using Koopman eigenfunctions from the paper: [*Finding separatrices of dynamical flows with Deep Koopman Eigenfunctions*](https://arxiv.org/abs/2505.15231).

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

## Baselines experiments included in the repo
To run the experiments, run the following:

```bash
python experiments/simple_experiment.py configs/[config_name].py
```

- **``configs/bistable2d.py``**: Simple 2D bistable system.
    ```math
    \dot x = x - x^3 \\
    \dot y = -y
    ```

- **``configs/duffing.py``**: Classic 2D nonlinear bistable oscillator 
    ```math
    \dot x = -y \\
    \dot y = x - x^3
    ```

- **``configs/1bitflipflop2D.py``**: 2-unit GRU RNN trained to do 1-bit flip with bistable dynamics. Pre-trained params included in repository.

- **``configs/1bitflipflop64D.py``**: 64-unit Vanilla RNN trained to do 1-bit flip with bistable dynamics. Pre-trained params included in repository.

Resulting figures and models are saved in `results`.

## Model Architectures

- **ResNet**: Residual network with skip connections,
```python
from separatrix_locator.core.models import ResNet
```
- **RBF**: Radial basis functions


## Quick start:

Example: Learning a the Koopman eigenfuncion for a 1D system with $\dot x = \sin(x)$, implemented by inheriting from `separatix_locator/dynamics/base.py`'s `DynamicalSystem`:

```python
import torch
from torch import nn

from separatrix_locator.dynamics.base import DynamicalSystem
from separatrix_locator.core.separatrix_locator import SeparatrixLocator

# Define the dynamics as a torch function
dynamical_function = torch.sin 

# Define Neural network architecture to approximate the Koopman Eigenfunction
model = nn.Sequential(
    nn.Linear(1, 100),
    nn.Tanh(),
    nn.Linear(100, 100),
    nn.Tanh(),
    nn.Linear(100, 1)
)

# Define Separatix Locator object, providing the model a kwarg.
locator = SeparatrixLocator(
    dynamics_dim=1,
    models=[model], # provide a list of models. Even if just one. 
    epochs=200, # training epochs (same as steps)
    verbose=True,
)

# Choose a sampling distribution for initial conditions (mean 0, std 1)
distribution = torch.distributions.Uniform(low=torch.tensor([-4.0]), high=torch.tensor([4.0]))

# Train models to learn Koopman eigenfunctions
locator.fit(dynamical_function, distribution)
```


