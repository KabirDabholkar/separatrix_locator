from separatrix_locator.core.models import ResNet
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.dynamics import DuffingOscillator
from pathlib import Path
from base import *

dim = 2
epochs = 1000
batch_size = 1000
balance_loss_lambda = 1e-2
RHS_function = "lambda psi: psi-psi**3"


dynamics = DuffingOscillator()

model = ResNet(
    input_dim=dim,
    hidden_size=400,
    num_layers=20,
    output_dim=1,
    input_scale_factor=1.0,
    input_center=None,
    scale_last_layer_by_inv_sqrt_hidden=True
)

dist = MultivariateGaussian(
    dim=dim,
    mean=None,
    covariance_matrix=None
)

dists = multiscaler(dist, [0.1, 0.5, 1.0, 2.0])

# Plotting limits
x_limits = (-4, 4)
y_limits = (-4, 4)

save_dir = Path("results") / f"{dynamics.name}_{dists.name}" / f"{model.name}"