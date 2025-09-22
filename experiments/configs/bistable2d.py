from src.core.models import ResNet
from src.distributions import MultivariateGaussian, multiscaler
from src.dynamics import Bistable2D
from pathlib import Path

dynamics = Bistable2D()

model = ResNet(
    input_dim=2,
    hidden_size=128,
    num_layers=20,
    output_dim=1,
    input_scale_factor=1.0,
    input_center=None,
    scale_last_layer_by_inv_sqrt_hidden=False
)

dist = MultivariateGaussian(
    dim=2,
    mean=None,
    covariance_matrix=None
)

dists = multiscaler(dist, [0.1, 0.5, 1.0])

save_dir = Path("results") / f"{dynamics.name}_{dists.name}" / f"{model.name}"