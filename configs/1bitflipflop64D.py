from separatrix_locator.core.models import ResNet
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.dynamics.rnn import RNN, get_autonomous_dynamics_from_model, discrete_to_continuous, set_model_with_checkpoint
from pathlib import Path
from base import *
import torch
from separatrix_locator.utils import run_odeint_to_final
from separatrix_locator.core.separatrix_point import find_separatrix_point_along_line

# 64D 1-bit flip-flop RNN-driven dynamics, mirroring old main_1bitflipflop64D.yaml

dim = 64

# Training parameters (from old YAML main_1bitflipflop64D)
epochs = 1000
batch_size = 100
balance_loss_lambda = 5e-2
RHS_function = "lambda psi: psi-psi**3"

# RNN hyperparameters mirroring the previous setup
rnn_params = {
    'ob_size': 1,       # k_bit
    'act_size': 1,      # k_bit
    'num_h': dim,       # hidden/state dimension equals system dimension
    'tau': 10.0,
    'speed_factor': 6.0,
}

# Dataset parameters (kept consistent with 2D config, safe defaults)
dataset_params = {
    'n_trials': 16,
    'n_time': 50,
    'n_bits': 1,
    'p': 0.2,
    'random_seed': 2,
}

# RNN model and checkpoint
rnn_model = RNN(
    num_h=rnn_params['num_h'],
    ob_size=rnn_params['ob_size'],
    act_size=rnn_params['act_size'],
    RNN_class='RNN'
)

# Optional: load pretrained weights if available
rnn_checkpoint_path = Path("rnn_params/1bitflipflop64D/RNNmodel.torch")
if rnn_checkpoint_path.exists():
    checkpoint = torch.load(rnn_checkpoint_path, weights_only=True)
    rnn_model = set_model_with_checkpoint(rnn_model, checkpoint)

# Build autonomous continuous-time dynamics from the RNN
discrete_dynamics = get_autonomous_dynamics_from_model(rnn_model)
continuous_dynamics = discrete_to_continuous(discrete_dynamics, delta_t=1.0)

def dynamics_function(x: torch.Tensor) -> torch.Tensor:
    return continuous_dynamics(x) * rnn_params['speed_factor']


class RNNFlipFlopDynamics:
    def __init__(self, func, name=f"RNNFlipFlop1Bit{dim}D"):
        self.func = func
        self.name = name
        self.dim = dim

    def function(self, x):
        return self.func(x)


dynamics = RNNFlipFlopDynamics(dynamics_function)

model = ResNet(
    input_dim=dim,
    hidden_size=400,
    num_layers=20,
    output_dim=1,
    input_scale_factor=1.0,
    input_center=None,
    scale_last_layer_by_inv_sqrt_hidden=True
)

sep_save_default = Path("results/1bitflipflop64D_precompute/point_on_separatrix.pt")
if sep_save_default.exists():
    separatrix_point = torch.load(sep_save_default)
else:
    # Fallback: estimate attractors quickly and compute separatrix point on the fly
    with torch.no_grad():
        y0 = torch.randn(128, dim)
        finals = run_odeint_to_final(lambda x: dynamics_function(x), y0, 30.0, return_last_only=True)
        from sklearn.cluster import KMeans
        km = KMeans(n_clusters=2, random_state=0).fit(finals.cpu())
        centers = torch.tensor(km.cluster_centers_, dtype=finals.dtype)
        a1, a2 = centers[0], centers[1]
    separatrix_point = find_separatrix_point_along_line(dynamics_function, None, (a1, a2), num_points=20, num_iterations=4, final_time=30)

dist = MultivariateGaussian(dim=dim, mean=separatrix_point, covariance_matrix=None)

# Multiscale sampling centered at separatrix point (matching old YAML intent)
dists = multiscaler(dist, [0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 2.0])

save_dir = Path("results") / f"{dynamics.name}_{dists.name}" / f"{model.name}"


