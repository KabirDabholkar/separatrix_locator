from separatrix_locator.core.models import ResNet
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.dynamics.rnn import RNN, get_autonomous_dynamics_from_model, discrete_to_continuous, set_model_with_checkpoint, get_spectral_norm
from pathlib import Path
from base import *
import torch
from separatrix_locator.utils import run_odeint_to_final
from separatrix_locator.core.separatrix_point import find_separatrix_point_along_line
from preprocess_1bitflipflop64D import estimate_attractors

# 64D 1-bit flip-flop RNN-driven dynamics, mirroring old main_1bitflipflop64D.yaml

dim = 64

# Training parameters (from old YAML main_1bitflipflop64D)
epochs = 1000
batch_size = 1000
balance_loss_lambda = 1e-2 #5e-3
# RHS_function = "lambda psi: psi-psi**3"
RHS_function = "lambda psi: psi"

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
        self._attractors = None
        self._precompute_dir = Path("results/1bitflipflop64D_precompute")

    def function(self, x):
        return self.func(x)

    def get_attractors(self):
        """Return cached attractors if available; otherwise load from disk or compute and save.

        Workflow:
        1) If previously computed in this session, return the cached attribute.
        2) If an on-disk cache exists, load and cache it, then return.
        3) Otherwise compute via estimate_attractors(dim), save to disk, cache, and return.
        """
        if self._attractors is not None:
            return self._attractors

        self._precompute_dir.mkdir(parents=True, exist_ok=True)
        attractors_path = self._precompute_dir / "attractors.pt"

        if attractors_path.exists():
            loaded = torch.load(attractors_path)
            if isinstance(loaded, (tuple, list)) and len(loaded) == 2:
                a1, a2 = loaded
            else:
                # Fallback: recompute if format is unexpected
                a1, a2 = estimate_attractors(dim)
                torch.save((a1, a2), attractors_path)
        else:
            a1, a2 = estimate_attractors(dim)
            torch.save((a1, a2), attractors_path)

        self._attractors = (a1, a2)
        return self._attractors


dynamics = RNNFlipFlopDynamics(dynamics_function)

model = ResNet(
    input_dim=dim,
    hidden_size=550,
    num_layers=8,
    output_dim=1,
    input_scale_factor=1.0,
    input_center=None,
    scale_last_layer_by_inv_sqrt_hidden=False
)

sep_save_default = Path("results/1bitflipflop64D_precompute/point_on_separatrix.pt")
if sep_save_default.exists():
    separatrix_point = torch.load(sep_save_default)
else:
    # Fallback: estimate attractors quickly and compute separatrix point on the fly
    with torch.no_grad():
        a1, a2 = dynamics.get_attractors()
    separatrix_point = find_separatrix_point_along_line(dynamics_function, None, (a1, a2), num_points=20, num_iterations=4, final_time=30)

dist = MultivariateGaussian(dim=dim, mean=separatrix_point, covariance_matrix=None)

# Compute spectral norm from hidden-state covariance and scale IC radii accordingly
with torch.no_grad():
    seq_len = dataset_params['n_time']
    batch_size_local = dataset_params['n_trials']
    input_size = rnn_params['ob_size']

    # Simple synthetic input resembling flip-flop stats (Bernoulli with prob p)
    bern_prob = dataset_params.get('p', 0.2)
    inputs = torch.bernoulli(torch.full((seq_len, batch_size_local, input_size), bern_prob, dtype=torch.float32))

    # Run model to obtain hidden states (shape: seq_len, batch, hidden_dim)
    _, hidden_seq = rnn_model(inputs, return_hidden=True)
    spectral_norm = get_spectral_norm(hidden_seq)

base_scales = [0.001, 0.005, 0.01, 0.1, 0.5, 1.0, 2.0]
# print(f"Spectral norm: {spectral_norm}")
# scaled_scales = [s * spectral_norm for s in base_scales]

# Multiscale sampling centered at separatrix point with spectral-norm scaling
dists = multiscaler(dist, base_scales)

save_dir = Path("results") / f"{dynamics.name}_{dists.name}" / f"{model.name}"

print('separatrix_point[:10]',separatrix_point[:10])

if __name__ == "__main__":
    attractors = RNNFlipFlopDynamics(dynamics_function).get_attractors()
    # print('attractors',attractors)

    print("||dynamics_function(attractors[0])|| =", torch.norm(dynamics_function(attractors[0])).item())
    print("||dynamics_function(attractors[1])|| =", torch.norm(dynamics_function(attractors[1])).item())