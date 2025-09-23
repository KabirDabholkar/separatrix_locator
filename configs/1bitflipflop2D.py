from separatrix_locator.core.models import ResNet
from separatrix_locator.distributions import MultivariateGaussian, multiscaler
from separatrix_locator.dynamics import FlipFlop1Bit2D
from separatrix_locator.dynamics.rnn import GRU_RNN, get_autonomous_dynamics_from_model, discrete_to_continuous, set_model_with_checkpoint
from pathlib import Path
from configs.base import *
import torch

dim = 2

# Training parameters from old config
epochs = 300
batch_size = 1000
balance_loss_lambda = 1e-2
RHS_function = "lambda psi: psi-psi**3"

# RNN parameters from old YAML config
# GRU_RNN with ob_size=1, act_size=1, num_h=2 (dim=2, k_bit=1)
rnn_params = {
    'ob_size': 1,      # k_bit
    'act_size': 1,     # k_bit  
    'num_h': 2,        # dim
    'tau': 10.0,       # time constant for flip-flop dynamics
    'speed_factor': 6.0,  # factor from change_speed function
}

# Dataset parameters from old config
dataset_params = {
    'n_trials': 16,
    'n_time': 50,
    'n_bits': 1,       # k_bit
    'p': 0.1,          # probability parameter
    'random_seed': 2,
}

# Attractor and separatrix parameters
attractor_params = {
    'num_clusters': 2,
    'separatrix_iterations': 4,
    'separatrix_points': 20,
    'final_time': 30,
}

# RNN model setup
rnn_model = GRU_RNN(
    num_h=rnn_params['num_h'],
    ob_size=rnn_params['ob_size'], 
    act_size=rnn_params['act_size']
)

# Load pre-trained RNN weights (placeholder for now)
rnn_checkpoint_path = Path("rnn_params/1bitflipflop2D/RNNmodel.torch")

if rnn_checkpoint_path.exists():
    checkpoint = torch.load(rnn_checkpoint_path, weights_only=True)
    rnn_model = set_model_with_checkpoint(rnn_model, checkpoint)

# Get autonomous dynamics from the RNN model
discrete_dynamics = get_autonomous_dynamics_from_model(rnn_model)
continuous_dynamics = discrete_to_continuous(discrete_dynamics, delta_t=1.0)

# Apply speed factor from old config
def dynamics_function(x):
    return continuous_dynamics(x) * rnn_params['speed_factor']

# Create a dynamics object that wraps the RNN-based dynamics
class RNNFlipFlopDynamics:
    def __init__(self, func, name="RNNFlipFlop1Bit2D"):
        self.func = func
        self.name = name
        self.dim = 2
    
    def function(self, x):
        return self.func(x)

dynamics = RNNFlipFlopDynamics(dynamics_function)

model = ResNet(
    input_dim=dim,
    hidden_size=400,  # from old config
    num_layers=20,    # from old config
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

# Scale ranges from old config: [1e-2,0.1,0.5,1.0,2.0,4.0]
dists = multiscaler(dist, [0.01, 0.1, 0.5, 1.0, 2.0, 4.0])

# Plotting limits from old config



save_dir = Path("results") / f"{dynamics.name}_{dists.name}" / f"{model.name}"
