from pathlib import Path
from functools import partial
import torch
import scipy.io as sio

from configs.base import *  # epochs, batch_size defaults etc.; will be overridden below
from separatrix_locator.core.models import ResNet
from separatrix_locator.utils.odeint_utils import run_odeint_to_final
from separatrix_locator.core.separatrix_point import find_separatrix_point_along_line
from separatrix_locator.dynamics.rnn import (
    get_autonomous_dynamics_from_model,
    discrete_to_continuous,
)
from separatrix_locator.utils.finkelstein_fontolan_RNN import (
    init_network,
    extract_opposite_attractors_from_model,
)
from separatrix_locator.utils.finkelstein_fontolan_task import initialize_task


# -----------------------------
# Data/params sources
# -----------------------------
_PARAMS_DIR = Path("rnn_params/finkelstein_fontolan")
_INPUT_DATA_DIR = _PARAMS_DIR / "input_data"


def _load_params_dict(params_mat_path: Path) -> dict:
    """Build the params_dict expected by init_network from params_data_wramp.mat.

    Mirrors the extraction logic used in initialize_task and utils file.
    """
    params_mat = sio.loadmat(str(params_mat_path))
    # MATLAB struct access is nested; follow utils.initialize_task
    N = int(params_mat['params']['N'][0][0])
    params = {
        "N": N,
        "dt": params_mat['params']['dt'][0][0][0, 0],
        "tau": params_mat['params']['tau'][0][0][0, 0],
        "f0": params_mat['params']['f0'][0][0][0, 0],
        "beta0": float(params_mat['params']['beta0'][0][0]),
        "theta0": float(params_mat['params']['theta0'][0][0]),
        # Recurrent weights and bias (neurons only)
        "M": params_mat['params']['M'][0, 0][:N, :N],
        "h": params_mat['params']['h'][0, 0][:N].flatten(),
        # Effective step and noise
        "eff_dt": params_mat['params']['eff_dt'][0][0],
        "sigma_noise_cd": 100.0 / N,
        # For x_init computation
        "des_out_left": params_mat['params']['des_out_left'],
        "des_out_right": params_mat['params']['des_out_right'],
        # Task inputs
        "ramp_train": params_mat['params']['ramp_train'],
    }
    return params


# -----------------------------
# Build RNN dynamics (autonomous, continuous-time)
# -----------------------------
_params_mat = _INPUT_DATA_DIR.parent / "params_data_wramp.mat"
_params_dict = _load_params_dict(_params_mat)

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_rnn_model = init_network(_params_dict, device=_device)

_discrete = get_autonomous_dynamics_from_model(
    _rnn_model,
    device=_device,
    rnn_submodule_name=None,  # RNNModel already has the forward
    kwargs={"deterministic": True, "batch_first": False},
    output_id=1,  # return hidden state trajectory
)
_continuous = discrete_to_continuous(_discrete, delta_t=1.0)

def dynamics_function(x: torch.Tensor, external_input: torch.Tensor = None) -> torch.Tensor:
    # External input is a 3D vector (chirp, stim, ramp). If None, assume zeros.
    if external_input is None:
        external_input = torch.zeros(x.shape[-2] if x.dim() > 1 else 1, 3, device=x.device)
    return _continuous(x, external_input)


class FinkelsteinFontolanDynamics:
    def __init__(self, func, dim: int, name: str = "FinkelsteinFontolanRNN"):
        self.func = func
        self.name = name
        self.dim = dim

    def function(self, x: torch.Tensor, external_input: torch.Tensor = None) -> torch.Tensor:
        return self.func(x, external_input)


# Dimension from params
dim = int(_params_dict["N"])  # expected 668

# Increase speed consistent with YAML factor 60.0 by wrapping dynamics
def _speed_scaled(x: torch.Tensor, external_input: torch.Tensor = None) -> torch.Tensor:
    return dynamics_function(x, external_input) * 60.0

dynamics = FinkelsteinFontolanDynamics(_speed_scaled, dim=dim)


# -----------------------------
# Static external input and attractors for separatrix point
# -----------------------------
# Build dataset generator from MAT files
dataset = initialize_task(str(_INPUT_DATA_DIR) + "/", N_trials_cd=10)

# Choose ramp range around 0.9-0.92 as in YAML
_input_range = (0.9, 0.92)

# Find two opposite attractors and refine by short integration at fixed external input
_attractors_unrefined = extract_opposite_attractors_from_model(
    _rnn_model, dataset, input_range=_input_range
)

_static_external_input = torch.tensor([0.0, 0.0, _input_range[0]], dtype=torch.float32, device=_device)

with torch.no_grad():
    _attractors = run_odeint_to_final(
        lambda x, u: dynamics.function(x, u),
        torch.tensor(_attractors_unrefined, dtype=torch.float32, device=_device),
        T=50,
        inputs=_static_external_input,
        steps=2,
        return_last_only=True,
        no_grad=True,
    )

# Compute a point on the separatrix between the two attractors
with torch.no_grad():
    point_on_separatrix = find_separatrix_point_along_line(
        lambda x: dynamics.function(x, _static_external_input),
        _static_external_input,
        (torch.as_tensor(_attractors[0]), torch.as_tensor(_attractors[1])),
        num_points=10,
        num_iterations=4,
        final_time=10,
    )


# -----------------------------
# Model and training hyperparameters (mirroring YAML)
# -----------------------------
model = ResNet(
    input_dim=dim,
    hidden_size=800,
    num_layers=25,
    output_dim=1,
    input_scale_factor=0.02,
    input_center=point_on_separatrix.detach().cpu(),
    scale_last_layer_by_inv_sqrt_hidden=True,
)

# Training parameters
epochs = 5000
batch_size = 3000
balance_loss_lambda = 5e-2
learning_rate = 1e-4
optimizer = partial(torch.optim.AdamW, lr=learning_rate, weight_decay=1e-4)
RHS_function = "lambda psi: psi-psi**3"


# -----------------------------
# Output directory
# -----------------------------
save_dir = Path("results") / f"{dynamics.name}" / f"{model.name}"


