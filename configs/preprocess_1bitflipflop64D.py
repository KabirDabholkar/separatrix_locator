from pathlib import Path
import torch
from base import *
from separatrix_locator.dynamics.rnn import RNN, get_autonomous_dynamics_from_model, discrete_to_continuous, set_model_with_checkpoint
from separatrix_locator.utils import get_estimate_attractor_func
from separatrix_locator.core.separatrix_point import find_separatrix_point_along_line


dim = 64

# RNN setup (vanilla RNN)
rnn_model = RNN(num_h=dim, ob_size=1, act_size=1, RNN_class='RNN')
ckpt = Path("rnn_params/1bitflipflop64D/RNNmodel.torch")
if ckpt.exists():
    checkpoint = torch.load(ckpt, weights_only=True)
    rnn_model = set_model_with_checkpoint(rnn_model, checkpoint)


discrete_dynamics = get_autonomous_dynamics_from_model(rnn_model)
continuous_dynamics = discrete_to_continuous(discrete_dynamics, delta_t=1.0)


def dynamics_function(x: torch.Tensor) -> torch.Tensor:
    return continuous_dynamics(x) * 6.0


# Estimate attractors utility obtained from utils
estimate_attractors = get_estimate_attractor_func(dynamics_function)


def main(save_dir: Path = Path("results/1bitflipflop64D_precompute")):
    save_dir.mkdir(parents=True, exist_ok=True)
    a1, a2 = estimate_attractors(dim)
    point_on_sep = find_separatrix_point_along_line(dynamics_function, None, (a1, a2), num_points=20, num_iterations=4, final_time=30)
    torch.save(point_on_sep, save_dir / "point_on_separatrix.pt")
    print("Saved:", save_dir / "point_on_separatrix.pt")


if __name__ == "__main__":
    main()


