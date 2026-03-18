"""
Discrete-time dimensionality reduction for Koopman-style consistency.

This subpackage provides:
- a learnable encoder `x -> z`
- a learnable discrete latent dynamics `z_{t+1} = g(z_t, u_t)`
- a learnable latent eigenfunction `psi(z)`
- losses that enforce one-step Koopman consistency and geometry preservation.
"""

from .models import EncoderNN, LatentDynamicsNN, LatentEigenfunctionNN
from .losses import (
    pairwise_stress_loss,
    discrete_one_step_state_loss,
    discrete_one_step_koopman_loss,
    call_rhs_function,
)
from .training_discrete import train_discrete_reduction
from .trajectory_utils import (
    rollout_rnn_hidden_states,
    extract_one_step_pairs,
)

__all__ = [
    "EncoderNN",
    "LatentDynamicsNN",
    "LatentEigenfunctionNN",
    "pairwise_stress_loss",
    "discrete_one_step_state_loss",
    "discrete_one_step_koopman_loss",
    "call_rhs_function",
    "train_discrete_reduction",
    "rollout_rnn_hidden_states",
    "extract_one_step_pairs",
]

