"""
Reusable training utilities for recurrent neural networks on the flip-flop task.

The goal of this module is to provide a lightweight, Hydra-free interface for
training and analysing RNNs that can be imported both in scripts and in
tutorial notebooks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from torch import nn

from .rnn import GRU_RNN, RNN


DatasetCallable = Callable[[], Tuple[np.ndarray, np.ndarray]]
PathLike = Union[str, Path]


@dataclass
class OptimizerConfig:
    """Configuration for the Adam optimizer."""

    lr: float = 5e-4
    weight_decay: float = 1e-2
    betas: Tuple[float, float] = (0.9, 0.999)


@dataclass
class TrainingConfig:
    """High-level knobs controlling the training loop."""

    epochs: int = 400
    log_interval: int = 100
    device: Optional[str] = None
    save_dir: Optional[PathLike] = None
    save_checkpoint: bool = True
    save_loss_plot: bool = True
    checkpoint_name: str = "RNNmodel.torch"
    loss_plot_name: str = "RNN_loss_hist.png"

    def resolved_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resolved_save_dir(self) -> Optional[Path]:
        if self.save_dir is None:
            return None
        path = Path(self.save_dir)
        path.mkdir(parents=True, exist_ok=True)
        return path


@dataclass
class TrainingResult:
    """Container for the outputs of a training run."""

    model: nn.Module
    loss_history: List[float]
    device: torch.device
    config: TrainingConfig = field(repr=False)

    def save_checkpoint(self, path: Optional[PathLike] = None) -> Path:
        """Save the trained model weights to ``path``."""
        if path is None:
            save_dir = self.config.resolved_save_dir()
            if save_dir is None:
                raise ValueError("No save directory configured for this training run.")
            path = save_dir / self.config.checkpoint_name
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), path)
        return path

    def plot_loss(self, path: Optional[PathLike] = None) -> Optional[Path]:
        """Plot the training loss history."""
        if not self.loss_history:
            return None

        plt.figure()
        plt.plot(self.loss_history)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Flip-Flop RNN Training Loss")

        if path is None:
            save_dir = self.config.resolved_save_dir()
            if save_dir is None:
                plt.close()
                return None
            path = save_dir / self.config.loss_plot_name
        else:
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(path, dpi=300, bbox_inches="tight")
        plt.close()
        return path

    def evaluate(self, dataset: DatasetCallable) -> Tuple[np.ndarray, np.ndarray]:
        """Run the trained model on a dataset and return inputs and outputs."""
        self.model.eval()
        with torch.no_grad():
            inputs, targets = dataset()
            inputs_tensor = torch.from_numpy(inputs).to(device=self.device, dtype=torch.float32)
            outputs = self.model(inputs_tensor).detach().cpu().numpy()
        self.model.train()
        return inputs, outputs

    def compute_hidden_pca(
        self,
        dataset: DatasetCallable,
        n_components: int = 2,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute PCA on the hidden states produced by the model for a dataset.

        Returns:
            hidden_states: (seq_len, batch, hidden_dim) array
            hidden_pca: (seq_len, batch, n_components) array
            explained_variance_ratio: length ``n_components`` array
        """
        self.model.eval()
        with torch.no_grad():
            inputs, _ = dataset()
            inputs_tensor = torch.from_numpy(inputs).to(device=self.device, dtype=torch.float32)
            outputs, hidden_states = self.model(inputs_tensor, return_hidden=True)
            hidden_states = hidden_states.detach().cpu().numpy()

        hidden_flat = hidden_states.reshape(-1, hidden_states.shape[-1])
        pca = PCA(n_components=n_components)
        hidden_pca = pca.fit_transform(hidden_flat)
        hidden_pca = hidden_pca.reshape(*hidden_states.shape[:-1], n_components)

        self.model.train()
        return hidden_states, hidden_pca, pca.explained_variance_ratio_


def build_rnn_model(
    input_size: int,
    output_size: int,
    hidden_size: int = 64,
    cell_type: str = "GRU",
) -> nn.Module:
    """
    Construct a recurrent network compatible with the training utilities.

    Args:
        input_size: Dimensionality of the input.
        output_size: Dimensionality of the output.
        hidden_size: Number of hidden units.
        cell_type: ``"GRU"``, ``"RNN"``, or ``"LSTM"`` (case-insensitive).
    """
    cell = cell_type.upper()
    if cell == "GRU":
        return GRU_RNN(num_h=hidden_size, ob_size=input_size, act_size=output_size)

    if cell in {"RNN", "LSTM"}:
        return RNN(num_h=hidden_size, ob_size=input_size, act_size=output_size, RNN_class=cell)

    raise ValueError(f"Unsupported cell type '{cell_type}'. Expected 'GRU', 'RNN', or 'LSTM'.")


def train_flipflop_rnn(
    dataset: DatasetCallable,
    input_size: int,
    output_size: int,
    hidden_size: int = 64,
    criterion: Optional[nn.Module] = None,
    optimizer_config: Optional[OptimizerConfig] = None,
    training_config: Optional[TrainingConfig] = None,
    cell_type: str = "GRU",
    progress_callback: Optional[Callable[[int, float], None]] = None,
) -> TrainingResult:
    """
    Train an RNN on the flip-flop task using the provided dataset generator.

    Args:
        dataset: Callable returning ``(inputs, targets)`` NumPy arrays with
            shapes ``(seq_len, batch, input_size)`` and ``(seq_len, batch, output_size)``.
        input_size: Dimension of the input bits.
        output_size: Dimension of the target outputs.
        hidden_size: Number of hidden units in the RNN.
        criterion: Loss function (defaults to ``nn.MSELoss``).
        optimizer_config: Hyper-parameters for Adam.
        training_config: High-level training configuration.
        cell_type: RNN cell to build (``"GRU"``, ``"RNN"``, or ``"LSTM"``).
        progress_callback: Optional callable invoked as ``callback(iteration, loss)``.
    """
    if optimizer_config is None:
        optimizer_config = OptimizerConfig()
    if training_config is None:
        training_config = TrainingConfig()
    if criterion is None:
        criterion = nn.MSELoss()

    device = training_config.resolved_device()
    model = build_rnn_model(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        cell_type=cell_type,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=optimizer_config.lr,
        weight_decay=optimizer_config.weight_decay,
        betas=optimizer_config.betas,
    )

    loss_history: List[float] = []

    model.train()
    for iteration in range(training_config.epochs):
        inputs_np, targets_np = dataset()

        inputs = torch.from_numpy(inputs_np).to(device=device, dtype=torch.float32)
        targets = torch.from_numpy(targets_np).to(device=device, dtype=torch.float32)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        loss_value = float(loss.item())
        loss_history.append(loss_value)

        if progress_callback:
            progress_callback(iteration, loss_value)

        if training_config.log_interval and (iteration + 1) % training_config.log_interval == 0:
            print(f"[{iteration + 1:04d}] loss: {loss_value:.5f}")

    result = TrainingResult(model=model, loss_history=loss_history, device=device, config=training_config)

    save_dir = training_config.resolved_save_dir()
    if save_dir is not None:
        if training_config.save_checkpoint:
            result.save_checkpoint(save_dir / training_config.checkpoint_name)
        if training_config.save_loss_plot:
            result.plot_loss(save_dir / training_config.loss_plot_name)

    return result


__all__ = [
    "OptimizerConfig",
    "TrainingConfig",
    "TrainingResult",
    "build_rnn_model",
    "train_flipflop_rnn",
]

