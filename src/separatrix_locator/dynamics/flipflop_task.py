"""
Utilities for generating flip-flop task data used to train recurrent
neural networks.

These helpers wrap the data generation utilities from the fixed point
finder project and adapt them into simple callable dataset objects that
return NumPy arrays suitable for PyTorch training loops.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .FlipFlopData import FlipFlopData


class FlipFlopDataset(FlipFlopData):
    """
    Callable dataset for the k-bit flip-flop task.

    Parameters mirror those of :class:`FlipFlopData`, with the addition of
    ``n_trials`` controlling the number of generated trials and ``repeats``
    duplicating the dataset to increase its size.
    """
    def __init__(self, n_trials: int, repeats: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.n_trials = n_trials
        self.repeats = repeats

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        data = self.generate_data(n_trials=self.n_trials)
        inputs, targets = data["inputs"].swapaxes(0, 1), data["targets"].swapaxes(0, 1)

        if self.repeats > 1:
            inputs = np.repeat(inputs, self.repeats, axis=0)
            targets = np.repeat(targets, self.repeats, axis=0)

        return inputs, targets


class FlipFlopSweepDataset(FlipFlopDataset):
    """
    Deterministic sweep dataset used to probe network responses.

    Generates a set of initial inputs with the first bit sweeping between
    ``-1`` and ``1`` halfway through the sequence.
    """

    def __init__(self, sign: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.sign = sign

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.n_time, self.n_trials, self.n_bits), dtype=float)
        inputs[0] = -1
        inputs[self.n_time // 2, :, 0] = np.linspace(-1, 1, self.n_trials)
        inputs = np.repeat(inputs, self.repeats, axis=0)

        targets = np.full_like(inputs, np.nan)

        inputs = inputs * self.sign
        return inputs, targets


__all__ = ["FlipFlopDataset", "FlipFlopSweepDataset"]

