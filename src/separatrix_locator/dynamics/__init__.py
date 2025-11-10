"""
Dynamical systems for separatrix location.

This module contains various dynamical systems that can be used with the
separatrix locator, from simple 1D bistable systems to complex neural circuits.
"""

from .base import DynamicalSystem
from .bistableND import BistableND
from .duffing import DuffingOscillator
from .FlipFlopData import FlipFlopData
from .flipflop import FlipFlop1Bit2D, FlipFlop2Bit2D
from .flipflop_task import FlipFlopDataset, FlipFlopSweepDataset
from .rnn_training import (
    OptimizerConfig,
    TrainingConfig,
    TrainingResult,
    build_rnn_model,
    train_flipflop_rnn,
)

# Backward-compatible shims for previous class names
class Bistable1D(BistableND):
    def __init__(self, **kwargs):
        super().__init__(dim=1, **kwargs)


class Bistable2D(BistableND):
    def __init__(self, **kwargs):
        super().__init__(dim=2, **kwargs)


class Bistable3D(BistableND):
    def __init__(self, **kwargs):
        super().__init__(dim=3, **kwargs)

__all__ = [
    "DynamicalSystem",
    "BistableND",
    "Bistable1D",
    "Bistable2D", 
    "Bistable3D",
    "DuffingOscillator",
    "FlipFlopData",
    "FlipFlop1Bit2D",
    "FlipFlop2Bit2D",
    "FlipFlopDataset",
    "FlipFlopSweepDataset",
    "OptimizerConfig",
    "TrainingConfig",
    "TrainingResult",
    "build_rnn_model",
    "train_flipflop_rnn",
]
