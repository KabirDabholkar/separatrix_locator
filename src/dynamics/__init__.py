"""
Dynamical systems for separatrix location.

This module contains various dynamical systems that can be used with the
separatrix locator, from simple 1D bistable systems to complex neural circuits.
"""

from .base import DynamicalSystem
from .bistableND import BistableND
from .duffing import DuffingOscillator

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
]
