"""
Default distributions for systems.

This package decouples sampling distributions from dynamics definitions.
"""

from .base import BaseDistribution
from .gaussian import MultivariateGaussian

__all__ = [
    "BaseDistribution",
    "MultivariateGaussian",
]


