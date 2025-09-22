"""
Default distributions for systems.

This package decouples sampling distributions from dynamics definitions.
"""

from .base import BaseDistribution, BaseDistributionList
from .gaussian import MultivariateGaussian, multiscaler

__all__ = [
    "BaseDistribution",
    "BaseDistributionList",
    "MultivariateGaussian",
    "multiscaler",
]


