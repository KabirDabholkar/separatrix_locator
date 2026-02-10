"""
Default distributions for systems.

This package decouples sampling distributions from dynamics definitions.
"""

from .base import BaseDistribution, BaseDistributionList
from .empirical import EmpiricalDistribution
from .gaussian import MultivariateGaussian, multiscaler

__all__ = [
    "BaseDistribution",
    "BaseDistributionList",
    "EmpiricalDistribution",
    "MultivariateGaussian",
    "multiscaler",
]


