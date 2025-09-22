"""
Base class for sampling distributions used to generate initial conditions
or noise for dynamical systems.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Union

import torch


class BaseDistribution(ABC):
    """
    Abstract base class for separatrix-locator distributions.

    Requirements:
    - dim: dimensionality of samples produced
    - name: human-readable identifier for the distribution
    - sample(): produce samples as a tensor with trailing dimension dim
    """

    def __init__(self, dim: int, name: str):
        if not isinstance(dim, int) or dim < 1:
            raise ValueError("dim must be a positive integer")
        if not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("name must be a non-empty string")
        self._dim = int(dim)
        self._name = name

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def sample(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        """
        Generate samples.

        Args:
            sample_shape: Optional leading shape for number of samples
                          (e.g., 10, (batch_size,), or (num_batches, batch_size)).
        Returns:
            Tensor of shape (*sample_shape, dim)
        """
        raise NotImplementedError

    def __call__(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        return self.sample(sample_shape)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, dim={self.dim})"


