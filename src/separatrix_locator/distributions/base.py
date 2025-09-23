"""
Base class for sampling distributions used to generate initial conditions
or noise for dynamical systems.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

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


class BaseDistributionList:
    """
    A container class for holding a list of BaseDistribution instances.
    
    Args:
        distributions: List of BaseDistribution instances
        name: Optional name for the list. If None, generates a name based on the distributions
    """
    
    def __init__(
        self, 
        distributions: List[BaseDistribution], 
        name: Optional[str] = None
    ):
        if not isinstance(distributions, list):
            raise TypeError("distributions must be a list")
        if not distributions:
            raise ValueError("distributions list cannot be empty")
        
        # Validate that all items are BaseDistribution instances
        for i, dist in enumerate(distributions):
            if not isinstance(dist, BaseDistribution):
                raise TypeError(f"distributions[{i}] must be a BaseDistribution instance, got {type(dist)}")
        
        self._distributions = distributions
        
        # Generate name if not provided
        if name is None:
            # Get unique distribution types
            dist_types = set(type(dist).__name__ for dist in distributions)
            if len(dist_types) == 1:
                dist_type = list(dist_types)[0]
                self._name = f"{dist_type}List{len(distributions)}"
            else:
                self._name = f"DistributionList{len(distributions)}"
        else:
            self._name = name
    
    @property
    def name(self) -> str:
        """Name of the distribution list."""
        return self._name
    
    @name.setter
    def name(self, value: str) -> None:
        if not isinstance(value, str) or len(value.strip()) == 0:
            raise ValueError("name must be a non-empty string")
        self._name = value
    
    @property
    def distributions(self) -> List[BaseDistribution]:
        """List of BaseDistribution instances."""
        return self._distributions
    
    @property
    def dim(self) -> int:
        """Dimensionality of the distributions (assumes all have same dim)."""
        if not self._distributions:
            raise ValueError("Cannot get dim from empty distribution list")
        return self._distributions[0].dim
    
    def __len__(self) -> int:
        """Number of distributions in the list."""
        return len(self._distributions)
    
    def __getitem__(self, index: int) -> BaseDistribution:
        """Get distribution by index."""
        return self._distributions[index]
    
    def __iter__(self):
        """Iterate over distributions."""
        return iter(self._distributions)
    
    def __repr__(self) -> str:
        return f"BaseDistributionList(name={self.name!r}, len={len(self)}, dim={self.dim})"


