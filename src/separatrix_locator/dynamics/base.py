"""
Base class for dynamical systems.
"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from separatrix_locator.utils import get_estimate_attractor_func


class DynamicalSystem(ABC):
    """
    Abstract base class for dynamical systems.
    
    All dynamical systems should inherit from this class and implement
    the required methods.
    """
    
    def __init__(self, dim: int, name: str):
        self.dim = dim
        if not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("name must be a non-empty string")
        # Note: subclasses may override `name` as a property; in that case
        # this assignment may be ignored by the subclass' setter.
        self.name = name
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.function(x)

    @abstractmethod
    def function(self, x: torch.Tensor) -> torch.Tensor:
        """
        The dynamical system function f(x) -> dx/dt.
        
        Parameters:
        -----------
        x : torch.Tensor
            State vector of shape (batch_size, dim)
            
        Returns:
        --------
        torch.Tensor
            Time derivative dx/dt of shape (batch_size, dim)
        """
        pass
    
    def get_attractors(self) -> Optional[torch.Tensor]:
        """
        Default implementation: estimate attractors by clustering final states of
        trajectories integrated from random initial conditions. Subclasses with
        analytical attractors should override this method.

        Returns
        -------
        Optional[torch.Tensor]
            Attractor points of shape (num_attractors, dim) if defined, else None
        """
        try:
            estimate_attractors = get_estimate_attractor_func(self.function)
            a1, a2 = estimate_attractors(self.dim)
            return (a1, a2)
        except Exception as exc:
            raise NotImplementedError("This system does not define attractors") from exc
    
    def sample_initial_conditions(self, distribution: torch.distributions.Distribution, batch_size: int) -> torch.Tensor:
        """
        Sample initial conditions from a provided distribution.
        
        Parameters:
        -----------
        distribution : torch.distributions.Distribution
            Distribution to sample initial conditions from. Any object with a
            `sample(sample_shape)` method returning a tensor is supported.
        batch_size : int
            Number of samples to generate
            
        Returns:
        --------
        torch.Tensor
            Initial conditions of shape (batch_size, dim)
        """
        return distribution.sample((batch_size,))
    
    def integrate(self, x0: torch.Tensor, t_span: torch.Tensor) -> torch.Tensor:
        """
        Integrate the system from initial conditions.
        
        Parameters:
        -----------
        x0 : torch.Tensor
            Initial conditions of shape (batch_size, dim)
        t_span : torch.Tensor
            Time points for integration
            
        Returns:
        --------
        torch.Tensor
            Trajectories of shape (len(t_span), batch_size, dim)
        """
        try:
            from torchdiffeq import odeint
        except ImportError:
            raise ImportError("torchdiffeq is required for integration. Install with: pip install torchdiffeq")
        
        def ode_func(t, x):
            return self.function(x)
        
        return odeint(ode_func, x0, t_span)
    
    def __str__(self) -> str:
        return f"{self.name}(dim={self.dim})"
    
    def __repr__(self) -> str:
        return self.__str__()
