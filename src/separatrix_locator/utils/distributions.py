"""
Distribution utilities for separatrix location.

This module provides various distributions for sampling initial conditions
and external inputs for dynamical systems. (Deprecated: moved to separatrix_locator.distributions)
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple
from torch.distributions import Distribution, Normal, Uniform, MultivariateNormal


class IsotropicGaussian(Distribution):
    """
    Isotropic Gaussian distribution with multiple centers and scales.
    
    This is useful for sampling initial conditions around multiple
    attractors or regions of interest.
    """
    
    def __init__(self, means: torch.Tensor, scales: torch.Tensor):
        """
        Parameters:
        -----------
        means : torch.Tensor
            Centers of the Gaussians, shape (num_centers, dim)
        scales : torch.Tensor
            Scales for each center, shape (num_centers,)
        """
        self.means = means
        self.scales = scales
        self.num_centers = means.shape[0]
        self.dim = means.shape[1]
        
        # Create individual normal distributions
        self.distributions = [
            MultivariateNormal(
                loc=self.means[i], 
                covariance_matrix=self.scales[i] * torch.eye(self.dim)
            )
            for i in range(self.num_centers)
        ]
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the mixture distribution."""
        # Sample which component to use
        component_indices = torch.randint(0, self.num_centers, sample_shape)
        
        # Sample from each component
        samples = []
        for i in range(self.num_centers):
            mask = component_indices == i
            if mask.any():
                component_samples = self.distributions[i].sample((mask.sum(),))
                samples.append(component_samples)
        
        if samples:
            return torch.cat(samples, dim=0)
        else:
            return torch.empty((*sample_shape, self.dim))
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability under the mixture."""
        log_probs = []
        for dist in self.distributions:
            log_probs.append(dist.log_prob(value))
        
        # Take the maximum (approximation for mixture)
        return torch.stack(log_probs, dim=0).max(dim=0)[0]


class MultiGapNormal(Distribution):
    """
    Normal distribution with gaps at specified points.
    
    This is useful for sampling initial conditions while avoiding
    certain regions (like separatrices).
    """
    
    def __init__(
        self, 
        gap_points: List[float], 
        epsilon: float = 1e-3,
        loc: float = 0.0,
        scale: float = 1.0,
        dim: int = 1
    ):
        """
        Parameters:
        -----------
        gap_points : list
            Points where sampling should be avoided
        epsilon : float
            Width of the gaps around gap_points
        loc : float
            Mean of the underlying normal distribution
        scale : float
            Standard deviation of the underlying normal distribution
        dim : int
            Dimensionality of the distribution
        """
        self.gap_points = gap_points
        self.epsilon = epsilon
        self.loc = loc
        self.scale = scale
        self.dim = dim
        
        # Base normal distribution
        if dim == 1:
            self.base_dist = Normal(loc=loc, scale=scale)
        else:
            self.base_dist = MultivariateNormal(
                loc=torch.full((dim,), loc),
                covariance_matrix=scale * torch.eye(dim)
            )
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the distribution, rejecting points in gaps."""
        max_attempts = 1000
        samples = []
        
        for _ in range(max_attempts):
            if self.dim == 1:
                candidate = self.base_dist.sample(sample_shape)
            else:
                candidate = self.base_dist.sample(sample_shape)
            
            # Check if candidate is in any gap
            in_gap = False
            for gap_point in self.gap_points:
                if self.dim == 1:
                    if torch.any(torch.abs(candidate - gap_point) < self.epsilon):
                        in_gap = True
                        break
                else:
                    # For multi-dimensional, check first dimension
                    if torch.any(torch.abs(candidate[..., 0] - gap_point) < self.epsilon):
                        in_gap = True
                        break
            
            if not in_gap:
                samples.append(candidate)
                if len(samples) >= sample_shape[0] if sample_shape else 1:
                    break
        
        if samples:
            return torch.cat(samples, dim=0)[:sample_shape[0] if sample_shape else 1]
        else:
            # Fallback to base distribution if no valid samples found
            return self.base_dist.sample(sample_shape)


def make_iid_multivariate(dist: Distribution, dim: int) -> Distribution:
    """
    Create an IID multivariate distribution from a 1D distribution.
    
    Parameters:
    -----------
    dist : torch.distributions.Distribution
        Base 1D distribution
    dim : int
        Dimensionality of the resulting multivariate distribution
        
    Returns:
    --------
    torch.distributions.Distribution
        IID multivariate distribution
    """
    if isinstance(dist, Normal):
        return MultivariateNormal(
            loc=torch.full((dim,), dist.loc),
            covariance_matrix=dist.scale * torch.eye(dim)
        )
    elif isinstance(dist, Uniform):
        return Uniform(
            low=torch.full((dim,), dist.low),
            high=torch.full((dim,), dist.high)
        )
    else:
        raise ValueError(f"Unsupported distribution type: {type(dist)}")


def create_separatrix_avoiding_distribution(
    separatrix_points: torch.Tensor,
    base_dist: Distribution,
    epsilon: float = 0.1
) -> Distribution:
    """
    Create a distribution that avoids sampling near separatrix points.
    
    Parameters:
    -----------
    separatrix_points : torch.Tensor
        Points to avoid, shape (num_points, dim)
    base_dist : torch.distributions.Distribution
        Base distribution to modify
    epsilon : float
        Minimum distance from separatrix points
        
    Returns:
    --------
    torch.distributions.Distribution
        Modified distribution that avoids separatrix points
    """
    # This is a simplified implementation
    # In practice, you might want to use rejection sampling or other methods
    
    if isinstance(base_dist, Normal):
        gap_points = separatrix_points[:, 0].tolist()  # Use first dimension
        return MultiGapNormal(
            gap_points=gap_points,
            epsilon=epsilon,
            loc=base_dist.loc,
            scale=base_dist.scale,
            dim=1
        )
    else:
        # For other distributions, return the base distribution
        return base_dist


class MixtureDistribution(Distribution):
    """
    General mixture distribution for combining multiple distributions.
    """
    
    def __init__(self, distributions: List[Distribution], weights: Optional[torch.Tensor] = None):
        """
        Parameters:
        -----------
        distributions : list
            List of distributions to mix
        weights : torch.Tensor, optional
            Mixing weights. If None, uniform weights are used.
        """
        self.distributions = distributions
        self.num_components = len(distributions)
        
        if weights is None:
            self.weights = torch.ones(self.num_components) / self.num_components
        else:
            self.weights = weights / weights.sum()  # Normalize weights
    
    def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
        """Sample from the mixture distribution."""
        # Sample which component to use
        component_indices = torch.multinomial(self.weights, sample_shape[0], replacement=True)
        
        # Sample from each component
        samples = []
        for i in range(self.num_components):
            mask = component_indices == i
            if mask.any():
                component_samples = self.distributions[i].sample((mask.sum(),))
                samples.append(component_samples)
        
        if samples:
            return torch.cat(samples, dim=0)
        else:
            return torch.empty((*sample_shape, self.distributions[0].event_shape[0]))
    
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        """Compute log probability under the mixture."""
        log_probs = []
        for i, dist in enumerate(self.distributions):
            log_probs.append(dist.log_prob(value) + torch.log(self.weights[i]))
        
        # Log-sum-exp for numerical stability
        log_probs = torch.stack(log_probs, dim=0)
        return torch.logsumexp(log_probs, dim=0)
