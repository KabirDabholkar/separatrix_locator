
from typing import Optional, Tuple, Union

import torch

from .base import BaseDistribution


class EmpiricalDistribution(BaseDistribution):
    """
    Empirical distribution that samples with replacement from a given dataset.

    Args:
        dataset: A tensor of shape (num_samples, dim) containing the dataset.
        name: Optional name for the distribution.
    """

    def __init__(self, dataset: torch.Tensor, name: Optional[str] = None):
        if not isinstance(dataset, torch.Tensor):
            raise TypeError("dataset must be a torch.Tensor")
        if dataset.dim() != 2:
            raise ValueError(f"dataset must be 2D, got shape {dataset.shape}")

        self.dataset = dataset
        self.num_samples = dataset.shape[0]
        dim = dataset.shape[1]

        super().__init__(dim=dim, name=name or "EmpiricalDistribution")

    def sample(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        """
        Generate samples by resampling with replacement from the dataset.

        Args:
            sample_shape: Optional leading shape for number of samples.

        Returns:
            Tensor of shape (*sample_shape, dim)
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        # Calculate total number of samples needed
        num_requested = 1
        for s in sample_shape:
            num_requested *= s

        # Sample indices with replacement
        indices = torch.randint(
            0, self.num_samples, (num_requested,), device=self.dataset.device
        )

        # Gather samples
        samples = self.dataset[indices]

        # Reshape to match requested shape + (dim,)
        return samples.view(*sample_shape, self.dim)


class EmpiricalNoisyDistribution(EmpiricalDistribution):
    """
    Empirical distribution that samples with replacement from a given dataset
    and adds isotropic Gaussian noise to the samples.

    Args:
        dataset: A tensor of shape (num_samples, dim) containing the dataset.
        noise_scale: Standard deviation of the isotropic Gaussian noise.
        name: Optional name for the distribution.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        noise_scale: float,
        name: Optional[str] = None,
    ):
        super().__init__(dataset, name=name or "EmpiricalNoisyDistribution")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        self.noise_scale = noise_scale

    def sample(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        """
        Generate samples by resampling with replacement from the dataset
        and adding isotropic Gaussian noise.

        Args:
            sample_shape: Optional leading shape for number of samples.

        Returns:
            Tensor of shape (*sample_shape, dim)
        """
        # Get base samples from the empirical distribution
        samples = super().sample(sample_shape)

        # Add noise
        noise = torch.randn_like(samples) * self.noise_scale
        return samples + noise


class EmpiricalConvexCombinationDistribution(EmpiricalDistribution):
    """
    Empirical distribution over convex combinations of random pairs of points
    from the dataset.

    Each sample has the form
        x = t * x_i + (1 - t) * x_j
    where x_i and x_j are sampled (with replacement) from the dataset and
    t ~ Uniform(0, 1) independently for each sample.

    Args:
        dataset: A tensor of shape (num_samples, dim) containing the dataset.
        noise_scale: Optional standard deviation of the isotropic Gaussian
            noise added on top of the convex combinations. If 0.0, no noise is
            added.
        name: Optional name for the distribution.
    """

    def __init__(
        self,
        dataset: torch.Tensor,
        noise_scale: float = 0.0,
        name: Optional[str] = None,
    ):
        super().__init__(dataset, name=name or "EmpiricalConvexCombinationDistribution")
        if noise_scale < 0:
            raise ValueError("noise_scale must be non-negative")
        self.noise_scale = noise_scale

    def sample(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        """
        Generate samples by taking convex combinations of random pairs of
        points from the dataset.

        Args:
            sample_shape: Optional leading shape for number of samples.

        Returns:
            Tensor of shape (*sample_shape, dim)
        """
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)

        # Total number of convex-combination samples needed
        num_requested = 1
        for s in sample_shape:
            num_requested *= s

        device = self.dataset.device

        # Sample two sets of indices (with replacement) for the pairs
        idx1 = torch.randint(0, self.num_samples, (num_requested,), device=device)
        idx2 = torch.randint(0, self.num_samples, (num_requested,), device=device)

        x1 = self.dataset[idx1]
        x2 = self.dataset[idx2]

        # Sample mixing coefficients t ~ Uniform(0, 1)
        t = torch.rand(num_requested, 1, device=device)

        samples = t * x1 + (1.0 - t) * x2

        # Optionally add isotropic Gaussian noise
        if self.noise_scale > 0.0:
            noise = torch.randn_like(samples) * self.noise_scale
            samples = samples + noise

        return samples.view(*sample_shape, self.dim)
