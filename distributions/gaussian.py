"""
Multivariate Gaussian distribution compatible with BaseDistribution.
"""

from typing import Optional, Tuple, Union

import torch

from .base import BaseDistribution


class MultivariateGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with mean and covariance.

    Args:
        dim: Dimensionality of the distribution
        mean: Optional tensor of shape (dim,). Defaults to zeros
        covariance_matrix: Optional PSD tensor of shape (dim, dim). Defaults to identity
    """

    def __init__(
        self,
        dim: int,
        mean: Optional[torch.Tensor] = None,
        covariance_matrix: Optional[torch.Tensor] = None,
        name: Optional[str] = None,
    ):
        # Pass a placeholder name; our property makes it dynamic and ignores assignments
        super().__init__(dim=dim, name=name or "MultivariateGaussian")

        if mean is None:
            mean = torch.zeros(dim, dtype=torch.float32)
        if mean.shape != (dim,):
            raise ValueError(f"mean must have shape ({dim},), got {tuple(mean.shape)}")

        if covariance_matrix is None:
            covariance_matrix = torch.eye(dim, dtype=mean.dtype, device=mean.device)
        if covariance_matrix.shape != (dim, dim):
            raise ValueError(
                f"covariance_matrix must have shape ({dim}, {dim}), got {tuple(covariance_matrix.shape)}"
            )

        self._mean = mean
        self._covariance_matrix = covariance_matrix

    @property
    def name(self) -> str:
        """Dynamic name including dimensionality (e.g., "MultivariateGaussian3D")."""
        return f"MultivariateGaussian{self.dim}D"

    @name.setter
    def name(self, value: Optional[str]) -> None:
        # Ignore external name assignments; keep name dynamic
        return

    @property
    def mean(self) -> torch.Tensor:
        return self._mean

    @property
    def covariance_matrix(self) -> torch.Tensor:
        return self._covariance_matrix

    def _torch_distribution(self) -> torch.distributions.MultivariateNormal:
        return torch.distributions.MultivariateNormal(
            loc=self._mean, covariance_matrix=self._covariance_matrix
        )

    def sample(self, sample_shape: Union[int, Tuple[int, ...]] = ()) -> torch.Tensor:
        if isinstance(sample_shape, int):
            sample_shape = (sample_shape,)
        return self._torch_distribution().sample(sample_shape)