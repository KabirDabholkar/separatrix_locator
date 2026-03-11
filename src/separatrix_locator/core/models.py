"""
Model architectures for Koopman eigenfunction learning.

This module contains various neural network architectures that can be used
to learn Koopman eigenfunctions for separatrix location.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union, Tuple


class ResNet(nn.Module):
    """Residual network for Koopman eigenfunction learning.
    
    This model uses residual connections across layers.
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_size: int = 400,
        num_layers: int = 20,
        output_dim: int = 1,
        input_scale_factor: float = 1.0,
        input_center: Optional[torch.Tensor] = None,
        prewhitening_matrix: Optional[torch.Tensor] = None,
        scale_last_layer_by_inv_sqrt_hidden: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.scale_last_layer_by_inv_sqrt_hidden = scale_last_layer_by_inv_sqrt_hidden
        
        # Input preprocessing
        self.input_scale_factor = input_scale_factor
        if input_center is not None:
            self.register_buffer('input_center', input_center)
        else:
            self.register_buffer('input_center', torch.zeros(input_dim))

        # Optional prewhitening matrix (non-trainable, precomputed)
        if prewhitening_matrix is not None:
            if prewhitening_matrix.shape != (input_dim, input_dim):
                raise ValueError(
                    f"prewhitening_matrix must have shape ({input_dim}, {input_dim}), "
                    f"got {prewhitening_matrix.shape}"
                )
            self.register_buffer("prewhitening_matrix", prewhitening_matrix)
            self._has_prewhitening = True
        else:
            self.register_buffer("prewhitening_matrix", torch.eye(input_dim))
            self._has_prewhitening = False
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        
        # Hidden layers with residual connections (standard linear blocks)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        output_layer = nn.Linear(hidden_size, output_dim)
        layers.append(output_layer)
        
        self.network = nn.ModuleList(layers)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        # Input preprocessing
        x = (x - self.input_center) * self.input_scale_factor
        # Apply fixed prewhitening transform (premultiply by square matrix)
        # Using row-vector convention: x' = x W^T
        if self.prewhitening_matrix is not None:
            x = torch.matmul(x, self.prewhitening_matrix.t())
        
        # Forward through network with residual connections
        h = x
        for i, layer in enumerate(self.network):
            if i == 0:
                # First layer
                h = self.activation(layer(h))
            elif i < len(self.network) - 1:
                # Hidden layers with residual connection
                residual = h
                h = self.activation(layer(h))
                h = h + residual  # Residual connection
            else:
                # Output layer (no activation)
                h = layer(h)
                # Apply scaling during forward pass if enabled
                if self.scale_last_layer_by_inv_sqrt_hidden:
                    h = h * (1.0 / (self.hidden_size ** 0.5))
        
        return h

    @property
    def name(self) -> str:
        """Descriptive model name encoding all constructor hyperparameters."""
        # Convert center to a standard Python list for stable, readable formatting
        center_list = (
            self.input_center.detach().cpu().tolist()
            if isinstance(self.input_center, torch.Tensor)
            else None
        )
        has_nonzero_center = (
            any(x != 0 for x in center_list)
            if center_list is not None
            else False
        )
        return (
            f"ResNet_"
            f"inputdim{self.input_dim}_"
            f"hiddensize{self.hidden_size}_"
            f"numlayers{self.num_layers}_"
            f"outputdim{self.output_dim}_"
            f"inputscalefactor{self.input_scale_factor}_"
            f"hasnonzero_center{has_nonzero_center}_"
            f"smalllastlayer{self.scale_last_layer_by_inv_sqrt_hidden}_"
            f"hasprewhitening{self._has_prewhitening}"
        )


def compute_input_center_and_whitening(
    dataset: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute an input mean (for `input_center`) and a whitening matrix
    suitable for use with `ResNet(prewhitening_matrix=...)`.

    The whitening matrix W is defined such that, for centered data x,
        x_whitened = x @ W^T
    has approximately identity covariance.

    Args:
        dataset: Tensor of shape (num_samples, dim) containing the data.
        eps: Small regularization added to eigenvalues to avoid division
            by zero when inverting the square root of the covariance.

    Returns:
        input_center: Tensor of shape (dim,) with the empirical mean.
        prewhitening_matrix: Tensor of shape (dim, dim) with the
            whitening transform.
    """
    if not isinstance(dataset, torch.Tensor):
        raise TypeError("dataset must be a torch.Tensor")
    if dataset.dim() != 2:
        raise ValueError(f"dataset must be 2D, got shape {dataset.shape}")

    # Empirical mean
    input_center = dataset.mean(dim=0)

    # Empirical covariance (unbiased)
    centered = dataset - input_center
    num_samples = centered.shape[0]
    if num_samples <= 1:
        raise ValueError("dataset must contain at least two samples to compute covariance")

    cov = centered.t().matmul(centered) / (num_samples - 1)

    # Symmetric eigendecomposition
    eigvals, eigvecs = torch.linalg.eigh(cov)

    # Inverse square root of eigenvalues with regularization
    inv_sqrt_eigvals = (eigvals + eps).clamp_min(eps).rsqrt()

    # Whitening matrix W such that cov_whitened ≈ I
    # W = diag(inv_sqrt_eigvals) @ eigvecs.T
    prewhitening_matrix = torch.diag(inv_sqrt_eigvals).matmul(eigvecs.t())

    return input_center, prewhitening_matrix