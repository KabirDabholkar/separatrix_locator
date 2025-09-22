"""
Model architectures for Koopman eigenfunction learning.

This module contains various neural network architectures that can be used
to learn Koopman eigenfunctions for separatrix location.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union


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
        
        # Build the network layers
        layers = []
        
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_size))
        
        # Hidden layers with residual connections (standard linear blocks)
        for i in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
        
        # Output layer
        output_layer = nn.Linear(hidden_size, output_dim)
        if scale_last_layer_by_inv_sqrt_hidden:
            output_layer.weight.data *= (1.0 / (hidden_size ** 0.5))
        layers.append(output_layer)
        
        self.network = nn.ModuleList(layers)
        self.activation = nn.ReLU()
        
    def forward(self, x):
        # Input preprocessing
        x = (x - self.input_center) * self.input_scale_factor
        
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
            f"smalllastlayer{self.scale_last_layer_by_inv_sqrt_hidden}"
        )