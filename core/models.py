"""
Model architectures for Koopman eigenfunction learning.

This module contains various neural network architectures that can be used
to learn Koopman eigenfunctions for separatrix location.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Union


class KoopmanEigenfunctionModel(nn.Module):
    """Basic neural network for learning Koopman eigenfunctions."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class ResNet(nn.Module):
    """Residual network for Koopman eigenfunction learning.
    
    This model uses residual connections across layers. Despite the previous
    name, it does not rely on explicit RBF layers.
    """
    
    def __init__(
        self, 
        input_dim: int,
        hidden_size: int = 400,
        num_layers: int = 20,
        num_kernels: int = 10,
        output_dim: int = 1,
        input_scale_factor: float = 1.0,
        input_center: Optional[torch.Tensor] = None,
        scale_last_layer_by_inv_sqrt_hidden: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_kernels = num_kernels
        self.output_dim = output_dim
        
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

# Backward-compatibility alias
ResRBF = ResNet


class LinearModel(nn.Module):
    """Simple linear model for baseline comparisons."""
    
    def __init__(self, input_dim: int, output_dim: int = 1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return self.linear(x)


class DeepKoopmanModel(nn.Module):
    """Deep neural network with more sophisticated architecture for complex systems."""
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: List[int] = [512, 256, 128],
        output_dim: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)
