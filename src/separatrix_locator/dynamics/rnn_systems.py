"""
Recurrent neural network dynamical systems.

These systems model neural circuits and decision-making networks,
often exhibiting complex separatrix structures.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any

from .base import DynamicalSystem


class RNNDecisionMaking(DynamicalSystem):
    """
    RNN-based decision-making system.
    
    This system models a recurrent neural network that performs binary decision-making,
    with separatrix structures that separate different decision outcomes.
    """
    
    def __init__(
        self, 
        hidden_size: int = 64,
        tau: float = 10.0,
        input_strength: float = 1.0
    ):
        super().__init__(dim=hidden_size, name="RNNDecisionMaking")
        self.hidden_size = hidden_size
        self.tau = tau
        self.input_strength = input_strength
        
        # Simple RNN weights (can be replaced with more sophisticated architectures)
        self.W_rec = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.1)
        self.W_in = nn.Parameter(torch.randn(2, hidden_size) * input_strength)  # 2 inputs for binary choice
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        
        # Initialize weights
        nn.init.orthogonal_(self.W_rec)
        
    
    def function(self, x: torch.Tensor) -> torch.Tensor:
        """RNN dynamics with external input"""
        # For simplicity, assume no external input in this base implementation
        # In practice, external inputs would be passed as additional parameters
        
        # RNN dynamics: dx/dt = (1/tau) * (-x + tanh(W_rec @ x + W_in @ u + bias))
        # For now, assume u = 0 (no external input)
        u = torch.zeros(x.shape[0], 2, device=x.device)  # No external input
        
        recurrent_input = torch.matmul(x, self.W_rec.T)
        input_contribution = torch.matmul(u, self.W_in.T)
        
        total_input = recurrent_input + input_contribution + self.bias
        dx_dt = (1.0 / self.tau) * (-x + torch.tanh(total_input))
        
        return dx_dt
    
    def function_with_input(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """RNN dynamics with explicit external input"""
        recurrent_input = torch.matmul(x, self.W_rec.T)
        input_contribution = torch.matmul(u, self.W_in.T)
        
        total_input = recurrent_input + input_contribution + self.bias
        dx_dt = (1.0 / self.tau) * (-x + torch.tanh(total_input))
        
        return dx_dt
    
    def get_attractors(self) -> torch.Tensor:
        """Get approximate attractor locations (these would need to be computed numerically)"""
        # For RNNs, attractors are typically computed numerically
        # Here we return approximate locations based on the network structure
        attractor1 = torch.ones(self.hidden_size) * 0.5
        attractor2 = torch.ones(self.hidden_size) * -0.5
        return torch.stack([attractor1, attractor2])
    
    def get_separatrix(self) -> torch.Tensor:
        """Get the separatrix (typically near the origin for RNNs)"""
        return torch.zeros(1, self.hidden_size)
    
    def set_external_input(self, input_value: float):
        """Set a constant external input for the decision-making task"""
        self.external_input = input_value
        
    def function_with_external_input(self, x: torch.Tensor, external_input: float = None) -> torch.Tensor:
        """Dynamics function with external input"""
        if external_input is None:
            external_input = getattr(self, 'external_input', 0.0)
        
        # Add external input to the first component
        x_modified = x.clone()
        x_modified[:, 0] += external_input
        
        return self.function(x_modified)
    
    def compute_fixed_points(self, num_points: int = 1000, max_iterations: int = 1000) -> torch.Tensor:
        """
        Compute fixed points of the RNN system numerically.
        
        This is a simplified implementation - in practice, you might use
        more sophisticated fixed point finding algorithms.
        """
        fixed_points = []
        
        for _ in range(num_points):
            # Start from random initial condition
            x = torch.randn(self.hidden_size) * 0.1
            
            # Iterate until convergence
            for _ in range(max_iterations):
                x_new = torch.tanh(torch.matmul(x, self.W_rec.T) + self.bias)
                if torch.norm(x_new - x) < 1e-6:
                    fixed_points.append(x_new)
                    break
                x = x_new
        
        if fixed_points:
            return torch.stack(fixed_points)
        else:
            return torch.zeros(1, self.hidden_size)
