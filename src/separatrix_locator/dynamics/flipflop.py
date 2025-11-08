"""
Flip-flop dynamical systems.

These are discrete-state systems that can switch between different stable states,
commonly used in neural circuit modeling and decision-making studies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import DynamicalSystem


class FlipFlop1Bit2D(DynamicalSystem):
    """
    2D 1-bit flip-flop system.
    
    This system models a simple binary decision-making circuit with two stable states
    representing different decisions.
    """
    
    def __init__(self, tau: float = 10.0):
        super().__init__(dim=2, name="FlipFlop1Bit2D")
        self.tau = tau  # Time constant
    
    def function(self, x: torch.Tensor) -> torch.Tensor:
        """dx/dt = (1/tau) * (-x + tanh(x + y))"""
        x_coord = x[..., 0:1]
        y_coord = x[..., 1:2]
        
        # Simple flip-flop dynamics
        dx_dt = (1.0 / self.tau) * (-x_coord + torch.tanh(x_coord + y_coord))
        dy_dt = (1.0 / self.tau) * (-y_coord + torch.tanh(x_coord - y_coord))
        
        return torch.cat([dx_dt, dy_dt], dim=-1)
    
    def get_attractors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the stable fixed points (approximate locations)"""
        # These are approximate - the exact locations depend on the tanh nonlinearity
        attractor1 = torch.tensor([1.0, -1.0], dtype=torch.float32)
        attractor2 = torch.tensor([-1.0, 1.0], dtype=torch.float32)
        return (attractor1, attractor2)
    
    def get_separatrix(self) -> torch.Tensor:
        """Get the unstable fixed point (separatrix) at (0, 0)"""
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)


class FlipFlop2Bit2D(DynamicalSystem):
    """
    2D 2-bit flip-flop system.
    
    This system models a more complex decision-making circuit with multiple
    possible stable states.
    """
    
    def __init__(self, tau: float = 10.0):
        super().__init__(dim=2, name="FlipFlop2Bit2D")
        self.tau = tau
    
    def function(self, x: torch.Tensor) -> torch.Tensor:
        """More complex 2-bit flip-flop dynamics"""
        x_coord = x[..., 0:1]
        y_coord = x[..., 1:2]
        
        # 2-bit flip-flop with coupling
        dx_dt = (1.0 / self.tau) * (-x_coord + torch.tanh(2*x_coord - y_coord))
        dy_dt = (1.0 / self.tau) * (-y_coord + torch.tanh(-x_coord + 2*y_coord))
        
        return torch.cat([dx_dt, dy_dt], dim=-1)
    
    def get_attractors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get the stable fixed points (approximate locations)"""
        # These are approximate - the exact locations depend on the tanh nonlinearity
        attractor1 = torch.tensor([1.0, 1.0], dtype=torch.float32)
        attractor2 = torch.tensor([-1.0, -1.0], dtype=torch.float32)
        return (attractor1, attractor2)
    
    def get_separatrix(self) -> torch.Tensor:
        """Get the unstable fixed point (separatrix) at (0, 0)"""
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)
