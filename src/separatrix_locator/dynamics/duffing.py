"""
Duffing oscillator system.

The Duffing oscillator is a classic nonlinear dynamical system that exhibits
rich bifurcation behavior and is commonly used in separatrix location studies.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple

from .base import DynamicalSystem


class DuffingOscillator(DynamicalSystem):
    """
    2D Duffing oscillator system:
    dx/dt = y
    dy/dt = -y + x - x^3
    
    This system has:
    - Two stable fixed points at (±1, 0)
    - One unstable fixed point at (0, 0) (the separatrix)
    - Damping term that creates spiral behavior
    """
    
    def __init__(self, damping: float = 1.0):
        super().__init__(dim=2, name="DuffingOscillator")
        self.damping = damping
    
    @property
    def name(self) -> str:
        """Always include fixed dimension in the name."""
        return f"DuffingOscillator{self.dim}D"

    @name.setter
    def name(self, value: Optional[str]) -> None:
        # Ignore external name assignments; keep name dynamic
        return
    
    def function(self, x: torch.Tensor) -> torch.Tensor:
        """dx/dt = [y, -damping*y + x - x^3]"""
        x_coord = x[..., 0:1]
        y_coord = x[..., 1:2]
        
        dx_dt = y_coord
        dy_dt = -self.damping * y_coord + x_coord - x_coord**3
        
        return torch.cat([dx_dt, dy_dt], dim=-1)
    
    def get_attractors(self) -> torch.Tensor:
        """Get the stable fixed points at (±1, 0)"""
        return (torch.tensor([1.0, 0.0], dtype=torch.float32),
                torch.tensor([-1.0, 0.0], dtype=torch.float32))
    
    def get_separatrix(self) -> torch.Tensor:
        """Get the unstable fixed point (separatrix) at (0, 0)"""
        return torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    


if __name__ == "__main__":
    # Basic shape tests
    system = DuffingOscillator(damping=0.7)
    batch_size = 4
    x = torch.randn(batch_size, system.dim)
    fx = system.function(x)
    assert isinstance(fx, torch.Tensor), "function(x) must return a torch.Tensor"
    assert fx.shape == x.shape, f"Expected shape {x.shape}, got {fx.shape}"

    # Name dynamic behavior reflects fixed 2D
    assert system.name == "DuffingOscillator2D", f"Unexpected name: {system.name}"

    # Integration test (skips if torchdiffeq not installed)
    try:
        t_span = torch.linspace(0.0, 1.0, steps=21)
        x0 = torch.randn(batch_size, system.dim)
        traj = system.integrate(x0, t_span)
        assert traj.shape == (t_span.numel(), batch_size, system.dim), (
            f"Expected trajectory shape {(t_span.numel(), batch_size, system.dim)}, got {traj.shape}"
        )
        print("Duffing integrate: OK")
    except ImportError:
        print("torchdiffeq not installed; skipping integrate test for DuffingOscillator")

    print("DuffingOscillator tests passed.")
