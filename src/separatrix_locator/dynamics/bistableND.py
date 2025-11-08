"""
General N-dimensional bistable dynamical system.

One axis (configurable) is bistable with dynamics x - x^3, while all other
axes are linearly stable with dynamics -x.
"""

import torch
from typing import Optional, Tuple

from .base import DynamicalSystem


class BistableND(DynamicalSystem):
    """
    N-dimensional bistable system.

    Dynamics:
    - On the bistable axis i = `bistable_axis`: dx_i/dt = x_i - x_i^3
    - On all other axes j != `bistable_axis`: dx_j/dt = -x_j

    This system has:
    - Two stable fixed points at (+/-1 on the bistable axis, 0 elsewhere)
    - One unstable fixed point at the origin (the separatrix)
    """

    def __init__(self, dim: int, bistable_axis: int = 0):
        if dim < 1:
            raise ValueError("dim must be >= 1")
        if not (0 <= bistable_axis < dim):
            raise ValueError("bistable_axis must be in [0, dim)")

        # Pass a placeholder name; our property makes it dynamic and ignores assignments
        super().__init__(dim=dim, name="BistableND")
        self.bistable_axis = int(bistable_axis)

    @property
    def name(self) -> str:
        """Name of the system.

        Always dynamic: includes the current dimension and bistable axis
        (e.g., "Bistable3D_axis0").
        """
        return f"Bistable{self.dim}D_axis{self.bistable_axis}"

    @name.setter
    def name(self, value: Optional[str]) -> None:
        # Ignore external name assignments; keep name dynamic
        return

    def function(self, x: torch.Tensor) -> torch.Tensor:
        """Compute f(x) for the N-D system.

        Args:
            x: Tensor of shape (..., dim)

        Returns:
            Tensor of shape (..., dim) with the vector field values.
        """
        # Ensure last dimension is state dimension
        if x.shape[-1] != self.dim:
            raise ValueError(f"Expected input last dim {self.dim}, got {x.shape[-1]}")

        # Create output initialized as linear stable dynamics: -x
        fx = -x.clone()

        # Replace the bistable axis with x - x^3
        xa = x[..., self.bistable_axis:self.bistable_axis + 1]
        fx[..., self.bistable_axis:self.bistable_axis + 1] = xa - xa ** 3
        return fx

    def get_attractors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return the two stable fixed points.

        (+1 at bistable axis, 0 elsewhere) and (-1 at bistable axis, 0 elsewhere)
        Returns a tuple of two tensors, each of shape (dim,)
        """
        plus = torch.zeros(self.dim, dtype=torch.float32)
        minus = torch.zeros(self.dim, dtype=torch.float32)
        plus[self.bistable_axis] = 1.0
        minus[self.bistable_axis] = -1.0
        return (plus, minus)



if __name__ == "__main__":
    # Basic shape tests
    system = BistableND(dim=3, bistable_axis=1)
    batch_size = 5
    x = torch.randn(batch_size, system.dim)
    fx = system.function(x)
    assert isinstance(fx, torch.Tensor), "function(x) must return a torch.Tensor"
    assert fx.shape == x.shape, f"Expected shape {x.shape}, got {fx.shape}"

    # Name dynamic behavior: includes dim and axis
    assert system.name == "Bistable3D_axis1", f"Unexpected name: {system.name}"
    system.dim = 2
    assert system.name == "Bistable2D_axis1", f"Unexpected name after dim change: {system.name}"
    system.bistable_axis = 0
    assert system.name == "Bistable2D_axis0", f"Unexpected name after axis change: {system.name}"

    # Integration test (skips if torchdiffeq not installed)
    try:
        t_span = torch.linspace(0.0, 1.0, steps=11)
        x0 = torch.randn(batch_size, system.dim)
        traj = system.integrate(x0, t_span)
        assert traj.shape == (t_span.numel(), batch_size, system.dim), (
            f"Expected trajectory shape {(t_span.numel(), batch_size, system.dim)}, got {traj.shape}"
        )
        print("BistableND integrate: OK")
    except ImportError:
        print("torchdiffeq not installed; skipping integrate test for BistableND")

    print("BistableND tests passed.")
