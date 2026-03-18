import pytest

torch = pytest.importorskip("torch")
from separatrix_locator.dynamics.duffing import DuffingOscillator


def test_function_shape_and_type():
    system = DuffingOscillator(damping=0.7)
    batch_size = 4
    x = torch.randn(batch_size, system.dim)
    fx = system.function(x)
    assert isinstance(fx, torch.Tensor)
    assert fx.shape == x.shape


def test_integrate_runs_and_shape():
    pytest.importorskip("torchdiffeq")
    system = DuffingOscillator()
    t_span = torch.linspace(0.0, 1.0, steps=21)
    x0 = torch.randn(3, system.dim)
    traj = system.integrate(x0, t_span)
    assert traj.shape == (t_span.numel(), x0.shape[0], system.dim)


def test_name_dynamic_fixed_dim():
    system = DuffingOscillator()
    assert system.dim == 2
    assert system.name == "DuffingOscillator2D"


