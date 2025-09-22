import pytest

torch = pytest.importorskip("torch")
from separatrix_locator.dynamics.bistableND import BistableND


def test_function_shape_and_type():
    system = BistableND(dim=3, bistable_axis=1)
    batch_size = 5
    x = torch.randn(batch_size, system.dim)
    fx = system.function(x)
    assert isinstance(fx, torch.Tensor)
    assert fx.shape == x.shape


def test_integrate_runs_and_shape():
    pytest.importorskip("torchdiffeq")
    system = BistableND(dim=2, bistable_axis=0)
    t_span = torch.linspace(0.0, 1.0, steps=11)
    x0 = torch.randn(4, system.dim)
    traj = system.integrate(x0, t_span)
    assert traj.shape == (t_span.numel(), x0.shape[0], system.dim)


def test_name_dynamic_with_dim_and_axis():
    system = BistableND(dim=2, bistable_axis=1)
    assert system.name == "Bistable2D_axis1"
    system.dim = 3
    assert system.name == "Bistable3D_axis1"
    system.bistable_axis = 0
    assert system.name == "Bistable3D_axis0"


