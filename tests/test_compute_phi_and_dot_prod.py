import pytest

torch = pytest.importorskip("torch")

from src.separatrix_locator.core.training import compute_phi_and_dot_prod


def test_compute_phi_and_dot_prod_matches_jvp():
    torch.manual_seed(0)

    batch_size = 5
    input_dim = 3

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 8),
        torch.nn.Tanh(),
        torch.nn.Linear(8, 1),
    ).double()

    base_x = torch.randn(batch_size, input_dim, dtype=torch.double)
    F_x_val = 0.5 * base_x

    x_batch_grad = base_x.clone().detach().requires_grad_(True)
    phi_grad, dot_grad = compute_phi_and_dot_prod(
        model=model,
        x_batch=x_batch_grad,
        input_to_model=x_batch_grad,
        F_x=F_x_val.clone(),
        use_jvp=False,
    )

    x_batch_jvp = base_x.clone().detach().requires_grad_(True)
    phi_jvp, dot_jvp = compute_phi_and_dot_prod(
        model=model,
        x_batch=x_batch_jvp,
        input_to_model=x_batch_jvp,
        F_x=F_x_val.clone(),
        use_jvp=True,
    )

    assert torch.allclose(phi_grad, phi_jvp, atol=1e-7, rtol=1e-7)
    assert torch.allclose(dot_grad, dot_jvp, atol=1e-7, rtol=1e-7)
