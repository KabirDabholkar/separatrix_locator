import pytest

torch = pytest.importorskip("torch")

from separatrix_locator.reduction import EncoderNN, LatentDynamicsNN, LatentEigenfunctionNN
from separatrix_locator.reduction.losses import pairwise_stress_loss, discrete_one_step_koopman_loss


def test_pairwise_stress_loss_symmetric_and_nonnegative():
    torch.manual_seed(0)
    x = torch.randn(12, 4)
    z = torch.randn(12, 3)

    loss_xz = pairwise_stress_loss(x_points=x, z_points=z)
    loss_zx = pairwise_stress_loss(x_points=z, z_points=x)

    assert loss_xz.item() >= 0.0
    assert torch.allclose(loss_xz, loss_zx, atol=1e-6, rtol=1e-6)


def test_discrete_one_step_koopman_loss_backprop():
    torch.manual_seed(0)

    batch = 16
    x_dim = 5
    latent_dim = 2
    u_dim = 3

    encoder = EncoderNN(input_dim=x_dim, latent_dim=latent_dim, hidden_dim=16, num_layers=2).double()
    latent_dyn = LatentDynamicsNN(latent_dim=latent_dim, u_dim=u_dim, hidden_dim=16, num_layers=2).double()
    eigenfunction = LatentEigenfunctionNN(latent_dim=latent_dim, hidden_dim=16, num_layers=2).double()

    x_t = torch.randn(batch, x_dim, dtype=torch.double)
    x_next = torch.randn(batch, x_dim, dtype=torch.double)
    u_t = torch.randn(batch, u_dim, dtype=torch.double)

    # Squashed RHS used in the repo. Signature accepts optional u_t.
    def rhs_fn(psi, u_t_optional=None):
        return psi - psi ** 3

    loss = discrete_one_step_koopman_loss(
        encoder=encoder,
        latent_dynamics=latent_dyn,
        eigenfunction=eigenfunction,
        rhs_function=rhs_fn,
        x_t=x_t,
        x_next=x_next,
        u_t=u_t,
        kappa=1.3,
    )

    assert loss.ndim == 0
    loss.backward()

    for name, p in list(encoder.named_parameters()) + list(latent_dyn.named_parameters()) + list(eigenfunction.named_parameters()):
        assert p.grad is not None, f"Missing grad for {name}"
        assert torch.isfinite(p.grad).all(), f"Non-finite grad for {name}"

