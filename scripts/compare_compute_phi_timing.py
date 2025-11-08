import time

import torch

from separatrix_locator.core.training import compute_phi_and_dot_prod


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def run_comparison(batch_size=256, input_dim=30, dtype=torch.float32, seed=0, device=DEFAULT_DEVICE, repeats=10):
    torch.manual_seed(seed)

    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 128),
        torch.nn.Tanh(),
        torch.nn.Linear(128, 1),
    ).to(dtype=dtype, device=device)

    x_base = torch.randn(batch_size, input_dim, dtype=dtype, device=device, requires_grad=True)
    F_x = torch.randn_like(x_base)

    grad_durations = []
    jvp_durations = []

    for _ in range(repeats):
        x_grad = x_base.detach().clone().requires_grad_(True)
        F_grad = F_x.detach().clone()

        start_grad = time.perf_counter()
        compute_phi_and_dot_prod(
            model=model,
            x_batch=x_grad,
            input_to_model=x_grad,
            F_x=F_grad,
            use_jvp=False,
        )
        grad_durations.append(time.perf_counter() - start_grad)

        x_jvp = x_base.detach().clone().requires_grad_(True)
        F_jvp = F_x.detach().clone()

        start_jvp = time.perf_counter()
        compute_phi_and_dot_prod(
            model=model,
            x_batch=x_jvp,
            input_to_model=x_jvp,
            F_x=F_jvp,
            use_jvp=True,
        )
        jvp_durations.append(time.perf_counter() - start_jvp)

    grad_avg = sum(grad_durations) / repeats
    jvp_avg = sum(jvp_durations) / repeats

    print(f"Gradient method avg time ({repeats} runs): {grad_avg:.6f} seconds")
    print(f"JVP method avg time ({repeats} runs):      {jvp_avg:.6f} seconds")
    if grad_avg > 0:
        speedup = grad_avg / jvp_avg if jvp_avg > 0 else float("inf")
        print(f"Speedup (grad/jvp):                      {speedup:.2f}x")


if __name__ == "__main__":
    run_comparison()

