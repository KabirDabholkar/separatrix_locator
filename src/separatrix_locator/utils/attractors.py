import torch
from .odeint_utils import run_odeint_to_final


def get_estimate_attractor_func(dynamics_function):
    """
    Returns a function that estimates two attractors for a given dynamics function
    by integrating from random initial conditions and clustering the final states.

    The returned function has signature: estimate_attractors(dim, num_inits=128, T=30.0)
    and returns a tuple of two tensors (a1, a2) each of shape (dim,).
    """

    def estimate_attractors(dim: int, num_inits: int = 128, T: float = 30.0):
        with torch.no_grad():
            # Sample standard normal initial conditions in R^dim
            y0 = torch.randn(num_inits, dim)
            finals = run_odeint_to_final(dynamics_function, y0, T, return_last_only=True)

            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=2, random_state=0).fit(finals.detach().cpu())
            centers = torch.tensor(km.cluster_centers_, dtype=finals.dtype).to(finals.device)
            return centers[0], centers[1]

    return estimate_attractors


