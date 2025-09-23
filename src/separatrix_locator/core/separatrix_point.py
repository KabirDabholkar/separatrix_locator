import torch
from torchdiffeq import odeint
from sklearn.cluster import KMeans


def find_separatrix_point_along_line(dynamics_function, external_input, attractors, num_points=5, num_iterations=5, time_points=None, return_all_points=False, final_time=5000):
    if time_points is None:
        time_points = torch.linspace(0, final_time, 2)

    attractor1, attractor2 = attractors
    current_points = attractor1 + torch.linspace(0, 1, num_points).unsqueeze(-1) * (attractor2 - attractor1)
    current_t_values = torch.linspace(0, 1, num_points)

    for _ in range(num_iterations):
        with torch.no_grad():
            if external_input is None:
                trajectories = odeint(lambda t, x: dynamics_function(x), current_points, time_points).detach().cpu()
            else:
                trajectories = odeint(lambda t, x: dynamics_function(x, external_input[None].repeat(num_points, 1)), current_points, time_points).detach().cpu()

        final_points = trajectories[-1]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(final_points)
        labels = kmeans.labels_

        switch_idx = None
        for i in range(len(current_points) - 1):
            if labels[i] != labels[i + 1]:
                switch_idx = i
                break

        if switch_idx is None:
            break

        new_t_values = torch.linspace(current_t_values[switch_idx], current_t_values[switch_idx + 1], num_points)
        current_points = attractor1 + new_t_values.unsqueeze(-1) * (attractor2 - attractor1)
        current_t_values = new_t_values

    if return_all_points:
        return current_points, trajectories, labels
    else:
        return torch.mean(current_points, dim=0)


def find_saddle_point(dynamics_function, point_on_separatrix, T=1, steps=100, return_all=False):
    def kinetic_energy(x):
        return torch.sum(dynamics_function(x) ** 2) / 2

    time_points = torch.linspace(0, T, steps)
    with torch.no_grad():
        trajectory = odeint(lambda t, x: dynamics_function(x), point_on_separatrix, time_points)

    ke_traj = torch.tensor([kinetic_energy(x) for x in trajectory])
    min_ke_idx = torch.argmin(ke_traj)
    saddle_point = trajectory[min_ke_idx]

    x = saddle_point.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=0.005)

    saddle_point = x.detach()

    if return_all:
        x = saddle_point.clone().detach().requires_grad_(True)
        jacobian = torch.autograd.functional.jacobian(dynamics_function, x)
        eigenvalues = torch.linalg.eigvals(jacobian)
        return saddle_point, eigenvalues, trajectory, ke_traj
    else:
        return saddle_point


