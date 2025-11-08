"""
ODE trajectory analysis utilities for separatrix locator.

This module provides utilities for analyzing ODE trajectories, including
computing trajectories along lines between attractors and performing PCA analysis.
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Union
from sklearn.decomposition import PCA

from .odeint_utils import run_odeint_to_final


def compute_line_trajectories(
    dynamics_function,
    attractors: List[torch.Tensor],
    num_points: int = 100,
    T: float = 25.0,
    steps: int = 100,
    static_external_input: Optional[torch.Tensor] = None,
    device: str = "cpu"
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute trajectories for points along a line joining attractors.
    
    Parameters:
    -----------
    dynamics_function : callable
        The dynamics function f(x) that defines the ODE dx/dt = f(x)
    attractors : List[torch.Tensor]
        List of attractor points to connect with a line
    num_points : int
        Number of points along the line (default: 100)
    T : float
        Integration time (default: 25.0)
    steps : int
        Number of integration steps (default: 100)
    static_external_input : Optional[torch.Tensor]
        Static external input to pass to dynamics function
    device : str
        Device to use for computations (default: "cpu")
        
    Returns:
    --------
    Tuple[torch.Tensor, torch.Tensor]
        (trajectories, times) where trajectories shape is (steps, num_points, dim)
        and times shape is (steps,)
    """
    # Create points along the line joining attractors
    alpha = torch.linspace(0, 1, num_points, device=device)[:, None]
    
    # Handle case with more than 2 attractors by using first two
    if len(attractors) >= 2:
        line_points = attractors[0] * (1 - alpha) + attractors[1] * alpha
    else:
        raise ValueError("Need at least 2 attractors to create a line")
    
    # Create time vector
    times = torch.linspace(0, T, steps, device=device)
    
    # Run ODE integration for each point
    trajectories = run_odeint_to_final(
        dynamics_function,
        line_points,
        T,
        inputs=static_external_input,
        steps=steps,
        return_last_only=False,
        no_grad=True
    )
    
    return trajectories, times


def compute_pca_trajectories(
    trajectories: torch.Tensor,
    n_components: int = 2
) -> torch.Tensor:
    """
    Compute PCA of trajectories and return projected trajectories.
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        Trajectories tensor of shape (steps, num_points, dim)
    n_components : int
        Number of PCA components (default: 2)
        
    Returns:
    --------
    torch.Tensor
        PCA-transformed trajectories of shape (steps, num_points, n_components)
    """
    # Reshape trajectories for PCA: (steps * num_points, dim)
    original_shape = trajectories.shape
    trajectories_flat = trajectories.reshape(-1, trajectories.shape[-1])
    
    # Convert to numpy for sklearn
    trajectories_np = trajectories_flat.detach().cpu().numpy()
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_trajectories_flat = pca.fit_transform(trajectories_np)
    
    # Reshape back to original structure
    pca_trajectories = pca_trajectories_flat.reshape(
        original_shape[0], original_shape[1], n_components
    )
    
    return torch.from_numpy(pca_trajectories)


def analyze_trajectory_convergence(
    trajectories: torch.Tensor,
    attractors: List[torch.Tensor],
    convergence_threshold: float = 0.1
) -> dict:
    """
    Analyze convergence of trajectories to attractors.
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        Trajectories tensor of shape (steps, num_points, dim)
    attractors : List[torch.Tensor]
        List of attractor points
    convergence_threshold : float
        Distance threshold for considering convergence
        
    Returns:
    --------
    dict
        Dictionary containing convergence analysis results
    """
    final_positions = trajectories[-1, :, :]  # Last time point
    
    # Compute distances to each attractor
    distances_to_attractors = []
    for attractor in attractors:
        distances = torch.norm(final_positions - attractor, dim=-1)
        distances_to_attractors.append(distances)
    
    # Find closest attractor for each trajectory
    distances_tensor = torch.stack(distances_to_attractors, dim=-1)
    closest_attractor_idx = torch.argmin(distances_tensor, dim=-1)
    min_distances = torch.min(distances_tensor, dim=-1)[0]
    
    # Count converged trajectories
    converged = min_distances < convergence_threshold
    convergence_counts = {}
    
    for i, attractor in enumerate(attractors):
        converged_to_this = (closest_attractor_idx == i) & converged
        convergence_counts[f'attractor_{i}'] = converged_to_this.sum().item()
    
    return {
        'convergence_counts': convergence_counts,
        'total_converged': converged.sum().item(),
        'total_trajectories': trajectories.shape[0],
        'convergence_rate': converged.float().mean().item(),
        'final_distances': min_distances,
        'closest_attractor_indices': closest_attractor_idx
    }


def compute_trajectory_statistics(trajectories: torch.Tensor) -> dict:
    """
    Compute various statistics for trajectories.
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        Trajectories tensor of shape (steps, num_points, dim)
        
    Returns:
    --------
    dict
        Dictionary containing trajectory statistics
    """
    # Transpose to (num_points, steps, dim) for easier computation
    trajectories_t = trajectories.transpose(0, 1)
    
    # Compute trajectory lengths
    trajectory_lengths = torch.norm(torch.diff(trajectories_t, dim=1), dim=-1).sum(dim=1)
    
    # Compute velocity statistics
    velocities = torch.diff(trajectories_t, dim=1)
    velocity_norms = torch.norm(velocities, dim=-1)
    
    # Compute acceleration statistics
    accelerations = torch.diff(velocities, dim=1)
    acceleration_norms = torch.norm(accelerations, dim=-1)
    
    return {
        'trajectory_lengths': trajectory_lengths,
        'mean_trajectory_length': trajectory_lengths.mean().item(),
        'std_trajectory_length': trajectory_lengths.std().item(),
        'velocity_norms': velocity_norms,
        'mean_velocity': velocity_norms.mean().item(),
        'max_velocity': velocity_norms.max().item(),
        'acceleration_norms': acceleration_norms,
        'mean_acceleration': acceleration_norms.mean().item(),
        'max_acceleration': acceleration_norms.max().item()
    }


if __name__ == "__main__":
    pass