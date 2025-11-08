"""
ODE trajectory plotting utilities for separatrix locator.

This module provides plotting functions for visualizing ODE trajectories,
including PCA projections, convergence analysis, and trajectory statistics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

from ..utils.ode_trajectory import (
    compute_line_trajectories,
    compute_pca_trajectories,
    analyze_trajectory_convergence,
    compute_trajectory_statistics
)


def plot_ode_line_trajectories(
    dynamics_function,
    attractors: List[torch.Tensor],
    save_dir: str,
    num_points: int = 100,
    T: float = 25.0,
    steps: int = 100,
    static_external_input: Optional[torch.Tensor] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8),
    device: str = "cpu"
) -> Dict[str, Any]:
    """
    Plot ODE trajectories along a line joining attractors with PCA analysis.
    
    This function replicates the functionality of plot_ODE_line_IC from main.py
    but with improved organization and additional analysis.
    
    Parameters:
    -----------
    dynamics_function : callable
        The dynamics function f(x) that defines the ODE dx/dt = f(x)
    attractors : List[torch.Tensor]
        List of attractor points to connect with a line
    save_dir : str
        Directory to save plots
    num_points : int
        Number of points along the line (default: 100)
    T : float
        Integration time (default: 25.0)
    steps : int
        Number of integration steps (default: 100)
    static_external_input : Optional[torch.Tensor]
        Static external input to pass to dynamics function
    show : bool
        Whether to display plots (default: True)
    figsize : Tuple[int, int]
        Figure size (default: (12, 8))
    device : str
        Device to use for computations (default: "cpu")
        
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing analysis results and plot paths
    """
    # Compute trajectories
    trajectories, times = compute_line_trajectories(
        dynamics_function=dynamics_function,
        attractors=attractors,
        num_points=num_points,
        T=T,
        steps=steps,
        static_external_input=static_external_input,
        device=device
    )
    
    # Perform PCA analysis
    pca_trajectories = compute_pca_trajectories(trajectories, n_components=2)
    
    # Analyze convergence
    convergence_analysis = analyze_trajectory_convergence(trajectories, attractors)
    
    # Compute trajectory statistics
    trajectory_stats = compute_trajectory_statistics(trajectories)
    
    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: PC1 vs Time (original plot from main.py)
    plt.figure(figsize=figsize)
    plt.subplot(2, 2, 1)
    
    # Plot PC1 trajectories
    times_np = times.detach().cpu().numpy()
    pca_trajectories_np = pca_trajectories.detach().cpu().numpy()
    
    for i in range(min(num_points, 20)):  # Limit to 20 trajectories for clarity
        plt.plot(times_np, pca_trajectories_np[:, i, 0], alpha=0.7, linewidth=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('PC1')
    plt.title('PC1 vs Time for Trajectories')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: PC2 vs Time
    plt.subplot(2, 2, 2)
    for i in range(min(num_points, 20)):
        plt.plot(times_np, pca_trajectories_np[:, i, 1], alpha=0.7, linewidth=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('PC2')
    plt.title('PC2 vs Time for Trajectories')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Trajectory in PC1-PC2 space
    plt.subplot(2, 2, 3)
    for i in range(min(num_points, 20)):
        plt.plot(pca_trajectories_np[:, i, 0], pca_trajectories_np[:, i, 1], 
                alpha=0.7, linewidth=0.8)
    
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('Trajectories in PC1-PC2 Space')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Plot 4: Convergence analysis
    plt.subplot(2, 2, 4)
    convergence_counts = convergence_analysis['convergence_counts']
    labels = list(convergence_counts.keys())
    counts = list(convergence_counts.values())
    
    plt.bar(labels, counts)
    plt.xlabel('Attractor')
    plt.ylabel('Number of Converged Trajectories')
    plt.title(f'Convergence Analysis (Total: {convergence_analysis["total_converged"]}/{convergence_analysis["total_trajectories"]})')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_path = save_path / 'ode_line_trajectories_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Create additional detailed plots
    
    # Plot: Individual PC1 trajectories (original style from main.py)
    plt.figure(figsize=(10, 6))
    for i in range(num_points):
        plt.plot(times_np, pca_trajectories_np[:, i, 0], alpha=0.6, linewidth=0.5)
    
    plt.xlabel('Time')
    plt.ylabel('PC1')
    plt.title('PC1 vs Time for All Trajectories')
    plt.grid(True, alpha=0.3)
    
    pc1_plot_path = save_path / 'PC1_line_trajectories.png'
    plt.savefig(pc1_plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot: Trajectory length distribution
    plt.figure(figsize=(8, 6))
    lengths = trajectory_stats['trajectory_lengths'].detach().cpu().numpy()
    plt.hist(lengths, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Trajectory Length')
    plt.ylabel('Frequency')
    plt.title(f'Trajectory Length Distribution\nMean: {trajectory_stats["mean_trajectory_length"]:.2f}, Std: {trajectory_stats["std_trajectory_length"]:.2f}')
    plt.grid(True, alpha=0.3)
    
    length_plot_path = save_path / 'trajectory_length_distribution.png'
    plt.savefig(length_plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # Plot: Final position distances
    plt.figure(figsize=(8, 6))
    final_distances = convergence_analysis['final_distances'].detach().cpu().numpy()
    plt.hist(final_distances, bins=30, alpha=0.7, edgecolor='black')
    plt.xlabel('Distance to Nearest Attractor')
    plt.ylabel('Frequency')
    plt.title('Distribution of Final Distances to Attractors')
    plt.grid(True, alpha=0.3)
    
    distance_plot_path = save_path / 'final_distances_distribution.png'
    plt.savefig(distance_plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return {
        'trajectories': trajectories,
        'pca_trajectories': pca_trajectories,
        'times': times,
        'convergence_analysis': convergence_analysis,
        'trajectory_statistics': trajectory_stats,
        'plot_paths': {
            'main_analysis': plot_path,
            'pc1_trajectories': pc1_plot_path,
            'length_distribution': length_plot_path,
            'distance_distribution': distance_plot_path
        }
    }


def plot_trajectory_phase_portrait(
    trajectories: torch.Tensor,
    attractors: List[torch.Tensor],
    save_dir: str,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 8),
    max_trajectories: int = 50
) -> Path:
    """
    Plot phase portrait of trajectories.
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        Trajectories tensor of shape (steps, num_points, dim)
    attractors : List[torch.Tensor]
        List of attractor points
    save_dir : str
        Directory to save plot
    show : bool
        Whether to display plot (default: True)
    figsize : Tuple[int, int]
        Figure size (default: (10, 8))
    max_trajectories : int
        Maximum number of trajectories to plot (default: 50)
        
    Returns:
    --------
    Path
        Path to saved plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    trajectories_np = trajectories.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Plot trajectories
    for i in range(min(trajectories.shape[1], max_trajectories)):
        plt.plot(trajectories_np[:, i, 0], trajectories_np[:, i, 1], 
                alpha=0.6, linewidth=0.8)
    
    # Plot attractors
    for i, attractor in enumerate(attractors):
        attractor_np = attractor.detach().cpu().numpy()
        plt.scatter(attractor_np[0], attractor_np[1], 
                   s=100, marker='*', color=f'C{i}', 
                   label=f'Attractor {i}', zorder=5)
    
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.title('Phase Portrait of Trajectories')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    plot_path = save_path / 'trajectory_phase_portrait.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return plot_path


def plot_velocity_analysis(
    trajectories: torch.Tensor,
    times: torch.Tensor,
    save_dir: str,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 8)
) -> Path:
    """
    Plot velocity analysis of trajectories.
    
    Parameters:
    -----------
    trajectories : torch.Tensor
        Trajectories tensor of shape (steps, num_points, dim)
    times : torch.Tensor
        Time vector
    save_dir : str
        Directory to save plot
    show : bool
        Whether to display plot (default: True)
    figsize : Tuple[int, int]
        Figure size (default: (12, 8))
        
    Returns:
    --------
    Path
        Path to saved plot
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    # Transpose to (num_points, steps, dim) for velocity computation
    trajectories_t = trajectories.transpose(0, 1)
    
    # Compute velocities
    velocities = torch.diff(trajectories_t, dim=1)
    velocity_norms = torch.norm(velocities, dim=-1)
    times_vel = times[1:]  # One fewer point for velocities
    
    trajectories_np = trajectories.detach().cpu().numpy()
    velocity_norms_np = velocity_norms.detach().cpu().numpy()
    times_np = times.detach().cpu().numpy()
    times_vel_np = times_vel.detach().cpu().numpy()
    
    plt.figure(figsize=figsize)
    
    # Plot 1: Velocity norms vs time
    plt.subplot(2, 2, 1)
    for i in range(min(trajectories.shape[1], 20)):
        plt.plot(times_vel_np, velocity_norms_np[i, :], alpha=0.7, linewidth=0.8)
    
    plt.xlabel('Time')
    plt.ylabel('Velocity Norm')
    plt.title('Velocity Norm vs Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Mean velocity vs time
    plt.subplot(2, 2, 2)
    mean_velocity = velocity_norms_np.mean(axis=0)
    plt.plot(times_vel_np, mean_velocity, linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Mean Velocity Norm')
    plt.title('Mean Velocity vs Time')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Velocity distribution
    plt.subplot(2, 2, 3)
    all_velocities = velocity_norms_np.flatten()
    plt.hist(all_velocities, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Velocity Norm')
    plt.ylabel('Frequency')
    plt.title('Velocity Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Velocity vs position (if 2D)
    if trajectories.shape[-1] >= 2:
        plt.subplot(2, 2, 4)
        for i in range(min(trajectories.shape[1], 10)):
            plt.scatter(trajectories_np[:, i, 0], trajectories_np[:, i, 1], 
                       c=velocity_norms_np[i, :], cmap='viridis', 
                       alpha=0.6, s=1)
        
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.title('Trajectories Colored by Velocity')
        plt.colorbar(label='Velocity Norm')
    
    plt.tight_layout()
    
    plot_path = save_path / 'velocity_analysis.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return plot_path
