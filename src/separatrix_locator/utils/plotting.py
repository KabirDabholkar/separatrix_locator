"""
Plotting utilities for separatrix location.

This module provides functions for visualizing trajectories, separatrices,
and dynamical system behavior.
"""

import os
import shutil
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Optional, Tuple, List, Union
import seaborn as sns

# Set up matplotlib for better plots
use_tex = os.environ.get("USE_TEX", "0") == "1"
if use_tex and shutil.which("latex") is None:
    warnings.warn("USE_TEX requested but LaTeX not found; falling back to Matplotlib text rendering.")
    use_tex = False
plt.rcParams['text.usetex'] = use_tex
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["mathtext.fontset"] = "dejavuserif"
if use_tex:
    plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'


def plot_trajectories(
    trajectories: Union[torch.Tensor, List[torch.Tensor]], 
    title: str = "Trajectories",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.7,
    colors: Optional[List[str]] = None
) -> plt.Figure:
    """
    Plot trajectories from dynamical system integration.
    
    Parameters:
    -----------
    trajectories : torch.Tensor or list
        Trajectories of shape (time_steps, batch_size, dim) or list of such tensors
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    alpha : float
        Transparency for trajectory lines
    colors : list, optional
        Colors for different trajectories
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if isinstance(trajectories, torch.Tensor):
        trajectories = [trajectories]
    
    if colors is None:
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
    
    for i, traj in enumerate(trajectories):
        if traj.dim() == 3:  # (time_steps, batch_size, dim)
            for j in range(traj.shape[1]):
                ax.plot(traj[:, j, 0].numpy(), traj[:, j, 1].numpy(), 
                       color=colors[i], alpha=alpha, linewidth=1)
        elif traj.dim() == 2:  # (time_steps, dim)
            ax.plot(traj[:, 0].numpy(), traj[:, 1].numpy(), 
                   color=colors[i], alpha=alpha, linewidth=2)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_separatrix(
    separatrix_points: torch.Tensor,
    attractors: Optional[torch.Tensor] = None,
    title: str = "Separatrix",
    xlabel: str = "x",
    ylabel: str = "y",
    figsize: Tuple[int, int] = (8, 6),
    separatrix_color: str = "red",
    attractor_color: str = "green",
    separatrix_size: float = 20.0,
    attractor_size: float = 50.0
) -> plt.Figure:
    """
    Plot separatrix points and attractors.
    
    Parameters:
    -----------
    separatrix_points : torch.Tensor
        Separatrix points of shape (num_points, dim)
    attractors : torch.Tensor, optional
        Attractor points of shape (num_attractors, dim)
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    separatrix_color, attractor_color : str
        Colors for separatrix and attractor points
    separatrix_size, attractor_size : float
        Point sizes
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot separatrix points
    if separatrix_points.dim() == 2 and separatrix_points.shape[1] >= 2:
        ax.scatter(separatrix_points[:, 0].numpy(), separatrix_points[:, 1].numpy(),
                  c=separatrix_color, s=separatrix_size, alpha=0.7, 
                  label="Separatrix", marker='x')
    
    # Plot attractors
    if attractors is not None and attractors.dim() == 2 and attractors.shape[1] >= 2:
        ax.scatter(attractors[:, 0].numpy(), attractors[:, 1].numpy(),
                  c=attractor_color, s=attractor_size, alpha=0.8,
                  label="Attractors", marker='o')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_phase_portrait(
    dynamics_func,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    grid_size: int = 20,
    title: str = "Phase Portrait",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot a phase portrait of a 2D dynamical system.
    
    Parameters:
    -----------
    dynamics_func : callable
        The dynamical system function f(x) -> dx/dt
    xlim, ylim : tuple
        Limits for the plot
    grid_size : int
        Number of grid points in each direction
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], grid_size)
    y = np.linspace(ylim[0], ylim[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # Compute vector field
    points = torch.stack([torch.tensor(X.flatten()), torch.tensor(Y.flatten())], dim=1)
    vectors = dynamics_func(points)
    
    U = vectors[:, 0].reshape(grid_size, grid_size).numpy()
    V = vectors[:, 1].reshape(grid_size, grid_size).numpy()
    
    # Plot vector field
    ax.quiver(X, Y, U, V, alpha=0.7)
    
    # Add nullclines (where dx/dt = 0 or dy/dt = 0)
    # This is system-specific and would need to be implemented per system
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig


def plot_eigenfunction_landscape(
    eigenfunction_func,
    xlim: Tuple[float, float] = (-3, 3),
    ylim: Tuple[float, float] = (-3, 3),
    resolution: int = 100,
    title: str = "Eigenfunction Landscape",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot the landscape of a learned eigenfunction.
    
    Parameters:
    -----------
    eigenfunction_func : callable
        The eigenfunction to plot
    xlim, ylim : tuple
        Limits for the plot
    resolution : int
        Resolution of the grid
    title : str
        Plot title
    figsize : tuple
        Figure size
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create grid
    x = np.linspace(xlim[0], xlim[1], resolution)
    y = np.linspace(ylim[0], ylim[1], resolution)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate eigenfunction
    points = torch.stack([torch.tensor(X.flatten()), torch.tensor(Y.flatten())], dim=1)
    with torch.no_grad():
        values = eigenfunction_func(points)
    
    Z = values.reshape(resolution, resolution).numpy()
    
    # Plot contour
    contour = ax.contourf(X, Y, Z, levels=50, cmap='RdYlBu_r', alpha=0.8)
    ax.contour(X, Y, Z, levels=[0], colors='black', linewidths=2, linestyles='--')
    
    plt.colorbar(contour, ax=ax, label='Eigenfunction Value')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(title)
    
    return fig


def plot_training_curves(
    losses: List[float],
    title: str = "Training Loss",
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    figsize: Tuple[int, int] = (8, 6),
    log_scale: bool = True
) -> plt.Figure:
    """
    Plot training loss curves.
    
    Parameters:
    -----------
    losses : list
        List of loss values
    title : str
        Plot title
    xlabel, ylabel : str
        Axis labels
    figsize : tuple
        Figure size
    log_scale : bool
        Whether to use log scale for y-axis
        
    Returns:
    --------
    matplotlib.figure.Figure
        The created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(len(losses))
    ax.plot(epochs, losses, linewidth=2)
    
    if log_scale:
        ax.set_yscale('log')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    return fig
