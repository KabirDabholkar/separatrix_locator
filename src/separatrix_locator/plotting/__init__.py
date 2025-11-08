"""Plotting utilities and visualization helpers for separatrix_locator.

This package is intended to contain functions and classes for visualizing
trajectories, vector fields, separatrices, and experiment results.
"""

from .plots import (
    plot_dynamics_1D,
    plot_dynamics_2D,
    evaluate_on_grid,
    plot_flow_streamlines,
    dynamics_to_kinetic_energy,
    remove_frame,
)
from .hermite import (
    plot_hermite_curves_with_separatrix,
)

__all__ = [
    'plot_dynamics_1D',
    'plot_dynamics_2D',
    'evaluate_on_grid',
    'plot_flow_streamlines',
    'dynamics_to_kinetic_energy',
    'remove_frame',
    'plot_hermite_curves_with_separatrix',
]


