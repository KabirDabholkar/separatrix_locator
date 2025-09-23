"""
Utility functions for separatrix location.

This module contains helper functions for plotting, distributions, composition,
validation, and various training utilities.
"""

from .plotting import plot_trajectories, plot_separatrix, plot_phase_portrait, plot_eigenfunction_landscape
from .distributions import IsotropicGaussian, MultiGapNormal, make_iid_multivariate, MixtureDistribution
from .compose import compose
from .helpers import (
    l_norm, rbf_gaussian, rbf_inv, rbf_laplacian, mutual_information_loss,
    restrict_to_distribution_loss, gmm_sample_from_residuals, reset_adam_momentum,
    train_model_on_trajectories_sgd, BistableKEF, compute_loss
)

__all__ = [
    "plot_trajectories",
    "plot_separatrix",
    "plot_phase_portrait", 
    "plot_eigenfunction_landscape",
    "IsotropicGaussian",
    "MultiGapNormal",
    "make_iid_multivariate",
    "MixtureDistribution",
    "compose",
    "l_norm",
    "rbf_gaussian", 
    "rbf_inv",
    "rbf_laplacian",
    "mutual_information_loss",
    "restrict_to_distribution_loss",
    "gmm_sample_from_residuals",
    "reset_adam_momentum",
    "train_model_on_trajectories_sgd",
    "BistableKEF",
    "compute_loss",
]
