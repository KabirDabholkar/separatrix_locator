from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple, Sequence

import numpy as np
import torch
import matplotlib.pyplot as plt

from .plots import remove_frame
from ..utils.interpolation import generate_curves_between_points


def _predict_kef(
    predictor: Callable[[torch.Tensor], torch.Tensor],
    points: torch.Tensor,
) -> torch.Tensor:
    """Safely call predictor on points of shape (num_curves, num_points, dim) or (N, dim).

    Returns a tensor of shape (num_curves, num_points) with scalar KEF values.
    """
    original_shape = points.shape
    if points.dim() == 3:
        num_curves, num_points, dim = original_shape
        flat = points.reshape(-1, dim)
    elif points.dim() == 2:
        num_curves, num_points = 1, points.shape[0]
        flat = points
    else:
        raise ValueError("points must be (num_curves, num_points, dim) or (N, dim)")

    with torch.no_grad():
        vals = predictor(flat)
    if vals.dim() == 2 and vals.shape[-1] == 1:
        vals = vals[..., 0]
    vals = vals.reshape(num_curves, num_points)
    return vals


def find_separatrix_along_curve_using_ODE(
    dynamics_function: Callable[[torch.Tensor], torch.Tensor],
    alphas: np.ndarray | torch.Tensor,
    curve_points: np.ndarray | torch.Tensor,
    integration_time: float = 30.0,
    kmeans_random_state: int = 42,
    clustering_method: str = "attractor_eps",
    attractor_epsilon: float = 0.02,
    attractors: Optional[torch.Tensor] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Integrate curve points through the provided dynamics and infer separatrix locations.

    Parameters
    ----------
    dynamics_function : Callable[[torch.Tensor], torch.Tensor]
        Vector field defining the ODE dynamics.
    alphas : array-like
        1-D array of curve parameters corresponding to the second dimension of curve_points.
    curve_points : array-like
        Array of shape (num_curves, num_points, dim) containing points along each curve.
    integration_time : float, optional
        Time horizon for integrating the dynamics.
    kmeans_random_state : int, optional
        Random seed used when clustering with k-means.
    clustering_method : {"kmeans", "attractor_eps"}, optional
        Strategy for classifying integrated endpoints into basins.
    attractor_epsilon : float, optional
        Distance threshold for the "attractor_eps" method.
    attractors : torch.Tensor, optional
        Tensor of shape (2, dim) with attractor locations, required if clustering_method=="attractor_eps".

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - change_points_alpha: array of length num_curves with estimated separatrix parameter values.
        - labels_bt: array of shape (num_curves, num_points) with class labels along each curve.
    """
    from sklearn.cluster import KMeans
    from torchdiffeq import odeint

    if isinstance(alphas, torch.Tensor):
        alpha_range = alphas.detach().cpu().numpy()
    else:
        alpha_range = np.asarray(alphas)

    if alpha_range.ndim != 1:
        raise ValueError("alphas must be a 1-D array of curve parameters")

    if isinstance(curve_points, torch.Tensor):
        points_bt = curve_points.detach()
    else:
        points_bt = torch.tensor(curve_points, dtype=torch.float32)

    if points_bt.ndimension() != 3:
        raise ValueError("curve_points must have shape (num_curves, num_points, dim)")

    num_curves, num_points, dim = points_bt.shape
    device = points_bt.device

    flat_points = points_bt.reshape(-1, dim)
    t = torch.tensor([0.0, integration_time], dtype=points_bt.dtype, device=device)

    with torch.no_grad():
        trajectories = odeint(lambda tt, xx: dynamics_function(xx), flat_points, t)
    final_states = trajectories[-1].detach().cpu().numpy()

    if clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=2, random_state=kmeans_random_state)
        labels_flat = kmeans.fit_predict(final_states)
    elif clustering_method == "attractor_eps":
        if attractors is None:
            raise ValueError("attractors must be provided when clustering_method='attractor_eps'")
        attractors_np = attractors.detach().cpu().numpy()
        dists = np.stack(
            [
                np.linalg.norm(final_states - attractors_np[0], axis=1),
                np.linalg.norm(final_states - attractors_np[1], axis=1),
            ],
            axis=1,
        )
        nearest = np.argmin(dists, axis=1)
        min_d = dists[np.arange(dists.shape[0]), nearest]
        labels_flat = np.where(min_d <= attractor_epsilon, nearest, -1)
    else:
        raise ValueError("clustering_method must be 'kmeans' or 'attractor_eps'")

    labels_bt = labels_flat.reshape(num_curves, num_points)

    if clustering_method == "kmeans":
        changes_bt = np.diff(labels_bt, axis=1) != 0
        change_points_idx = np.argmax(changes_bt, axis=1)
        no_changes = ~np.any(changes_bt, axis=1)
        change_points_idx[no_changes] = num_points // 2
        change_points_alpha = alpha_range[change_points_idx]
        return change_points_alpha, labels_bt

    change_points_alpha = np.zeros(num_curves, dtype=float)

    def left_change_index(labels_row: np.ndarray, target_label: int) -> Optional[int]:
        idxs = np.where(labels_row == target_label)[0]
        if len(idxs) == 0 or idxs[0] != 0:
            return None
        stops = np.where(labels_row != target_label)[0]
        if len(stops) == 0:
            return num_points - 1
        return int(stops[0] - 1)

    def right_change_index(labels_row: np.ndarray, target_label: int) -> Optional[int]:
        idxs = np.where(labels_row == target_label)[0]
        if len(idxs) == 0 or idxs[-1] != num_points - 1:
            return None
        stops = np.where(labels_row[::-1] != target_label)[0]
        if len(stops) == 0:
            return 0
        return int(num_points - stops[0])

    for i in range(num_curves):
        row = labels_bt[i]
        left_idx_opt = left_change_index(row, target_label=0)
        right_idx_opt = right_change_index(row, target_label=1)

        alpha_left = alpha_range[left_idx_opt] if left_idx_opt is not None else None
        alpha_right = alpha_range[right_idx_opt] if right_idx_opt is not None else None

        if (alpha_left is not None) and (alpha_right is not None):
            change_points_alpha[i] = 0.5 * (alpha_left + alpha_right)
        elif alpha_left is not None:
            change_points_alpha[i] = alpha_left
        elif alpha_right is not None:
            change_points_alpha[i] = alpha_right
        else:
            change_points_alpha[i] = 0.5 * (alpha_range[0] + alpha_range[-1])

    return change_points_alpha, labels_bt


def plot_hermite_curves_with_separatrix(
    dynamics_function: Callable[[torch.Tensor], torch.Tensor],
    attractors: torch.Tensor,
    kef_predictor: Callable[[torch.Tensor], torch.Tensor],
    input_concatenator: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    x_limits: Optional[Tuple[float, float]] = None,
    y_limits: Optional[Tuple[float, float]] = None,
    num_curves: int = 100,
    num_points: int = 100,
    rand_scale: float = 5.0,
    alpha_lims: Tuple[float, float] = (0.0, 1.0),
    integration_time: float = 30.0,
    kmeans_random_state: int = 42,
    clustering_method: str = "attractor_eps", #"kmeans",
    attractor_epsilon: float = 0.02,
    save_dir: Optional[str] = None,
    show: bool = True,
) -> None:
    """
    Generate random cubic Hermite curves between two attractors, integrate their endpoints
    through the dynamics to label basins via clustering, and compare true separatrix points
    against zeros/argmins of the approximated Koopman eigenfunction (KEF).

    Parameters
    - dynamics_function: f(x) -> dx/dt callable
    - attractors: tensor of shape (2, dim)
    - kef_predictor: callable mapping points -> scalar KEF value; accepts (N, dim) or (B, T, dim)
    - input_concatenator: optional function to augment state inputs before kef_predictor
      (e.g., concatenate static external inputs). Receives (B, T, dim) points and returns same-shape dim'.
    - x_limits, y_limits: optional axis limits for 2D PCA plots
    - num_curves, num_points: sampling parameters along Hermite curves
    - rand_scale: random tangent perturbation scale
    - alpha_lims: parameter range along the curve
    - integration_time: time horizon for endpoint integration used for clustering basins
    - kmeans_random_state: RNG seed for clustering
    - save_dir: optional directory to save figures
    - show: whether to display figures; if False, figures are closed
    """
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    from sklearn.metrics import r2_score
    from torchdiffeq import odeint

    Path(save_dir).mkdir(parents=True, exist_ok=True) if save_dir is not None else None

    torch.manual_seed(0)
    np.random.seed(0)

    # Prepare endpoints and random Hermite curves in numpy
    x_np, y_np = attractors.detach().cpu().numpy()
    all_points_np, alpha_range = generate_curves_between_points(
        x_np,
        y_np,
        lims=list(alpha_lims),
        num_points=num_points,
        num_curves=num_curves,
        rand_scale=rand_scale,
        return_alpha=True,
    )

    # PCA for visualization across all points
    flat_points_np = all_points_np.reshape(-1, all_points_np.shape[-1])
    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(flat_points_np)

    # Integrate dynamics from each sampled point, but we only need basin label at final time
    points_tensor = torch.tensor(flat_points_np, dtype=torch.float32)
    t = torch.tensor([0.0, integration_time])
    with torch.no_grad():
        trajectories = odeint(lambda tt, xx: dynamics_function(xx), points_tensor, t)
    final_states = trajectories[-1].detach().cpu().numpy()

    # 2-cluster final states to infer basins
    if clustering_method == "kmeans":
        kmeans = KMeans(n_clusters=2, random_state=kmeans_random_state)
        labels_flat = kmeans.fit_predict(final_states)
    elif clustering_method == "attractor_eps":
        # Assign label by nearest attractor if within epsilon; else label -1 (undecided)
        attractors_np = attractors.detach().cpu().numpy()
        dists = np.stack([
            np.linalg.norm(final_states - attractors_np[0], axis=1),
            np.linalg.norm(final_states - attractors_np[1], axis=1),
        ], axis=1)
        nearest = np.argmin(dists, axis=1)
        min_d = dists[np.arange(dists.shape[0]), nearest]
        labels_flat = np.where(min_d <= attractor_epsilon, nearest, -1)
    else:
        raise ValueError("clustering_method must be 'kmeans' or 'attractor_eps'")

    # Reshape data back to (num_curves, num_points, ...)
    pca_points_bt = pca_points.reshape(num_curves, num_points, 2)
    labels_bt = labels_flat.reshape(num_curves, num_points)

    # Plot PCA-colored by label segments and mark change points
    fig, ax = plt.subplots(figsize=(5, 5))
    cluster_colors = ['C0', 'C1', 'C2']
    changes = np.diff(labels_bt, axis=1) != 0
    for i in range(num_curves):
        change_indices = np.where(changes[i])[0]
        if len(change_indices) == 0:
            label_color_index = labels_bt[i, 0] if labels_bt[i, 0] in (0, 1) else 2
            ax.plot(pca_points_bt[i, :, 0], pca_points_bt[i, :, 1], c=cluster_colors[label_color_index], alpha=0.5)
        else:
            start_idx = 0
            for change_idx in change_indices:
                ax.plot(
                    pca_points_bt[i, start_idx:change_idx + 1, 0],
                    pca_points_bt[i, start_idx:change_idx + 1, 1],
                    c=cluster_colors[labels_bt[i, start_idx] if labels_bt[i, start_idx] in (0, 1) else 2],
                    alpha=0.5,
                )
                ax.plot(pca_points_bt[i, change_idx, 0], pca_points_bt[i, change_idx, 1], 'x', color='red', markersize=3, alpha=0.8)
                start_idx = change_idx + 1
            if start_idx < len(pca_points_bt[i]):
                ax.plot(
                    pca_points_bt[i, start_idx:, 0],
                    pca_points_bt[i, start_idx:, 1],
                    c=cluster_colors[labels_bt[i, start_idx] if labels_bt[i, start_idx] in (0, 1) else 2],
                    alpha=0.5,
                )

    # Plot endpoints in PCA space
    endpoints_np = np.stack([x_np, y_np], axis=0)
    pca_endpoints = pca.transform(endpoints_np)
    ax.scatter(pca_endpoints[:, 0], pca_endpoints[:, 1], color='lightgreen', s=50, label="Endpoints", zorder=1)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    remove_frame(ax)
    if save_dir is not None:
        fig.savefig(Path(save_dir) / f"hermite_cubic_interpolations_plus_clustering_scale{rand_scale}.png", dpi=300)
        fig.savefig(Path(save_dir) / f"hermite_cubic_interpolations_plus_clustering_scale{rand_scale}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close(fig)

    # KEF evaluation along curves
    points_bt = torch.tensor(all_points_np, dtype=torch.float32)
    if input_concatenator is not None:
        points_for_kef = input_concatenator(points_bt)
    else:
        points_for_kef = points_bt
    kef_vals = _predict_kef(kef_predictor, points_for_kef)

    # True separatrix alpha from labels
    if clustering_method == "kmeans":
        # First change along alpha
        changes_bt = np.diff(labels_bt, axis=1) != 0
        change_points_idx = np.argmax(changes_bt, axis=1)
        no_changes = ~np.any(changes_bt, axis=1)
        change_points_idx[no_changes] = num_points // 2
        change_points_alpha = alpha_range[change_points_idx]
    else:
        # Epsilon-to-attractor method: compute change from both directions and average
        change_points_alpha = np.zeros(num_curves, dtype=float)

        def left_change_index(labels_row: np.ndarray, target_label: int) -> Optional[int]:
            # Find the last index of the first contiguous run of target_label from the left
            idxs = np.where(labels_row == target_label)[0]
            if len(idxs) == 0:
                return None
            # Only consider the initial run starting at the first index that equals target_label if it's at 0
            if idxs[0] != 0:
                return None
            # Find first position where label stops being target_label
            stops = np.where(labels_row != target_label)[0]
            if len(stops) == 0:
                return num_points - 1
            return int(stops[0] - 1)

        def right_change_index(labels_row: np.ndarray, target_label: int) -> Optional[int]:
            # Find the first index of the last contiguous run of target_label from the right
            idxs = np.where(labels_row == target_label)[0]
            if len(idxs) == 0:
                return None
            if idxs[-1] != num_points - 1:
                return None
            stops = np.where(labels_row[::-1] != target_label)[0]
            if len(stops) == 0:
                return 0
            # Convert from reversed index to original: position where run ends from the right
            return int(num_points - (stops[0]))

        for i in range(num_curves):
            row = labels_bt[i]
            left_idx_opt = left_change_index(row, target_label=0)
            right_idx_opt = right_change_index(row, target_label=1)

            if left_idx_opt is not None and 0 <= left_idx_opt < num_points:
                alpha_left = alpha_range[left_idx_opt]
            else:
                alpha_left = None

            if right_idx_opt is not None and 0 <= right_idx_opt < num_points:
                # right change index returns the index of the first point of the right run
                alpha_right = alpha_range[right_idx_opt]
            else:
                alpha_right = None

            if (alpha_left is not None) and (alpha_right is not None):
                change_points_alpha[i] = 0.5 * (alpha_left + alpha_right)
            elif alpha_left is not None:
                change_points_alpha[i] = alpha_left
            elif alpha_right is not None:
                change_points_alpha[i] = alpha_right
            else:
                change_points_alpha[i] = 0.5 * (alpha_lims[0] + alpha_lims[1])

        # Build synthetic indices from alpha for downstream selection logic
        change_points_idx = np.clip(
            np.searchsorted(alpha_range, change_points_alpha, side='left'),
            0,
            num_points - 1,
        )

    # KEF-based estimate: argmin |psi| along each curve
    abs_kef = np.abs(kef_vals.detach().cpu().numpy())
    argmin_idx = np.argmin(abs_kef, axis=1)
    argmin_alpha = alpha_range[argmin_idx]

    # Scatter comparison and R^2
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2))
    ax.scatter(change_points_alpha, argmin_alpha)
    xlim = ax.get_xlim(); ylim = ax.get_ylim()
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel(r'true separatrix point $\alpha$')
    ax.set_ylabel(r'$\psi=0$ point $\alpha$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xticks([0.0, 0.5, 1.0])
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.spines['left'].set_bounds(0, 1)
    ax.spines['bottom'].set_bounds(0, 1)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    r2 = r2_score(change_points_alpha, argmin_alpha)
    ax.text(0.3, 0.7, f'$R^2={r2:.3f}$', transform=ax.transAxes, verticalalignment='top', fontsize=12)
    ax.set_aspect('equal')
    fig.tight_layout()
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / f"separatrix_position_along_curves_scale{rand_scale}.png", dpi=300)
        fig.savefig(Path(save_dir) / f"separatrix_position_along_curves_scale{rand_scale}.pdf", dpi=300)
    plt.close(fig)

    # Grid of KEF values with true separatrix alpha lines
    fig, axes = plt.subplots(4, 10, figsize=(20, 8))
    axes = axes.flatten()
    for i in range(min(num_curves, len(axes))):
        ax = axes[i]
        ax.plot(alpha_range, kef_vals[i, :].detach().cpu().numpy())
        ax.axvline(x=change_points_alpha[i], color='r', linestyle='--', alpha=0.7)
        ax.set_title(f'Curve {i + 1}')
        ax.grid(True)
    for i in range(num_curves, len(axes)):
        axes[i].set_visible(False)
    plt.tight_layout()
    if show:
        plt.show()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / f"hermite_cubic_interpolations_KEFvals_scale{rand_scale}.png", dpi=300)
    plt.close(fig)

    # Selected curves: min/median/max change points
    min_idx = int(np.argmin(change_points_idx))
    max_idx = int(np.argmax(change_points_idx))
    median_idx = int(np.argsort(change_points_idx)[len(change_points_idx) // 2])
    selected_indices: Sequence[int] = [min_idx, median_idx, max_idx]
    labels = ['Curve 1', 'Curve 2', 'Curve 3']
    colors = ['blue', 'green', 'red']

    fig, ax = plt.subplots(figsize=(4, 3))
    for idx, label, color in zip(selected_indices, labels, colors):
        ax.plot(alpha_range, kef_vals[idx, :].detach().cpu().numpy(), color=color, lw=1, label=label)
        ax.axvline(x=change_points_alpha[idx], color=color, linestyle='--', alpha=1)
    ax.axhline(0, ls='solid', c='k', alpha=0.3)
    ax.set_xlabel(r'Curve parameter $\alpha$', fontsize=13)
    ax.set_ylabel(r'KEF Value $\psi$', fontsize=13)
    remove_frame(ax)
    ax.set_xticks([0, 0.5, 1])
    ax.spines['bottom'].set_bounds(0, 1)
    ax.spines['left'].set_bounds(-1.0, 1.0)
    ax.set_yticks([-1.0, 0.0, 1.0])
    fig.tight_layout()
    if save_dir is not None:
        fig.savefig(Path(save_dir) / f"hermite_cubic_interpolations_KEFvals_selected_scale{rand_scale}.png", dpi=300)
        fig.savefig(Path(save_dir) / f"hermite_cubic_interpolations_KEFvals_selected_scale{rand_scale}.pdf", dpi=300)
    if show:
        plt.show()
    plt.close(fig)


