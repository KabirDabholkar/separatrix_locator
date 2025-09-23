"""
Helper utility functions for separatrix location.

This module contains various utility functions for training, optimization,
and analysis that support the main separatrix location algorithms.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Callable, Dict, Any
import numpy as np
from functools import partial


def l_norm(x, p=2):
    """Compute L-p norm along the last dimension."""
    return torch.norm(x, p=p, dim=-1)


def rbf_gaussian(x):
    """Gaussian RBF function."""
    return (-x.pow(2)).exp()


def rbf_inv(x):
    """Inverse RBF function."""
    return (1 / x).exp()


def rbf_laplacian(x):
    """Laplacian RBF function."""
    return (-x.pow(2).sqrt()).exp()


def mutual_information_loss(psi, eps=1e-8):
    """
    Compute mutual information based loss.
    
    Args:
        psi: tensor of shape (batch_size, num_classes) with softmax outputs.
    
    Returns:
        L_MI = E[H(psi(x))] - H(E[psi(x)])
    """
    # Conditional entropy per sample: H(psi(x))
    cond_entropy = -torch.sum(psi * torch.log(psi + eps), dim=1).mean()

    # Marginal distribution over classes: average over batch
    p_y = psi.mean(dim=0)
    marg_entropy = -torch.sum(p_y * torch.log(p_y + eps))

    return cond_entropy - marg_entropy


def restrict_to_distribution_loss(x_batch, phi_x, dist, threshold=-4.0):
    """
    Compute regularization loss to restrict model to distribution.
    
    Args:
        x_batch: Input batch
        phi_x: Model output
        dist: Distribution for computing log probabilities
        threshold: Threshold for weighting
        
    Returns:
        Regularization loss
    """
    # Compute the log probability for the current batch samples
    log_probs = dist.log_prob(x_batch)

    # Define a weighting function that is high when log_probs are low
    weight = torch.sigmoid(threshold - log_probs)

    # Regularisation: penalise high |phi(x)| in low probability regions
    reg_loss = torch.mean(weight * torch.abs(phi_x))
    return reg_loss


def gmm_sample_from_residuals(
    x_batch: torch.Tensor,
    residuals: torch.Tensor,
    batch_size: int,
    n_components: int = 2,
    oversample_factor: int = 1,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Sample a batch of points using a Gaussian Mixture Model (GMM) fitted on x_batch,
    weighted by residuals.

    Args:
        x_batch (torch.Tensor): Original batch of input points (N, D).
        residuals (torch.Tensor): Residuals at those points (N,) or (N, 1).
        batch_size (int): Number of new samples to draw.
        n_components (int): Number of GMM components.
        oversample_factor (int): Number of samples to draw (with replacement) for fitting GMM.
        device (str): Device to return the final tensor on.

    Returns:
        torch.Tensor: A (batch_size, D) tensor of new points sampled from GMM.
    """
    # Import GMM here to avoid circular imports
    try:
        from gmm_torch.gmm import GaussianMixture
    except ImportError:
        raise ImportError("gmm_torch package is required for GMM sampling. Install with: pip install gmm-torch")

    # Ensure residuals are 1D
    if residuals.ndim > 1:
        residuals = residuals.squeeze(-1)

    # Convert to NumPy
    residuals_np = residuals.detach().cpu().numpy()
    x_np = x_batch.detach().cpu().numpy()

    # Convert residuals to sampling probabilities
    probs = residuals_np - residuals_np.min() + 1e-6
    probs = probs / probs.sum()

    # Resample x values based on residual probabilities
    resample_indices = np.random.choice(len(x_np), size=oversample_factor * batch_size, p=probs)
    resampled_x = x_np[resample_indices]

    # Fit GMM
    gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    gmm.fit(resampled_x)

    # Sample from the GMM
    x_batch_np_new, _ = gmm.sample(batch_size)
    x_batch_new = torch.tensor(x_batch_np_new, dtype=torch.float32).to(device)
    x_batch_new.requires_grad_(True)

    return x_batch_new


def reset_adam_momentum(optimizer):
    """Reset Adam optimizer momentum."""
    for group in optimizer.param_groups:
        for p in group['params']:
            if p.grad is not None:
                state = optimizer.state[p]
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)


def reset_adam_momentum_periodically(optimizer, epoch, reset_interval=50):
    """Reset Adam optimizer momentum periodically."""
    if epoch % reset_interval == 0:
        reset_adam_momentum(optimizer)


def train_model_on_trajectories_sgd(
    trajectories, model, t_values, batch_size=32, num_epochs=1000, 
    learning_rate=0.01, device='cpu'
):
    """
    Train the neural network model using SGD with minibatches to minimize the loss:
    |ln(psi(x(t))) - ln(psi(x(0))) - t|^2

    Args:
        trajectories (torch.Tensor): Input tensor of shape (n_trials, T, d).
        model (nn.Module): PyTorch neural network mapping from R^d to R.
        t_values (torch.Tensor): Time values for the trajectories.
        batch_size (int): Minibatch size for SGD.
        num_epochs (int): Number of training epochs.
        learning_rate (float): Learning rate for SGD optimizer.
        device (str): Device to train on ('cpu' or 'cuda').
    """
    from torch.utils.data import DataLoader, TensorDataset
    import torch.optim as optim

    # Move data to the specified device
    trajectories = trajectories.to(device)
    model.to(device)

    # Prepare dataset and data loader for minibatch training
    dataset = TensorDataset(trajectories)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Define optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0

        for batch in data_loader:
            batch_trajectories = batch[0]  # Extract trajectories from dataset
            batch_size_here = batch_trajectories.shape[0]

            # Extract initial points x(0) and all time steps x(t)
            x_0 = batch_trajectories[:, 0:1, :]  # shape (batch_size, d)
            x_t = batch_trajectories  # shape (batch_size, T, d)

            # Compute f(x) and psi(x)
            f_x_0 = torch.abs(model(x_0.view(-1, x_0.shape[-1])))  # shape (batch_size, 1)
            psi_x_0 = torch.exp(f_x_0) - 1  # shape (batch_size, 1)

            f_x_t = torch.abs(model(x_t.view(-1, x_t.shape[-1])))  # shape (batch_size * T, 1)
            psi_x_t = torch.exp(f_x_t) - 1  # shape (batch_size * T, 1)

            # Reshape to (batch_size, T, 1)
            psi_x_0 = psi_x_0.view(batch_size_here, -1, 1)
            psi_x_t = psi_x_t.view(batch_size_here, -1, 1)

            # Compute logarithms
            log_psi_x_0 = torch.log(psi_x_0)  # shape (batch_size, 1, 1)
            log_psi_x_t = torch.log(psi_x_t)  # shape (batch_size, T, 1)

            # Compute the loss |ln(psi(x(t))) - ln(psi(x(0))) - t|^2
            loss = torch.mean((log_psi_x_t - log_psi_x_0 - t_values.unsqueeze(0).unsqueeze(-1)) ** 2)
            loss /= t_values[-1] ** 2

            # Backpropagation
            optimizer.zero_grad(set_to_none=True)
            loss.backward(retain_graph=True)
            optimizer.step()

            epoch_loss += loss.item()

        # Print progress
        if epoch % 1 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {epoch_loss / len(data_loader):.6f}")

    print("Training completed.")
    return model


class BistableKEF(nn.Module):
    """Analytical Koopman eigenfunction for bistable system."""
    
    def forward(self, x):
        return x / torch.sqrt(torch.abs((1 - x**2)))


def compute_loss(model, x, F, epoch, decay_factor=1.0):
    """
    Compute the regularized loss with optional decay.

    Args:
        model (torch.nn.Module): The model being trained.
        x (torch.Tensor): Input batch.
        F (callable): Dynamical system function.
        epoch (int): Current epoch number.
        decay_factor (float): Weight for the decay term.

    Returns:
        torch.Tensor: Total loss.
    """
    phi_x = model(x)

    # Compute phi'(x) using autograd
    x.requires_grad_(True)
    phi_x_prime = torch.autograd.grad(
        outputs=model(x),
        inputs=x,
        grad_outputs=torch.ones_like(model(x)),
        create_graph=True
    )[0]

    # Main loss term
    dot_prod = (phi_x_prime * F(x)).sum(axis=-1, keepdim=True)
    main_loss = torch.mean((dot_prod - phi_x) ** 2)

    # Variance penalty
    phi_mean = torch.mean(phi_x)
    phi_variance = torch.mean((phi_x - phi_mean) ** 2)
    variance_penalty = (phi_variance - 1) ** 2

    # Decay term (-l0 with weight)
    l0 = torch.abs(phi_x).mean()

    # Combine losses
    total_loss = main_loss + variance_penalty - decay_factor * l0

    return total_loss
