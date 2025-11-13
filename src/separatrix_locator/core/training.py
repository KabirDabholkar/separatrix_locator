"""
Training functions for Koopman eigenfunction models.

This module contains the core training functions for learning Koopman eigenfunctions
from dynamical systems data.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd.functional import jvp
from functools import partial
from typing import Optional, List, Callable, Dict, Any
import numpy as np

from ..utils.helpers import restrict_to_distribution_loss


def log_metrics(logger, metrics: Dict[str, float], epoch: int):
    """
    Log metrics to the given logger(s).

    Args:
        logger (None, callable, list of callables): Logger(s) to log metrics. Each logger should have a `.log()` or `.add_scalar()` method.
        metrics (dict): Dictionary of metric names and their values.
        epoch (int): Current epoch number.
    """
    if logger is None:
        return

    if not isinstance(logger, list):
        logger = [logger]

    for log in logger:
        if hasattr(log, "add_scalar"):  # TensorBoard-like logger
            for key, value in metrics.items():
                log.add_scalar(key, value, epoch)
        elif hasattr(log, "log"):  # WandB-like logger
            log.log({**metrics, "epoch": epoch})


class DecayModule:
    """Optional module to compute a decaying weight factor with a delay."""
    
    def __init__(self, initial_decay=1.0, decay_rate=0.95, start_epoch=1000):
        """
        Args:
            initial_decay (float): Initial weight for the decayed term.
            decay_rate (float): Exponential decay rate per epoch.
            start_epoch (int): The epoch after which decay begins.
        """
        self.initial_decay = initial_decay
        self.decay_rate = decay_rate
        self.start_epoch = start_epoch

    def get_decay_factor(self, epoch):
        """Calculate the decay factor for the given epoch."""
        if epoch < self.start_epoch:
            return 1.0
        adjusted_epoch = epoch - self.start_epoch
        return self.initial_decay * (self.decay_rate ** adjusted_epoch)


def variance_normaliser(x, y, axis=None, return_terms=False):
    """Variance-based normalizer for loss functions."""
    numerator = torch.mean((x - y) ** 2, axis=axis)
    denominator = torch.std(x, axis=axis) * torch.std(y, axis=axis)
    ratio = numerator / denominator
    if return_terms:
        return ratio, numerator, denominator
    return ratio


def shuffle_normaliser(x, y, axis=0, return_terms=False):
    """Shuffle-based normalizer for loss functions."""
    permutation = np.random.permutation(x.shape[0])
    numerator = torch.mean((x - y) ** 2, axis=axis)
    denominator = torch.mean((x - y[permutation]) ** 2, axis=axis)
    ratio = numerator / denominator
    if return_terms:
        return ratio, numerator, denominator
    return ratio


def distance_weighted_normaliser(x, y, positions, axis=0, return_terms=False, distance_threshold=1.0):
    """Distance-weighted normalizer for loss functions."""
    permutation = np.random.permutation(x.shape[0])
    distances = torch.norm(positions - positions[permutation], dim=-1) / positions.shape[0]
    distance_threshold = np.quantile(distances.flatten().detach().cpu().numpy(), 0.3)

    weights = torch.exp(-distances / distance_threshold)
    numerators = (x - y) ** 2
    denominators = (x - y[permutation]) ** 2
    numerator_sum = torch.sum(numerators * weights[:, None], axis=axis)
    denominator_sum = torch.sum(denominators * weights[:, None], axis=axis)
    ratio = numerator_sum / denominator_sum
    if return_terms:
        return ratio, numerator_sum, denominator_sum
    return ratio


def compute_phi_and_dot_prod(model, x_batch, input_to_model, F_x, use_jvp=False, external_inputs=None):
    """
    Compute phi(x) and the directional derivative phi'(x)F(x).

    Args:
        model: The Koopman eigenfunction model.
        x_batch: Tensor of input states requiring gradients.
        input_to_model: Input tensor fed directly into the model (may include external inputs).
        F_x: Vector field evaluated at x_batch.
        use_jvp: Whether to use Jacobian-vector products (True) or explicit gradients (False).
        external_inputs: Optional tensor of external inputs concatenated with x_batch.

    Returns:
        Tuple[phi_x, dot_prod] where dot_prod aligns with phi'(x)F(x).
    """
    if use_jvp:
        def model_fn(x):
            if external_inputs is None:
                return model(x)
            return model(torch.cat((x, external_inputs), dim=-1))

        phi_x, directional_derivative = jvp(
            model_fn,
            (x_batch,),
            (F_x,),
            create_graph=True,
            strict=True,
        )
        dot_prod = directional_derivative
    else:
        phi_x = model(input_to_model)
        phi_x_prime = torch.autograd.grad(
            outputs=phi_x,
            inputs=x_batch,
            grad_outputs=torch.ones_like(phi_x),
            create_graph=True,
        )[0]
        dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True)
    return phi_x, dot_prod


def eval_loss(
    model, F, dist, external_input_dist=None, dist_requires_dim=True, 
    batch_size=64, dynamics_dim=1, eigenvalue=1, drop_values_outside_range=None, 
    normaliser=shuffle_normaliser, scale_dist=1, ext_inp_batch_size=None
):
    """
    Evaluate the loss of a model on a dynamical system.
    
    Args:
        model: The Koopman eigenfunction model
        F: The dynamical system function
        dist: Distribution for sampling initial conditions
        external_input_dist: Optional external input distribution
        dist_requires_dim: Whether distribution requires explicit dimensionality
        batch_size: Batch size for evaluation
        dynamics_dim: Dimensionality of the dynamical system
        eigenvalue: Eigenvalue for the Koopman equation
        drop_values_outside_range: Optional range for filtering values
        normaliser: Function for normalizing the loss
        scale_dist: Scaling factor for the distribution
        ext_inp_batch_size: Batch size for external inputs
        
    Returns:
        Normalized loss value
    """
    sample_shape = [batch_size]
    if dist_requires_dim:
        sample_shape += [dynamics_dim]
    x_batch = dist.sample(sample_shape=sample_shape)

    x_batch.requires_grad_(True)
    x_batch = x_batch * scale_dist

    input_to_model = x_batch
    if external_input_dist is not None:
        if ext_inp_batch_size is None:
            ext_inp_batch_size = batch_size
        else:
            assert batch_size % ext_inp_batch_size == 0, "ext_inp_batch_size must divide batch_size evenly."

        ext_sample_shape = [ext_inp_batch_size]
        if dist_requires_dim:
            ext_sample_shape += [dynamics_dim]
        external_inputs = external_input_dist.sample(sample_shape=ext_sample_shape)

        repeats = batch_size // ext_inp_batch_size
        external_inputs = external_inputs.repeat(repeats, *([1] * (external_inputs.dim() - 1)))
        input_to_model = torch.concat((input_to_model, external_inputs), dim=-1)

    # Forward pass and compute phi(x)
    # Ensure input is on the same device as the model
    if hasattr(model, 'parameters'):
        model_device = next(model.parameters()).device
        input_to_model = input_to_model.to(model_device)
        x_batch = x_batch.to(model_device)
    
    # Make sure x_batch requires gradients for the computation
    x_batch = x_batch.requires_grad_(True)
    
    # For models that only use the original input (no external inputs), use x_batch directly
    if external_input_dist is None:
        phi_x = model(x_batch)
    else:
        phi_x = model(input_to_model)
    
    points_to_use = torch.ones_like(x_batch)[..., 0:1]
    if drop_values_outside_range is not None:
        points_to_use = (phi_x > drop_values_outside_range[0]) & (phi_x < drop_values_outside_range[1])

    # Compute phi'(x) - use x_batch for gradient computation
    phi_x_prime = torch.autograd.grad(
        outputs=phi_x,
        inputs=x_batch,
        grad_outputs=torch.ones_like(phi_x),
        create_graph=True,
        allow_unused=True
    )[0]

    # Compute F(x_batch)
    F_inputs = [x_batch] + ([] if external_input_dist is None else [external_inputs])
    F_x = F(*F_inputs).detach()

    # Main loss term: ||phi'(x) F(x) - phi(x)||^2
    dot_prod = (phi_x_prime * F_x).sum(axis=-1, keepdim=True).detach()

    if external_input_dist is not None and batch_size != ext_inp_batch_size:
        new_shape_dot_prod = (batch_size // ext_inp_batch_size, ext_inp_batch_size) + dot_prod.shape[1:]
        dot_prod = dot_prod.view(new_shape_dot_prod)
        
        new_shape_phi_x = (batch_size // ext_inp_batch_size, ext_inp_batch_size) + phi_x.shape[1:]
        phi_x = phi_x.view(new_shape_phi_x)
        
        main_loss = normaliser(dot_prod / eigenvalue, phi_x, axis=(0, 1))
    else:
        main_loss = normaliser(dot_prod / eigenvalue, phi_x)

    return main_loss


def train_with_logger(
        model, F, dists, external_input_dist=None, dist_requires_dim=False, num_epochs=1000,
        batch_size=512,
        dynamics_dim=1, decay_module=None, logger=None, lr_scheduler=None,
        eigenvalue=1.0, print_every_num_epochs=10, device='cpu', param_specific_hyperparams=[],
        normaliser=partial(shuffle_normaliser, axis=None, return_terms=True),
        # normaliser=partial(variance_normaliser, axis=None, return_terms=True),
        verbose=False,
        restrict_to_distribution_lambda=0,
        ext_inp_batch_size=None,
        ext_inp_reg_coeff=0,
        metadata=None,
        optimizer = partial(torch.optim.Adam,lr=1e-4),
        fixed_x_batch=None,
        fixed_external_inputs=None,
        epoch_callbacks=[],
        dist_weights=None,
        balance_loss_lambda=0.01,
        RHS_function=lambda psi: psi - psi ** 3,
        use_jvp=True,
):
    if len(param_specific_hyperparams) == 0:
        param_specific_hyperparams = model.parameters()
    else:
        param_specific_hyperparams = _evaluate_param_specific_hyperparams(model, param_specific_hyperparams)

    optimizer = optimizer(param_specific_hyperparams)
    if lr_scheduler is not None:
        lr_scheduler = lr_scheduler(optimizer)

    sample_shape = (batch_size, dynamics_dim) if dist_requires_dim else (batch_size,)

    # Set default weights to 1.0 if not provided
    if dist_weights is None:
        dist_weights = [1.0] * len(dists)
    elif len(dist_weights) != len(dists):
        raise ValueError(f"Length of dist_weights ({len(dist_weights)}) must match length of dists ({len(dists)})")

    for epoch in range(num_epochs):
        # Run any epoch callbacks
        for callback in epoch_callbacks:
            callback(optimizer, epoch)
            
        total_loss = 0
        normalised_losses = []
        reg_term_values = []
        balance_losses = []

        for i, dist in enumerate(dists):
            external_input_dist_single = None if external_input_dist is None else external_input_dist[i]

            if fixed_x_batch is not None:
                x_batch = fixed_x_batch.to(device)
                batch_size = x_batch.shape[0]
            else:
                x_batch = dist.sample(sample_shape=sample_shape).to(device)

            x_batch.requires_grad_(True)
            input_to_model = x_batch
            external_inputs = None

            if external_input_dist_single is not None:
                if fixed_external_inputs is not None:
                    external_inputs = fixed_external_inputs.to(device)
                    ext_inp_batch_size = external_inputs.shape[0]
                else:
                    ext_inp_batch_size = ext_inp_batch_size or batch_size
                    ext_sample_shape = [ext_inp_batch_size] + ([dynamics_dim] if dist_requires_dim else [])
                    external_inputs = external_input_dist_single.sample(sample_shape=ext_sample_shape).to(device)

                repeats = batch_size // ext_inp_batch_size
                remainder = batch_size % ext_inp_batch_size
                external_inputs = external_inputs.repeat(repeats, *([1] * (external_inputs.dim() - 1)))

                input_to_model = torch.cat((input_to_model, external_inputs), dim=-1)

            F_inputs = [x_batch] + ([] if external_input_dist_single is None else [external_inputs])
            F_x = F(*F_inputs)

            phi_x, dot_prod = compute_phi_and_dot_prod(
                model=model,
                x_batch=x_batch,
                input_to_model=input_to_model,
                F_x=F_x,
                use_jvp=use_jvp,
                external_inputs=external_inputs,
            )
            # main_loss = torch.mean((dot_prod/eigenvalue - phi_x) ** 2)
            rhs_phi_x = RHS_function(phi_x)
            main_loss = torch.mean((dot_prod/eigenvalue - rhs_phi_x) ** 2)

            # normalised_loss, _, _ = normaliser(dot_prod/eigenvalue, phi_x, axis=None, return_terms=True)
            normalised_loss, _, _ = normaliser(dot_prod/eigenvalue, rhs_phi_x, axis=None, return_terms=True)
            normalised_losses.append(normalised_loss.item())
            total_loss += dist_weights[i] * normalised_loss

            if restrict_to_distribution_lambda > 0:
                reg_loss = restrict_to_distribution_loss(x_batch, phi_x, dist, threshold=-4.0)
                total_loss += restrict_to_distribution_lambda * reg_loss

            if balance_loss_lambda > 0:
                balance_loss = torch.abs( torch.mean(phi_x[..., 1]) / torch.std(phi_x[..., 1]) )
                balance_losses.append(balance_loss.item())
                total_loss += balance_loss_lambda * balance_loss

            if external_input_dist_single is not None and ext_inp_reg_coeff > 0:
                group_counts = [batch_size // ext_inp_batch_size + (1 if i < (batch_size % ext_inp_batch_size) else 0)
                                for i in range(ext_inp_batch_size)]
                start_idx = 0
                group_mean_squared_values = []
                for count in group_counts:
                    group_phi = phi_x[start_idx:start_idx + count]
                    group_mean_sq = torch.mean(group_phi ** 2)
                    group_mean_squared_values.append(group_mean_sq)
                    start_idx += count
                group_mean_squared_values = torch.stack(group_mean_squared_values)
                reg_term_value = (torch.std(group_mean_squared_values) / torch.mean(group_mean_squared_values)) ** 2
                reg_loss = ext_inp_reg_coeff * reg_term_value
                total_loss += reg_loss
                reg_term_values.append(reg_term_value.item())
        # Log metrics
        metrics = {
            "Loss/Total": total_loss.item(),
            "Loss/Main": main_loss.item(),
            "Learning Rate": optimizer.param_groups[0]['lr'],
        }

        # Add normalised losses for each distribution to metrics
        for i, (n_loss, reg_term_value) in enumerate(zip(normalised_losses, reg_term_values)):
            metrics[f"Loss/NormalisedLoss_Dist_{i}"] = n_loss
            metrics[f"Loss/RegTermValue_Dist_{i}"] = reg_term_value
            if balance_loss_lambda > 0:
                metrics[f"Loss/BalanceLoss_Dist_{i}"] = balance_losses[i]

        # Add metadata to metrics if provided
        if metadata is not None:
            metrics.update(metadata)

        log_metrics(logger, metrics, epoch)

        # Backpropagation and optimization step
        optimizer.zero_grad()
        total_loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        param_norm = (sum([torch.linalg.norm(g) for g in grads]).item() if len(grads) > 0 else 0.0)
        # Replace any NaN gradients with 0 to maintain stability
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data[torch.isnan(param.grad.data)] = 0
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        if epoch % print_every_num_epochs == 0 and verbose:
            print(
                f"Epoch {epoch}, Loss: {total_loss.item()}, Normalised losses: {[n_loss for n_loss in normalised_losses]}, "
                f"Regularisation term values: {[reg_term_value for reg_term_value in reg_term_values]}, "
                + (f"Balance losses: {balance_losses}, " if balance_loss_lambda > 0 else "") +
                f"param norm: {param_norm}, Learning Rate: {optimizer.param_groups[0]['lr']}, "
                f"len(model.parameters()): {len(list(model.parameters()))}, "
            )
            # if x_batch.shape[-1] == 1:
            #
            #     fig, axs = plt.subplots(2, 1, sharex=True)
            #     ax = axs[0]
            #     ax.axhline(0, ls='dashed', c='grey')
            #     ax.scatter(x_batch.detach().cpu().numpy(), phi_x.detach().cpu().numpy(), s=3)
            #     ax.scatter(x_batch.detach().cpu().numpy(), dot_prod.detach().cpu().numpy(), s=3)
            #
            #     ax = axs[1]
            #     ax.hist(x_batch.detach().cpu().numpy(), density=True, bins=50)
            #     plt.savefig(f"test_outputs/{epoch}.png")
            #     plt.close()

def _evaluate_param_specific_hyperparams(model, param_specific_hyperparams):
    """Helper function to evaluate parameter-specific hyperparameters."""
    params = dict(model.named_parameters())
    param_specific_hyperparams_complete = []
    for param_list in param_specific_hyperparams:
        new_param_list = dict(param_list)
        new_param_list['params'] = [params[p] for p in param_list['params']]
        param_specific_hyperparams_complete.append(new_param_list)
    return param_specific_hyperparams_complete
    