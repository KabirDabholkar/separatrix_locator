"""
Gradient descent functions for separatrix location.

This module contains the gradient descent algorithms used to find separatrices
by optimizing on the learned Koopman eigenfunctions.
"""

import torch
from functools import partial
from typing import Optional, Tuple, Union, Callable, Any
from copy import deepcopy
import numpy as np


def process_initial_conditions(
    func, init_cond_dist, initial_conditions, input_dim, dist_needs_dim,
    batch_size, threshold, resample_above_threshold, external_inputs_fixed=None
):
    """
    Process initial conditions for optimization.

    If `initial_conditions` is None and resampling is requested, it samples candidates
    from `init_cond_dist` until a full batch of points satisfying 
    func(concat(candidate, external_inputs_fixed)) > threshold is obtained.
    
    Args:
        func: Callable, a differentiable scalar-valued function that accepts a single tensor.
        init_cond_dist: A PyTorch distribution for sampling initial conditions.
        initial_conditions: Optional tensor of initial conditions.
        input_dim: Dimension of the initial conditions.
        dist_needs_dim: Boolean indicating whether to add an extra dimension to the sample.
        batch_size: Number of initial points to optimize.
        threshold: Threshold value used to filter points.
        resample_above_threshold: If True, only accept points where func(concat(point, external_inputs_fixed)) > threshold.
        external_inputs_fixed: Optional tensor of external inputs (detached) to be concatenated with candidates for threshold evaluation.

    Returns:
        initial_conditions: A tensor of initial conditions with gradients enabled.
        batch_size: The effective batch size.
        orig_indices: A tensor containing the indices of the original initial conditions.
    """
    def build_input(candidates):
        if external_inputs_fixed is not None:
            return torch.cat((candidates, external_inputs_fixed), dim=-1)
        else:
            return candidates

    if initial_conditions is None:
        if resample_above_threshold:
            if threshold is None:
                raise ValueError("When resample_above_threshold is True, threshold must be provided.")
            accepted_points = []
            while sum(pt.shape[0] for pt in accepted_points) < batch_size:
                sample_shape = [batch_size] + ([input_dim] if dist_needs_dim else [])
                candidates = init_cond_dist.sample(sample_shape=sample_shape)
                with torch.no_grad():
                    candidate_input = build_input(candidates)
                    candidate_losses = func(candidate_input)
                mask = candidate_losses[..., 0] > threshold
                valid = candidates[mask]
                if valid.numel() > 0:
                    accepted_points.append(valid)
            accepted_points = torch.cat(accepted_points, dim=0)[:batch_size]
            initial_conditions = accepted_points.requires_grad_()
        else:
            sample_shape = [batch_size] + ([input_dim] if dist_needs_dim else [])
            initial_conditions = init_cond_dist.sample(sample_shape=torch.Size(sample_shape))
            
            # Ensure the shape is correct: (batch_size, input_dim)
            if initial_conditions.dim() == 1:
                initial_conditions = initial_conditions.unsqueeze(-1)
            elif initial_conditions.dim() == 2 and initial_conditions.shape[1] != input_dim:
                if initial_conditions.shape[0] == input_dim:
                    initial_conditions = initial_conditions.T
            
            # Clone to ensure it's a leaf tensor, then set requires_grad
            initial_conditions = initial_conditions.clone().requires_grad_()
        orig_indices = torch.arange(initial_conditions.shape[0])
        batch_size = initial_conditions.shape[0]
    else:
        if resample_above_threshold:
            batch_size = initial_conditions.shape[0]
            orig_indices = torch.arange(batch_size)
            initial_conditions = initial_conditions.requires_grad_()
        else:
            with torch.no_grad():
                candidate_input = build_input(initial_conditions)
                candidate_losses = func(candidate_input)
            mask = candidate_losses[..., 0] > threshold
            orig_indices = torch.nonzero(mask, as_tuple=False).flatten()
            initial_conditions = initial_conditions[mask].requires_grad_()
            batch_size = initial_conditions.shape[0]

    return initial_conditions, batch_size, orig_indices


def runGD_basic(
    func, initial_conditions=None, external_inputs=None, num_steps=100,
    partial_optim=partial(torch.optim.Adam, lr=1e-2), threshold=5e-2,
    lr_scheduler=None, optimize_initial_conditions=True,
    optimize_external_inputs=False, return_indices=False, return_mask=False,
    save_trajectories_every=10000
):
    """
    Basic gradient descent optimization for finding separatrices.
    
    Args:
        func: Callable, a differentiable scalar-valued function.
        initial_conditions: Optional tensor of initial conditions.
        external_inputs: Optional tensor of external inputs.
        num_steps: Number of optimization steps.
        partial_optim: Partial function for creating the optimizer.
        threshold: Threshold value for filtering points.
        lr_scheduler: Optional learning rate scheduler.
        optimize_initial_conditions: If True, initial_conditions are optimized.
        optimize_external_inputs: If True, external_inputs are optimized.
        return_indices: If True, also returns indices.
        return_mask: If True, returns a mask.
        save_trajectories_every: Save trajectories every N iterations.
        
    Returns:
        trajectories_initial, below_threshold_points, and optionally indices/mask
    """
    if hasattr(threshold, 'start_threshold'):
        start_threshold = threshold['start_threshold']
        end_threshold = threshold['end_threshold']
    else:
        start_threshold = threshold
        end_threshold = threshold

    # Set gradient requirements based on optimization flags
    if optimize_initial_conditions:
        initial_conditions = initial_conditions.requires_grad_()
    else:
        initial_conditions = initial_conditions.detach()
    if external_inputs is not None:
        if optimize_external_inputs:
            external_inputs = external_inputs.requires_grad_()
        else:
            external_inputs = external_inputs.detach()

    # Collect parameters to optimize
    params_to_optimize = []
    if optimize_initial_conditions:
        params_to_optimize.append(initial_conditions)
    if optimize_external_inputs:
        params_to_optimize.append(external_inputs)
    optimizer = partial_optim(params_to_optimize)
    scheduler = lr_scheduler(optimizer) if lr_scheduler else None

    trajectories_initial = []
    trajectories_external = []
    below_threshold_mask = torch.zeros(initial_conditions.shape[0], dtype=torch.bool)
    below_threshold_points = []
    below_threshold_indices = []

    for step in range(num_steps):
        if step % save_trajectories_every == 0:
            trajectories_initial.append(initial_conditions.clone().detach().cpu().numpy())
            if external_inputs is not None:
                trajectories_external.append(external_inputs.clone().detach().cpu().numpy())

        optimizer.zero_grad()

        # Concatenate initial conditions and external inputs along the last dimension
        inputs = initial_conditions
        if external_inputs is not None:
            inputs = torch.cat((initial_conditions, external_inputs), dim=-1)
        losses = func(inputs)
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Identify initial conditions that drop below the threshold
        newly_below_threshold = (losses[..., 0] < end_threshold) & ~below_threshold_mask
        if newly_below_threshold.any():
            indices = newly_below_threshold.nonzero(as_tuple=True)[0]
            below_threshold_selection = initial_conditions[indices].detach().clone()
            if external_inputs is not None:
                below_threshold_selection = torch.cat([
                    below_threshold_selection, external_inputs[indices].detach().clone()
                ], axis=-1)
            below_threshold_points.append(below_threshold_selection)
            below_threshold_mask[indices] = True

    # Stack trajectories
    trajectories_initial = np.stack(trajectories_initial) if len(trajectories_initial) > 0 else None

    if below_threshold_points:
        below_threshold_points = torch.cat(below_threshold_points, dim=0)
        below_threshold_indices = torch.cat(below_threshold_indices, dim=0)
    else:
        below_threshold_points = torch.empty((0,))
        below_threshold_indices = torch.empty((0,), dtype=torch.long)

    to_return = [trajectories_initial, below_threshold_points]
    if return_indices:
        to_return += [below_threshold_indices]
    if return_mask:
        mask = torch.zeros(initial_conditions.shape[0], dtype=torch.bool)
        mask[below_threshold_indices] = True
        to_return += [mask]
    return tuple(to_return)


def runGD(
    func, init_cond_dist, external_input_dist=None, initial_conditions=None,
    external_inputs=None, input_dim=1, external_input_dim=None, dist_needs_dim=True,
    num_steps=100, partial_optim=partial(torch.optim.Adam, lr=1e-2), batch_size=64,
    threshold=5e-2, lr_scheduler=None, resample_above_threshold=False,
    optimize_initial_conditions=True, optimize_external_inputs=False,
    return_indices=False, return_mask=False, save_trajectories_every=10000,
    device: Optional[Union[str, torch.device]] = None
):
    """
    Advanced gradient descent optimization for finding separatrices with distribution sampling.
    
    Optimizes a scalar-valued function using full-batch Adam over initial conditions 
    and optionally external inputs, and records trajectories.

    Args:
        func: Callable, a differentiable scalar-valued function that accepts a single tensor.
        init_cond_dist: A PyTorch distribution for sampling initial conditions.
        external_input_dist: A PyTorch distribution for sampling external inputs.
        initial_conditions: Optional tensor of initial conditions. If None, they are sampled.
        external_inputs: Optional tensor of external inputs. If None and external_input_dist is provided, they are sampled.
        input_dim: Dimension of the initial conditions.
        external_input_dim: Dimension of the external inputs. If None, defaults to input_dim.
        dist_needs_dim: Boolean indicating whether to add an extra dimension to the sample.
        num_steps: Number of optimization steps.
        partial_optim: Partial function for creating the optimizer.
        batch_size: Number of points to optimize (for both initial conditions and external inputs).
        threshold: Threshold value for filtering initial conditions (applied on the first output of func).
        lr_scheduler: Optional learning rate scheduler.
        resample_above_threshold: If True, only initial conditions with func(concat(point, external_inputs_fixed)) > threshold
                                  are used. (When provided initial_conditions, all are assumed valid.)
        optimize_initial_conditions: If True, initial_conditions are optimized.
        optimize_external_inputs: If True, external_inputs are optimized. Otherwise, they remain fixed.
        return_indices: If True, also returns the original indices of the points that dropped below threshold.
        return_mask: If True, returns a mask indicating which initial conditions dropped below threshold.
        save_trajectories_every: Save trajectories every N iterations.

    Returns:
        A tuple containing:
            trajectories_initial: Tensor of shape (num_steps, batch_size, input_dim) recording the initial conditions trajectory.
            trajectories_external: Tensor of shape (num_steps, batch_size, external_input_dim) recording the external inputs trajectory.
            below_threshold_points: Tensor containing points (from initial_conditions) that dropped below the threshold.
            below_threshold_indices (optional): Tensor of original indices corresponding to below_threshold_points (if return_indices=True).
            mask (optional): Boolean tensor mask indicating which initial conditions dropped below threshold (if return_mask=True).
    """
    if hasattr(threshold, 'start_threshold'):
        start_threshold = threshold['start_threshold']
        end_threshold = threshold['end_threshold']
    else:
        start_threshold = threshold
        end_threshold = threshold

    if external_input_dim is None:
        external_input_dim = 0

    # First, sample or use provided external inputs
    if external_input_dist is not None:
        sample_shape = [batch_size] + ([external_input_dim] if dist_needs_dim else [])
        external_inputs = external_inputs if external_inputs is not None else external_input_dist.sample(sample_shape=sample_shape)
    else:
        if external_input_dim > 0:
            external_inputs = external_inputs if external_inputs is not None else torch.zeros((batch_size, external_input_dim))
        else:
            # No external inputs needed
            external_inputs = None
    
    # For threshold filtering, use a fixed copy (detached) of external_inputs
    external_inputs_fixed = external_inputs.detach() if external_inputs is not None else None

    # Process initial conditions using the fixed external inputs
    initial_conditions, batch_size, orig_indices = process_initial_conditions(
        func, init_cond_dist, initial_conditions, input_dim, dist_needs_dim, batch_size, start_threshold,
        resample_above_threshold, external_inputs_fixed=external_inputs_fixed
    )

    # Set gradient requirements based on optimization flags
    if optimize_initial_conditions:
        initial_conditions = initial_conditions.requires_grad_()
    else:
        initial_conditions = initial_conditions.detach()
    if external_inputs is not None:
        if optimize_external_inputs:
            external_inputs = external_inputs.requires_grad_()
        else:
            external_inputs = external_inputs.detach()

    # Collect parameters to optimize
    params_to_optimize = []
    if optimize_initial_conditions:
        params_to_optimize.append(initial_conditions)
    if optimize_external_inputs and external_inputs is not None:
        params_to_optimize.append(external_inputs)
    optimizer = partial_optim(params_to_optimize)
    scheduler = lr_scheduler(optimizer) if lr_scheduler else None

    trajectories_initial = []
    trajectories_external = []
    below_threshold_points = []
    below_threshold_indices = []

    # Determine target device
    if device is not None:
        target_device = torch.device(device)
    else:
        # Infer from func/model if possible
        if hasattr(func, 'func') and hasattr(func.func, 'parameters'):
            target_device = next(func.func.parameters()).device
        elif hasattr(func, 'parameters'):
            target_device = next(func.parameters()).device
        else:
            target_device = torch.device('cpu')

    # Move tensors to target device
    initial_conditions = initial_conditions.to(target_device)
    if external_inputs is not None:
        external_inputs = external_inputs.to(target_device)

    below_threshold_mask = torch.zeros(batch_size, dtype=torch.bool, device=target_device)

    for step in range(num_steps):
        optimizer.zero_grad()

        # Concatenate initial conditions and external inputs along the last dimension
        if external_inputs is not None:
            inputs = torch.cat((initial_conditions, external_inputs), dim=-1)
        else:
            inputs = initial_conditions
        
        # Ensure inputs are on the correct device
        inputs = inputs.to(target_device)
        losses = func(inputs)
        loss = losses.sum()
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()

        # Identify initial conditions that drop below the threshold
        newly_below_threshold = (losses[..., 0] < end_threshold) & ~below_threshold_mask
        if newly_below_threshold.any():
            indices = newly_below_threshold.nonzero(as_tuple=True)[0]
            below_threshold_selection = initial_conditions[indices].detach().clone()
            if external_inputs is not None:
                below_threshold_selection = torch.cat([
                    below_threshold_selection, external_inputs[indices].detach().clone()
                ], axis=-1)
            below_threshold_points.append(below_threshold_selection)
            below_threshold_indices.append(orig_indices[indices].detach().clone())
            below_threshold_mask[indices] = True

    # Stack trajectories
    trajectories_initial = None  # torch.stack(trajectories_initial)

    if below_threshold_points:
        below_threshold_points = torch.cat(below_threshold_points, dim=0)
        below_threshold_indices = torch.cat(below_threshold_indices, dim=0)
    else:
        below_threshold_points = torch.empty((0, input_dim))
        below_threshold_indices = torch.empty((0,), dtype=torch.long)

    to_return = [trajectories_initial, below_threshold_points]
    if return_indices:
        to_return += [below_threshold_indices]
    if return_mask:
        mask = torch.zeros(orig_indices.shape[0], dtype=torch.bool)
        mask[below_threshold_indices] = True
        to_return += [mask]
    return tuple(to_return)
