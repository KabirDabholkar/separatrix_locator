"""
Main SeparatrixLocator class for locating separatrices in dynamical systems.

This is the core class that orchestrates the training of multiple Koopman eigenfunction
models and uses them to locate separatrices through gradient descent.
"""

from sklearn.base import BaseEstimator
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union, Dict, Any, List
import os

from separatrix_locator.utils.compose import compose


class SeparatrixLocator(BaseEstimator):
    """
    A tool for locating separatrices in black-box dynamical systems.
    
    This class trains multiple Koopman eigenfunction models and uses gradient descent
    to find separatrices in the learned eigenfunction landscape.
    
    Parameters:
    -----------
    num_models : int, default=10
        Number of models to train for ensemble separatrix location
    dynamics_dim : int, default=1
        Dimensionality of the dynamical system
    models : list,
        List of models to use for training.
    lr : float, default=1e-3
        Learning rate for training
    epochs : int, default=100
        Number of training epochs
    use_multiprocessing : bool, default=True
        Whether to use multiprocessing for training multiple models
    verbose : bool, default=False
        Whether to print training progress
    device : str, default="cpu"
        Device to use for training ("cpu" or "cuda")
    """
    
    def __init__(
        self, 
        num_models: int = 10, 
        dynamics_dim: int = 1,
        models: Optional[List[nn.Module]] = None,
        lr: float = 1e-4, 
        epochs: int = 100, 
        use_multiprocessing: bool = False, 
        verbose: bool = False, 
        device: str = "cpu"
    ):
        super().__init__()
        self.num_models = num_models
        self.lr = lr
        self.epochs = epochs
        self.use_multiprocessing = use_multiprocessing
        self.device = device
        self.verbose = verbose
        self.dynamics_dim = dynamics_dim
        self.models = models if models is not None else []
        self.scores = None
        self.functions_for_gradient_descent = None

    def fit(self, func, distribution, **kwargs):
        """
        Train the ensemble of models on the given dynamical system.
        
        Parameters:
        -----------
        func : callable
            The dynamical system function f(x) -> dx/dt
        distribution : torch.distributions.Distribution or list
            Distribution for sampling initial conditions
        **kwargs : dict
            Additional training parameters
        """
        from .training import train_with_logger
        
        # Accept a single distribution or a list. Always work with a list.
        if not isinstance(distribution, Iterable) or isinstance(distribution, (torch.Tensor, bytes, str)):
            distributions = [distribution]
        else:
            distributions = list(distribution)
        
        # Allow overriding the optimizer via kwargs; default to Adam(lr=self.lr)
        optimizer_factory = kwargs.pop('optimizer', partial(torch.optim.Adam, lr=self.lr))

        train_single_model_ = partial(
            train_with_logger, 
            F=func, 
            dists=distributions, 
            dynamics_dim=self.dynamics_dim, 
            num_epochs=self.epochs,
            optimizer=optimizer_factory,
            **kwargs
        )


        if self.verbose:
            print(f"Training {len(self.models)} models...")

        if self.use_multiprocessing:
            mp.set_start_method('spawn', force=True)
            devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]

            with mp.Pool(processes=len(devices)) as pool:
                results = [
                    pool.apply_async(
                        train_single_model_,
                        args=(self.models[i],),
                        kwds=dict(verbose=self.verbose, device=self.device, metadata={"model_id": int(i)}),
                    ) for i in range(self.num_models)
                ]
                # Wait for all results
                for result in results:
                    result.get()
        else:
            for i, model in enumerate(self.models):
                if self.verbose:
                    print(f"Training model {i+1}/{self.num_models}")
                train_single_model_(
                    model,
                    verbose=self.verbose,
                    device=self.device,
                    metadata={"model_id": int(i)}
                )

        return self
    
    def _train_simple(self, func, distribution, **kwargs):
        """Simplified training method as fallback."""
        if len(self.models) == 0:
            self.init_models()
        
        if self.verbose:
            print("Using simplified training method...")
            print(f"Training {len(self.models)} models...")
        
        # Simple training loop for each model
        for i, model in enumerate(self.models):
            if self.verbose:
                print(f"Training model {i+1}/{len(self.models)}...")
            
            optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
            
            for epoch in range(self.epochs):
                # Sample from distribution
                x = distribution.sample((1000,)).to(self.device)
                
                # Compute dynamics
                with torch.no_grad():
                    F_x = func(x)
                
                # Compute eigenfunction and its derivative
                x.requires_grad_(True)
                phi_x = model(x)
                
                # Compute gradient of eigenfunction
                phi_x_prime = torch.autograd.grad(
                    phi_x.sum(), x, create_graph=True, retain_graph=True
                )[0]
                
                # Koopman eigenfunction loss: ||phi'(x) F(x) - lambda phi(x)||^2
                eigenvalue = kwargs.get('eigenvalue', 1.0)
                dot_prod = (phi_x_prime * F_x).sum(dim=-1, keepdim=True)
                loss = torch.mean((dot_prod - eigenvalue * phi_x)**2)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if self.verbose and epoch % 10 == 0:
                    print(f"  Epoch {epoch}, Loss: {loss.item():.4f}")
        
        if self.verbose:
            print("Training completed!")
    
    def predict(self, inputs: torch.Tensor, no_grad: bool = True) -> torch.Tensor:
        """
        Predict the KEF outputs for the given inputs using the trained models.

        Parameters:
        -----------
        inputs : torch.Tensor
            Input tensor of shape (batch_size, input_dim)
        no_grad : bool, default=True
            If True, run without gradient computation

        Returns:
        --------
        torch.Tensor
            KEF outputs of shape (num_models, batch_size, output_dim)
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() first.")
            
        kef_outputs = []
        for model in self.models:
            if no_grad:
                with torch.no_grad():
                    kef_output = model(inputs.to(self.device))
            else:
                kef_output = model(inputs.to(self.device))
            kef_outputs.append(kef_output.cpu())
        return torch.concat(kef_outputs, axis=-1)

    def to(self, device: str):
        """Move all models to the specified device."""
        self.device = device
        for model in self.models:
            model.to(device)

    def score(self, func, distribution, **kwargs):
        """
        Evaluate the models on the given function and distribution.
        
        Returns:
        --------
        torch.Tensor
            Scores for each model
        """
        from .training import eval_loss
        
        scores = []
        for model in self.models:
            if isinstance(distribution, Iterable):
                model_scores = []
                for dist in distribution:
                    score = eval_loss(model, func, dist, dynamics_dim=self.dynamics_dim, **kwargs)
                    model_scores.append(score)
                scores.append(torch.stack(model_scores))
            else:
                score = eval_loss(model, func, distribution, dynamics_dim=self.dynamics_dim, **kwargs)
                scores.append(score)
        self.scores = scores
        return torch.stack(self.scores)
    
    def save_models(self, savedir: str, filename: Optional[str] = None):
        """Save all trained models to disk."""
        os.makedirs(Path(savedir) / "models", exist_ok=True)
        for i, model in enumerate(self.models):
            if filename is None:
                save_filename = f"{model.__class__.__name__}_{i}.pt"
            else:
                save_filename = f"{filename}_{i}.pt"
            torch.save(model.state_dict(), Path(savedir) / "models" / save_filename)

    def load_models(self, savedir: str, filename: Optional[str] = None):
        """Load trained models from disk."""
        if not self.models:
            self.init_models()
            
        for i, model in enumerate(self.models):
            if filename is None:
                load_filename = f"{model.__class__.__name__}_{i}.pt"
            else:
                load_filename = f"{filename}_{i}.pt"
            # Try new PyTorch API with weights_only; fallback for older versions
            try:
                state_dict = torch.load(
                    Path(savedir) / "models" / load_filename,
                    weights_only=True,
                    map_location=torch.device(self.device),
                )
            except TypeError:
                state_dict = torch.load(
                    Path(savedir) / "models" / load_filename,
                    map_location=torch.device(self.device),
                )
            self.models[i].load_state_dict(state_dict)

    def filter_models(self, threshold: float):
        """Filter models based on their scores."""
        assert self.scores is not None, "Scores not available. Call score() first."
        scores = self.scores
        self.models = [m for m, s in zip(self.models, scores) if torch.mean(s) < threshold]
        self.num_models = len(self.models)
        return self

    def prepare_models_for_gradient_descent(self, distribution, **kwargs):
        """
        Prepare all models for gradient descent by composing and normalizing their functions.
        
        This is a key step that transforms the raw model outputs into the proper
        Koopman eigenfunction form for separatrix location.
        """
        if self.verbose:
            print('Preparing models for gradient descent...')
            
        # Compose the functions without normalization first
        self.functions_for_gradient_descent = [
            self._compose_model_functions(model, **kwargs) for model in self.models
        ]
        
        # Then normalize all functions
        self.functions_for_gradient_descent = self._normalize_functions(
            self.functions_for_gradient_descent,
            distribution,
            **kwargs
        )
        
        if self.verbose:
            print('Models are prepared for gradient descent.')
        return self.functions_for_gradient_descent

    def _compose_model_functions(self, model, **kwargs):
        """Compose the transformation functions for a single model without normalization."""
        # Start with pre-functions if provided
        functions = []
        if 'pre_functions' in kwargs:
            functions.extend(kwargs['pre_functions'])
            
        # Add the main chain of functions
        functions.extend([
            torch.log,
            lambda x: x + 1,
            torch.exp,
            partial(torch.sum, dim=-1, keepdims=True),
            torch.log,
            torch.abs,
            model
        ])
        
        # Add post-functions if provided
        if 'post_functions' in kwargs:
            functions.extend(kwargs['post_functions'])
            
        return compose(*functions)

    def _normalize_functions(self, functions, distribution, dist_needs_dim=False, **kwargs):
        """Normalize the given functions using samples from the distribution."""
        normalized_functions = []
        for f in functions:
            # Sample initial conditions
            shape = [1000] + ([self.dynamics_dim] if dist_needs_dim else [])
            samples_ic = distribution.sample(sample_shape=torch.Size(shape))
            
            # Ensure samples have the right shape (batch_size, input_dim)
            if samples_ic.dim() == 1:
                # If 1D, reshape to (batch_size, 1)
                samples_ic = samples_ic.unsqueeze(-1)
            elif samples_ic.dim() == 2 and samples_ic.shape[1] != self.dynamics_dim:
                # If 2D but wrong dimension, transpose
                if samples_ic.shape[0] == self.dynamics_dim:
                    samples_ic = samples_ic.T
            
            if self.verbose:
                print(f"DEBUG: samples_ic shape: {samples_ic.shape}")
                print(f"DEBUG: dynamics_dim: {self.dynamics_dim}")
                print(f"DEBUG: dist_needs_dim: {dist_needs_dim}")
            
            # Handle external inputs if provided
            if "external_input_dist" in kwargs:
                ext_input_dist = kwargs["external_input_dist"]
                ext_input_dim = kwargs.get("external_input_dim", 0)
                if ext_input_dim > 0:
                    shape_ext = [1000] + ([ext_input_dim] if dist_needs_dim else [])
                    samples_ext = ext_input_dist.sample(sample_shape=shape_ext)
                    if self.verbose:
                        print(f"DEBUG: samples_ext shape: {samples_ext.shape}")
                    # Ensure compatible shapes
                    if samples_ic.shape[0] != samples_ext.shape[0]:
                        min_batch_size = min(samples_ic.shape[0], samples_ext.shape[0])
                        samples_ic = samples_ic[:min_batch_size]
                        samples_ext = samples_ext[:min_batch_size]
                else:
                    # Create empty external inputs
                    if len(samples_ic.shape) == 3:
                        samples_ext = torch.zeros(samples_ic.shape[0], 0, samples_ic.shape[2])
                    else:
                        samples_ext = torch.zeros(samples_ic.shape[0], 0)
            else:
                # Create empty external inputs
                if len(samples_ic.shape) == 3:
                    samples_ext = torch.zeros(samples_ic.shape[0], 0, samples_ic.shape[2])
                else:
                    samples_ext = torch.zeros(samples_ic.shape[0], 0)
            
            # Combine samples and calculate normalization
            if samples_ext.shape[1] > 0:
                combined_samples = torch.cat((samples_ic, samples_ext), dim=-1)
            else:
                combined_samples = samples_ic
            
            # Ensure samples are on the correct device
            combined_samples = combined_samples.to(self.device)
                
            norm_val = float(torch.mean(torch.sum(f(combined_samples) ** 2, dim=-1)).sqrt().detach().cpu().numpy())
            
            # Normalize the function
            normalized_f = compose(lambda x: x / norm_val, f)
            normalized_functions.append(normalized_f)
            
        return normalized_functions

    def find_separatrix(self, distribution, dist_needs_dim=False, **kwargs):
        """
        Find separatrix using gradient descent on the learned eigenfunctions.
        
        Parameters:
        -----------
        distribution : torch.distributions.Distribution
            Distribution for sampling initial conditions for gradient descent
        dist_needs_dim : bool, default=False
            Whether the distribution requires explicit dimensionality
        **kwargs : dict
            Additional parameters for gradient descent
        
        Returns:
        --------
        tuple
            (trajectories, below_threshold_points) from gradient descent
        """
        if self.functions_for_gradient_descent is None:
            raise ValueError("Models not prepared for gradient descent. Call prepare_models_for_gradient_descent() first.")
        
        from .gradient_descent import runGD
        
        all_traj, all_below = [], []
        
        for f in self.functions_for_gradient_descent:
            ret = runGD(
                f,
                distribution,
                input_dim=self.dynamics_dim,
                dist_needs_dim=dist_needs_dim,
                device=self.device,
                **kwargs
            )
            all_traj.append(ret[0])
            all_below.append(ret[1])
        
        return all_traj, all_below
