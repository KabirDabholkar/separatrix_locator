"""
Main SeparatrixLocator class for locating separatrices in dynamical systems.

This is the core class that orchestrates the training of multiple Koopman eigenfunction
models and uses them to locate separatrices through gradient descent.
"""

from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from functools import partial
from pathlib import Path
from typing import Iterable, Optional, Union, Dict, Any, List, Sequence, Tuple
import os

from separatrix_locator.utils.compose import compose
from separatrix_locator.utils.interpolation import generate_curves_between_points
from separatrix_locator.plotting.hermite import find_separatrix_along_curve_using_ODE
import matplotlib.pyplot as plt


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
        num_models: int = 1, 
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

    def validate_with_curves(
        self,
        dynamics_function,
        attractors: Union[np.ndarray, torch.Tensor],
        num_curves: int = 100,
        num_points: int = 100,
        rand_scale: float = 3.0,
        alpha_lims: Tuple[float, float] = (0.0, 1.0),
        integration_time: float = 500.0,
        attractor_epsilon: float = 0.02,
        kmeans_random_state: int = 42,
        clustering_method: str = "attractor_eps",
        kef_component: int = 0,
        plot_pca: bool = False,
        plot_kef: bool = False,
        plot_scatter: bool = False,
    ) -> float:
        """
        Validate the separatrix locator by comparing curve-derived separatrix estimates
        with the zero-crossing of the Koopman eigenfunction predictions.

        Parameters
        ----------
        dynamics_function : Callable[[torch.Tensor], torch.Tensor]
            Vector field used to integrate points along each curve.
        attractors : array-like or torch.Tensor of shape (2, dim)
            Endpoints between which curves are generated.
        num_curves : int, default=100
            Number of Hermite curves to sample.
        num_points : int, default=100
            Number of evaluation points per curve.
        rand_scale : float, default=3.0
            Random tangent perturbation scale for curve generation.
        alpha_lims : tuple(float, float), default=(0.0, 1.0)
            Inclusive limits for the curve parameter alpha.
        integration_time : float, default=500.0
            Time horizon for the ODE integration inside the separatrix finder.
        attractor_epsilon : float, default=0.02
            Distance threshold used by the attractor-epsilon clustering method.
        kmeans_random_state : int, default=42
            Random state forwarded to k-means when clustering is enabled.
        clustering_method : str, default="attractor_eps"
            Basin classification method passed to `find_separatrix_along_curve_using_ODE`.
        kef_component : int, default=0
            Index of the KEF component used when searching for zero crossings.
        plot_pca : bool, default=False
            If True, render a PCA projection of the curves with basin labels and change points.
        plot_kef : bool, default=False
            If True, plot KEF values along each curve.
        plot_scatter : bool, default=False
            If True, scatter `change_points_alpha` against `zero_point_alphas`.

        Returns
        -------
        float
            R² score comparing ODE-derived separatrix alphas with KEF zero alphas.
        """
        if not self.models:
            raise ValueError("Models not trained. Call fit() before validation.")

        if isinstance(attractors, torch.Tensor):
            attractors_np = attractors.detach().cpu().numpy()
        else:
            attractors_np = np.asarray(attractors)
        if attractors_np.shape[0] != 2:
            raise ValueError("attractors must have shape (2, dim).")

        curves_np, alphas = generate_curves_between_points(
            attractors_np[0],
            attractors_np[1],
            num_curves=num_curves,
            num_points=num_points,
            rand_scale=rand_scale,
            lims=list(alpha_lims),
            return_alpha=True,
        )

        change_points_alpha, labels_bt = find_separatrix_along_curve_using_ODE(
            dynamics_function=dynamics_function,
            attractors=torch.tensor(attractors_np, dtype=torch.float32),
            alphas=alphas,
            curve_points=curves_np,
            integration_time=integration_time,
            attractor_epsilon=attractor_epsilon,
            kmeans_random_state=kmeans_random_state,
            clustering_method=clustering_method,
        )

        curves_tensor = torch.tensor(curves_np, dtype=torch.float32, device=self.device)
        kef_vals = self.predict(curves_tensor, no_grad=True)

        if kef_vals.dim() == 2:
            kef_component_vals = kef_vals
        elif kef_vals.dim() == 3:
            if kef_component >= kef_vals.shape[-1]:
                raise IndexError(
                    f"kef_component {kef_component} is out of range for KEF output size {kef_vals.shape[-1]}."
                )
            kef_component_vals = kef_vals[..., kef_component]
        else:
            raise ValueError(
                f"Unexpected KEF prediction shape {kef_vals.shape}. Expected rank 2 or 3 tensor."
            )

        kef_component_vals = kef_component_vals.cpu()
        abs_kef = torch.abs(kef_component_vals)
        zero_idx = torch.argmin(abs_kef, dim=1)
        alphas_tensor = torch.tensor(alphas, dtype=torch.float32)
        zero_point_alphas = alphas_tensor[zero_idx].detach().cpu().numpy()

        if plot_pca:
            num_curves, num_points, dim = curves_np.shape
            flat_points = curves_np.reshape(-1, dim)
            pca = PCA(n_components=2)
            curve_points_pca = pca.fit_transform(flat_points)
            curve_points_pca_bt = curve_points_pca.reshape(num_curves, num_points, 2)
            labels_bt_np = np.asarray(labels_bt)
            max_label = np.max(labels_bt_np[labels_bt_np >= 0]) if np.any(labels_bt_np >= 0) else 1.0

            fig, ax = plt.subplots(figsize=(7, 7))
            for i in range(num_curves):
                for j in range(num_points - 1):
                    label_val = labels_bt_np[i, j]
                    if label_val < 0 or max_label <= 0:
                        color = "gray"
                    else:
                        color = plt.cm.coolwarm(label_val / max_label)
                    ax.plot(
                        curve_points_pca_bt[i, j:j + 2, 0],
                        curve_points_pca_bt[i, j:j + 2, 1],
                        color=color,
                        alpha=0.8,
                        linewidth=1,
                    )
                change_alpha = change_points_alpha[i]
                change_idx = int(np.argmin(np.abs(alphas - change_alpha)))
                ax.plot(
                    curve_points_pca_bt[i, change_idx, 0],
                    curve_points_pca_bt[i, change_idx, 1],
                    marker="x",
                    markersize=8,
                    markeredgewidth=2,
                    color="black",
                    linestyle="None",
                    label="Change point" if i == 0 else None,
                )
            ax.set_title("Hermite curves (PCA) with basin labels\nChange points marked with x's")
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            if labels_bt_np.ndim == 2 and np.any(labels_bt_np >= 0):
                sm = plt.cm.ScalarMappable(cmap="coolwarm")
                sm.set_array(np.linspace(0, max_label, num=100))
                fig.colorbar(sm, ax=ax, label="Basin label")
            ax.legend(loc="upper right")
            plt.show()

        if plot_kef:
            kef_vals_np = kef_component_vals.detach().cpu().numpy()
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(alphas, kef_vals_np.T)
            ax.set_xlabel(r"Curve parameter $\alpha$")
            ax.set_ylabel(r"KEF value")
            ax.set_title("KEF values along curves")
            ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.4)
            plt.show()

        if plot_scatter:
            fig, ax = plt.subplots(figsize=(3.5, 3.5))
            ax.scatter(change_points_alpha, zero_point_alphas)
            ax.plot([alphas[0], alphas[-1]], [alphas[0], alphas[-1]], "k--", alpha=0.5)
            ax.set_xlabel(r"True separatrix $\alpha$")
            ax.set_ylabel(r"KEF zero $\alpha$")
            ax.set_title("Separatrix vs KEF zero")
            ax.set_aspect("equal", adjustable="box")
            plt.show()

        score = r2_score(change_points_alpha, zero_point_alphas)
        return float(score)
