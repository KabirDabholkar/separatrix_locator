"""
Configuration system for separatrix locator experiments.

This module provides configuration classes for different aspects of experiments:
- Dynamics configuration
- Model configuration  
- Distribution configuration
- Training configuration
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Union
import torch
from pathlib import Path


@dataclass
class DynamicsConfig:
    """Configuration for dynamical systems."""
    
    # System type and parameters
    system_type: str = "Bistable2D"
    dim: int = 2
    bistable_axis: int = 0
    
    # Additional system parameters (can be extended for other systems)
    system_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.dim < 1:
            raise ValueError("dim must be >= 1")
        if not (0 <= self.bistable_axis < self.dim):
            raise ValueError("bistable_axis must be in [0, dim)")


@dataclass
class ModelConfig:
    """Configuration for neural network models."""
    
    # Model type and architecture
    model_type: str = "ResNet"
    
    # ResNet parameters
    hidden_size: int = 128
    num_layers: int = 5
    num_kernels: int = 10
    output_dim: int = 1
    input_scale_factor: float = 1.0
    scale_last_layer_by_inv_sqrt_hidden: bool = True
    
    # DeepKoopmanModel parameters
    hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    dropout: float = 0.1
    
    # KoopmanEigenfunctionModel parameters
    hidden_dim: int = 128
    
    def get_model_kwargs(self, input_dim: int) -> Dict[str, Any]:
        """Get model initialization kwargs based on model type."""
        if self.model_type == "ResNet":
            return {
                "input_dim": input_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "num_kernels": self.num_kernels,
                "output_dim": self.output_dim,
                "input_scale_factor": self.input_scale_factor,
                "scale_last_layer_by_inv_sqrt_hidden": self.scale_last_layer_by_inv_sqrt_hidden
            }
        elif self.model_type == "DeepKoopmanModel":
            return {
                "input_dim": input_dim,
                "hidden_dims": self.hidden_dims,
                "output_dim": self.output_dim,
                "dropout": self.dropout
            }
        elif self.model_type == "KoopmanEigenfunctionModel":
            return {
                "input_dim": input_dim,
                "hidden_dim": self.hidden_dim,
                "output_dim": self.output_dim
            }
        elif self.model_type == "LinearModel":
            return {
                "input_dim": input_dim,
                "output_dim": self.output_dim
            }
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")


@dataclass
class DistributionConfig:
    """Configuration for initial condition distributions."""
    
    # Distribution type and parameters
    dist_type: str = "MultivariateGaussian"
    gaussian_scale: float = 2.0
    mean: Optional[List[float]] = None
    covariance_matrix: Optional[List[List[float]]] = None
    
    # Multiscaler parameters (for multiple scales)
    use_multiscaler: bool = True
    scales: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    
    def get_distribution_kwargs(self, dim: int) -> Dict[str, Any]:
        """Get distribution initialization kwargs based on distribution type."""
        if self.dist_type == "MultivariateGaussian":
            kwargs = {"dim": dim}
            
            if self.mean is not None:
                kwargs["mean"] = torch.tensor(self.mean, dtype=torch.float32)
            else:
                kwargs["mean"] = torch.zeros(dim)
                
            if self.covariance_matrix is not None:
                kwargs["covariance_matrix"] = torch.tensor(self.covariance_matrix, dtype=torch.float32)
            else:
                kwargs["covariance_matrix"] = self.gaussian_scale * torch.eye(dim)
                
            return kwargs
        else:
            raise ValueError(f"Unknown distribution type: {self.dist_type}")


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    num_models: int = 1
    epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 128
    
    # Training options
    use_multiprocessing: bool = False
    verbose: bool = True
    device: str = "cpu"
    
    # Additional training parameters
    eigenvalue: float = 1.0
    balance_loss_lambda: float = 1e-1
    RHS_function: str = "lambda psi: psi-psi**3"
    
    # Gradient descent parameters
    num_steps: int = 100
    threshold: float = 5e-2
    save_trajectories_every: int = 10000


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    
    # Core configurations
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # Experiment metadata
    experiment_name: str = "bistable2d_experiment"
    save_dir: Optional[str] = None
    
    # Plotting parameters
    plot_limits: Dict[str, tuple] = field(default_factory=lambda: {
        "x_limits": (-2, 2),
        "y_limits": (-2, 2)
    })
    show_plots: bool = False
    
    def __post_init__(self):
        """Generate save directory if not provided."""
        if self.save_dir is None:
            dynamics_name = f"{self.dynamics.system_type}_{self.dynamics.bistable_axis}"
            dist_name = f"{self.distribution.dist_type}{self.dynamics.dim}D"
            
            save_dir = Path("results") / f"{dynamics_name}_{dist_name}" / (
                f"{self.training.epochs}_epochs_{self.training.learning_rate}_learning_rate_"
                f"{self.training.batch_size}_batch_size_{self.distribution.gaussian_scale}_gaussian_scale"
            )
            self.save_dir = str(save_dir)


# Predefined configurations
def get_bistable2d_config() -> ExperimentConfig:
    """Get a standard Bistable2D configuration."""
    return ExperimentConfig(
        dynamics=DynamicsConfig(
            system_type="Bistable2D",
            dim=2,
            bistable_axis=0
        ),
        model=ModelConfig(
            model_type="ResNet",
            hidden_size=128,
            num_layers=5
        ),
        distribution=DistributionConfig(
            dist_type="MultivariateGaussian",
            gaussian_scale=2.0,
            use_multiscaler=True,
            scales=[0.1, 0.5, 1.0]
        ),
        training=TrainingConfig(
            num_models=1,
            epochs=1000,
            learning_rate=1e-3,
            batch_size=128,
            use_multiprocessing=False,
            verbose=True,
            device="cpu"
        )
    )


def get_quick_test_config() -> ExperimentConfig:
    """Get a quick test configuration for debugging."""
    return ExperimentConfig(
        dynamics=DynamicsConfig(
            system_type="Bistable2D",
            dim=2,
            bistable_axis=0
        ),
        model=ModelConfig(
            model_type="ResNet",
            hidden_size=64,
            num_layers=3
        ),
        distribution=DistributionConfig(
            dist_type="MultivariateGaussian",
            gaussian_scale=1.0,
            use_multiscaler=False,
            scales=[1.0]
        ),
        training=TrainingConfig(
            num_models=1,
            epochs=10,
            learning_rate=1e-3,
            batch_size=64,
            use_multiprocessing=False,
            verbose=True,
            device="cpu"
        )
    )
