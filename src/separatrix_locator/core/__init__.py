"""
Core components for separatrix location.

This module contains the main algorithms and models for locating separatrices
in dynamical systems using Koopman eigenfunctions.
"""

from .separatrix_locator import SeparatrixLocator
from .models import ResNet
from .training import train_with_logger, eval_loss, log_metrics, DecayModule
from .gradient_descent import runGD, runGD_basic, process_initial_conditions
from .advanced_models import (
    InputScaler, InputCenterAndScale, ODEBlock, AttentionSelectorDNN, 
    ParallelModels, ExpOutput, LogOutput, AttentionNN, OneHotOutputNN,
    AttentionOneHotNN, create_phi_network, concat_last_dim
)

__all__ = [
    "SeparatrixLocator",
    "ResNet",
    "train_with_logger",
    "eval_loss",
    "log_metrics",
    "DecayModule",
    "runGD",
    "runGD_basic",
    "process_initial_conditions",
    "InputScaler",
    "InputCenterAndScale", 
    "ODEBlock",
    "AttentionSelectorDNN",
    "ParallelModels",
    "ExpOutput",
    "LogOutput",
    "AttentionNN",
    "OneHotOutputNN",
    "AttentionOneHotNN",
    "create_phi_network",
    "concat_last_dim",
]
