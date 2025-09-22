"""
Advanced neural network architectures for Koopman eigenfunction learning.

This module contains sophisticated model architectures including RBF networks,
attention mechanisms, and specialized neural ODE blocks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, List, Union, Tuple
from torchdiffeq import odeint


class InputScaler(nn.Module):
    """A simple PyTorch module for scaling input data by a constant factor."""
    
    def __init__(self, scale_factor=1.0, trainable=False):
        """
        Args:
            scale_factor (float): Constant to multiply the input by
            trainable (bool): If True, scale_factor will be learned during training
        """
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(scale_factor), requires_grad=trainable)
            
    def forward(self, x):
        """Scale the input by multiplying with scale_factor."""
        return x * self.scale
            
    def inverse_transform(self, x):
        """Reverse the scaling transformation."""
        return x / self.scale


class InputCenterAndScale(nn.Module):
    """Center inputs by an optional vector and scale by a factor."""
    
    def __init__(self, center=None, scale_factor=1.0, trainable_scale=False):
        """
        Args:
            center (Tensor|list|float|None): Value to subtract from inputs. If None, uses zeros like x.
            scale_factor (float): Multiply inputs after centering by this factor.
            trainable_scale (bool): If True, scale is learned.
        """
        super().__init__()
        self.register_buffer("center_buffer", None, persistent=False)
        if center is not None:
            center_tensor = torch.as_tensor(center, dtype=torch.get_default_dtype())
            self.register_buffer("center_buffer", center_tensor, persistent=False)
        self.scale = nn.Parameter(torch.tensor(scale_factor), requires_grad=trainable_scale)

    def _get_center(self, x):
        if self.center_buffer is None:
            return torch.zeros_like(x)

        center = self.center_buffer

        # Ensure center is 1D over features (last dim)
        if center.dim() == 0:
            center = center.view(1)
        elif center.dim() > 1:
            center = center.reshape(-1)

        desired_features = x.size(-1)
        current_features = center.size(0)

        # Pad with zeros at the end if shorter; slice if longer
        if current_features < desired_features:
            pad_amount = desired_features - current_features
            center = nn.functional.pad(center, (0, pad_amount))
        elif current_features > desired_features:
            center = center[:desired_features]

        # Expand to match x shape
        expand_shape = [1] * (x.dim() - 1) + [desired_features]
        center = center.to(x.device).view(*expand_shape)
        return center.expand_as(x)

    def forward(self, x):
        center = self._get_center(x)
        return (x - center) * self.scale

    def inverse_transform(self, x):
        center = self._get_center(x)
        return x / self.scale + center


class ScaledLinear(nn.Linear):
    """Linear layer with scaled weights."""
    
    def __init__(self, in_features, out_features, bias=True, scale=0.1):
        super().__init__(in_features, out_features, bias)
        self.scale = scale
        self._scale_weights()

    def _scale_weights(self):
        self.weight.data *= self.scale

    def forward(self, x):
        return nn.functional.linear(x, self.weight * self.scale, self.bias)


class ODEBlock(nn.Module):
    """Neural ODE block for continuous-time dynamics."""
    
    def __init__(self, odefunc, odefunc_dim, input_dim=1, output_dim=1):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()
        self.output_dim = output_dim
        self.input_layer = nn.Linear(input_dim, odefunc_dim)
        self.readout = nn.Linear(odefunc_dim, output_dim)
        self.atol = None
        self.rtol = None

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        x = self.input_layer(x)
        out = odeint(lambda t, y: self.odefunc(y), x, self.integration_time)
        return self.readout(out[1])

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class AttentionSelectorDNN(nn.Module):
    """Deep neural network with attention mechanism for output selection."""
    
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()

        # Fully connected layers
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        # Last hidden layer produces `output_dim` features
        self.feature_extractor = nn.Sequential(*layers)
        self.last_layer = nn.Linear(prev_dim, output_dim)

        # Attention mechanism (produces weights for output selection)
        self.attention = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # Extract deep features
        features = self.feature_extractor(x)
        outputs = self.last_layer(features)

        # Compute attention weights
        attention_scores = self.attention(outputs)
        attention_weights = F.softmax(attention_scores, dim=1)

        # Weighted sum (soft attention)
        selected_output = (attention_weights * outputs).sum(dim=1, keepdim=True)

        return selected_output, attention_weights


class ParallelModels(nn.Module):
    """Parallel ensemble of models with output selection."""
    
    def __init__(self, base_model, num_models, prod_output=True, select_max=False):
        """
        Args:
            base_model: An instance of torch.nn.Module representing the base architecture.
            num_models: Number of independent models to train in parallel.
            prod_output: If True, use product-based output selection.
            select_max: If True, select maximum output.
        """
        super().__init__()
        self.num_models = num_models
        self.models = nn.ModuleList([base_model() for _ in range(num_models)])
        self.prod_output = prod_output
        self.select_max = select_max

    def forward(self, x):
        """Forward pass through all models and combine outputs."""
        outputs = [model(x) for i, model in enumerate(self.models)]
        if self.prod_output:
            stack_outputs = torch.stack(outputs, dim=-1)
            mean_outputs = torch.mean(torch.abs(stack_outputs), dim=-2)
            ids = torch.argmax(mean_outputs, dim=-1)
            select_outputs = torch.zeros_like(mean_outputs)
            for i in range(select_outputs.shape[0]):
                select_outputs[i] = stack_outputs[i, ..., ids[i]]
            outputs = select_outputs
        else:
            outputs = torch.concatenate(outputs, dim=-1)
        return outputs


class ExpOutput(nn.Module):
    """Wrapper that applies exponential transformation to base model output."""
    
    def __init__(self, base_model):
        super(ExpOutput, self).__init__()
        self.base_model = base_model

    def forward(self, x):
        output = self.base_model(x)
        output = torch.exp(torch.abs(output)) - 1
        return output


class LogOutput(nn.Module):
    """Wrapper that applies logarithmic transformation to base model output."""
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        output = self.base_model(x)
        output = -torch.log(torch.abs(output))
        return output


class AttentionNN(nn.Module):
    """Neural network with attention mechanism."""
    
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of attention weights).
        """
        super(AttentionNN, self).__init__()
        self.temperature = temperature
        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)  # Feature extraction
        self.fc2 = nn.Linear(128, 64)
        self.query = nn.Linear(64, 64)  # Query vector
        self.key = nn.Linear(64, 64)  # Key vector
        self.value = nn.Linear(64, output_dim)  # Value vector (output space)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Attention mechanism
        query = self.query(x)  # Compute queries
        key = self.key(x)  # Compute keys
        value = self.value(x)  # Compute values

        # Attention scores: scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)

        # Compute the context vector as the weighted sum of values
        context = torch.matmul(attention_weights, value)

        return context


class OneHotOutputNN(nn.Module):
    """Neural network with one-hot-like output using softmax."""
    
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of output).
        """
        super(OneHotOutputNN, self).__init__()
        self.temperature = temperature
        # Define the network layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass of the network.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Hidden layers with non-linear activation
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # Output layer (scores)
        scores = self.fc3(x)
        # Apply softmax with temperature
        probabilities = F.softmax(scores / self.temperature, dim=1)
        # Compute one-hot-like output
        output = probabilities * scores  # Smooth approximation of argmax
        return output


class AttentionOneHotNN(nn.Module):
    """Neural network with attention and one-hot-like output."""
    
    def __init__(self, input_dim, output_dim, temperature=1.0):
        """
        Args:
            input_dim (int): Dimensionality of the input space (R^N).
            output_dim (int): Dimensionality of the output space (R^M).
            temperature (float): Temperature for softmax (controls smoothness of attention weights).
        """
        super(AttentionOneHotNN, self).__init__()
        self.temperature = temperature
        # Define layers
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.query = nn.Linear(64, 64)
        self.key = nn.Linear(64, 64)
        self.value = nn.Linear(64, output_dim)

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        # Feature extraction
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        # Attention mechanism
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)

        # Attention scores: scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        attention_weights = F.softmax(attention_scores / self.temperature, dim=-1)

        # Compute the context vector as the weighted sum of values
        context = torch.matmul(attention_weights, value)

        # Output layer: apply softmax to context vector to get a one-hot-like output
        output = F.softmax(context, dim=-1) * value
        return output


def concat_last_dim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Concatenate two tensors along the last dimension.
    Shapes must be broadcastable except for last dim.
    """
    return torch.cat((a, b), dim=-1)


def create_phi_network(input_dim=1, hidden_dim=200, output_dim=1, num_layers=4, nonlin=nn.Tanh):
    """Create a standard neural network for phi (eigenfunction) learning."""
    args = []
    args.append(nn.Linear(input_dim, hidden_dim))
    args.append(nonlin())

    for i in range(num_layers - 2):
        args.append(nn.Linear(hidden_dim, hidden_dim))
        args.append(nonlin())

    args.append(nn.Linear(hidden_dim, output_dim))
    model = nn.Sequential(*args)
    return model
