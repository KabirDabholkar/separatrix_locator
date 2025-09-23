from functools import partial
import torch

# Default hyperparameters for experiments. Individual config files can import
# these and override as needed. CLI flags should take final precedence.

# Training loop hyperparameters
epochs = 500
batch_size = 128

# Optimizer factory (callable returning an optimizer when given params)
learning_rate = 1e-4
optimizer = partial(torch.optim.AdamW, lr=learning_rate)

# Training loss specifics
RHS_function = "lambda psi: psi-psi**3"
balance_loss_lambda = 1e-2


