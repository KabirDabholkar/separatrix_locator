from torchdiffeq import odeint
import torch


def add_t_arg_to_dynamical_function(f, *args, **kwargs):
    def new_f(t, x):
        return f(x, *args, **kwargs)
    return new_f


def run_odeint_to_final(func, y0, T, inputs=None, steps=2, return_last_only=True, no_grad=True):
    times = torch.linspace(0, T, steps).type_as(y0)

    args = []
    if inputs is not None:
        if len(y0.shape) > 1:
            for _ in range(len(y0.shape) - 1):
                inputs = inputs[None]
            inputs = inputs.repeat(*y0.shape[:-1], 1)
        args += [inputs]

    if no_grad:
        with torch.no_grad():
            traj = odeint(
                add_t_arg_to_dynamical_function(func, *args),
                y0,
                times,
            )
    else:
        traj = odeint(
            add_t_arg_to_dynamical_function(func, *args),
            y0,
            times,
        )

    if return_last_only:
        return traj[-1]
    return traj


