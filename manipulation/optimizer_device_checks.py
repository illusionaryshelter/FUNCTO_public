"""Synchronous tensor validation and declared model-constant conversion.

Planning evaluates host constraint programs and eager robot rollouts. Invalid
inputs therefore raise at the operation that consumes them, before indexing
or polynomial evaluation. Model constants use the consuming tensor's device.
"""

# BEGIN eager validation replaces the retired optimizer graph fault scope.
import torch


def require_tensor(condition, message, *, error_type=ValueError):
    """Require one scalar Boolean predicate and preserve its declared error."""
    if (not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool
            or condition.ndim != 0):
        raise ValueError("A tensor check requires a scalar boolean tensor")
    if not bool(condition):
        raise error_type(message)


def constant_tensor(value, like, *, dtype=None):
    """Convert declared constants with the original dtype, layout and gradients."""
    return torch.as_tensor(value, dtype=like.dtype if dtype is None else dtype,
                           device=like.device)
# END the caller observes a validation error synchronously in its own request.
