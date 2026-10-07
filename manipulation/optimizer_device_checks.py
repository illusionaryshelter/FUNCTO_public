"""Collect optimizer input faults without reading device values in a graph.

Public calls retain synchronous validation. Only a bounded optimizer block
enters ``DeviceChecks.scope``; its caller must reset the mask before the block
and raise recorded faults immediately after it. Each guarded operation must
also select a safe tensor on failure, so invalid arithmetic cannot reach an
unsafe indexing/native operation while the exception is deferred.
"""

# BEGIN block-local device validation; no process-wide torch/solver patches.
from contextlib import contextmanager
from contextvars import ContextVar
import pickle

import torch
import numpy as np


_active_checks = ContextVar("functo_optimizer_device_checks", default=None)


# BEGIN expose the existing graph/warmup boundary without exposing its mask.
def has_active_device_checks():
    """Whether this context must retain device-only optimizer control flow."""
    return _active_checks.get() is not None
# END eager shortcuts must never specialize a captured graph to warmup data.


class DeviceChecks:
    """One fixed device mask with stable, warmup-registered error categories."""

    def __init__(self, device):
        self._mask = torch.zeros((), dtype=torch.int64, device=device)
        self._errors = {}
        self._constants = {}

    @property
    def storage_bytes(self):
        """Device tensor payload; host key metadata uses native host admission."""
        return (self._mask.numel() * self._mask.element_size()
                + sum(value.numel()*value.element_size() for value in self._constants.values()))

    def clear_constants(self):
        """Release only after all graphs referencing this scope were reset."""
        self._constants.clear()

    def reset(self):
        """Clear before a block, outside its graph; keep the captured address."""
        self._mask.zero_()

    @contextmanager
    def scope(self):
        """Apply only to this execution context, including its backward pass."""
        token = _active_checks.set(self)
        try:
            yield self
        finally:
            _active_checks.reset(token)

    def record(self, condition, message, error_type):
        if condition.device != self._mask.device:
            raise ValueError("Optimizer checks must use the block's device")
        key = (message, error_type)
        if key not in self._errors:
            if len(self._errors) >= 63:
                raise ValueError("Optimizer block exceeded its device fault categories")
            self._errors[key] = 1 << len(self._errors)
        # The integer OR has no floating reduction or rounding. Repeated
        # checks in captured iterations always update the same device scalar.
        self._mask.bitwise_or_((~condition).to(torch.int64) * self._errors[key])
        return condition

    def raise_if_failed(self):
        """Read once after the complete block, before returning any result."""
        faults = int(self._mask.item())
        for (message, error_type), bit in self._errors.items():
            if faults & bit:
                raise error_type(message)


def require_tensor(condition, message, *, error_type=ValueError):
    """Validate eagerly, or return the recorded predicate for safe selection.

    A returned tensor is never a Python branch condition. The guarded caller
    must use it in ``torch.where`` with an explicitly safe value. Outside an
    optimizer scope this function raises immediately or returns ``None``.
    """
    if (not isinstance(condition, torch.Tensor) or condition.dtype != torch.bool
            or condition.ndim != 0):
        raise ValueError("A device check requires a scalar boolean tensor")
    checks = _active_checks.get()
    if checks is None:
        if not bool(condition):
            raise error_type(message)
        return None
    return checks.record(condition, message, error_type)


def _host_constant_key(value, dtype):
    """Describe host bits/types/layout without any numeric promotion."""
    strides = value.strides if isinstance(value, np.ndarray) else None
    return (pickle.dumps(value, protocol=5), strides, dtype)


def constant_tensor(value, like, *, dtype=None):
    """Keep model constants alive for capture using their original conversion.

    The cache belongs to a single optimizer's graph lifetime. Keys describe
    host values only; tensor inputs always keep the original ``as_tensor``
    path, including their gradients. No device values are read to form keys.
    Host serialization describes a key without promoting mixed Python values;
    the actual conversion receives the original value/type. Array strides are
    also part of the key because layout can select a different numeric kernel.
    """
    target_dtype = like.dtype if dtype is None else dtype
    checks = _active_checks.get()
    if checks is None or isinstance(value, torch.Tensor) or like.device.type != "cuda":
        # CPU as_tensor may alias a caller-owned ndarray. Retain that ordinary
        # behavior without placing a mutable alias under a saved content key.
        return torch.as_tensor(value, dtype=target_dtype, device=like.device)
    if like.device != checks._mask.device:
        raise ValueError("Optimizer constants must use the block's device")
    # Only serialize; never deserialize or turn these bytes into tensor data.
    # np.asarray([large_int, float]) would round the integer in the key even
    # though torch.as_tensor(..., dtype=int64) preserves it in the real value.
    key = _host_constant_key(value, target_dtype)
    if key not in checks._constants:
        checks._constants[key] = torch.as_tensor(value, dtype=target_dtype, device=like.device)
    return checks._constants[key]
# END callers explicitly own the graph boundary and invalid-input isolation.
