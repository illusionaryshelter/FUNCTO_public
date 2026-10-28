"""Immutable completed numerical evidence shared by reports and bindings.

Only completed CPU records enter this module. It does not cache a calculation,
identify similar candidates, or decide acceptance. Mutable solver arrays,
request inputs and bookkeeping keep their own ownership boundaries.
"""

from collections.abc import Mapping
import math
from uuid import uuid4

import numpy as np


def _readonly(*args, **kwargs):
    raise TypeError("Completed evidence is read-only")


class _FrozenDict(dict):
    """JSON-compatible mapping whose public mutation operations are disabled."""

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


class _FrozenList(list):
    """Preserve ordinary JSON/list equality while owning immutable elements."""

    __setitem__ = __delitem__ = append = clear = extend = insert = pop = remove = reverse = sort = _readonly
    __iadd__ = __imul__ = _readonly

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self


def _freeze(value):
    """Detach numeric arrays and mutable containers once, preserving values."""
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, Mapping):
        entries = [(key, _freeze(item)) for key, item in value.items()]
        return _FrozenDict((key, item) for key, (item, _) in entries), all(
            type(key) is str and finite for key, (_, finite) in entries)
    if isinstance(value, (list, tuple)):
        entries = [_freeze(item) for item in value]
        return _FrozenList(item for item, _ in entries), all(finite for _, finite in entries)
    if type(value) is float:
        return value, math.isfinite(value)
    if type(value) in (str, bool, int) or value is None:
        return value, True
    raise TypeError(f"Completed evidence contains a non-CPU record value: {type(value).__name__}")


class CompletedEvidence(_FrozenDict):
    """One completed record, with a file identity independent of its contents.

    Sharing is safe because the entire public value tree is immutable. The
    identifier names this record, not a numerical equivalence class. Nonfinite
    diagnostics retain their values; formal conversion uses the existing
    execution-binding rules and is performed at most once for this record.
    """

    def __init__(self, value, *, record_id=None):
        frozen, self._finite = _freeze(value)
        dict.__init__(self, frozen)
        self.record_id = str(uuid4()) if record_id is None else record_id
        self._formal = None

    def formal(self):
        """Return the original finite record or its owned formal conversion."""
        if self._finite:
            return self
        if self._formal is None:
            from .execution_binding import plain
            self._formal = CompletedEvidence(plain(dict(self)))
        return self._formal


def completed_evidence(value):
    """Freeze one completed record without copying an already immutable one."""
    return value if isinstance(value, CompletedEvidence) else CompletedEvidence(value)
