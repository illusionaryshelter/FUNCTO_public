"""Shared monotone clock for reference, task and scene event queries."""
# BEGIN all consumers derive event identities from the same binary64 record.
from dataclasses import dataclass, field
import numpy as np

def progress_anchor_parameters(u, s, anchors):
    """Return canonical binary64 (time parameter, progress) anchor records.

    A plateau contributes its entrance and exit. Native tensor consumers
    additionally verify float32 representability in _ProgressMap; continuous
    CPU geometry retains the original binary64 owner-path clock partitions.
    """
    pairs = []
    for value in anchors:
        equal = np.flatnonzero(s == value)
        if len(equal):
            pairs.append((float(u[equal[0]]), float(value)))
            if equal[-1] != equal[0]:
                pairs.append((float(u[equal[-1]]), float(value)))
        else:
            right = int(np.searchsorted(s, value, side="right"))
            if right == 0 or right == len(s):
                raise ValueError("An event anchor is outside the planned progress interval")
            fraction = (value - s[right-1]) / (s[right] - s[right-1])
            pairs.append((float(u[right-1] + fraction * (u[right] - u[right-1])), float(value)))
    pairs = np.asarray(sorted(set(pairs)), dtype=float).reshape(-1, 2)
    return pairs


def progress_query_parameters(u, s, times, duration, anchors):
    """Sample one physical clock and restore its canonical event coordinates."""
    times = np.asarray(times, float)
    parameters = times / duration
    progress = np.interp(parameters, u, s)
    for parameter, value in progress_anchor_parameters(u, s, anchors):
        matching = times == parameter * duration
        parameters[matching] = parameter
        progress[matching] = value
    return parameters, progress


@dataclass(frozen=True)
class _ProgressMap:
    """A fixed, monotone phase schedule; repeated s values encode pauses."""

    u: np.ndarray
    s: np.ndarray
    # BEGIN original intent and device activation are distinct bound facts.
    event_clock: dict = field(default_factory=dict, compare=False)
    # END only initial binding creates this record.

    def __post_init__(self):
        if (self.u.ndim != 1 or self.s.shape != self.u.shape or len(self.u) < 2
                or not np.isfinite(self.u).all() or not np.isfinite(self.s).all()
                or self.u[0] != 0 or self.u[-1] != 1 or self.s[0] < 0
                or np.any(np.diff(self.u) <= 0) or np.any(np.diff(self.s) < 0)
                or np.any(np.diff(self.u.astype(np.float32)) <= 0)):
            raise ValueError("A progress map needs increasing u and nondecreasing s")

    def anchor_parameters(self, anchors):
        """Preserve both ends of an s plateau, including a paused event."""
        pairs = progress_anchor_parameters(self.u, self.s, anchors)
        # Required source knots and event boundaries must remain distinct in
        # the native float32 process tensors.  Reject an unrepresentable clock
        # instead of silently merging or moving a supplied hard anchor.
        if any(len(np.unique(pairs[:, column])) != len(np.unique(pairs[:, column].astype(np.float32)))
               for column in (0, 1)):
            raise ValueError("Required progress anchors exceed native float32 clock precision")
        return pairs

    # BEGIN one event clock supplies exact time, progress and relation identity.
    def query_parameters(self, times, duration, anchors):
        """Restore declared event coordinates at their identical physical keys.

        Multiplication by duration creates the authoritative physical time.
        Division need not recover the same binary64 parameter. Event progress
        and normalized time therefore come directly from the same anchor
        record; ordinary times retain piecewise-linear interpolation.
        """
        return progress_query_parameters(self.u, self.s, times, duration, anchors)
    # END no epsilon or rounded cache key changes event membership.

# END monotone progress may include finite physical-time plateaus.
