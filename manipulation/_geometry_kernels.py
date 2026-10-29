"""Compiled float64 three-dimensional bounds for the existing geometry queries.

Boxes and halfspaces share the caller's Cartesian coordinate frame. These
functions implement numerical bounds, not material models or collision rules.
The scene retains ownership of geometry, permissions and query evidence.
"""

# BEGIN common geometry arithmetic uses the existing Numba dependency.
import numpy as np
from numba import njit

_EPS = np.finfo(np.float64).eps
# Forward-error allowance for at most 32 scalar operations per coordinate.
# This coefficient bounds rounding; it is not a physical/model tolerance.
_GAMMA = 32 * _EPS / (1 - 32 * _EPS)


@njit(fastmath=False, cache=True, inline="always")
def _plane_sums(low, high, normal):
    """Keep x/y/z multiplication and addition order for one halfspace.

    Each row has exactly three coordinates. The original NumPy calculation
    reduces this length-three axis; the probe compares its four sums directly
    as well as the final classification. Error sums use the same selected
    products, and no fused multiply-add or reassociation is requested.
    """
    minimum = 0.0
    maximum = 0.0
    low_absolute = 0.0
    high_absolute = 0.0
    for axis in range(3):
        coefficient = normal[axis]
        if coefficient >= 0.0:
            low_term = coefficient * low[axis]
            high_term = coefficient * high[axis]
        else:
            low_term = coefficient * high[axis]
            high_term = coefficient * low[axis]
        minimum = minimum + low_term
        maximum = maximum + high_term
        low_absolute = low_absolute + abs(low_term)
        high_absolute = high_absolute + abs(high_term)
    return minimum, maximum, low_absolute, high_absolute


@njit(fastmath=False, cache=True)
def classify_box(low, high, normals, bounds):
    """Return 1=inside, -1=outside, 0=mixed with original precedence.

    A region is a conjunction of halfspaces. The caller supplies its complete
    finite rows; zero rows represent the whole space, matching NumPy all([]).
    """
    inside = True
    outside = False
    for plane in range(normals.shape[0]):
        minimum, maximum, low_absolute, high_absolute = _plane_sums(
            low, high, normals[plane]
        )
        low_error = _GAMMA * low_absolute
        high_error = _GAMMA * high_absolute
        if not maximum + high_error <= bounds[plane]:
            inside = False
        if minimum - low_error > bounds[plane]:
            outside = True
    if inside:
        return 1
    if outside:
        return -1
    return 0


@njit(fastmath=False, cache=True, inline="always")
def norm3(x, y, z):
    """Use the original length-three sqrt(dot(v,v)) operation order.

    NumPy's one-dimensional norm takes this dot/sqrt path. Numba's generic
    norm instead calls a scaled BLAS nrm2, which is a different computation
    for underflow/overflow and can round differently at ordinary magnitudes.
    Explicit products and left-to-right additions implement that three-term
    formula. Boundary probes and actual-query comparisons establish equality
    for this environment; a different BLAS reduction is a distinct baseline.
    """
    return np.sqrt((x*x + y*y) + z*z)


@njit(fastmath=False, cache=True)
def box_distance(low_a, high_a, low_b, high_b):
    """Lower-bound complete box separation with the original error allowance."""
    gx = max(0., max(low_a[0]-high_b[0], low_b[0]-high_a[0]))
    gy = max(0., max(low_a[1]-high_b[1], low_b[1]-high_a[1]))
    gz = max(0., max(low_a[2]-high_b[2], low_b[2]-high_a[2]))
    value = norm3(gx, gy, gz)
    magnitude = 0.
    # The original concatenate/max visits these same twelve finite endpoints.
    # A maximum introduces no floating-point addition or reassociation.
    for axis in range(3):
        magnitude = max(magnitude, abs(low_a[axis]), abs(high_a[axis]),
                        abs(low_b[axis]), abs(high_b[axis]))
    error = _GAMMA * (value + magnitude)
    return max(0., np.nextafter(value-error, -np.inf))



@njit(fastmath=False, cache=True)
def projection_lower(low, high, direction, native_low, native_high, length):
    """Project a complete box using original xyz sums and outward rounding."""
    minimum, maximum, low_absolute, high_absolute = _plane_sums(low, high, direction)
    box_low = np.nextafter(minimum - _GAMMA*low_absolute, -np.inf)
    box_high = np.nextafter(maximum + _GAMMA*high_absolute, np.inf)
    gap = max(0., box_low-native_high, native_low-box_high)
    error = _GAMMA * (((abs(box_low)+abs(box_high))+abs(native_low))+abs(native_high))
    return max(0., np.nextafter((gap-error)/(length*(1.+_GAMMA)), -np.inf))
# END shared numerical kernels preserve query and model ownership.
