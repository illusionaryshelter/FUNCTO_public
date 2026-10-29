"""Regression boundaries for the shared compiled geometry calculations.

The original NumPy formulas serve as a numerical oracle. Classification tests
exercise adjacent binary64 values at permission boundaries; complete query
semantics are covered by test_implicit_distance and test_native_field_distance.
"""

# BEGIN test the production APIs with independent NumPy reductions.
import numpy as np
import pytest

from manipulation._geometry_kernels import _GAMMA, projection_lower, norm3
from manipulation.implicit_distance import PairPermission, box_distance


def numpy_classify(box, region):
    normals, bounds = region
    low, high = box
    minimum = np.where(normals >= 0, normals * low, normals * high)
    maximum = np.where(normals >= 0, normals * high, normals * low)
    if np.all(maximum.sum(-1) + _GAMMA * np.abs(maximum).sum(-1) <= bounds):
        return "inside"
    if np.any(minimum.sum(-1) - _GAMMA * np.abs(minimum).sum(-1) > bounds):
        return "outside"
    return "mixed"


def layout(values, mode):
    """Create original-shape views whose stride and mutability differ."""
    if mode == "strided":
        storage = np.empty((*values.shape[:-1], values.shape[-1] * 2))
        storage[..., ::2] = values
        return storage[..., ::2]
    values = values.copy()
    if mode == "readonly":
        values.flags.writeable = False
    return values


@pytest.mark.parametrize("mode", ["contiguous", "readonly", "strided"])
def test_permission_inclusive_boundary_and_adjacent_values(mode):
    rng = np.random.default_rng(728315)
    for count in (0, 1, 6, 17):
        for _ in range(24):
            center = rng.normal(size=3)
            radius = rng.random(3)
            box = tuple(layout(x, mode) for x in (center-radius, center+radius))
            normals = layout(rng.normal(size=(count, 3)), mode)
            minimum = np.where(normals >= 0, normals*box[0], normals*box[1])
            maximum = np.where(normals >= 0, normals*box[1], normals*box[0])
            edges = (minimum.sum(-1)-_GAMMA*np.abs(minimum).sum(-1),
                     maximum.sum(-1)+_GAMMA*np.abs(maximum).sum(-1))
            for edge in edges:
                for direction in (-np.inf, None, np.inf):
                    bounds = edge if direction is None else np.nextafter(edge, direction)
                    region = normals, layout(bounds, mode)
                    assert PairPermission._classify(box, region) == numpy_classify(box, region)


def test_range_distance_mixed_array_layouts():
    rng = np.random.default_rng(443)
    for scale in (1e-100, 1e-8, 1., 1e8, 1e100):
        for _ in range(40):
            a, b = np.sort(rng.normal(size=(2, 2, 3))*scale, axis=1)
            a = (layout(a[0], "readonly"), layout(a[1], "strided"))
            b = (layout(b[0], "strided"), b[1])
            gap = np.maximum(0., np.maximum(a[0]-b[1], b[0]-a[1]))
            value = float(np.linalg.norm(gap))
            error = _GAMMA*(value+np.max(np.abs(np.concatenate((*a, *b)))))
            expected = max(0., float(np.nextafter(value-error, -np.inf)))
            assert box_distance(a, b) == expected
            assert box_distance(a, b) == box_distance(b, a)


def test_projection_outward_rounding_including_zero_gap():
    rng = np.random.default_rng(128)
    for _ in range(256):
        low, high = np.sort(rng.normal(size=(2, 3)), axis=0)
        direction = rng.normal(size=3)
        length = float(np.linalg.norm(direction))
        assert norm3(*direction) == length
        minimum = np.where(direction >= 0, low, high)*direction
        maximum = np.where(direction >= 0, high, low)*direction
        box_low = np.nextafter(minimum.sum()-_GAMMA*np.abs(minimum).sum(), -np.inf)
        box_high = np.nextafter(maximum.sum()+_GAMMA*np.abs(maximum).sum(), np.inf)
        for native_low, native_high in ((box_high+1., box_high+2.),
                                       (box_low-2., box_low-1.), (box_low, box_high)):
            gap = max(0., box_low-native_high, native_low-box_high)
            error = _GAMMA*(abs(box_low)+abs(box_high)+abs(native_low)+abs(native_high))
            expected = max(0., float(np.nextafter((gap-error)/(length*(1+_GAMMA)), -np.inf)))
            assert projection_lower(low, high, direction, native_low, native_high, length) == expected
# END changed arithmetic is checked at its actual geometry boundaries.
