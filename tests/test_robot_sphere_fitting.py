"""CPU regressions for full-cell sphere fitting to native convex surfaces."""

# BEGIN geometric counterexamples and scale checks for mesh-constrained covers.
from itertools import product
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial import ConvexHull
import trimesh

sys.path.insert(0, str(Path(__file__).parent / "planning"))
from robot_fixture import _fit_cell_sphere_center
from refine_robot_fixture import _checked_support_excess


@pytest.mark.parametrize("scale", [1e-3, 1., 1e3])
@pytest.mark.parametrize("affine", [np.eye(3),
    np.array([[1., .1, 0.], [0., .7, .05], [0., 0., 1.2]])])
def test_complete_boundary_cell_fits_native_planes_across_scales(scale, affine):
    """A thin boundary cell exposes why minimum radius is the wrong objective.

    Every source point is a convex-cell vertex. Independent hull planes and
    distances verify the fitted ball after the same outward float32 boundary
    used by cuRobo. The affine case also forbids silently normalizing a native
    transform into a rotation and thereby changing its physical solid.
    """
    native = np.asarray(list(product((-1., 1.), repeat=3))) @ affine.T * scale
    cell = np.asarray(list(product((.8, 1.), (-.4, .4), (-.4, .4)))) @ affine.T * scale
    planes = ConvexHull(native).equations
    initial, _ = trimesh.nsphere.minimum_nsphere(cell)
    initial_radius = np.linalg.norm(cell-initial, axis=1).max()
    before = (planes[:, :3] @ initial+planes[:, 3]+initial_radius).max()
    center, evidence = _fit_cell_sphere_center(cell, planes)
    serialized_center = np.asarray(center, np.float32).astype(float)
    maximum = float(np.linalg.norm(cell-serialized_center, axis=1).max())
    radius = float(np.nextafter(np.float32(maximum+1e-6), np.float32(np.inf)))
    after = (planes[:, :3] @ serialized_center+planes[:, 3]+radius).max()
    assert radius-maximum >= 1e-6
    assert after < before-.05*scale
    assert evidence["constraint_violation_m"] < 1e-7*scale
    assert evidence["fitted_support_plane_excess_m"] <= evidence["initial_support_plane_excess_m"]
    # All convex combinations of enclosed vertices are enclosed; checking
    # sparse surface samples would not establish this volume statement.
    if np.array_equal(affine, np.eye(3)):
        analytic_excess = (np.sqrt(1.+.4**2+.4**2)-1.)*scale
        assert np.linalg.norm(center) < 1e-6*scale
        assert abs(evidence["fitted_support_plane_excess_m"]-analytic_excess) < 1e-7*scale


def test_invalid_native_plane_cannot_select_an_unchecked_fit():
    cell = np.asarray(list(product((-1., 1.), repeat=3)))
    with pytest.raises(ValueError, match="nonzero support-plane normals"):
        _fit_cell_sphere_center(cell, np.zeros((1, 4)))


def test_refinement_rejects_serialized_radius_growth_under_native_affine_transform():
    """A translated stretched cube provides an analytic directional oracle."""
    native = np.asarray(list(product((-1., 1.), repeat=3)))
    transform = np.diag([2., 3., 4., 1.])
    transform[:3, 3] = [.1, -.2, .3]
    center = transform[:3, 3].tolist()
    previous = [{"center": center, "radius": 6.}]
    refined = [{"center": center, "radius": 5.5}]
    before, after = _checked_support_excess(previous, refined, ConvexHull(native).equations, transform)
    assert (before, after) == pytest.approx((4., 3.5))
    # A misleading old metadata value cannot excuse worsening actual balls:
    # this API accepts only serialized balls and the original solid planes.
    with pytest.raises(ValueError, match="worsened serialized"):
        _checked_support_excess(refined, previous, ConvexHull(native).equations, transform)
# END no saved task pose, collision exclusion, or scene-specific fit appears here.
