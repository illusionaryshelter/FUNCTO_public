"""Independent CPU checks of conservative observed-volume sphere clearance.

The oracle measures Euclidean distances directly to unknown voxel boxes and
the complement of the finite voxel domain. It does not interpolate an EDT or
reuse the production clearance formula, so agreement checks geometric safety.
"""
# BEGIN geometric bounds and differentiability regressions for observed volume.
from itertools import product

import numpy as np
import pytest
import torch

from manipulation.planning_scene import ObservedSpace


def _coverage(known, origin, step):
    return ObservedSpace(known, origin=origin, voxel_size=step,
                         observation_ids=('measured-grid',), reference_frame='world')


def _clearance(coverage, positions, dtype):
    # One metre exceeds every clearance in these small fixtures. Subtracting
    # the residual therefore recovers the production clearance without reading
    # its private numerical grid or assuming a particular implementation.
    centres = torch.as_tensor(positions, dtype=dtype)
    spheres = torch.cat((centres, torch.ones((*centres.shape[:-1], 1), dtype=dtype)), -1)
    return (spheres[..., 3] - coverage.violation(spheres)).detach().numpy()


def _exact_unknown_box_clearance(known, origin, step, positions):
    """Distance to actual unknown cell volumes, including all exterior space."""
    points = np.asarray(positions, dtype=np.float64)
    origin = np.asarray(origin, dtype=np.float64)
    lower = origin - step / 2
    upper = origin + (np.asarray(known.shape[::-1]) - .5) * step
    # The complement of the finite cell domain is unknown. Its distance is
    # the nearest exterior face for interior points and zero outside the box.
    result = np.minimum(points - lower, upper - points).min(axis=-1).clip(min=0)
    unknown_zyx = np.argwhere(~known)
    if unknown_zyx.size:
        centres = origin + unknown_zyx[:, ::-1] * step
        offsets = np.maximum(np.abs(points[:, None, :] - centres[None, :, :]) - step / 2, 0)
        distance = np.linalg.norm(offsets, axis=-1).min(axis=-1)
        result = np.minimum(result, distance)
    return result


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('seed', [0, 17, 42, 101])
def test_random_grid_clearance_never_exceeds_distance_to_unknown_boxes(dtype, seed):
    rng = np.random.default_rng(seed)
    known = rng.random((5, 6, 7)) > .2
    origin, step = np.array([-.21, .17, .04]), .02
    extent = (np.asarray(known.shape[::-1]) - 1) * step
    positions = origin + rng.uniform(-.5, 1.5, (96, 3)) * extent
    # Include every cell centre as well as arbitrary interpolation positions.
    centres_zyx = np.indices(known.shape).reshape(3, -1).T
    positions = np.concatenate((positions, origin + centres_zyx[:, ::-1] * step))
    measured = _clearance(_coverage(known, origin, step), positions, dtype)
    actual = _exact_unknown_box_clearance(known, origin, step, positions)
    tolerance = 2e-7 if dtype == torch.float32 else 2e-9
    assert np.isfinite(measured).all()
    assert (measured >= -tolerance).all()
    assert (measured <= actual + tolerance).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_known_grid_still_has_unknown_exterior_and_positive_interior(dtype):
    known = np.ones((9, 9, 9), dtype=bool)
    origin, step = np.array([-.08] * 3), .02
    positions = np.array([[0., 0., 0.], [-.08, 0., 0.], [-.081, 0., 0.],
                          [.1, 0., 0.], [0., .081, 0.]])
    measured = _clearance(_coverage(known, origin, step), positions, dtype)
    actual = _exact_unknown_box_clearance(known, origin, step, positions)
    assert measured[0] > .05
    assert np.allclose(measured[1:], 0, atol=2e-7, rtol=0)
    assert (measured <= actual + 2e-7).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_all_unknown_grid_has_no_positive_clearance(dtype):
    known = np.zeros((5, 5, 5), dtype=bool)
    positions = np.array([[0., 0., 0.], [.011, -.003, .019], [.2, .1, 0.]])
    measured = _clearance(_coverage(known, [-.04] * 3, .02), positions, dtype)
    assert np.array_equal(measured, np.zeros_like(measured))


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_single_unknown_box_preserves_a_provably_clear_nearby_sphere(dtype):
    known = np.ones((9, 9, 9), dtype=bool)
    known[4, 4, 4] = False
    coverage = _coverage(known, [-.08] * 3, .02)
    positions = np.array([[0., 0., 0.], [.03, 0., 0.], [.041, .013, -.007]])
    measured = _clearance(coverage, positions, dtype)
    actual = _exact_unknown_box_clearance(known, [-.08] * 3, .02, positions)
    assert (measured <= actual + 2e-7).all()
    assert measured[0] == 0
    # The unknown box ends at x=.01, and this sphere starts at x=.02.
    # Certifying this gap catches the former excessive interpolation deduction.
    sphere = torch.tensor([[.03, 0., 0., .01]], dtype=dtype)
    assert coverage.violation(sphere).item() == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_diagonal_clearance_stays_below_exact_unknown_box_corner_distance(dtype):
    # Keep padding far enough away that the central unknown voxel, rather
    # than the exterior, is nearest at every surrounding interpolation node.
    known = np.ones((17, 17, 17), dtype=bool)
    known[8, 8, 8] = False
    coverage = _coverage(known, [-.16] * 3, .02)
    # Diagonal approach directions touch an unknown box corner. The voxel
    # enclosing sphere is tight here, making these checks more demanding than
    # ordinary random points with slack between the two geometric surfaces.
    positions = np.array([np.asarray(signs) * offset
        for signs in product((-1., 1.), repeat=3) for offset in (.025, .03, .035)])
    measured = _clearance(coverage, positions, dtype)
    actual = _exact_unknown_box_clearance(known, [-.16] * 3, .02, positions)
    assert (measured <= actual + 2e-7).all()
    assert (actual - measured < 1e-5).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_large_coordinate_origin_does_not_overstate_unknown_clearance(dtype):
    known = np.ones((9, 9, 9), dtype=bool)
    known[4, 4, 4] = False
    origin, step = np.array([1000., -800., 1200.]), .02
    positions = origin + .08 + np.array([[.03, .03, .03], [-.025, .025, .025],
                                        [0., 0., 0.], [.1, 0., 0.]])
    # Compare the oracle at exactly the coordinates representable by the
    # requested dtype, separating input rounding from the clearance guarantee.
    positions = torch.as_tensor(positions, dtype=dtype).double().numpy()
    measured = _clearance(_coverage(known, origin, step), positions, dtype)
    actual = _exact_unknown_box_clearance(known, origin, step, positions)
    assert np.isfinite(measured).all()
    assert (measured >= 0).all()
    assert (measured <= actual + 2e-7).all()


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_map_boundary_and_interior_grid_face_have_continuous_clearance(dtype):
    known = np.ones((9, 9, 9), dtype=bool)
    known[4, 4, 4] = False
    coverage = _coverage(known, [-.08] * 3, .02)
    epsilon = 1e-5 if dtype == torch.float32 else 1e-8
    tolerance = 2e-7 if dtype == torch.float32 else 2e-9
    # Test both the finite map boundary and a cell face with positive clearance.
    for face in (-.08, .04):
        positions = np.array([[face - epsilon, .011, .009],
                              [face, .011, .009], [face + epsilon, .011, .009]])
        measured = _clearance(coverage, positions, dtype)
        assert np.ptp(measured) <= 4 * epsilon + tolerance
        if face == -.08:
            assert measured[0] == measured[1] == 0


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_gradients_stay_finite_at_unknown_centres_zero_radius_and_grid_faces(dtype):
    known = np.ones((9, 9, 9), dtype=bool)
    known[4, 4, 4] = False
    coverage = _coverage(known, [-.08] * 3, .02)
    spheres = torch.tensor([[0., 0., 0., .05], [.001, .002, -.001, .05],
        [.03, 0., 0., .05], [-.08, 0., 0., .05], [-.081, 0., 0., .05],
        [.04, 0., 0., .05], [0., 0., 0., 0.]], dtype=dtype, requires_grad=True)
    residual = coverage.violation(spheres)
    assert torch.isfinite(residual).all()
    assert residual[-1] == 0
    residual.sum().backward()
    assert torch.isfinite(spheres.grad).all()
    assert spheres.grad[2, :3].abs().max() > 0
# END the oracle uses unknown boxes rather than implementation-derived distances.
