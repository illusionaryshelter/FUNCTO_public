"""Production cell selection versus the same original full material sum.

The full-query oracle changes only index selection inside this test process.
Every TSDF/native query, permission and pose derivative still uses production
code. Small deterministic fields test coordinate arithmetic independently of
any task-specific scene, model or execution result.
"""

# BEGIN original point identities and floating-coordinate preimages are explicit.
import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation import sampled_geometry_objective as objective
from manipulation.implicit_distance import ObservedField, PairPermission
from geometry_objective_fixtures import _moving_scene


def field(shape=(4, 4, 4), *, origin=(0., 0., 0.), size=.1, mixed=False):
    grid = np.stack(np.meshgrid(*[np.arange(n) for n in shape], indexing="ij"), -1)
    values = grid[..., 0] + .37 * grid[..., 1] - .21 * grid[..., 2] - 1.1 if mixed else -np.ones(shape)
    return ObservedField(values, np.ones(shape, bool), origin=origin, voxel_size=size)


def pose(position, quaternion=(1., 0., 0., 0.)):
    return torch.tensor(position, dtype=torch.float64), torch.tensor(quaternion, dtype=torch.float64)


def full_indices(source, *_):
    return np.arange(len(source.points), dtype=np.int64)


def value_gradient(function, row):
    value = function(row) + row.sum() * 0
    derivative, = torch.autograd.grad(value, row)
    return value.detach(), derivative


def test_closed_cell_endpoint_retains_both_original_records():
    material = objective.MaterialSamples(field((2, 2, 2), origin=(.1,) * 3))
    query = np.full(3, .1)
    assert material.geometry.tree.query_ball_point(query, .05, p=np.inf, eps=0.) == []
    selected = material.indices_in_box((query, query))
    copies = np.flatnonzero(np.all(material.points == query, axis=1))
    assert len(copies) == 2
    assert np.isin(copies, selected).all()


def test_cell_ranges_preserve_order_duplicates_and_immutable_shared_storage():
    material = objective.MaterialSamples(field((6, 6, 6)))
    full = material.indices_in_box(material.point_bounds)
    np.testing.assert_array_equal(full, np.arange(len(material.points)))
    assert not np.array_equal(material.geometry.tree.indices, np.arange(len(material.geometry.cells)))
    assert len(np.unique(material.points, axis=0)) < len(material.points)
    assert material.cell_ranges.nbytes + material.cell_radius.nbytes == 16 * len(material.geometry.cells) + 24
    assert not material.cell_ranges.flags.writeable and not material.cell_radius.flags.writeable
    small = material.indices_in_box(np.array([[.1, .1, .1], [.2, .2, .2]]))
    assert len(small) < len(full) and np.all(np.diff(small) > 0)
    for cell in np.unique(material.cells[small]):
        start, stop = material.cell_ranges[cell]
        assert np.all(material.cells[start:stop] == cell)
    np.testing.assert_array_equal(material.indices_in_box(material.point_bounds), full)


def test_full_numeric_preimage_retains_rotated_scaled_and_translated_candidates():
    rng = np.random.default_rng(90217)
    materials = {
        scale: (objective.MaterialSamples(field(size=scale, mixed=True)),
                objective.MaterialSamples(field((3, 3, 3), origin=(-scale,) * 3, size=scale, mixed=True)),
                objective.MaterialSamples(coal.Box(scale * 1.2, scale * .8, scale * .6)))
        for scale in (1e-6, 1e-2, 1e2)}
    possible_count = 0
    for trial in range(384):
        scale = (1e-6, 1e-2, 1e2)[trial % 3]
        source, observed, native = materials[scale]
        target = observed if trial % 2 else native
        rotations = Rotation.random(2, random_state=rng)
        quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
        common = rng.uniform(-1, 1, 3) * (0., 1., 1e6, 1e10)[trial % 4]
        source_pose = pose(common, quaternions[0])
        chosen = source.points[rng.integers(len(source.points))]
        world = chosen @ rotations[0].as_matrix().T + common
        corner = np.where(rng.integers(2, size=3), target.negative_bounds[0], target.negative_bounds[1])
        target_pose = pose(world - corner @ rotations[1].as_matrix().T, quaternions[1])
        selected = objective._material_indices(source, source_pose, target, target_pose)
        _, numeric = objective._numeric_points(source.points, source_pose, target_pose)
        possible = np.flatnonzero(np.all((numeric >= target.negative_bounds[0])
                                        & (numeric <= target.negative_bounds[1]), axis=1))
        assert np.isin(possible, selected).all(), (trial, scale, common)
        assert np.all(np.diff(selected) > 0)
        possible_count += len(possible)
    assert possible_count > 4000


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("permission_kind", ["none", "finite", "cells", "both"])
def test_selected_scalar_and_all_pose_derivatives_match_full_sum(monkeypatch, native, permission_kind):
    rng = np.random.default_rng(12)
    first = objective.MaterialSamples(coal.Box(.24, .23, .22) if native else field(mixed=True))
    second = objective.MaterialSamples(field((5, 4, 3), origin=(-.2, -.1, -.1), mixed=True))
    region = (np.array([[1., 0., 0.], [0., -1., 0.]]), np.array([.02, .01]))
    permission = (None if permission_kind == "none" else PairPermission(
        [(region, region)] if permission_kind in ("finite", "both") else [],
        excluded_second_cells=(0, len(second.geometry.cells) - 1)
        if permission_kind in ("cells", "both") else ()))
    nonzero = 0
    for _ in range(8):
        rotations = Rotation.from_rotvec(rng.normal(size=(2, 3)) * .2).as_quat()[:, [3, 0, 1, 2]]
        positions = rng.uniform(-.06, .06, (2, 3))
        row = torch.tensor(np.r_[positions[0], rotations[0], positions[1], rotations[1]],
                           dtype=torch.float64, requires_grad=True)
        def value(x):
            return objective._pair_value(first, (x[:3], x[3:7]), second,
                (x[7:10], x[10:]), permission=permission, report={})
        with monkeypatch.context() as full:
            full.setattr(objective, "_material_indices", full_indices)
            expected = value_gradient(value, row)
        actual = value_gradient(value, row)
        torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
        nonzero += bool(actual[0] != 0)
    if permission_kind == "none":
        assert nonzero > 0


@pytest.mark.parametrize("gap", [-.25, 0., .25])
def test_native_squared_negative_part_queries_once_and_omits_only_zero_graph(monkeypatch, gap):
    first = objective.MaterialSamples(coal.Sphere(.5))
    second = objective.MaterialSamples(coal.Sphere(.5))
    row = torch.tensor([0., 0., 0., 1., 0., 0., 0., 1. + gap, 0., 0., 1., 0., 0., 0.],
                       dtype=torch.float64, requires_grad=True)
    transforms = np.tile(np.eye(4), (2, 1, 1))
    transforms[1, 0, 3] = 1. + gap
    queries, distance = [], coal.distance
    def counted(*args):
        queries.append(1)
        return distance(*args)
    monkeypatch.setattr(coal, "distance", counted)
    if gap >= 0:
        monkeypatch.setattr(objective, "_world", lambda *args: pytest.fail("nonnegative pair read witnesses"))
    value = objective.native_pair_objective(first, (row[:3], row[3:7]), transforms[0],
        second, (row[7:10], row[10:]), transforms[1])
    assert len(queries) == 1
    gradient, = torch.autograd.grad(value + row.sum() * 0, row)
    length = (first.length_scale + second.length_scale) / 2
    assert float(value.detach()) == pytest.approx(max(0., -gap / length) ** 2)
    expected = torch.zeros_like(row)
    if gap < 0:
        expected[0], expected[7] = -2 * gap / length ** 2, 2 * gap / length ** 2
    else:
        assert not value.requires_grad
    torch.testing.assert_close(gradient, expected, rtol=1e-12, atol=1e-12)


def test_release_anchor_remains_the_same_source_of_scene_gradient(monkeypatch, tmp_path):
    from manipulation.planning_scene import Attachment
    scene = _moving_scene(tmp_path)
    scene.owners[0].attachments = (Attachment((0., .5), [0., 0., 0.], [1., 0., 0., 0.],
                                             "declared", "release"),)
    progress = torch.tensor([[.25, .5, .75]], dtype=torch.float64)
    row = torch.tensor([-.015, -.005, .08], dtype=torch.float64, requires_grad=True)
    def value(x):
        position = torch.stack((x, x * 0 + 1, x * 0), -1)[None]
        quaternion = torch.zeros((1, 3, 4), dtype=x.dtype); quaternion[..., 0] = 1
        q = torch.stack((x * 0 + .04, x * 0 - .02), -1)[None]
        names = ("left_slide", "right_slide")
        return scene.optimization_objective(position, quaternion, progress, q, names,
            trajectory_time=progress, robot_link_poses=scene.kinematics.poses(q, names))
    with monkeypatch.context() as full:
        full.setattr(objective, "_material_indices", full_indices)
        expected = value_gradient(value, row)
    actual = value_gradient(value, row)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    assert actual[0] > 0 and actual[1][1] < 0 and actual[1][2] == 0


def test_diagnostic_counts_describe_actual_queries_while_full_denominator_is_preserved(monkeypatch):
    first = objective.MaterialSamples(field((9, 9, 9), origin=(-.4,) * 3))
    second = objective.MaterialSamples(field((2, 2, 2), origin=(-.05,) * 3))
    poses = pose([0., 0., 0.]), pose([0., 0., 0.])
    expected_report, actual_report = {}, {}
    with monkeypatch.context() as full:
        full.setattr(objective, "_material_indices", full_indices)
        expected = objective.pair_objective(first, poses[0], second, poses[1], report=expected_report)
    actual = objective.pair_objective(first, poses[0], second, poses[1], report=actual_report)
    assert actual == expected and actual > 0
    assert actual_report["negative_sample_count"] == expected_report["negative_sample_count"]
    assert actual_report["unsupported_sample_count"] < expected_report["unsupported_sample_count"]
# END the full-query oracle and all task-independent fixtures remain test-only.
