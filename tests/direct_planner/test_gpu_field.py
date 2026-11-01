"""Compare fused rigid queries with the original supported geometry APIs."""

# BEGIN these fixtures exercise geometry semantics independently of a task.
from types import SimpleNamespace

import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from manipulation.direct_planner.batch_distance import NativePointQuery
from manipulation.direct_planner.gpu_field import DeviceField, submit_pair_maxima
from manipulation.implicit_distance import ObservedField


def query_fixture():
    rng = np.random.default_rng(524)
    points = np.vstack((rng.uniform(-.4, .4, (1043, 3)), [[0., 0., 0.], [.5, .5, .5]]))
    source = np.repeat(np.eye(4)[None], 3, axis=0)
    target = source.copy()
    source[1:, :3, :3] = Rotation.from_rotvec([[.2, -.3, .1], [-.1, .1, .3]]).as_matrix()
    target[1:, :3, :3] = Rotation.from_rotvec([[-.2, .1, .2], [.2, .3, -.1]]).as_matrix()
    source[1:, :3, 3] = [[.13, -.06, .02], [-.1, .03, .05]]
    target[1:, :3, 3] = [[-.03, .02, -.04], [.04, .02, .06]]
    # Each pair requires both world-space half-space sets; the two pairs
    # form a union, and the last row deliberately has no permitted region.
    pairs = [((np.array([[1., 0., 0.]]), np.array([-.1])),
              (np.array([[0., 1., 0.]]), np.array([.08]))),
             ((np.array([[-1., 0., 0.]]), np.array([-.16])),
              (np.array([[0., 0., 1.]]), np.array([.03])))]
    return points, source, target, [SimpleNamespace(regions=pairs),
                                   SimpleNamespace(regions=pairs[:1]), None]


def original_queries(points, source, target, permissions):
    local, allowed = [], []
    for first, second, permission in zip(source, target, permissions):
        world = points @ first[:3, :3].T+first[:3, 3]
        local.append((world-second[:3, 3]) @ second[:3, :3])
        row = np.zeros(len(points), bool)
        if permission is not None:
            for side_a, side_b in permission.regions:
                row |= ((world @ side_a[0].T <= side_a[1]).all(1)
                        & (world @ side_b[0].T <= side_b[1]).all(1))
        allowed.append(row)
    return np.asarray(local), np.asarray(allowed)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_fused_field_matches_original_rigid_queries_and_permission_union(device):
    wp.config.kernel_cache_dir = '/tmp/direct006-warp-cache'
    xyz = np.stack(np.meshgrid(*([np.linspace(-.5, .5, 11)]*3), indexing='ij'), axis=-1)
    observed = np.ones(xyz.shape[:-1], bool)
    observed[3:5, 4:6, 5:7] = False
    field = ObservedField((xyz[..., 0]+2*xyz[..., 1]*xyz[..., 2]).astype(np.float32),
        observed, origin=np.full(3, -.5), voxel_size=.1)
    points, source, target, permissions = query_fixture()
    local, allowed = original_queries(points, source, target, permissions)
    scalar, support, _, _ = field.sample(local.reshape(-1, 3))
    scalar, support = scalar.reshape(3, -1), support.reshape(3, -1)
    expected_flags = np.where(allowed, 0, np.where(support, 1, 2))
    expected_values = np.where((expected_flags == 1), -scalar/.1, -np.inf)
    values, winners, counts, unknowns = submit_pair_maxima(DeviceField(field, device),
        torch.as_tensor(points, dtype=torch.float64, device=device), source, target, permissions, .1).cpu().numpy().T
    np.testing.assert_array_equal(counts, (expected_flags != 0).sum(1))
    np.testing.assert_array_equal(unknowns, (expected_flags == 2).sum(1))
    np.testing.assert_allclose(values, expected_values.max(1), atol=2e-14)
    np.testing.assert_array_equal(winners, expected_values.argmax(1))
    assert set(expected_flags.ravel()) == {0, 1, 2}


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_fused_registered_analytic_solids_match_original_values(device):
    points, source, target, permissions = query_fixture()
    local, allowed = original_queries(points, source, target, permissions)
    for shape in (coal.Box(.4, .6, .5), coal.Cylinder(.3, .5)):
        native = NativePointQuery(shape, device=device)
        scalar, _ = native.sample(local.reshape(-1, 3))
        expected = np.where(allowed, -np.inf, np.maximum(-1., -scalar.reshape(3, -1)/.1))
        values, winners, counts, unknowns = submit_pair_maxima(native, torch.as_tensor(points, dtype=torch.float64,
            device=device), source, target, permissions, .1).cpu().numpy().T
        # Both interfaces retain the same -1 tail in the complete row maximum.
        np.testing.assert_allclose(values, expected.max(1), atol=2e-14)
        np.testing.assert_array_equal(counts, (~allowed).sum(1))
        np.testing.assert_array_equal(unknowns, np.zeros(len(source), int))
# END CPU and actual CUDA execute the same numerical and support definitions.
