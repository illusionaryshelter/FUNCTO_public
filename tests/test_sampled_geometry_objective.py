"""Actual finite geometry values, analytic derivatives and original verdicts."""

import coal
import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField, PairPermission, query_distance
from manipulation.sampled_geometry_objective import MaterialSamples, pair_objective, _pair_value
from geometry_objective_fixtures import _moving_scene, _quaternion


def pose(data):
    return data[:3], _quaternion(data[3:6])


def cube(*, value=-.1, origin=(-.13, -.11, -.09), size=.2, observed=True):
    return ObservedField(np.full((2, 2, 2), value), np.full((2, 2, 2), observed),
                         origin=origin, voxel_size=size)


def plane():
    coordinates = np.stack(np.meshgrid(*[np.linspace(-1, 1, 5)] * 3, indexing="ij"), -1)
    values = coordinates[..., 0] + .3 * coordinates[..., 1] + .2 * coordinates[..., 2]
    return ObservedField(values, np.ones_like(values, bool), origin=[-1] * 3, voxel_size=.5)


def test_both_moving_frames_match_the_same_trial_scalar_gradient():
    first, second = MaterialSamples(cube()), MaterialSamples(plane())
    data = torch.tensor([-.24, .03, .07, .1, -.05, .08, .02, -.04, .01, -.08, .02, .04],
                        dtype=torch.float64, requires_grad=True)

    def value(row):
        return pair_objective(first, pose(row[:6]), second, pose(row[6:]))

    assert value(data) > 0
    assert torch.autograd.gradcheck(value, data, eps=1e-6, atol=1e-8, rtol=1e-5)
    # Revisiting a trial after an unrelated trial must return exactly the
    # same value; sample coordinates cannot be refreshed by the query.
    expected = value(data).detach()
    value(data + .1)
    torch.testing.assert_close(value(data).detach(), expected, rtol=0, atol=0)


def test_clear_and_penetrating_trials_use_the_original_interpolant():
    first, second = MaterialSamples(cube()), MaterialSamples(plane())
    origin = torch.zeros(6, dtype=torch.float64)
    separated = origin.clone(); separated[0] = .45
    penetrating = origin.clone(); penetrating[0] = -.45
    assert pair_objective(first, pose(separated), second, pose(origin)) == 0
    assert pair_objective(first, pose(penetrating), second, pose(origin)) > 0


def test_finite_regions_remove_only_their_actual_overlap_points():
    first, second = MaterialSamples(cube()), MaterialSamples(plane())
    state = torch.zeros(6, dtype=torch.float64)
    state[0] = -.4
    fixed = torch.zeros_like(state)
    region = np.vstack((np.eye(3), -np.eye(3))), np.full(6, 2.)
    full = PairPermission([(region, region)])
    assert pair_objective(first, pose(state), second, pose(fixed), permission=full) == 0
    distant = np.array([[1., 0., 0.]]), np.array([-3.])
    partial = PairPermission([(distant, distant)])
    torch.testing.assert_close(pair_objective(first, pose(state), second, pose(fixed), permission=partial),
                               pair_objective(first, pose(state), second, pose(fixed)))


def test_original_cell_exclusion_keeps_adjacent_cell_and_support_boundary():
    field = ObservedField(-np.ones((3, 2, 2)), np.ones((3, 2, 2), bool),
                          origin=[0, 0, 0], voxel_size=1.)
    values, valid, _, _ = field.sample(np.array([[.5, .5, .5], [1., .5, .5], [1.5, .5, .5]]),
                                      excluded_cells=(0,))
    assert valid.tolist() == [False, True, True]
    np.testing.assert_equal(values[1:], [-1., -1.])
    # On the internal plane the remaining adjacent cell owns the same value.
    _, valid, _, _ = field.sample(np.array([[1., .5, .5]]), excluded_cells=(1,))
    assert valid.item()


def test_source_and_target_cell_permissions_apply_in_both_directions():
    first = MaterialSamples(cube(origin=(-.5, -.5, -.5), size=1.))
    second = MaterialSamples(cube(origin=(-.25, -.25, -.25), size=.5))
    fixed = pose(torch.zeros(6, dtype=torch.float64))
    assert pair_objective(first, fixed, second, fixed) > 0
    assert pair_objective(first, fixed, second, fixed,
                          permission=PairPermission([], excluded_second_cells=(0,))) == 0


def test_unknown_samples_remain_unknown_and_do_not_acquire_a_field_direction():
    first, second = MaterialSamples(cube()), MaterialSamples(cube(observed=False))
    state = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    report = {}
    value = pair_objective(first, pose(state), second, pose(state * 0), report=report)
    assert value == 0 and report["zero_contribution_separated_pair_count"] == 1
    assert report["supported_sample_count"] == report["unsupported_sample_count"] == 0
    assert torch.count_nonzero(torch.autograd.grad(value, state)[0]) == 0
    values, supported, _, _ = second.geometry.sample(np.array([[0., 0., 0.]]))
    assert not supported.item() and np.isnan(values).all()


# BEGIN interval separation preserves both finite sums and every pose derivative.
@pytest.mark.parametrize('native', [False, True])
@pytest.mark.parametrize('separation', [0., .2, 3.])
@pytest.mark.parametrize('permitted', [False, True])
def test_separation_matches_unpruned_value_and_fourteen_pose_derivatives(native, separation, permitted):
    first = MaterialSamples(coal.Box(.3, .25, .2) if native else cube())
    second = MaterialSamples(cube(origin=(-.16, -.14, -.12), size=.3))
    quaternions = [_quaternion(torch.tensor(rotation, dtype=torch.float64)).numpy()
                   for rotation in ([.2, -.1, .3], [-.12, .25, -.08])]
    row = torch.tensor(np.r_[[separation, .015, -.02], quaternions[0],
                            [.03, -.025, .04], quaternions[1]],
                       dtype=torch.float64, requires_grad=True)
    region = (np.vstack((np.eye(3), -np.eye(3))), np.full(6, .1))
    permission = PairPermission([(region, region)]) if permitted else None
    poses = (row[:3], row[3:7]), (row[7:10], row[10:14])
    full = _pair_value(first, poses[0], second, poses[1], permission=permission, report={})
    full_gradient, = torch.autograd.grad(full, row)
    report = {}
    pruned = pair_objective(first, poses[0], second, poses[1], permission=permission, report=report)
    pruned_gradient, = torch.autograd.grad(pruned, row)
    torch.testing.assert_close(pruned, full, atol=0., rtol=0.)
    torch.testing.assert_close(pruned_gradient, full_gradient, atol=1e-12, rtol=1e-12)
    if separation == 3.:
        assert report['zero_contribution_separated_pair_count'] == 1
        assert report['supported_sample_count'] == report['unsupported_sample_count'] == 0
        assert torch.count_nonzero(pruned_gradient) == 0


def test_touching_enclosures_retain_the_full_query_and_original_surface_semantics():
    first = MaterialSamples(coal.Box(2., 2., 2.))
    second = MaterialSamples(cube(origin=(1., -.1, -.1), size=.2))
    state = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    report = {}
    pair_objective(first, pose(state), second, pose(state * 0), report=report)
    assert report['zero_contribution_separated_pair_count'] == 0
    assert report['supported_sample_count'] + report['unsupported_sample_count'] == len(first.points)
# END the optimization enclosure never supplies a collision verdict.


def test_constant_zero_overlap_keeps_original_collision_without_inventing_a_direction():
    geometry = cube(value=0.)
    first, second = MaterialSamples(geometry), MaterialSamples(geometry)
    state = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert query_distance(geometry, np.eye(4), geometry, np.eye(4))["status"] == "intersection"
    value = pair_objective(first, pose(state), second, pose(state * 0))
    assert value == 0
    assert torch.count_nonzero(torch.autograd.grad(value, state)[0]) == 0


def test_observed_material_inside_native_shape_has_a_signed_distance_direction():
    native = MaterialSamples(coal.Box(2., 2., 2.))
    observed = MaterialSamples(cube(origin=(-.12, -.1, -.08), size=.05))
    state = torch.tensor([.05, -.1, .4, .1, -.05, .04], dtype=torch.float64, requires_grad=True)
    fixed = torch.zeros_like(state)

    def value(row):
        return pair_objective(native, pose(fixed), observed, pose(row))

    report = {}
    assert pair_objective(native, pose(fixed), observed, pose(state), report=report) > 0
    assert report["negative_native_sample_count"] > 0
    assert torch.autograd.gradcheck(value, state, eps=1e-6, atol=1e-6, rtol=1e-5)


def test_scene_objective_release_uses_the_shared_release_state(tmp_path):
    from manipulation.planning_scene import Attachment
    scene = _moving_scene(tmp_path)
    scene.owners[0].attachments = (Attachment((0., .5), [0., 0., 0.], [1., 0., 0., 0.],
                                               "declared", "release"),)
    progress = torch.tensor([[.25, .5, .75]], dtype=torch.float64)
    values = torch.tensor([-.015, -.005, .08], dtype=torch.float64, requires_grad=True)

    def objective(x, report=None):
        position = torch.stack((x, x * 0 + 1, x * 0), -1)[None]
        quaternion = torch.zeros((1, 3, 4), dtype=x.dtype); quaternion[..., 0] = 1
        q = torch.stack((x * 0 + .04, x * 0 - .02), -1)[None]
        links = scene.kinematics.poses(q, ("left_slide", "right_slide"))
        return scene.optimization_objective(position, quaternion, progress, q,
            ("left_slide", "right_slide"), trajectory_time=progress,
            robot_link_poses=links, report_out=report)

    report = {}
    result = objective(values, report)
    derivative = torch.autograd.grad(result, values)[0]
    assert result > 0 and derivative[1] < 0 and derivative[2] == 0
    assert torch.autograd.gradcheck(objective, values, eps=1e-6, atol=1e-8, rtol=1e-5)
    assert report["scope"] == "fixed_original_material_samples"
    assert report["relationship_components"]["relationship/missing_release/material/0"] == [[0., 0., 0.]]


# BEGIN required surface contact keeps its original decision and finite objective.
def test_required_contact_has_a_trial_derivative_and_original_surface_evidence(tmp_path):
    from manipulation.planning_scene import ContactRule
    scene = _moving_scene(tmp_path)
    scene.contacts = (ContactRule(rule_id="surface", actor="material", target_owner="background",
        interval=(0., 1.), material_id="declared_face", required=True,
        feature_points=[[0., -.04, -.04], [0., .04, -.04],
                        [0., .04, .04], [0., -.04, .04]],
        feature_normal=[-1., 0., 0.], patch_origin=[0., 1., 0.],
        patch_axes=np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
        patch_half_size=[.04, .04], gap_bounds=(-.001, .001),
        normal_tolerance_rad=.1, modeling_allowance_m=.001, geometry_error_m=.001,
        source="declared finite surface"),)

    def inputs(x):
        position = torch.stack((x, x * 0 + 1, x * 0)).reshape(1, 1, 3)
        quaternion = x.new_tensor([[[1., 0., 0., 0.]]])
        progress, q = x.new_tensor([[.5]]), x.new_tensor([[[.04, -.02]]])
        return position, quaternion, progress, q, ("left_slide", "right_slide")

    def objective(x):
        args = inputs(x)
        return scene.optimization_objective(*args, trajectory_time=args[2],
            robot_link_poses=scene.kinematics.poses(args[3], args[4]))

    x = torch.tensor(.03, dtype=torch.float64, requires_grad=True)
    value = objective(x)
    assert value > 0 and torch.autograd.grad(value, x)[0] > 0
    assert torch.autograd.gradcheck(objective, x, eps=1e-6, atol=1e-7, rtol=1e-5)
    for offset, expected in ((.03, 1.), (0., 0.)):
        args = inputs(x.new_tensor(offset))
        actual = scene.components(None, *args, trajectory_time=args[2])
        assert actual["contact/surface"].item() == expected
        assert actual["unresolved/contact/surface"].item() == 0.
# END the finite optimization direction does not supply the contact verdict.
