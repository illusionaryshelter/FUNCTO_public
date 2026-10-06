"""True finite edges share the existing polygon/whole-intersection contract."""

# BEGIN CPU correspondence and counterexamples for native edge material.
from dataclasses import asdict

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
import torch

from manipulation.planning_scene import ContactRule
from manipulation.robot_geometry_check import NativePrimitive, RobotGeometryChecker, _feature_region
from test_independent_scene_validation import binding, checker
from test_manipulation_planning_scene import scene


def edge_rule(**changes):
    args = dict(rule_id="true_edge", actor="robot", target_owner="target", interval=(0., 1.),
        sphere_indices=(0,), material_id="contact_pad", feature_kind="edge",
        feature_points=[[0., -.05, 0.], [0., .05, 0.]], feature_normal=[0., 0., -1.],
        patch_origin=[0., 0., 0.], patch_axes=np.eye(3), patch_half_size=[.2, .2],
        gap_bounds=(-.0001, .0001), normal_tolerance_rad=.0001,
        modeling_allowance_m=.0001, geometry_error_m=.0001, minimum_thickness_m=.1,
        source="Complete native box edge and explicit finite test target")
    args.update(changes)
    return ContactRule(**args)


def pose(rotation=(0., 0., 0.), position=(0., 0., 0.)):
    value = np.eye(4)
    value[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    value[:3, 3] = position
    return value


def tensors(transform):
    return (torch.tensor(transform[:3, 3], dtype=torch.float64).reshape(1, 1, 3),
        torch.tensor(Rotation.from_matrix(transform[:3, :3]).as_quat()[[3, 0, 1, 2]],
                     dtype=torch.float64).reshape(1, 1, 4))


def torch_policy(rule, actor=np.eye(4), target=np.eye(4), *, actor_position=None):
    actor_pair = tensors(actor)
    if actor_position is not None:
        actor_pair = (actor_position, actor_pair[1])
    return scene()._permission(rule, torch.zeros((1, 1, 1, 4), dtype=torch.float64),
        actor_pair, tensors(target), torch.tensor([[.5]], dtype=torch.float64),
        torch.zeros((1, 1, 0), dtype=torch.float64), torch.zeros((1, 1, 0), dtype=torch.float64))[0]


@pytest.mark.parametrize("changes", [
    {"feature_kind": "point"}, {"feature_points": [[0., 0., 0.]]},
    {"feature_points": [[0., 0., 0.], [0., 0., 0.]]},
    {"feature_points": [[0., -.05, 0.], [0., 0., 0.], [0., .05, 0.]]},
    {"feature_normal": [0., 1., 0.]},
    {"feature_kind": "polygon"},
])
def test_edge_requires_exactly_two_distinct_endpoints_and_perpendicular_normal(changes):
    with pytest.raises(ValueError):
        edge_rule(**changes)


@pytest.mark.parametrize("global_pose", [np.eye(4), pose((.4, -.2, .3), (.13, -.27, .41))])
@pytest.mark.parametrize("reverse", [False, True])
def test_torch_and_independent_edge_halfspaces_agree_after_rigid_frame_changes(global_pose, reverse):
    rule = edge_rule()
    if reverse:
        rule = edge_rule(feature_points=rule.feature_points[::-1])
    target = global_pose @ pose((0., .00005, 0.))
    policy = torch_policy(rule, global_pose, target)
    region, reason = _feature_region(asdict(rule), global_pose, target, None)
    assert reason is None and policy["normal_valid"].item()
    # Torch uses the target owner's coordinates; the independent checker uses
    # world coordinates. Compare the complete transformed finite regions.
    normals = policy["normals"].numpy()[0, 0] @ target[:3, :3].T
    bounds = policy["bounds"].numpy()[0, 0]+normals @ target[:3, 3]
    # Halfspace row order has no geometric meaning; the two established
    # consumers enumerate the material normal's +/- planes in opposite order.
    expected = np.column_stack((region[0], region[1]))
    actual = np.column_stack((normals, bounds))
    left, right = linear_sum_assignment(np.linalg.norm(actual[:, None]-expected[None], axis=-1))
    np.testing.assert_allclose(actual[left], expected[right], atol=2e-14, rtol=0)


def test_edge_gap_projects_to_line_not_the_unoccupied_rest_of_a_plane():
    # The target normal tilts toward X. Every point on this Y edge has gap 0;
    # projecting onto an infinite Z plane would incorrectly add an X term.
    rule = edge_rule(patch_axes=Rotation.from_rotvec([0., .00005, 0.]).as_matrix())
    policy = torch_policy(rule)
    region, reason = _feature_region(asdict(rule), np.eye(4), np.eye(4), None)
    assert reason is None
    np.testing.assert_array_equal(policy["normals"].numpy()[0, 0, -2:], np.zeros((2, 3)))
    np.testing.assert_array_equal(region[0][-2:], np.zeros((2, 3)))
    np.testing.assert_allclose(region[1][-2:], [.000101, .000101], atol=1e-15, rtol=0)


def test_edge_gap_torch_gradient_matches_independent_translated_geometry():
    rule = edge_rule()
    position = torch.zeros((1, 1, 3), dtype=torch.float64, requires_grad=True)
    policy = torch_policy(rule, actor_position=position)
    policy["bounds"][0, 0, -2].backward()
    np.testing.assert_allclose(position.grad.numpy()[0, 0], [0., 0., -1.], atol=1e-14, rtol=0)
    step = 1e-7
    upper = []
    for z in (-step, step):
        region, reason = _feature_region(asdict(rule), pose(position=(0., 0., z)), np.eye(4), None)
        assert reason is None
        upper.append(region[1][-2])
    assert (upper[1]-upper[0])/(2*step) == pytest.approx(-1., abs=1e-10)


def test_edge_has_no_extra_lateral_material_allowance_or_unbounded_endpoints():
    rule = edge_rule()
    region, _ = _feature_region(asdict(rule), np.eye(4), np.eye(4), None)
    a, b = region
    assert np.all(a @ [0., 0., 0.] <= b)
    # Fifty micrometres is within the explicitly supplied normal allowance,
    # but no equally wide lateral tube or longer edge was authorized.
    assert np.all(a @ [0., 0., -.00005] <= b)
    assert np.any(a @ [.00005, 0., 0.] > b)
    assert np.any(a @ [0., .05005, 0.] > b)


def native_edge_query(checker, *, penetration=0., patch_half_size=(.2, .2), gap_offset=0.):
    local = pose((0., np.pi/4, 0.), (0., 0., np.sqrt(2)*.05-penetration))
    native = RobotGeometryChecker(checker.urdf_path, base_frame="base", geometries=[
        NativePrimitive("pad", "base", "box", (.1, .1, .1), local, "Complete tilted native box")])
    rule = edge_rule(feature_points=[[0., -.05, -penetration], [0., .05, -penetration]],
        patch_half_size=patch_half_size, patch_origin=[0., 0., gap_offset])
    model = binding([asdict(rule)])
    native.bind_scene(model)
    state = native.query(joint_names=native.joint_names, joint_positions=[0., 0.], T_reference_base=np.eye(4), pairs=())
    return native.query_scene(state, tool_pose=np.eye(4), progress=.5, finger_positions=[0., 0.], state_kind="actual")


def test_native_complete_edge_contact_is_permitted_without_disabling_its_whole_box(checker):
    result = native_edge_query(checker)
    assert result["passed"]
    row = result["pairs"][0]
    assert row["permission"] == "true_edge"
    assert row["contact_region_checks"][0]["contained"] is True


@pytest.mark.parametrize("changes", [
    {"penetration": .00005}, {"patch_half_size": (.2, .01)}, {"gap_offset": .001},
])
def test_native_full_side_overlap_edge_end_or_gap_cannot_borrow_edge_permission(checker, changes):
    result = native_edge_query(checker, **changes)
    assert not result["passed"]
    assert result["pairs"][0]["permission"] is None
    assert result["violations"]


def test_serialized_edge_is_restored_without_inventing_polygon_or_extra_vertices():
    rule = edge_rule()
    restored = ContactRule(**asdict(rule))
    assert restored.feature_kind == "edge" and restored.feature_points.shape == (2, 3)
    np.testing.assert_array_equal(restored._feature_edge_normals, rule._feature_edge_normals)
# END the same original full native solids remain active in every counterexample.
