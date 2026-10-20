"""Permission compatibility does not manufacture visible contact evidence."""

# BEGIN continuous finger-domain and material-specific grasp-region regressions.
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment
from scipy.spatial.transform import Rotation

from manipulation.grasp_region import check_grasp_region_compatibility
from manipulation.grasp_region import _feature_region_halfspaces


def permission(side, *, half_gap=.02, material=None, rule_id=None, joints=1, joint=0):
    """An explicitly calibrated pad and fixed target patch, without spheres."""
    points = np.array([[0., y, z] for y in (-.01, .01) for z in (-.01, .01)])
    mapping = np.zeros((4, 3, joints))
    mapping[:, 0, joint] = side
    return dict(rule_id=rule_id or f"region_{side}", actor="robot", target_owner="target",
                material_id=material or f"pad_{side}", source="explicit independent geometric fixture",
                feature_kind="polygon", feature_points=points,
                feature_normal=[-side, 0., 0.], feature_joint_translation=mapping,
                patch_origin=[side*half_gap, 0., 0.],
                patch_axes=np.column_stack(([0., 1., 0.], [0., 0., side], [side, 0., 0.])),
                patch_half_size=[.01, .01], gap_bounds=[-.00001, .00001],
                normal_tolerance_rad=.01, modeling_allowance_m=0., geometry_error_m=0.,
                numerical_position_tolerance_m=.000001,
                numerical_angle_tolerance_rad=.000001)


def query(rules, limits=((0., .04),), tool=None, owner=None):
    return check_grasp_region_compatibility(rules,
        T_reference_tool=np.eye(4) if tool is None else tool,
        T_reference_owner=np.eye(4) if owner is None else owner,
        finger_position_limits=limits)


def test_intermediate_shared_finger_position_is_not_rejected_by_endpoint_sampling():
    rules = [permission(-1), permission(1)]
    original = deepcopy(rules)
    result = query(rules)
    assert result["status"] == "compatible"
    q = np.asarray(result["hypothetical_finger_positions_m"])
    assert .01999 < q[0] < .02001
    # Both endpoints are genuinely impossible; solving the continuous domain
    # is necessary. Missing side observations are not inputs to this check.
    assert query(rules, ((0., 0.),))["status"] == "contradicted"
    assert query(rules, ((.04, .04),))["status"] == "contradicted"
    assert result["actual_contact_verified"] is False
    assert result["observed_target_surface_verified"] is False
    assert result["holding_verified"] is False
    assert result["finger_trajectory_predicted"] is False
    for rule, before in zip(rules, original):
        for key in rule:
            np.testing.assert_array_equal(rule[key], before[key])


def test_each_material_needs_a_common_q_not_separate_incompatible_witnesses():
    rules = [permission(-1, half_gap=.01), permission(1, half_gap=.03)]
    assert all(query([row])["status"] == "compatible" for row in rules)
    result = query(rules)
    assert result["status"] == "contradicted"
    # Independent measured coordinates are not arbitrarily coupled to one
    # common closure fraction. The same asymmetric target is then admissible.
    independent = [permission(-1, half_gap=.01, joints=2, joint=0),
                   permission(1, half_gap=.03, joints=2, joint=1)]
    accepted = query(independent, ((0., .04), (0., .04)))
    assert accepted["status"] == "compatible"
    np.testing.assert_allclose(accepted["hypothetical_finger_positions_m"], [.01, .03], atol=2e-6)

def test_source_feature_heights_are_preserved_without_plane_projection():
    rule = permission(-1)
    rule["feature_points"][1, 0] = 4e-7
    original = rule["feature_points"].copy()
    assert query([rule])["status"] == "compatible"
    np.testing.assert_array_equal(rule["feature_points"], original)


def test_alternative_rules_for_one_material_are_or_not_and():
    unreachable = permission(-1, rule_id="left_elsewhere")
    unreachable["patch_origin"][1] = .1
    reachable = permission(-1, rule_id="left_here")
    result = query([unreachable, reachable, permission(1)])
    assert result["status"] == "compatible"
    assert "left_here" in result["selected_rule_ids"]
    assert "left_elsewhere" not in result["selected_rule_ids"]
    assert result["checked_rule_combinations"] == 2


def test_grasp_origin_in_patch_does_not_replace_finite_pad_reachability():
    rule = permission(-1, half_gap=0.)
    rule["feature_points"][:, 1] += .05
    # The tool origin equals patch origin, but its actual pad is displaced
    # tangentially. Closing along X cannot bring this finite face to the patch.
    assert query([rule])["status"] == "contradicted"


def test_normal_permissions_are_material_specific_without_automatic_pad_swaps():
    rules = [permission(-1), permission(1)]
    rotated = np.eye(4)
    rotated[:3, :3] = Rotation.from_rotvec([0., 0., np.pi]).as_matrix()
    result = query(rules, tool=rotated)
    assert result["status"] == "contradicted"
    assert result["normal_rejected_rule_ids"]


def test_rigid_gauge_change_and_witnesses_agree_with_existing_fixed_q_checker():
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([.2, -.4, .7]).as_matrix()
    transform[:3, 3] = [.13, -.52, .81]
    rules = [permission(-1), permission(1)]
    result = query(rules, tool=transform, owner=transform)
    assert result["status"] == "compatible"
    q = np.asarray(result["hypothetical_finger_positions_m"])
    for rule, point in zip(rules, result["hypothetical_reference_contact_points"]):
        region, _, reason = _feature_region_halfspaces(rule, transform, transform, q)
        assert reason is None
        assert np.max(region[0] @ point-region[1]) <= 2e-9


@pytest.mark.parametrize("kind", ["polygon", "edge"])
def test_affine_region_derivative_preserves_the_fixed_q_native_law(kind):
    rule = permission(-1, joints=2)
    rule["feature_kind"] = kind
    if kind == "edge":
        rule["feature_points"] = np.array([[0., -.01, 0.], [0., .01, 0.]])
    rule["feature_joint_translation"] = np.repeat(
        np.array([[[-1., .2], [.3, -.1], [.1, .2]]]), len(rule["feature_points"]), axis=0)
    rule["normal_tolerance_rad"] = .1
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([.01, .02, -.01]).as_matrix()
    q0, dq = np.array([.02, .01]), np.array([.003, -.005])
    region0, derivative, reason0 = _feature_region_halfspaces(rule, tool, np.eye(4), q0)
    region1, _, reason1 = _feature_region_halfspaces(rule, tool, np.eye(4), q0+dq)
    assert reason0 is reason1 is None
    predicted = np.column_stack((region0[0], region0[1]+derivative@dq))
    actual = np.column_stack(region1)
    # ConvexHull may enumerate equivalent polygon edges in another order.
    a, b = linear_sum_assignment(np.linalg.norm(predicted[:, None]-actual[None], axis=-1))
    np.testing.assert_allclose(predicted[a], actual[b], atol=3e-14, rtol=0)


def test_solver_incompleteness_is_unresolved_not_impossible(monkeypatch):
    monkeypatch.setattr("manipulation.grasp_region.linprog",
                        lambda *args, **kwargs: SimpleNamespace(success=False, status=4))
    result = query([permission(-1), permission(1)])
    assert result["status"] == "unresolved"
    assert result["incomplete_rule_combinations"][0]["solver_status"] == 4


@pytest.mark.parametrize("mutation", [
    lambda row: row.pop("gap_bounds"),
    lambda row: row.pop("numerical_position_tolerance_m"),
    lambda row: row.update(actor="payload"),
    lambda row: row.update(feature_joint_translation=np.zeros((4, 3, 2))),
    lambda row: row.update(normal_tolerance_rad=float("nan")),
])
def test_missing_or_misbound_geometry_does_not_gain_defaults(mutation):
    rule = permission(-1)
    mutation(rule)
    with pytest.raises(ValueError):
        query([rule])
# END these tests prove declared-domain geometry only, never a physical grasp.
