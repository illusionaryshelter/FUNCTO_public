"""Continuous-model interval evidence and shared physical query budgets."""

from types import SimpleNamespace

import coal
import numpy as np
import pinocchio as pin
import pytest
from scipy.interpolate import CubicHermiteSpline

from manipulation.implicit_distance import ObservedField
from manipulation.interval_geometry import (
    SceneMotionBounds, _polynomial_ranges, check_interval_geometry,
)
from manipulation.native_field_distance import query_native_field
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from manipulation.motion import MotionModel
from test_geometry_motion_bound import model


def trajectory(*, duration=.025, position=None, velocity=None, names=("q",)):
    q = np.asarray([[0.], [duration]] if position is None else position, float)
    dq = np.ones_like(q) if velocity is None else np.asarray(velocity, float)
    times = np.linspace(0., duration, len(q))
    return SimpleNamespace(duration=duration, times=times, joint_names=names,
        _curve=CubicHermiteSpline(times, q, dq, axis=0),
        progress_u=np.array([0., 1.]), progress_s=np.array([0., 1.]))


class ConstantMotion:
    def __init__(self, speed=1., *, boundary=None, unknown=False):
        self.speed, self.boundary, self.unknown = speed, boundary, unknown

    def declaration_times(self, curve):
        return {} if self.boundary is None else {self.boundary: {-1, 1}}

    def partition_times(self, curve):
        return [0., curve.duration] if self.boundary is None else [0., self.boundary, curve.duration]

    def interval(self, curve, start, end):
        return {"speeds": {"robot/body": self.speed, "background": 0.},
                "unavailable": {"robot/finger": {"reason": "unmodelled_joint_motion"}} if self.unknown else {}}


def pair(**values):
    return {"component": "collision/body/background", "first": "body", "second": "background",
            "status": "clear", "lower_m": .1, "permission_kind": "none",
            "geometry_entity_key": {"first": f"robot/{values.get('first', 'body')}", "second": "background"},
            **values}


def callback(function, *, sides=(0,), anchor_cost=0):
    calls = []

    def query(times, *, point_budget):
        needed = len(set(times))+anchor_cost
        calls.append((list(times), point_budget))
        if needed > point_budget:
            return {"states": [], "query_points": 0, "capacity_exhausted": True}
        return {"states": [{"time_s": time, "side": side, "pairs": function(time, side)}
                           for time in times for side in sides],
                "query_points": needed, "capacity_exhausted": False}

    query.calls = calls
    return query


def test_original_curve_midpoint_finds_thin_observed_entity_between_grid_samples():
    curve = trajectory()
    field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[.0115, -.001, -.001], voxel_size=.002)
    shape = coal.Box(.002, .002, .002)

    def evidence(time, side):
        pose = np.eye(4)
        pose[0, 3] = curve._curve(time)[0]
        return [pair(**query_native_field(shape, pose, field, np.eye(4)))]

    query = callback(evidence)
    result = check_interval_geometry(curve, ConstantMotion(), query,
        max_query_points=20, spatial_step_m=.001)
    assert result["confirmed_intersections"][0]["time_s"] == .0125
    assert result["query_points"] == 3
    assert len(query.calls) == 2
    assert result["unresolved_intervals"][0]["reason"] == "not_checked_after_intersection"


def test_separation_uses_full_interval_motion_and_can_prove_without_subdivision():
    query = callback(lambda time, side: [pair(lower_m=.04)])
    result = check_interval_geometry(trajectory(), ConstantMotion(), query,
        max_query_points=2, spatial_step_m=.0001)
    assert len(result["proved_intervals"]) == 1
    assert result["query_points"] == 2
    assert not result["unresolved_intervals"]


def test_finite_permission_lower_bound_cannot_be_used_as_whole_entity_separation():
    query = callback(lambda time, side: [pair(lower_m=100., permission_kind="finite_region")])
    result = check_interval_geometry(trajectory(), ConstantMotion(), query,
        max_query_points=40, spatial_step_m=.01)
    assert not result["proved_intervals"]
    assert not result["confirmed_intersections"]
    assert all(row["maximum_entity_movement_m"] <= .01 for row in result["unresolved_intervals"])
    assert all(row["pairs"][0]["reason"] == "dynamic_permission" for row in result["unresolved_intervals"])


# BEGIN independent complete-entity evidence can prove a permitted pair clear.
def test_finite_permission_uses_separate_full_entity_bound_without_subdivision():
    query = callback(lambda time, side: [pair(lower_m=0., full_entity_lower_m=.04,
                                              permission_kind="finite_region")])
    result = check_interval_geometry(trajectory(), ConstantMotion(), query,
        max_query_points=2, spatial_step_m=.001)
    assert result["query_points"] == 2 and len(query.calls) == 1
    assert len(result["proved_intervals"]) == 1
    assert not result["unresolved_intervals"] and not result["confirmed_intersections"]


def test_large_clipped_bound_cannot_replace_inconclusive_complete_entity_bound():
    query = callback(lambda time, side: [pair(lower_m=100., full_entity_lower_m=0.,
                                              permission_kind="finite_region")])
    result = check_interval_geometry(trajectory(), ConstantMotion(), query,
        max_query_points=40, spatial_step_m=.001)
    assert not result["proved_intervals"] and not result["confirmed_intersections"]
    assert result["unresolved_intervals"] and result["query_points"] == 33
# END enclosure overlap remains unresolved under the existing sampling rule.


def test_actual_anchor_points_consume_shared_budget_and_capacity_remains_unresolved():
    query = callback(lambda time, side: [pair(lower_m=0.)], anchor_cost=1)
    result = check_interval_geometry(trajectory(), ConstantMotion(), query,
        max_query_points=4, spatial_step_m=.001)
    assert result["query_points"] == 3
    assert not result["confirmed_intersections"]
    assert any(row["reason"] == "sampling_capacity" for row in result["unresolved_intervals"])
    assert all(budget >= 0 for _, budget in query.calls)


def test_cached_sides_remain_distinct_at_declared_relation_boundary():
    curve, boundary = trajectory(), .0125
    query = callback(lambda time, side: [pair(lower_m=.1 if side != 1 else 0.)], sides=(-1, 0, 1))
    result = check_interval_geometry(curve, ConstantMotion(boundary=boundary), query,
        max_query_points=3, spatial_step_m=.02)
    assert len(result["proved_intervals"]) == 2  # Far opposite endpoints prove both spans.
    assert result["query_points"] == 3
    query_missing = callback(lambda time, side: [pair()])
    result = check_interval_geometry(curve, ConstantMotion(boundary=boundary), query_missing,
        max_query_points=8, spatial_step_m=.02)
    assert not result["proved_intervals"]
    assert all(row["reason"] == "missing_boundary_side" for row in result["unresolved_intervals"])


def test_previously_charged_initial_states_are_reused_with_zero_remaining_budget():
    curve = trajectory()
    initial = [{"time_s": time, "side": 0, "pairs": [pair()]} for time in (0., curve.duration)]
    query = callback(lambda time, side: pytest.fail("Cached physical points must be reused"))
    result = check_interval_geometry(curve, ConstantMotion(), query,
        max_query_points=0, spatial_step_m=.001, initial_states=initial)
    assert len(result["proved_intervals"]) == 1
    assert result["query_points"] == 0


def test_unknown_finger_does_not_prevent_known_body_interval_refinement():
    def evidence(time, side):
        return [pair(lower_m=0., status="intersection" if time == .0125 else "unresolved"),
            pair(first="finger", status="unmodelled_motion", lower_m=None)]
    result = check_interval_geometry(trajectory(), ConstantMotion(unknown=True), callback(evidence),
        max_query_points=6, spatial_step_m=.001)
    assert result["confirmed_intersections"][0]["time_s"] == .0125


def test_bernstein_derivative_enclosure_covers_all_original_polynomial_pieces():
    rng = np.random.default_rng(74)
    curve = trajectory(duration=2., position=rng.normal(size=(8, 3)),
        velocity=rng.normal(size=(8, 3)), names=("a", "b", "c"))
    for start, end in [(0., 2.), (.21, 1.67), (.71, .73)]:
        lo, hi = _polynomial_ranges(curve._curve, start, end, 1)
        values = curve._curve(np.linspace(start, end, 500), 1)
        assert np.all(values >= lo) and np.all(values <= hi)


def bound_scene(*, owners=(), nominal=None):
    # BEGIN bind scene tests to the same native model type as production.
    native = model([0, 1], [pin.JointModelRZ(), pin.JointModelPX()],
        names=("arm", "finger"), limits=[[-2., 0.], [2., .1]])
    scene = SimpleNamespace(robot_model=SimpleNamespace(_geometries={
        "body": ("link1", coal.Box(.02, .02, .02), "declared", np.eye(4)),
        "finger": ("link2", coal.Box(.02, .02, .02), "declared", np.eye(4))}),
        _native_local_bounds={name: np.array([[-.01]*3, [.01]*3]) for name in ("body", "finger")},
        owners=owners, gripper=SimpleNamespace(nominal_finger_trajectory=nominal),
        required_progress_anchors=tuple(value for owner in owners for relation in owner.attachments
                                        for value in relation.interval),
        kinematics=SimpleNamespace(model=native),
        robot_validator=SimpleNamespace(T_reference_base=np.eye(4)))
    # END full finger motion is carried by the native joint tree.
    return scene


def test_binding_tracks_complete_finger_tree_and_attached_original_field():
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[.2, 0., 0.], voxel_size=.1,
        name="owner", source="synthetic RGB-D samples")
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "declared pose")
    owner = CollisionOwner("owner", field, path, "observed field", attachments=(
        Attachment((.25, .75), [0., 0., 0.], [1., 0., 0., 0.], "hold", "release"),))
    nominal = trajectory(duration=1., names=("finger",), position=[[0.], [.1]], velocity=[[.1], [.1]])
    nominal.valid_until_s = .5
    scene = bound_scene(owners=(owner,), nominal=nominal)
    bound = SceneMotionBounds.from_scene(scene, tool_link="link1", T_link_tool=np.eye(4))
    curve = trajectory(duration=1., names=("arm",))
    early = bound.interval(curve, .3, .4)
    late = bound.interval(curve, .6, .7)
    released = bound.interval(curve, .8, .9)
    assert early["speeds"]["owner/owner"] >= np.sqrt(.3**2+.1**2+.1**2)
    assert early["speeds"]["robot/finger"] > early["speeds"]["robot/body"]
    assert "robot/finger" in late["unavailable"] and "robot/body" not in late["unavailable"]
    assert released["speeds"]["owner/owner"] == 0.
    np.testing.assert_array_equal(bound.partition_times(curve), [0., .25, .5, .75, 1.])


def test_progress_plateau_requests_only_its_actual_relation_sides():
    bound = SceneMotionBounds.from_scene(bound_scene(), tool_link="link1", T_link_tool=np.eye(4))
    bound.scene.required_progress_anchors = (.5,)
    curve = trajectory(duration=1., names=("arm",))
    curve.progress_u = np.array([0., .25, .75, 1.])
    curve.progress_s = np.array([0., .5, .5, 1.])
    assert bound.declaration_times(curve) == {.25: {-1}, .75: {1}}

    def query(times, *, point_budget):
        states = []
        for time in times:
            sides = [0, *bound.declaration_times(curve).get(time, ())]
            states.extend({"time_s": time, "side": side,
                "pairs": [pair(first="body", lower_m=1.)]} for side in sides)
        return {"states": states, "query_points": len(times), "capacity_exhausted": False}

    result = check_interval_geometry(curve, bound, query, max_query_points=10, spatial_step_m=.001)
    assert len(result["proved_intervals"]) == 3
    assert not result["unresolved_intervals"]


def test_declared_unwrapped_mechanism_turn_retains_motion_when_endpoint_poses_coincide():
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[0., 0., 0.], voxel_size=.01,
        name="owner", source="synthetic RGB-D samples")
    reference = np.eye(4)
    reference[0, 3] = .2
    coordinates = np.array([[0.], [2*np.pi]])
    mechanism = MotionModel("revolute", coordinates, reference, np.eye(3), np.zeros(3),
        .001, .001, {"source": "declared coordinate relation"})
    path = PosePath(np.array([0., 1.]), mechanism.poses(coordinates), "declared mechanism",
        motion_model=mechanism, coordinates=coordinates)
    owner = CollisionOwner("owner", field, path, "observed field", moving=True)
    bound = SceneMotionBounds.from_scene(bound_scene(owners=(owner,)),
        tool_link="link1", T_link_tool=np.eye(4))
    curve = trajectory(duration=1., names=("arm",))
    result = bound.interval(curve, 0., 1.)
    np.testing.assert_allclose(path.poses[0], path.poses[1], atol=1e-15)
    assert result["speeds"]["owner/owner"] >= 2*np.pi*(.2+np.sqrt(3)*.01)


def test_original_polynomial_knots_do_not_force_redundant_geometry_queries():
    bound = SceneMotionBounds.from_scene(bound_scene(), tool_link="link1", T_link_tool=np.eye(4))
    curve = trajectory(duration=1., position=np.linspace(0., 1., 650)[:, None],
        velocity=np.ones((650, 1)), names=("arm",))
    np.testing.assert_array_equal(bound.partition_times(curve), [0., 1.])


def test_same_display_name_retains_distinct_robot_and_owner_motion_bounds():
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[0., 0., 0.], voxel_size=.01,
        name="body", source="synthetic RGB-D samples")
    path = PosePath(np.array([0., 1.]), np.stack([np.eye(4), np.eye(4)]), "fixed observed owner")
    owner = CollisionOwner("body", field, path, "observed field")
    bound = SceneMotionBounds.from_scene(bound_scene(owners=(owner,)),
        tool_link="link1", T_link_tool=np.eye(4))
    curve = trajectory(duration=1., names=("arm",))
    speeds = bound.interval(curve, 0., 1.)["speeds"]
    assert speeds["robot/body"] > .01 and speeds["owner/body"] < 1e-12
    query = callback(lambda time, side: [pair(first="body", second="body",
        component="collision/robot/body", lower_m=.001,
        status="intersection" if time == .5 else "clear",
        geometry_entity_key={"first": "robot/body", "second": "owner/body"})])
    result = check_interval_geometry(curve, bound, query, max_query_points=4, spatial_step_m=.001)
    assert result["confirmed_intersections"][0]["time_s"] == .5


def test_scene_pair_producer_namespaces_an_owner_equal_to_a_robot_geometry(tmp_path):
    import torch
    from test_native_scene_contract import make_scene
    from test_geometry_query_reuse import owner
    scene = make_scene(tmp_path)
    scene.owners = (owner("first"),)
    evidence = []
    scene.evaluate(None, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.zeros((1, 1)),
        torch.tensor([[[.04, -.02]]], dtype=torch.float64), ("left_slide", "right_slide"),
        trajectory_time=torch.zeros((1, 1)), geometry_evidence_out=evidence)
    same_name = next(row for row in evidence if row["first"] == row["second"] == "first")
    assert same_name["geometry_entity_key"] == {"first": "robot/first", "second": "owner/first"}


def test_unknown_robot_finger_does_not_hide_an_owner_with_the_same_name(tmp_path, monkeypatch):
    import torch
    from manipulation.planning_scene import GripperState
    from test_native_scene_contract import make_scene
    from test_geometry_query_reuse import owner
    scene = make_scene(tmp_path)
    scene.owners = (owner("first"),)
    scene.gripper = SimpleNamespace(joint_names=("left_slide",),
        state=lambda progress, q, names, **kwargs: GripperState(q[..., :1], q[..., :1]*0,
            progress*0, torch.ones_like(progress, dtype=torch.bool)))
    monkeypatch.setattr(scene.robot_model, "geometry_depending_on", lambda names: ("first",))
    evidence = []
    scene.evaluate(None, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.zeros((1, 1)),
        torch.tensor([[[.04, -.02]]], dtype=torch.float64), ("left_slide", "right_slide"),
        trajectory_time=torch.zeros((1, 1)), geometry_evidence_out=evidence)
    known_pair = next(row for row in evidence
        if row["geometry_entity_key"] == {"first": "robot/second", "second": "owner/first"})
    assert known_pair["status"] == "clear"
    assert any(row["status"] == "unmodelled_motion" for row in evidence
        if row["geometry_entity_key"]["first"] == "robot/first")


def test_scene_output_contains_complete_pair_records_through_evaluate(tmp_path):
    import torch
    from test_native_scene_contract import make_scene
    scene = make_scene(tmp_path)
    q = torch.tensor([[[.04, -.02]]], dtype=torch.float64)
    evidence = []
    scene.evaluate(None, torch.zeros((1, 1, 3), dtype=torch.float64),
        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64), torch.zeros((1, 1)),
        q, ("left_slide", "right_slide"), trajectory_time=torch.zeros((1, 1)),
        geometry_evidence_out=evidence)
    assert len(evidence) == 3
    native = next(row for row in evidence if row["component"] == "collision/robot/self")
    assert 0 < native["lower_m"] <= .001
    assert native["distance_scope"] == "full_entities"
    assert all("permission_kind" in row and "state_index" in row for row in evidence)
