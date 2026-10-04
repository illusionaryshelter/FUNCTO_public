"""CPU geometric sampling, owner motion and bounded export-history regressions."""
from copy import copy
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation
import torch

from manipulation.geometry_motion_bound import GeometryMotionBound
from manipulation.motion import MotionModel
from manipulation.planning import (JointTrajectory, PlanningBudget, _ProgressMap,
    _cubic_speed_upper, _export_sample_times, _owner_path_speed, _export_motion_inputs, _anchor_capacity,
    _geometry_finger_model, _validation_storage_admission)
from manipulation.collision_records import encode_array, decode_array
from manipulation.voxel_layers import LayeredVoxelQuery
from manipulation.planning_problem import MotionPhase, PosePath
from manipulation.planning_scene import Attachment, PlanningScene
from test_manipulation_planning import cartesian_problem, constant_path, export_fixture


def test_derivative_vertex_and_reversing_excursion_are_not_endpoint_motion():
    curve = CubicHermiteSpline([0., 1.], [[0., 0.], [1., 0.]], [[0., 4.], [0., -4.]])
    bound = _cubic_speed_upper(curve, np.array([0.]), np.array([1.]))[0]
    assert 1.5 <= bound[0] < 1.5+1e-12
    assert 4. <= bound[1] < 4.+1e-12
    # The second coordinate returns to its start after a 1 m excursion.
    assert curve(.5)[1] == 1.


def test_random_derivative_intervals_enclose_dense_polynomial_oracle():
    rng = np.random.default_rng(743)
    curve = CubicHermiteSpline(np.arange(12.), rng.normal(size=(12, 5)), rng.normal(size=(12, 5)))
    left, right = np.arange(11.)+.07, np.arange(11.)+.93
    bound = _cubic_speed_upper(curve, left, right)
    for i in range(len(left)):
        assert np.all(np.abs(curve(np.linspace(left[i], right[i], 1001), 1)).max(axis=0) <= bound[i])


def test_expanded_work_never_expands_original_native_anchor_capacity():
    budget = PlanningBudget(max_validation_points=32768, max_refinement_points=32768)
    clock = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    assert _anchor_capacity(clock, [0., 1.], budget) == 8192
    with pytest.raises(ValueError, match="anchors exceed"):
        _anchor_capacity(clock, np.linspace(0., 1., 8193), budget)


def test_full_record_capacity_counts_typed_modes_bodies_targets_and_moving_actors():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    layer = LayeredVoxelQuery.__new__(LayeredVoxelQuery)
    scene = NS(required_progress_anchors=(0., 1.), background=layer, owners=(),
               robot_collision_bodies=tuple({"body_id": f"body-{i}"} for i in range(13)))
    metadata = {"active_joint_names": problem.joint_names}
    report = _validation_storage_admission(problem, scene, metadata, PlanningBudget(max_validation_points=32768))
    assert report["typed_modes"] == 3 and report["layered_body_records_per_typed_state"] == 13
    assert 160*1024**2 < report["encoded_record_bytes_upper"] < 512*1024**2
    scene.owners = (NS(owner_id="payload", moving=True, query=layer),)
    report = _validation_storage_admission(problem, scene, metadata, PlanningBudget(max_validation_points=32768))
    assert report["layered_actor_target_pairs"] == 3
    assert report["layered_body_records_per_typed_state"] == 27
    scene.robot_collision_bodies = tuple({"body_id": f"body-{i}"} for i in range(64))
    with pytest.raises(ValueError, match="Complete typed collision records"):
        _validation_storage_admission(problem, scene, metadata, PlanningBudget(max_validation_points=65536))


@pytest.mark.parametrize("kind", ["support", "interior", "coupling", "retained_phase_outputs"])
def test_large_full_history_tensors_are_rejected_using_metadata_before_allocation(kind):
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    # Broadcast metadata makes even the counterexample input allocation tiny.
    points = np.broadcast_to(0., (1000, 3))
    planes = np.broadcast_to(0., (1000, 4))
    if kind == "support":
        problem.supports = (NS(body_points=points, contact_points=points, surface=NS(_hull=NS(equations=planes))),)
    elif kind == "interior":
        problem.interiors = (NS(body_points=points, halfspaces=planes),)
    elif kind == "coupling":
        problem.couplings = (NS(matrix=np.broadcast_to(0., (8192, 3))),)
    else:
        problem.phases = (problem.phases[0],)*1000
    scene = NS(required_progress_anchors=(), background=None, owners=())
    with pytest.raises(ValueError, match="Full-history process tensors"):
        _validation_storage_admission(problem, scene, {"active_joint_names": problem.joint_names},
                                      PlanningBudget(max_validation_points=65536))


def test_analytic_solids_use_explicit_metric_step_and_voxels_only_tighten_it():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    planner, _ = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                               np.zeros((2, 3)), np.zeros((2, 3)), 1.)
    params, scene = planner._verification_fk.config.kinematics_config, planner._verification_scene
    scene.background = NS(independent_geometry={"solids": [{"kind": "cuboid"}]})
    assert _export_motion_inputs(params, scene, problem, spatial_step_m=.001)[1] == .001
    scene.background.independent_geometry["solids"] = [{"kind": "voxel", "voxel_size": .0015}]
    assert _export_motion_inputs(params, scene, problem, spatial_step_m=.001)[1] == .00075
    with pytest.raises(ValueError, match="finite positive"):
        _export_motion_inputs(params, scene, problem, spatial_step_m=0.)


def test_out_of_range_cubic_cannot_use_a_smaller_declared_reach_bound():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    planner, _ = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                               np.zeros((2, 3)), np.zeros((2, 3)), 1.)
    params = planner._verification_fk.config.kinematics_config
    params.joint_limits.position = torch.tensor([[-.1]*3, [.1]*3])
    motion, _, _ = _export_motion_inputs(params, planner._verification_scene, problem, spatial_step_m=.001)
    # Endpoints are legal, but the first coordinate reaches 1 m between them.
    curve = JointTrajectory(problem.joint_names, np.array([0., 1.]), np.zeros((2, 3)),
                            np.array([[4., 0., 0.], [-4., 0., 0.]]), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    curve.finger_trajectory = None
    with pytest.raises(ValueError, match="inside its declared joint position limits"):
        _export_sample_times(curve, motion, spatial_step=.001, anchors=[0., 1.], budget=PlanningBudget())


def test_nominal_finger_excursion_is_sampled_even_when_executable_command_stays_still():
    curve = JointTrajectory(("arm",), np.array([0., 1.]), np.zeros((2, 1)), np.zeros((2, 1)),
                            np.array([0., 1.]), np.array([0., 1.]), (), (), {})
    command = NS(joint_names=("finger",), _curve=CubicHermiteSpline([0., 1.], [[0.], [0.]], [[0.], [0.]]))
    nominal = NS(joint_names=("finger",), _curve=CubicHermiteSpline([0., .2, .21, .22, 1.],
        [[0.], [0.], [.02], [0.], [0.]], np.zeros((5, 1))))
    curve.finger_trajectory = command
    hand = NS(controller="timed_reference", finger_trajectory=command, nominal_finger_trajectory=nominal)
    bound = GeometryMotionBound(("finger cover",), ("arm", "finger"), np.array([[0., 1.]]),
                                np.array([[-1., -1.], [1., 1.]]), np.array([1.]))
    times, report = _export_sample_times(curve, bound, spatial_step=.001, anchors=[0., 1.],
        budget=PlanningBudget(), finger_geometry=_geometry_finger_model(hand))
    assert .21 in times and nominal._curve(times).max() == .02
    assert len(times) > 80 and report["maximum_interval_relative_displacement_bound_m"] <= .001
    assert curve.finger_trajectory is command
    assert command._curve(times).max() == 0.


@pytest.mark.parametrize("has_nominal", [False, True])
def test_binary_hand_with_or_without_nominal_keeps_its_actual_geometry_contract(has_nominal):
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    planner, _ = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                               np.zeros((2, 3)), np.zeros((2, 3)), 1.)
    nominal = NS(joint_names=("finger",), _curve=CubicHermiteSpline([0., 1.], [[0.], [.02]], [[0.], [0.]]))
    hand = NS(controller="binary", nominal_finger_trajectory=nominal if has_nominal else None,
        finger_trajectory=None, joint_names=("finger",), joint_limits=np.array([[0., .04]]),
        sphere_indices=(0,), local_spheres=np.array([[0., 0., 0., .001]]),
        sphere_joint_translation=np.array([[[1.], [0.], [0.]]]))
    scene = planner._verification_scene
    scene.gripper = hand
    bound, _, _ = _export_motion_inputs(planner._verification_fk.config.kinematics_config,
                                       scene, problem, spatial_step_m=.001)
    if has_nominal:
        assert bound.joint_names[-1] == "finger"
        assert _geometry_finger_model(hand) is nominal
    else:
        assert bound.joint_names == problem.joint_names
        assert _geometry_finger_model(hand) is None
        # Same mechanical envelope as hand.spheres: centre .02, radius .021.
        assert .041 <= bound.local_radius_bounds_m[0] < .041+1e-12
        assert "unknown_hand_mechanical_envelope" in bound.body_names


def owner(path, *, spheres=((1., 0., 0., .2),), interval=(0., 1.)):
    return NS(owner_id="test owner", pose_path=path, local_spheres=np.array(spheres),
              sphere_cover_error_m=.03, pose_interval=interval)


def test_free_owner_small_angle_nlerp_peak_and_full_cover_offset():
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, :3, :3] = Rotation.from_rotvec([0., 0., .04]).as_matrix()
    offset = np.eye(4)
    offset[:3, 3] = [.5, 0., 0.]
    path = PosePath(np.array([0., 1.]), poses, "analytic path", attachment=offset)
    rate = _owner_path_speed(owner(path, interval=(2., 4.)))
    expected = 4*np.tan(.04/4)*(1.2+.03+.5)/2
    assert expected <= rate < expected+1e-12
    assert rate > .04*(1.2+.03+.5)/2  # slerp speed alone underbounds nlerp.


@pytest.mark.parametrize("kind,coordinates,expected", [
    ("revolute", [[0.], [2.]], 2*(2.+1.23)),
    ("prismatic", [[0.], [-.8]], .8),
    ("planar", [[0., 0., 0.], [.3, -.4, 2.]], .7+2*1.23),
    ("translation_plane", [[0., 0.], [.3, -.4]], .7),
])
def test_all_existing_mechanism_families_have_point_motion_bounds(kind, coordinates, expected):
    reference = np.eye(4)
    reference[:3, 3] = [2., 0., 0.]
    model = MotionModel(kind, np.asarray(coordinates), reference, np.eye(3), np.zeros(3), .01, .01, {})
    path = PosePath(np.array([0., 1.]), model.poses(coordinates), "analytic mechanism",
                    motion_model=model, coordinates=np.asarray(coordinates))
    actual = _owner_path_speed(owner(path))
    assert expected <= actual < expected+1e-11


def test_unknown_owner_motion_cannot_be_replaced_by_zero_speed():
    with pytest.raises(ValueError, match="custom pose sampler"):
        _owner_path_speed(owner(NS(sample=lambda x: x)))
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[1, 0, 3] = 1.
    value = owner(PosePath(np.array([0., 1.]), poses, "moving owner"))
    value.local_spheres = None
    with pytest.raises(ValueError, match="complete declared sphere cover"):
        _owner_path_speed(value)


def test_static_mechanism_needs_no_cover_but_full_rotation_is_not_static():
    model = MotionModel("revolute", np.zeros((2, 1)), np.eye(4), np.eye(3), np.zeros(3), .01, .01, {})
    fixed = PosePath(np.array([0., 1.]), model.poses([[0.], [0.]]), "fixed mechanism",
                     motion_model=model, coordinates=np.zeros((2, 1)))
    value = owner(fixed)
    value.local_spheres = None
    assert _owner_path_speed(value) == 0.
    value.pose_path = PosePath(np.array([0., 1.]), model.poses([[0.], [2*np.pi]]), "full rotation",
                               motion_model=model, coordinates=np.array([[0.], [2*np.pi]]))
    with pytest.raises(ValueError, match="complete declared sphere cover"):
        _owner_path_speed(value)


def test_actual_progress_rate_adds_external_relative_motion_and_budget_fails_before_grid():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    planner, _ = export_fixture(problem, _ProgressMap(np.array([0., .2, 1.]), np.array([0., 1., 1.])),
                               np.zeros((2, 3)), np.zeros((2, 3)), 1.)
    motion, _, _ = _export_motion_inputs(planner._verification_fk.config.kinematics_config,
                                        planner._verification_scene, problem, spatial_step_m=.001)
    curve = JointTrajectory(problem.joint_names, np.array([0., 1.]), np.zeros((2, 3)), np.zeros((2, 3)),
                            np.array([0., .2, 1.]), np.array([0., 1., 1.]), (), (), {})
    curve.finger_trajectory = None
    times, report = _export_sample_times(curve, motion, spatial_step=.01, anchors=[0., 1.],
        budget=PlanningBudget(max_validation_points=4096), owner_speed_per_progress=1.)
    assert len(times) >= 1001
    assert report["maximum_interval_relative_displacement_bound_m"] <= .01
    with patch("manipulation.planning.np.linspace", side_effect=AssertionError("must reject before grid allocation")):
        with pytest.raises(ValueError, match="finite sample budget"):
            _export_sample_times(curve, motion, spatial_step=.0001, anchors=[0., 1.],
                budget=PlanningBudget(max_validation_points=4096), owner_speed_per_progress=1.)


def test_export_geometry_chunks_preserve_full_process_history_and_actual_mode_records():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                     np.zeros((641, 3)), np.zeros((641, 3)), .025)
    scene = planner._verification_scene
    scene.required_progress_anchors = (.5,)
    scene.robot_owner_permissions = ()
    planner.anchors = np.array([0., .5, 1.])
    horizon = []
    fk = planner._verification_fk.compute_kinematics
    planner._verification_fk.compute_kinematics = lambda state: (horizon.append(state.position.shape[1]) or fk(state))
    coverage_calls = []
    scene.check_robot_coverage = lambda q, names, t, **kw: (coverage_calls.append((len(t[0]), kw.get("entrance", False))) or [])
    def evaluate(spheres, p, r, progress, q, names, *, layer_selection_out, **kwargs):
        layer_selection_out["robot/background"] = {"body_ids": ["body"],
            "body_fine": encode_array((progress > .5).unsqueeze(-1).detach().cpu())}
        return torch.zeros_like(progress)
    scene.evaluate = evaluate
    original = problem.residuals
    problem.residuals = Mock(side_effect=original)
    curve, count = planner.export(result)
    assert count == 1281 and max(horizon) <= 256 and len(horizon) > 1
    assert problem.residuals.call_count == 1
    assert problem.residuals.call_args.args[0].shape[1] == count
    assert sum(n for n, _ in coverage_calls) == count
    assert sum(flag for _, flag in coverage_calls) == 1
    records = planner.last_validation["scene_layer_queries"]
    assert sum(row["sample_count"] for row in records) == count
    assert all(decode_array(row["time_s"]).shape[1] == 3*row["sample_count"] for row in records)
    assert all(row["query_order"] == ["exact", "left", "right"] for row in records)
    assert all(decode_array(row["layer_selection"]["robot/background"]["body_fine"]).shape[1]
               == decode_array(row["time_s"]).shape[1] for row in records)


def test_release_history_is_frozen_before_every_geometry_chunk():
    problem = cartesian_problem((MotionPhase("free", constant_path()),))
    t = np.linspace(0., 1., 641)
    q = np.column_stack((t, np.zeros((len(t), 2))))
    dq = np.column_stack((np.ones(len(t)), np.zeros((len(t), 2))))
    planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), q, dq, 1/640)
    scene = planner._verification_scene
    relation = Attachment((0., .4), np.zeros(3), np.array([1., 0., 0., 0.]), "holding", "release")
    payload = owner(constant_path(), spheres=((0., 0., 0., .1),))
    payload.attachments = (relation,)
    payload.query = NS(independent_geometry={"solids": []})
    scene.owners = (payload,)
    scene.required_progress_anchors = (0., .4)
    scene.robot_owner_permissions = ()
    scene._fixed_release_poses = {}
    scene._release_pose = PlanningScene._release_pose
    def clone():
        value = copy(scene)
        value._fixed_release_poses = {}
        return value
    scene.clone_for_rollout = clone
    scene.with_release_history = lambda *args: PlanningScene.with_release_history(scene, *args)
    checked = []
    # Bind evaluation to each clone's own release history; other scene state
    # is shared only for this CPU oracle.
    def clone_with_evaluate():
        value = clone()
        def evaluate(spheres, p, r, progress, q, names, **kwargs):
            saved = value._fixed_release_poses[(payload.owner_id, 0)][0]
            checked.append(saved.clone())
            return torch.zeros_like(progress)
        value.evaluate = evaluate
        return value
    scene.clone_for_rollout = clone_with_evaluate
    planner.anchors = np.array([0., .4, 1.])
    planner.export(result)
    assert len(checked) > 1
    assert all(torch.allclose(value[..., 0], torch.tensor([[.4]], dtype=torch.float64)) for value in checked)
    assert scene._fixed_release_poses == {}  # No candidate mutates another's release history.
