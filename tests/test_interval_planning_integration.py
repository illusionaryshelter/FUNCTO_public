"""Exact candidate queries connect interval evidence to original scene verdicts."""
# BEGIN CPU integration keeps real native/field queries and an analytic robot FK.
from copy import deepcopy
from contextlib import nullcontext
from manipulation.pair_evidence import PairBatch
from pair_evidence_fixtures import records as evidence_records, state_record
from types import SimpleNamespace
from unittest.mock import Mock

import coal
import numpy as np
import pytest
import torch

from manipulation.implicit_distance import ObservedField
from manipulation.native_field_distance import query_native_field
from manipulation.observed_mapping import FrozenObservedField
from manipulation.online_geometry import OnlinePlanningScene
from manipulation.planning import JointTrajectory, PlanningBudget, _ConstrainedPlanner, _ProgressMap, _scene_boundary_evaluate
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner
from test_interval_geometry import ConstantMotion


def candidate():
    return JointTrajectory(("q",), np.array([0., .025]), np.array([[0.], [.025]]),
        np.ones((2, 1)), np.array([0., 1.]), np.array([0., 1.]), (), (), {})


def planner(monkeypatch, *, permission="none", residual=0., release=False):
    curve = candidate()
    instance = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    instance.torch = torch
    instance.joint_names = ("q",)
    instance.check_request = Mock()
    instance.anchors = np.array([.5]) if release else np.array([])
    instance.budget = PlanningBudget(max_validation_points=100, max_task_anchors=32,
        validation_spatial_step_m=.01, interpolation_dt=.025, interpolation_buffer_size=2048)
    instance.robot_metadata = {'active_joint_names': ['q'], 'links': 1}
    instance.problem = SimpleNamespace(tool_link="tool", tool_transform=np.eye(4), start_progress=0.,
        end_progress=1., residuals=lambda *args, **kwargs: {}, verification_issues=lambda progress: [],
        transform_tool_pose=lambda p, q: (p, q), transform_robot_spheres=lambda x: x)
    instance._candidate_domain_validation = lambda value: {}
    # The analytic FK fixture explicitly has no native Pinocchio session.
    instance._candidate_kinematics = lambda scene: nullcontext(None)
    shape = coal.Box(.002, .002, .002)
    field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[.0115, -.001, -.001], voxel_size=.002)
    queries = []
    instance.scene_queries = []

    # BEGIN analytic CPU states preserve the production candidate helper output.
    def robot_state(trajectory, candidate_scene, times, progress, *, fk_session=None):
        states = [torch.as_tensor(trajectory._curve(times, order))[None] for order in range(4)]
        q, dq = states[:2]
        position = torch.cat((q, q.new_zeros((*q.shape[:2], 2))), -1)
        quaternion = q.new_zeros((*q.shape[:2], 4)); quaternion[..., 0] = 1.
        linear = torch.cat((dq, dq.new_zeros((*dq.shape[:2], 2))), -1)
        angular = torch.zeros_like(linear)
        links = {"tool": (position, quaternion)}
        return states, (q, dq, ("q",), links, (position, quaternion), (linear, angular))

    instance._candidate_robot_state = robot_state
    # END interval tests retain their original scene and release computations.

    def evaluate(self, spheres, p, quaternion, progress, joints, names, *, trajectory_time,
                 components_out, geometry_evidence_out, contact_permission_progress=None, **kwargs):
        queries.append(trajectory_time.numpy().copy())
        permissions = progress if contact_permission_progress is None else contact_permission_progress
        values = torch.zeros_like(progress)
        geometry_evidence_out.bind_clock(trajectory_time.numpy(), progress.numpy(), permissions.numpy())
        # BEGIN consume the same production release relation as the real scene.
        relation_costs = {}
        owner_position, _ = self._owner_poses(p, quaternion, progress, None, relation_costs)["material"]
        record = {"times": trajectory_time.numpy().copy(), "progress": progress.numpy().copy(),
                  "owner_position": owner_position.numpy().copy(),
                  "fixed_release_keys": tuple(self._fixed_release_poses), "pairs": []}
        # END a cached release affects modeled owner poses, not just call counts.
        for index in np.ndindex(progress.shape):
            matrix = np.eye(4); matrix[0, 3] = float(p[index][0])
            evidence = query_native_field(shape, matrix, field, np.eye(4))
            if permission != "none":
                evidence = {"status": "clear", "lower_m": 100.}
            values[index] = float(evidence["status"] == "intersection")
            row = {"state_index": list(index), "time_s": float(trajectory_time[index]),
                "progress": float(progress[index]), "permission_progress": float(permissions[index]),
                "first": "body", "second": "background", "component": "collision/body/background",
                "geometry_entity_key": {"first": "robot/body", "second": "background"},
                "permission_kind": permission, **evidence}
            geometry_evidence_out.append_observed(index, row["component"], row["first"], row["second"],
                evidence, geometry_entity_key=row["geometry_entity_key"], permission_kind=permission)
            record["pairs"].append(deepcopy(row))
        geometry_evidence_out.seal()
        components_out["collision/body/background"] = values
        components_out["gripper/controller"] = torch.full_like(progress, residual)
        components_out.update(relation_costs)
        instance.scene_queries.append(record)
        return sum(components_out.values())

    # BEGIN the fixture exercises candidate-private production release storage.
    class Scene:
        clone_for_rollout = OnlinePlanningScene.clone_for_rollout
        with_release_history = OnlinePlanningScene.with_release_history
        _release_pose = staticmethod(OnlinePlanningScene._release_pose)
        _owner_poses = OnlinePlanningScene._owner_poses

    Scene.evaluate = evaluate
    relations = (Attachment((0., .5), [.003, 0., 0.], [1., 0., 0., 0.],
                            "declared_attachment", "declared_release"),) if release else ()
    material = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.001] * 3, voxel_size=.002,
        name="material", source="synthetic RGB-D field")
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared reference")
    scene = Scene()
    scene.owners = (CollisionOwner("material", material, path, "observed material",
                                   moving=release, attachments=relations),)
    scene.contacts, scene.required_progress_anchors = (), (.5,) if release else ()
    scene._fixed_release_poses, scene.last_query_report, scene.gripper = {}, {}, None
    # The analytic fixture declares its tiny query buffers explicitly; the
    # production evidence/Pin allocation estimator still runs unchanged.
    scene.robot_model = SimpleNamespace(geometry_names=('body',), joint_names=('q',))
    scene.estimated_workspace_allocation = lambda *args: dict(cpu_resident=0, cpu_pending_shared=0,
        cpu_live=0, gpu_shared=0, gpu_live=0, gpu_retained=0)
    # END the original scene and every selected candidate own distinct dictionaries.
    instance.scene = scene
    monkeypatch.setattr("manipulation.interval_geometry.SceneMotionBounds.from_scene",
        lambda *args, **kwargs: ConstantMotion(boundary=.0125 if release else None))
    report = {"accepted": True, "native_success": True, "residuals": {"interpolated:process_scene": 0.},
        "collision_residuals": {}, "scene_update_residuals": {}, "coverage_residuals": {},
        "conditional_unknown_residuals": {},
        # This integration fixture isolates interval evidence from scoring.
        "fixed_objective": 0., "objective_components": {"fixture_objective": 0.}}
    # BEGIN fixed queries feed their complete original evidence to interval checks.
    instance._candidate_objective = Mock(return_value={"fixed_objective": 0.})
    # END both stages now exercise the same production query and release context.
    return instance, curve, report, queries


def test_interval_collision_rejects_sampled_candidate_and_keeps_exact_failure(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch)
    returned, result, points = instance.evaluate_candidate(curve, record_work=Mock())
    assert returned is curve and not result["accepted"]
    assert result["residuals"]["interval:process_scene"] == 1.
    assert result["collision_residuals"]["interval:scene_collision"] == 1.
    assert result["continuous_geometry"]["confirmed_intersections"][0]["time_s"] == .0125
    assert points == 3 and len(queries) == 2
    assert instance.last_validation is None


def test_unresolved_finite_permission_preserves_warning_and_original_hard_result(monkeypatch):
    instance, curve, report, _ = planner(monkeypatch, permission="finite_region")
    _, result, _ = instance.evaluate_candidate(curve, record_work=Mock())
    assert result["accepted"]
    geometry = result["continuous_geometry"]
    assert geometry["unresolved_intervals"] and not geometry["proved_intervals"]
    assert not geometry["confirmed_intersections"]
    assert instance.export(None, result) == (curve, 0)


def test_interval_sampling_keeps_controller_residual_separate_from_intersection(monkeypatch):
    instance, curve, report, _ = planner(monkeypatch, permission="finite_region", residual=.2)
    _, result, _ = instance.evaluate_candidate(curve, record_work=Mock())
    assert not result["accepted"] and result["residuals"]["process_scene"] == .2
    assert result["collision_residuals"]["collision"] == 0.
    assert result["scene_components"]["gripper/controller"]["maximum"] == .2
    assert "continuous_geometry" not in result


def test_release_query_keeps_original_anchor_and_physical_sides_in_one_call(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch, release=True)
    _, result, _ = instance.evaluate_candidate(curve, record_work=Mock())
    assert len(queries) == 1 and np.count_nonzero(queries[0] == .0125) == 3
    assert not result["accepted"] and "continuous_geometry" not in result
    assert result["residuals"]["process_scene"] == 1.
    assert instance.scene_queries[0]["times"].shape == (1, 5)


# BEGIN one immutable candidate reuses its first complete release evidence.
def test_release_cache_charges_each_new_time_once_and_preserves_full_scene_result(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch, permission="finite_region", release=True)
    _, result, points = instance.evaluate_candidate(curve, record_work=Mock())
    saved = deepcopy(instance.scene_queries)
    assert result["accepted"] and points == 5
    assert len(queries) == 2 and np.count_nonzero(queries[0] == .0125) == 3
    assert all(not np.any(times == .0125) for times in queries[1:])
    assert sum(len(np.unique(times)) for times in queries) == points
    assert saved[0]["fixed_release_keys"] == ()
    assert all(row["fixed_release_keys"] == (("material", 0),) for row in saved[1:])
    assert not instance.scene._fixed_release_poses

    # One original-scene query over the entire sampled set is an independent
    # uncached reference for owner positions, pair evidence and hard residuals.
    times = np.unique(np.concatenate([value.ravel() for value in queries]))
    _, _, _, progress = curve.sample(times)
    states, (_, _, _, _, (position, quaternion), _) = instance._candidate_robot_state(
        curve, instance.scene, times, progress)
    evidence = PairBatch()
    value = _scene_boundary_evaluate(instance.scene, instance.problem, _ProgressMap(curve.progress_u, curve.progress_s),
        position.new_empty((1, len(times), 0, 4)), position, quaternion,
        torch.as_tensor(progress)[None], states[0], ("q",), trajectory_time=torch.as_tensor(times)[None],
        normalized_time=torch.as_tensor(times / curve.duration)[None], geometry_evidence_out=evidence,
        component_diagnostics={})
    assert float(value.max()) == result["continuous_geometry"]["hard_scene_residual"] == 0.

    def positions(records):
        return {(float(row["times"][index]), float(row["progress"][index])):
                row["owner_position"][index].tolist()
                for row in records for index in np.ndindex(row["times"].shape)}

    def pair_records(records):
        return {(row["time_s"], row["progress"], row["permission_progress"]):
                {key: value for key, value in row.items() if key != "state_index"}
                for record in records for row in record["pairs"]}

    assert positions(saved) == positions([instance.scene_queries[-1]])
    assert pair_records(saved) == pair_records([instance.scene_queries[-1]])
    later = [point for (time, progress), point in positions(saved).items() if time > .0125]
    np.testing.assert_allclose(np.asarray(later)[:, 0], .0155, rtol=0, atol=1e-15)
    assert not result["continuous_geometry"]["confirmed_intersections"]
    assert result["continuous_geometry"]["unresolved_intervals"]
# END unresolved permissions retain their existing warning and hard result.


def test_rejected_task_or_joint_domain_does_not_spend_interval_geometry_work(monkeypatch):
    instance, curve, report, queries = planner(monkeypatch)
    instance._candidate_domain_validation = lambda value: value.residuals.update(continuous_position=.1) or {}
    _, result, points = instance.evaluate_candidate(curve, record_work=Mock())
    assert not result["accepted"] and points == 2 and len(queries) == 1
    assert "continuous_geometry" not in result
# END only actual scene violations reject; interval proof gaps retain their own meaning.


# BEGIN fixed evidence and new batches share one exact candidate clock.
def test_fixed_queries_supply_complete_states_and_release_without_duplicate_work(monkeypatch):
    instance, curve, _, queries = planner(monkeypatch, permission='finite_region', release=True)
    record_work = Mock()
    _, result, points = instance.evaluate_candidate(curve, record_work=record_work)
    assert result['continuous_geometry']['initial_query_points'] == 3
    assert result['continuous_geometry']['query_points'] == 2
    assert result['continuous_geometry']['query_batch_sizes'] == [2]
    assert points == 5
    assert [call.kwargs['points'] for call in record_work.call_args_list if 'points' in call.kwargs] == [3, 2]
    assert len(queries) == 2
    assert instance.scene_queries[1]['fixed_release_keys'] == (('material', 0),)


def test_event_ulp_identity_matches_interval_partition_and_boundary_sides(monkeypatch):
    from manipulation.interval_geometry import _progress_times
    u = np.array([0., .038557137974987485, .2702740738263834, .998034361190033, 1.])
    s = np.array([0., .6608281629629073, .6925010468048584, .7285616397930895, 1.])
    anchor, duration = .08538524674570802, 10.4040002823
    mapping = _ProgressMap(u, s)
    parameter = float(mapping.anchor_parameters([anchor])[0, 0])
    physical_time = parameter*duration
    assert float(np.interp(physical_time/duration, u, s)) != anchor
    curve = SimpleNamespace(progress_u=u, progress_s=s, duration=duration)
    assert _progress_times(curve, [anchor]) == [physical_time]
    parameters, progress = mapping.query_parameters([physical_time], duration, [anchor])
    assert parameters.tolist() == [parameter] and progress.tolist() == [anchor]

    class Scene:
        required_progress_anchors, contacts = (anchor,), ()
        def evaluate(self, spheres, position, quaternion, progress, joints, names, **kwargs):
            self.progress = progress.numpy().ravel().tolist()
            batch = kwargs['geometry_evidence_out']
            batch.bind_clock(kwargs['trajectory_time'].numpy(), progress.numpy(), progress.numpy())
            for index in np.ndindex(progress.shape):
                batch.append_observed(index, 'collision/fixture', 'body', 'background',
                    {'status': 'clear'}, geometry_entity_key={'first': 'robot/body', 'second': 'background'})
            batch.seal()
            return torch.zeros_like(progress)

    scene, evidence = Scene(), PairBatch()
    tensor = lambda value: torch.as_tensor(value, dtype=torch.float64)[None]
    _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.), mapping,
        torch.empty((1, 1, 0, 4)), torch.zeros((1, 1, 3)), torch.tensor([[[1., 0., 0., 0.]]]),
        tensor(progress), torch.zeros((1, 1, 1)), ('q',), trajectory_time=tensor([physical_time]),
        normalized_time=tensor(parameters), geometry_evidence_out=evidence)
    assert [row['side'] for row in evidence_records(evidence)] == [0, -1, 1]
    assert scene.progress == [anchor, np.nextafter(anchor, -np.inf), np.nextafter(anchor, np.inf)]


def test_actual_candidate_array_and_split_queries_have_identical_cached_release_results(monkeypatch):
    import manipulation.interval_geometry as interval

    def invoke(groups):
        instance, curve, _, _ = planner(monkeypatch, permission='finite_region', release=True)
        _, _, points, (scene, initial) = instance._candidate_metrics(curve, instance.scene, record_work=Mock())
        responses = []
        def capture(curve, motion, query, **kwargs):
            assert kwargs['initial_states'] == initial
            responses.extend(query(group, point_budget=100) for group in groups)
            return {'query_points': sum(row['query_points'] for row in responses)}
        monkeypatch.setattr(interval, 'check_interval_geometry', capture)
        instance._candidate_geometry(curve, scene, initial_states=initial, remaining_points=100-points,
                                     record_work=Mock())
        return {(state.time_s, state.side): state_record(state)
                for response in responses for state in response['states']}

    assert invoke([[.00625, .01875]]) == invoke([[.00625], [.01875]])
# END actual scene evidence, release poses and physical counts agree for both layouts.


def test_each_candidate_recomputes_its_own_release_and_complete_evidence(monkeypatch):
    instance, first, _, queries = planner(monkeypatch, permission='finite_region', release=True)
    _, result, _ = instance.evaluate_candidate(first, record_work=Mock())
    first_records = deepcopy(instance.scene_queries)
    assert result['accepted']
    second = JointTrajectory(first.joint_names, first.times.copy(), first.position+.1,
        first.velocity.copy(), first.progress_u.copy(), first.progress_s.copy(), (), (), {})
    _, result, _ = instance.evaluate_candidate(second, record_work=Mock())
    second_records = instance.scene_queries[len(first_records):]
    assert result['accepted'] and len(first_records) == len(second_records) == 2
    assert second_records[0]['fixed_release_keys'] == ()
    assert second_records[1]['fixed_release_keys'] == (('material', 0),)
    np.testing.assert_allclose(second_records[1]['owner_position'][second_records[1]['progress'] > .5, 0], .1155, atol=1e-15)
    np.testing.assert_allclose(first_records[1]['owner_position'][first_records[1]['progress'] > .5, 0], .0155, atol=1e-15)
    assert not instance.scene._fixed_release_poses


def test_actual_dense_progress_clock_is_admitted_before_fixed_query(monkeypatch):
    instance, curve, _, _ = planner(monkeypatch, permission='finite_region')
    u = np.linspace(0., 1., 1001)
    curve = JointTrajectory(curve.joint_names, curve.times, curve.position, curve.velocity,
        u, u.copy(), (), (), {})
    from manipulation import planning
    actual = planning._planning_workspace_bytes
    clocks = []
    def workspace(*args, **kwargs):
        clocks.append(kwargs['clock_nodes'])
        return actual(*args, **kwargs)
    monkeypatch.setattr(planning, '_planning_workspace_bytes', workspace)
    instance._candidate_metrics(curve, instance.scene, record_work=Mock())
    assert clocks == [1001]
