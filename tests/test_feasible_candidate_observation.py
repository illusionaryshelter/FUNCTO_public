"""Bounded CPU contracts for seed/block feasibility retention and native hooks."""

# BEGIN complete candidate observations reuse the existing planner test oracles.
from copy import deepcopy
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import test_curobo_solver_components as core_fixtures
import test_manipulation_planning as planning_fixtures
from manipulation.curobo_solver_components import bind_solver_components
from manipulation.physical_joint_domain import JointPositionDomain
from manipulation.planning import PlanningBudget, PlanningWorkExhausted, _CuroboPlanner, _ProgressMap
from manipulation.planning_problem import PoseEventRequirement


class FeasibleObservationTests(unittest.TestCase):
    def fixture(self, flags=(True, False), residuals=(0., .1)):
        contract = planning_fixtures.PlannerRetryContractTests()
        planner, native, curves = contract.scripted_planner(flags, residuals, quality_scores=[2., 1.])
        planner._observe_actions = MethodType(_CuroboPlanner._observe_actions, planner)
        planner._functional_candidate_mask = Mock(return_value=np.array([True]))
        planner._observed_native_result = Mock(return_value=native[0])
        planner._optimizer_reports = Mock(return_value=[{
            "seed_index": 0, "unchanged_from_seed": False, "maximum_control_update": .1, "cores": []}])
        return contract, planner, native, curves

    def observe_then_return(self, planner, native, *, kind="initial_seed", error=None, duplicate=False):
        original = planner.optimize.side_effect

        def optimize(seed):
            original(seed)
            initial = native[0].solution.clone()
            actions, seen = initial[0].clone(), [[]]
            planner._observe_actions(actions, {"kind": kind, "iterations": [0 if kind == "initial_seed" else 4]},
                                     initial, seen)
            if duplicate:
                planner._observe_actions(actions, {"kind": "optimization_block", "iterations": [8]},
                                         initial, seen)
            # Later metrics overwrite this mutable pointer. A returned binding
            # must continue to carry the observation's original validation.
            planner.last_validation = {"measurements": {"candidate_index": 999}}
            if error is not None:
                raise error
            return native[1]

        planner.optimize.side_effect = optimize

    def test_initial_and_block_feasible_curves_survive_lower_scalar_cost_failure(self):
        for kind in ("initial_seed", "optimization_block"):
            with self.subTest(kind=kind):
                contract, planner, native, curves = self.fixture()
                native[0].seed_cost.fill_(100.)
                native[1].seed_cost.fill_(1.)
                self.observe_then_return(planner, native, kind=kind, duplicate=True)
                result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
                self.assertTrue(result.success)
                self.assertIs(result.trajectory, curves[0])
                self.assertEqual(result.execution_binding["validation"]["measurements"]["candidate_index"], 0)
                self.assertEqual(result.diagnostics["returned_candidate"]["candidate_origin"]["kind"], kind)
                self.assertEqual(result.request_ledger["consumed_planning"]["validation_passes"], 1)
                self.assertEqual(result.diagnostics["candidate_observation_work"],
                                 {"export_passes": 1, "export_queries": 17})
                self.assertEqual(planner._observed_native_result.call_count, 1)
                self.assertEqual(len(result.diagnostics["candidate_observations"]), 1)

    def test_pose_screen_cannot_bypass_native_or_independent_geometry(self):
        for flags, residuals in (((False, False), (0., .1)), ((True, False), (.002, .1))):
            with self.subTest(flags=flags):
                contract, planner, native, _ = self.fixture(flags, residuals)
                self.observe_then_return(planner, native)
                result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
                self.assertFalse(result.success)
                self.assertEqual(planner.bound_candidates, [])

    def test_failed_pose_screen_never_runs_source_metrics_or_export(self):
        contract, planner, native, _ = self.fixture()
        planner._functional_candidate_mask.return_value = np.array([False])
        self.observe_then_return(planner, native)
        result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
        self.assertFalse(result.success)
        planner._observed_native_result.assert_not_called()
        planner.export.assert_not_called()

    def test_final_duplicate_reuses_verdict_without_borrowing_overwritten_validation(self):
        contract, planner, native, curves = self.fixture((True, True), (0., 0.))
        native[1].solution.copy_(native[0].solution)
        self.observe_then_return(planner, native)
        result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
        self.assertTrue(result.success)
        self.assertIs(result.trajectory, curves[0])
        self.assertEqual(planner.export.call_count, 1)
        self.assertEqual(len(planner.bound_candidates), 1)
        self.assertEqual(result.execution_binding["validation"]["measurements"]["candidate_index"], 0)
        self.assertTrue(result.diagnostics["attempts"][-1]["previously_considered"])

    def test_rejected_optimized_curve_can_trigger_a_new_map_without_new_time_witnesses(self):
        contract = planning_fixtures.PlannerRetryContractTests()
        planner, native, _ = contract.scripted_planner([False, True], [.1, 0.], quality_scores=[2., 1.])
        planner.refine_background_for_seed = Mock(side_effect=[{"updated": False}, {"updated": True}])
        planner.scene = SimpleNamespace(describe=lambda: {"scope": "refined test scene"})
        result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=2))
        self.assertTrue(result.success)
        self.assertEqual(planner.refine_background_for_seed.call_count, 2)
        torch.testing.assert_close(planner.refine_background_for_seed.call_args.args[0], native[0].solution)
        # BEGIN map changes now share the AL/working-set replacement boundary.
        self.assertEqual(result.diagnostics["attempts"][0]["next_action"], "optimize_updated_constraints")
        self.assertTrue(result.diagnostics["attempts"][0]["subproblem_changes"]["background"])
        # END the same selected curve continues on the rebuilt objective.
        planner.ompl_seed.assert_not_called()

    def test_only_work_exhaustion_preserves_an_observed_binding(self):
        for error, succeeds in ((PlanningWorkExhausted("same request exhausted"), True),
                                (RuntimeError("native API failed"), False)):
            with self.subTest(error=type(error).__name__):
                contract, planner, native, _ = self.fixture()
                self.observe_then_return(planner, native, error=error)
                result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
                self.assertEqual(result.success, succeeds)
                self.assertEqual(result.request_ledger["consumed_planning"]["validation_passes"], 1)
                if not succeeds:
                    self.assertIsNone(result.execution_binding)
                    self.assertEqual(result.diagnostics["error_type"], "RuntimeError")

    def test_query_work_is_charged_before_native_metrics_and_preserves_bound_result(self):
        contract, planner, native, _ = self.fixture()
        original_observe = planner._observe_actions

        def observe(*args, **kwargs):
            original_observe(*args, **kwargs)
            planner._charge_candidate_queries(10**9)

        planner._observe_actions = observe
        self.observe_then_return(planner, native)
        result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
        self.assertTrue(result.success)
        self.assertGreater(result.request_ledger["consumed_planning"]["validation_queries"], 10**9)
        self.assertIn("validation_queries", result.diagnostics["quality_stop_reason"])

    def test_scene_refinement_stops_after_a_candidate_has_been_bound(self):
        contract = planning_fixtures.PlannerRetryContractTests()
        planner, _, _ = contract.scripted_planner([True, True], [0., 0.], quality_scores=[2., 1.])
        planner.refine_background_for_seed = Mock(return_value={"updated": False})
        planner.scene = SimpleNamespace(describe=lambda: {"scope": "refined test scene"})
        result = contract.run_planner(planner, PlanningBudget(maximum_trajectory_solves=2))
        self.assertTrue(result.success)
        planner.refine_background_for_seed.assert_called_once()
        self.assertEqual(result.diagnostics["background_refinements"][-1]["reason"],
                         "preserve_qualified_candidate_scene")

    def test_observed_native_postprocess_uses_full_unsorted_batch(self):
        actions = torch.zeros((2, 8, 3))
        joint = SimpleNamespace(dt=torch.full((2, 1), .2))
        metrics = SimpleNamespace(state=SimpleNamespace(joint_state=joint), actions=actions)
        output = object()
        trajopt = SimpleNamespace(solve_state=SimpleNamespace(batch_size=1, num_seeds=2),
            horizon=49, interpolation_steps=4, joint_names=["a", "b", "c"],
            metrics_rollout=SimpleNamespace(compute_metrics_from_action=Mock(return_value=metrics)),
            _interpolate_and_compute_metrics=Mock(return_value=("metrics", "trajectory", "last")),
            end_source_world_reuse=Mock(),
            _get_result=Mock(return_value=output), _get_best_result=Mock(return_value=output))
        planner = SimpleNamespace(trajopt=trajopt, torch=torch, formal_clock={"valid_count": 385},
            anchors=[0., 1.], refinement_parameters=np.array([.4]), _charge_candidate_queries=Mock())
        result = _CuroboPlanner._observed_native_result(planner, actions)
        self.assertIs(result, output)
        trajopt._get_result.assert_called_once_with(actions, metrics, "metrics", "trajectory", "last", 2, 2)
        trajopt._get_best_result.assert_called_once_with(output, 2)
        trajopt.end_source_world_reuse.assert_called_once_with()
        self.assertIs(joint.knot, actions)
        torch.testing.assert_close(joint.knot_dt, torch.full((2, 1), .8))
        planner._charge_candidate_queries.assert_called_once_with(2*(49+385+6))

    # BEGIN candidate sessions follow actual native calls and all error exits.
    def test_source_reuse_session_begins_per_candidate_and_closes_on_errors(self):
        from manipulation.planning import _bind_candidate_source_world_reuse

        for failure in (None, "ordinary", "dense"):
            with self.subTest(failure=failure):
                events = []
                active = [False]
                dense_scene = object()

                def begin(scene):
                    self.assertIs(scene, dense_scene)
                    events.append("begin")
                    active[0] = True

                def end():
                    events.append("end")
                    active[0] = False

                def evaluate(kind):
                    self.assertTrue(active[0])
                    events.append(kind)
                    if failure == kind:
                        raise RuntimeError(kind)
                    return kind

                ordinary_scene = SimpleNamespace(begin_source_world_reuse=begin,
                                                  end_source_world_reuse=end)
                rollout = SimpleNamespace(metrics_constraint_manager=SimpleNamespace(_scene=ordinary_scene),
                    compute_metrics_from_action=lambda _: evaluate("ordinary"))
                optimizer = SimpleNamespace(metrics_rollout=rollout,
                    additional_metrics_rollouts={"interpolated_rollout": SimpleNamespace(
                        metrics_constraint_manager=SimpleNamespace(_scene=dense_scene))},
                    _interpolate_and_compute_metrics=lambda _: evaluate("dense"))
                cleanup = _bind_candidate_source_world_reuse(optimizer)
                if failure is None:
                    for _ in range(2):
                        self.assertEqual(rollout.compute_metrics_from_action("actions"), "ordinary")
                        self.assertEqual(optimizer._interpolate_and_compute_metrics("joint"), "dense")
                        self.assertFalse(active[0])
                    self.assertEqual(events, ["begin", "ordinary", "dense", "end"]*2)
                else:
                    with self.assertRaisesRegex(RuntimeError, failure):
                        rollout.compute_metrics_from_action("actions")
                        optimizer._interpolate_and_compute_metrics("joint")
                    self.assertFalse(active[0])
                    self.assertEqual(events[-1], "end")
                cleanup()
                self.assertFalse(active[0])
    # END no interpolation failure can leave a candidate report cache active.

    def test_pose_gate_checks_exact_event_parameter_and_original_tolerance(self):
        from manipulation.planning import _ProgressMap

        position = torch.tensor([[[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]],
                                 [[0., 0., 0.], [.502, 0., 0.], [1., 0., 0.]]])
        velocity = torch.zeros_like(position)
        velocity[..., 0] = 1.
        state = SimpleNamespace(joint_state=SimpleNamespace(position=position, velocity=velocity,
                                                            dt=torch.tensor([.5, .5])))

        def kinematics(joint):
            q = joint.position
            return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tcp"],
                position=q[:, :, None], quaternion=q.new_tensor([1., 0., 0., 0.]).expand(*q.shape[:2], 1, 4)))

        goal = SimpleNamespace(progress=.7, position_tolerance_m=.001, rotation_tolerance_rad=.01,
            errors=lambda p, r: ((p-p.new_tensor([.5, 0., 0.])).norm(dim=-1), p.new_zeros(p.shape[0])))
        planner = SimpleNamespace(problem=SimpleNamespace(pose_events=[goal], start_progress=0., tool_link="tcp",
                transform_tool_pose=lambda p, r: (p, r)),
            torch=torch, trajopt=SimpleNamespace(horizon=3,
                metrics_rollout=SimpleNamespace(compute_state_from_action=Mock(return_value=state))),
            progress_map=_ProgressMap(np.array([0., .5, 1.]), np.array([0., .7, 1.])),
            device_cfg=SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float32)),
            _verification_fk=SimpleNamespace(compute_kinematics=kinematics), joint_names=["x", "y", "z"],
            JointState=SimpleNamespace(from_position=lambda q, **kwargs: SimpleNamespace(position=q)),
            _native_layout=SimpleNamespace(apply=lambda value: value), _charge_candidate_queries=Mock())
        mask = _CuroboPlanner._functional_candidate_mask(planner, torch.zeros((2, 8, 3)))
        np.testing.assert_array_equal(mask, [True, False])
        planner._charge_candidate_queries.assert_called_once_with(8)

    # BEGIN a point-goal screen preserves target identity and first-arrival semantics.
    @staticmethod
    def pose_screen_fixture(goals, progress_map):
        """Use exact q(t)=(t,0,0) and Cartesian FK to expose query-time mistakes."""
        position = torch.zeros((1, 5, 3), dtype=torch.float64)
        position[0, :, 0] = torch.linspace(0., 1., 5, dtype=position.dtype)
        velocity = torch.zeros_like(position)
        velocity[..., 0] = 1.
        state = SimpleNamespace(joint_state=SimpleNamespace(position=position, velocity=velocity,
                                                            dt=torch.tensor([.25], dtype=position.dtype)))

        def kinematics(joint):
            q = joint.position
            return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=["tcp"],
                position=q[:, :, None], quaternion=q.new_tensor([1., 0., 0., 0.]).expand(*q.shape[:2], 1, 4)))

        return SimpleNamespace(problem=SimpleNamespace(pose_events=goals, start_progress=0., tool_link="tcp",
                transform_tool_pose=lambda p, r: (p, r)), torch=torch,
            trajopt=SimpleNamespace(horizon=5,
                metrics_rollout=SimpleNamespace(compute_state_from_action=Mock(return_value=state))),
            progress_map=progress_map,
            device_cfg=SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64)),
            _verification_fk=SimpleNamespace(compute_kinematics=Mock(side_effect=kinematics)),
            joint_names=["x", "y", "z"],
            JointState=SimpleNamespace(from_position=lambda q, **kwargs: SimpleNamespace(position=q)),
            _native_layout=SimpleNamespace(apply=lambda value: value), _charge_candidate_queries=Mock())

    @staticmethod
    def point_goal(progress, target_x):
        target = np.eye(4)
        target[0, 3] = target_x
        return PoseEventRequirement(progress, np.eye(4), target, .001, .01, "declared CPU point goal")

    def assert_pose_screen(self, planner, expected, query_times):
        result = _CuroboPlanner._functional_candidate_mask(planner, torch.zeros((1, 8, 3)))
        np.testing.assert_array_equal(result, [expected])
        query = planner._verification_fk.compute_kinematics.call_args.args[0].position
        np.testing.assert_allclose(query[0, :, 0].numpy(), query_times, rtol=0., atol=1e-14)
        # The mocked native state work plus actual FK rows must equal charging.
        planner._charge_candidate_queries.assert_called_once_with(planner.trajopt.horizon+query.shape[1])

    def test_pose_screen_unsorted_goals_keep_their_different_targets(self):
        clock = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        planner = self.pose_screen_fixture([self.point_goal(.75, .75), self.point_goal(.25, .25)], clock)
        self.assert_pose_screen(planner, True, [.75, .25])

    def test_pose_screen_accepts_first_arrival_even_when_plateau_end_leaves_target(self):
        clock = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
        planner = self.pose_screen_fixture([self.point_goal(.5, .25)], clock)
        self.assert_pose_screen(planner, True, [.25])
        # The later pose is outside tolerance but has no second point event.
        self.assertGreater(abs(.75-planner.problem.pose_events[0].target[0, 3]), .001)

    def test_pose_screen_rejects_bad_first_arrival_even_when_plateau_end_matches(self):
        clock = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
        planner = self.pose_screen_fixture([self.point_goal(.5, .75)], clock)
        self.assert_pose_screen(planner, False, [.25])

    def test_pose_screen_shared_progress_keeps_each_distinct_goal_and_counts_its_query(self):
        clock = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        for second_target, expected in ((.5005, True), (.502, False)):
            with self.subTest(second_target=second_target):
                planner = self.pose_screen_fixture(
                    [self.point_goal(.5, .5), self.point_goal(.5, second_target)], clock)
                self.assert_pose_screen(planner, expected, [.5, .5])
    # END pause endpoints and duplicate source progress never replace a point goal.

    def test_solve_clears_verdicts_and_observers_even_when_native_api_fails(self):
        core = SimpleNamespace(candidate_observer=None)
        planner = SimpleNamespace(physical_domain=SimpleNamespace(validate=Mock()),
            trajopt=SimpleNamespace(disable_tool_pose_tracking=Mock(), core=SimpleNamespace(
                optimizers=[SimpleNamespace(_core=core)])),
            _candidate_qualified=Mock(), _observe_actions=Mock(), _goal=Mock(),
            problem=SimpleNamespace(phase_count=1), budget=PlanningBudget(), torch=torch,
            start=object(), _observed_verdicts=[{"stale": True}])

        def solve(**kwargs):
            self.assertEqual(planner._observed_verdicts, [])
            self.assertTrue(callable(core.candidate_observer))
            raise RuntimeError("native solve API")

        planner.trajopt.solve_pose = solve
        with self.assertRaisesRegex(RuntimeError, "native solve API"):
            _CuroboPlanner.optimize(planner, torch.zeros((1, 1, 8, 3)))
        self.assertIsNone(core.candidate_observer)


class NativeObservationHookTests(unittest.TestCase):
    def test_seed_and_real_blocks_are_observed_without_changing_native_result(self):
        factory = core_fixtures.NativeComponentTest()
        domain = JointPositionDomain(torch.tensor([.1, -.8]), torch.tensor([1., .9]))
        solver, optimizer, _ = factory.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        records = []
        core = optimizer._core
        core.candidate_observer = lambda action, origin: records.append((action.clone(), deepcopy(origin)))
        seed = torch.tensor([[[.4, .2]]])
        with patch("curobo._src.runtime.cuda_event_timers", False):
            optimizer.reinitialize(seed)
            observed = optimizer.optimize(seed)
            observed_records = deepcopy(records)
            core.candidate_observer = None
            optimizer.reinitialize(seed)
            ordinary = optimizer.optimize(seed)
        torch.testing.assert_close(observed, ordinary, rtol=0, atol=0)
        self.assertEqual([origin["kind"] for _, origin in observed_records],
                         ["initial_seed", "optimization_block", "optimization_block"])
        self.assertEqual([origin["iterations"] for _, origin in observed_records], [[0], [2], [4]])
        torch.testing.assert_close(observed_records[0][0], seed, rtol=0, atol=0)

    def test_observer_error_is_not_suppressed_by_native_reinitialization(self):
        factory = core_fixtures.NativeComponentTest()
        domain = JointPositionDomain(torch.tensor([.1, -.8]), torch.tensor([1., .9]))
        solver, optimizer, _ = factory.make_solver(domain, 1)
        bind_solver_components(solver, domain)
        optimizer._core.candidate_observer = Mock(side_effect=RuntimeError("candidate metrics API"))
        with self.assertRaisesRegex(RuntimeError, "candidate metrics API"):
            optimizer.reinitialize(torch.tensor([[[.4, .2]]]))


if __name__ == "__main__":
    unittest.main()
# END GPU warmup and source geometry remain outside these small CPU contracts.
