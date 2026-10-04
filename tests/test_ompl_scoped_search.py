"""Native OMPL searches with explicit CPU Cartesian kinematics.

These tests exercise the installed planner, projection, typed relations and
scene release history. Their analytic robot is a test fixture, not a second
production IK backend or evidence about a physical robot's dynamics.
"""

# BEGIN scoped manifolds and true allowed-goal search regressions.
from types import SimpleNamespace
from dataclasses import replace
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.planning import (_CuroboPlanner, _ProgressMap, PlanningBudget,
                                    _project_allowed_boundary, _orientation_vjp_type, _ompl_stage)
from manipulation.goals import SupportSurface
from manipulation.motion import MotionModel
from manipulation.planning_problem import (MotionPhase, PlanningProblem,
    CoordinateEventRequirement, MechanismRequirement, PoseEventRequirement, PoseProcessRequirement,
    SupportRequirement)
from tests.test_manipulation_planning_scene import (Attachment, CollisionOwner,
                                                    SphereQuery, pose_path, scene)


def cartesian_planner(problem, world=None):
    """Three named translation joints with a fixed tool orientation."""
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.torch, planner.problem = torch, problem
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.budget = PlanningBudget(search_seconds=3., validation_joint_step=.08)
    planner.joint_names = problem.joint_names
    planner.scene = scene() if world is None else world
    planner.anchors = np.unique(np.r_[problem.anchors, planner.scene.required_progress_anchors])
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.tensor(value, dtype=torch.float64))
    planner.start = SimpleNamespace(position=torch.tensor(problem.start_q[None], dtype=torch.float64))
    dof = len(problem.joint_names)
    planner.trajopt = SimpleNamespace(action_horizon=8, transition_model=SimpleNamespace(
        action_bound_lows=torch.full((dof,), -2.), action_bound_highs=torch.full((dof,), 2.)))
    planner.formal_clock, planner.finger_trajectory = {"duration": 1.}, None
    planner.physical_domain = SimpleNamespace(initialize=lambda values: values)
    planner.JointState, planner._RobotState = SimpleNamespace, SimpleNamespace

    def fk(state):
        q = state.position
        orientation = q.new_zeros((*q.shape[:2], 1, 4)); orientation[..., 0] = 1.
        jacobian = q.new_zeros((*q.shape[:2], 1, 6, dof))
        if dof == 4:
            orientation[..., 0, 0] = torch.cos(q[..., 3]/2)
            orientation[..., 0, 3] = torch.sin(q[..., 3]/2)
            jacobian[..., 0, 5, 3] = 1.
        # A small attached test proxy, with a known finite collision model.
        spheres = torch.cat((q[..., :3], q.new_full((*q.shape[:2], 1), .01)), -1)[:, :, None]
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
            position=q[:, :, None, :3], quaternion=orientation), robot_spheres=spheres,
            tool_jacobians=jacobian)
    planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
    planner._orientation_vjp = _orientation_vjp_type()
    planner._native_layout = SimpleNamespace(apply=lambda value: value)
    planner._verification_native = SimpleNamespace(get_cost=lambda name: None,
        compute_costs=lambda state: SimpleNamespace(names=(), values=()))
    return planner


def problem(**requirements):
    return PlanningProblem(joint_names=("x", "y", "z"), start_q=np.zeros(3), start_dq=np.zeros(3),
        tool_link="tool", reference_frame="world", base_pose=np.eye(4), tool_transform=np.eye(4),
        phases=(MotionPhase("ordinary reference", pose_path()),), **requirements)


def pose_goal(progress, position, tolerance=.02):
    target = np.eye(4); target[:3, 3] = position
    return PoseEventRequirement(progress, np.eye(4), target, tolerance, .01, "actual functional goal")


class ScopedOmplSearchTest(unittest.TestCase):
    def search(self, planner, curve):
        return planner.ompl_seed(torch.zeros((1, 1, 8, len(planner.joint_names)), dtype=torch.float64),
                                SimpleNamespace(duration=1., _curve=curve))

    def test_native_goal_projection_preserves_free_direction_and_fixed_time(self):
        # Two bands, one free tangent, one duplicate equation. A projected
        # member stays near its proposal; no goal-center pose is substituted.
        def residual(values, derivative):
            x, y, z, _ = values
            if derivative:
                return np.array([[0., 1., 0., 0.], [0., 2., 0., 0.],
                    [1. if x > .6 else -1. if x < .4 else 0., 0., 0., 0.]])
            return np.array([y, 2*y, max(.4-x, 0., x-.6)])
        projected = _project_allowed_boundary(np.array([0., .3, .73, .45]), residual, 2, 1e-6,
            np.array([-2., -2., -2., 0.]), np.array([2., 2., 2., 1.]), deadline=time.monotonic()+1.)
        self.assertIsNotNone(projected)
        self.assertEqual(projected[2], .73)
        self.assertEqual(projected[3], .45)
        self.assertGreaterEqual(projected[0], .4)
        self.assertLess(projected[0], .5)
        np.testing.assert_allclose(projected[1], 0., atol=1e-12)

    def test_real_scope_uses_local_projected_chart_between_free_intervals(self):
        bounds = np.tile([-np.inf, np.inf], (6, 1)); bounds[1] = 0.
        relation = PoseProcessRequirement(.3, .7, pose_path((0., .2, 0.)), bounds, "required y plane")
        planner = cartesian_planner(problem(pose_relations=(relation,)))
        seed, report = self.search(planner, lambda t: np.array([t, 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., .3, .7, 1.])
        successful = [row for row in report["branches"] if "failure" not in row]
        self.assertEqual([row["space"] for row in successful], ["ordinary", "projected", "ordinary"])
        self.assertEqual(successful[1]["equalities"], 1)
        self.assertAlmostEqual(successful[0]["boundary_state"][1], .2)
        self.assertAlmostEqual(successful[1]["boundary_state"][1], .2)

    def test_midpoint_and_terminal_pose_use_actual_allowed_sets(self):
        goals = (pose_goal(.4, [.5, .3, 0.]), pose_goal(1., [1., 0., 0.]))
        planner = cartesian_planner(problem(pose_events=goals))
        seed, report = self.search(planner, lambda t: np.array([.2*t, 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., .4, 1.])
        rows = [row for row in report["branches"] if "failure" not in row]
        for row, goal in zip(rows, goals):
            witness = np.asarray(row["boundary_state"][:3])
            self.assertLessEqual(np.linalg.norm(witness-goal.target[:3, 3]), goal.position_tolerance_m)
            # A band member remains distinct from the nominal center.
            self.assertGreater(np.linalg.norm(witness-goal.target[:3, 3]), .005)
        self.assertGreater(seed[0, :, :, 1].max().item(), .15)

    def support_goal(self):
        support = SupportSurface(np.zeros(3), np.eye(3)[:, :2], [0., 0., 1.],
                                 [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]])
        return SupportRequirement(.5, np.eye(4), [[0., 0., 0.]], [[0., 0., 0.]],
                                  [0., 0., -1.], support, .001, .01, 0., "true finite support")

    def test_support_projection_keeps_tangential_member_and_searches_it(self):
        planner = cartesian_planner(problem(supports=(self.support_goal(),)))
        seed, report = self.search(planner, lambda t: np.array([.4*t, .3*t, .3]))
        self.assertIsNotNone(seed, report)
        boundary = report["branches"][0]["boundary_state"]
        np.testing.assert_allclose(boundary[:2], [.2, .15], atol=1e-12)
        self.assertLessEqual(abs(boundary[2]), .001)

    # BEGIN a small boundary budget must reach an independently sampled posture.
    def test_two_attempts_sample_a_new_posture_without_bypassing_projection_or_collision(self):
        for block_random in (False, True):
            with self.subTest(block_random=block_random):
                # The first obstacle blocks only the curve endpoint. The
                # larger alternative also blocks the projected random goal,
                # while leaving the measured start free in both cases.
                center, radius = ((.565, -.5, 0.), .57) if block_random else ((.4, 0., 0.), .08)
                world = scene(background=SphereQuery(("background",), center=center, radius=radius))
                support = replace(self.support_goal(), progress=1.)
                planner = cartesian_planner(problem(supports=(support,)), world)
                planner.budget = replace(planner.budget, ik_seeds=2, trajectory_seeds=1)
                sampled = np.random.default_rng(planner.budget.random_seed).uniform(-2., 2., 3)
                self.assertGreater(abs(sampled[2]), support.gap_tolerance)

                seed, report = self.search(planner, lambda t: np.array([.4*t, 0., .3]))
                first, second = report["branches"]
                self.assertEqual([first["proposal_source"], second["proposal_source"]],
                                 ["candidate_curve", "joint_box_random"])
                self.assertEqual([first["attempt"], second["attempt"]], [1, 2])
                self.assertEqual(first["failure"], "boundary_candidate_not_feasible")
                self.assertEqual(first["boundary"], "goal")
                self.assertGreater(first["first_invalid_states"][0]["terms"]["scene"], 0.)
                boundary = np.asarray(second["boundary_state"])
                # Native projection changes the unsupported height and the
                # out-of-polygon y, preserving the unconstrained x and time.
                self.assertAlmostEqual(boundary[0], sampled[0])
                self.assertLessEqual(abs(boundary[1]), 1.+planner.problem.numerical.position_m)
                self.assertLessEqual(abs(boundary[2]), support.gap_tolerance)
                self.assertEqual(boundary[-1], 1.)
                self.assertLessEqual(report["state_queries"], planner.budget.max_validation_points)
                if block_random:
                    self.assertIsNone(seed)
                    self.assertEqual(second["failure"], "boundary_candidate_not_feasible")
                    self.assertEqual(second["boundary"], "goal")
                    self.assertGreater(second["first_invalid_states"][0]["terms"]["scene"], 0.)
                else:
                    self.assertIsNotNone(seed, report)
                    self.assertTrue(report["success"])
                    self.assertGreater(second["motion_checks"], 0)
                    self.assertGreater(second["validity_checks"], 2)
    # END both sampled outcomes use the installed projection and validity path.

    def test_failed_suffix_revisits_a_different_member_of_previous_allowed_set(self):
        planner = cartesian_planner(problem(supports=(self.support_goal(),)))
        def connector(start, goal, *args, **kwargs):
            # Controlled geometric search failure isolates the backtracking
            # contract. Every accepted connection still uses native OMPL.
            if start[-1] == .5 and start[0] < .3:
                return None, {"failure": "test suffix blocked for this boundary member"}
            return _ompl_stage(start, goal, *args, **kwargs)
        with patch("manipulation.planning._ompl_stage", side_effect=connector):
            seed, report = self.search(planner, lambda t: np.array([.4*t, 0., .3]))
        self.assertIsNotNone(seed, report)
        self.assertGreaterEqual(report["branches"][-1]["boundary_state"][0], -.001)
        prefix_members = [row["boundary_state"][0] for row in report["branches"]
                          if row["chart"] == 0 and "boundary_state" in row]
        self.assertLess(prefix_members[0], .3)
        self.assertGreaterEqual(prefix_members[-1], .3)

    def test_ordinary_progress_plateaus_are_queries_without_extra_search_charts(self):
        planner = cartesian_planner(problem())
        planner.progress_map = _ProgressMap(np.array([0., .2, .8, 1.]), np.array([0., 0., 1., 1.]))
        seed, report = self.search(planner, lambda t: np.array([t, 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., 1.])

    def test_goal_on_plateau_is_consumed_once_and_later_motion_remains_free(self):
        planner = cartesian_planner(problem(pose_events=(pose_goal(.5, [.4, 0., 0.]),)))
        planner.progress_map = _ProgressMap(np.array([0., .4, .7, 1.]), np.array([0., .5, .5, 1.]))
        # The once-only goal is reached at .4 s. A later state on the same
        # progress plateau may leave it, and the final point is unrestricted.
        seed, report = self.search(planner, lambda t: np.array([min(t, .8-t), 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., .4, 1.])
        self.assertLess(report["branches"][-1]["boundary_state"][0], 0.)

    def test_shared_query_budget_never_overshoots_on_complete_path_batch(self):
        planner = cartesian_planner(problem())
        planner.budget = replace(planner.budget, max_validation_points=32, max_refinement_points=32)
        seed, report = self.search(planner, lambda t: np.array([t, 0., 0.]))
        self.assertLessEqual(report["state_queries"], 32)
        self.assertIsNone(seed)
        self.assertTrue(report["budget_exhausted"])

    def test_postrelease_queries_use_selected_prefix_and_keep_original_scene_unbound(self):
        attachment = Attachment((0., .4), [.3, 0., 0.], [1., 0., 0., 0.],
                                "current holding", "actual release event")
        owner = CollisionOwner("payload", SphereQuery(("payload",), center=(0., 0., 0.), radius=.02),
            pose_path((.3, 0., 0.)), "observed payload", True,
            local_spheres=[[0., 0., 0., .02]], attachments=(attachment,))
        world = scene((owner,))
        planner = cartesian_planner(problem(pose_events=(pose_goal(.4, [.4, 0., 0.]),)), world)
        seed, report = self.search(planner, lambda t: np.array([min(t, .8-t), 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., .4, 1.])
        self.assertFalse(world._fixed_release_poses)
        self.assertFalse(any("unknown/release_anchor" in str(row.get("first_invalid_states", ""))
                             for row in report["branches"]))

    def test_two_turn_path_survives_local_screening_and_checks_full_lifted_history(self):
        model = MotionModel("revolute", np.zeros((1, 1)), np.eye(4), np.eye(3), np.zeros(3),
                            1e-6, 1e-6, {"source": "current axis"})
        mechanism = MechanismRequirement(0., 1., model, np.eye(4), [[0., 4*np.pi+.1]],
                                         [0.], "observed native coordinate zero")
        goal = CoordinateEventRequirement(1., mechanism, 4*np.pi, 1e-5, "two complete turns")
        request = PlanningProblem(joint_names=("x", "y", "z", "turn"), start_q=np.zeros(4),
            start_dq=np.zeros(4), tool_link="tool", reference_frame="world", base_pose=np.eye(4),
            tool_transform=np.eye(4), phases=(MotionPhase("zero-turn reference", pose_path()),),
            mechanisms=(mechanism,), coordinate_events=(goal,))
        planner = cartesian_planner(request)
        planner.trajopt.transition_model.action_bound_lows[-1] = 0.
        planner.trajopt.transition_model.action_bound_highs[-1] = 4*np.pi+.1
        seed, report = self.search(planner, lambda t: np.array([0., 0., 0., 4*np.pi*t]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["branches"][0]["space"], "projected")
        self.assertGreater(seed[0, 0, -1, -1].item(), 3*np.pi)
        self.assertTrue(report["success"])
        # The identical endpoint pose with no actual turns cannot become a
        # complete search seed merely because every point passes local IK.
        seed, report = self.search(planner, lambda t: np.zeros(4))
        self.assertTrue(any(row.get("failure") == "complete_path_process_or_history_rejected"
                            for row in report["branches"]))


if __name__ == "__main__":
    unittest.main()
# END these CPU searches make no native GPU or robot execution claim.
