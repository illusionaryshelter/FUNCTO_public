"""Native OMPL searches with explicit CPU Cartesian kinematics.

These tests exercise the installed planner, projection, typed relations and
scene release history. Their analytic robot is a test fixture, not a second
production IK backend or evidence about a physical robot's dynamics.
"""

# BEGIN scoped manifolds and true allowed-goal search regressions.
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch

from manipulation.planning import (_ConstrainedPlanner, _ProgressMap, PlanningBudget,
                                    _project_allowed_boundary, _orientation_vjp_type, _ompl_stage)
from manipulation.goals import SupportSurface
from manipulation.motion import MotionModel
from manipulation.planning_problem import (MotionPhase, PlanningProblem,
    CoordinateEventRequirement, MechanismRequirement, PoseEventRequirement, PoseProcessRequirement,
    SupportRequirement, PosePath)
from manipulation.online_geometry import OnlinePlanningScene
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace
from manipulation.robot_geometry_check import NativePrimitive, PairQuery, RobotGeometryChecker


# BEGIN synthetic task observations use the same raw trilinear field interface.
def pose_path(position=(0., 0., 0.)):
    poses = np.repeat(np.eye(4)[None], 2, axis=0)
    poses[:, :3, 3] = position
    return PosePath(np.array([0., 1.]), poses, source={'kind': 'synthetic_test'})


def observed_sphere(name, center=(8., 0., 0.), radius=.1):
    """Sample a small analytic field to test the observation representation.

    Task geometry is supplied as values/support/weights, with no task CAD or
    primitive query. These analytic samples are test inputs, not sensor claims.
    """
    step = radius/4
    origin = np.asarray(center)-5*step
    grid = np.moveaxis(np.indices((11, 11, 11)), 0, -1)*step+origin
    values = np.linalg.norm(grid-np.asarray(center), axis=-1)-radius
    return FrozenObservedField(values, np.ones(values.shape, bool), weights=np.ones(values.shape),
        origin=origin, voxel_size=step, name=name, source='synthetic sampled observation fixture')


def scene(owners=(), *, background=None):
    coverage = ObservedSpace(np.ones((41, 41, 41), bool), origin=[-2.]*3, voxel_size=.1,
        observation_ids=['synthetic_view'], reference_frame='world')
    return SimpleNamespace(owners=owners, background=observed_sphere('background') if background is None else background,
        coverage=coverage, _fixed_release_poses={})


def cartesian_planner(problem, world=None, *, fixed_self_box=None):
    """Native Cartesian robot solids and observed task fields, all on CPU.

    Only the robot receives registered native boxes. Its complete named
    joints and link transforms also drive the production scene's Pinocchio FK.
    """
    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    planner.torch, planner.problem = torch, problem
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.budget = PlanningBudget(search_seconds=3., validation_joint_step=.08)
    planner.joint_names = problem.joint_names
    source = scene() if world is None else world
    dof = len(problem.joint_names)
    links, joints, parent = ['<link name="base"/>'], [], 'base'
    for index, name in enumerate(problem.joint_names):
        child = 'tool' if index == dof-1 else f'axis_{index}'
        links.append(f'<link name="{child}"/>')
        axis = ('1 0 0', '0 1 0', '0 0 1', '0 0 1')[index]
        kind = 'revolute' if index == 3 else 'prismatic'
        joints.append(f'<joint name="{name}" type="{kind}"><parent link="{parent}"/>'
            f'<child link="{child}"/><axis xyz="{axis}"/>'
            '<limit lower="-20" upper="20" effort="10" velocity="20"/></joint>')
        parent = child
    geometry = [NativePrimitive('tool_body', 'tool', 'box', (.02,)*3, np.eye(4), 'declared test robot model')]
    if fixed_self_box is not None:
        local = np.eye(4); local[:3, 3] = fixed_self_box[:3]
        geometry.append(NativePrimitive('base_body', 'base', 'box', (2*fixed_self_box[3],)*3,
            local, 'declared fixed test robot model'))
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'cartesian.urdf'
        path.write_text('<robot name="cartesian">'+''.join(links+joints)+'</robot>')
        checker = RobotGeometryChecker(path, base_frame='base', geometries=geometry)
    planner.scene = OnlinePlanningScene(background=source.background, owners=source.owners,
        coverage=source.coverage, reference_frame='world', robot_model=checker,
        fixed_geometry_names=() if fixed_self_box is None else ('base_body',),
        tool_geometry_names=('tool_body',), has_gripper=False,
        unknown_policy={'outside_regions':'forbid','conditional_regions':[]})
    pairs = () if fixed_self_box is None else (PairQuery('tool_body', 'base_body', 'required', 'test independent robot links'),)
    planner.robot_validator = SimpleNamespace(checker=checker, pairs=pairs, required_pairs=pairs,
        T_reference_base=np.eye(4))
    planner.check_request = lambda: None
    planner.scene.bind_online_robot(validator=planner.robot_validator, check_request=planner.check_request)
    planner.goal_ik_candidates = {}
    # Functional IK construction has its own CPU/native contracts. This fixture
    # injects explicit candidates to isolate production search consumption.
    planner._build_goal_ik_candidates = lambda reference_seeds, check_search: check_search()
    planner.anchors = np.unique(np.r_[problem.anchors, planner.scene.required_progress_anchors])
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.tensor(value, dtype=torch.float64))
    planner.start = SimpleNamespace(position=torch.tensor(problem.start_q[None], dtype=torch.float64))
    planner.position_domain = SimpleNamespace(lower=torch.full((dof,), -2.), upper=torch.full((dof,), 2.))
    planner.formal_clock, planner.finger_trajectory = {'duration': 1.}, None

    def fit_test_initializer(parameters, positions, *, prior_controls):
        # Spatial fitting is isolated from physical spline acceptance here;
        # the production continuous-domain tests own that later boundary.
        output = np.column_stack([np.interp(np.linspace(0., 1., len(prior_controls)),
            parameters, positions[:, joint]) for joint in range(positions.shape[-1])])
        return output, {'model': 'test Cartesian sampled initializer'}

    planner.physical_domain = SimpleNamespace(fit_samples=fit_test_initializer,
        evaluate=lambda controls, parameters: np.column_stack([
            np.interp(parameters, np.linspace(0., 1., len(controls)), controls[:, j])
            for j in range(controls.shape[-1])]))
    planner.JointState, planner._RobotState = SimpleNamespace, SimpleNamespace

    def fk(state):
        q = state.position
        orientation = q.new_zeros((*q.shape[:2], 1, 4)); orientation[..., 0] = 1.
        jacobian = q.new_zeros((*q.shape[:2], 1, 6, dof))
        if dof == 4:
            orientation[..., 0, 0] = torch.cos(q[..., 3]/2)
            orientation[..., 0, 3] = torch.sin(q[..., 3]/2)
            jacobian[..., 0, 5, 3] = 1.
        return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
            position=q[:, :, None, :3], quaternion=orientation),
            robot_spheres=q.new_empty((*q.shape[:2], 0, 4)), tool_jacobians=jacobian)

    planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
    planner._orientation_vjp = _orientation_vjp_type()
    planner._native_layout = SimpleNamespace(apply=lambda value: value)
    limits = SimpleNamespace(joint_names=planner.joint_names,
        velocity=torch.tensor([[-20.]*dof, [20.]*dof], dtype=torch.float64))
    cspace = SimpleNamespace(config=SimpleNamespace(joint_limits=limits))
    planner._verification_native = SimpleNamespace(get_cost=lambda name: cspace if name == 'cspace' else None,
        compute_costs=lambda state: SimpleNamespace(names=(), values=()))
    return planner
# END all task objects retain observed fields and branch-local release histories.


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

    # BEGIN native self costs receive the same named state used for scene FK.
    def test_native_cost_receives_current_named_joint_state(self):
        planner = cartesian_planner(problem())
        received = []
        def costs(state):
            received.append(state.joint_state.position.detach().clone())
            self.assertEqual(tuple(state.joint_state.joint_names), planner.joint_names)
            self.assertEqual(state.cuda_robot_model_state.robot_spheres.shape[-2], 0)
            return SimpleNamespace(names=(), values=())
        planner._verification_native.compute_costs = costs
        seed, report = self.search(planner, lambda t: np.array([.2*t, 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertTrue(received)
        self.assertTrue(any(torch.allclose(row[0, 0], torch.tensor([.2, 0., 0.], dtype=row.dtype)) for row in received))
    # END the original robot scene and native state are both evaluated.

    # BEGIN required connections share one deadline without an artificial chart quota.
    def _search_with_charged_clock(self, cost_per_query, *, attempts=1):
        """Charge unique scene queries while retaining the installed OMPL connector.

        Only the first connection is expensive in this deterministic fixture.
        No real sleep or GPU timing is involved: the clock isolates whether a
        necessary connection receives the request's remaining time.
        """
        bounds = np.tile([-np.inf, np.inf], (6, 1)); bounds[1] = 0.
        relation = PoseProcessRequirement(.3, .7, pose_path((0., .2, 0.)), bounds, "required y plane")
        planner = cartesian_planner(problem(pose_relations=(relation,)))
        planner.budget = replace(planner.budget, ik_seeds=attempts, trajectory_seeds=1)
        clock, calls = [0.], []

        def connector(start, goal, lower, upper, valid, equality, **kwargs):
            calls.append({"allocated_seconds": kwargs["seconds"], "started": clock[0]})
            seen = set()

            def charged_valid(value):
                key = np.asarray(value, dtype=float).tobytes()
                if key not in seen:
                    seen.add(key)
                    clock[0] += cost_per_query if len(calls) == 1 else 0.
                return valid(value)

            return _ompl_stage(start, goal, lower, upper, charged_valid, equality, **kwargs)

        with patch("manipulation.planning.time", SimpleNamespace(monotonic=lambda: clock[0])), \
                patch("manipulation.planning._ompl_stage", side_effect=connector):
            seed, report = self.search(planner, lambda t: np.array([t, 0., 0.]))
        return seed, report, calls, clock[0]

    def test_first_required_connection_can_use_remaining_global_time(self):
        seed, report, calls, elapsed = self._search_with_charged_clock(.2)
        self.assertIsNotNone(seed, report)
        self.assertEqual(len(calls), 3)
        self.assertGreater(elapsed, .75)
        self.assertLess(elapsed, 3.)
        self.assertEqual(report["elapsed_seconds"], elapsed)
        for call, branch in zip(calls, report["branches"]):
            self.assertAlmostEqual(call["allocated_seconds"], 3.-call["started"])
            self.assertEqual(branch["allocated_seconds"], call["allocated_seconds"])
        self.assertGreater(report["branches"][0]["elapsed_seconds"], .75)

    def test_expired_global_deadline_stops_later_charts_and_attempts(self):
        seed, report, calls, elapsed = self._search_with_charged_clock(.8, attempts=2)
        self.assertIsNone(seed)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(report["branches"]), 1)
        self.assertGreaterEqual(elapsed, 3.)
        # A finite state check can cross the deadline; it cannot grant a new
        # connection or candidate its own renewed three-second budget.
        self.assertLessEqual(elapsed, 3.8)
        self.assertEqual(report["elapsed_seconds"], elapsed)
        self.assertEqual(report["branches"][0]["allocated_seconds"], 3.)
        self.assertEqual(report["branches"][0]["elapsed_seconds"], elapsed)
        self.assertEqual(report["branches"][0]["failure"], "whole_search_time_budget_exhausted")
    # END existing state-query and alternative-proposal caps remain independently tested.

    # BEGIN failed alternative IK cannot consume the only qualified boundary attempt.
    def test_functional_boundary_uses_qualified_indices_before_preferring_an_alternative(self):
        good = {"task_satisfied": True, "model_valid": True, "allowed_set_ik_success": True}
        failed = {"task_satisfied": False, "model_valid": False, "allowed_set_ik_success": False}
        cases = (
            ([([.5, 0., 0.], good), ([1.8, 1.8, 1.8], failed)], 0, [.5, 0., 0.]),
            ([([.5, 0., 0.], good), ([1.8, 1.8, 1.8], failed), ([.51, 0., 0.], good)], 0, [.5, 0., 0.]),
            ([([float("nan"), 0., 0.], good), ([1.8, 1.8, 1.8], failed)], None, [.5, 0., 0.]),
        )
        for candidates, index, expected in cases:
            with self.subTest(selected_original_index=index):
                planner = cartesian_planner(problem(pose_events=(pose_goal(1., [.5, 0., 0.]),)))
                planner.budget = replace(planner.budget, ik_seeds=2, trajectory_seeds=1)
                planner.goal_ik_candidates = {1.: {"candidates": [
                    {"joint_position": q, "report": report} for q, report in candidates]}}
                seed, result = self.search(planner, lambda t: np.array([.5*t, 0., 0.]))
                self.assertIsNotNone(seed, result)
                first = result["branches"][0]
                self.assertNotIn("failure", first)
                np.testing.assert_allclose(first["boundary_state"][:3], expected, atol=1e-12)
                if index is None:
                    self.assertEqual(first["proposal_source"], "candidate_curve")
                    self.assertNotIn("functional_ik_candidate_index", first)
                else:
                    self.assertEqual(first["proposal_source"], "ranked_functional_ik")
                    self.assertEqual(first["functional_ik_candidate_index"], index)
                    self.assertEqual(first["functional_ik_report"], good)
    # END native OMPL projection and scene/connection validation remain active.

    # BEGIN registered robot pairs constrain every OMPL goal proposal.
    def test_registered_robot_self_pair_rejects_an_otherwise_free_boundary(self):
        request=problem(pose_events=(pose_goal(1.,[.5,0.,0.],tolerance=.001),))
        planner=cartesian_planner(request,fixed_self_box=(.5,0.,0.,.1))
        planner.budget=replace(planner.budget,ik_seeds=2,trajectory_seeds=1)
        seed,report=self.search(planner,lambda t:np.array([.5*t,0.,0.]))
        self.assertIsNone(seed)
        self.assertEqual(len(report['branches']),2)
        for branch in report['branches']:
            self.assertEqual(branch['failure'],'boundary_candidate_not_feasible')
            self.assertGreater(branch['first_invalid_states'][0]['terms']['scene'],0.)
    # END no separate source checker supplies or overrides this online result.

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
            [0., 0., -1.], support, .001, .01, 0., "declared finite support",
            # These projection tests explicitly exercise all four components;
            # the production task declaration decides its own hard subset.
            hard_components=("gap", "body", "footprint", "normal"))

    def test_support_projection_keeps_tangential_member_and_searches_it(self):
        planner = cartesian_planner(problem(supports=(self.support_goal(),)))
        seed, report = self.search(planner, lambda t: np.array([.4*t, .3*t, .3]))
        self.assertIsNotNone(seed, report)
        boundary = report["branches"][0]["boundary_state"]
        np.testing.assert_allclose(boundary[:2], [.2, .15], atol=1e-12)
        self.assertLessEqual(abs(boundary[2]), .001)

    # BEGIN boundary initialization preserves the independent support requirements.
    def test_support_goal_rows_preserve_the_allowed_set_and_independent_gradients(self):
        support = self.support_goal()
        request = problem(supports=(support,))
        position = torch.tensor([[.2, .1, .0005], [1.2, .3, .3],
                                 [.2, .1, -.01], [.2, .1, .0005]],
                                dtype=torch.float64, requires_grad=True)
        quaternion = torch.tensor([[1., 0., 0., 0.]]*4, dtype=torch.float64)
        quaternion[3] = torch.tensor([np.cos(.05), np.sin(.05), 0., 0.])
        progress = torch.full((4,), support.progress, dtype=torch.float64)
        residual = request.goal_residual(position, quaternion, progress,
                                        goal_ids=(("supports", 0),))
        self.assertEqual(residual.shape, (4, 4))
        # The existing physical metric still determines membership: splitting
        # its nonnegative components may not change any original threshold.
        original = support.evaluate(position, quaternion, request.numerical)
        torch.testing.assert_close(residual.sum(-1), original, rtol=0., atol=0.)
        torch.testing.assert_close((residual <= 0).all(-1),
                                   torch.tensor([True, False, False, False]))
        self.assertGreater(float(residual[2, 1].detach()), 0.)  # body below plane
        self.assertGreater(float(residual[3, 3].detach()), 0.)  # tilted material face
        # Height and finite-region excess are simultaneously active but have
        # independent correction directions; an aggregate scalar loses these.
        gap_gradient = torch.autograd.grad(residual[1, 0], position, retain_graph=True)[0][1]
        footprint_gradient = torch.autograd.grad(residual[1, 2], position)[0][1]
        torch.testing.assert_close(gap_gradient, torch.tensor([0., 0., 1.], dtype=torch.float64))
        torch.testing.assert_close(footprint_gradient, torch.tensor([1., 0., 0.], dtype=torch.float64))

    def test_native_support_projection_corrects_two_components_without_moving_free_tangent(self):
        request = problem()
        # Binary-exact synthetic boundaries isolate the vector projection from
        # rounding at a physical robot's float32 FK boundary. Other tests retain
        # the ordinary tight support gap and the complete search/scene checks.
        support = replace(self.support_goal(), gap_tolerance=.125,
                          margin=request.numerical.position_m)
        request.supports = (support,)

        def residual(values, derivative):
            point = torch.tensor(values, dtype=torch.float64, requires_grad=derivative)
            def evaluate(value):
                return request.goal_residual(value[:3][None], value.new_tensor([[1., 0., 0., 0.]]),
                    value[-1:], goal_ids=(("supports", 0),)).reshape(-1)
            answer = torch.autograd.functional.jacobian(evaluate, point) if derivative else evaluate(point)
            return answer.detach().numpy()

        proposal = np.array([1.25, .375, .5, .5])
        projected = _project_allowed_boundary(proposal, residual, 0, 1e-6,
            np.array([-2., -2., -2., 0.]), np.array([2., 2., 2., 1.]), deadline=time.monotonic()+1.)
        self.assertIsNotNone(projected)
        self.assertTrue(np.all(residual(projected, False) <= 0))
        self.assertEqual(projected[1], proposal[1])
        self.assertEqual(projected[-1], proposal[-1])
        self.assertLessEqual(projected[0], 1.)
        self.assertLessEqual(projected[2], support.gap_tolerance)
    # END actual native projection keeps all unchanged hard membership checks.

    # BEGIN the last admitted Newton update must receive its own strict verdict.
    def test_boundary_projection_checks_update_24_without_performing_update_25(self):
        from ompl import base as ob

        for terminal in (0., .25, float("nan"), "expired"):
            with self.subTest(terminal=terminal):
                calls, clock = [], [0.]

                class ControlledConstraint:
                    # A controlled update isolates iteration accounting from
                    # Newton convergence speed; native projection is exercised
                    # by the neighboring independent-support regression.
                    def __init__(self, dimension, rank, tolerance):
                        pass

                    def setMaxIterations(self, count):
                        self.asserted_count = count

                    def project(self, point):
                        self_test.assertEqual(self.asserted_count, 1)
                        calls.append(point.copy())
                        # Reflection of these deliberately incomplete updates
                        # remains infeasible until the final controlled step.
                        point[0] -= .25
                        if len(calls) == 24:
                            point[0] = 0. if terminal == "expired" else terminal
                            if terminal == "expired":
                                clock[0] = 2.
                        return False

                def residual(point, derivative):
                    return np.array([[1., 0.]]) if derivative else np.array([abs(point[0])])

                self_test = self
                with patch.object(ob, "Constraint", ControlledConstraint), \
                     patch("manipulation.planning.time.monotonic", side_effect=lambda: clock[0]):
                    projected = _project_allowed_boundary(np.array([24., .5]), residual, 0, 1e-6,
                        np.array([-30., 0.]), np.array([30., 1.]), deadline=1.)
                self.assertEqual(len(calls), 24)
                if terminal == 0.:
                    np.testing.assert_array_equal(projected, [0., .5])
                else:
                    self.assertIsNone(projected)
    # END no positive, nonfinite or expired final state gains acceptance.

    # BEGIN reflected native steps address quantized evaluations without relaxed gates.
    def test_native_projection_crosses_float32_plateau_with_original_threshold(self):
        # The strict bound lies just below a representable float32 value.
        # Repeated tiny Newton corrections remain in the same quantization bin
        # for all 24 updates; reflection of the initial step enters the set.
        spacing = 1.-float(np.nextafter(np.float32(1.), np.float32(0.)))
        bound = 1.-spacing/128.
        queried = []

        def residual(point, derivative):
            violation = max(float(np.float32(point[0]))-bound, 0.)
            if derivative:
                return np.array([[1. if violation > 0 else 0., 0., 0.]])
            queried.append(violation)
            return np.array([violation])

        proposal = np.array([1.+2.**-19, .375, .5])
        projected = _project_allowed_boundary(proposal, residual, 0, 1e-6,
            np.array([0., -2., 0.]), np.array([2., 2., 1.]), deadline=time.monotonic()+1.)
        self.assertIsNotNone(projected)
        self.assertTrue(any(0. < value < spacing for value in queried))
        self.assertLessEqual(float(np.float32(projected[0])), bound)
        np.testing.assert_array_equal(projected[1:], proposal[1:])

    def test_reflection_cannot_accept_a_new_violation_of_another_goal(self):
        spacing = 1.-float(np.nextafter(np.float32(1.), np.float32(0.)))
        upper = 1.-spacing/128.
        # The narrow allowed interval contains the preceding float32 value,
        # but the reflected initial update jumps below its lower boundary.
        lower = 1.-2*spacing
        saw_reflected_violation = []

        def residual(point, derivative):
            value = float(np.float32(point[0]))
            high, low = max(value-upper, 0.), max(lower-value, 0.)
            if derivative:
                return np.array([[1. if high > 0 else 0., 0., 0.],
                                 [-1. if low > 0 else 0., 0., 0.]])
            if high == 0. and low > 0.:
                saw_reflected_violation.append(point.copy())
            return np.array([high, low])

        proposal = np.array([1.+2.**-19, .375, .5])
        projected = _project_allowed_boundary(proposal, residual, 0, 1e-6,
            np.array([0., -2., 0.]), np.array([2., 2., 1.]), deadline=time.monotonic()+1.)
        self.assertTrue(saw_reflected_violation)
        for point in saw_reflected_violation:
            np.testing.assert_array_equal(point[1:], proposal[1:])
        # A subsequent valid candidate is allowed, but a reflected overshoot
        # must never be accepted on the basis of the original active row alone.
        if projected is not None:
            self.assertTrue(np.all(residual(projected, False) <= 0))
            np.testing.assert_array_equal(projected[1:], proposal[1:])
    # END every component, free coordinate and fixed time survives the extra trial.

    # BEGIN a small boundary budget must reach an independently sampled posture.
    def test_two_attempts_sample_a_new_posture_without_bypassing_projection_or_collision(self):
        for block_random in (False, True):
            with self.subTest(block_random=block_random):
                # The first obstacle blocks only the curve endpoint. The
                # larger alternative also blocks the projected random goal,
                # while leaving the measured start free in both cases.
                center, radius = ((.565, -.5, 0.), .57) if block_random else ((.4, 0., 0.), .08)
                world = scene(background=observed_sphere("background", center=center, radius=radius))
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
        planner.budget = replace(planner.budget, max_validation_points=32,max_task_anchors=32)
        seed, report = self.search(planner, lambda t: np.array([t, 0., 0.]))
        self.assertLessEqual(report["state_queries"], 32)
        self.assertIsNone(seed)
        self.assertTrue(report["budget_exhausted"])

    def test_postrelease_queries_use_selected_prefix_and_keep_original_scene_unbound(self):
        attachment = Attachment((0., .4), [.3, 0., 0.], [1., 0., 0., 0.],
                                "current holding", "actual release event")
        owner = CollisionOwner("payload", observed_sphere("payload", center=(0., 0., 0.), radius=.02),
            pose_path((.3, 0., 0.)), "observed payload", True,
            attachments=(attachment,))
        world = scene((owner,))
        planner = cartesian_planner(problem(pose_events=(pose_goal(.4, [.4, 0., 0.]),)), world)
        seed, report = self.search(planner, lambda t: np.array([min(t, .8-t), 0., 0.]))
        self.assertIsNotNone(seed, report)
        self.assertEqual(report["chart_times"], [0., .4, 1.])
        self.assertFalse(world._fixed_release_poses)
        self.assertFalse(any("unsupported/release_anchor" in str(row.get("first_invalid_states", ""))
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
        planner.position_domain.lower[-1] = 0.
        planner.position_domain.upper[-1] = 4*np.pi+.1
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

# BEGIN task-valid functional proposals receive the selected prefix's scene verdict.
def test_later_functional_proposal_uses_current_release_history_and_keeps_collision_checks():
    attachment = Attachment((0., .4), [.3, 0., 0.], [1., 0., 0., 0.],
        'declared rigid relationship', 'declared release event')
    owner = CollisionOwner('payload', observed_sphere('payload', center=(0., 0., 0.), radius=.02),
        pose_path((.3, 0., 0.)), 'observed task field', attachments=(attachment,))
    for later_x, succeeds in ((.2, True), (.7, False)):
        planner = cartesian_planner(problem(pose_events=(
            pose_goal(.4, [.4, 0., 0.], tolerance=.001),
            pose_goal(.8, [later_x, 0., 0.], tolerance=.001))), scene((owner,)))
        planner.budget = replace(planner.budget, ik_seeds=1, trajectory_seeds=1)
        previous_evidence = {'task_satisfied': True, 'model_valid': False,
            'scene_components': {'relationship/missing_release/payload/0': {'maximum': 1.}}}
        planner.goal_ik_candidates = {.8: {'candidates': [
            {'joint_position': [later_x, 0., 0.], 'report': previous_evidence}]}}
        seed, report = planner.ompl_seed(torch.zeros((1, 1, 8, 3), dtype=torch.float64),
            SimpleNamespace(duration=1., _curve=lambda t: np.array([min(t, .8-t), 0., 0.])))
        later = next(row for row in report['branches'] if row['chart'] == 1)
        assert later['proposal_source'] == 'ranked_functional_ik'
        assert later['functional_ik_report']['model_valid'] is False
        assert not planner.scene._fixed_release_poses
        assert (seed is not None) is succeeds, report
        if succeeds:
            assert 'failure' not in later
        else:
            assert later['failure'] == 'boundary_candidate_not_feasible'
            assert later['first_invalid_states'][0]['terms']['scene'] > 0
# END a missing old prefix cannot veto a proposal or authorize a current collision.
