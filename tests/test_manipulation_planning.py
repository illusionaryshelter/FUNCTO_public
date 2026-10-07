"""Independent CPU checks for planner calculus and exported-curve semantics.

SciPy supplies the rotation and cubic-curve oracles. The export fixture uses a
declared Cartesian test kinematics, so these tests do not claim native robot
IK, GPU execution, collision geometry or physical grasp certification.
"""

# BEGIN independent calculus, exact-anchor and execution-boundary regressions.
from copy import deepcopy
# BEGIN exact-anchor native self-collision fixtures keep GPU resources isolated.
from contextlib import contextmanager, ExitStack, nullcontext
# END the installed native cost remains active in CPU integration tests.
# BEGIN preserve the complete event payload in the shared test clock.
from dataclasses import asdict
# END the exported occurrence must match its frozen source metadata.
import json
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from scipy.interpolate import BSpline, CubicHermiteSpline
from scipy.spatial.transform import Rotation, Slerp
import torch

from manipulation.planning import (
    PlanningBudget,
    # BEGIN exact executable-polynomial acceleration regressions.
    JointTrajectory,
    cubic_derivative_report,
    # END shared production calculus.
    _CuroboPlanner,
    _ProgressMap,
    _formal_native_clock,
    _fixed_spline_grid,
    _timed_native_state,
    # BEGIN actual-anchor allocation and projection preflight regressions.
    _anchor_capacity,
    _planning_workspace_bytes,
    _PlanningWorkspace,
    _projection_tolerance,
    _require_planning_memory,
    # END actual-anchor allocation and projection preflight regressions.
    _bounded_robot_configuration,
    _hermite_at_parameters,
    _native_gradient_layout_type,
    _orientation_vjp_type,
    # BEGIN native cost-configuration ownership regression.
    _process_manager_type,
    # END native cost-configuration ownership regression.
    _progress_for_cubic_boundaries,
    _verification_stop_preference,
    plan_trajectory,
)
from manipulation.planning_problem import (
    MotionPhase,
    PlanningEvent,
    PlanningProblem,
    PosePath,
)


# BEGIN analytic one-body fixtures declare their separate source-self result.
def separated_source_self(joints, names, times, *, valid=None, diagnostics=None):
    """No self pair exists in the synthetic rigid Cartesian test body."""
    if diagnostics is not None:
        diagnostics.update(checked_states=int(times.numel() if valid is None else valid.sum()),
                           unmodelled_geometry_names=[], worst_state=None)
    return torch.zeros_like(times)
# END real source geometry is covered by the dedicated Coal integration tests.


# BEGIN reference-chain fixtures separate static evidence from real cubic fitting.
def bind_analytic_ik_screening(planner, report_for=None):
    """Supply static point reports without claiming CPU fixtures perform FK.

    Native success describes only the declared nominal query. Static task and
    source checks are independent synthetic facts. A real CPU cubic domain and
    fixed model velocity scales exercise conversion of the selected samples;
    they do not simulate source geometry or native robot IK convergence.
    """
    from manipulation.planning_ik import rank_ik_candidates
    from manipulation.physical_joint_domain import CubicJointDomain

    def rank(candidates, success, previous, parameter, goal_ids=()):
        reports = []
        for row, flag in zip(candidates, success):
            evidence = {"task_satisfied": True, "model_valid": bool(np.isfinite(row).all()),
                "native_pose_ik_success": bool(flag), "candidate_source": "native_ik",
                "environment_clearance_m": 1., "self_clearance_m": 1.,
                "joint_limit_margin_fraction": .5}
            if report_for is not None:
                evidence.update(report_for(row, bool(flag), parameter))
            reports.append(evidence)
        indices = rank_ik_candidates(candidates, reports, previous, count=planner.budget.ik_seeds,
            joint_tolerance=planner.problem.numerical.joint,
            query_kind="functional" if goal_ids else "reference")
        return indices, reports

    planner._rank_ik_candidates = Mock(side_effect=rank)
    planner.check_request = Mock()
    planner._build_goal_ik_candidates = Mock(return_value={})
    planner.device_cfg = SimpleNamespace(to_device=lambda values: torch.as_tensor(
        values, dtype=planner.start.position.dtype))
    dimensions = planner.start.position.shape[-1]
    planner.joint_names = tuple(f"joint{column}" for column in range(dimensions))
    limits = SimpleNamespace(joint_names=list(planner.joint_names),
        velocity=torch.tensor([[-2.]*dimensions, [2.]*dimensions], dtype=planner.start.position.dtype))
    planner._verification_native = SimpleNamespace(get_cost=lambda name:
        SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
    planner.formal_clock = {"knot_dt": .2, "duration": .2*(planner.trajopt.action_horizon+4)}
    if not hasattr(planner.start, "velocity"):
        planner.start.velocity = torch.zeros_like(planner.start.position)
    planner.physical_domain = CubicJointDomain(torch.full((dimensions,), -200.),
        torch.full((dimensions,), 200.), planner.start.position, planner.start.velocity,
        planner.formal_clock["knot_dt"], planner.trajopt.action_horizon)
# END declared static evidence cannot substitute for source checking or complete trajectory acceptance.


def serial_rotations(angles):
    """Independent three-revolute chain with two rigidly mounted tool frames.

    A world Z axis precedes a moving Y axis and then a moving X axis. The
    fourth joint is prismatic and has no angular contribution. World angular
    Jacobian columns are the independently rotated physical joint axes.
    """
    angles = np.asarray(angles)
    shape = angles.shape[:-1]
    mounts = (Rotation.identity(), Rotation.from_euler("xyz", [.63, -.47, .82]))
    quaternions, jacobians = [], []
    for configuration in angles.reshape(-1, 4):
        first = Rotation.from_rotvec([0., 0., configuration[0]])
        second = Rotation.from_rotvec([0., configuration[1], 0.])
        third = Rotation.from_rotvec([configuration[2], 0., 0.])
        orientation = first * second * third
        columns = np.column_stack((
            [0., 0., 1.], first.apply([0., 1., 0.]),
            (first * second).apply([1., 0., 0.]), np.zeros(3)))
        quaternions.append([(orientation * mount).as_quat()[[3, 0, 1, 2]]
                            for mount in mounts])
        jacobians.append([columns, columns])
    return (np.asarray(quaternions).reshape(*shape, 2, 4),
            np.asarray(jacobians).reshape(*shape, 2, 3, 4))


def constant_path():
    """A measured constant pose used only for an unconstrained test phase."""
    return PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))),
                    "independently specified constant Cartesian pose")


def cartesian_problem(phases, events=()):
    """Three explicit translation coordinates for export checks, not a robot."""
    return PlanningProblem(
        joint_names=("cartesian_x", "cartesian_y", "cartesian_z"),
        start_q=np.zeros(3), start_dq=np.zeros(3), tool_link="analytic_tool",
        reference_frame="analytic_world", base_pose=np.eye(4),
        tool_transform=np.eye(4), phases=tuple(phases), events=tuple(events),
        source_timestamps=np.array([0., float(len(phases))]), source_progress=np.array([0., float(len(phases))]))


def export_fixture(problem, progress_map, position, velocity, dt):
    """Exercise the real exporter with bounded, explicitly analytic FK.

    Native integration is a separate test responsibility. Here FK maps the
    supplied three coordinates to Cartesian translation exactly; all scene
    and sampled native-limit residuals are empty/zero. Explicit broad analytic
    acceleration bounds keep ordinary fixtures focused on process/timing;
    acceleration regressions replace those bounds with their stated values.
    """
    position = torch.as_tensor(position, dtype=torch.float64)[None]
    velocity = torch.as_tensor(velocity, dtype=torch.float64)[None]

    class CartesianKinematics:
        def compute_kinematics(self, state):
            q = state.position
            quaternion = q.new_zeros((*q.shape[:2], 1, 4))
            quaternion[..., 0] = 1.
            spheres = q.new_zeros((*q.shape[:2], 1, 4))
            spheres[..., :3] = q[:, :, None]
            spheres[..., 3] = .001
            return SimpleNamespace(
                tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
                                           position=q[:, :, None], quaternion=quaternion),
                robot_spheres=spheres)

    native = SimpleNamespace(position=position, velocity=velocity,
                             dt=torch.tensor([dt], dtype=torch.float64))
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.check_request = Mock()
    planner.physical_domain = SimpleNamespace(initialize=lambda values: values, validate=lambda values: None)
    planner.problem = problem
    # The Cartesian test coordinates span metres; declare their metric sampling
    # resolution explicitly while individual spatial tests select finer values.
    planner.budget = PlanningBudget(validation_spatial_step_m=.5)
    # BEGIN the mock uses the same exact occurrence clock as the native exporter.
    clock = {(event.skill_id, event.event_id, event.occurrence): {"event": asdict(event), "time": float(progress_map.anchor_parameters([event.progress])[0, 0]*(position.shape[1]-1)*dt),
        "hold_until": float(progress_map.anchor_parameters([event.progress])[-1, 0]*(position.shape[1]-1)*dt)} for event in problem.events}
    # END the source event pair cannot silently match a later attempt.
    planner.progress_map = _ProgressMap(progress_map.u, progress_map.s, clock)
    planner.anchors = problem.anchors
    planner.joint_names = problem.joint_names
    planner.torch = torch
    planner.device_cfg = SimpleNamespace(to_device=lambda x: torch.as_tensor(x, dtype=torch.float64))
    planner.JointState = SimpleNamespace
    planner._RobotState = lambda **kwargs: SimpleNamespace(**kwargs, robot_spheres=kwargs["cuda_robot_model_state"].robot_spheres)
    planner.trajopt = SimpleNamespace(get_active_js=lambda state: state)
    planner.formal_clock = {"valid_count": position.shape[1], "interpolation_dt": dt,
                            "duration": (position.shape[1]-1)*dt}
    planner.finger_trajectory = None
    planner.start = SimpleNamespace(position=position[:, 0], velocity=velocity[:, 0])
    planner._verification_fk = CartesianKinematics()
    # BEGIN the analytic Cartesian fixture declares its complete native tree.
    fixed = torch.eye(4, dtype=torch.float64)[:3].repeat(4, 1, 1)
    planner._verification_fk.config = SimpleNamespace(kinematics_config=SimpleNamespace(
        fixed_transforms=fixed, link_map=torch.tensor([0, 0, 1, 2]),
        joint_map=torch.tensor([-1, 0, 1, 2]), joint_map_type=torch.tensor([-1, 0, 1, 2]),
        joint_offset_map=torch.tensor([1., 0.]*4), joint_names=list(problem.joint_names),
        joint_limits=SimpleNamespace(position=torch.tensor([[-1e6]*3, [1e6]*3])),
        link_name_to_idx_map={"base": 0, "x": 1, "y": 2, problem.tool_link: 3},
        link_spheres=torch.tensor([[[0., 0., 0., .001]]]), link_sphere_idx_map=torch.tensor([3])))
    # END no omitted FK metadata can silently bypass geometric sampling.
    planner._verification_scene = SimpleNamespace(
        gripper=None, owners=(), required_progress_anchors=(),
        background=SimpleNamespace(independent_geometry={"solids": [{"kind": "tsdf", "voxel_size": 1.}]}),
        # BEGIN exporter fixtures explicitly supply the source-world acceptance API.
        evaluate_source=lambda spheres, p, r, s, q, names, **kwargs: torch.zeros_like(s),
        # END source geometry numerics remain the dedicated integration tests' responsibility.
        check_robot_self_collision=separated_source_self,
        check_robot_coverage=lambda *args, **kwargs: [])
    # BEGIN give the real exporter explicit, finite native acceleration bounds.
    limits = SimpleNamespace(joint_names=list(problem.joint_names),
        position=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64),
        velocity=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64),
        acceleration=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64))
    cspace = SimpleNamespace(enabled=True, config=SimpleNamespace(joint_limits=limits))
    planner._verification_native = SimpleNamespace(
        get_cost={"cspace": cspace}.get,
        compute_costs=lambda state: SimpleNamespace(names=(), values=()))
    # END these analytic fixture bounds are not a robot-model assertion.
    return planner, SimpleNamespace(batch_size=1, num_seeds=1, get_interpolated_plan=lambda: native)


class PlannerCalculusTests(unittest.TestCase):
    # BEGIN formal time and complete-volume gates are independent of point costs.
    # BEGIN fixed-clock sampling regressions; no native CUDA execution here.
    def test_formal_spline_clock_preserves_raw_duration_and_exact_dense_grid(self):
        from curobo._src.types.control_space import ControlSpace
        for native_dt, subdivisions, count in ((.2, 32, 641), (.05, 8, 161)):
            with self.subTest(native_dt=native_dt):
                native = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=native_dt,
                    maximum_trajectory_dt=native_dt, interpolation_dt=.025), interpolation_steps=4,
                    transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3), action_horizon=16)
                clock = _formal_native_clock(native)
                self.assertEqual(clock["valid_count"], count)
                self.assertEqual(clock["subdivisions"], subdivisions)
                self.assertEqual(clock["duration"], 80*float(np.float32(native_dt)))
                self.assertEqual(clock["duration"], (count-1)*float(np.float32(.025)))
                self.assertEqual(np.float32(clock["interpolation_dt"]*subdivisions), np.float32(clock["knot_dt"]))
        native.config.minimum_trajectory_dt = .04
        with self.assertRaisesRegex(ValueError, "fixed request-wide"):
            _formal_native_clock(native)

    def test_exact_grid_supports_general_representable_clock_and_rejects_retiming(self):
        grid = _fixed_spline_grid(intervals=10, native_dt=.125,
                                 interpolation_steps=3, interpolation_dt=.03125)
        self.assertEqual((grid["subdivisions"], grid["valid_count"], grid["duration"]), (12, 121, 3.75))
        cases = ((.2, 4, .03), (.20625, 4, .025), (.2, 3, .025),
                 (float(np.nextafter(np.float32(.2), np.float32(np.inf))), 4, .025))
        for native_dt, steps, output_dt in cases:
            with self.subTest(native_dt=native_dt, steps=steps, output_dt=output_dt):
                with self.assertRaisesRegex(ValueError, "Exact output grid"):
                    _fixed_spline_grid(intervals=20, native_dt=native_dt,
                                       interpolation_steps=steps, interpolation_dt=output_dt)
        for output_dt in (0., 1e-50, 1e50, float("nan"), float("inf")):
            with self.subTest(output_dt=output_dt):
                with self.assertRaisesRegex(ValueError, "finite positive native float32"):
                    _fixed_spline_grid(intervals=20, native_dt=.2,
                                       interpolation_steps=4, interpolation_dt=output_dt)

    def test_budget_admits_only_the_required_fixed_capacity_without_losing_limits(self):
        budget = PlanningBudget(knots=16, interpolation_buffer_size=641)
        self.assertEqual(budget.interpolation_buffer_size, 641)
        self.assertEqual(budget.max_validation_points, 4096)
        with self.assertRaisesRegex(ValueError, "Interpolation buffer"):
            PlanningBudget(knots=16, interpolation_buffer_size=640)
        with self.assertRaisesRegex(ValueError, "Interpolation buffer"):
            PlanningBudget(interpolation_buffer_size=2049)
        with self.assertRaisesRegex(ValueError, "Exact output grid"):
            PlanningBudget(interpolation_dt=.03)

    def test_nonzero_boundary_cubic_survives_same_time_dense_export(self):
        # SciPy evaluates one continuous cubic independently of the grid
        # helper. Coarse and dense samples must share its physical clock.
        grid = _fixed_spline_grid(intervals=10, native_dt=.2,
                                 interpolation_steps=4, interpolation_dt=.025)
        q0, dq0 = np.array([.3, -.8]), np.array([.17, -.09])
        controls = np.array([[.6, -.4], [.4, .2], [-.2, .7], [.1, -.3], [.8, .1], [.5, .6]])
        coefficients = np.vstack((q0 + np.array([-1., 0., 1., 2.])[:, None]*dq0*grid["knot_dt"],
                                  controls, np.repeat(controls[-1:], 3, axis=0)))
        oracle = BSpline(np.arange(-3., 14.)*grid["knot_dt"], coefficients, 3, axis=0)
        dense_times = np.arange(grid["valid_count"])*grid["interpolation_dt"]
        exported = JointTrajectory(("joint_a", "joint_b"), dense_times, oracle(dense_times),
            oracle(dense_times, 1), np.array([0., 1.]), np.array([0., 1.]), (), (), {})
        times = np.unique(np.r_[np.arange(41)*grid["native_dt"],
                                (dense_times[:-1]+dense_times[1:])/2, grid["duration"]])
        for order, tolerance in ((0, 1e-12), (1, 1e-11), (2, 1e-9)):
            np.testing.assert_allclose(exported._curve(times, order), oracle(times, order),
                                       rtol=0, atol=tolerance)
        np.testing.assert_allclose(exported.sample([0.])[0][0], q0, rtol=0, atol=1e-14)
        np.testing.assert_allclose(exported.sample([0.])[1][0], dq0, rtol=0, atol=1e-14)
        self.assertEqual(exported.duration, 40*grid["native_dt"])

    # END no synthetic zero-speed case substitutes for boundary preservation.



    # BEGIN a rounded spline boundary is a reference state, not measured q0.
    # END no tolerance, start replacement or alternate curve enters acceptance.

    def test_initializer_plateau_inverse_is_explicitly_earliest(self):
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.physical_domain = SimpleNamespace(initialize=lambda values: values, validate=lambda values: None)
        planner.progress_map = _ProgressMap(np.array([0., .2, .4, 1.]), np.array([0., 1., 1., 2.]))
        self.assertEqual(planner._seed_u(1.), .2)

    def test_native_self_collision_receives_timed_spheres_without_mutating_fk(self):
        from manipulation.planning_scene import PrismaticGripper
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        q = torch.tensor([[[.3, .1, 0.]]], dtype=torch.float64, requires_grad=True)
        quaternion = q.new_tensor([[[[1., 0., 0., 0.]]]])
        original = torch.cat((q[:, :, None]+q.new_tensor([.04, 0., 0.]),
                              q.new_full((1, 1, 1, 1), .001)), -1)
        kinematics = SimpleNamespace(robot_spheres=original,
            tool_poses=SimpleNamespace(tool_frames=[problem.tool_link], position=q[:, :, None], quaternion=quaternion),
            tool_jacobians=q.new_zeros((1, 1, 1, 6, 3)))

        class State:
            def __init__(self):
                self.joint_state = SimpleNamespace(position=q)
                self.cuda_robot_model_state = kinematics

            @property
            def robot_spheres(self):
                return self.cuda_robot_model_state.robot_spheres

            @property
            def tool_poses(self):
                return self.cuda_robot_model_state.tool_poses

        hand = PrismaticGripper(controller="timed_reference", control_timestep_s=.002, joint_names=("finger",),
            current_positions=[.04], joint_limits=[[0, .04]], joint_bounds=None,
            finger_trajectory_factory=lambda *_: None, coupling_matrix=np.empty((0, 1)), coupling_offset=[],
            sphere_indices=(0,), local_spheres=[[0, 0, 0, .001]],
            sphere_joint_translation=[[[1], [0], [0]]], self_collision_pairs=(),
            commands=[{"skill_id": "test", "event_id": "close", "occurrence": 0, "command": "close", "progress": 0.}],
            source="analytic timed native-self input")
        hand.finger_trajectory = SimpleNamespace(sample_torch=lambda t: (
            (.04-.02*t)[..., None], torch.full((*t.shape, 1), -.02, dtype=t.dtype)))
        hand.nominal_finger_trajectory = hand.finger_trajectory
        original_state = State()
        replaced = _timed_native_state(original_state, problem, SimpleNamespace(gripper=hand),
                                         problem.joint_names, q.new_tensor([[1.]]))
        self.assertIsNot(replaced.cuda_robot_model_state, original_state.cuda_robot_model_state)
        self.assertAlmostEqual(original_state.robot_spheres[0, 0, 0, 0].item(), .34)
        self.assertAlmostEqual(replaced.robot_spheres[0, 0, 0, 0].item(), .32)
        # The process manager wraps the returned sphere tensor again before
        # concatenating anchors. This layout identity remains idempotent.
        wrapped = _native_gradient_layout_type().apply(replaced.robot_spheres)
        gradient = torch.autograd.grad(wrapped[..., 0].sum(), q)[0]
        torch.testing.assert_close(gradient, q.new_tensor([[[1., 0., 0.]]]), rtol=0, atol=0)
    # END no nominal point objective can bypass native volume acceptance.

    # BEGIN independent full-horizon endpoint-family gradient regression.
    # END independent full-horizon endpoint-family gradient regression.

    def test_native_backward_receives_contiguous_gradients_after_batched_anchor_views(self):
        # A native FK callback has a stricter memory-layout contract than a
        # regular Torch op. Concatenation slices along H have a batch stride
        # from the longer combined horizon, even when values were cloned.
        seen = []

        class StrictNativeIdentity(torch.autograd.Function):
            @staticmethod
            def forward(ctx, value):
                return value.clone()

            @staticmethod
            def backward(ctx, gradient):
                seen.append(gradient.is_contiguous())
                if not gradient.is_contiguous():
                    raise AssertionError("native FK requires a contiguous output gradient")
                return gradient

        main = torch.randn((2, 4, 3, 4), dtype=torch.float64, requires_grad=True)
        anchors = torch.randn((2, 2, 3, 4), dtype=torch.float64, requires_grad=True)
        order = np.array([0, 4, 1, 5, 2, 3])
        weight = torch.arange(2*6*3*3, dtype=torch.float64).reshape(2, 6, 3, 3)/17.
        layout = _native_gradient_layout_type()
        combined = torch.cat((layout.apply(StrictNativeIdentity.apply(main)),
                              layout.apply(StrictNativeIdentity.apply(anchors))), dim=1)
        value = combined[:, order, :, :3]
        loss = (value*weight).sum()
        loss.backward()
        self.assertEqual(seen, [True, True])
        expected = np.zeros((2, 6, 3, 4))
        expected[..., :3] = weight.numpy()[:, np.argsort(order)]
        np.testing.assert_array_equal(main.grad.numpy(), expected[:, :4])
        np.testing.assert_array_equal(anchors.grad.numpy(), expected[:, 4:])
        np.testing.assert_array_equal(value.detach().numpy(),
                                      np.concatenate((main.detach().numpy(), anchors.detach().numpy()), axis=1)[:, order, :, :3])

    def test_world_angular_jacobian_vjp_matches_independent_fk_finite_differences(self):
        # Large compound rotations and a near-pi orientation expose the former
        # body/world convention error; identity-only examples cannot do so.
        angles = np.array([[[2.8, -.7, 1.1, .2], [np.pi-1e-4, .01, -.03, -.1]],
                           [[-.8, 1.2, -2.1, .5], [1.7, -.9, 2.6, .3]]])
        native_q, native_jacobian = serial_rotations(angles)
        target = Rotation.from_euler("xyz", [.31, -.72, 1.18]).as_quat()[[3, 0, 1, 2]]
        weights = np.array([.7, 1.3])

        def independent_loss(value):
            orientations, _ = serial_rotations(value)
            # Squared quaternion dot is a physical, double-cover-invariant
            # orientation objective, independent of the production VJP math.
            return np.sum((1.-np.square(orientations @ target))*weights)

        expected = np.zeros_like(angles)
        epsilon = 2e-6
        for index in np.ndindex(angles.shape):
            plus, minus = angles.copy(), angles.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (independent_loss(plus)-independent_loss(minus))/(2*epsilon)

        for quaternion_sign in (1., -1.):
            with self.subTest(quaternion_sign=quaternion_sign):
                joints = torch.tensor(angles, dtype=torch.float64, requires_grad=True)
                native_quaternion = torch.tensor(native_q*quaternion_sign, dtype=torch.float64,
                                                 requires_grad=True)
                output = _orientation_vjp_type().apply(
                    joints, native_quaternion, torch.tensor(native_jacobian, dtype=torch.float64))
                loss = ((1.-(output @ torch.tensor(target)).square())*torch.tensor(weights)).sum()
                loss.backward()
                np.testing.assert_allclose(joints.grad.numpy(), expected, rtol=2e-8, atol=5e-10)
                self.assertIsNone(native_quaternion.grad,
                                  "The native quaternion backward must not add a second convention-dependent gradient")
                np.testing.assert_array_equal(joints.grad.numpy()[..., 3], 0.)
                self.assertGreater(np.max(np.abs(joints.grad.numpy()[..., :3])), .1)

    def test_hermite_values_and_derivatives_match_scipy_with_distinct_counts_and_dt(self):
        rng = np.random.default_rng(481)
        position, velocity = rng.normal(size=(2, 3, 9, 4))
        counts = np.array([9, 6, 2])
        dt = np.array([.07, .13, .4])
        parameters = np.broadcast_to([0., .023, .17, 1/3, .501, .873, 1.], (3, 7)).copy()
        expected_q, expected_dq = [], []
        for row, count in enumerate(counts):
            curve = CubicHermiteSpline(np.arange(count)*dt[row], position[row, :count],
                                       velocity[row, :count], axis=0)
            times = parameters[row]*(count-1)*dt[row]
            expected_q.append(curve(times))
            expected_dq.append(curve(times, 1))
            # Invalid native padding must never stretch the skill schedule or
            # become a real knot, even when its values are conspicuously bad.
            position[row, count:] = 1e5
            velocity[row, count:] = -2e5
        actual_q, actual_dq = _hermite_at_parameters(
            torch.tensor(position), torch.tensor(velocity), torch.tensor(dt),
            torch.tensor(counts), torch.tensor(parameters))
        np.testing.assert_allclose(actual_q.numpy(), expected_q, rtol=3e-13, atol=3e-13)
        np.testing.assert_allclose(actual_dq.numpy(), expected_dq, rtol=3e-13, atol=3e-13)

    def test_exact_curve_anchor_gradients_exclude_padding_and_match_scipy(self):
        rng = np.random.default_rng(382)
        data = rng.normal(size=(2, 5, 2))
        count, dt = 3, .19
        parameters = np.array([[0., .17, .49, .79, 1.]])
        weights = rng.normal(size=(2, 5, 2))

        def independent_loss(value):
            curve = CubicHermiteSpline(np.arange(count)*dt, value[0, :count],
                                       value[1, :count], axis=0)
            times = parameters[0]*(count-1)*dt
            return np.sum(curve(times)*weights[0] + curve(times, 1)*weights[1])

        q = torch.tensor(data[0][None], requires_grad=True)
        dq = torch.tensor(data[1][None], requires_grad=True)
        result_q, result_dq = _hermite_at_parameters(
            q, dq, torch.tensor([dt]), torch.tensor([count]), torch.tensor(parameters))
        loss = (result_q*torch.tensor(weights[0]) + result_dq*torch.tensor(weights[1])).sum()
        loss.backward()
        actual = np.stack((q.grad.numpy()[0], dq.grad.numpy()[0]))
        epsilon = 1e-6
        expected = np.zeros_like(data)
        for index in np.ndindex(data.shape):
            plus, minus = data.copy(), data.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (independent_loss(plus)-independent_loss(minus))/(2*epsilon)
        np.testing.assert_allclose(actual, expected, rtol=3e-7, atol=2e-8)
        np.testing.assert_array_equal(actual[:, count:], 0.)


class NativeCubicClockTests(unittest.TestCase):
    """Check installed cubic boundary layout against SciPy basis functions.

    The independent layout oracle has four prescribed initial coefficients,
    N free actions and three copies of the last action. Uniform knots expose
    its real support and Greville times; no native GPU call is used here.
    """

    @staticmethod
    def native_cubic(actions, initial=0., initial_velocity=0., dt=1.):
        count = len(actions)
        prefix = initial+np.array([-1., 0., 1., 2.])*initial_velocity*dt
        coefficients = np.r_[prefix, actions, np.repeat(actions[-1], 3)]
        return BSpline(np.arange(-3., count+8.), coefficients, 3)

    # BEGIN exact-stop objective separate from physical feasibility tolerance.
    def test_stop_preference_has_an_interior_gradient_at_each_event_once(self):
        progress = torch.tensor([[.2, .3, .3, .3+0.5e-7, .3+2e-7, .6],
                                 [.3, .3, .3, .4, .6, .7]], dtype=torch.float64)
        velocity = torch.tensor([.25e-4, -.5e-4, .75e-4], dtype=torch.float64)
        velocity = velocity.expand(2, 6, 3).clone().requires_grad_(True)
        result = _verification_stop_preference(progress, velocity, [.3, .3, .6])
        # Duplicate queries and later plateau samples do not turn a single
        # stop event into an unrequested maintained stop objective.
        selected = np.array([[False, True, False, False, False, True],
                             [True, False, False, False, True, False]])
        expected = np.where(selected, np.sum(velocity.detach().numpy()**2, axis=-1), 0.)
        np.testing.assert_array_equal(result.detach().numpy(), expected)
        result.sum().backward()
        np.testing.assert_array_equal(velocity.grad.numpy(),
                                      np.where(selected[..., None], 2*velocity.detach().numpy(), 0.))
        self.assertGreater(float(velocity.grad[selected].abs().min()), 0.)
        self.assertTrue(bool(torch.all(velocity.detach().abs() < 1e-4)))
        np.testing.assert_array_equal(_verification_stop_preference(progress, velocity, ()).detach().numpy(),
                                      np.zeros(progress.shape))

    def test_stop_preference_never_changes_the_original_raw_acceptance_test(self):
        event = PlanningEvent("stop", "analytic", .3, 1, "verification", "test source", stop=True)
        problem = cartesian_problem((MotionPhase("free", constant_path()),), (event,))
        progress = torch.tensor([[.3, .3, .4]], dtype=torch.float64)
        position = torch.zeros((1, 3, 3), dtype=torch.float64)
        quaternion = torch.zeros((1, 3, 4), dtype=torch.float64)
        quaternion[..., 0] = 1.
        velocity = torch.tensor([[[.5e-4, 0., 0.], [1.25e-4, 0., 0.], [9., 0., 0.]]],
                                dtype=torch.float64, requires_grad=True)
        before = problem.residuals(position, quaternion, progress, joint_velocity=velocity)
        objective = _verification_stop_preference(progress, velocity, problem.stop_progress)
        objective.sum().backward()
        after = problem.residuals(position, quaternion, progress, joint_velocity=velocity)
        self.assertEqual(before.keys(), after.keys())
        for name in before:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
        np.testing.assert_array_equal(after["stop:analytic:stop"].detach().numpy(), [[0., 0., 0.]])
        self.assertGreater(float(objective[0, 0].detach()), 0.)
        self.assertEqual(float(objective[0, 1].detach()), 0.)
        self.assertEqual(float(objective[0, 2].detach()), 0.)
        bad_first = velocity.detach().clone(); bad_first[0, 0, 0] = 1.25e-4
        rejected = problem.residuals(position, quaternion, progress, joint_velocity=bad_first)
        np.testing.assert_allclose(rejected["stop:analytic:stop"].numpy(), [[.25e-4, 0., 0.]], atol=1e-20)
    # END exact-stop objective separate from physical feasibility tolerance.

    # BEGIN installed native retiming clamp contract without CUDA imports.
    def test_equal_native_dt_bounds_prevent_implicit_post_solve_retiming(self):
        import ast
        from pathlib import Path
        import curobo

        # Execute the installed arithmetic methods verbatim on CPU. Importing
        # the full trajectory solver would also import CUDA interpolation/JIT;
        # parsing these two functions avoids creating that unrelated runtime.
        root = Path(curobo.__file__).resolve().parent / "_src"
        functions = []
        for relative, name in (("util/trajectory.py", "calculate_dt_no_clamp"),
                               ("solver/solver_trajopt.py", "compute_trajectory_dt")):
            tree = ast.parse((root/relative).read_text())
            function = next(node for node in ast.walk(tree)
                            if isinstance(node, ast.FunctionDef) and node.name == name)
            function.decorator_list = []
            functions.append(function)
        future = ast.parse("from __future__ import annotations").body
        module = ast.fix_missing_locations(ast.Module(body=future+functions, type_ignores=[]))
        scope = {"torch": torch, "log_and_raise": lambda message: (_ for _ in ()).throw(ValueError(message))}
        exec(compile(module, "installed_native_dt_arithmetic", "exec"), scope)
        calculate = scope["compute_trajectory_dt"]
        limits = SimpleNamespace(max_velocity=torch.ones(2), max_acceleration=torch.ones(2),
                                 max_jerk=torch.ones(2))
        solver = SimpleNamespace(transition_model=limits,
                                 config=SimpleNamespace(minimum_trajectory_dt=.002, maximum_trajectory_dt=.2))
        velocity = torch.zeros((1, 2, 4, 2))
        velocity[:, 1] = 10.
        state = SimpleNamespace(position=torch.zeros_like(velocity), velocity=velocity,
                                acceleration=torch.zeros_like(velocity), jerk=torch.zeros_like(velocity),
                                dt=torch.full((1, 2), .2))
        before_cuda = torch.cuda.is_initialized()
        with patch("torch.cuda.init", side_effect=AssertionError("retiming arithmetic must remain on CPU")):
            # The former interval does shrink a slow/constant trajectory, so
            # this fixture would expose a post-solve clock change.
            old = calculate(solver, state)
            self.assertLess(float(old[0, 0]), .01)
            solver.config.minimum_trajectory_dt = .2
            for scaled in (False, True):
                fixed = calculate(solver, state, scale_dt=scaled)
                torch.testing.assert_close(fixed, torch.full((1, 2), .2), rtol=0, atol=0)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)
    # END installed native retiming clamp contract.

    # BEGIN independent native-basis progress warp and retained-path contracts.
    def test_source_progress_stays_fixed_on_actual_native_boundary_plateaus(self):
        source = _ProgressMap(np.array([0., .3, .55, .8, 1.]), np.array([0., .6, .6, 1.7, 2.]))
        for count in (6, 8, 12):
            with self.subTest(free_controls=count):
                curve = self.native_cubic(np.linspace(.1, 1., count))
                start, finish = curve.t[curve.k], curve.t[-curve.k-1]
                first_moving, last_moving = curve.t[curve.k+1], curve.t[-curve.k-2]
                leading = np.linspace(start, first_moving, 7)
                trailing = np.linspace(last_moving, finish, 7)
                np.testing.assert_allclose(curve(leading), 0., atol=1e-15)
                np.testing.assert_allclose(curve(trailing), 1., atol=1e-15)
                np.testing.assert_allclose(curve.derivative()(np.r_[leading, trailing]), 0., atol=1e-15)
                mapped, warp = _progress_for_cubic_boundaries(
                    source, count, np.zeros(3), stop_progress=(), anchors=source.s)
                np.testing.assert_array_equal(np.interp(leading/finish, mapped.u, mapped.s), 0.)
                np.testing.assert_array_equal(np.interp(trailing/finish, mapped.u, mapped.s), 2.)
                physical_source_times = np.interp(source.u, warp.s, warp.u)
                physical_source_times[0], physical_source_times[-1] = first_moving/finish, last_moving/finish
                np.testing.assert_allclose(np.interp(physical_source_times, mapped.u, mapped.s), source.s, atol=1e-14)
                np.testing.assert_allclose(mapped.anchor_parameters([.6])[:, 0], physical_source_times[[1, 2]], atol=1e-14)
                # Moving a free final action still changes the final joint
                # state. Padding the physical clock never fixes that action.
                shifted = np.linspace(.1, 1., count)
                shifted[-1] += .02
                self.assertAlmostEqual(self.native_cubic(shifted)(finish)-curve(finish), .02)

    def test_nonzero_measured_initial_velocity_does_not_get_a_fictitious_pause(self):
        source = _ProgressMap(np.array([0., .4, 1.]), np.array([0., .8, 2.]))
        measured = np.array([.27, -.13])
        curve = self.native_cubic(np.linspace(.2, .8, 8), initial=.1, initial_velocity=measured[0], dt=.2)
        self.assertAlmostEqual(curve(0.), .1)
        self.assertAlmostEqual(curve.derivative()(0.)/.2, measured[0])
        self.assertGreater(curve(.5), curve(0.))
        mapped, _ = _progress_for_cubic_boundaries(
            source, 8, measured, stop_progress=(), anchors=source.s)
        self.assertGreater(np.interp(.5/12., mapped.u, mapped.s), 0.)
        self.assertEqual(mapped.anchor_parameters([0.]).shape[0], 1)
        self.assertEqual(mapped.anchor_parameters([2.]).shape[0], 2)
        np.testing.assert_array_equal(measured, [.27, -.13])

    def test_internal_stop_matches_a_feasible_native_cubic_with_bounded_clock_error(self):
        # An odd number of free controls makes the single central stop exactly
        # symmetric: six increasing controls, three extra copies of .5, then
        # six increasing controls. This is an independently specified scalar
        # problem in the actual native basis, not a sampled robot trajectory.
        count = 15
        controls = np.r_[np.arange(1, 7)/12., [.5]*3, .5+np.arange(1, 7)/12.]
        exact_clock = self.native_cubic(controls)
        exact_angle = self.native_cubic(.45*np.maximum(2*controls-1., 0.))
        source = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        mapping, warp = _progress_for_cubic_boundaries(
            source, count, np.zeros(2), stop_progress=(1.,), anchors=(0., 1., 2.))
        query = np.linspace(0., 1., 20001)
        exact_parameter = exact_clock(query*(count+4))
        parameter = np.interp(query, warp.u, warp.s)
        progress = np.interp(query, mapping.u, mapping.s)
        self.assertLessEqual(float(np.max(np.abs(parameter-exact_parameter))), 1e-6+1e-14)
        np.testing.assert_allclose(progress, 2*parameter, atol=1e-14)
        # The same native q basis follows the entire required scalar carry
        # once its progress clock has the actual stop. No extra knot, relaxed
        # tolerance, or frozen internal robot boundary is used by this oracle.
        actual_angle = exact_angle(query*(count+4))
        wanted_angle = .45*np.maximum(progress-1., 0.)
        self.assertLess(float(np.max(np.abs(actual_angle-wanted_angle))), 1e-6)
        np.testing.assert_allclose(mapping.anchor_parameters([1.]),
                                   [[9/19, 1.], [10/19, 1.]], atol=2e-15)
        np.testing.assert_allclose(warp.anchor_parameters([.5]),
                                   [[9/19, .5], [10/19, .5]], atol=2e-15)
        stop_time = np.linspace(9., 10., 31)
        np.testing.assert_array_equal(exact_clock.derivative()(stop_time), 0.)
        np.testing.assert_array_equal(exact_clock.derivative(2)(stop_time), 0.)
        np.testing.assert_array_equal(np.interp(stop_time/19, mapping.u, mapping.s), 1.)
        self.assertLessEqual(len(mapping.u), 16384)
        self.assertLessEqual(len(warp.u), 16384)

    def test_nonzero_initial_velocity_clock_has_the_native_linear_prefix_and_later_stop(self):
        count, parameter_velocity = 15, 1/16
        # A nonzero measured velocity forbids a manufactured starting pause.
        # The scalar clock's first six actions continue its linear prefix and
        # reach .5, after which the same genuine four-control stop applies.
        controls = np.r_[np.arange(3, 9)*parameter_velocity, [.5]*3,
                         .5+np.arange(1, 7)/12.]
        exact = self.native_cubic(controls, initial_velocity=parameter_velocity)
        source = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        measured = np.array([.27, -.13])
        mapping, warp = _progress_for_cubic_boundaries(
            source, count, measured, stop_progress=(1.,), anchors=(0., 1., 2.))
        query = np.linspace(0., 1., 10001)
        self.assertLessEqual(float(np.max(np.abs(np.interp(query, warp.u, warp.s)
                                                 -exact(query*19)))), 1e-6+1e-14)
        first_interval = np.linspace(0., 1., 17)
        np.testing.assert_allclose(np.interp(first_interval/19, warp.u, warp.s),
                                   first_interval*parameter_velocity, atol=1e-14)
        self.assertEqual(mapping.anchor_parameters([0.]).shape, (1, 2))
        np.testing.assert_allclose(mapping.anchor_parameters([1.]),
                                   [[9/19, 1.], [10/19, 1.]], atol=2e-15)
        np.testing.assert_array_equal(measured, [.27, -.13])

    def test_required_short_returning_motion_keeps_every_exact_source_anchor(self):
        # This 40 mm out-and-back is much shorter in source progress than an
        # ordinary rollout sample. It must survive the warp as two distinct
        # source anchors; nearest-sample snapping would erase the excursion.
        source_progress = np.array([0., .49, .49002, .49004, 1.])
        poses = np.broadcast_to(np.eye(4), (len(source_progress), 4, 4)).copy()
        poses[2, 0, 3] = .04
        problem = cartesian_problem((MotionPhase(
            "short required return", PosePath(source_progress, poses, "analytic short source excursion")),))
        source = _ProgressMap(np.array([0., .2, .55, .75, 1.]),
                              np.array([0., .3, .3, .8, 1.]))
        before = (source.u.copy(), source.s.copy())
        mapping, warp = _progress_for_cubic_boundaries(
            source, 15, np.zeros(3), stop_progress=(.3,), anchors=problem.anchors)
        pairs = mapping.anchor_parameters(source_progress[1:4])
        np.testing.assert_array_equal(pairs[:, 1], source_progress[1:4])
        np.testing.assert_allclose(np.interp(pairs[:, 0], mapping.u, mapping.s),
                                   source_progress[1:4], atol=1e-15)
        self.assertTrue(np.all(np.diff(pairs[:, 0].astype(np.float32)) > 0))
        # The exact progress pair is what the native process manager supplies
        # at each appended FK anchor. Its float32 pose interpolation must keep
        # the actual return, including the middle source observation.
        p, _ = problem.sample_reference(torch.tensor(pairs[None, :, 1], dtype=torch.float32))
        np.testing.assert_allclose(p[0, :, 0].numpy(), [0., .04, 0.], atol=2e-9)
        source_u = np.interp(pairs[:, 1], source.s, source.u)
        np.testing.assert_allclose(np.interp(pairs[:, 0], warp.u, warp.s), source_u, atol=2e-15)
        np.testing.assert_array_equal(source.u, before[0])
        np.testing.assert_array_equal(source.s, before[1])

    def test_stop_count_clock_keys_and_native_precision_fail_with_explicit_bounds(self):
        source = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        with self.assertRaisesRegex(ValueError, "(?i)(stop|control|knot|budget)"):
            _progress_for_cubic_boundaries(source, 8, np.zeros(1),
                                           stop_progress=(.3, .7), anchors=(.3, .7))
        # Retaining all distinct preexisting path vertices is part of the
        # contract. An oversized input may be rejected, never decimated into
        # a different source or OMPL q/s path to fit the fixed key budget.
        crowded = _ProgressMap(np.linspace(0., 1., 16385), np.linspace(0., 1., 16385))
        with self.assertRaisesRegex(ValueError, "(?i)(key|clock|budget)"):
            _progress_for_cubic_boundaries(crowded, 8, np.zeros(1),
                                           stop_progress=(), anchors=())
        with self.assertRaisesRegex(ValueError, "(?i)(float32|precision)"):
            _progress_for_cubic_boundaries(source, 8, np.zeros(1),
                                           stop_progress=(), anchors=(.5, .5+1e-10))

    def test_native_key_alias_drops_only_redundant_grid_points_and_keeps_exact_anchors(self):
        # In this native clock's linear interior, z=.3 maps to u=4.4/12.
        # Forward grid arithmetic and inverse-anchor arithmetic yield nearby
        # float64 values that are the SAME float32 key. The optional grid key
        # can be replaced; the actual source anchor must keep its exact value.
        source = _ProgressMap(np.array([0., .3, .8, 1.]), np.array([0., .6, 1.6, 2.]))
        mapping, warp = _progress_for_cubic_boundaries(
            source, 8, np.zeros(1), stop_progress=(), anchors=source.s)
        self.assertTrue(np.all(np.diff(warp.u.astype(np.float32)) > 0))
        self.assertTrue(np.all(np.diff(mapping.u.astype(np.float32)) > 0))
        pairs = mapping.anchor_parameters([.6, 1.6])
        np.testing.assert_array_equal(pairs[:, 1], [.6, 1.6])
        np.testing.assert_allclose(pairs[:, 0], (2+8*np.array([.3, .8]))/12, atol=2e-15)
        retained = np.searchsorted(warp.u, pairs[:, 0])
        np.testing.assert_array_equal(warp.u[retained], pairs[:, 0])
        np.testing.assert_array_equal(warp.s[retained], [.3, .8])
        exact = self.native_cubic(np.arange(1, 9)/8.)
        query = np.linspace(0., 1., 20001)
        self.assertLessEqual(float(np.max(np.abs(np.interp(query, warp.u, warp.s)-exact(query*12)))),
                             1e-6+1e-14)

        # Two true anchors can be distinct in the source's float32 clock yet
        # collide after this compression. They must be rejected together;
        # choosing either one would discard a required process observation.
        linear = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        close = np.array([.6, .6+2*float(np.spacing(np.float32(.3)))])
        self.assertEqual(linear.anchor_parameters(close).shape, (2, 2))
        with self.assertRaisesRegex(ValueError, "(?i)(float32|precision)"):
            _progress_for_cubic_boundaries(linear, 8, np.zeros(1), stop_progress=(), anchors=close)

    # BEGIN a longer spatial route retains the identical binding and plateau time.
    def test_ompl_initializer_preserves_clock_when_q_moves_during_progress_pause(self):
        from manipulation.physical_joint_domain import CubicJointDomain

        problem = cartesian_problem((MotionPhase("reference", constant_path()),),
            (PlanningEvent("query", "analytic", .3, 10, "function", "fixture"),))
        mapping = _ProgressMap(np.array([0., .25, .7, 1.]), np.array([0., .3, .3, 1.]))
        path = np.array([[0., 0., 0., 0.], [.3, 0., 0., .25],
                         [.3, .6, 0., .7], [1.5, .1, 0., 1.]])
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.torch, planner.problem, planner.progress_map = torch, problem, mapping
        planner.budget = PlanningBudget(knots=15, interpolation_buffer_size=768)
        planner.anchors, planner.joint_names = problem.anchors, problem.joint_names
        planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
        planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64),
                                        velocity=torch.zeros((1, 3), dtype=torch.float64))
        planner.trajopt = SimpleNamespace(action_horizon=15, transition_model=SimpleNamespace(
            action_bound_lows=torch.full((3,), -2.), action_bound_highs=torch.full((3,), 2.)))
        planner.formal_clock, planner.finger_trajectory = {"duration": 2.}, None
        planner.physical_domain = CubicJointDomain(torch.full((3,), -2.), torch.full((3,), 2.),
            planner.start.position, planner.start.velocity, 2./19., 15)
        limits = SimpleNamespace(joint_names=list(planner.joint_names),
                                velocity=torch.tensor([[-10.]*3, [10.]*3]))
        planner._verification_native = SimpleNamespace(get_cost=lambda name:
            SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
        planner.JointState = SimpleNamespace
        def fk(state):
            q = state.position
            quaternion = q.new_zeros((*q.shape[:2], 1, 4)); quaternion[..., 0] = 1.
            return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
                position=q[:, :, None], quaternion=quaternion),
                tool_jacobians=q.new_zeros((*q.shape[:2], 1, 6, 3)))
        planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
        planner._orientation_vjp = SimpleNamespace(apply=lambda q, quaternion, jacobian: quaternion)
        planner._native_layout = SimpleNamespace(apply=lambda value: value)
        scene = SimpleNamespace(project_residual=lambda progress, q, names: q[..., :0],
                                required_progress_anchors=())
        scene.clone_for_rollout = lambda: scene
        planner.scene = scene
        # BEGIN this clock-only search fixture declares no functional IK proposals.
        planner.goal_ik_candidates = {}
        # END the existing reference/candidate proposals still exercise the paused clock.
        candidate = SimpleNamespace(duration=2., _curve=lambda moment: path[-1, :-1])
        with patch("manipulation.planning._ompl_stage", return_value=(path, {"success": True})) as search:
            seed, report = planner.ompl_seed(torch.zeros((1, 1, 15, 3)), candidate)
        self.assertTrue(report["success"])
        self.assertFalse(report["clock_changed"])
        self.assertIs(planner.progress_map, mapping)
        self.assertEqual(search.call_count, 1)
        np.testing.assert_array_equal(search.call_args.args[0][-1:], [0.])
        np.testing.assert_array_equal(search.call_args.args[1][-1:], [1.])
        self.assertIn(.25, search.call_args.kwargs["query_parameters"])
        self.assertIn(.7, search.call_args.kwargs["query_parameters"])
        rebuilt = planner.physical_domain.evaluate(seed[0, 0], path[:, -1])
        np.testing.assert_allclose(rebuilt, path[:, :-1], atol=1e-5)
        self.assertTrue(report["cubic_fit"]["success"])
        paused = planner.physical_domain.evaluate(seed[0, 0], np.linspace(.25, .7, 11))
        self.assertGreater(float(np.ptp(paused[:, 1])), .1)
    # END spatial distance no longer creates new event times.
    # END independent native-basis progress warp and retained-path contracts.

    def test_every_free_action_uses_its_scipy_greville_time_without_consuming_q0(self):
        count = 8
        source = _ProgressMap(np.array([0., .35, .6, 1.]), np.array([0., .7, .7, 2.]))
        mapping, _ = _progress_for_cubic_boundaries(
            source, count, np.zeros(1), stop_progress=(), anchors=source.s)
        curve = self.native_cubic(np.linspace(.1, 1., count))
        free_indices = np.arange(4, 4+count)
        greville = np.array([curve.t[index+1:index+curve.k+1].mean() for index in free_indices])
        clock = greville/curve.t[-curve.k-1]
        requested, requested_parameters = [], []

        def solve_pose(goal, **kwargs):
            requested.append(goal)
            requested_parameters.append(float(planner.seed_parameter.item()))
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor([[[goal], [goal+.2]]])),
                                   success=torch.tensor([[True, True]]))

        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.torch = torch
        planner.budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=1e-5))
        planner.start = SimpleNamespace(position=torch.tensor([[-.2]]))
        # BEGIN explicitly model the native initializer's independent time input.
        planner.seed_parameter = torch.zeros((1, 1))
        # END progress alone cannot identify a point on a pause interval.
        planner.progress_map = mapping
        planner.trajopt = SimpleNamespace(action_horizon=count)
        planner.ik = SimpleNamespace(solve_pose=solve_pose, get_active_js=lambda state: state)
        planner._goal = lambda progress: progress
        bind_analytic_ik_screening(planner)
        seeds, success_counts = planner.reference_seeds()
        np.testing.assert_allclose(requested, np.interp(clock, mapping.u, mapping.s), atol=1e-14)
        np.testing.assert_array_equal(requested_parameters, clock.astype(np.float32).astype(float))
        self.assertEqual(seeds.shape, (1, 1, count, 1))
        self.assertEqual(success_counts, [2]*count)
        self.assertTrue(torch.all(seeds[:, :, 0] != planner.start.position).item())
        self.assertTrue(np.all(np.diff(requested) >= 0))

    # BEGIN native constant-seed initialization contract after failed pose IK.
    def test_two_seeds_keep_failed_ik_diagnostic_and_measured_state_initializer(self):
        from curobo.types import DeviceCfg
        from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator

        horizon = 8
        measured_q = torch.tensor([[.1, -.2, .3]], dtype=torch.float64)
        measured_dq = torch.tensor([[.4, -.5, .6]], dtype=torch.float64)
        position_before, velocity_before = measured_q.clone(), measured_dq.clone()
        native_generator = TrajectorySeedGenerator(horizon, 3, DeviceCfg("cpu"))
        helper = Mock(wraps=native_generator.generate_constant_seeds)
        goals, supplied_seed_shapes, requested_parameters = [], [], []

        def failed_pose_ik(goal, **kwargs):
            goals.append(goal)
            requested_parameters.append(float(planner.seed_parameter.item()))
            supplied_seed_shapes.append(tuple(kwargs["seed_config"].shape))
            # Finite failed poses remain diagnostic evidence; neither this
            # query's seed nor the next native warm start may use them.
            first = measured_q+goal
            second = first+1.
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.stack((first, second), dim=1)),
                                   success=torch.tensor([[False, False]]))

        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.torch = torch
        planner.budget = PlanningBudget(ik_seeds=2, trajectory_seeds=2)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=1e-5))
        planner.start = SimpleNamespace(position=measured_q, velocity=measured_dq)
        # BEGIN failed pose IK still receives the real control-point clock.
        planner.seed_parameter = torch.zeros((1, 1))
        # END diagnostic failure does not remove the independent time input.
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        planner.trajopt = SimpleNamespace(action_horizon=horizon,
                                         _trajectory_seed_generator=SimpleNamespace(generate_constant_seeds=helper))
        planner.ik = SimpleNamespace(solve_pose=failed_pose_ik, get_active_js=lambda state: state)
        planner._goal = lambda progress: progress
        bind_analytic_ik_screening(planner)
        seeds, success_counts = planner.reference_seeds()
        np.testing.assert_array_equal(requested_parameters,
            ((np.arange(horizon)+3.)/(horizon+4.)).astype(np.float32).astype(float))
        self.assertEqual(helper.call_count, 2)
        self.assertIs(helper.call_args.args[0], measured_q)
        self.assertEqual(helper.call_args.args[1], 1)
        self.assertEqual(tuple(seeds.shape), (1, 2, horizon, 3))
        self.assertTrue(seeds.is_contiguous())
        self.assertEqual(success_counts, [0]*horizon)
        self.assertEqual(supplied_seed_shapes, [(1, 1, 3)]*horizon)
        # BEGIN all-failed references preserve evidence without a fabricated anchor chain.
        self.assertEqual(planner.reference_ik_chains, [])
        self.assertTrue(all(row["reference_status"] == "FAILED_REFERENCE"
                            for row in planner.reference_ik_queries))
        self.assertEqual(planner.initial_seed_strategies, ["measured_state_constant"]*2)
        np.testing.assert_array_equal(seeds[0, 0].numpy(), np.broadcast_to(measured_q.numpy(), (horizon, 3)))
        # END the independent measured-state initializer retains the original entrance.
        np.testing.assert_array_equal(seeds[0, 1].numpy(), np.broadcast_to(measured_q.numpy(), (horizon, 3)))
        torch.testing.assert_close(measured_q, position_before, rtol=0, atol=0)
        torch.testing.assert_close(measured_dq, velocity_before, rtol=0, atol=0)
        self.assertIs(planner.start.position, measured_q)
        self.assertIs(planner.start.velocity, measured_dq)

        # A broken IK batch loses only that reference sample; the legal
        # measured-state initializer remains available in the same budget.
        planner.ik.solve_pose = lambda *args, **kwargs: SimpleNamespace(
            js_solution=SimpleNamespace(position=torch.full((1, 2, 3), float("nan"))),
            success=torch.tensor([[False, False]]))
        seeds, failures = planner.reference_seeds()
        self.assertTrue(bool(torch.isfinite(seeds).all()))
        self.assertEqual(failures, [0]*horizon)
        self.assertEqual(planner.reference_ik_chains, [])
        self.assertTrue(all(row["warm_start_from_query"] is None for row in planner.reference_ik_queries))
        # Each raw query is screened once. Reused warm starts only initialize
        # IK and never masquerade as successful states at a different time.
        self.assertEqual(planner._rank_ik_candidates.call_count, 2*horizon)
    # END native constant-seed initialization contract after failed pose IK.


# BEGIN native-result diversity and whole-window initializer selection contracts.
class ReferenceSeedDiversityTests(unittest.TestCase):
    """Declared IK outputs test selection, never actual native pose feasibility."""

    @staticmethod
    def planner_for(pools, flags, *, seed_count=2, joint_tolerance=1e-5, report_for=None):
        from curobo.types import DeviceCfg
        from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator

        horizon, candidates = len(pools), len(pools[0])
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.torch = torch
        planner.budget = PlanningBudget(knots=horizon, ik_seeds=candidates, trajectory_seeds=seed_count)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=joint_tolerance))
        planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64),
                                        velocity=torch.tensor([[.1, -.2, .3]], dtype=torch.float64))
        # BEGIN seed selection fixtures retain the production time parameter.
        planner.seed_parameter = torch.zeros((1, 1), dtype=torch.float64)
        # END no implicit default clock is added to the production planner.
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        generator = TrajectorySeedGenerator(horizon, 3, DeviceCfg("cpu"))
        constant = Mock(wraps=generator.generate_constant_seeds)
        planner.trajopt = SimpleNamespace(action_horizon=horizon,
                                         _trajectory_seed_generator=SimpleNamespace(generate_constant_seeds=constant))
        warm_starts = []

        def solve(goal, **kwargs):
            index = len(warm_starts)
            np.testing.assert_array_equal(planner.seed_parameter.numpy(),
                                          [[(index+3.)/(horizon+4.)]])
            warm_starts.append(kwargs["seed_config"].clone())
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor(pools[index][None], dtype=torch.float64)),
                                   success=torch.tensor(flags[index][None], dtype=torch.bool))

        planner.ik = SimpleNamespace(solve_pose=solve, get_active_js=lambda state: state,
                                     config=SimpleNamespace(exit_early=seed_count == 1))
        planner._goal = lambda progress: progress
        bind_analytic_ik_screening(planner, report_for)
        return planner, constant, warm_starts

    @staticmethod
    def two_tracks():
        primary = np.array([[.1+.01*index, .02*index, 0.] for index in range(6)])
        alternative = np.array([[1.5+.02*index, .2+.01*index, 0.] for index in range(6)])
        pools, flags = [], []
        for index, (a, b) in enumerate(zip(primary, alternative)):
            # The third successful candidate is close to the primary. The
            # fourth is invalid and much farther away than any usable seed.
            values = np.stack((a, a+[.15, 0., 0.], b, [-100., 0., 0.]))
            success = np.array([True, True, True, False])
            order = np.array([0, 1, 2, 3] if index % 2 == 0 else [2, 3, 1, 0])
            pools.append(values[order])
            flags.append(success[order])
        return primary, alternative, pools, flags

    def test_whole_chain_continuity_retains_successes_despite_result_order_changes(self):
        primary, alternative, pools, flags = self.two_tracks()
        planner, constant, warm_starts = self.planner_for(pools, flags)
        measured_q, measured_dq = planner.start.position.clone(), planner.start.velocity.clone()
        seeds, success = planner.reference_seeds()
        chains = planner.reference_ik_chains
        np.testing.assert_allclose(chains[0]["joint_positions"], primary, atol=1e-14)
        self.assertEqual(len(chains), 2)
        self.assertGreater(float(np.max(np.abs(np.asarray(chains[0]["joint_positions"])
                                              -np.asarray(chains[1]["joint_positions"])))),
                           planner.problem.numerical.joint)
        for chain in chains:
            self.assertTrue(chain["reference_success"])
            for query, index in enumerate(chain["candidate_indices"]):
                self.assertTrue(flags[query][index])
                np.testing.assert_array_equal(chain["joint_positions"][query], pools[query][index])
        self.assertEqual(success, [3]*6)
        self.assertEqual(planner.reference_ik_distinct_counts, [3]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "reference_ik_alternative"])
        constant.assert_not_called()
        self.assertEqual(tuple(warm_starts[0].shape), (1, 1, 3))
        for index in range(1, 6):
            self.assertEqual(tuple(warm_starts[index].shape), (1, 3, 3))
            self.assertEqual({tuple(row) for row in warm_starts[index].numpy()[0]},
                             {tuple(row) for row in pools[index-1][flags[index-1]]})
        torch.testing.assert_close(planner.start.position, measured_q, rtol=0, atol=0)
        torch.testing.assert_close(planner.start.velocity, measured_dq, rtol=0, atol=0)
        self.assertTrue(seeds.is_contiguous())

    def test_a_single_success_can_merge_traces_and_later_successes_split_them_again(self):
        primary = np.array([[.1, 0., 0.], [.5, .1, 0.], [.4, .1, 0.],
                            [.42, .1, 0.], [.6, .1, 0.], [.5, .1, 0.]])
        alternative = np.array([[1.5, .2, 0.], [.5, .1, 0.], [.8, .2, 0.],
                                [.82, .2, 0.], [.6, .1, 0.], [.9, .2, 0.]])
        pools, flags = [], []
        for index, (a, b) in enumerate(zip(primary, alternative)):
            values = np.stack((a, b, [-50., 0., 0.], [50., 0., 0.]))
            success = np.array([True, index not in (1, 4), False, False])
            order = np.arange(4) if index % 2 == 0 else np.arange(3, -1, -1)
            pools.append(values[order])
            flags.append(success[order])
        planner, constant, _ = self.planner_for(pools, flags)
        seeds, success = planner.reference_seeds()
        chains = planner.reference_ik_chains
        self.assertEqual(len(chains), 2)
        for chain in chains:
            for query, index in enumerate(chain["candidate_indices"]):
                self.assertTrue(flags[query][index])
            np.testing.assert_array_equal(np.asarray(chain["joint_positions"])[[1, 4]], primary[[1, 4]])
        self.assertFalse(np.array_equal(chains[0]["joint_positions"], chains[1]["joint_positions"]))
        self.assertEqual(success, [2, 1, 2, 2, 1, 2])
        self.assertEqual(planner.reference_ik_distinct_counts, success)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "reference_ik_alternative"])
        constant.assert_not_called()

    def test_one_failed_control_pose_keeps_its_diagnostic_and_successful_alternative_chain(self):
        _, _, pools, flags = self.two_tracks()
        flags[2] = np.zeros(4, dtype=bool)
        planner, constant, _ = self.planner_for(pools, flags)
        seeds, success = planner.reference_seeds()
        self.assertEqual(success, [3, 3, 0, 3, 3, 3])
        self.assertEqual(planner.reference_ik_distinct_counts, [3, 3, 0, 3, 3, 3])
        self.assertEqual(planner.initial_seed_strategies,
                         ["reference_ik_with_fallback", "reference_ik_with_fallback_alternative"])
        self.assertEqual([row["query_index"] for row in planner.reference_ik_chains[0]["recovery_queries"]], [2])
        self.assertTrue(bool(torch.isfinite(seeds).all()))
        self.assertGreater(float(seeds[0, 0].abs().max()), 0.)
        self.assertEqual(len(planner.reference_ik_chains), 2)
        self.assertTrue(all(chain["candidate_indices"][2] is None for chain in planner.reference_ik_chains))
        constant.assert_not_called()

    def test_numerically_duplicate_native_results_do_not_claim_a_second_reference_trace(self):
        tolerance = 1e-5
        pools = [np.array([[.3+.01*index, 0., 0.], [.3+.01*index+.25*tolerance, 0., 0.]])
                 for index in range(6)]
        planner, constant, _ = self.planner_for(pools, [np.ones(2, dtype=bool)]*6, joint_tolerance=tolerance)
        seeds, success = planner.reference_seeds()
        self.assertEqual(success, [2]*6)
        self.assertEqual(planner.reference_ik_distinct_counts, [1]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "measured_state_constant"])
        np.testing.assert_array_equal(seeds.numpy()[0, 1], np.zeros((6, 3)))
        constant.assert_called_once()

    def test_distinction_uses_declared_joint_numerics_without_an_extra_branch_threshold(self):
        tolerance = 1e-5
        pools = [np.array([[.3+.01*index, 0., 0.], [.3+.01*index+4*tolerance, 0., 0.]])
                 for index in range(6)]
        planner, constant, _ = self.planner_for(pools, [np.ones(2, dtype=bool)]*6, joint_tolerance=tolerance)
        seeds, _ = planner.reference_seeds()
        self.assertEqual(planner.reference_ik_distinct_counts, [2]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "reference_ik_alternative"])
        selected = np.asarray([row["joint_positions"] for row in planner.reference_ik_chains])
        self.assertGreater(float(np.max(np.abs(selected[1]-selected[0]))), tolerance)
        self.assertLessEqual(float(np.max(np.abs(selected[1]-selected[0]))), 4*tolerance+1e-14)
        constant.assert_not_called()

    def test_single_seed_budget_keeps_its_nearest_native_continuation(self):
        primary, _, pools, flags = self.two_tracks()
        planner, constant, _ = self.planner_for(pools, flags, seed_count=1)
        seeds, success = planner.reference_seeds()
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))
        np.testing.assert_allclose(planner.reference_ik_chains[0]["joint_positions"], primary, atol=1e-14)
        rebuilt = planner.physical_domain.evaluate(seeds[0, 0], (np.arange(6)+3.)/10.)
        np.testing.assert_allclose(rebuilt, primary, atol=1e-5)
        self.assertEqual(success, [3]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik"])
        constant.assert_not_called()

    def test_reference_seed_chooses_a_complete_chain_before_fitting_coefficients(self):
        pools = [np.array([[0., 0., 0.], [.4, 0., 0.]])]
        pools.extend(np.array([[.5, 0., 0.], [.5, 0., 0.]]) for _ in range(5))
        planner, constant, warm_starts = self.planner_for(pools, [np.ones(2, bool)]*6, seed_count=1)
        seeds, _ = planner.reference_seeds()
        selected = np.asarray(planner.reference_ik_chains[0]["joint_positions"])
        self.assertEqual(selected[0, 0], .4)
        np.testing.assert_array_equal(selected[1:, 0], np.full(5, .5))
        self.assertEqual(tuple(warm_starts[1].shape), (1, 2, 3))
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))
        self.assertTrue(planner.reference_seed_fits[0]["success"])
        constant.assert_not_called()

    def test_failed_optional_reference_fit_retains_evidence_and_measured_initializer(self):
        _, _, pools, flags = self.two_tracks()
        planner, constant, _ = self.planner_for(pools, flags, seed_count=1)
        with patch.object(planner.physical_domain, "fit_reference",
                          return_value=(None, {"success": False, "reason": "declared_fit_failure"})):
            seeds, success = planner.reference_seeds()
        self.assertEqual(success, [3]*6)
        self.assertTrue(planner.reference_ik_chains[0]["reference_success"])
        np.testing.assert_array_equal(seeds.numpy()[0, 0], np.zeros((6, 3)))
        self.assertEqual(planner.initial_seed_strategies, ["measured_state_after_reference_fit_failure"])
        fit = planner.reference_seed_fits[0]
        self.assertFalse(fit["success"])
        self.assertEqual(fit["reason"], "declared_fit_failure")
        self.assertEqual(fit["initializer_after_failure"], "measured_state_constant")
        constant.assert_called_once()

    # BEGIN actual source priorities and off-grid functional hints retain the original seed budget.
    def test_extra_clearance_does_not_override_qualified_whole_chain_continuity(self):
        primary, _, pools, flags = self.two_tracks()

        def report(row, success, parameter):
            return {"task_satisfied": success, "model_valid": success,
                    "environment_clearance_m": .002 if row[0] > 1. else .0001,
                    "self_clearance_m": 1., "joint_limit_margin_fraction": .5}

        planner, constant, warm_starts = self.planner_for(pools, flags, seed_count=1, report_for=report)
        seeds, success = planner.reference_seeds()
        np.testing.assert_allclose(planner.reference_ik_chains[0]["joint_positions"], primary, atol=1e-14)
        self.assertEqual(success, [3]*6)
        self.assertEqual(len(warm_starts), 6)
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))
        self.assertEqual(planner._rank_ik_candidates.call_count, 6)
        constant.assert_not_called()

    def test_off_grid_functional_goal_is_fitted_at_its_actual_time_with_fixed_entrance(self):
        _, _, pools, flags = self.two_tracks()
        ordinary, _, _ = self.planner_for(pools, flags)
        baseline, _ = ordinary.reference_seeds()
        planner, _, _ = self.planner_for(pools, flags)
        original_q, original_dq = planner.start.position.clone(), planner.start.velocity.clone()
        valid = {"task_satisfied": True, "model_valid": True}
        rejected = {"task_satisfied": True, "model_valid": False}
        target_postures = ([.9, .8, .7], [.6, .5, .4])
        planner._build_goal_ik_candidates.return_value = {
            0.: {"candidates": [{"joint_position": [99., 99., 99.], "report": valid}]},
            .47: {"candidates": [
                {"joint_position": target_postures[0], "report": valid},
                {"joint_position": [90., 90., 90.], "report": rejected},
                {"joint_position": target_postures[1], "report": valid}]}}
        seeds, success = planner.reference_seeds()
        control_u = (np.arange(6)+3.)/10.
        self.assertNotIn(.47, control_u)
        before = ordinary.physical_domain.evaluate(baseline[0], np.array([.47]))[:, 0]
        after = planner.physical_domain.evaluate(seeds[0], np.array([.47]))[:, 0]
        self.assertTrue(np.all(np.linalg.norm(after-target_postures, axis=-1)
                               < np.linalg.norm(before-target_postures, axis=-1)))
        # Seven soft observations share six coefficients; the event remains
        # an optimization hint and does not become an interpolation equation.
        self.assertFalse(np.allclose(after, target_postures, atol=1e-5))
        np.testing.assert_allclose(planner.physical_domain.evaluate(seeds[0], np.array([0.]))[:, 0],
                                   np.broadcast_to(original_q.numpy(), (2, 3)), atol=1e-14)
        np.testing.assert_allclose(planner.physical_domain.evaluate(seeds[0], np.array([0.]), 1)[:, 0],
                                   np.broadcast_to(original_dq.numpy(), (2, 3)), atol=1e-14)
        torch.testing.assert_close(planner.start.position, original_q, rtol=0, atol=0)
        torch.testing.assert_close(planner.start.velocity, original_dq, rtol=0, atol=0)
        self.assertEqual(success, [3]*6)
        self.assertEqual(len(planner.goal_ik_fit_hints), 2)
        self.assertTrue(all(row["event_u"] == .47 and "control_index" not in row
                            for row in planner.goal_ik_fit_hints))
        self.assertTrue(all(value.endswith("+ranked_functional_ik") for value in planner.seed_strategies))

    def test_functional_events_sharing_a_nearest_control_keep_separate_physical_queries(self):
        _, _, pools, flags = self.two_tracks()
        planner, _, _ = self.planner_for(pools, flags, seed_count=1)
        valid = {"task_satisfied": True, "model_valid": True}
        planner._build_goal_ik_candidates.return_value = {
            .47: {"candidates": [{"joint_position": [.18, .15, .1], "report": valid}]},
            .49: {"candidates": [{"joint_position": [.2, .17, .12], "report": valid}]}}
        original_fit = planner.physical_domain.fit_reference
        with patch.object(planner.physical_domain, "fit_reference", wraps=original_fit) as fitting:
            seeds, _ = planner.reference_seeds()
        control_u = (np.arange(6)+3.)/10.
        self.assertEqual(int(np.argmin(abs(control_u-.47))), int(np.argmin(abs(control_u-.49))))
        np.testing.assert_array_equal(fitting.call_args.args[0][-2:], [.47, .49])
        self.assertEqual([row["event_u"] for row in planner.goal_ik_fit_hints], [.47, .49])
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))

    def test_failed_optional_functional_hint_keeps_the_preceding_curve_exactly(self):
        _, _, pools, flags = self.two_tracks()
        baseline_planner, _, _ = self.planner_for(pools, flags, seed_count=1)
        baseline, _ = baseline_planner.reference_seeds()
        planner, _, _ = self.planner_for(pools, flags, seed_count=1)
        planner._build_goal_ik_candidates.return_value = {
            .47: {"candidates": [{"joint_position": [.9, .8, .7],
                "report": {"task_satisfied": True, "model_valid": True}}]}}
        original_fit = planner.physical_domain.fit_reference

        def fit(samples, values, **kwargs):
            if len(samples) > planner.trajopt.action_horizon:
                return None, {"success": False, "reason": "declared_hint_fit_failure"}
            return original_fit(samples, values, **kwargs)

        with patch.object(planner.physical_domain, "fit_reference", side_effect=fit):
            seeds, _ = planner.reference_seeds()
        torch.testing.assert_close(seeds, baseline, atol=0., rtol=0.)
        hint = planner.reference_seed_fits[0]["functional_hint_fit"]
        self.assertFalse(hint["success"])
        self.assertEqual(hint["reason"], "declared_hint_fit_failure")
        self.assertFalse(planner.goal_ik_fit_hints[0]["applied"])
        self.assertTrue(planner.initial_seed_strategies[0].endswith("+functional_hint_fit_failed"))
    # END fitted pose proposals retain their own event times and permit residual error.
# END native-result diversity and whole-window initializer selection contracts.


# BEGIN installed native IK seed-coverage control flow without CUDA imports.
class NativeIkCoverageTests(unittest.TestCase):
    def test_one_successful_seed_skips_native_optimization_only_when_early_exit_is_enabled(self):
        import ast
        from pathlib import Path
        import curobo

        # Execute the installed method's actual branching on CPU. The fake
        # optimizer makes all eight supplied seeds successful, so its call
        # boundary distinguishes seed coverage from convergence claims. This
        # deliberately does not simulate real robot IK or CUDA numerics.
        source = Path(curobo.__file__).resolve().parent / "_src/solver/solver_ik.py"
        tree = ast.parse(source.read_text())
        function = next(node for node in ast.walk(tree)
                        if isinstance(node, ast.FunctionDef) and node.name == "_solve_impl")
        function.decorator_list = []
        future = ast.parse("from __future__ import annotations").body
        module = ast.fix_missing_locations(ast.Module(body=future+[function], type_ignores=[]))

        class CpuTimer:
            def start(self):
                return self

            def stop(self):
                return 0.

        scope = {"torch": torch, "CudaEventTimer": CpuTimer,
                 "log_and_raise": lambda message: (_ for _ in ()).throw(ValueError(message))}
        exec(compile(module, "installed_native_ik_seed_coverage", "exec"), scope)
        solve = scope["_solve_impl"]
        original = torch.arange(24., dtype=torch.float64).reshape(1, 8, 3)
        goal = SimpleNamespace(ndim=5)
        goal.reorder_links = Mock(return_value=goal)
        before_cuda = torch.cuda.is_initialized()
        with patch("torch.cuda.init", side_effect=AssertionError("native control-flow oracle must remain on CPU")):
            for early in (True, False):
                with self.subTest(exit_early=early):
                    optimizer = SimpleNamespace(reinitialize=Mock(), optimize=Mock(side_effect=lambda values: values+1))

                    def result_for(values, metrics, seeds, returned):
                        success = torch.zeros((1, 8), dtype=torch.bool)
                        success[:, 0] = True
                        if not torch.equal(values, original):
                            success[:] = True
                        return SimpleNamespace(success=success)

                    native = SimpleNamespace(
                        config=SimpleNamespace(exit_early=early, exit_early_batch_success_threshold=1.),
                        kinematics=SimpleNamespace(tool_frames=["analytic_tool"]),
                        _fill_current_state_dt=lambda value: value,
                        _prepare_goal_buffer=lambda **kwargs: (None, None),
                        prepare_action_seeds=Mock(return_value=original),
                        _update_rollout_params=Mock(),
                        metrics_rollout=SimpleNamespace(compute_metrics_from_action=lambda value: value),
                        _get_result=result_for, _compute_solution_velocity=Mock(), optimizer=optimizer)
                    result = solve(native, SimpleNamespace(batch_size=1), goal, 8, return_seeds=8)
                    if early:
                        optimizer.optimize.assert_not_called()
                        self.assertEqual(int(result.success.sum()), 1)
                    else:
                        optimizer.reinitialize.assert_called_once_with(original)
                        optimizer.optimize.assert_called_once_with(original)
                        self.assertEqual(int(result.success.sum()), 8)
                    native.prepare_action_seeds.assert_called_once_with(1, 8, seed_config=None)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)
# END installed native IK seed-coverage control flow without CUDA imports.


# BEGIN native manager configuration ownership without GPU allocation.
class NativeCostManagerOwnershipTests(unittest.TestCase):
    """Run installed native constructors while excluding GPU-only resources.

    Only CUDA stream creation and Warp runtime initialization are replaced.
    Native configuration binding, cost construction, weight normalization and
    enable/disable behavior execute unchanged on CPU tensors. These checks
    establish lifecycle ownership, not collision or trajectory numerics.
    """

    def setUp(self):
        before_cuda = torch.cuda.is_initialized()
        guard = patch("torch.cuda._lazy_init", side_effect=AssertionError(
            "Native configuration ownership tests must not initialize CUDA"))
        guard.start()
        self.addCleanup(guard.stop)
        for name, options in (
            ("curobo._src.rollout.cost_manager.cost_manager_robot.create_cuda_stream_pair",
             {"return_value": (None, None)}),
            ("curobo._src.cost.cost_cspace_base.init_warp", {}),
        ):
            replacement = patch(name, **options)
            replacement.start()
            self.addCleanup(replacement.stop)
        self.addCleanup(lambda: self.assertEqual(torch.cuda.is_initialized(), before_cuda))

        from curobo._src.robot.types.joint_limits import JointLimits
        from curobo._src.types.device_cfg import DeviceCfg
        self.JointLimits = JointLimits
        self.device_cfg = DeviceCfg("cpu")
        # BEGIN native-only configuration declares an explicitly empty AL state.
        # Empty process_mode selects native costs alone; no task, scene or FK
        # stand-in is needed to exercise the actual integration boundary.
        from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
        self.config_type = _process_manager_type(None, None, None, None, None,
            functional_constraints=FunctionalAugmentedLagrangian(()))
        # END configuration ownership tests do not invent functional requirements.
        self.raw = {
            "self_collision_cfg": {"weight": 5000., "convert_to_binary": True},
            "cspace_cfg": {
                "weight": [10., 20., 30., 40., 50.],
                "activation_distance": [0.] * 5,
                "squared_l2_regularization_weight": [1000., 10000., 5., 0., 10000.],
                "cost_type": "STATE",
            },
        }

    def configuration(self):
        return self.config_type.create(deepcopy(self.raw), device_cfg=self.device_cfg)

    def transition(self, interpolation_steps=4, *, teleport=False, bound_scale=1.):
        bounds = torch.tensor([[-1., -2.], [1., 2.]]) * bound_scale
        limits = self.JointLimits(
            ["first_joint", "second_joint"], bounds, bounds * 2, bounds * 3,
            bounds * 4, bounds * 5, self.device_cfg)
        # Native constructors only bind the self-collision metadata here;
        # no collision forward/kernel is invoked by this lifecycle test.
        robot = SimpleNamespace(
            get_self_collision_config=lambda: SimpleNamespace(num_spheres=2),
            total_spheres=2)
        return SimpleNamespace(
            robot_model=robot, interpolation_steps=interpolation_steps,
            action_dim=2, teleport_mode=teleport, get_state_bounds=lambda: limits)

    def manager(self, config, transition):
        result = config.class_type(self.device_cfg)
        result.initialize_from_config(config, transition)
        return result

    # BEGIN required/refinement anchors share native self-collision semantics.
    @contextmanager
    def anchor_self_collision_fixture(self, mode, *, required=(), hand=None, padding=0., capacity=2):
        """Keep native cost/buffer behavior; replace GPU FK and its pair kernel.

        The analytic robot has a sphere at x=q[0] and a fixed sphere at zero.
        Endpoint states are separated; a cubic interpolation can collide at
        an exact anchor. Two batches exercise the native gradient layout.
        """
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.state.state_robot import RobotState
        from curobo.types import JointState

        def kinematics(joints):
            q = joints.position
            p = torch.cat((q, torch.zeros_like(q[..., :1])), -1)
            r = q.new_zeros((*q.shape[:2], 1, 4)); r[..., 0] = 1.
            moving = torch.cat((q[..., :1], torch.zeros_like(q), torch.full_like(q[..., :1], .2)), -1)
            fixed = torch.cat((torch.zeros_like(p), torch.full_like(q[..., :1], .2)), -1)
            return SimpleNamespace(tool_poses=SimpleNamespace(position=p.unsqueeze(-2), quaternion=r,
                tool_frames=["tool"]), tool_jacobians=q.new_zeros((*q.shape[:2], 1, 6, 2)),
                robot_spheres=torch.stack((moving, fixed), -2))

        def pair_kernel(spheres, output, gradient, pair_distance, sparse, weight, sphere_padding,
                        pairs, *unused):
            # Independent Euclidean sphere-pair oracle. Returning the supplied
            # output tensor preserves the native reusable-buffer contract.
            radius = spheres[..., 3] + sphere_padding
            first, second = pairs[:, 0].long(), pairs[:, 1].long()
            squared_distance = (spheres[..., first, :3]-spheres[..., second, :3]).square().sum(-1)
            squared_overlap = (radius[..., first]+radius[..., second]).square()-squared_distance
            active = (radius[..., first] >= 0) & (radius[..., second] >= 0)
            value = .5*weight*torch.where(active, squared_overlap.clamp_min(0), 0.).amax(-1, keepdim=True)
            output.copy_(value)
            return output

        forwarded = []
        original_forward = SelfCollisionCost.forward

        def native_forward(cost, spheres):
            value = original_forward(cost, spheres)
            forwarded.append({"cost": cost, "spheres": spheres.detach().clone(),
                "distance_buffer": cost._out_distance.data_ptr(), "gradient_buffer": cost._out_grad.data_ptr(),
                "value": value.detach().clone()})
            return value

        q = torch.tensor([[[-1., 0.], [1., 0.]], [[-1.1, 0.], [.9, 0.]]], requires_grad=True)
        joints = JointState.from_position(q, joint_names=["first_joint", "second_joint"])
        joints.velocity = torch.zeros_like(q)
        joints.dt = torch.ones(2)
        state = RobotState(joint_state=joints, cuda_robot_model_state=kinematics(joints))
        problem = SimpleNamespace(tool_link="tool", start_progress=0., phase_count=1., end_progress=1., base_pose=np.eye(4),
            mechanisms=(), stop_progress=(), transform_tool_pose=lambda p, r: (p, r),
            transform_robot_spheres=lambda value: value,
            evaluate=lambda p, *args, **kwargs: p[..., 0]*0,
            optimizer_preferences=lambda p, *args, **kwargs: p[..., 0]*0)
        # BEGIN this geometry oracle has no functional point inequalities.
        problem.functional_constraint_keys = ()
        problem.functional_inequalities = lambda *args, **kwargs: ({}, {})
        problem.residuals = lambda *args, **kwargs: {"synthetic_process": problem.evaluate(*args, **kwargs)}
        # END the original process tensor remains unchanged under the named interface.
        scene = SimpleNamespace(gripper=hand, robot_coverage=None, robot_owner_permissions=(),
            required_progress_anchors=(), evaluate=lambda spheres, p, *args, **kwargs: p[..., 0]*0,
            # BEGIN native metrics query an explicit source-world fixture independently of self.
            evaluate_source=Mock(side_effect=lambda spheres, p, *args, **kwargs: p[..., 0]*0),
            # END optimizer guidance retains its existing differentiable scene callback.
            check_robot_self_collision=Mock(side_effect=separated_source_self),
            robot_self_collision_guidance=lambda spheres, p, *args, **kwargs: p[..., 0]*0,
            optimizer_preferences=lambda progress, *args: progress*0)
        scene.clone_for_rollout = lambda: scene
        transition = self.transition()
        transition.joint_names = ("first_joint", "second_joint")
        transition.robot_model.tool_frames = ["tool"]
        transition.robot_model.config = object()
        transition.robot_model.get_self_collision_config = lambda: SimpleNamespace(num_spheres=2,
            sphere_padding=torch.tensor(padding), collision_pairs=torch.tensor([[0, 1]], dtype=torch.int16),
            num_blocks_per_batch=1, max_threads_per_block=32)
        with ExitStack() as stack:
            stack.enter_context(patch("curobo.kinematics.Kinematics",
                return_value=SimpleNamespace(compute_kinematics=kinematics)))
            stack.enter_context(patch("manipulation.planning._orientation_vjp_type",
                return_value=SimpleNamespace(apply=lambda q, r, jacobian: r)))
            stack.enter_context(patch.object(RobotCostManager, "_stream_context", side_effect=lambda _: nullcontext()))
            stack.enter_context(patch("curobo._src.rollout.cost_manager.cost_manager_robot.synchronize_cuda_streams"))
            stack.enter_context(patch("curobo._src.cost.cost_self_collision.SelfCollisionDistance.apply",
                side_effect=pair_kernel))
            stack.enter_context(patch.object(SelfCollisionCost, "forward", autospec=True, side_effect=native_forward))
            config_type = _process_manager_type(problem, scene,
                _ProgressMap(np.array([0., 1.]), np.array([0., 1.])), required,
                PlanningBudget(constraint_weight=777., max_task_anchors=capacity),
                clock={"duration": 2.}, seed_parameter=torch.tensor([[.3]]))
            data = {"process_mode": mode}
            if mode.endswith("optimizer"):
                data["self_collision_cfg"] = {"weight": 8., "convert_to_binary": False}
            config = config_type.create(data, device_cfg=self.device_cfg)
            manager = self.manager(config, transition)
            yield SimpleNamespace(manager=manager, state=state, q=q, forwarded=forwarded, config=config,
                                  scene=scene, kinematics=kinematics, config_type=config_type,
                                  problem=problem, capacity=capacity)

    def test_required_self_collision_anchors_retain_native_values_and_gradient(self):
        for mode in ("trajectory_optimizer",):
            for source in ("required",):
                with self.subTest(mode=mode, source=source):
                    # u=.55 is inside the endpoint spheres' overlap region but
                    # off its symmetric maximum, so its gradient must be nonzero.
                    sites = {source: (.55,)}
                    with self.anchor_self_collision_fixture(mode, **sites) as fixture:
                        costs = fixture.manager.compute_costs(fixture.state)
                        self.assertIn("anchor_self_collision", costs.names)
                        self.assertEqual(len(fixture.forwarded), 2)
                        main, anchor = fixture.forwarded
                        self.assertIsNot(main["cost"], anchor["cost"])
                        self.assertNotEqual(main["distance_buffer"], anchor["distance_buffer"])
                        self.assertNotEqual(main["gradient_buffer"], anchor["gradient_buffer"])
                        self.assertEqual(main["cost"]._weight.item(), 2.)
                        self.assertEqual(anchor["cost"]._weight.item(), 2.)
                        self.assertEqual(fixture.config.self_collision_cfg.weight.item(), 8.)
                        self.assertTrue(bool((main["value"] == 0).all()))
                        self.assertTrue(bool((anchor["value"] > 0).all()))
                        main_cost = costs.values[costs.names.index("self_collision")]
                        anchor_cost = costs.values[costs.names.index("anchor_self_collision")]
                        torch.testing.assert_close(main_cost, main["value"], rtol=0., atol=0.)
                        self.assertEqual(anchor_cost.shape, main_cost.shape)
                        torch.testing.assert_close(anchor_cost.sum(1), (anchor["value"]*fixture.manager._anchor_valid[..., None]).sum(1), rtol=0., atol=0.)
                        # The native weight/binary operation is authoritative:
                        # no project constraint weight or second square applies.
                        center = anchor["spheres"][..., 0, 0]
                        expected = (.4**2-center.square()).clamp_min(0)
                        if mode.endswith("metrics"):
                            expected = torch.where(expected > 0, expected.clamp_max(1)+1, expected)
                        torch.testing.assert_close(anchor_cost.sum(1).squeeze(-1), (expected*fixture.manager._anchor_valid).sum(1))
                        sum(value.sum() for value in costs.values).backward()
                        self.assertTrue(bool(torch.isfinite(fixture.q.grad).all()))
                        self.assertGreater(fixture.q.grad.abs().max().item(), 0.)

    # BEGIN fixed admission capacity does not create padded FK/scene work.
    def test_real_anchor_count_and_rebind_preserve_capacity_and_required_times(self):
        from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
        with self.anchor_self_collision_fixture("trajectory_optimizer", required=(.55,), capacity=256) as fixture:
            manager = fixture.manager
            pointer = manager._anchor_u.data_ptr()
            self.assertEqual(manager._anchor_u.shape[1], 256)
            manager.compute_costs(fixture.state)
            self.assertEqual(fixture.forwarded[-1]["spheres"].shape[1], 1)
            self.assertEqual(manager._query_counts["anchor_count"], 1)
            self.assertEqual(manager._query_counts["exact_states"], 2*(2+1))
            # The same native manager and fixed input allocation accept another
            # request with two genuine witnesses, rather than 254 duplicates.
            mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
            budget = PlanningBudget(constraint_weight=777., max_task_anchors=256)
            fixture.config_type.bind_request(fixture.problem, fixture.scene, mapping,
                (.55, .65), budget, FunctionalAugmentedLagrangian(()))
            self.assertEqual(manager._anchor_u.data_ptr(), pointer)
            self.assertEqual(manager._anchor_count, 2)
            manager.compute_costs(fixture.state)
            self.assertEqual(fixture.forwarded[-1]["spheres"].shape[1], 2)
            self.assertEqual(manager._query_counts["exact_states"], 2*(2+2))
            torch.testing.assert_close(manager._anchor_u[:, :2], torch.tensor([[.55, .65]]))
            with self.assertRaisesRegex(ValueError, "Task anchors"):
                fixture.config_type.bind_request(fixture.problem, fixture.scene, mapping,
                    np.linspace(0., 1., 257), budget, FunctionalAugmentedLagrangian(()))
    # END overflow still fails admission; no original event witness is dropped.

    def test_disabled_self_collision_and_reference_ik_do_not_add_anchor_checks(self):
        for mode in ("trajectory_optimizer", "seed_optimizer"):
            with self.subTest(mode=mode):
                with self.anchor_self_collision_fixture(mode, required=(.55, .65)) as fixture:
                    if mode.startswith("trajectory_"):
                        fixture.manager.disable_cost_component("self_collision")
                    costs = fixture.manager.compute_costs(fixture.state)
                    self.assertNotIn("anchor_self_collision", costs.names)
                    self.assertEqual(len(fixture.forwarded), 1 if mode.startswith("seed_") else 0)

    # BEGIN native metrics retain their installed complete success formula.
    @staticmethod
    def native_success(collection, *, position_error=0., interpolated=None):
        from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints
        from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult
        batch = collection.values[0].shape[0]
        convergence = CostCollection()
        convergence.add(torch.full((batch, 1, 1), position_error), "tool_pose_position_tolerance")
        metrics = SimpleNamespace(costs_and_constraints=CostsAndConstraints(constraints=collection),
                                  convergence=convergence)
        result = SimpleNamespace(batch_size=batch, num_seeds=1, metrics=metrics,
            interpolated_metrics=(None if interpolated is None else SimpleNamespace(
                costs_and_constraints=CostsAndConstraints(constraints=interpolated))),
            position_tolerance=.1, orientation_tolerance=.1)
        TrajOptSolverResult._process_metrics(result)
        return result.success



            # END padding cannot become source-world evidence through the scene callback.

    # END source collisions replace only self feasibility, never native joint or pose requirements.

    def test_anchor_self_collision_uses_timed_hand_geometry_and_native_padding(self):
        for controller in ("timed_reference", "binary"):
            for nominal in (False, True):
                with self.subTest(controller=controller, nominal=nominal):
                    times = []

                    def hand_state(progress, q, names, *, trajectory_time):
                        times.append(trajectory_time.detach().clone())
                        # BEGIN native and scene geometry consume one uncertainty flag.
                        from manipulation.planning_scene import GripperState
                        return GripperState((.1*trajectory_time)[..., None], torch.zeros_like(trajectory_time),
                            None, torch.zeros_like(trajectory_time, dtype=torch.bool) if nominal else trajectory_time > 0)
                        # END a command-only future is not a measured hand width.

                    def hand_spheres(spheres, position, quaternion, fingers, half):
                        # One translating finger sphere has an independent
                        # physical-time command. Arm geometry stays untouched.
                        finger = torch.cat((fingers, torch.zeros_like(spheres[..., 1, 1:3]),
                                            spheres[..., 1, 3:4]), -1)
                        return torch.stack((spheres[..., 0, :], finger), -2)

                    hand = SimpleNamespace(controller=controller, state=hand_state, spheres=hand_spheres,
                        nominal_finger_trajectory=object() if nominal else None,
                        sphere_indices=(1,), sphere_joint_translation=[[[1.], [0.], [0.]]])
                    with self.anchor_self_collision_fixture("trajectory_optimizer", required=(.55, .65), hand=hand, padding=.3) as fixture:
                        original = fixture.state.robot_spheres.detach().clone()
                        costs = fixture.manager.compute_costs(fixture.state)
                        self.assertEqual(len(times), 2)
                        torch.testing.assert_close(times[0], torch.tensor([[0., 2.], [0., 2.]]))
                        torch.testing.assert_close(times[1], torch.tensor([[1.1, 1.3], [1.1, 1.3]]))
                        main, anchor = fixture.forwarded
                        torch.testing.assert_close(anchor["spheres"][..., 1, 0], .1*times[1])
                        self.assertAlmostEqual(anchor["cost"].config.self_collision_kin_config.sphere_padding.item(), .3)
                        torch.testing.assert_close(fixture.state.robot_spheres, original, rtol=0., atol=0.)
                        anchor_cost = costs.values[costs.names.index("anchor_self_collision")]
                        if nominal:
                            torch.testing.assert_close(anchor["spheres"][..., 1, 3], torch.full((2, 2), .2))
                            self.assertTrue(bool((anchor_cost[:, -1] > 0).all()))
                        else:
                            # Missing future physical finger geometry remains
                            # disabled even after the native radius padding.
                            torch.testing.assert_close(anchor["spheres"][..., 1, 3], torch.full((2, 2), -1.3))
                            self.assertTrue(bool((anchor_cost == 0).all()))
                            self.assertTrue(bool((main["spheres"][:, 0, 1, 3] > 0).all()))
    # END anchors augment native self collision without changing its physical semantics.

    def test_repeated_managers_and_verifier_each_normalize_self_weight_once(self):
        template = self.configuration()
        managers = []
        # Six users reproduce metrics/auxiliary/interpolated rollout sharing;
        # distinct interpolation counts must use their own native divisor.
        for steps in (4, 4, 4, 4, 4, 4, 1, 8):
            manager = self.manager(template, self.transition(steps))
            managers.append(manager)
            self.assertIsNot(manager.config, template)
            self.assertEqual(manager.get_cost("self_collision")._weight.item(), 5000. / steps)
            self.assertTrue(manager.get_cost("self_collision").enabled)
            torch.testing.assert_close(manager.get_cost("cspace")._weight,
                                       torch.tensor(self.raw["cspace_cfg"]["weight"]), rtol=0, atol=0)
            torch.testing.assert_close(
                manager.get_cost("cspace").config.squared_l2_regularization_weight,
                torch.tensor(self.raw["cspace_cfg"]["squared_l2_regularization_weight"]), rtol=0, atol=0)

        # The verifier inherits the metrics configuration with only native
        # cspace limits; source solids replace its former sphere hard gate.
        metrics_template = deepcopy(template)
        metrics_template.self_collision_cfg = None
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.physical_domain = SimpleNamespace(initialize=lambda values: values, validate=lambda values: None)
        planner.device_cfg = self.device_cfg
        planner.trajopt = SimpleNamespace(
            config=SimpleNamespace(metrics_rollout_config=SimpleNamespace(constraint_cfg=metrics_template)),
            transition_model=self.transition())
        for _ in range(2):
            planner._initialize_verification_constraints()
            self.assertIsNone(planner._verification_native.get_cost("self_collision"))
            torch.testing.assert_close(planner._verification_native.get_cost("cspace")._weight,
                                       torch.tensor(self.raw["cspace_cfg"]["weight"]), rtol=0, atol=0)
        self.assertEqual(template.self_collision_cfg.weight.item(), 5000.)
        self.assertIsNone(template.self_collision_cfg.self_collision_kin_config)
        self.assertIsNone(template.cspace_cfg.joint_limits)
        self.assertEqual(len({id(manager.config) for manager in managers}), len(managers))

        # Native runtime toggles must restore this instance's normalized
        # weight and leave sibling managers enabled with their own weights.
        managers[0].disable_cost_component("self_collision")
        self.assertFalse(managers[0].get_cost("self_collision").enabled)
        self.assertEqual(managers[1].get_cost("self_collision")._weight.item(), 1250.)
        managers[0].enable_cost_component("self_collision")
        self.assertEqual(managers[0].get_cost("self_collision")._weight.item(), 1250.)

    def test_transition_binding_keeps_cspace_types_limits_and_smoothness_private(self):
        template = self.configuration()
        teleport_transition = self.transition(1, teleport=True)
        state_transition = self.transition(4, bound_scale=3.)
        teleport = self.manager(template, teleport_transition).get_cost("cspace")
        state = self.manager(template, state_transition).get_cost("cspace")
        # The native teleport conversion selects POSITION fields. A later
        # STATE manager must still receive every original derivative weight.
        self.assertEqual(teleport.config.cost_type.name, "POSITION")
        self.assertEqual(state.config.cost_type.name, "STATE")
        self.assertEqual(template.cspace_cfg.cost_type.name, "STATE")
        torch.testing.assert_close(teleport._weight, torch.tensor([10., 50.]), rtol=0, atol=0)
        torch.testing.assert_close(state._weight, torch.tensor([10., 20., 30., 40., 50.]), rtol=0, atol=0)
        torch.testing.assert_close(state.config.squared_l2_regularization_weight,
                                   torch.tensor([1000., 10000., 5., 0., 10000.]), rtol=0, atol=0)
        for cost, transition in ((teleport, teleport_transition), (state, state_transition)):
            for name in ("position", "velocity", "acceleration", "jerk", "effort"):
                torch.testing.assert_close(getattr(cost.config.joint_limits, name),
                                           getattr(transition.get_state_bounds(), name), rtol=0, atol=0)
        self.assertIsNone(template.cspace_cfg.joint_limits)

    def test_failed_native_binding_does_not_contaminate_a_later_manager(self):
        template = self.configuration()
        broken = self.transition()
        # Native initialization binds self-collision before requesting the
        # cspace limits. Fail at that later boundary to expose partial writes.
        broken.get_state_bounds = Mock(side_effect=ValueError("unavailable test bounds"))
        with self.assertRaisesRegex(ValueError, "unavailable test bounds"):
            self.manager(template, broken)
        healthy = self.manager(template, self.transition())
        self.assertEqual(healthy.get_cost("self_collision")._weight.item(), 1250.)
        self.assertEqual(template.self_collision_cfg.weight.item(), 5000.)
        self.assertIsNone(template.self_collision_cfg.self_collision_kin_config)
        self.assertIsNone(template.cspace_cfg.joint_limits)

    # BEGIN reference IK is numerical initialization, with full trajectory checks retained.
    def test_reference_ik_cost_graph_skips_scene_while_trajectory_keeps_scene_and_guidance(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.rollout.metrics import CostCollection

        def kinematics(joints):
            q = joints.position
            p = torch.cat((q, torch.zeros_like(q[..., :1])), -1)
            r = q.new_zeros((*q.shape[:2], 1, 4)); r[..., 0] = 1.
            return SimpleNamespace(tool_poses=SimpleNamespace(position=p.unsqueeze(-2), quaternion=r),
                tool_jacobians=q.new_zeros((*q.shape[:2], 1, 6, 2)),
                robot_spheres=torch.cat((p, torch.full_like(p[..., :1], .01)), -1).unsqueeze(-2))

        def native_cost(manager, state, cost_collection=None, goal=None, **kwargs):
            collection = CostCollection()
            collection.add(state.joint_state.position.square().sum(-1, keepdim=True), "native")
            return collection

        for mode in ("seed_optimizer", "seed_metrics", "trajectory_optimizer", "trajectory_metrics"):
            with self.subTest(mode=mode):
                q = torch.tensor([[[.1, .2], [.3, .4]]], requires_grad=True)
                joints = SimpleNamespace(position=q, velocity=torch.zeros_like(q), dt=torch.tensor([1.]))
                native = kinematics(joints)
                state = SimpleNamespace(joint_state=joints, tool_poses=native.tool_poses,
                    cuda_robot_model_state=native, robot_spheres=native.robot_spheres)
                mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
                problem = SimpleNamespace(tool_link="tool", start_progress=0., phase_count=1., end_progress=1.,
                    mechanisms=(), stop_progress=(), transform_tool_pose=lambda p, r: (p, r),
                    transform_robot_spheres=lambda value: value,
                    evaluate=Mock(side_effect=lambda p, *args, **kwargs: p.square().sum(-1)),
                    optimizer_preferences=Mock(side_effect=lambda p, *args, **kwargs: p[..., 0]*0))
                # BEGIN preserve the original nonfunctional process and gradient oracle.
                problem.functional_constraint_keys = ()
                problem.functional_inequalities = lambda *args, **kwargs: ({}, {})
                problem.residuals = lambda *args, **kwargs: {"synthetic_process": problem.evaluate(*args, **kwargs)}
                # END an empty AL contribution cannot change native scene ownership.
                # BEGIN the synthetic scene explicitly declares no owner contact permissions.
                scene = SimpleNamespace(gripper=None, contacts=(), owners=(), robot_coverage=object(), robot_owner_permissions=(),
                    required_progress_anchors=(0., 1.),
                    check_robot_self_collision=Mock(side_effect=separated_source_self),
                    # BEGIN nonzero guidance checks the original summed optimizer objective.
                    robot_self_collision_guidance=lambda spheres, p, *args, **kwargs: p.square().sum(-1),
                    # END source metrics never consume this differentiable proxy value.
                    evaluate=Mock(side_effect=lambda spheres, p, *args, **kwargs: p.square().sum(-1)),
                    # BEGIN an explicitly separated source world contributes no proxy gradient.
                    evaluate_source=Mock(side_effect=lambda spheres, p, r, progress, *args, **kwargs: torch.zeros_like(progress)),
                    # END process constraints remain independently differentiable in metrics.
                    robot_coverage_guidance=Mock(side_effect=lambda poses, progress: poses.square().sum(-1)),
                    optimizer_preferences=lambda progress, *args: torch.zeros_like(progress))
                scene.clone_for_rollout = Mock(return_value=scene)
                # END the mock follows the same required scene contract as production.
                transition = self.transition()
                transition.joint_names = ("first_joint", "second_joint")
                transition.robot_model.tool_frames = ["tool"]
                transition.robot_model.config = object()
                with patch("manipulation.planning._native_gradient_layout_type",
                           return_value=SimpleNamespace(apply=lambda value: value)), \
                     patch("manipulation.planning._orientation_vjp_type",
                           return_value=SimpleNamespace(apply=lambda q, r, jacobian: r)), \
                     patch("curobo.kinematics.Kinematics", return_value=SimpleNamespace(
                           compute_kinematics=kinematics)) as anchor_fk, \
                     patch.object(RobotCostManager, "compute_costs", autospec=True,
                                  side_effect=native_cost) as native_evaluate:
                    config_type = _process_manager_type(problem, scene, mapping, (0., 1.),
                        PlanningBudget(max_task_anchors=2), clock={"duration": 1.}, seed_parameter=torch.tensor([[.4]]))
                    config = config_type.create({"process_mode": mode}, device_cfg=self.device_cfg)
                    manager = self.manager(config, transition)
                    collection = manager.compute_costs(state)
                    native_evaluate.assert_called_once()
                    loss = sum(value.sum() for value in collection.values)
                    loss.backward()
                    self.assertTrue(bool(torch.isfinite(q.grad).all()))
                    if mode.startswith("seed_"):
                        self.assertEqual(collection.names, ["native"])
                        torch.testing.assert_close(q.grad, 2*q.detach(), rtol=0., atol=0.)
                        problem.evaluate.assert_not_called()
                        scene.evaluate.assert_not_called()
                        scene.evaluate_source.assert_not_called()
                        scene.robot_coverage_guidance.assert_not_called()
                        anchor_fk.assert_not_called()
                    else:
                        self.assertIn("skill_process", collection.names)
                        self.assertIn("process_scene", collection.names)
                        problem.evaluate.assert_called_once()
                        anchor_fk.assert_called_once()
                        self.assertTrue(bool((q.grad > 2*q.detach()).all()))
                        if mode == "trajectory_optimizer":
                            # BEGIN the optimizer weights the same online residual.
                            scene.evaluate.assert_called_once()
                            scene.evaluate_source.assert_not_called()
                            # The online scene already owns self collision;
                            # no second source/proxy term is added to E(q).
                            regular = q.square().sum(-1).square()
                            expected = regular.clone()
                            expected[:, -1] += regular.sum(1)
                            scene_cost = collection.values[collection.names.index("process_scene")]
                            torch.testing.assert_close(scene_cost[..., 0],
                                expected*PlanningBudget().constraint_weight)
                        else:
                            # BEGIN success metrics use that same online residual.
                            scene.evaluate.assert_called_once()
                            scene.evaluate_source.assert_not_called()
                            # END reference seed modes above query neither scene callback.
                            scene.robot_coverage_guidance.assert_not_called()

    def test_reference_ik_still_supplies_same_clock_finger_geometry_to_native_costs(self):
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager

        for mode in ("seed_optimizer",):
            with self.subTest(mode=mode):
                scene = SimpleNamespace(gripper=SimpleNamespace(controller="timed_reference"))
                scene.clone_for_rollout = lambda: scene
                transition = self.transition()
                transition.joint_names = ("first_joint", "second_joint")
                # BEGIN this seed-only timing fixture has no functional inequalities.
                from manipulation.functional_augmented_lagrangian import FunctionalAugmentedLagrangian
                config_type = _process_manager_type(None, scene, None, None, None,
                    clock={"duration": 2.}, seed_parameter=torch.tensor([[.375]]),
                    functional_constraints=FunctionalAugmentedLagrangian(()))
                # END native finger-clock assertions still execute unchanged.
                config = config_type.create({"process_mode": mode}, device_cfg=self.device_cfg)
                manager = self.manager(config, transition)
                state = SimpleNamespace(joint_state=SimpleNamespace(position=torch.zeros((3, 1, 2))))
                timed_state, native_costs = object(), object()
                with patch("manipulation.planning._timed_native_state", return_value=timed_state) as timed, \
                     patch.object(RobotCostManager, "compute_costs", return_value=native_costs) as native:
                    self.assertIs(manager.compute_costs(state), native_costs)
                native.assert_called_once_with(timed_state, None, None)
                self.assertIs(timed.call_args.args[0], state)
                torch.testing.assert_close(timed.call_args.args[4], torch.full((3, 1), .75), rtol=0., atol=0.)
    # END reference success cannot bypass scene, process, or trajectory acceptance.
# END native manager configuration ownership without GPU allocation.


# BEGIN preserve complete source timelines under request-sized allocation bounds.
class RequiredAnchorAllocationTests(unittest.TestCase):
    @staticmethod
    def timeline_problem(counts=(252, 9, 60)):
        # These phase lengths match the existing 319-row placement input.
        # Analytic poses isolate source retention from robot/scene feasibility.
        phases = tuple(MotionPhase(f"phase_{index}", PosePath(
            np.linspace(0., 1., count), np.broadcast_to(np.eye(4), (count, 4, 4)),
            "complete supplied source timeline")) for index, count in enumerate(counts))
        events = (PlanningEvent("release", "current", 2., 259, "verification",
                                "original release boundary", "actual release feedback", True),)
        return cartesian_problem(phases, events)

    @staticmethod
    def clock(problem):
        return _progress_for_cubic_boundaries(
            _ProgressMap(np.array([0., 1.]), np.array([0., float(problem.phase_count)])),
            8, problem.start_dq, stop_progress=problem.stop_progress, anchors=problem.anchors)[0]

    @staticmethod
    def robot_metadata():
        return {"active_joint_names": tuple(f"q{index}" for index in range(16)),
                "links": 64, "spheres": 512}

    # BEGIN ordinary source density does not consume hard-query capacity.
    def test_complete_reference_timeline_keeps_only_real_query_anchors(self):
        problem = self.timeline_problem()
        self.assertEqual(sum(len(phase.reference.progress)-1 for phase in problem.phases)+1, 319)
        np.testing.assert_array_equal(problem.anchors, [0., 2., 3.])
        mapping = self.clock(problem)
        pairs = mapping.anchor_parameters(problem.anchors)
        self.assertEqual(len(pairs), 6)
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, PlanningBudget()), 256)
        exact = PlanningBudget(max_task_anchors=6)
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, exact), 6)
        with self.assertRaisesRegex(ValueError, "Task anchors"):
            _anchor_capacity(mapping, problem.anchors, PlanningBudget(max_task_anchors=5))
    # END reference vertices remain source data instead of required IK/validation poses.

    def test_true_scene_query_overflow_fails_before_native_construction(self):
        problem = self.timeline_problem()
        scene = SimpleNamespace(bind_robot_validator=Mock(), validate_problem=Mock(), required_progress_anchors=np.linspace(0., 3., 129),
                                reference_frame=problem.reference_frame, gripper=None)
        # BEGIN metadata validation precedes the separate allocation boundary.
        validator = SimpleNamespace(validate_planning_binding=Mock())
        # END this fixture still forbids native construction for excess anchors.
        with patch("manipulation.planning._bounded_robot_configuration") as robot, \
                patch("manipulation.planning._process_manager_type") as manager:
            with self.assertRaisesRegex(ValueError, "anchors|source knots"):
                _CuroboPlanner(problem, scene, object(), PlanningBudget(
                    max_validation_points=128, max_task_anchors=128), "cuda:0", robot_validator=validator)
        validator.validate_planning_binding.assert_called_once_with(problem, None)
        robot.assert_not_called()
        manager.assert_not_called()

    # BEGIN mandatory native midpoints must fit before any expensive initialization.

    # BEGIN source identity is checked before device allocation, against consumed XML.
    def test_source_configuration_rejects_changed_xml_overrides_and_unmodelled_locks(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        import xml.etree.ElementTree as ET

        original = '<robot name="source"><link name="base"/></robot>'
        frozen = ET.tostring(ET.fromstring(original), encoding="unicode")
        scenarios = (
            ("unchanged", {}, False, False, False, True),
            ("live_file_changed", {}, True, False, False, False),
            ("different_base", {"base_link": "other"}, False, False, False, False),
            ("extra_link", {"extra_links": {"tool": {}}}, False, False, False, False),
            ("object_link", {"add_object_link": True}, False, False, False, False),
            ("measured_finger", {"lock_joints": {"finger": .1}}, False, True, False, True),
            ("native_finger", {"lock_joints": {"finger": float(np.float32(.1))}}, False, True, False, True),
            ("changed_finger", {"lock_joints": {"finger": .11}}, False, True, False, False),
            ("locked_arm", {"lock_joints": {"arm": .1}}, False, True, False, False),
            ("finger_moves_tcp", {"lock_joints": {"finger": .1}}, False, True, True, False),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory)/"robot.urdf"
            for label, changes, changed_xml, has_hand, tool_descendant, admitted in scenarios:
                with self.subTest(case=label):
                    path.write_text(original.replace('name="source"', 'name="changed"') if changed_xml else original)
                    checker = SimpleNamespace(base_frame="base", model_record=lambda: {"urdf_xml": frozen},
                        _frame_id=lambda name: 0, model=SimpleNamespace(
                            frames=[SimpleNamespace(parentJoint=1 if tool_descendant else 0)],
                            names=["root", "finger"], parents=[0, 0]))
                    # BEGIN isolate XML/tree identity from calibrated metadata.
                    validator = SimpleNamespace(checker=checker, tool_frame="tool",
                        validate_planning_binding=Mock())
                    # END production metadata validation is invoked, not bypassed.
                    hand = (SimpleNamespace(controller="binary", joint_names=("finger",), current_positions=(.1,),
                        control_timestep_s=.01, bind_clock=Mock(), finger_trajectory=None) if has_hand else None)
                    problem = cartesian_problem((MotionPhase("free", constant_path()),))
                    # BEGIN geometry binding is explicit even in the CPU source-identity fixture.
                    scene = SimpleNamespace(bind_robot_validator=Mock(), bind_source_scene=Mock(), validate_problem=Mock(),
                        required_progress_anchors=(), gripper=hand, reference_frame=problem.reference_frame)
                    # END XML, joint-lock and ancestor checks below retain their original assertions.
                    robot = {"robot_cfg": {"kinematics": {"urdf_path": str(path), "base_link": "base", **changes}}}
                    planner = _CuroboPlanner.__new__(_CuroboPlanner)
                    planner.check_request = Mock()
                    with patch("torch.cuda._lazy_init", side_effect=AssertionError("source preflight must stay CPU-only")), \
                         patch("manipulation.planning._bounded_robot_configuration", return_value=(robot, {})), \
                         patch("manipulation.planning._planning_workspace_bytes", side_effect=RuntimeError("source admitted")) as storage:
                        with self.assertRaisesRegex(RuntimeError if admitted else ValueError,
                                "source admitted" if admitted else "source URDF|joint locks|ancestor"):
                            planner._initialize(problem, scene, robot, PlanningBudget(), "cuda:0", robot_validator=validator)
                    self.assertEqual(storage.call_count, int(admitted))
                    validator.validate_planning_binding.assert_called_once_with(problem, hand)
    # END a later edit of the original URDF cannot silently redefine consumed source geometry.

    # END neither impossible admission nor a later overflow weakens verification.

    def test_native_anchor_estimate_matches_installed_meta_buffer_layout(self):
        # Execute only the installed pure buffer factory on Torch's meta
        # device: no native library import, GPU context or physical allocation.
        import ast
        from importlib.util import find_spec
        from pathlib import Path
        source = Path(find_spec("curobo").origin).parent / "_src/curobolib/cuda_ops/kinematics.py"
        tree = ast.parse(source.read_text())
        definition = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                          and node.name == "KinematicsFusedFunction")
        factory = next(node for node in definition.body if isinstance(node, ast.FunctionDef)
                       and node.name == "create_buffers")
        factory.decorator_list = []
        module = ast.Module(body=[ast.ImportFrom(module="__future__",
                            names=[ast.alias(name="annotations")], level=0), factory], type_ignores=[])
        scope = {"torch": torch, "DeviceCfg": lambda: SimpleNamespace(device="meta", dtype=torch.float32)}
        exec(compile(ast.fix_missing_locations(module), str(source), "exec"), scope)
        metadata = self.robot_metadata()
        native = SimpleNamespace(num_pose_links=1, num_spheres=512, num_dof=16, num_links=64)
        scene = SimpleNamespace(required_progress_anchors=(), contacts=(), estimated_workspace_allocation=lambda batch, horizon, **kwargs:
                                {"gpu_retained": 0, "gpu_live": 0, "cpu_shared": 0, "cpu_live": 0, "gpu_shared": 0, "gpu_graph_live": 0})
        budget = PlanningBudget(max_validation_points=8192)
        before_cuda = torch.cuda.is_initialized()
        # BEGIN the anchor admission bound includes native self-cost buffers.
        # Construct actual native tensors on meta: this checks shapes/dtypes
        # without allocating their multi-GiB represented storage or running
        # CUDA. Use the smallest installed block capacity to verify the
        # conservative all-pairs bound used before native model construction.
        from curobo._src.cost.cost_self_collision import SelfCollisionCost
        from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg
        collision_layout = SelfCollisionKinematicsCfg()
        block_capacity = min(
            collision_layout._num_checks_per_thread_small_collision_pairs
            * collision_layout._max_threads_per_block_small_collision_pairs,
            collision_layout._num_checks_per_thread_large_collision_pairs
            * collision_layout._max_threads_per_block_large_collision_pairs)
        pair_count = native.num_spheres*(native.num_spheres-1)//2
        self_cost = SelfCollisionCost.__new__(SelfCollisionCost)
        self_cost.config = SimpleNamespace(store_pair_distance=False,
            self_collision_kin_config=SimpleNamespace(num_spheres=native.num_spheres,
                num_blocks_per_batch=(pair_count+block_capacity-1)//block_capacity))
        self_cost.device_cfg = SimpleNamespace(device="meta", dtype=torch.float32)
        self_cost._batch_size, self_cost._horizon = None, None
        # END the isolated setup executes the installed native allocator.
        for count in (322, 386, 8192):
            buffers = scope["create_buffers"](44, count, native)
            expected = sum(value.numel()*value.element_size() for value in buffers.values())
            # BEGIN include every anchor-count-dependent native collision array.
            self_cost.setup_batch_tensors(44, count)
            expected += sum(value.numel()*value.element_size() for name, value in vars(self_cost).items()
                            if isinstance(value, torch.Tensor) and name != "_pair_distance")
            # Disabling optional pair diagnostics leaves one fixed scratch
            # scalar, covered by the existing fixed native memory reserve.
            self.assertEqual(self_cost._pair_distance.numel(), 1)
            # END no real GPU allocation is needed to validate the admission bound.
            actual = _planning_workspace_bytes(scene, metadata, budget, count).anchor_fk_bytes
            self.assertEqual(actual, expected)
        self.assertGreater(actual, 6*1024**3)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)

    def test_torch_cache_covers_tensors_but_not_the_other_native_reserve(self):
        native_reserve = 2*1024**3
        cuda = SimpleNamespace(mem_get_info=Mock(return_value=(6*1024**3, 8*1024**3)),
                               memory_reserved=Mock(return_value=400), memory_allocated=Mock(return_value=101))
        runtime = SimpleNamespace(cuda=cuda)
        workspace = _PlanningWorkspace(0, 0, 200, 0, 600, False)
        # Existing cached pages count in device-used but can satisfy future
        # tensors. One byte less reuse must fail the actual physical limit.
        # BEGIN shared allocation admission reports the whole-device boundary.
        with self.assertRaisesRegex(ValueError, "gpu_device_used"):
            _require_planning_memory(runtime, "cuda:0", workspace,
                                     max_device_used_bytes=4*1024**3+500)
        cuda.memory_allocated.return_value = 100
        self.assertIs(_require_planning_memory(runtime, "cuda:0", workspace,
                                               max_device_used_bytes=4*1024**3+500), workspace)
        cuda.mem_get_info.return_value = native_reserve-1, 12*1024**3
        cuda.memory_reserved.return_value = 10*1024**3
        with self.assertRaisesRegex(ValueError, "gpu_native_reserve"):
            _require_planning_memory(runtime, "cuda:0", workspace, max_device_used_bytes=8*1024**3)
        # END native storage never borrows reserved Torch pages.


    # BEGIN refinement admission cannot delete old witnesses or destroy a usable solver.
    # END a caller cannot replace the old witness with a new one to fit the budget.

    def test_close_releases_both_native_cycles_and_remains_idempotent(self):
        import gc
        import weakref
        destroyed = []

        class CyclicSolver:
            def __init__(self, name):
                self.name = name
                self.core = SimpleNamespace(callback=self.destroy)

            def destroy(self):
                destroyed.append(self.name)

        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.physical_domain = SimpleNamespace(initialize=lambda values: values, validate=lambda values: None)
        planner.ik, planner.trajopt = CyclicSolver("ik"), CyclicSolver("trajectory")
        references = (weakref.ref(planner.ik), weakref.ref(planner.trajopt))
        with patch("manipulation.planning.gc.collect", wraps=gc.collect) as collect:
            planner.close()
            self.assertEqual(destroyed, ["ik", "trajectory"])
            self.assertTrue(all(reference() is None for reference in references))
            self.assertIsNone(planner.ik)
            self.assertIsNone(planner.trajopt)
            collect.assert_called_once_with()
            planner.close()
        self.assertEqual(destroyed, ["ik", "trajectory"])

    def test_scene_budget_and_removed_legacy_parameter_remain_strict(self):
        # BEGIN explicit TSDF storage capacity does not weaken workspace admission.
        self.assertEqual(PlanningBudget().max_scene_gpu_workspace_bytes, 1024**3)
        self.assertEqual(PlanningBudget(max_scene_gpu_workspace_bytes=6*1024**3).max_scene_gpu_workspace_bytes,
                         6*1024**3)
        with self.assertRaisesRegex(ValueError, "six-GiB allocation budget"):
            PlanningBudget(max_scene_gpu_workspace_bytes=6*1024**3+1)
        # END requests above the local ceiling still fail before allocations.
        scene = SimpleNamespace(required_progress_anchors=(), estimated_workspace_allocation=lambda batch, horizon, **kwargs:
                                {"gpu_retained": 1024**3, "gpu_live": 0, "cpu_shared": 0, "cpu_live": 0, "gpu_shared": 0, "gpu_graph_live": 0})
        with self.assertRaisesRegex(ValueError, "scene query workspaces"):
            _planning_workspace_bytes(scene, self.robot_metadata(), PlanningBudget(), 322)
        with self.assertRaises(TypeError):
            PlanningBudget(max_anchors=128)
        with self.assertRaises(TypeError):
            PlanningBudget(max_refinement_points=1)

    # BEGIN a large finite scene must fit by lifetime, without dropping queries.
    def test_graph_admission_preserves_eager_budget_and_counts_shared_grids_once(self):
        # BEGIN graph pools exclude CPU source validation, even in guidance rows.
        # Distinct full-live/captured peaks catch accidentally duplicating CPU
        # query/packing/dedup scratch in the two CUDA pools. Immutable grids
        # remain shared across both graph pools and all independent scenes.
        estimator = Mock(side_effect=lambda batch, horizon, *, robot_coverage_horizon:
            {"gpu_retained": 5, "gpu_live": 100 if robot_coverage_horizon else 150,
             "cpu_shared": 17, "cpu_live": 31, "gpu_shared": 23,
             "gpu_graph_live": 60 if robot_coverage_horizon else 90})
        scene = SimpleNamespace(required_progress_anchors=(), contacts=(),
                                estimated_workspace_allocation=estimator)
        metadata = self.robot_metadata()
        eager_bytes = 13*5+150+23
        graph_bytes = eager_bytes+2*60
        # END the ordinary eager bound still includes the complete 150-byte peak.
        for capacity, expected_bytes, expected_graph in (
                (eager_bytes, eager_bytes, False),
                (graph_bytes-1, eager_bytes, False),
                (graph_bytes, eager_bytes, False)):
            with self.subTest(capacity=capacity):
                estimator.reset_mock()
                actual = _planning_workspace_bytes(scene, metadata,
                    PlanningBudget(max_scene_gpu_workspace_bytes=capacity), 1)
                self.assertEqual((actual.gpu_bytes, actual.trajectory_graph), (expected_bytes, expected_graph))
                self.assertEqual(actual.cpu_bytes, 17+31)
                self.assertEqual(estimator.call_count, 5)
        scene.contacts = (object(),)
        actual = _planning_workspace_bytes(scene, metadata,
            PlanningBudget(max_scene_gpu_workspace_bytes=graph_bytes), 1)
        self.assertEqual((actual.gpu_bytes, actual.trajectory_graph), (eager_bytes, False))
        with self.assertRaisesRegex(ValueError, "required .* allowed"):
            _planning_workspace_bytes(scene, metadata,
                PlanningBudget(max_scene_gpu_workspace_bytes=eager_bytes-1), 1)

    def test_scene_allocation_keeps_all_caches_and_only_one_sequential_live_graph(self):
        from dataclasses import replace
        from manipulation.planning_scene import PlanningScene
        world = PlanningScene.__new__(PlanningScene)
        world.required_progress_anchors = (0., 1.)
        world.robot_sphere_materials = (None,)*416
        # BEGIN this admission fixture declares the absence of a frozen source allocation.
        world.robot_validator = None
        world._source_scene_checker = None
        world.robot_collision_bodies = ()
        # END actual source snapshots are accounted by the same method when bound.
        world.background = None
        world.owners = tuple(SimpleNamespace(moving=False, owner_id=f"fixed-{i}", query=None) for i in range(4))
        world.coverage = SimpleNamespace(_distance=SimpleNamespace(numel=lambda: 503840))
        # BEGIN this metadata-only hand owns no timed polynomial cache.
        world.gripper = SimpleNamespace(self_collision_pairs=range(28672),
                                        finger_trajectory=None, nominal_finger_trajectory=None)
        # END real hand references are separately included in retained bytes.
        world.robot_coverage = SimpleNamespace(required_link_names=range(13), baseline_joint_names=range(9))
        # Broadcast views supply metadata only: no native buffers, tree, large
        # sampled cloud or GPU context is constructed for this admission test.
        planes = (200, 3174, 300, 300, 300, 300, 200, 200, 200, 32, 12, 32, 12)
        world._coverage_geometry = {"solids": [{"baseline_halfspaces_world": np.broadcast_to(0., (n, 4))}
                                                for n in planes]}
        world._free_distance = world.coverage._distance
        world._free_mask = world.coverage._distance
        world.contacts = tuple(SimpleNamespace(_feature_edge_normals=np.zeros((4, 3))) for _ in range(2))
        # BEGIN metadata implements the contact model's explicit storage/workspace interface.
        class ContactStorage(SimpleNamespace):
            @property
            def storage_bytes(self):
                return self.centres.nbytes+self.halves.nbytes+self.rotations.nbytes

            def workspace_bytes(self):
                return self.pair_chunk*1600

        cells = ContactStorage(pair_chunk=32768, centres=np.empty((0, 3)), halves=np.empty((0, 3)),
                               rotations=np.empty((0, 3, 3)))
        # END changing only complete component inventory still changes shared bytes exactly once.
        world._contact_cells = {"target": cells}
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1, knots=16, interpolation_steps=4,
            interpolation_buffer_size=768, max_task_anchors=256, max_validation_points=512)
        metadata = {"active_joint_names": range(7), "links": 13, "spheres": 416}
        cuda_before = torch.cuda.is_initialized()
        empty = _planning_workspace_bytes(world, metadata, budget, 22)
        cells.centres = np.broadcast_to(0., (250000, 3))
        cells.halves = np.broadcast_to(0., (250000, 3))
        cells.rotations = np.broadcast_to(0., (250000, 3, 3))
        with patch.object(world, "estimated_workspace_allocation", wraps=world.estimated_workspace_allocation) as estimate:
            complete = _planning_workspace_bytes(world, metadata, budget, 22)
        self.assertEqual(complete.cpu_bytes-empty.cpu_bytes, 250000*15*8)
        self.assertEqual(complete.gpu_bytes, empty.gpu_bytes)
        self.assertEqual(complete.trajectory_graph, empty.trajectory_graph)
        self.assertLess(complete.gpu_bytes, budget.max_scene_gpu_workspace_bytes)
        self.assertTrue(any(call.args == (4, 3*(81+22))
                            and call.kwargs["robot_coverage_horizon"] == 81+22
                            for call in estimate.call_args_list))
        with self.assertRaisesRegex(ValueError, "required .* allowed"):
            _planning_workspace_bytes(world, metadata,
                replace(budget, max_scene_gpu_workspace_bytes=complete.gpu_eager_bytes-1), 22)
        # BEGIN complete validation must fit both planner and native-query guards.
        from manipulation.planning_scene import CuroboSphereQuery
        cells.centres = np.broadcast_to(0., (45655, 3))
        cells.halves = np.broadcast_to(0., (45655, 3))
        cells.rotations = np.broadcast_to(0., (45655, 3, 3))
        expanded = replace(budget, max_validation_points=1536, max_scene_gpu_workspace_bytes=2*1024**3)
        complete = _planning_workspace_bytes(world, metadata, expanded, 20)
        # BEGIN total validation work does not enlarge live geometry batches.
        large_work = replace(expanded, max_validation_points=32768)
        self.assertEqual(complete, _planning_workspace_bytes(world, metadata, large_work, 20))
        self.assertLess(complete.gpu_bytes, expanded.max_scene_gpu_workspace_bytes)
        # BEGIN shared source storage and one serial query workspace are each counted once.
        world._source_scene_checker = SimpleNamespace(scene_storage_nbytes=131072,
                                                       scene_query_workspace_nbytes=98304)
        # BEGIN the bounded ordinary/dense report store is live once per candidate.
        admitted = _planning_workspace_bytes(world, metadata, expanded, 20)
        self.assertEqual(admitted.cpu_bytes-complete.cpu_bytes,
                         131072+98304+32*1024**2)
        self.assertEqual(admitted.gpu_bytes, complete.gpu_bytes)
        # END it is neither a per-rollout nor a captured-graph allocation.
        # END rollout counts multiply neither the immutable source scene nor its live CPU query.
        query = CuroboSphereQuery(SimpleNamespace(get_sphere_distance_raw=lambda *args, **kwargs: None),
                                 owner_ids=("fixture",), source="complete validation allocation metadata")
        native_spheres = 3*256*metadata["spheres"]
        self.assertLessEqual(native_spheres, query.max_query_spheres)
        self.assertLessEqual(native_spheres*20+8, query.max_cache_bytes)
        # BEGIN complete refined coverage still fits without dropping samples.
        maximum_robot_query = 3*256*expanded.max_robot_spheres
        self.assertLessEqual(maximum_robot_query, query.max_query_spheres)
        self.assertLessEqual(query.max_query_spheres*20+8, query.max_cache_bytes)
        self.assertGreater(3*3072*metadata["spheres"], query.max_query_spheres)
        # END all 32768 finite queries are retained across bounded chunks.
        # END both the sphere bound and byte bound remain finite and enforced.
        # END additional samples are admitted intact, never thinned to fit cache.
        self.assertEqual(torch.cuda.is_initialized(), cuda_before)
    # END the original three mode queries and every finite-cell pair remain.

    def test_projection_accuracy_respects_the_strictest_source_mechanism(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-6)
        model = SimpleNamespace(position_tolerance=1e-6, rotation_tolerance=2e-6)
        problem.mechanisms = (SimpleNamespace(model=model),)
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-7)
        model.rotation_tolerance = 1e-8
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-9)
# END preserve complete source timelines under request-sized allocation bounds.


class RobotAllocationPreflightTests(unittest.TestCase):
    """Inspect installed metadata only; native model constructors remain unused."""

    @classmethod
    def setUpClass(cls):
        from curobo.content import get_robot_configs_path
        from curobo._src.util.config_io import load_yaml, join_path
        cls.robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
        kinematics = cls.robot["robot_cfg"]["kinematics"]
        locked = set(kinematics["lock_joints"])
        cls.problem = SimpleNamespace(tool_link="panda_hand", joint_names=tuple(
            name for name in kinematics["cspace"]["joint_names"] if name not in locked))
        # This exact installed fixture includes inactive attachment slots;
        # omitting those slots undercounts the native S-by-S allocation.
        cls.scene = SimpleNamespace(robot_sphere_materials=(None,)*65, required_robot_links=())

    def test_installed_robot_metadata_is_valid_without_initializing_cuda(self):
        initialized = torch.cuda.is_initialized()
        with patch("torch.cuda.init", side_effect=AssertionError("metadata preflight must remain CPU only")):
            config, info = _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget())
        self.assertEqual(info["unlocked_dof"], 9)
        self.assertEqual(len(info["active_joint_names"]), 7)
        self.assertEqual(info["links"], 13)
        self.assertEqual(info["spheres"], 65)
        self.assertEqual(config["robot_cfg"]["kinematics"]["tool_frames"], ["panda_hand"])
        self.assertEqual(torch.cuda.is_initialized(), initialized)

    def test_oversized_attachment_slots_are_rejected_before_urdf_or_native_allocation(self):
        robot = deepcopy(self.robot)
        robot["robot_cfg"]["kinematics"]["extra_collision_spheres"]["attached_object"] = 10**7
        with patch("yourdfpy.URDF.load", side_effect=AssertionError("count guard must run before model parsing")):
            with self.assertRaisesRegex(ValueError, "sphere budget"):
                _bounded_robot_configuration(robot, self.problem, self.scene, PlanningBudget())

    def test_active_joint_and_material_identity_mismatches_fail_preflight(self):
        wrong_joints = SimpleNamespace(tool_link="panda_hand", joint_names=self.problem.joint_names[:-1])
        with self.assertRaisesRegex(ValueError, "active joints"):
            _bounded_robot_configuration(self.robot, wrong_joints, self.scene, PlanningBudget())
        wrong_materials = SimpleNamespace(robot_sphere_materials=(None,)*64, required_robot_links=())
        with self.assertRaisesRegex(ValueError, "complete native sphere layout"):
            _bounded_robot_configuration(self.robot, self.problem, wrong_materials, PlanningBudget())

    def test_occupied_geometry_cannot_request_links_outside_the_bounded_robot_tree(self):
        scene = SimpleNamespace(robot_sphere_materials=(None,)*65,
                                required_robot_links=("unobserved_extra_camera_tree",))
        with self.assertRaisesRegex(ValueError, "link|tree"):
            _bounded_robot_configuration(self.robot, self.problem, scene, PlanningBudget())

    def test_limits_cover_the_unlocked_intermediate_tree_and_all_selected_links(self):
        # Seven final active arm joints do not make the native nine-joint
        # model used during lock processing safe under an eight-DOF budget.
        with self.assertRaisesRegex(ValueError, "joint|DOF"):
            _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget(max_robot_dof=8))
        with self.assertRaisesRegex(ValueError, "link"):
            _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget(max_robot_links=12))


# BEGIN independent native-result selection and continuation regressions.
class PlannerCandidateSelectionTests(unittest.TestCase):
    """Use real CPU native results so seed/time reshaping cannot be mocked away."""

    @staticmethod
    def native_candidates(flags=(False, True), native_violations=(.02, 0.)):
        from curobo.types import DeviceCfg, JointState
        from curobo._src.rollout.metrics import CostCollection, CostsAndConstraints, RolloutMetrics
        from curobo._src.solver.solver_trajopt_result import TrajOptSolverResult

        # The native failed-seed cost loses small differences beside 1e16 in
        # float32. Actual process violations must determine continuation.
        shape = (1, 2, 9, 3)
        position = torch.stack((torch.ones((9, 3)), 2*torch.ones((9, 3))))[None]
        position[:, :, 0] = 0.
        position[:, 1, 5:] = 1000.
        state = JointState(position=position, velocity=torch.zeros(shape),
                           acceleration=torch.zeros(shape), jerk=torch.zeros(shape),
                           dt=torch.tensor([[.07, .11]]), joint_names=["x", "y", "z"],
                           device_cfg=DeviceCfg("cpu"))
        solution = torch.stack((torch.ones((4, 3)), 2*torch.ones((4, 3))))[None]
        constraints = CostCollection(
            values=[torch.tensor(native_violations)[:, None, None].expand(2, 9, 1).clone()],
            names=["cspace"])
        metrics = RolloutMetrics(costs_and_constraints=CostsAndConstraints(constraints=constraints))
        return TrajOptSolverResult(
            success=torch.tensor([flags]), solution=solution, optimized_seeds=solution.clone(),
            js_solution=state.clone(), interpolated_trajectory=state,
            interpolated_last_tstep=torch.tensor([[9, 5]]),
            metrics=metrics, interpolated_metrics=None,
            position_error=torch.zeros((1, 2)), rotation_error=torch.zeros((1, 2)),
            goalset_index=torch.zeros((1, 2, 1), dtype=torch.long),
            seed_cost=torch.tensor([[1e16, 1e16]], dtype=torch.float32),
            total_cost_reshaped=torch.tensor([[1e16, 1e16]], dtype=torch.float32),
            seed_rank=torch.tensor([[0, 1]]), batch_size=1, num_seeds=2)


    def test_native_single_seed_gather_preserves_clock_valid_length_and_original_rank(self):
        from manipulation.planning import _trajectory_candidate

        result = self.native_candidates()
        original_rank = result.seed_rank.clone()
        for index in (1, 0):
            selected = _trajectory_candidate(result, index)
            self.assertEqual((selected.batch_size, selected.num_seeds), (1, 1))
            self.assertEqual(tuple(selected.solution.shape), (1, 1, 4, 3))
            torch.testing.assert_close(selected.solution, result.solution[:, index:index+1])
            curve = selected.get_interpolated_plan()
            self.assertEqual(tuple(curve.position.shape), (1, 1, (9, 5)[index], 3))
            self.assertLess(float(curve.position.max()), 3.)
            np.testing.assert_allclose(curve.dt.numpy(), [[(.07, .11)[index]]], rtol=1e-7)
            torch.testing.assert_close(result.seed_rank, original_rank, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            _trajectory_candidate(result, 2)

    # BEGIN locked-joint native result layout: active knots and full robot state.
    def test_locked_finger_state_keeps_seven_control_dofs_and_nine_state_dofs_separate(self):
        from curobo.types import DeviceCfg, JointState
        from manipulation.planning import _trajectory_candidate

        for interpolated_has_knots in (False, True):
            with self.subTest(interpolated_has_knots=interpolated_has_knots):
                result = self.native_candidates()
                controls = torch.stack((torch.ones((12, 7)), 2*torch.ones((12, 7))))[None]
                result.solution, result.optimized_seeds = controls, controls.clone()
                # The installed V2 solver restores locked fingers in sampled
                # states, then stores its seven active action controls in the
                # same JointState.knot field. Its generic gather incorrectly
                # assumes these two last dimensions are always identical.
                position = torch.stack((torch.ones((9, 9)), 2*torch.ones((9, 9))))[None]
                position[:, 1, 5:, :7] = 1000.
                position[..., 7:] = .04
                state = JointState(
                    position=position, velocity=torch.zeros_like(position),
                    acceleration=torch.zeros_like(position), jerk=torch.zeros_like(position),
                    dt=torch.tensor([[.07, .11]]), knot=controls, knot_dt=torch.tensor([[.28, .44]]),
                    joint_names=[*(f"arm_{index}" for index in range(7)), "finger_left", "finger_right"],
                    device_cfg=DeviceCfg("cpu"))
                result.js_solution = state.clone()
                result.interpolated_trajectory = state.clone()
                if not interpolated_has_knots:
                    result.interpolated_trajectory.knot = None
                    result.interpolated_trajectory.knot_dt = None
                selected = _trajectory_candidate(result, 1)
                self.assertEqual(tuple(selected.solution.shape), (1, 1, 12, 7))
                self.assertEqual(tuple(selected.optimized_seeds.shape), (1, 1, 12, 7))
                self.assertEqual(tuple(selected.js_solution.position.shape), (1, 1, 9, 9))
                self.assertEqual(tuple(selected.js_solution.knot.shape), (1, 1, 12, 7))
                torch.testing.assert_close(selected.js_solution.knot, controls[:, 1:2], rtol=0, atol=0)
                torch.testing.assert_close(selected.js_solution.knot_dt, torch.tensor([[.44]]), rtol=0, atol=0)
                curve = selected.get_interpolated_plan()
                self.assertEqual(tuple(curve.position.shape), (1, 1, 5, 9))
                torch.testing.assert_close(curve.position[..., 7:], torch.full((1, 1, 5, 2), .04), rtol=0, atol=0)
                self.assertLess(float(curve.position.max()), 3.)
                if interpolated_has_knots:
                    torch.testing.assert_close(selected.interpolated_trajectory.knot, controls[:, 1:2], rtol=0, atol=0)
                else:
                    self.assertIsNone(selected.interpolated_trajectory.knot)
                # Extraction must leave the original all-seed state intact.
                self.assertEqual(tuple(result.js_solution.knot.shape), (1, 2, 12, 7))
                self.assertEqual(tuple(result.js_solution.position.shape), (1, 2, 9, 9))
                torch.testing.assert_close(result.js_solution.knot, controls, rtol=0, atol=0)
    # END locked-joint active/full dimension regression.


    # BEGIN immediate feasibility archival and phase-independent quality ranking.

    # END hard acceptance is required before either phase can select a result.




    # BEGIN global budget failure must keep its identity and stop seed work.
    # END an admitted alternative never replenishes the request budget.
# END independent native-result selection and continuation regressions.


# BEGIN exact knot-side, barrier and named-bound acceleration regressions.
# BEGIN candidate ranking cannot trigger independent per-seed geometry exports.
def test_native_selection_uses_shared_residuals_and_exports_no_rejected_seed():
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.check_request = Mock()
    planner.seed_strategies = ["reference", "alternate"]
    planner.last_optimizer_diagnostics = []
    planner._native_curve = Mock(return_value=object())
    planner.export = Mock(side_effect=AssertionError("Selection must not run output delivery"))
    candidates = PlannerCandidateSelectionTests.native_candidates()
    _, curve, report = planner.select_candidate(candidates)
    assert report["accepted"] and report["selected_seed_index"] == 1
    assert [row["accepted"] for row in report["candidates"]] == [False, True]
    planner._native_curve.assert_called_once()
    planner.export.assert_not_called()
# END only the selected native candidate reaches the later delivery boundary.


class ContinuousCubicDerivativeTests(unittest.TestCase):
    @staticmethod
    def curve():
        # The moving coordinate is t**3 on [0,1], then 1+3*(t-1).
        # All stored-knot right accelerations are zero, while the left
        # acceleration at t=1 is six. q/dq are continuous at that same knot.
        return JointTrajectory(("stationary", "moving"), np.array([0., 1., 2.]),
            np.array([[0., 0.], [0., 1.], [0., 4.]]),
            np.array([[0., 0.], [0., 3.], [0., 3.]]),
            np.array([0., 1.]), np.array([0., 1.]), (), (), {})

    def test_left_acceleration_limit_and_reordered_asymmetric_bounds(self):
        curve = self.curve()
        np.testing.assert_array_equal(curve._curve(curve.times, 2), np.zeros((3, 2)))
        report = cubic_derivative_report(curve, [[-2., -.1], [5., .1]],
                                         joint_names=("moving", "stationary"))
        self.assertEqual(report["joint_names"], ["stationary", "moving"])
        self.assertEqual(report["acceleration_bounds"], [[-.1, -2.], [.1, 5.]])
        self.assertEqual(report["acceleration"]["maximum"], [0., 6.])
        self.assertEqual(report["acceleration"]["upper_excess"], [0., 1.])
        self.assertEqual(report["acceleration"]["maximum_violation"], {
            "joint_name": "moving", "time_s": 1., "segment_index": 0,
            "side": "segment_end_left_limit", "acceleration": 6., "bound": 5., "excess": 1.})

    def test_prefix_clips_inside_piece_and_keeps_left_side_at_exact_barrier(self):
        curve = self.curve()
        arguments = dict(acceleration_bounds=[[-2., -.1], [5., .1]],
                         joint_names=("moving", "stationary"))
        partial = cubic_derivative_report(curve, until=.5, **arguments)
        self.assertEqual(partial["checked_interval_s"], [0., .5])
        self.assertEqual(partial["acceleration"]["maximum"], [0., 3.])
        self.assertTrue(partial["acceleration"]["bounds_satisfied"])
        exact = cubic_derivative_report(curve, until=1., **arguments)
        self.assertEqual(exact["checked_piece_count"], 1)
        self.assertEqual(exact["acceleration"]["maximum"], [0., 6.])
        self.assertEqual(exact["continuity"]["internal_breakpoints"], 0)
        self.assertFalse(exact["acceleration"]["bounds_satisfied"])
        empty = cubic_derivative_report(curve, until=0., **arguments)
        self.assertEqual(empty["acceleration"]["maximum"], [0., 0.])

    def test_c1_curve_with_acceleration_jump_is_accepted_inside_existing_bounds(self):
        report = cubic_derivative_report(self.curve(), [[-8., -.1], [8., .1]],
                                         joint_names=("moving", "stationary"))
        self.assertTrue(report["acceleration"]["bounds_satisfied"])
        self.assertTrue(report["continuity"]["c1_by_hermite_construction"])
        self.assertFalse(report["continuity"]["acceleration_jump_is_rejection"])
        self.assertEqual(report["continuity"]["position_maximum_absolute_jump"], [0., 0.])
        self.assertEqual(report["continuity"]["velocity_maximum_absolute_jump"], [0., 0.])
        self.assertEqual(report["continuity"]["acceleration_maximum_absolute_jump"], [0., 6.])

    def test_negative_acceleration_uses_lower_bound_and_curve_drift_is_rejected(self):
        curve = self.curve()
        curve.position *= -1.
        curve.velocity *= -1.
        curve.__post_init__()
        report = cubic_derivative_report(curve, [[-.1, -5.], [.1, 20.]],
                                         joint_names=curve.joint_names)
        self.assertEqual(report["acceleration"]["lower_excess"], [0., 1.])
        self.assertEqual(report["acceleration"]["maximum_violation"]["acceleration"], -6.)
        curve.position[1, 1] += .1
        with self.assertRaisesRegex(ValueError, "differ from.*polynomial"):
            cubic_derivative_report(curve, [[-.1, -5.], [.1, 20.]], joint_names=curve.joint_names)

    def test_missing_duplicate_and_nonfinite_effective_bounds_are_rejected(self):
        for names, bounds in ((("moving",), [[-5.], [5.]]),
                              (("moving", "moving"), [[-5., -5.], [5., 5.]]),
                              (("stationary", "moving"), [[-5., -5.], [5., np.inf]])):
            with self.subTest(names=names, bounds=bounds), self.assertRaisesRegex(ValueError, "named.*bounds"):
                cubic_derivative_report(self.curve(), bounds, joint_names=names)
# END acceleration jumps remain diagnostic, with no new C2 or jerk-jump gate.


class ExportedCurveContractTests(unittest.TestCase):
    # BEGIN observer failures cannot alter checks or replace the budget error.

    # END evidence collection cannot revoke a numerically unchanged candidate.
    # BEGIN exported source checks preserve the exact native query precision and failure time.
    # END a source collision cannot disappear behind passing native joint constraints.

    # BEGIN the real exporter consumes the complete scalar acceleration check.
    def test_analytic_acceleration_violation_enters_existing_candidate_residuals(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            [[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]],
            [[0., 0., 0.], [3., 0., 0.], [3., 0., 0.]], 1.)
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        limits.acceleration[:, 0] = torch.tensor([-2., 5.], dtype=torch.float64)
        from manipulation.planning import MotionPlanningError
        with self.assertRaises(MotionPlanningError) as caught:
            planner.export(result, {"accepted": True, "residuals": {}})
        self.assertEqual(caught.exception.reason, "output_inconsistent")
    # END a zero sampled native residual cannot hide a left-limit violation.

    def test_multiple_native_seeds_cannot_be_flattened_into_one_export(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((3, 3)), np.zeros((3, 3)), .25)
        result.num_seeds = 2
        result.get_interpolated_plan = Mock(side_effect=AssertionError("must reject before reading mixed curves"))
        with self.assertRaisesRegex(ValueError, "one.*seed|selected seed"):
            planner.export(result, {"accepted": True, "residuals": {}})
        result.get_interpolated_plan.assert_not_called()

    # BEGIN changing ordinary reference shape cannot invent a process obligation.
    def test_reference_short_loop_does_not_require_motion_or_a_hard_anchor(self):
        knots = np.array([0., .501, .502, .503, 1.])
        reference = np.broadcast_to(np.eye(4), (len(knots), 4, 4)).copy()
        reference[:, 0, 3] = [0., 0., .02, 0., 0.]
        problem = cartesian_problem((MotionPhase("reference loop", PosePath(knots, reference, "reference")),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((9, 3)), np.zeros((9, 3)), .125)
        trajectory, _ = planner.export(result, {"accepted": True, "residuals": {}})
        self.assertNotIn(.502, planner.last_validation["u"])
        self.assertTrue(all(value == 0. for value in trajectory.residuals.values()))
        self.assertFalse(any(key.startswith("phase:") for key in trajectory.residuals))
    # END independent PoseProcessRequirement tests cover truly required paths.

    def test_plateau_has_two_validation_anchors_but_one_original_event_trigger(self):
        phases = tuple(MotionPhase(f"free {index}", constant_path()) for index in range(3))
        events = (
            PlanningEvent("close", "first", 1., 73, "close", "original first close"),
            PlanningEvent("verify", "first", 1., 79, "verification", "first grasp boundary",
                          "actual stable Holding feedback", True),
            PlanningEvent("close-again", "first", 2., 115, "close", "original second close"),
            PlanningEvent("verify-again", "first", 2., 121, "verification", "second grasp boundary",
                          "another actual Holding observation", True),
        )
        problem = cartesian_problem(phases, events)
        mapping = _ProgressMap(np.array([0., .2, .45, .75, 1.]), np.array([0., 1., 1., 2., 3.]))
        np.testing.assert_allclose(mapping.anchor_parameters([1.]), [[.2, 1.], [.45, 1.]])
        planner, result = export_fixture(problem, mapping, np.zeros((7, 3)), np.zeros((7, 3)), .2)
        trajectory, _ = planner.export(result, {"accepted": True, "residuals": {}})
        self.assertEqual(len(trajectory.events), len(events))
        self.assertEqual([(event["skill_id"], event["event_id"]) for event in trajectory.events],
                         [(event.skill_id, event.event_id) for event in events])
        first = trajectory.events[0]
        self.assertAlmostEqual(first["time"], .2*trajectory.duration)
        self.assertAlmostEqual(first["hold_until"], .45*trajectory.duration)
        self.assertAlmostEqual(trajectory.executable_until, first["time"])
        self.assertEqual(trajectory.next_verification["source_frame"], 79)
        # Returned barrier details cannot mutate the trajectory's authorization
        # boundary or turn a future verification into supplied evidence.
        returned = trajectory.next_verification
        returned["time"] = trajectory.duration
        self.assertAlmostEqual(trajectory.executable_until, first["time"])

    def test_no_evidence_barrier_allows_the_complete_exported_duration(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((3, 3)), np.zeros((3, 3)), .25)
        trajectory, _ = planner.export(result, {"accepted": True, "residuals": {}})
        self.assertIsNone(trajectory.next_verification)
        self.assertEqual(trajectory.executable_until, trajectory.duration)



# BEGIN measured-entrance necessary conditions without invented derivatives.
class InitialStatePreflightTests(unittest.TestCase):
    """CPU integration boundaries, without claiming native collision numerics."""

    @staticmethod
    def fixture(*, position=(.1, -.3, .5), velocity=(.7, -1.7, 2.4),
                native_collision=0., scene_name="source_world_collision", scene_value=0.):
        base = np.eye(4)
        base[:3, :3] = Rotation.from_euler("z", .8).as_matrix()
        base[:3, 3] = [.8, -.6, .2]
        tool = np.eye(4)
        tool[:3, 3] = [.02, -.03, .04]
        problem = PlanningProblem(
            joint_names=("cartesian_x", "cartesian_y", "cartesian_z"),
            start_q=np.asarray(position), start_dq=np.asarray(velocity),
            tool_link="actual_tool", reference_frame="analytic_world",
            base_pose=base, tool_transform=tool,
            phases=(MotionPhase("free", constant_path()),))
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.check_request = Mock()
        planner.physical_domain = SimpleNamespace(initialize=lambda values: values, validate=lambda values: None)
        planner.problem, planner.joint_names = problem, problem.joint_names
        # BEGIN entrance source checks receive the same explicit progress/time mapping.
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        # END this fixture has no material or attachment mode switches.
        planner.budget = PlanningBudget()
        planner.torch, planner.JointState = torch, SimpleNamespace
        planner._RobotState = lambda **kwargs: SimpleNamespace(**kwargs, robot_spheres=kwargs["cuda_robot_model_state"].robot_spheres)
        planner.start = SimpleNamespace(
            position=torch.tensor([position], dtype=torch.float64),
            velocity=torch.tensor([velocity], dtype=torch.float64))
        # Native limits deliberately have a different name order. Using raw
        # columns would accept/reject different physical axes in these cases.
        limits = SimpleNamespace(
            joint_names=["cartesian_z", "cartesian_x", "cartesian_y"],
            position=torch.tensor([[-.6, -.2, -.4], [.6, .2, .4]], dtype=torch.float64),
            velocity=torch.tensor([[-3., -1., -2.], [3., 1., 2.]], dtype=torch.float64))
        cspace = SimpleNamespace(enabled=True, config=SimpleNamespace(joint_limits=limits),
            forward=Mock(side_effect=AssertionError("unknown high derivatives cannot enter STATE cost")))
        collision = SimpleNamespace(enabled=True, config=SimpleNamespace(self_collision_kin_config=SimpleNamespace(sphere_padding=0.)), setup_batch_tensors=Mock(),
            forward=Mock(return_value=torch.tensor([[native_collision]], dtype=torch.float64)))
        planner._verification_native = SimpleNamespace(
            get_cost={"cspace": cspace, "self_collision": collision}.get,
            compute_costs=Mock(side_effect=AssertionError("complete manager would invent missing derivatives")))

        def fk(state):
            q = state.position
            quaternion = q.new_zeros((1, 1, 2, 4))
            quaternion[..., 0] = 1.
            spheres = torch.cat((q[:, :, None], q.new_full((1, 1, 1, 1), .01)), dim=-1)
            return SimpleNamespace(robot_spheres=spheres, tool_poses=SimpleNamespace(
                tool_frames=["unrelated_tool", problem.tool_link],
                position=torch.stack((q+7., q), dim=2), quaternion=quaternion))

        planner._verification_fk = SimpleNamespace(compute_kinematics=Mock(side_effect=fk))
        # BEGIN independently named source FK supplies the same analytic model.
        def source_fk(*, joint_positions, **kwargs):
            link = np.eye(4)
            link[:3, 3] = joint_positions
            return {"frame_poses": {problem.tool_link: base @ link}}

        planner.robot_validator = SimpleNamespace(joint_names=problem.joint_names,
            tool_frame=problem.tool_link, T_reference_base=base, T_frame_tcp=tool,
            _planning_positions=lambda q, names, **kwargs: (np.asarray(q), ()),
            checker=SimpleNamespace(query=Mock(side_effect=source_fk)),
            _pose_difference=lambda actual, expected: {
                "translation_m": float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                "rotation_rad": float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())})
        # END this diagnostic compares identical q without relaxing any gate.
        planner.scene = SimpleNamespace(gripper=None, required_progress_anchors=(),
            contacts=(), owners=(), robot_owner_permissions=(),
            evaluate=Mock(side_effect=lambda spheres, p, r, progress, *args, **kwargs:
                          torch.full_like(progress, scene_value)))
        planner._verification_scene = planner.scene
        planner.scene_workspace_bytes, planner.anchor_fk_workspace_bytes, planner.anchor_capacity = 64, 32, 4
        planner.scene_cpu_workspace_bytes, planner.scene_gpu_workspace_bytes = 24, 40
        planner.reference_seeds = Mock(side_effect=AssertionError("an infeasible fixed entrance must stop before IK"))
        planner.optimize, planner.ompl_seed, planner.close = Mock(), Mock(), Mock()
        return planner, cspace, collision

    def test_checks_effective_named_limits_and_actual_fk_without_unknown_derivatives(self):
        planner, cspace, collision = self.fixture()
        result = planner.initial_state_diagnostics()
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_components"], [])
        measured = planner._verification_fk.compute_kinematics.call_args.args[0]
        torch.testing.assert_close(measured.position[:, 0], planner.start.position, rtol=0, atol=0)
        torch.testing.assert_close(measured.velocity[:, 0], planner.start.velocity, rtol=0, atol=0)
        for field in ("acceleration", "jerk", "dt"):
            self.assertIsNone(getattr(measured, field, None))
        cspace.forward.assert_not_called()
        planner._verification_native.compute_costs.assert_not_called()
        collision.setup_batch_tensors.assert_called_once_with(1, 1)
        collision.forward.assert_called_once()
        spheres, position, quaternion, progress, joints, names = planner.scene.evaluate.call_args.args
        expected_position = (planner.problem.base_pose[:3, :3]
                             @ (planner.problem.start_q+planner.problem.tool_transform[:3, 3])
                             + planner.problem.base_pose[:3, 3])
        np.testing.assert_allclose(position[0, 0], expected_position, atol=1e-14, rtol=0)
        np.testing.assert_allclose(spheres[0, 0, 0, :3],
            planner.problem.base_pose[:3, :3]@planner.problem.start_q+planner.problem.base_pose[:3, 3],
            atol=1e-14, rtol=0)
        torch.testing.assert_close(progress, torch.zeros((1, 1), dtype=torch.float64), rtol=0, atol=0)
        torch.testing.assert_close(joints[:, 0], planner.start.position, rtol=0, atol=0)
        self.assertEqual(names, planner.joint_names)
    def test_each_immutable_violation_rejects_the_same_measured_entrance(self):
        for component, arguments in (("position", {"position": (.21, -.3, .5)}),
            ("velocity", {"velocity": (.7, -2.3, 2.4)}),
            ("self_collision", {"native_collision": 1.}),
            ("scene", {"scene_value": 1.}), ("scene", {"scene_value": np.nan})):
            with self.subTest(component=component):
                planner, _, _ = self.fixture(**arguments)
                before = (planner.start.position.clone(), planner.start.velocity.clone())
                result = planner.initial_state_diagnostics()
                self.assertFalse(result["passed"])
                self.assertIn(component, result["failed_components"])
                planner.reference_seeds.assert_not_called()
                planner.optimize.assert_not_called()
                planner.ompl_seed.assert_not_called()
                torch.testing.assert_close(planner.start.position, before[0], rtol=0, atol=0)
                torch.testing.assert_close(planner.start.velocity, before[1], rtol=0, atol=0)
# END measured-entrance necessary conditions without invented derivatives.


    # END the archive preserves qualified motion but cannot mask environment failures.


if __name__ == "__main__":
    unittest.main()
# END independent calculus, exact-anchor and execution-boundary regressions.
