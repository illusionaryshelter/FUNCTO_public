# BEGIN tests migrated to native motion attempts and retained numerical contracts.
"""Independent CPU checks for planner calculus and exported-curve semantics.

SciPy supplies the rotation and cubic-curve oracles. The export fixture uses a
declared Cartesian test kinematics, so these tests do not claim native robot
IK, GPU execution, collision geometry or physical grasp certification.
"""

from copy import deepcopy
from contextlib import contextmanager, ExitStack, nullcontext
from dataclasses import asdict
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
    JointTrajectory,
    cubic_derivative_report,
    _CuroboPlanner,
    _ProgressMap,
    _formal_native_clock,
    _fixed_spline_grid,
    _timed_native_state,
    _anchor_capacity,
    _planning_workspace_bytes,
    _PlanningWorkspace,
    _require_planning_memory,
    _bounded_robot_configuration,
    _hermite_at_parameters,
    _native_gradient_layout_type,
    _orientation_vjp_type,
    _process_manager_type,
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
    """Exercise real polynomial export with explicit analytic joint bounds.

    Native integration and collision checks have separate tests. Broad bounds
    keep these fixtures focused on process/timing; derivative regressions
    replace those bounds with their stated physical values.
    """
    position = torch.as_tensor(position, dtype=torch.float64)[None]
    velocity = torch.as_tensor(velocity, dtype=torch.float64)[None]

    native = SimpleNamespace(position=position, velocity=velocity,
                             dt=torch.tensor([dt], dtype=torch.float64))
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.check_request = Mock()
    planner.problem = problem
    planner.budget = PlanningBudget()
    clock = {(event.skill_id, event.event_id, event.occurrence): {"event": asdict(event), "time": float(progress_map.anchor_parameters([event.progress])[0, 0]*(position.shape[1]-1)*dt),
        "hold_until": float(progress_map.anchor_parameters([event.progress])[-1, 0]*(position.shape[1]-1)*dt)} for event in problem.events}
    planner.progress_map = _ProgressMap(progress_map.u, progress_map.s, clock)
    planner.anchors = problem.anchors
    planner.joint_names = problem.joint_names
    planner.torch = torch
    planner.trajopt = SimpleNamespace(get_active_js=lambda state: state)
    planner.formal_clock = {"valid_count": position.shape[1], "interpolation_dt": dt,
                            "duration": (position.shape[1]-1)*dt}
    planner.finger_trajectory = None
    planner.start = SimpleNamespace(position=position[:, 0], velocity=velocity[:, 0])
    limits = SimpleNamespace(joint_names=list(problem.joint_names),
        position=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64),
        velocity=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64),
        acceleration=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64))
    cspace = SimpleNamespace(enabled=True, config=SimpleNamespace(joint_limits=limits))
    planner._verification_native = SimpleNamespace(
        get_cost={"cspace": cspace}.get,
        compute_costs=lambda state: SimpleNamespace(names=(), values=()))
    return planner, SimpleNamespace(batch_size=1, num_seeds=1, get_interpolated_plan=lambda: native)


class PlannerCalculusTests(unittest.TestCase):
    def test_formal_spline_clock_preserves_raw_duration_and_exact_dense_grid(self):
        from curobo._src.types.control_space import ControlSpace
        for native_dt, subdivisions, count in ((.2, 32, 641), (.05, 8, 161)):
            with self.subTest(native_dt=native_dt):
                native = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=native_dt,
                    maximum_trajectory_dt=native_dt, interpolation_dt=.025), interpolation_steps=4,
                    transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3), action_horizon=18)
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
        from manipulation.gripper_trajectory import NominalFingerGeometry
        hand.nominal_finger_trajectory = NominalFingerGeometry(("finger",), [0., 1.], [[.04], [.02]],
            [[-.02], [-.02]], 1., 1., "explicit synthetic free finger geometry")
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
    """Check shared physical clocks, scalar reference warps and seed queries.

    The scalar reference oracle has an independently chosen positive initial
    rate and a stopped end. It never represents the joint action boundary;
    that separate map is tested in test_cubic_trajectory and physical domains.
    """

    @staticmethod
    def scalar_reference_cubic(actions, initial=0., initial_velocity=0., dt=1.):
        count = len(actions)
        prefix = initial+np.array([-1., 0., 1., 2.])*initial_velocity*dt
        coefficients = np.r_[prefix, actions, np.repeat(actions[-1], 3)]
        return BSpline(np.arange(-3., count+8.), coefficients, 3)

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

    def test_source_progress_keeps_source_and_terminal_plateaus_without_entrance_pause(self):
        source = _ProgressMap(np.array([0., .3, .55, .8, 1.]), np.array([0., .6, .6, 1.7, 2.]))
        for count in (6, 8, 12):
            with self.subTest(free_controls=count):
                finish, last_moving = count+4, count+3
                leading = np.linspace(0., 1., 7)
                trailing = np.linspace(last_moving, finish, 7)
                mapped, warp = _progress_for_cubic_boundaries(
                    source, count, stop_progress=(), anchors=source.s)
                self.assertEqual(mapped.anchor_parameters([0.]).shape, (1, 2))
                self.assertTrue(np.all(np.interp(leading[1:]/finish, mapped.u, mapped.s) > 0.))
                np.testing.assert_array_equal(np.interp(trailing/finish, mapped.u, mapped.s), 2.)
                physical_source_times = np.interp(source.u, warp.s, warp.u)
                physical_source_times[0], physical_source_times[-1] = 0., last_moving/finish
                np.testing.assert_allclose(np.interp(physical_source_times, mapped.u, mapped.s), source.s, atol=1e-14)
                np.testing.assert_allclose(mapped.anchor_parameters([.6])[:, 0], physical_source_times[[1, 2]], atol=1e-14)

    def test_internal_stop_matches_a_feasible_native_cubic_with_bounded_clock_error(self):
        # A positive scalar entrance rate reaches the same central stop;
        # this is a reference parameter, not a prescribed robot velocity.
        count = 15
        controls = np.r_[np.arange(3, 9)/16., [.5]*3, .5+np.arange(1, 7)/12.]
        exact_clock = self.scalar_reference_cubic(controls, initial_velocity=1/16)
        exact_angle = self.scalar_reference_cubic(.45*np.maximum(2*controls-1., 0.))
        source = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        mapping, warp = _progress_for_cubic_boundaries(
            source, count, stop_progress=(1.,), anchors=(0., 1., 2.))
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
        exact = self.scalar_reference_cubic(controls, initial_velocity=parameter_velocity)
        source = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        measured = np.array([.27, -.13])
        mapping, warp = _progress_for_cubic_boundaries(
            source, count, stop_progress=(1.,), anchors=(0., 1., 2.))
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
            source, 15, stop_progress=(.3,), anchors=problem.anchors)
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
            _progress_for_cubic_boundaries(source, 8,
                                           stop_progress=(.3, .7), anchors=(.3, .7))
        # Retaining all distinct preexisting path vertices is part of the
        # contract. An oversized input may be rejected, never decimated into
        # a different source or OMPL q/s path to fit the fixed key budget.
        crowded = _ProgressMap(np.linspace(0., 1., 16385), np.linspace(0., 1., 16385))
        with self.assertRaisesRegex(ValueError, "(?i)(key|clock|budget)"):
            _progress_for_cubic_boundaries(crowded, 8,
                                           stop_progress=(), anchors=())
        with self.assertRaisesRegex(ValueError, "(?i)(float32|precision)"):
            _progress_for_cubic_boundaries(source, 8,
                                           stop_progress=(), anchors=(.5, .5+1e-10))

    def test_native_key_alias_drops_only_redundant_grid_points_and_keeps_exact_anchors(self):
        # In this scalar clock's linear interior, z=.3 maps to u=3/12.
        # Forward grid arithmetic and inverse-anchor arithmetic yield nearby
        # float64 values that are the SAME float32 key. The optional grid key
        # can be replaced; the actual source anchor must keep its exact value.
        source = _ProgressMap(np.array([0., .3, .8, 1.]), np.array([0., .6, 1.6, 2.]))
        mapping, warp = _progress_for_cubic_boundaries(
            source, 8, stop_progress=(), anchors=source.s)
        self.assertTrue(np.all(np.diff(warp.u.astype(np.float32)) > 0))
        self.assertTrue(np.all(np.diff(mapping.u.astype(np.float32)) > 0))
        pairs = mapping.anchor_parameters([.6, 1.6])
        np.testing.assert_array_equal(pairs[:, 1], [.6, 1.6])
        np.testing.assert_allclose(pairs[:, 0], (10*np.array([.3, .8]))/12, atol=2e-15)
        retained = np.searchsorted(warp.u, pairs[:, 0])
        np.testing.assert_array_equal(warp.u[retained], pairs[:, 0])
        np.testing.assert_array_equal(warp.s[retained], [.3, .8])
        exact = self.scalar_reference_cubic(np.arange(3, 11)/10., initial_velocity=.1)
        query = np.linspace(0., 1., 20001)
        self.assertLessEqual(float(np.max(np.abs(np.interp(query, warp.u, warp.s)-exact(query*12)))),
                             1e-6+1e-14)

        # These nearby true source anchors remain distinct with the corrected
        # entrance map; do not preserve the old compression's rejection.
        linear = _ProgressMap(np.array([0., 1.]), np.array([0., 2.]))
        close = np.array([.6, .6+2*float(np.spacing(np.float32(.3)))])
        self.assertEqual(linear.anchor_parameters(close).shape, (2, 2))
        revised, _ = _progress_for_cubic_boundaries(linear, 8, stop_progress=(), anchors=close)
        self.assertTrue(np.all(np.diff(revised.anchor_parameters(close)[:, 0].astype(np.float32)) > 0))


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
        self.config_type = _process_manager_type(None, None, None, None, None)
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


class ExportedCurveContractTests(unittest.TestCase):


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

    def test_multiple_native_seeds_cannot_be_flattened_into_one_export(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((3, 3)), np.zeros((3, 3)), .25)
        result.num_seeds = 2
        result.get_interpolated_plan = Mock(side_effect=AssertionError("must reject before reading mixed curves"))
        with self.assertRaisesRegex(ValueError, "one.*seed|selected seed"):
            planner.export(result, {"accepted": True, "residuals": {}})
        result.get_interpolated_plan.assert_not_called()

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


if __name__ == "__main__":
    unittest.main()
# END native attempt-planner test migration.
