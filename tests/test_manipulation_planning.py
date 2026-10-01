"""Independent CPU checks for planner calculus and exported-curve semantics.

SciPy supplies the rotation and cubic-curve oracles. The export fixture uses a
declared Cartesian test kinematics, so these tests do not claim native robot
IK, GPU execution, collision geometry or physical grasp certification.
"""

# BEGIN independent calculus, exact-anchor and execution-boundary regressions.
from copy import deepcopy
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
    _timed_native_state,
    # BEGIN actual-anchor allocation and projection preflight regressions.
    _anchor_capacity,
    _planning_workspace_bytes,
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
        tool_transform=np.eye(4), phases=tuple(phases), events=tuple(events))


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
            spheres[..., 3] = .001
            return SimpleNamespace(
                tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
                                           position=q[:, :, None], quaternion=quaternion),
                robot_spheres=spheres)

    native = SimpleNamespace(position=position, velocity=velocity,
                             dt=torch.tensor([dt], dtype=torch.float64))
    planner = _CuroboPlanner.__new__(_CuroboPlanner)
    planner.problem = problem
    planner.budget = PlanningBudget()
    planner.progress_map = progress_map
    planner.anchors = problem.anchors
    planner.joint_names = problem.joint_names
    planner.torch = torch
    planner.device_cfg = SimpleNamespace(to_device=lambda x: torch.as_tensor(x, dtype=torch.float64))
    planner.JointState = SimpleNamespace
    planner._RobotState = SimpleNamespace
    planner.trajopt = SimpleNamespace(get_active_js=lambda state: state)
    planner.formal_clock = {"valid_count": position.shape[1], "interpolation_dt": dt,
                            "duration": (position.shape[1]-1)*dt}
    planner.finger_trajectory = None
    planner.start = SimpleNamespace(position=position[:, 0], velocity=velocity[:, 0])
    planner._verification_fk = CartesianKinematics()
    planner._verification_scene = SimpleNamespace(
        gripper=None, evaluate=lambda spheres, p, r, s, q, names, **kwargs: torch.zeros_like(s),
        check_robot_coverage=lambda *args, **kwargs: [])
    # BEGIN give the real exporter explicit, finite native acceleration bounds.
    limits = SimpleNamespace(joint_names=list(problem.joint_names),
        acceleration=torch.tensor([[-1e6]*3, [1e6]*3], dtype=torch.float64))
    cspace = SimpleNamespace(enabled=True, config=SimpleNamespace(joint_limits=limits))
    planner._verification_native = SimpleNamespace(
        get_cost={"cspace": cspace}.get,
        compute_costs=lambda state: SimpleNamespace(names=(), values=()))
    # END these analytic fixture bounds are not a robot-model assertion.
    return planner, SimpleNamespace(batch_size=1, num_seeds=1, get_interpolated_plan=lambda: native)


class PlannerCalculusTests(unittest.TestCase):
    # BEGIN formal time and complete-volume gates are independent of point costs.
    def test_formal_spline_clock_reuses_native_count_not_rollout_duration(self):
        from curobo._src.types.control_space import ControlSpace
        native = SimpleNamespace(config=SimpleNamespace(minimum_trajectory_dt=.05,
            maximum_trajectory_dt=.05, interpolation_dt=.025), interpolation_steps=4,
            transition_model=SimpleNamespace(control_space=ControlSpace.BSPLINE_3), action_horizon=16)
        clock = _formal_native_clock(native)
        self.assertEqual(clock["valid_count"], 181)
        self.assertEqual(clock["duration"], 180*float(np.float32(.025)))
        self.assertNotEqual(clock["duration"], 80*float(np.float32(.05)))
        native.config.minimum_trajectory_dt = .04
        with self.assertRaisesRegex(ValueError, "fixed request-wide"):
            _formal_native_clock(native)

    def test_export_full_volume_rejection_is_hard_even_when_all_gpu_costs_zero(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
        mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        planner, native = export_fixture(problem, mapping, np.zeros((2, 3)), np.zeros((2, 3)), 1.)
        calls = []

        def coverage(q, names, t, *, entrance=False):
            calls.append((q.numpy().copy(), t.numpy().copy(), entrance))
            return [{"passed": bool(value != .5), "failure": None if value != .5 else "new_unknown_volume"}
                    for value in t.reshape(-1).tolist()]

        planner._verification_scene.check_robot_coverage = coverage
        curve, count = planner.export(native)
        self.assertEqual(count, 3)
        self.assertEqual(curve.residuals["unknown/new_robot_volume"], 1.)
        self.assertTrue(calls[0][2])
        self.assertFalse(calls[1][2])
        self.assertEqual(calls[1][0].dtype, np.float64)
        self.assertEqual(planner.last_validation["robot_coverage"]["failed"][0]["time_s"], .5)

    def test_initializer_plateau_inverse_is_explicitly_earliest(self):
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.progress_map = _ProgressMap(np.array([0., .2, .4, 1.]), np.array([0., 1., 1., 2.]))
        self.assertEqual(planner._seed_u(1.), .2)

    def test_native_self_collision_receives_timed_spheres_without_mutating_fk(self):
        from manipulation.planning_scene import PrismaticGripper
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
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

        hand = PrismaticGripper(controller="timed_reference", joint_names=("finger",),
            current_positions=[.04], joint_limits=[[0, .04]], joint_bounds=None,
            finger_trajectory_factory=lambda *_: None, coupling_matrix=np.empty((0, 1)), coupling_offset=[],
            sphere_indices=(0,), local_spheres=[[0, 0, 0, .001]],
            sphere_joint_translation=[[[1], [0], [0]]], self_collision_pairs=(),
            commands=[{"skill_id": "test", "event_id": "close", "command": "close", "progress": 0.}],
            source="analytic timed native-self input")
        hand.finger_trajectory = SimpleNamespace(sample_torch=lambda t: (
            (.04-.02*t)[..., None], torch.full((*t.shape, 1), -.02, dtype=t.dtype)))
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
    def test_coupled_endpoint_and_release_frame_vjp_matches_scipy_whole_horizon(self):
        # Event poses are optimization variables. Their derivatives must reach
        # earlier motion, the still-held successor and the released retreat;
        # a correct local FK derivative alone does not establish that chain.
        def transform(position, angles):
            result = np.eye(4)
            result[:3, 3] = position
            result[:3, :3] = Rotation.from_euler("xyz", angles).as_matrix()
            return result

        attachment = transform([.08, -.04, .11], [.43, -.52, .71])
        body = np.stack((transform([.2, -.3, .5], [.1, -.2, .3]),
                         transform([.3, -.2, .35], [-.1, .4, .7]),
                         transform([.4, -.1, .2], [.2, -.1, .9])))
        held = np.stack((body[-1], transform([.42, -.08, .2], [.2, -.1, 1.])))
        retreat = np.stack((held[-1] @ attachment,
                            held[-1] @ attachment @ transform([.03, -.02, .07], [.1, -.15, .08])))
        source = "independent noncommuting endpoint and release-frame fixture"
        phases = (
            MotionPhase("place", PosePath(np.array([0., .5, 1.]), body, source,
                                           attachment=attachment),
                        endpoint_family={"move_start": False, "end": "variable"}),
            MotionPhase("hold", PosePath(np.array([0., 1.]), held, source,
                                          attachment=attachment),
                        endpoint_family={"move_start": True, "end": "unanchored"}),
            MotionPhase("retreat", PosePath(np.array([0., 1.]), retreat, source),
                        frame_anchor={"progress": 2., "reference_pose": retreat[0]}),
        )
        problem = cartesian_problem(phases)
        progress = np.array([0., .2, .5, .8, 1., 1.25, 1.5, 1.8, 2., 2.2, 2.5, 2.8, 3.])
        rng = np.random.default_rng(730)
        # Seven explicit analytic coordinates: XYZ translation, Z/Y/X serial
        # rotations, and an unused coordinate. This is not a robot fixture.
        values = rng.normal(scale=.22, size=(2, len(progress), 7))
        values[..., :3] += np.array([.3, -.1, .4])
        values[..., 3:6] += np.array([.8, -.45, .3])
        sample_weight = np.linspace(.4, 1.6, len(progress))[None]
        sample_weight[:, [4, 8]] = 0.
        phase_paths = (body, held, retreat)
        phase_knots = (np.array([0., .5, 1.]), np.array([0., 1.]), np.array([0., 1.]))

        def sample(poses, knots, local):
            output = np.broadcast_to(np.eye(4), (len(local), 4, 4)).copy()
            output[:, :3, 3] = np.column_stack([np.interp(local, knots, poses[:, axis, 3])
                                                for axis in range(3)])
            output[:, :3, :3] = Slerp(knots, Rotation.from_matrix(poses[:, :3, :3]))(local).as_matrix()
            return output

        def independent_loss(configurations):
            quaternions, _ = serial_rotations(configurations[..., 3:])
            rotations = Rotation.from_quat(quaternions[..., 0, :].reshape(-1, 4)[:, [1, 2, 3, 0]])
            actual = np.broadcast_to(np.eye(4), (*configurations.shape[:2], 4, 4)).copy()
            actual[..., :3, 3] = configurations[..., :3]
            actual[..., :3, :3] = rotations.as_matrix().reshape(*configurations.shape[:2], 3, 3)
            violation = np.zeros(configurations.shape[:2])
            for batch, observed in enumerate(actual):
                for index in range(3):
                    local = np.clip(progress-index, 0., 1.)
                    reference = sample(phase_paths[index], phase_knots[index], local)
                    if index < 2:
                        initial = body[0] if index == 0 else observed[4] @ np.linalg.inv(attachment)
                        mapped = reference @ np.linalg.inv(phase_paths[index][0]) @ initial
                        if index == 0:
                            goal = observed[4] @ np.linalg.inv(attachment)
                            mapped_end = body[-1] @ np.linalg.inv(body[0]) @ initial
                            mapped[:, :3, 3] += local[:, None]*(goal[:3, 3]-mapped_end[:3, 3])
                            correction = goal[:3, :3] @ mapped_end[:3, :3].T
                            turns = Slerp([0., 1.], Rotation.from_matrix(np.stack((np.eye(3), correction))))(local)
                            mapped[:, :3, :3] = turns.as_matrix() @ mapped[:, :3, :3]
                        target = mapped @ attachment
                    else:
                        target = observed[8] @ np.linalg.inv(retreat[0]) @ reference
                    relative = np.swapaxes(target[:, :3, :3], -1, -2) @ observed[:, :3, :3]
                    position_error = np.einsum("hji,hj->hi", target[:, :3, :3],
                                               observed[:, :3, 3]-target[:, :3, 3])
                    errors = np.column_stack((position_error, Rotation.from_matrix(relative).as_rotvec()))
                    tolerance = np.array([problem.numerical.position_m]*3+[problem.numerical.rotation_rad]*3)
                    term = np.maximum(np.abs(errors)-tolerance, 0.).sum(-1)
                    violation[batch] += term*((progress >= index) & (progress <= index+1))
            return np.sum(violation**2*sample_weight)

        expected = np.zeros_like(values)
        epsilon = 2e-6
        for index in np.ndindex(values.shape):
            plus, minus = values.copy(), values.copy()
            plus[index] += epsilon
            minus[index] -= epsilon
            expected[index] = (independent_loss(plus)-independent_loss(minus))/(2*epsilon)
        for sign in (1., -1.):
            with self.subTest(quaternion_sign=sign):
                coordinates = torch.tensor(values, dtype=torch.float64, requires_grad=True)
                quaternions, jacobians = serial_rotations(values[..., 3:])
                orientation = _orientation_vjp_type().apply(
                    coordinates[..., 3:], torch.tensor(sign*quaternions, dtype=torch.float64),
                    torch.tensor(jacobians, dtype=torch.float64))[..., 0, :]
                residual = problem.evaluate(coordinates[..., :3], orientation,
                                            torch.tensor(np.broadcast_to(progress, values.shape[:2]).copy()))
                loss = (residual.square()*torch.tensor(sample_weight)).sum()
                self.assertAlmostEqual(loss.item(), independent_loss(values), places=10)
                loss.backward()
                np.testing.assert_allclose(coordinates.grad.numpy(), expected, rtol=2e-7, atol=2e-7)
                np.testing.assert_array_equal(coordinates.grad.numpy()[..., -1], 0.)
                # The event nodes have zero direct loss weight. Nonzero event
                # gradients therefore prove the required cross-phase coupling.
                self.assertGreater(np.linalg.norm(coordinates.grad.numpy()[:, 4, :6]), .1)
                self.assertGreater(np.linalg.norm(coordinates.grad.numpy()[:, 8, :6]), .1)
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
    def test_stop_preference_has_an_interior_gradient_and_covers_each_plateau_once(self):
        progress = torch.tensor([[.2, .3, .3, .3+0.5e-7, .3+2e-7, .6],
                                 [.3, .3, .3, .4, .6, .7]], dtype=torch.float64)
        velocity = torch.tensor([.25e-4, -.5e-4, .75e-4], dtype=torch.float64)
        velocity = velocity.expand(2, 6, 3).clone().requires_grad_(True)
        result = _verification_stop_preference(progress, velocity, [.3, .3, .6])
        # The union of explicit events owns the objective; duplicate event
        # progress does not double its weight. Every saved plateau point and
        # the declared 1e-7 event comparison band receives the same target.
        selected = np.array([[False, True, True, True, False, True],
                             [True, True, True, False, True, False]])
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
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),), (event,))
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
        np.testing.assert_allclose(after["stop:analytic:stop"].detach().numpy(), [[0., .25e-4, 0.]], atol=1e-20)
        self.assertGreater(float(objective[0, 0].detach()), 0.)
        self.assertEqual(float(objective[0, 2].detach()), 0.)
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

    def test_ompl_initializer_uses_the_same_warp_when_joint_motion_continues_during_a_pause(self):
        # Real OMPL search is covered separately. This fixture supplies a
        # declared valid q/s path and exercises the production post-search
        # assembly, with analytic FK only. Its second edge changes q while s
        # remains .3; interpolating q as a function of s would lose that edge.
        event = PlanningEvent("pause-stop", "analytic-skill", .3, 10, "test stop", "fixture", stop=True)
        problem = cartesian_problem((MotionPhase("free connection", constant_path(), kind="free"),), (event,))
        paths = (
            np.array([[0., 0., 0., 0.], [.3, 0., 0., .3]]),
            np.array([[.3, 0., 0., .3], [.3, .6, 0., .3],
                      [.6, .6, 0., .55], [1.5, .1, 0., 1.]]),
        )
        path = np.concatenate((paths[0], paths[1][1:]))
        old_u = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        old_u /= old_u[-1]
        source = _ProgressMap(old_u, path[:, -1])
        expected_map, warp = _progress_for_cubic_boundaries(
            source, 15, problem.start_dq, stop_progress=problem.stop_progress, anchors=problem.anchors)
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.torch, planner.problem = torch, problem
        planner.budget = PlanningBudget(knots=15, search_seconds=3., interpolation_buffer_size=768)
        planner.anchors, planner.joint_names = problem.anchors, problem.joint_names
        planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
        planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64))
        planner.trajopt = SimpleNamespace(action_horizon=15, transition_model=SimpleNamespace(
            action_bound_lows=torch.full((3,), -2.), action_bound_highs=torch.full((3,), 2.)))
        candidates = torch.tensor(path[[0, 1, -1]][None, None, :, :3], dtype=torch.float64)
        planner._boundary_candidates = lambda *args, **kwargs: candidates.clone()

        class AnalyticState(SimpleNamespace):
            @classmethod
            def from_position(cls, position, **kwargs):
                return cls(position=position, **kwargs)

        def fk(state):
            q = state.position
            quaternion = q.new_zeros((*q.shape[:2], 1, 4))
            quaternion[..., 0] = 1.
            return SimpleNamespace(tool_poses=SimpleNamespace(tool_frames=[problem.tool_link],
                                                              position=q[:, :, None], quaternion=quaternion),
                                   tool_jacobians=q.new_zeros((*q.shape[:2], 1, 6, 3)))

        planner.JointState = AnalyticState
        planner._verification_fk = SimpleNamespace(compute_kinematics=fk)
        planner._orientation_vjp = SimpleNamespace(apply=lambda q, quaternion, jacobian: quaternion)
        planner._native_layout = SimpleNamespace(apply=lambda value: value)
        scene = SimpleNamespace()
        scene.with_fixed_endpoints = lambda *args: scene
        scene.clone_for_rollout = lambda: scene
        planner.scene = scene
        with patch("manipulation.planning._project_boundary", side_effect=lambda value, *args, **kwargs: value), \
                patch("manipulation.planning._ompl_stage", side_effect=[(value, {"success": True}) for value in paths]) as search:
            seed, mapping, report = planner.ompl_seed(torch.zeros((1, 1, 15, 3)), None)
        self.assertTrue(report["success"])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(tuple(seed.shape), (1, 1, 15, 3))
        np.testing.assert_array_equal(mapping.u, expected_map.u)
        np.testing.assert_array_equal(mapping.s, expected_map.s)
        greville = (np.arange(15)+3.)/19.
        old_parameter = np.interp(greville, warp.u, warp.s)
        expected_q = np.column_stack([np.interp(old_parameter, old_u, path[:, index]) for index in range(3)])
        np.testing.assert_allclose(seed.numpy()[0, 0], expected_q, atol=1e-14)
        progress_at_controls = np.interp(greville, mapping.u, mapping.s)
        pause_q = seed.numpy()[0, 0][progress_at_controls == .3]
        self.assertGreaterEqual(len(pause_q), 2)
        self.assertGreater(float(np.ptp(pause_q[:, 1])), .1)
        # The stop is at the original pause's first arrival; validation keeps
        # both ends of the entire pause, including the subsequent q movement.
        pause = mapping.anchor_parameters([.3])
        self.assertEqual(pause.shape, (2, 2))
        np.testing.assert_allclose(np.interp(pause[:, 0], warp.u, warp.s), old_u[1:3], atol=1e-14)
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
        requested = []

        def solve_pose(goal, **kwargs):
            requested.append(goal)
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor([[[goal], [goal+.2]]])),
                                   success=torch.tensor([[True, True]]))

        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.torch = torch
        planner.budget = PlanningBudget(ik_seeds=2, trajectory_seeds=1)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=1e-5))
        planner.start = SimpleNamespace(position=torch.tensor([[-10.]]))
        planner.seed_progress = torch.zeros((1, 1))
        planner.progress_map = mapping
        planner.trajopt = SimpleNamespace(action_horizon=count)
        planner.ik = SimpleNamespace(solve_pose=solve_pose, get_active_js=lambda state: state)
        planner._goal = lambda progress: progress
        seeds, success_counts = planner.reference_seeds()
        np.testing.assert_allclose(requested, np.interp(clock, mapping.u, mapping.s), atol=1e-14)
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
        goals, supplied_seed_shapes = [], []

        def failed_pose_ik(goal, **kwargs):
            goals.append(goal)
            supplied_seed_shapes.append(tuple(kwargs["seed_config"].shape))
            # Finite solutions of failed pose IK may initialize the optimizer,
            # but their zero success count must remain visible to its caller.
            first = measured_q+goal
            second = first+1.
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.stack((first, second), dim=1)),
                                   success=torch.tensor([[False, False]]))

        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.torch = torch
        planner.budget = PlanningBudget(ik_seeds=2, trajectory_seeds=2)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=1e-5))
        planner.start = SimpleNamespace(position=measured_q, velocity=measured_dq)
        planner.seed_progress = torch.zeros((1, 1))
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        planner.trajopt = SimpleNamespace(action_horizon=horizon,
                                         _trajectory_seed_generator=SimpleNamespace(generate_constant_seeds=helper))
        planner.ik = SimpleNamespace(solve_pose=failed_pose_ik, get_active_js=lambda state: state)
        planner._goal = lambda progress: progress
        seeds, success_counts = planner.reference_seeds()
        helper.assert_called_once()
        self.assertIs(helper.call_args.args[0], measured_q)
        self.assertEqual(helper.call_args.args[1], 1)
        self.assertEqual(tuple(seeds.shape), (1, 2, horizon, 3))
        self.assertTrue(seeds.is_contiguous())
        self.assertEqual(success_counts, [0]*horizon)
        self.assertEqual(supplied_seed_shapes, [(1, 1, 3)]+[(1, 2, 3)]*(horizon-1))
        np.testing.assert_allclose(seeds[0, 0].numpy(), measured_q.numpy()+np.asarray(goals)[:, None], atol=1e-14)
        np.testing.assert_array_equal(seeds[0, 1].numpy(), np.broadcast_to(measured_q.numpy(), (horizon, 3)))
        torch.testing.assert_close(measured_q, position_before, rtol=0, atol=0)
        torch.testing.assert_close(measured_dq, velocity_before, rtol=0, atol=0)
        self.assertIs(planner.start.position, measured_q)
        self.assertIs(planner.start.velocity, measured_dq)

        # A constant alternative does not conceal a corrupt native IK result.
        # Nonfinite candidates must still fail before any trajectory solve.
        planner.ik.solve_pose = lambda *args, **kwargs: SimpleNamespace(
            js_solution=SimpleNamespace(position=torch.full((1, 2, 3), float("nan"))),
            success=torch.tensor([[False, False]]))
        with self.assertRaisesRegex(ValueError, "nonfinite initialization"):
            planner.reference_seeds()
    # END native constant-seed initialization contract after failed pose IK.


# BEGIN native-result diversity and whole-window initializer selection contracts.
class ReferenceSeedDiversityTests(unittest.TestCase):
    """Declared IK outputs test selection, never actual native pose feasibility."""

    @staticmethod
    def planner_for(pools, flags, *, seed_count=2, joint_tolerance=1e-5):
        from curobo.types import DeviceCfg
        from curobo._src.util.trajectory_seed_generator import TrajectorySeedGenerator

        horizon, candidates = len(pools), len(pools[0])
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.torch = torch
        planner.budget = PlanningBudget(knots=horizon, ik_seeds=candidates, trajectory_seeds=seed_count)
        planner.problem = SimpleNamespace(numerical=SimpleNamespace(joint=joint_tolerance))
        planner.start = SimpleNamespace(position=torch.zeros((1, 3), dtype=torch.float64),
                                        velocity=torch.tensor([[.1, -.2, .3]], dtype=torch.float64))
        planner.seed_progress = torch.zeros((1, 1), dtype=torch.float64)
        planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
        generator = TrajectorySeedGenerator(horizon, 3, DeviceCfg("cpu"))
        constant = Mock(wraps=generator.generate_constant_seeds)
        planner.trajopt = SimpleNamespace(action_horizon=horizon,
                                         _trajectory_seed_generator=SimpleNamespace(generate_constant_seeds=constant))
        warm_starts = []

        def solve(goal, **kwargs):
            index = len(warm_starts)
            warm_starts.append(kwargs["seed_config"].clone())
            return SimpleNamespace(js_solution=SimpleNamespace(position=torch.tensor(pools[index][None], dtype=torch.float64)),
                                   success=torch.tensor(flags[index][None], dtype=torch.bool))

        planner.ik = SimpleNamespace(solve_pose=solve, get_active_js=lambda state: state,
                                     config=SimpleNamespace(exit_early=seed_count == 1))
        planner._goal = lambda progress: progress
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

    def test_first_nearest_and_farthest_successes_continue_despite_result_order_changes(self):
        primary, alternative, pools, flags = self.two_tracks()
        planner, constant, warm_starts = self.planner_for(pools, flags)
        measured_q, measured_dq = planner.start.position.clone(), planner.start.velocity.clone()
        seeds, success = planner.reference_seeds()
        np.testing.assert_allclose(seeds.numpy()[0, 0], primary, atol=1e-14)
        np.testing.assert_allclose(seeds.numpy()[0, 1], alternative, atol=1e-14)
        self.assertEqual(success, [3]*6)
        self.assertEqual(planner.reference_ik_distinct_counts, [3]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "reference_ik_alternative"])
        constant.assert_not_called()
        self.assertEqual(tuple(warm_starts[0].shape), (1, 1, 3))
        for index in range(1, 6):
            torch.testing.assert_close(warm_starts[index], seeds[:, :, index-1], rtol=0, atol=0)
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
        np.testing.assert_allclose(seeds.numpy()[0, 0], primary, atol=1e-14)
        np.testing.assert_allclose(seeds.numpy()[0, 1], alternative, atol=1e-14)
        self.assertEqual(success, [2, 1, 2, 2, 1, 2])
        self.assertEqual(planner.reference_ik_distinct_counts, success)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "reference_ik_alternative"])
        constant.assert_not_called()

    def test_one_failed_control_pose_keeps_its_diagnostic_and_uses_measured_state_second(self):
        _, _, pools, flags = self.two_tracks()
        flags[2] = np.zeros(4, dtype=bool)
        planner, constant, _ = self.planner_for(pools, flags)
        seeds, success = planner.reference_seeds()
        self.assertEqual(success, [3, 3, 0, 3, 3, 3])
        self.assertEqual(planner.reference_ik_distinct_counts, [3, 3, 0, 3, 3, 3])
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik", "measured_state_constant"])
        self.assertTrue(bool(torch.isfinite(seeds).all()))
        self.assertGreater(float(seeds[0, 0].abs().max()), 0.)
        np.testing.assert_array_equal(seeds.numpy()[0, 1], np.zeros((6, 3)))
        constant.assert_called_once()
        self.assertIs(constant.call_args.args[0], planner.start.position)
        self.assertEqual(constant.call_args.args[1], 1)

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
        np.testing.assert_allclose((seeds[0, 1]-seeds[0, 0]).numpy(), np.tile([4*tolerance, 0., 0.], (6, 1)), atol=1e-14)
        constant.assert_not_called()

    def test_single_seed_budget_keeps_its_nearest_native_continuation(self):
        primary, _, pools, flags = self.two_tracks()
        planner, constant, _ = self.planner_for(pools, flags, seed_count=1)
        seeds, success = planner.reference_seeds()
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))
        np.testing.assert_allclose(seeds.numpy()[0, 0], primary, atol=1e-14)
        self.assertEqual(success, [3]*6)
        self.assertEqual(planner.initial_seed_strategies, ["reference_ik"])
        constant.assert_not_called()
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
        # Empty process_mode selects native costs alone; no task, scene or FK
        # stand-in is needed to exercise the actual integration boundary.
        self.config_type = _process_manager_type(None, None, None, None, None, None)
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

        # Exercise the production verification constructor, including rebuilds;
        # its existing deepcopy must receive the still-unmodified template.
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.device_cfg = self.device_cfg
        planner.trajopt = SimpleNamespace(
            config=SimpleNamespace(metrics_rollout_config=SimpleNamespace(constraint_cfg=template)),
            transition_model=self.transition())
        for _ in range(2):
            planner._initialize_verification_constraints()
            self.assertEqual(planner._verification_native.get_cost("self_collision")._weight.item(), 1250.)
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

    def test_complete_319_node_timeline_and_all_322_clock_anchors_survive(self):
        problem = self.timeline_problem()
        mapping = self.clock(problem)
        pairs = mapping.anchor_parameters(problem.anchors)
        self.assertEqual(len(problem.anchors), 319)
        self.assertEqual(len(pairs), 322)
        np.testing.assert_array_equal(np.unique(pairs[:, 1]), problem.anchors)
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, PlanningBudget()), 386)
        no_refinement = PlanningBudget(max_refinement_points=0)
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, no_refinement), 322)
        with self.assertRaisesRegex(ValueError, "refinement-point budget"):
            _anchor_capacity(mapping, problem.anchors, no_refinement, [.123])
        tight = PlanningBudget(max_validation_points=330)
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, tight), 330)
        with self.assertRaisesRegex(ValueError, "refinement-point budget"):
            _anchor_capacity(mapping, problem.anchors, tight, np.linspace(.1, .9, 9))

    def test_total_source_and_pause_overflow_fail_before_native_construction(self):
        for counts, maximum in (((252, 9, 60), 128), ((12, 11, 10), 32)):
            with self.subTest(counts=counts):
                problem = self.timeline_problem(counts)
                scene = SimpleNamespace(validate_problem=Mock(), required_progress_anchors=(),
                                        reference_frame=problem.reference_frame)
                with patch("manipulation.planning._bounded_robot_configuration") as robot, \
                        patch("manipulation.planning._process_manager_type") as manager:
                    with self.assertRaisesRegex(ValueError, "anchors|source knots"):
                        _CuroboPlanner(problem, scene, object(), PlanningBudget(
                            max_validation_points=maximum, max_refinement_points=0), "cuda:0")
                robot.assert_not_called()
                manager.assert_not_called()

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
        scene = SimpleNamespace(estimated_workspace_bytes=lambda batch, horizon: 0)
        budget = PlanningBudget(max_validation_points=8192)
        before_cuda = torch.cuda.is_initialized()
        for count in (322, 386, 8192):
            buffers = scope["create_buffers"](44, count, native)
            expected = sum(value.numel()*value.element_size() for value in buffers.values())
            _, actual = _planning_workspace_bytes(scene, metadata, budget, count)
            self.assertEqual(actual, expected)
        self.assertGreater(actual, 6*1024**3)
        self.assertEqual(torch.cuda.is_initialized(), before_cuda)

    def test_torch_cache_covers_tensors_but_not_the_other_native_reserve(self):
        native_reserve = 2*1024**3
        cuda = SimpleNamespace(mem_get_info=Mock(return_value=(native_reserve+500, 12*1024**3)),
                               memory_reserved=Mock(return_value=400), memory_allocated=Mock(return_value=101))
        runtime = SimpleNamespace(cuda=cuda)
        with self.assertRaisesRegex(ValueError, "anchor FK"):
            _require_planning_memory(runtime, "cuda:0", 200, 600)
        cuda.memory_allocated.return_value = 100
        _require_planning_memory(runtime, "cuda:0", 200, 600)
        cuda.mem_get_info.return_value = native_reserve-1, 12*1024**3
        cuda.memory_reserved.return_value = 10*1024**3
        with self.assertRaisesRegex(ValueError, "GPU memory"):
            _require_planning_memory(runtime, "cuda:0", 200, 600)

    def test_replacement_rechecks_changed_pause_count_before_recreating_solver(self):
        import gc
        import weakref
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.budget = PlanningBudget(max_refinement_points=0)
        planner.anchors = np.array([0., 1., 2.])
        planner.problem = object()
        planner.scene = SimpleNamespace(estimated_workspace_bytes=lambda batch, horizon: batch*horizon)
        planner.robot_metadata = self.robot_metadata()
        planner.progress_map = _ProgressMap(np.array([0., .5, 1.]), planner.anchors)
        planner.seed_progress = object()
        planner.seed_parameter, planner.formal_clock, planner.robot_dict = object(), {}, {}
        planner.refinement_parameters = np.empty(0)
        planner.torch = SimpleNamespace(cuda=SimpleNamespace(
            mem_get_info=Mock(return_value=(12*1024**3, 12*1024**3)),
            memory_reserved=Mock(return_value=0), memory_allocated=Mock(return_value=0)))
        planner.device_cfg = SimpleNamespace(device="cuda:0")
        # Match the native optimizer/core bound-method cycle. destroy() only
        # records graph teardown; it deliberately does not break the cycle.
        destroyed = Mock()

        class CyclicSolver:
            def __init__(self):
                self.core = SimpleNamespace(callback=self.step)

            def step(self):
                pass

            def destroy(self):
                destroyed()

        planner.trajopt = CyclicSolver()
        previous = weakref.ref(planner.trajopt)

        def memory_after_release(device):
            self.assertIsNone(previous())
            return planner.torch.cuda.mem_get_info.return_value

        planner.torch.cuda.mem_get_info.side_effect = memory_after_release
        planner._trajectory_configuration = {}
        planner._optimizer_config_class = SimpleNamespace(create=Mock(return_value="new configuration"))
        planner._optimizer_class = Mock(return_value=SimpleNamespace(destroy=Mock()))
        planner._initialize_verification_constraints = Mock()
        planner._bind_formal_clock = Mock()
        mapping = _ProgressMap(np.array([0., .3, .6, 1.]), np.array([0., 1., 1., 2.]))
        with patch("manipulation.planning._process_manager_type", return_value="new manager") as manager, \
                patch("manipulation.planning.gc.collect", wraps=gc.collect) as collect:
            planner.replace_schedule(mapping)
            self.assertEqual(planner.anchor_capacity, 4)
            expected = _planning_workspace_bytes(planner.scene, planner.robot_metadata, planner.budget, 4)
            self.assertEqual((planner.scene_workspace_bytes, planner.anchor_fk_workspace_bytes), expected)
            destroyed.assert_called_once_with()
            collect.assert_called_once_with()
            manager.assert_called_once()
            planner._optimizer_class.assert_called_once_with("new configuration")
            # A second replacement must repeat the current device check. The
            # old solver is released, but an inadmissible new one is not built.
            planner.torch.cuda.mem_get_info.return_value = 2*1024**3, 12*1024**3
            planner._optimizer_class.reset_mock()
            with self.assertRaisesRegex(ValueError, "GPU memory"):
                planner.replace_schedule(mapping)
            planner._optimizer_class.assert_not_called()
            self.assertIsNone(planner.trajopt)
        self.assertEqual(planner.torch.cuda.mem_get_info.call_count, 2)

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
        scene = SimpleNamespace(estimated_workspace_bytes=lambda batch, horizon: 1024**3)
        with self.assertRaisesRegex(ValueError, "scene query workspaces"):
            _planning_workspace_bytes(scene, self.robot_metadata(), PlanningBudget(), 322)
        with self.assertRaises(TypeError):
            PlanningBudget(max_anchors=128)
        with self.assertRaisesRegex(ValueError, "Refinement points"):
            PlanningBudget(max_refinement_points=-1)

    def test_projection_accuracy_respects_the_strictest_source_mechanism(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-6)
        model = SimpleNamespace(position_tolerance=1e-6, rotation_tolerance=2e-6)
        problem.phases = (SimpleNamespace(reference=SimpleNamespace(motion_model=model)),)
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

    @staticmethod
    def selection_fixture(residuals, errors=()):
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.torch = torch
        planner.budget = PlanningBudget()
        exported = []

        def export(candidate):
            index = int(candidate.solution[0, 0, 0, 0].item())-1
            if index in errors:
                raise ValueError(f"candidate {index} exceeds finite sample budget")
            native = candidate.get_interpolated_plan()
            expected_count = (9, 5)[index]
            if tuple(native.position.shape) != (1, 1, expected_count, 3):
                raise AssertionError("A selected candidate must retain only its own valid samples")
            np.testing.assert_array_equal(native.position[0, 0, 1:].numpy(),
                                          np.full((expected_count-1, 3), index+1))
            np.testing.assert_allclose(native.dt.numpy(), [[(.07, .11)[index]]], rtol=1e-7)
            trajectory = SimpleNamespace(residuals={"required_process": residuals[index]},
                                         candidate_index=index, executable_until=.4, next_verification=None)
            planner.last_validation = {
                "measurements": {"candidate_index": index},
                "numerical_tolerances": {"position_m": 1e-4},
                "u": np.array([0., .5, 1.]),
                "violations": np.array([0., residuals[index], 0.]),
            }
            exported.append(index)
            return trajectory, expected_count

        planner.export = Mock(side_effect=export)
        return planner, exported

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

    def test_second_seed_with_native_and_export_success_cannot_be_discarded(self):
        result = self.native_candidates(flags=(False, True))
        planner, exported = self.selection_fixture((.01, 0.))
        chosen, trajectory, attempt = planner.select_candidate(result)
        self.assertEqual(exported, [0, 1])
        self.assertTrue(bool(chosen.success.all()))
        self.assertEqual(trajectory.candidate_index, 1)
        self.assertEqual(attempt["selected_seed_index"], 1)
        self.assertEqual([candidate["accepted"] for candidate in attempt["candidates"]], [False, True])
        self.assertEqual(attempt["validation_points"], 5)
        self.assertEqual(attempt["measurements"]["candidate_index"], 1)
        torch.testing.assert_close(chosen.solution, result.solution[:, 1:2])

    def test_failed_candidates_use_per_seed_actual_constraints_instead_of_float32_rank(self):
        result = self.native_candidates(flags=(False, False), native_violations=(.02, 0.))
        planner, exported = self.selection_fixture((.004, .009))
        chosen, trajectory, attempt = planner.select_candidate(result)
        self.assertEqual(exported, [0, 1])
        self.assertFalse(bool(chosen.success.any()))
        self.assertEqual(trajectory.candidate_index, 1)
        self.assertEqual(attempt["selected_seed_index"], 1)
        candidates = attempt["candidates"]
        self.assertFalse(any(item["accepted"] for item in candidates))
        self.assertLess(candidates[1]["continuation_violation"], candidates[0]["continuation_violation"])
        self.assertAlmostEqual(candidates[0]["native_metrics"]["rollout"]["constraints"]["cspace"], .02, places=7)
        self.assertEqual(candidates[1]["native_metrics"]["rollout"]["constraints"]["cspace"], 0.)

    def test_selected_validation_is_restored_for_refinement_and_native_failure_stays_failure(self):
        result = self.native_candidates(flags=(False, False), native_violations=(0., 0.))
        planner, _ = self.selection_fixture((0., .03))
        _, trajectory, attempt = planner.select_candidate(result)
        self.assertEqual(trajectory.candidate_index, 0)
        self.assertFalse(attempt["native_success"])
        self.assertFalse(any(item["accepted"] for item in attempt["candidates"]))
        self.assertEqual(planner.last_validation["measurements"]["candidate_index"], 0)
        np.testing.assert_array_equal(planner.last_validation["violations"], [0., 0., 0.])

    def test_one_unexportable_seed_does_not_hide_another_verified_curve(self):
        result = self.native_candidates(flags=(False, True))
        planner, exported = self.selection_fixture((.01, 0.), errors=(0,))
        _, trajectory, attempt = planner.select_candidate(result)
        self.assertEqual(exported, [1])
        self.assertEqual(trajectory.candidate_index, 1)
        self.assertEqual(attempt["selected_seed_index"], 1)
        self.assertEqual(len(attempt["candidates"]), 2)
        planner, _ = self.selection_fixture((.01, 0.), errors=(0, 1))
        with self.assertRaisesRegex(ValueError, "export|finite sample budget"):
            planner.select_candidate(result)
# END independent native-result selection and continuation regressions.


# BEGIN exact knot-side, barrier and named-bound acceleration regressions.
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
    # BEGIN the real exporter consumes the complete scalar acceleration check.
    def test_analytic_acceleration_violation_enters_existing_candidate_residuals(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free", require_end=False),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            [[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]],
            [[0., 0., 0.], [3., 0., 0.], [3., 0., 0.]], 1.)
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        limits.acceleration[:, 0] = torch.tensor([-2., 5.], dtype=torch.float64)
        trajectory, _ = planner.export(result)
        self.assertEqual(trajectory.residuals["continuous_acceleration"], 1.)
        self.assertEqual(planner.last_validation["cubic_derivatives"]["checked_interval_s"], [0., 2.])
        self.assertEqual(planner.last_validation["measurements"]["continuous_acceleration:maximum_violation_time"], 1.)
        self.assertEqual(planner.last_validation["u"][-1], .5)
        self.assertEqual(planner.last_validation["violations"][-1], 1.)
        planner.refinement_parameters = np.empty(0)
        planner.trajopt.horizon = 4
        # Neither ordinary sample failures nor pose anchors identify this
        # knot's left acceleration. Its analytic witness must drive the same
        # existing bounded refinement path as other curve violations.
        self.assertIn(.5, planner.refinement_samples())
        self.assertFalse(all(value <= 0. for value in trajectory.residuals.values()))
    # END a zero sampled native residual cannot hide a left-limit violation.

    def test_multiple_native_seeds_cannot_be_flattened_into_one_export(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((3, 3)), np.zeros((3, 3)), .25)
        result.num_seeds = 2
        result.get_interpolated_plan = Mock(side_effect=AssertionError("must reject before reading mixed curves"))
        with self.assertRaisesRegex(ValueError, "one.*seed|selected seed"):
            planner.export(result)
        result.get_interpolated_plan.assert_not_called()

    def test_source_short_loop_peak_cannot_hide_between_native_samples(self):
        # The candidate never moves. A velocity-only subdivision therefore
        # cannot discover this required 20 mm loop unless source anchors are
        # explicitly queried on the actual exported joint curve.
        knots = np.array([0., .501, .502, .503, 1.])
        reference = np.broadcast_to(np.eye(4), (len(knots), 4, 4)).copy()
        reference[:, 0, 3] = [0., 0., .02, 0., 0.]
        phase = MotionPhase("required loop", PosePath(knots, reference, "observed narrow loop"))
        problem = cartesian_problem((phase,))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((9, 3)), np.zeros((9, 3)), .125)
        trajectory, _ = planner.export(result)
        self.assertIn(.502, planner.last_validation["u"])
        self.assertGreater(trajectory.residuals["phase:0:required loop"], .019)
        self.assertAlmostEqual(planner.last_validation["measurements"]["phase:0:position_error_m"], .02)

    def test_plateau_has_two_validation_anchors_but_one_original_event_trigger(self):
        phases = tuple(MotionPhase(f"free {index}", constant_path(), kind="free") for index in range(3))
        events = (
            PlanningEvent("close", "first", 1., 73, "close", "original first close"),
            PlanningEvent("verify", "first", 1., 79, "verification", "first grasp boundary",
                          "actual stable Holding feedback", True),
            PlanningEvent("close", "second", 2., 115, "close", "original second close"),
            PlanningEvent("verify", "second", 2., 121, "verification", "second grasp boundary",
                          "another actual Holding observation", True),
        )
        problem = cartesian_problem(phases, events)
        mapping = _ProgressMap(np.array([0., .2, .45, .75, 1.]), np.array([0., 1., 1., 2., 3.]))
        np.testing.assert_allclose(mapping.anchor_parameters([1.]), [[.2, 1.], [.45, 1.]])
        planner, result = export_fixture(problem, mapping, np.zeros((7, 3)), np.zeros((7, 3)), .2)
        trajectory, _ = planner.export(result)
        self.assertEqual(len(trajectory.events), len(events))
        self.assertEqual([(event["skill_id"], event["event_id"]) for event in trajectory.events],
                         [(event.skill_id, event.event_id) for event in events])
        first = trajectory.events[0]
        self.assertAlmostEqual(first["time"], .2*trajectory.duration)
        self.assertAlmostEqual(first["hold_until"], .45*trajectory.duration)
        self.assertAlmostEqual(trajectory.executable_until, first["time"])
        self.assertEqual(trajectory.next_verification["source_frame"], 79)
        self.assertTrue(np.any(np.isclose(.2, planner.last_validation["u"], rtol=0, atol=1e-14)))
        self.assertTrue(np.any(np.isclose(.45, planner.last_validation["u"], rtol=0, atol=1e-14)))
        # Returned barrier details cannot mutate the trajectory's authorization
        # boundary or turn a future verification into supplied evidence.
        returned = trajectory.next_verification
        returned["time"] = trajectory.duration
        self.assertAlmostEqual(trajectory.executable_until, first["time"])

    def test_no_evidence_barrier_allows_the_complete_exported_duration(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((3, 3)), np.zeros((3, 3)), .25)
        trajectory, _ = planner.export(result)
        self.assertIsNone(trajectory.next_verification)
        self.assertEqual(trajectory.executable_until, trajectory.duration)

    def test_excessive_curve_subdivision_is_rejected_before_allocation(self):
        problem = cartesian_problem((MotionPhase("free", constant_path(), kind="free"),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((2, 3)), np.array([[1e9, 0., 0.], [-1e9, 0., 0.]]), 1.)
        with patch("manipulation.planning.np.linspace", wraps=np.linspace) as allocations:
            with self.assertRaisesRegex(ValueError, "finite sample budget"):
                planner.export(result)
            allocations.assert_not_called()


# BEGIN measured-entrance necessary conditions without invented derivatives.
class InitialStatePreflightTests(unittest.TestCase):
    """CPU integration boundaries, without claiming native collision numerics."""

    @staticmethod
    def fixture(*, position=(.1, -.3, .5), velocity=(.7, -1.7, 2.4),
                native_collision=0., scene_name="robot/background", scene_value=0.):
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
            phases=(MotionPhase("free", constant_path(), kind="free"),))
        planner = _CuroboPlanner.__new__(_CuroboPlanner)
        planner.problem, planner.joint_names = problem, problem.joint_names
        planner.torch, planner.JointState = torch, SimpleNamespace
        planner._RobotState = SimpleNamespace
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
        collision = SimpleNamespace(enabled=True, setup_batch_tensors=Mock(),
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
        planner._verification_scene = SimpleNamespace(gripper=None, check_robot_coverage=Mock(return_value=[]),
            components=Mock(return_value={scene_name: torch.tensor([[scene_value]], dtype=torch.float64)}))
        planner.scene_workspace_bytes, planner.anchor_fk_workspace_bytes, planner.anchor_capacity = 64, 32, 4
        planner.reference_seeds = Mock(side_effect=AssertionError("an infeasible fixed entrance must stop before IK"))
        planner.optimize, planner.ompl_seed, planner.close = Mock(), Mock(), Mock()
        return planner, cspace, collision

    def test_checks_effective_named_limits_and_actual_fk_without_unknown_derivatives(self):
        planner, cspace, collision = self.fixture()
        result = planner.initial_state_diagnostics()
        self.assertTrue(result["passed"])
        self.assertEqual(result["failed_components"], [])
        np.testing.assert_array_equal(result["joint_bounds"]["position"]["upper"], [.2, .4, .6])
        np.testing.assert_array_equal(result["joint_bounds"]["velocity"]["upper"], [1., 2., 3.])
        measured = planner._verification_fk.compute_kinematics.call_args.args[0]
        torch.testing.assert_close(measured.position[:, 0], planner.start.position, rtol=0, atol=0)
        torch.testing.assert_close(measured.velocity[:, 0], planner.start.velocity, rtol=0, atol=0)
        for field in ("acceleration", "jerk", "dt"):
            self.assertIsNone(getattr(measured, field, None))
        cspace.forward.assert_not_called()
        planner._verification_native.compute_costs.assert_not_called()
        collision.setup_batch_tensors.assert_called_once_with(1, 1)
        collision.forward.assert_called_once()
        spheres, position, quaternion, progress, joints, names = planner._verification_scene.components.call_args.args
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
        self.assertIn("unprovided_locked_joint_velocity", result["scope"]["not_checked"])
        self.assertIn("future_trajectory_feasibility", result["scope"]["not_checked"])

    def test_each_immutable_violation_stops_before_ik_optimization_or_search(self):
        cases = (
            ("native:position", {"position": (.21, -.3, .5)}, .01),
            ("native:velocity", {"velocity": (.7, -2.3, 2.4)}, .3),
            ("native:self_collision", {"native_collision": 1.}, 1.),
            ("scene:gripper/self", {"scene_name": "gripper/self", "scene_value": 4.5}, 4.5),
            ("scene:unknown_space", {"scene_name": "unknown_space", "scene_value": np.nan}, np.nan),
        )
        scene = SimpleNamespace(describe=lambda: {"scope": "analytic entrance fixture"})
        for component, arguments, expected in cases:
            with self.subTest(component=component):
                planner, _, _ = self.fixture(**arguments)
                before = (planner.start.position.clone(), planner.start.velocity.clone())
                with patch("manipulation.planning._CuroboPlanner", return_value=planner):
                    result = plan_trajectory(planner.problem, scene, object(), device="cpu")
                self.assertFalse(result.success)
                self.assertIsNone(result.trajectory)
                self.assertEqual(result.diagnostics["failure"], "initial_state_infeasible")
                entrance = result.diagnostics["initial_state"]
                self.assertEqual(entrance["failed_components"], [component])
                actual = entrance["residuals"][component]
                if np.isfinite(expected):
                    self.assertAlmostEqual(actual, expected)
                    self.assertEqual(entrance["nonfinite_components"], [])
                else:
                    # BEGIN strict JSON must retain the nonfinite rejection.
                    self.assertIsNone(actual)
                    self.assertEqual(entrance["nonfinite_components"], [component])
                    # END strict JSON must retain the nonfinite rejection.
                json.dumps(result.diagnostics, allow_nan=False)
                self.assertEqual(result.diagnostics["attempts"], [])
                self.assertFalse(result.diagnostics["used_ompl"])
                planner.reference_seeds.assert_not_called()
                planner.optimize.assert_not_called()
                planner.ompl_seed.assert_not_called()
                planner.close.assert_called_once_with()
                torch.testing.assert_close(planner.start.position, before[0], rtol=0, atol=0)
                torch.testing.assert_close(planner.start.velocity, before[1], rtol=0, atol=0)
# END measured-entrance necessary conditions without invented derivatives.


class PlannerRetryContractTests(unittest.TestCase):
    """Exercise authorization and finite retry semantics without native GPU work."""

    @staticmethod
    def scripted_planner(native_flags, residuals):
        # Distinct solutions and exports reveal accidental candidate leakage
        # across attempts; a graph-search seed is never an accepted curve.
        native = [SimpleNamespace(success=torch.tensor([[flag]]), batch_size=1, num_seeds=1,
                                  solution=torch.full((1, 1, 4, 3), float(index+1)),
                                  seed_cost=torch.tensor([[float(index)]]), metrics=None, interpolated_metrics=None)
                  for index, flag in enumerate(native_flags)]
        trajectories = [SimpleNamespace(residuals={"required_process": value}, executable_until=.4,
                                         next_verification={"time": .4, "skill_id": "grasp", "event_id": "verify",
                                                            "requires_evidence": "actual Holding observation"})
                        for value in residuals]
        planner = SimpleNamespace(
            budget=PlanningBudget(),
            # BEGIN explicit allocation diagnostics in the CPU retry fixture.
            scene_workspace_bytes=64, anchor_fk_workspace_bytes=32, anchor_capacity=4,
            # END explicit allocation diagnostics in the CPU retry fixture.
            progress_map=_ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            ik=SimpleNamespace(config=SimpleNamespace(exit_early=False)),
            reference_ik_distinct_counts=[2, 2, 2],
            initial_seed_strategies=["reference_ik", "reference_ik_alternative"],
            # BEGIN valid measured entrance for the independent retry fixture.
            initial_state_diagnostics=Mock(return_value={"passed": True, "residuals": {},
                "failed_components": [], "scope": {"kind": "analytic retry fixture"}}),
            # END valid measured entrance for the independent retry fixture.
            reference_seeds=Mock(return_value=(torch.zeros((1, 2, 4, 3)), [2, 2, 2])),
            optimize=Mock(side_effect=native), close=Mock(), replace_schedule=Mock(),
            refinement_samples=Mock(return_value=None),
            ompl_seed=Mock(return_value=(None, None, {"success": False})),
        )
        planner.native_diagnostics = MethodType(_CuroboPlanner.native_diagnostics, planner)
        planner.select_candidate = MethodType(_CuroboPlanner.select_candidate, planner)

        def export(result):
            index = next(index for index, candidate in enumerate(native) if candidate is result)
            planner.last_validation = {"measurements": {"position_error_m": residuals[index]},
                                       "numerical_tolerances": {"position_m": 1e-4}}
            return trajectories[index], 17

        planner.export = Mock(side_effect=export)
        return planner, native, trajectories

    def run_planner(self, planner, budget):
        scene = SimpleNamespace(describe=lambda: {"scope": "explicit analytic retry fixture"})
        with patch("manipulation.planning._CuroboPlanner", return_value=planner):
            result = plan_trajectory(object(), scene, object(), budget=budget)
        planner.close.assert_called_once_with()
        return result

    def test_native_and_exported_curve_must_both_pass_without_candidate_leakage(self):
        for native_success, residual in ((False, 0.), (True, .002), (True, np.nan)):
            with self.subTest(native_success=native_success, residual=residual):
                planner, _, _ = self.scripted_planner([native_success], [residual])
                result = self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
                self.assertFalse(result.success)
                self.assertIsNone(result.trajectory)
                self.assertEqual(planner.optimize.call_count, 1)
                planner.ompl_seed.assert_not_called()

    # BEGIN diagnose the completed native seed selection, not a static budget guess.
    def test_diagnostics_report_actual_seed_policy_and_native_early_exit_configuration(self):
        for strategies, distinct in ((["reference_ik", "reference_ik_alternative"], [3, 2, 3]),
                                     (["reference_ik", "measured_state_constant"], [3, 0, 3])):
            with self.subTest(strategies=strategies):
                planner, _, _ = self.scripted_planner([True], [0.])
                planner.initial_seed_strategies = ["not selected yet"]
                planner.reference_ik_distinct_counts = []

                def select():
                    planner.initial_seed_strategies = strategies.copy()
                    planner.reference_ik_distinct_counts = distinct.copy()
                    return torch.zeros((1, 2, 4, 3)), [3, distinct[1], 3]

                planner.reference_seeds.side_effect = select
                result = self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
                self.assertTrue(result.success)
                self.assertEqual(result.diagnostics["initial_seed_strategies"], strategies)
                self.assertEqual(result.diagnostics["reference_ik_distinct_counts"], distinct)
                self.assertEqual(result.diagnostics["reference_ik_success_counts"], [3, distinct[1], 3])
                self.assertIs(result.diagnostics["ik_exit_early"], False)
    # END diagnose the completed native seed selection.

    def test_ompl_success_is_only_a_seed_and_cannot_bypass_the_final_complete_solve(self):
        planner, _, _ = self.scripted_planner([False, True], [.004, .001])
        seed = torch.full((1, 1, 4, 3), 17.)
        mapping = _ProgressMap(np.array([0., .3, 1.]), np.array([0., .1, 1.]))
        planner.ompl_seed.return_value = seed, mapping, {"success": True}
        result = self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=2))
        self.assertFalse(result.success)
        self.assertIsNone(result.trajectory)
        self.assertTrue(result.diagnostics["used_ompl"])
        self.assertEqual(planner.optimize.call_count, 2)
        self.assertIs(planner.optimize.call_args_list[1].args[0], seed)
        planner.replace_schedule.assert_called_once_with(mapping)
        self.assertEqual(planner.ompl_seed.call_count, 1)
        self.assertEqual(len(result.diagnostics["attempts"]), 2)

    def test_curve_refinement_reoptimizes_whole_candidates_with_a_hard_solve_cap(self):
        planner, native, _ = self.scripted_planner([True, True, True], [.004, .002, .001])
        planner.refinement_samples.side_effect = [np.array([.27]), np.array([.27, .53])]
        result = self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=3, search_seconds=0.))
        self.assertFalse(result.success)
        self.assertIsNone(result.trajectory)
        self.assertEqual(planner.optimize.call_count, 3)
        self.assertEqual(planner.replace_schedule.call_count, 2)
        self.assertEqual(planner.refinement_samples.call_count, 2)
        for index in (1, 2):
            seed = planner.optimize.call_args_list[index].args[0]
            torch.testing.assert_close(seed, native[index-1].solution)
            self.assertNotEqual(seed.data_ptr(), native[index-1].solution.data_ptr())
        planner.ompl_seed.assert_not_called()

    def test_first_verified_export_stops_retries_and_preserves_its_evidence_barrier(self):
        planner, _, trajectories = self.scripted_planner([True, True], [.003, 0.])
        planner.refinement_samples.return_value = np.array([.27])
        result = self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=3, search_seconds=0.))
        self.assertTrue(result.success)
        self.assertIs(result.trajectory, trajectories[1])
        self.assertEqual(planner.optimize.call_count, 2)
        self.assertEqual(result.diagnostics["executable_until"], .4)
        self.assertEqual(result.diagnostics["next_verification"]["requires_evidence"], "actual Holding observation")
        self.assertEqual(planner.refinement_samples.call_count, 1)

    def test_export_rejection_releases_native_resources(self):
        planner, _, _ = self.scripted_planner([True], [0.])
        planner.export.side_effect = ValueError("finite sample budget exceeded")
        with self.assertRaisesRegex(ValueError, "finite sample budget"):
            self.run_planner(planner, PlanningBudget(maximum_trajectory_solves=1))
        planner.close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
# END independent calculus, exact-anchor and execution-boundary regressions.
