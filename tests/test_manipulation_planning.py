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
    _fixed_spline_grid,
    _anchor_capacity,
    _planning_workspace_bytes,
    _PlanningWorkspace,
    _require_planning_memory,
    _bounded_robot_configuration,
    _native_gradient_layout_type,
    _orientation_vjp_type,
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




# BEGIN returned-candidate fixtures retain actual CPU native result slicing.
# END fixtures declare complete saved curves and no independent acceptance oracle.


class PlannerCalculusTests(unittest.TestCase):
    def test_formal_spline_clock_preserves_raw_duration_and_exact_dense_grid(self):
        for native_dt, subdivisions, count in ((.2, 32, 641), (.05, 8, 161)):
            with self.subTest(native_dt=native_dt):
                clock = _fixed_spline_grid(intervals=20, native_dt=native_dt,
                    interpolation_steps=4, interpolation_dt=.025)
                self.assertEqual(clock["valid_count"], count)
                self.assertEqual(clock["subdivisions"], subdivisions)
                self.assertEqual(clock["duration"], 80*float(np.float32(native_dt)))

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




if __name__ == "__main__":
    unittest.main()
# END native attempt-planner test migration.
