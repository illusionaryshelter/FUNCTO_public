"""CPU planner contracts for fixed-clock curves, model input and candidate evidence."""

# BEGIN current constrained-planner APIs are exercised with explicit CPU fixtures.
from copy import deepcopy
from dataclasses import asdict
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
import torch

from manipulation.planning import (
    PlanningBudget, JointTrajectory, MotionPlanningError, cubic_derivative_report,
    _ConstrainedPlanner, _ProgressMap, _fixed_spline_grid, _anchor_capacity,
    _planning_workspace_bytes, _PlanningWorkspace, _projection_tolerance,
    _require_planning_memory, _bounded_robot_configuration,
    _native_gradient_layout_type, _orientation_vjp_type,
)
from manipulation.physical_joint_domain import CubicJointDomain, JointPositionDomain
from manipulation.pin_kinematics import PinKinematics
from manipulation.planning_problem import MotionPhase, PlanningEvent, PlanningProblem, PosePath
from manipulation.unified_solver import ConstrainedSolution




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





def cartesian_kinematics(problem):
    """Build a complete declared test robot with three independent translations."""
    links = ("base", "x_link", "y_link", problem.tool_link)
    parts = ["<robot name='cartesian_fixture'>", *(f"<link name='{name}'/>" for name in links)]
    for index, name in enumerate(problem.joint_names):
        axis = ['0', '0', '0']
        axis[index] = '1'
        parts.append(f"<joint name='{name}' type='prismatic'><parent link='{links[index]}'/>"
            f"<child link='{links[index+1]}'/><axis xyz='{' '.join(axis)}'/>"
            "<limit lower='-1000000' upper='1000000' effort='100' velocity='100'/></joint>")
    model = pin.buildModelFromXML(''.join(parts)+"</robot>")
    return PinKinematics(model, frames=(problem.tool_link,), T_reference_base=problem.base_pose)


def export_fixture(problem, progress_map, position, velocity, dt):
    """Bind exact cubic export and real Pin FK; interval geometry is isolated."""
    position, velocity = np.asarray(position, dtype=float), np.asarray(velocity, dtype=float)
    controls, duration = max(2, len(position)), (len(position)-1)*dt
    knot_dt = duration/(controls-1)
    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    planner.check_request = Mock()
    planner.problem, planner.torch, planner.joint_names = problem, torch, problem.joint_names
    planner.budget = PlanningBudget(validation_spatial_step_m=.5)
    clock = {(event.skill_id, event.event_id, event.occurrence): {'event': asdict(event),
        'time': float(progress_map.anchor_parameters([event.progress])[0, 0]*duration),
        'hold_until': float(progress_map.anchor_parameters([event.progress])[-1, 0]*duration)}
        for event in problem.events}
    planner.progress_map = _ProgressMap(progress_map.u, progress_map.s, clock)
    planner.anchors, planner.action_horizon = problem.anchors, controls
    planner.formal_clock = {'knot_dt': knot_dt, 'duration': duration}
    planner.finger_trajectory = None
    bounds = torch.tensor([[-1e6, 1e6]]*4, dtype=torch.float64)[:, :, None].repeat(1, 1, 3)
    planner.physical_domain = CubicJointDomain(bounds, torch.tensor(problem.start_q),
        torch.tensor(problem.start_dq), knot_dt, controls)
    planner.pin_kinematics = cartesian_kinematics(problem)
    limits = SimpleNamespace(joint_names=list(problem.joint_names),
        **{name: bounds[index] for index, name in enumerate(('position','velocity','acceleration','jerk'))})
    planner._verification_native = SimpleNamespace(get_cost=lambda name: SimpleNamespace(
        config=SimpleNamespace(joint_limits=limits)))
    planner.scene = SimpleNamespace(gripper=None, owners=(), contacts=(), required_progress_anchors=(),
        last_query_report={}, evaluate=lambda spheres,p,r,s,q,names,**kwargs: torch.zeros_like(s),
        robot_model=SimpleNamespace(geometry_names=(), joint_names=problem.joint_names),
        estimated_workspace_allocation=lambda *args: dict(cpu_resident=0, cpu_pending_shared=0,
            cpu_live=0, gpu_shared=0, gpu_live=0, gpu_retained=0))
    planner.robot_metadata = {'active_joint_names': list(problem.joint_names), 'links': 1}
    planner.last_validation, planner.delivery_candidate = None, None
    planner._candidate_geometry = Mock(return_value={'query_points': 0, 'hard_scene_residual': 0.,
        'scene_categories': {}, 'scene_components': {}, 'confirmed_intersections': [],
        'proved_intervals': [], 'unresolved_intervals': []})
    planner._candidate_metrics = Mock(return_value=({'accepted': True, 'residuals': {}}, np.array([0., 1.]), 0, (planner.scene, [])))
    actions = np.broadcast_to(position[-1], (controls, 3)).copy()
    return planner, ConstrainedSolution(actions, {'success': True, 'backend': 'declared CPU fixture'})




class PlannerCalculusTests(unittest.TestCase):
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
        # BEGIN a request keeps its maximum spacing until actual clock binding.
        self.assertEqual(PlanningBudget(interpolation_dt=.03).interpolation_dt, .03)


    def test_initializer_plateau_inverse_is_explicitly_earliest(self):
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
        planner.check_request = Mock()
        planner.progress_map = _ProgressMap(np.array([0., .2, .4, 1.]), np.array([0., 1., 1., 2.]))
        self.assertEqual(planner._seed_u(1.), .2)


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



# BEGIN the fixture models one reused native output buffer and explicit predecessor state.
def local_reference_fixture(samples=None, flags=None):
    """Bind real sample fitting around one native-shaped local solver."""
    samples = (np.array([[.1+.01*i, .02*i, -.01*i] for i in range(6)])
               if samples is None else np.asarray(samples, dtype=float))
    flags = np.ones(len(samples), dtype=bool) if flags is None else np.asarray(flags, dtype=bool)
    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
    planner.check_request, planner.torch, planner.JointState = Mock(), torch, SimpleNamespace
    planner.budget = PlanningBudget(knots=len(samples), ik_seeds=4, trajectory_seeds=2)
    planner.problem = cartesian_problem((MotionPhase('reference', constant_path()),))
    planner.problem.start_dq = np.array([.1, -.2, .3])
    planner.joint_names = planner.problem.joint_names
    planner.start = SimpleNamespace(position=torch.tensor(planner.problem.start_q)[None],
        velocity=torch.tensor(planner.problem.start_dq)[None], dt=torch.tensor([0.]))
    planner.progress_map = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    planner.source_progress_map, planner.reference_time_map = planner.progress_map, None
    planner.formal_clock = {'knot_dt': .2, 'duration': .2*(len(samples)-1)}
    planner.action_horizon = len(samples)
    planner.finger_trajectory = None
    planner.request_work = {'ik_queries': 0}
    planner.request_work_baseline = {'ik_queries': 0}
    planner.device_cfg = SimpleNamespace(to_device=lambda value: torch.as_tensor(value, dtype=torch.float64))
    planner._goal = lambda time: time
    calls, buffer = [], torch.zeros((1, 1, 3), dtype=torch.float64)
    def solve(goal, *, current_state, seed_config, return_seeds):
        index = len(calls) % len(samples)
        calls.append({'time_s': goal, 'q': current_state.position.clone(),
            'dq': current_state.velocity.clone(), 'dt': current_state.dt.clone(),
            'seed': seed_config.clone(), 'return_seeds': return_seeds})
        buffer.copy_(torch.tensor(samples[index])[None, None])
        return SimpleNamespace(js_solution=SimpleNamespace(position=buffer,
            velocity=torch.full_like(buffer, 999.)),
            success=torch.tensor([[flags[index]]]),
            position_error=torch.tensor([[1e-6 if flags[index] else .02]]),
            rotation_error=torch.tensor([[1e-5 if flags[index] else .1]]))
    planner.reference_ik = SimpleNamespace(config=SimpleNamespace(num_seeds=1),
        solve_pose=solve, get_active_js=lambda state: state)
    planner.ik = SimpleNamespace(config=SimpleNamespace(num_seeds=4), solve_pose=Mock())
    bounds = torch.tensor([[-2., 2.], [-2., 2.], [-20., 20.], [-100., 100.]],
        dtype=torch.float64)[:, :, None].repeat(1, 1, 3)
    planner.physical_domain = CubicJointDomain(bounds, planner.start.position, planner.start.velocity,
        .2, len(samples))
    fitting = Mock(wraps=planner.physical_domain.fit_samples)
    planner.physical_domain.fit_samples = fitting
    return planner, fitting, calls


class LocalReferenceInitializationTests(unittest.TestCase):
    def test_one_fit_and_one_actual_seed_preserve_original_measured_entry(self):
        planner, fitting, calls = local_reference_fixture()
        q0, dq0 = planner.start.position.clone(), planner.start.velocity.clone()
        seeds, success = planner.reference_seeds()
        self.assertEqual(tuple(seeds.shape), (1, 1, 6, 3))
        self.assertEqual(success, [1]*6)
        self.assertEqual(planner.seed_strategies, ['local_reference_ik'])
        fitting.assert_called_once()
        np.testing.assert_allclose(planner.physical_domain.evaluate(seeds[0, 0], [0.])[0], q0[0])
        np.testing.assert_allclose(planner.physical_domain.evaluate(seeds[0, 0], [0.], 1)[0], dq0[0], atol=1e-14)
        torch.testing.assert_close(planner.start.position, q0, rtol=0, atol=0)
        torch.testing.assert_close(planner.start.velocity, dq0, rtol=0, atol=0)
        self.assertEqual(planner.request_work['ik_queries'], 6)
        planner.ik.solve_pose.assert_not_called()

    def test_predecessor_velocity_and_real_nonuniform_dt_survive_native_buffer_reuse(self):
        planner, fitting, calls = local_reference_fixture()
        seeds, _ = planner.reference_seeds()
        samples = fitting.call_args.args[1]
        times = planner.physical_domain._curve_map.reference_parameters*planner.formal_clock['duration']
        previous = planner.start.position.numpy()[0]
        velocity = planner.start.velocity.numpy()[0]
        last_time = 0.
        for index, call in enumerate(calls):
            interval = times[index]-last_time
            np.testing.assert_allclose(call['q'][0], previous)
            np.testing.assert_allclose(call['dq'][0], velocity)
            np.testing.assert_allclose(call['dt'], [interval])
            np.testing.assert_allclose(call['seed'][0, 0], previous)
            self.assertEqual(call['return_seeds'], 1)
            velocity = (samples[index]-previous)/interval
            previous, last_time = samples[index], times[index]
        self.assertNotEqual(float(calls[0]['dt']), float(calls[-1]['dt']))
        np.testing.assert_allclose(planner.physical_domain.evaluate(seeds[0, 0],
            planner.physical_domain._curve_map.reference_parameters), samples, atol=2e-14)

    def test_new_sequence_uses_measured_entry_after_previous_sequence_completed(self):
        planner, _, calls = local_reference_fixture()
        first, _ = planner.reference_seeds()
        second, _ = planner.reference_seeds()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(calls[6]['q'], planner.start.position)
        torch.testing.assert_close(calls[6]['dq'], planner.start.velocity)
        self.assertEqual(len(planner.reference_ik_queries), 6)
# END native flags and finite approximations do not select or duplicate ordinary branches.



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
    def robot_metadata():
        return {"active_joint_names": tuple(f"q{index}" for index in range(16)),
                "links": 64, "spheres": 512}


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
                    checker = SimpleNamespace(base_frame="base", urdf_xml=frozen,
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
                    planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
                    planner.check_request = Mock()
                    with patch("torch.cuda._lazy_init", side_effect=AssertionError("source preflight must stay CPU-only")), \
                         patch("manipulation.planning._bounded_robot_configuration", return_value=(robot, {})), \
                         patch("manipulation.planning._reference_entrance", return_value=(np.eye(4), np.zeros(6))), \
                         patch("manipulation.planning._planning_workspace_bytes", side_effect=RuntimeError("source admitted")) as storage:
                        with self.assertRaisesRegex(RuntimeError if admitted else ValueError,
                                "source admitted" if admitted else "source URDF|joint locks|ancestor"):
                            planner._initialize(problem, scene, robot, PlanningBudget(), "cuda:0", robot_validator=validator)
                    self.assertEqual(storage.call_count, int(admitted))
                    validator.validate_planning_binding.assert_called_once_with(problem, hand)


    def test_torch_cache_covers_tensors_but_not_the_other_native_reserve(self):
        native_reserve = 2*1024**3
        cuda = SimpleNamespace(mem_get_info=Mock(return_value=(6*1024**3, 8*1024**3)),
                               memory_reserved=Mock(return_value=400), memory_allocated=Mock(return_value=101))
        runtime = SimpleNamespace(cuda=cuda)
        workspace = _PlanningWorkspace(0, 0, 0, 200, 600)
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


    def test_projection_accuracy_respects_the_strictest_source_mechanism(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-6)
        model = SimpleNamespace(position_tolerance=1e-6, rotation_tolerance=2e-6)
        problem.mechanisms = (SimpleNamespace(model=model),)
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-7)
        model.rotation_tolerance = 1e-8
        self.assertAlmostEqual(_projection_tolerance(problem), 1e-9)



class RobotAllocationPreflightTests(unittest.TestCase):
    """Exercise joint/link budgets without constructing a native GPU model."""

    def setUp(self):
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from test_robot_geometry_check import URDF
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name)/"kinematics.urdf"
        path.write_text(URDF)
        self.robot = {"robot_cfg": {"kinematics": {"urdf_path": str(path),
            "base_link": "base", "lock_joints": {"right_slide": 0.}}}}
        self.problem = SimpleNamespace(tool_link="tool", joint_names=("left_slide",))
        self.scene = SimpleNamespace(required_robot_links=("left", "right"))

    def test_pure_robot_metadata_is_valid_without_initializing_cuda(self):
        initialized = torch.cuda.is_initialized()
        with patch("torch.cuda.init", side_effect=AssertionError("metadata must remain on CPU")):
            config, info = _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget())
        self.assertEqual(info["unlocked_dof"], 2)
        self.assertEqual(info["active_joint_names"], ["left_slide"])
        self.assertEqual(info["links"], 4)
        self.assertEqual(info["spheres"], 0)
        native = config["robot_cfg"]["kinematics"]
        self.assertEqual(native["tool_frames"], ["tool"])
        self.assertFalse(native["load_collision_spheres"])
        self.assertEqual(native["collision_spheres"], {})
        self.assertEqual(torch.cuda.is_initialized(), initialized)

    def test_active_joint_identity_mismatch_fails_preflight(self):
        wrong = SimpleNamespace(tool_link="tool", joint_names=("unknown_joint",))
        with self.assertRaisesRegex(ValueError, "named joint/link"):
            _bounded_robot_configuration(self.robot, wrong, self.scene, PlanningBudget())

    def test_registered_geometry_requires_links_in_the_supplied_tree(self):
        scene = SimpleNamespace(required_robot_links=("unknown_link",))
        with self.assertRaisesRegex(ValueError, "joint/link"):
            _bounded_robot_configuration(self.robot, self.problem, scene, PlanningBudget())

    def test_limits_cover_the_unlocked_tree_and_every_link(self):
        with self.assertRaisesRegex(ValueError, "joint/link"):
            _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget(max_robot_dof=1))
        with self.assertRaisesRegex(ValueError, "joint/link"):
            _bounded_robot_configuration(self.robot, self.problem, self.scene, PlanningBudget(max_robot_links=3))

    def test_native_tree_overrides_do_not_change_registered_kinematics(self):
        robot = deepcopy(self.robot)
        robot["robot_cfg"]["kinematics"]["extra_links"] = {"extra": {}}
        with self.assertRaisesRegex(ValueError, "without extra links"):
            _bounded_robot_configuration(robot, self.problem, self.scene, PlanningBudget())


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
    def qualify(self, planner, result, trajectory=None):
        if trajectory is None:
            trajectory = planner._controls_curve(result.controls)
        # BEGIN this fixture supplies the independent ordinary score explicitly.
        # These cases exercise exact polynomial bounds, event timing and
        # delivery reuse. Reference-cost numerics have their own native tests.
        returned, report, points = planner.evaluate_candidate(trajectory,
            record_work=Mock(), objective={"fixed_objective": 0.})
        self.assertEqual(report["fixed_objective"], 0.)
        # END polynomial qualification retains its separate hard residuals.
        self.assertIs(returned, trajectory)
        self.assertEqual(points, 0)
        return trajectory, report


    def test_analytic_acceleration_violation_enters_existing_candidate_residuals(self):
        problem = cartesian_problem((MotionPhase("free", constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            [[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]],
            [[0., 0., 0.], [3., 0., 0.], [3., 0., 0.]], 1.)
        limits = planner._verification_native.get_cost("cspace").config.joint_limits
        limits.acceleration[:, 0] = torch.tensor([-2., 5.], dtype=torch.float64)
        analytic = JointTrajectory(problem.joint_names, np.array([0., 1., 2.]),
            np.array([[0., 0., 0.], [1., 0., 0.], [4., 0., 0.]]),
            np.array([[0., 0., 0.], [3., 0., 0.], [3., 0., 0.]]),
            np.array([0., 1.]), np.array([0., 1.]), (), (), {})
        trajectory, report = self.qualify(planner, result, analytic)
        self.assertFalse(report["accepted"])
        self.assertAlmostEqual(trajectory.residuals["continuous_acceleration"], 1.)
        self.assertAlmostEqual(report["residuals"]["continuous_acceleration"], 1.)
        self.assertIsNone(planner.last_validation)
        planner._candidate_geometry.assert_not_called()


    def test_reference_short_loop_does_not_require_motion_or_a_hard_anchor(self):
        knots = np.array([0., .501, .502, .503, 1.])
        reference = np.broadcast_to(np.eye(4), (len(knots), 4, 4)).copy()
        reference[:, 0, 3] = [0., 0., .02, 0., 0.]
        problem = cartesian_problem((MotionPhase("reference loop", PosePath(knots, reference, "reference")),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
                                         np.zeros((9, 3)), np.zeros((9, 3)), .125)
        trajectory, report = self.qualify(planner, result)
        self.assertTrue(report["accepted"])
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
        trajectory, report = self.qualify(planner, result)
        self.assertTrue(report["accepted"])
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
        trajectory, report = self.qualify(planner, result)
        self.assertTrue(report["accepted"])
        self.assertIsNone(trajectory.next_verification)
        self.assertEqual(trajectory.executable_until, trajectory.duration)
        validation = planner.last_validation
        calls = planner._candidate_geometry.call_count
        with patch.object(planner, "_controls_curve", side_effect=AssertionError("delivery must reuse the curve")), \
             patch.object(planner, "evaluate_candidate", side_effect=AssertionError("delivery must reuse validation")):
            delivered, points = planner.export(result, report)
        self.assertIs(delivered, trajectory)
        self.assertIs(planner.last_validation, validation)
        self.assertEqual(points, 0)
        self.assertEqual(planner._candidate_geometry.call_count, calls)
        self.assertEqual(calls, 1)



# END retained checks use the actual constrained planner and exact saved curves.


# BEGIN independent constrained solutions retain their own controls and termination.
class PlannerCandidateSelectionTests(unittest.TestCase):
    @staticmethod
    def solutions(flags=(False, True)):
        return [ConstrainedSolution(np.full((4, 3), index+1., dtype=float),
            {'success': bool(flag), 'status': 0 if flag else 9,
             'message': 'declared numerical termination', 'backend': 'scipy.optimize.SLSQP'})
            for index, flag in enumerate(flags)]

    def test_library_failure_still_receives_saved_curve_qualification(self):
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
        planner.seed_strategies, planner.scene = ['reference', 'alternate'], object()
        planner.last_validation = None
        planner._controls_curve = Mock(side_effect=lambda controls: float(controls[0, 0]))
        planner._candidate_objective = Mock(side_effect=lambda curve, scene: {'fixed_objective': curve})
        def qualify(curve, *, record_work, objective):
            record_work(passes=1)
            record_work(points=3)
            return curve, {'accepted': True, 'residuals': {}, **objective}, 3
        planner.evaluate_candidate = Mock(side_effect=qualify)
        work = Mock()
        selected, curve, report = planner.select_candidate(self.solutions(), record_work=work, on_candidate=Mock())
        self.assertEqual(curve, 1.)
        self.assertTrue(report['accepted'])
        self.assertFalse(report['optimizer']['success'])
        self.assertEqual(report['selected_seed_index'], 0)
        # BEGIN first complete qualification ends the original candidate stream.
        self.assertEqual(planner._controls_curve.call_count, 1)
        self.assertEqual(planner.evaluate_candidate.call_count, 1)
        self.assertEqual(work.call_args_list, [unittest.mock.call(passes=1), unittest.mock.call(points=3)])
        # END library success is still independent of saved-curve qualification.
        self.assertIs(planner.delivery_candidate, curve)
        np.testing.assert_array_equal(selected.controls, np.ones((4, 3)))

    def test_exported_curve_and_report_own_values_after_solution_storage_changes(self):
        problem = cartesian_problem((MotionPhase('free', constant_path()),))
        planner, result = export_fixture(problem, _ProgressMap(np.array([0., 1.]), np.array([0., 1.])),
            np.zeros((4, 3)), np.zeros((4, 3)), .25)
        trajectory = planner._controls_curve(result.controls)
        before = trajectory.sample([0., .2, .7])[0].copy()
        result.controls.fill(7.)
        np.testing.assert_array_equal(trajectory.sample([0., .2, .7])[0], before)

    def test_nonfinite_candidate_is_reported_and_other_candidate_is_checked(self):
        from test_reference_candidate_ranking import planner as selection_fixture
        candidates = self.solutions()
        candidates[0].controls[0, 0] = np.nan
        planner = selection_fixture([0., 1.], [0., 0.])
        _, curve, report = planner.select_candidate(candidates, record_work=Mock(), on_candidate=Mock())
        self.assertEqual(curve, 1)
        self.assertTrue(report['accepted'])
        self.assertFalse(report['candidates'][0]['accepted'])
        self.assertTrue(np.isinf(report['candidates'][0]['residuals']['finite_controls']))
        planner._controls_curve.assert_called_once()


class SerialAllocationTests(unittest.TestCase):
    def test_complete_reference_timeline_uses_only_declared_task_anchors(self):
        problem = RequiredAnchorAllocationTests.timeline_problem()
        self.assertEqual(sum(len(phase.reference.progress)-1 for phase in problem.phases)+1, 319)
        np.testing.assert_array_equal(problem.anchors, [0., 2., 3.])
        mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 3.]))
        np.testing.assert_allclose(mapping.anchor_parameters(problem.anchors), [[0., 0.], [2/3, 2.], [1., 3.]])
        self.assertEqual(_anchor_capacity(mapping, problem.anchors, PlanningBudget(max_task_anchors=3)), 3)
        with self.assertRaisesRegex(ValueError, 'Task anchors'):
            _anchor_capacity(mapping, problem.anchors, PlanningBudget(max_task_anchors=2))

    def test_excess_scene_anchors_stop_before_robot_or_ik_construction(self):
        problem = RequiredAnchorAllocationTests.timeline_problem()
        scene = SimpleNamespace(validate_problem=Mock(), required_progress_anchors=np.linspace(0., 3., 129),
            reference_frame=problem.reference_frame, gripper=None)
        validator = SimpleNamespace(validate_planning_binding=Mock())
        with patch('manipulation.planning._bounded_robot_configuration') as robot, \
             patch('manipulation.planning._ik_manager_type') as manager:
            with self.assertRaisesRegex(ValueError, 'anchors'):
                _ConstrainedPlanner(problem, scene, object(), PlanningBudget(
                    max_validation_points=128, max_task_anchors=128), 'cuda:0', robot_validator=validator)
        robot.assert_not_called()
        manager.assert_not_called()

    def test_one_cpu_query_and_native_ik_transfer_share_immutable_fields(self):
        estimator = Mock(return_value={'gpu_retained': 5, 'gpu_live': 150,
            'cpu_resident': 17, 'cpu_pending_shared': 19, 'cpu_live': 31,
            'gpu_shared': 23, 'gpu_graph_live': 60})
        scene = SimpleNamespace(required_progress_anchors=(.5,), owners=(),
            robot_model=SimpleNamespace(joint_names=tuple(range(5)), geometry_names=('body',)),
            estimated_workspace_allocation=estimator)
        budget = PlanningBudget(max_scene_gpu_workspace_bytes=178)
        metadata = {'active_joint_names': tuple(range(3)), 'links': 8, 'spheres': 0}
        workspace = _planning_workspace_bytes(scene, metadata, budget, 7, clock_nodes=2)
        states = max((budget.knots+4)*budget.interpolation_steps+1+7, budget.max_validation_points)
        self.assertEqual(estimator.call_args_list,
                         [unittest.mock.call(1, 3*states), unittest.mock.call(budget.ik_seeds, 3)])
        self.assertEqual(workspace.cpu_resident_bytes, 17)
        self.assertEqual(workspace.cpu_pending_shared_bytes, 19)
        self.assertGreater(workspace.cpu_live_bytes, 31+8*states*8*(7+6+3*6*5))
        self.assertEqual(workspace.pending_cpu_bytes, 19+workspace.cpu_live_bytes)
        self.assertEqual(workspace.gpu_bytes, 178)
        self.assertEqual(workspace.anchor_fk_bytes, 0)
        with self.assertRaisesRegex(ValueError, 'point-query'):
            _planning_workspace_bytes(scene, metadata, PlanningBudget(max_scene_gpu_workspace_bytes=177), 7, clock_nodes=2)

    def test_actual_shared_fields_count_once_and_larger_validation_capacity_costs_memory(self):
        from dataclasses import replace
        from manipulation.online_geometry import OnlinePlanningScene
        from manipulation.observed_mapping import FrozenObservedField
        field = lambda size: FrozenObservedField(np.ones((size,)*3), np.ones((size,)*3, bool),
            weights=np.ones((size,)*3), origin=[0.]*3, voxel_size=.01,
            name='shared', source='analytic observation fixture')
        world = OnlinePlanningScene.__new__(OnlinePlanningScene)
        world.required_progress_anchors = (0., 1.)
        world.required_robot_links = ('body', 'tool')
        world.contacts = ()
        # BEGIN this scene has resident observations and unbuilt pair/sample buffers.
        world.robot_validator, world._objective_material_samples = None, None
        # END pending allocations are distinct from the already observed fields.
        world.background = field(2)
        world.owners = tuple(SimpleNamespace(query=world.background, owner_id=f'owner{i}',
            pose_path=SimpleNamespace(progress=(0., 1.))) for i in range(2))
        world.robot_model = SimpleNamespace(geometry_names=('body','tool'), joint_names=('a','b'),
            native_query_storage_nbytes=131072, native_query_resident_nbytes=0,
            native_query_temporary_nbytes=262144, _native_pair_table=None)
        budget = PlanningBudget()
        metadata = {'active_joint_names': ('a','b'), 'links': 4, 'spheres': 0}
        initialized = torch.cuda.is_initialized()
        baseline = _planning_workspace_bytes(world, metadata, budget, 22, clock_nodes=2)
        self.assertEqual(baseline.cpu_resident_bytes, world.background.storage_nbytes)
        self.assertGreaterEqual(baseline.cpu_pending_shared_bytes, 131072)
        world.owners = ()
        self.assertEqual(_planning_workspace_bytes(world, metadata, budget, 22, clock_nodes=2).cpu_resident_bytes,
                         baseline.cpu_resident_bytes)
        larger = field(4)
        world.owners = (SimpleNamespace(query=larger, owner_id='owner', pose_path=SimpleNamespace(progress=(0., 1.))), SimpleNamespace(query=larger, owner_id='owner', pose_path=SimpleNamespace(progress=(0., 1.))))
        expanded = _planning_workspace_bytes(world, metadata, budget, 22, clock_nodes=2)
        self.assertGreaterEqual(expanded.cpu_resident_bytes-baseline.cpu_resident_bytes, larger.storage_nbytes)
        world.owners = (SimpleNamespace(query=larger, owner_id='owner', pose_path=SimpleNamespace(progress=(0., 1.))),)
        self.assertEqual(_planning_workspace_bytes(world, metadata, budget, 22, clock_nodes=2).cpu_resident_bytes,
                         expanded.cpu_resident_bytes)
        denser = _planning_workspace_bytes(world, metadata, replace(budget, max_validation_points=8192), 22, clock_nodes=2)
        self.assertGreater(denser.cpu_live_bytes, expanded.cpu_live_bytes)
        self.assertEqual(denser.gpu_bytes, expanded.gpu_bytes)
        self.assertEqual(torch.cuda.is_initialized(), initialized)

    def test_close_releases_native_ik_cycle_and_is_idempotent(self):
        import gc
        import weakref
        destroyed = []
        class CyclicSolver:
            def __init__(self):
                self.core = SimpleNamespace(callback=self.destroy)
            def destroy(self):
                destroyed.append('ik')
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
        planner.ik = CyclicSolver()
        planner.reference_ik = CyclicSolver()
        planner._caller_precision = None
        reference = weakref.ref(planner.ik)
        local_reference = weakref.ref(planner.reference_ik)
        with patch('manipulation.planning.gc.collect', wraps=gc.collect) as collect:
            planner.close()
            self.assertIsNone(reference())
            self.assertIsNone(local_reference())
            self.assertIsNone(planner.ik)
            self.assertIsNone(planner.reference_ik)
            collect.assert_called_once_with()
            planner.close()
        self.assertEqual(destroyed, ['ik', 'ik'])


class InitialStatePreflightTests(unittest.TestCase):
    @staticmethod
    def fixture(*, position=(.1,-.3,.5), velocity=(.7,-1.7,2.4), self_collision=0., scene_value=0.):
        problem = cartesian_problem((MotionPhase('free', constant_path()),))
        problem.start_q, problem.start_dq = np.asarray(position), np.asarray(velocity)
        problem.base_pose[:3, :3] = Rotation.from_euler('z', .8).as_matrix()
        problem.base_pose[:3, 3] = [.8,-.6,.2]
        problem.tool_transform[:3, 3] = [.02,-.03,.04]
        planner = _ConstrainedPlanner.__new__(_ConstrainedPlanner)
        planner.problem, planner.joint_names = problem, problem.joint_names
        planner.torch, planner.check_request, planner.budget = torch, Mock(), PlanningBudget()
        planner.progress_map = _ProgressMap(np.array([0.,1.]), np.array([0.,1.]))
        planner.formal_clock = {'knot_dt': .2}
        limits = SimpleNamespace(joint_names=['cartesian_z','cartesian_x','cartesian_y'],
            position=torch.tensor([[-.6,-.2,-.4],[.6,.2,.4]], dtype=torch.float64),
            velocity=torch.tensor([[-3.,-1.,-2.],[3.,1.,2.]], dtype=torch.float64),
            acceleration=torch.tensor([[-100.]*3,[100.]*3], dtype=torch.float64),
            jerk=torch.tensor([[-1000.]*3,[1000.]*3], dtype=torch.float64))
        planner._verification_native = SimpleNamespace(get_cost=lambda name: SimpleNamespace(config=SimpleNamespace(joint_limits=limits)))
        model = cartesian_kinematics(problem).model
        planner.robot_validator = SimpleNamespace(checker=SimpleNamespace(model=model))
        def evaluate(spheres, p, r, progress, *args, components_out, **kwargs):
            components_out['collision/robot/self'] = torch.full_like(progress, self_collision)
            components_out['collision/robot/background'] = torch.full_like(progress, scene_value)
            return sum(components_out.values(), torch.zeros_like(progress))
        planner.scene = SimpleNamespace(gripper=None, required_progress_anchors=(), contacts=(),
            last_query_report={}, robot_model=SimpleNamespace(_geometries={}), evaluate=Mock(side_effect=evaluate))
        planner._bind_computation = Mock()
        planner._bind_physical_domains()
        planner.pin_kinematics.states = Mock(wraps=planner.pin_kinematics.states)
        planner.reference_seeds, planner.optimize = Mock(), Mock()
        return planner

    def test_original_named_feedback_drives_real_pin_and_model_limits(self):
        planner = self.fixture()
        result = planner.initial_state_diagnostics()
        self.assertTrue(result['passed'])
        q, dq, names = planner.pin_kinematics.states.call_args.args
        np.testing.assert_array_equal(q.numpy()[0,0], planner.problem.start_q)
        np.testing.assert_array_equal(dq.numpy()[0,0], planner.problem.start_dq)
        self.assertEqual(names, planner.joint_names)
        spheres, position, quaternion, progress, joints, names = planner.scene.evaluate.call_args.args
        expected = planner.problem.base_pose[:3,:3] @ (planner.problem.start_q+planner.problem.tool_transform[:3,3]) + planner.problem.base_pose[:3,3]
        np.testing.assert_allclose(position[0,0], expected, atol=1e-14, rtol=0)
        self.assertEqual(spheres.shape, (1,1,0,4))
        torch.testing.assert_close(progress, torch.zeros((1,1), dtype=torch.float64), rtol=0, atol=0)
        np.testing.assert_array_equal(planner.physical_domain.bounds[0].numpy(), [[-.2,-.4,-.6],[.2,.4,.6]])

    def test_scene_components_and_observed_report_belong_to_the_same_query(self):
        planner = self.fixture()
        final_report = {'candidate_states':1, 'states':[{'pair':'robot/reference','passed':False}]}
        def evaluate(spheres,p,r,progress,*args,components_out,**kwargs):
            planner.scene.last_query_report = final_report
            components_out.update({'collision/robot/reference':torch.full_like(progress,.25),
                'scene_update/background/robot':torch.full_like(progress,.75),
                'unknown/forbidden/robot':torch.zeros_like(progress)})
            return sum(components_out.values(), torch.zeros_like(progress))
        planner.scene.evaluate.side_effect = evaluate
        result = planner.initial_state_diagnostics()
        planner.scene.evaluate.assert_called_once()
        planner.pin_kinematics.states.assert_called_once()
        self.assertEqual(result['failed_components'], ['scene'])
        self.assertEqual(result['residuals']['scene'], 1.)
        self.assertEqual(result['scene_components'], {
            'collision/robot/reference':{'maximum':.25,'time_s':0.,'progress':0.},
            'scene_update/background/robot':{'maximum':.75,'time_s':0.,'progress':0.}})
        self.assertEqual(result['observed_geometry'], final_report)
        final_report['states'][0]['passed'] = True
        self.assertFalse(result['observed_geometry']['states'][0]['passed'])

    def test_each_immutable_violation_keeps_original_binary64_feedback(self):
        for component, arguments in (
            ('position', {'position':(.2000000001,-.3,.5)}),
            ('velocity', {'velocity':(.7,-2.3,2.4)}),
            ('scene', {'self_collision':1.}), ('scene', {'scene_value':1.}),
            ('scene', {'scene_value':np.nan})):
            with self.subTest(component=component):
                planner = self.fixture(**arguments)
                q, dq = planner.problem.start_q.copy(), planner.problem.start_dq.copy()
                result = planner.initial_state_diagnostics()
                self.assertFalse(result['passed'])
                self.assertIn(component, result['failed_components'])
                np.testing.assert_array_equal(planner.physical_domain._curve_map.start_q, q)
                np.testing.assert_array_equal(planner.physical_domain._curve_map.start_dq, dq)
                planner.reference_seeds.assert_not_called()
                planner.optimize.assert_not_called()
# END model input, serial allocations and candidate qualification use current APIs.
