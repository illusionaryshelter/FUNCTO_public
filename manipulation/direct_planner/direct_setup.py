"""Bind the direct numerical planner from the existing robot configuration.

The maintained cuRobo configuration loader supplies named joint limits. The
scene's official Pinocchio model supplies all FK; the local SciPy initializer
and IPOPT solve consume the resulting physical domain. Request binding owns
the reference/event clock, calibrated entrance and declared hand polynomial.
"""

from dataclasses import asdict
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import torch

from manipulation.planning import (
    _ProgressMap, _initial_timing, _continuation_timing, reference_progress_clock,
    _fixed_spline_grid, _bind_event_clock, _phase_aligned_parameters,
    _anchor_capacity, _bounded_robot_configuration, _reference_entrance,
    _planning_workspace_bytes, _require_planning_host_memory, _require_planning_memory,
    cubic_derivative_report,
)
from manipulation.physical_joint_domain import (
    CubicJointDomain, JointPositionDomain, cubic_position_report,
)
from manipulation.timed_reference import BoundToolReference
from .differentiable_kinematics import DifferentiablePinKinematics


class DirectSetup:
    """Own the inputs actually consumed by the direct trajectory calculation.

    Compose before _ConstrainedPlanner, which supplies curve export, candidate
    selection and original entrance checks. This class owns construction,
    rebinding and precision restoration without inverse-kinematics workspaces.
    """

    def __init__(self, problem, scene, robot, budget, device, *, robot_validator,
                 continuation=None, goal_identities=None):
        from curobo._src.types.device_cfg import DeviceCfg
        self.torch = torch
        self._caller_precision = (torch.backends.cuda.matmul.allow_tf32,
                                  torch.backends.cudnn.allow_tf32)
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device_cfg = DeviceCfg(device)
        self.robot_validator = robot_validator
        self.check_request = lambda: None
        try:
            self._bind_request_clock(problem, scene, budget, continuation, goal_identities)
            self._bind_robot_configuration(robot)
            self._bind_request_computation()
        except BaseException:
            self.close()
            raise

    def _bind_request_clock(self, problem, scene, budget, continuation, goal_identities):
        """Bind original task/reference/hand clocks from the measured entrance."""
        self.robot_validator.validate_planning_binding(problem, scene.gripper)
        scene.validate_problem(problem, goal_identities=goal_identities)
        budget, source = (_initial_timing(problem, budget) if continuation is None else
                          _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.r_[problem.anchors, scene.required_progress_anchors])
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        # Preserve the original bounded input contract before binding a clock.
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.end_progress)):
            raise ValueError('Required source knots and scene anchors must fit the planning horizon')
        if problem.reference_frame != scene.reference_frame:
            raise ValueError('Problem and scene require the same calibrated reference frame')
        mapping, reference_times = source, None
        if continuation is None:
            mapping, reference_times = reference_progress_clock(source, budget.knots,
                stop_progress=problem.stop_progress, anchors=anchors)
        grid = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            before = mapping
            mapping = _bind_event_clock(problem, mapping, grid['duration'],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            reference_times = _ProgressMap(_phase_aligned_parameters(
                before, mapping, reference_times.u), reference_times.s)
        self.anchor_capacity = _anchor_capacity(mapping, anchors, budget)
        hand = scene.gripper
        if hand is not None:
            hand.bind_clock(grid['duration'], mapping)
        self.problem, self.scene, self.budget = problem, scene, budget
        self.source_progress_map, self.reference_time_map = source, reference_times
        self.progress_map, self.formal_clock, self.anchors = mapping, grid, anchors
        self.finger_trajectory = None if hand is None else hand.finger_trajectory

    def _bind_robot_configuration(self, robot):
        """Decode the same official joint order, limits and motion-cost values.

        KinematicsCfg is the maintained loader previously called inside both
        native IK constructors. Its get_joint_limits result is also returned
        by RobotStateTransition.get_state_bounds; the cost manager only clones
        it. Reading it here retains the same configured float32 limits before
        the physical domain's original conversion to binary64.

        The robot and fixed end effector retain their official model because
        their geometry and component relations are reusable across tasks.
        Calibration and named joint feedback place that model. Task entities
        continue to use the scene's observed TSDF and support representation.
        """
        from curobo.content import get_assets_path
        from curobo.kinematics import KinematicsCfg
        from curobo._src.util_file import load_yaml, join_path, get_task_configs_path
        self.robot_dict, self.robot_metadata = _bounded_robot_configuration(
            robot, self.problem, self.scene, self.budget)
        kinematics = self.robot_dict['robot_cfg']['kinematics']
        path = Path(kinematics.get('external_asset_path') or get_assets_path())/kinematics['urdf_path']
        checker = self.robot_validator.checker
        if (kinematics['base_link'] != checker.base_frame
                or ET.tostring(ET.parse(path).getroot(), encoding='unicode') != checker.urdf_xml):
            raise ValueError('Planning configuration must use the consumed calibrated robot tree')
        # The original lock declaration describes independent measured fingers.
        hand = self.scene.gripper
        independent = hand is not None and hand.controller in {'timed_reference', 'binary'}
        locks = kinematics.get('lock_joints') or {}
        allowed = dict(zip(hand.joint_names, hand.current_positions)) if independent else {}
        if not set(locks) <= set(allowed) or any(
                value != allowed[name] and value != float(np.float32(allowed[name]))
                for name, value in locks.items()):
            raise ValueError('Joint locks must retain measured independent finger coordinates')
        if independent:
            joint = int(checker.model.frames[checker._frame_id(self.robot_validator.tool_frame)].parentJoint)
            while joint:
                if str(checker.model.names[joint]) in hand.joint_names:
                    raise ValueError('An independently controlled finger cannot be an ancestor of the TCP')
                joint = int(checker.model.parents[joint])
        configuration = KinematicsCfg.from_data_dict(dict(kinematics), device_cfg=self.device_cfg)
        self.joint_limits = configuration.get_joint_limits()
        self.joint_names = tuple(self.joint_limits.joint_names)
        self.motion_configuration = load_yaml(join_path(get_task_configs_path(),
            'trajopt/lbfgs_bspline_trajopt.yml'))['rollout']['cost_cfg']['cspace_cfg']

    def _bind_request_computation(self):
        """Bind one original reference and physical domain, then local components."""
        scene, budget, problem = self.scene, self.budget, self.problem
        self.validation_storage = {'encoded_record_bytes_upper': 0}
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget,
            self.anchor_capacity, clock_nodes=len(self.progress_map.u))
        _require_planning_host_memory(workspace.pending_cpu_bytes, encoded_record_bytes_upper=0,
                                     max_host_rss_bytes=budget.max_host_rss_bytes)
        workspace = _require_planning_memory(torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.scene_cpu_live_bytes, self.anchor_fk_workspace_bytes = workspace.cpu_live_bytes, workspace.anchor_fk_bytes
        self._bind_online_robot()
        entrance_pose, entrance_twist = _reference_entrance(problem, scene, self.robot_validator)
        self.bound_reference = BoundToolReference(problem, self.progress_map.u, self.progress_map.s,
            self.formal_clock['duration'], entrance_pose, entrance_twist)
        self.last_validation = None
        self.last_optimizer_diagnostics, self.last_optimizer_runtime = [], []
        self.goal_ik_candidates = {}
        self.delivery_candidate = self.initialization_candidate = None
        self.diagnostic_optimizer_candidate = self.diagnostic_optimizer_provenance = None
        self.reference_initialization = {'status': 'not_started'}
        self.reference_ik_queries, self.initial_seed_strategies = [], []
        columns = [self.joint_limits.joint_names.index(name) for name in self.joint_names]
        bounds = torch.stack([getattr(self.joint_limits, name)[:, columns]
            for name in ('position', 'velocity', 'acceleration', 'jerk')]).cpu().double()
        self.position_domain = JointPositionDomain(bounds[0, 0], bounds[0, 1])
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.physical_domain = CubicJointDomain(bounds,
            torch.as_tensor(problem.start_q[columns]), torch.as_tensor(problem.start_dq[columns]),
            self.formal_clock['knot_dt'], budget.knots+5)
        frames = [problem.tool_link, *(row[0] for row in scene.robot_model._geometries.values())]
        self.pin_kinematics = DifferentiablePinKinematics(self.robot_validator.checker.model,
            frames=frames, T_reference_base=problem.base_pose)
        self.action_horizon = self.physical_domain.knots
        self.main_trajectory_iterations = budget.trajectory_iterations
        self._bind_computation()

    def rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None):
        """Replace request data while retaining the fixed official joint configuration."""
        self._bind_request_clock(problem, scene, budget, continuation, goal_identities)
        self._bind_request_computation()
        return self

    def close(self):
        """Restore the caller's arithmetic policy when this request owner closes."""
        torch.backends.cuda.matmul.allow_tf32 = self._caller_precision[0]
        torch.backends.cudnn.allow_tf32 = self._caller_precision[1]

    def _candidate_domain_validation(self, trajectory):
        """Evaluate original continuous polynomial limits from their direct owner."""
        limits = self.joint_limits
        columns = [limits.joint_names.index(name) for name in trajectory.joint_names]
        bounds = limits.position[:, columns].detach().cpu().numpy()
        position = cubic_position_report(trajectory._curve, *bounds)
        velocity_bounds = limits.velocity[:, columns].detach().cpu().numpy()
        velocity = cubic_position_report(trajectory._curve.derivative(), *velocity_bounds)
        if velocity['maximum_violation'] is not None:
            velocity['maximum_violation']['velocity'] = velocity['maximum_violation'].pop('position')
        derivatives = cubic_derivative_report(trajectory, limits.acceleration.detach().cpu().numpy(),
                                              joint_names=limits.joint_names)
        jerk = cubic_position_report(trajectory._curve.derivative(3),
            *limits.jerk[:, columns].detach().cpu().numpy())
        if jerk['maximum_violation'] is not None:
            jerk['maximum_violation']['jerk'] = jerk['maximum_violation'].pop('position')
        jerk['scope'] = 'piecewise third derivative; both adjacent segments at knots'
        entrance_columns = [self.joint_names.index(name) for name in trajectory.joint_names]
        trajectory.residuals = {
            'continuous_position': position['maximum_bound_excess'],
            'continuous_velocity': velocity['maximum_bound_excess'],
            'continuous_acceleration': derivatives['acceleration']['maximum_bound_excess'],
            'continuous_jerk': jerk['maximum_bound_excess'],
            'initial_position': max(0., float(np.max(np.abs(trajectory.position[0]-
                self.physical_domain._curve_map.start_q[entrance_columns])))-self.problem.numerical.joint),
            'initial_velocity': max(0., float(np.max(np.abs(trajectory.velocity[0]-
                self.physical_domain._curve_map.start_dq[entrance_columns])))-self.problem.numerical.velocity)}
        return {'continuous_position': position, 'continuous_velocity': velocity,
                'continuous_jerk': jerk, 'cubic_derivatives': derivatives,
                'numerical_tolerances': asdict(self.problem.numerical)}
