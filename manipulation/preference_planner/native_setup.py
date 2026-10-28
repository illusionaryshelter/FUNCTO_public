"""Native event IK setup using explicit pose roles and existing evidence.

Inputs are the current bound PlanningProblem, its observed scene, measured
joint entrance, calibrated robot model and original PlanningBudget. Event
poses use the problem reference frame and target-local swing/twist geometry.
The original clock, hand commands, resident native resources and joint/model
checks are retained. The local change supplies pose-prior attraction to event
IK optimization and uses the same preference to order equally qualified
proposals. Ordinary reference IK remains an approximate initializer.

The manager factory, initialization method and static-report method are
explicit local copies of their FUNCTO implementations. This module does not
extract code or replace library functions at runtime. NativeSetup precedes
_ConstrainedPlanner in the preference planner's bases, so the original
constructor dispatches to this initialization method and retains its cleanup.
"""

# BEGIN experiment-local native setup with one explicit pose-prior consumer.
from copy import deepcopy
from dataclasses import dataclass, fields

import numpy as np

from manipulation.planning import (
    _ProgressMap, _initial_timing, _continuation_timing, reference_progress_clock,
    _fixed_spline_grid, _bind_event_clock, _phase_aligned_parameters,
    _anchor_capacity, _bounded_robot_configuration, _reference_entrance,
    _planning_workspace_bytes, _require_planning_host_memory,
    _require_planning_memory, _orientation_vjp_type, _native_gradient_layout_type,
    _scene_boundary_evaluate, _ompl_terms_accepted,
)
from manipulation.planning_ik import _static_qualification, _validate_reports
from manipulation.resource_admission import require_device_memory
from .objective import pose_prior_cost


def rank_pose_ik_candidates(candidates, reports, previous, *, count, joint_tolerance):
    """Order finite distinct proposals by original evidence, then preference.

    The original four-channel static qualification has highest priority.
    Within equal qualification, the common weighted pose-prior cost precedes
    distance to the nearest supplied predecessor; source index breaks ties.
    The existing joint tolerance and seed count define duplicate removal and
    the selected pool size. Native success flags remain report data.
    """
    values = np.asarray(candidates, dtype=float)
    prior = np.asarray(previous, dtype=float)
    if (values.ndim != 2 or values.shape[1] < 1 or prior.ndim != 2
            or prior.shape[1] != values.shape[1] or not len(prior)
            or not np.isfinite(prior).all() or type(count) is not int or count < 1
            or not np.isfinite(joint_tolerance) or joint_tolerance <= 0):
        raise ValueError("IK ranking needs finite previous states and positive bounds")
    _validate_reports(reports, len(values))

    def key(index):
        continuity = float(np.linalg.norm(prior-values[index], axis=-1).min())
        return (*_static_qualification(reports[index]),
                reports[index]['pose_prior_cost'], continuity, int(index))

    selected = []
    for index in sorted(np.flatnonzero(np.isfinite(values).all(axis=-1)), key=key):
        if selected and np.any(np.max(np.abs(values[selected]-values[index]), axis=-1) <= joint_tolerance):
            continue
        selected.append(int(index))
        if len(selected) == count:
            break
    return np.asarray(selected, dtype=int)


def _ik_manager_type(problem, budget, *, clock, seed_parameter):
    """Bind original hard event metrics and explicit soft pose preferences.

    The optimizer receives the existing allowed-set cost plus pose_prior_cost
    at the same local event identities and existing constraint weight. Native
    seed_metrics contains the original hard quantities only. The ordinary
    reference solver uses its unchanged native manager and supplies proposals.
    """
    import torch
    from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
    from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import RobotCostManagerCfg
    orientation_vjp, native_layout = _orientation_vjp_type(), _native_gradient_layout_type()
    ik_goal_ids = ()

    class IKCostManager(RobotCostManager):
        def initialize_from_config(self, config, transition_model, scene_collision_checker=None, **kwargs):
            super().initialize_from_config(deepcopy(config), transition_model, scene_collision_checker, **kwargs)
            self._process_mode = config.process_mode
            self._joint_names = tuple(transition_model.joint_names)
            self._tool_index = transition_model.robot_model.tool_frames.index(problem.tool_link)
            transition_model.robot_model.compute_jacobian = True

        def compute_costs(self, state, cost_collection=None, goal=None, **kwargs):
            collection = super().compute_costs(state, cost_collection, goal, **kwargs)
            return self._add_event_costs(state, collection)

        def _add_event_costs(self, state, collection):
            """Append task components after the native collection is computed.

            Native RobotCostManager owns its cost registry and CUDA streams.
            This method owns only the existing task tensor/VJP computation
            and the added pose preference, making that boundary explicit.
            """
            if not self._process_mode or not ik_goal_ids:
                return collection
            q, tool = state.joint_state.position, state.tool_poses
            orientation = orientation_vjp.apply(q, tool.quaternion.detach(),
                state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach())
            position, quaternion = problem.transform_tool_pose(
                native_layout.apply(tool.position)[:, :, self._tool_index], orientation[:, :, self._tool_index])
            event = getattr(problem, ik_goal_ids[0][0])[ik_goal_ids[0][1]]
            progress = q.new_full(q.shape[:2], event.progress)
            times = seed_parameter.to(q).expand(q.shape[:2])*clock['duration']
            terms = problem.residuals(position, quaternion, progress, q, self._joint_names,
                joint_velocity=torch.zeros_like(q), trajectory_time=times,
                point_goal_ids=ik_goal_ids, direction_mode='point')
            process = sum((value for name, value in terms.items()
                if not name.startswith(('function_pose:', 'function_coordinate:', 'support:', 'stop:'))),
                torch.zeros_like(progress))
            goals = problem.goal_residual(position, quaternion, progress, goal_ids=ik_goal_ids,
                                          normalized=self._process_mode == 'seed_optimizer')
            value = (process.square()+goals.square().sum(-1) if self._process_mode == 'seed_optimizer'
                     else process+goals.sum(-1))
            collection.add(value[..., None]*budget.constraint_weight, 'event_allowed_set')
            # BEGIN soft event preferences affect optimizer attraction only.
            # goal_ids index this bound problem's local event collections.
            # Original identities remain in the formal execution binding.
            if self._process_mode == 'seed_optimizer':
                prior = pose_prior_cost(problem, position, quaternion, progress,
                                       times, goal_ids=ik_goal_ids)
                collection.add(prior[..., None]*budget.constraint_weight, 'pose_prior')
            # END native metrics never consume prior error as a hard residual.
            return collection

    @dataclass
    class IKCostManagerCfg(RobotCostManagerCfg):
        process_mode: str = ''

        def __post_init__(self):
            super().__post_init__()
            self.class_type = IKCostManager

        @staticmethod
        def create(data_dict, scene_collision_checker=None, device_cfg=None):
            base = RobotCostManagerCfg.create(data_dict, scene_collision_checker, device_cfg)
            values = {item.name: getattr(base, item.name) for item in fields(base) if item.name != 'class_type'}
            return IKCostManagerCfg(**values, process_mode=data_dict.get('process_mode', ''))

    def bind_request(new_problem, new_budget):
        nonlocal problem, budget, ik_goal_ids
        problem, budget, ik_goal_ids = new_problem, new_budget, ()
    def bind_ik_goals(goal_ids):
        nonlocal ik_goal_ids
        selected = tuple(tuple(identity) for identity in goal_ids)
        if len({getattr(problem, name)[index].progress for name, index in selected}) > 1:
            raise ValueError('One event IK query requires one declared progress')
        ik_goal_ids = selected
    IKCostManagerCfg.bind_request = staticmethod(bind_request)
    IKCostManagerCfg.bind_ik_goals = staticmethod(bind_ik_goals)
    return IKCostManagerCfg


class NativeSetup:
    """Use local event manager setup with the original planner lifecycle.

    Compose as ``class PreferencePlanner(NativeSetup, _ConstrainedPlanner)``.
    The inherited constructor owns resource cleanup and precision restoration;
    inherited rebind updates this manager through bind_request. All copied
    initialization operations retain their original order and resource bounds.
    """

    def _initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None):
        # Validate grounded evidence and bound independent scene workspaces
        # before importing native GPU solvers or creating any of their buffers.
        self.robot_validator = robot_validator
        # BEGIN native planning and measured execution share calibrated FK identity.
        robot_validator.validate_planning_binding(problem, scene.gripper)
        # END this checks named inputs only and performs no source geometry query.
        # BEGIN continuation keeps original goal identities after local filtering.
        scene.validate_problem(problem, goal_identities=goal_identities)
        # END finite arrival binding never treats an original ID as a local index.
        # END binding precedes native allocation and applies to direct planner construction.
        budget, initial_progress = (_initial_timing(problem, budget) if continuation is None
                                    else _continuation_timing(problem, budget, continuation))
        anchors = np.unique(np.concatenate((problem.anchors, np.asarray(scene.required_progress_anchors, float))))
        anchors = anchors[(anchors >= problem.start_progress) & (anchors <= problem.end_progress)]
        # BEGIN preserve all required source points within the total query budget.
        if (len(anchors) > budget.max_validation_points or not np.isfinite(anchors).all()
                or np.any(anchors < 0) or np.any(anchors > problem.end_progress)):
            raise ValueError("All required source knots and scene anchors must fit the finite planning horizon and budget")
        if problem.reference_frame != scene.reference_frame:
            raise ValueError("Problem and planning scene must use the same explicitly calibrated reference frame")
        self.progress_map = initial_progress
        # BEGIN retain the reference rhythm independently of free arm coefficients.
        self.source_progress_map, self.reference_time_map = initial_progress, None
        if continuation is None:
            self.progress_map, self.reference_time_map = reference_progress_clock(initial_progress,
                budget.knots, stop_progress=problem.stop_progress, anchors=anchors)
        initial_clock = _fixed_spline_grid(intervals=budget.knots+4,
            native_dt=budget.maximum_trajectory_dt, interpolation_steps=budget.interpolation_steps,
            interpolation_dt=budget.interpolation_dt)
        if continuation is None:
            before_tick = self.progress_map
            self.progress_map = _bind_event_clock(problem, self.progress_map, initial_clock["duration"],
                None if scene.gripper is None else scene.gripper.control_timestep_s)
            self.reference_time_map = _ProgressMap(_phase_aligned_parameters(
                before_tick, self.progress_map, self.reference_time_map.u), self.reference_time_map.s)
        # END task progress, hand commands and reference seeds share this fixed clock.
        # BEGIN freeze synchronized hand inputs before any rollout owns a clone.
        # IK, trajectory managers and independent verification each deep-copy
        # the scene. Binding only the parent after their construction leaves
        # those copies without a command/nominal polynomial. The admitted grid
        # already defines the exact native duration; verify it against the
        # constructed solver below without regenerating these commands.
        hand = scene.gripper
        if hand is not None:
            hand.bind_clock(initial_clock["duration"], self.progress_map)
        self.finger_trajectory = None if hand is None else hand.finger_trajectory
        # END every private scene receives the same already frozen hand clock.
        self.anchor_capacity = _anchor_capacity(self.progress_map, anchors, budget)
        robot_dict, self.robot_metadata = _bounded_robot_configuration(robot, problem, scene, budget)
        # BEGIN independent and differentiable FK share the same pure kinematic tree.
        # Compare the parsed XML with the validator's consumed snapshot, not
        # with its mutable source file. Runtime native tree overrides have no
        # counterpart in the bound source model and cannot change this tree.
        from curobo.content import get_assets_path
        from pathlib import Path
        import xml.etree.ElementTree as ET
        kinematics = robot_dict["robot_cfg"]["kinematics"]
        native_urdf = Path(kinematics.get("external_asset_path") or get_assets_path()) / kinematics["urdf_path"]
        checker = robot_validator.checker
        if (kinematics["base_link"] != robot_validator.checker.base_frame
                or kinematics.get("extra_links") or kinematics.get("add_object_link", False)
                or ET.tostring(ET.parse(native_urdf).getroot(), encoding="unicode")
                != checker.urdf_xml):
            raise ValueError("Native planning requires the consumed source URDF/base without tree overrides")
        locks = kinematics.get("lock_joints") or {}
        independent_hand = hand is not None and hand.controller in {"timed_reference", "binary"}
        allowed_locks = dict(zip(hand.joint_names, hand.current_positions)) if independent_hand else {}
        if (not set(locks) <= set(allowed_locks) or any(
                value != allowed_locks[name] and value != float(np.float32(allowed_locks[name]))
                for name, value in locks.items())):
            raise ValueError("Native joint locks may only retain the measured independent finger coordinates")
        if independent_hand:
            joint = int(checker.model.frames[checker._frame_id(robot_validator.tool_frame)].parentJoint)
            while joint:
                if str(checker.model.names[joint]) in hand.joint_names:
                    raise ValueError("An independently controlled finger cannot be an ancestor of the native TCP")
                joint = int(checker.model.parents[joint])
        # END source identity is established before allocating GPU solvers.
        # BEGIN bind the tool reference after the actual event/hold clock.
        from manipulation.timed_reference import BoundToolReference
        entrance_pose, entrance_twist = _reference_entrance(problem, scene, robot_validator)
        self.bound_reference = BoundToolReference(problem, self.progress_map.u, self.progress_map.s,
            initial_clock["duration"], entrance_pose, entrance_twist)
        # END this bound physical-time reference supplies ordinary IK proposals.
        self.validation_storage = {"encoded_record_bytes_upper": 0}
        workspace = _planning_workspace_bytes(scene, self.robot_metadata, budget, self.anchor_capacity,
            clock_nodes=len(self.progress_map.u))
        _require_planning_host_memory(workspace.pending_cpu_bytes,
            encoded_record_bytes_upper=self.validation_storage["encoded_record_bytes_upper"],
            max_host_rss_bytes=budget.max_host_rss_bytes)
        # END preserve all required source points within the total query budget.
        # BEGIN native IK proposes states; SLSQP owns the complete trajectory.
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.kinematics import Kinematics
        from curobo.types import JointState
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.types.tool_pose import GoalToolPose
        from curobo._src.util_file import load_yaml, join_path, get_task_configs_path
        self.problem, self.scene, self.budget = problem, scene, budget
        self.torch, self.JointState, self.GoalToolPose = torch, JointState, GoalToolPose
        self.device_cfg = DeviceCfg(device)
        workspace = _require_planning_memory(torch, self.device_cfg.device, workspace,
                                             max_device_used_bytes=budget.max_device_used_bytes)
        self.scene_workspace_bytes = workspace.total_bytes
        self.scene_cpu_workspace_bytes, self.scene_gpu_workspace_bytes = workspace.cpu_bytes, workspace.gpu_bytes
        self.scene_cpu_live_bytes, self.anchor_fk_workspace_bytes = workspace.cpu_live_bytes, workspace.anchor_fk_bytes
        self.anchors, self.robot_dict = anchors, robot_dict
        self.seed_parameter = self.device_cfg.to_device([[0.]])
        self.formal_clock = dict(initial_clock)
        self._manager_type = _ik_manager_type(problem, budget, clock=self.formal_clock,
                                               seed_parameter=self.seed_parameter)
        self.goal_ik_candidates = {}
        self.last_optimizer_diagnostics, self.last_optimizer_runtime = [], []
        self.delivery_candidate = self.initialization_candidate = None
        # BEGIN solver diagnostics own their curve separately from validated proposals.
        self.diagnostic_optimizer_candidate = self.diagnostic_optimizer_provenance = None
        # END an interrupted solver cannot replace a candidate's completed verdict.
        self.reference_initialization = {'status': 'not_started'}
        self.reference_ik_queries, self.initial_seed_strategies = [], []
        self.route = 'unified_constrained'

        def configuration(name, mode=None):
            data = deepcopy(load_yaml(join_path(get_task_configs_path(), name)))
            constraint = data['rollout'].setdefault('constraint_cfg', {})
            constraint.pop('scene_collision_cfg', None)
            constraint.pop('self_collision_cfg', None)
            if mode is not None:
                constraint['process_mode'] = mode
            return data

        ik_optimizer = configuration('ik/lbfgs_ik.yml', 'seed_optimizer')
        # BEGIN preserve the request's exact positive IK iteration count.
        # Native LBFGS requires num_iters to be divisible by inner_iters.
        # One update per group admits every PlanningBudget count. The native
        # outer loop retains optimizer state; fixed_iters executes all groups.
        ik_optimizer['optimizer']['inner_iters'] = 1
        # END grouping changes neither the budget nor the native solve loop.
        pose_cost = ik_optimizer['rollout']['cost_cfg']['tool_pose_cfg']
        pose_cost.update(use_lie_group=True, _terminal_pose_axes_weight_factor=[1.]*6,
                         _non_terminal_pose_axes_weight_factor=[1.]*6)
        pose_cost['weight'][1] *= 4.
        self.motion_configuration = deepcopy(load_yaml(join_path(get_task_configs_path(),
            'trajopt/lbfgs_bspline_trajopt.yml'))['rollout']['cost_cfg']['cspace_cfg'])
        # Native config decoding replaces dictionaries with typed config objects.
        # Separate ordinary inputs before either decoder can mutate its input.
        reference_optimizer = deepcopy(ik_optimizer)
        reference_optimizer['rollout']['constraint_cfg'].pop('process_mode')
        ik_cfg = InverseKinematicsCfg.create(robot=deepcopy(robot_dict), optimizer_configs=[ik_optimizer],
            metrics_rollout=configuration('metrics_base.yml', 'seed_metrics'), device_cfg=self.device_cfg,
            self_collision_check=False, num_seeds=budget.ik_seeds, seed_solver_num_seeds=budget.ik_seeds,
            max_batch_size=1, max_goalset=1, use_cuda_graph=False,
            position_tolerance=problem.numerical.position_m, orientation_tolerance=problem.numerical.rotation_rad,
            cost_manager_config_instance_type=self._manager_type,
            override_optimizer_num_iters={'lbfgs': budget.ik_iterations}, random_seed=budget.random_seed)
        # BEGIN fixed native instances separate approximate references from event allowed sets.
        ik_cfg.use_lm_seed = False
        ik_cfg.exit_early = False
        self.ik = InverseKinematics(ik_cfg)
        reference_cfg = InverseKinematicsCfg.create(robot=deepcopy(robot_dict),
            optimizer_configs=[reference_optimizer],
            metrics_rollout=configuration('metrics_base.yml'), device_cfg=self.device_cfg,
            self_collision_check=False, num_seeds=1, seed_solver_num_seeds=1,
            max_batch_size=1, max_goalset=1, use_cuda_graph=False,
            position_tolerance=problem.numerical.position_m, orientation_tolerance=problem.numerical.rotation_rad,
            optimization_dt=None, velocity_regularization_weight=.01, acceleration_regularization_weight=.01,
            override_optimizer_num_iters={'lbfgs': budget.ik_iterations}, random_seed=budget.random_seed)
        reference_cfg.use_lm_seed = reference_cfg.exit_early = False
        self.reference_ik = InverseKinematics(reference_cfg)
        # Native allocation is now resident; enforce the original physical caps
        # against measured usage, without charging the admission allowance twice.
        require_device_memory(torch, self.device_cfg.device, native_bytes=0,
                              max_device_used_bytes=budget.max_device_used_bytes)
        _require_planning_host_memory(workspace.pending_cpu_bytes, encoded_record_bytes_upper=0,
                                     max_host_rss_bytes=budget.max_host_rss_bytes)
        # END the ordinary solver uses the library cost manager and has one seed.
        # cuRobo sets precision flags during construction. Own and restore
        # that request-local change without altering the caller's cuDNN policy.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = self._caller_precision[1]
        self.joint_names = tuple(self.ik.joint_names)
        if (set(self.joint_names) != set(self.robot_metadata['active_joint_names'])
                or self.ik.kinematics.total_spheres != self.robot_metadata['spheres']):
            raise RuntimeError('Native robot layout differs from its checked metadata')
        columns = [problem.joint_names.index(name) for name in self.joint_names]
        self.start = JointState.from_numpy(list(self.joint_names), problem.start_q[columns][None],
            velocity=problem.start_dq[columns][None], device_cfg=self.device_cfg)
        self.start.dt = self.device_cfg.to_device([0.])
        self._bind_online_robot()
        self._verification_fk = Kinematics(self.ik.kinematics.config, compute_jacobian=True)
        self._orientation_vjp, self._native_layout = _orientation_vjp_type(), _native_gradient_layout_type()
        self._verification_scene = scene.clone_for_rollout()
        self._initialize_verification_constraints()
        self._bind_physical_domains()
        # END numerical trajectory search shares the independently loaded robot.

    def _rank_ik_candidates(self, candidates, success, previous, u, goal_ids):
        """Rank original static evidence, then prior cost from that same FK.

        A finite proposal receives one original FK, task, scene and native
        check. Its same tool pose supplies the common weighted prior cost;
        that preference affects ordering, never task/model qualification.
        """
        values, flags = np.asarray(candidates), np.asarray(success, bool)
        phase = float(np.interp(u, self.progress_map.u, self.progress_map.s))
        reports = []
        # BEGIN static IK qualification uses the same discrete collision model.
        # END this local scope retains no evidence across a different query/scene.
        for index, row in enumerate(values):
            report = {"allowed_set_ik_success": bool(flags[index]), "candidate_source": "event_allowed_set_ik",
                      "task_satisfied": False, "model_valid": False}
            reports.append(report)
            if not np.isfinite(row).all():
                # BEGIN an invalid proposal reports its actual missing input.
                report["invalid_joint_indices"] = np.flatnonzero(~np.isfinite(row)).tolist()
                # END non-finite joints never enter FK or acquire scene evidence.
                continue
            # BEGIN finite event proposals use their original allowed-set evidence.
            report["static_model_query"] = "evaluated"
            # END nominal pose-centre convergence has no functional veto.
            with self.torch.no_grad():
                q = self.device_cfg.to_device(row[None, None]).contiguous()
                zero = self.torch.zeros_like(q)
                js = self.JointState(position=q, velocity=zero, acceleration=zero, jerk=zero,
                    dt=self.device_cfg.to_device([self.budget.interpolation_dt]), joint_names=list(self.joint_names))
                state = self._verification_fk.compute_kinematics(js)
                index = state.tool_poses.tool_frames.index(self.problem.tool_link)
                p, r = self.problem.transform_tool_pose(state.tool_poses.position[:, :, index],
                                                       state.tool_poses.quaternion[:, :, index])
                progress, times = q.new_full(q.shape[:2], phase), q.new_full(q.shape[:2], u*self.formal_clock["duration"])
                # BEGIN reuse this FK for the same prior term as event IK.
                prior = pose_prior_cost(self.problem, p, r, progress, times,
                                       goal_ids=goal_ids)
                report['pose_prior_cost'] = float((self.budget.constraint_weight * prior.sum()).item())
                # END every qualification field below remains independently computed.
                task = self.problem.residuals(p, r, progress, q, self.joint_names,
                    # BEGIN seed qualification uses the same task set as curve delivery.
                    joint_velocity=zero, trajectory_time=times, point_goal_ids=goal_ids,
                    direction_mode="point")
                    # END nominal support preferences cannot become static rejection rules.
                task = {name: value for name, value in task.items() if not name.startswith("stop:")}
                # BEGIN reuse the existing static query for bounded rejection reasons.
                scene_components = {}
                # END prior ordering adds no second geometry evaluation.
                online = _scene_boundary_evaluate(self.scene, self.problem, self.progress_map,
                    self.problem.transform_robot_spheres(state.robot_spheres), p, r, progress, q,
                    self.joint_names, trajectory_time=times, normalized_time=q.new_full(q.shape[:2], u),
                    component_diagnostics=scene_components)
                native = self._RobotState(joint_state=js, cuda_robot_model_state=state)
                constraints = self._verification_native.compute_costs(native)
                model = {"scene": online, **dict(zip(constraints.names, constraints.values))}
                report.update(task_satisfied=_ompl_terms_accepted(task), model_valid=_ompl_terms_accepted(model))
                # BEGIN preserve positive/non-finite residuals in their original families.
                def failures(terms):
                    maxima = {}
                    for name, value in terms.items():
                        flat = value.detach().reshape(-1)
                        index = self.torch.where(self.torch.isfinite(flat), flat,
                            self.torch.full_like(flat, float("inf"))).argmax()
                        maxima[name] = float(flat[index].item())
                    return {name: value for name, value in maxima.items() if value > 0 or not np.isfinite(value)}
                report.update(task_violations=failures(task),
                    native_violations=failures(dict(zip(constraints.names, constraints.values))),
                    scene_components=scene_components)
                # END conditional scene entries retain their names and have no veto.
        ranked = rank_pose_ik_candidates(values, reports, np.asarray(previous), count=self.budget.ik_seeds,
            joint_tolerance=self.problem.numerical.joint)
        return ranked, reports

# END only event optimizer preference and static proposal ordering change.
