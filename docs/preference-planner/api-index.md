# 实际 API 定义及调用索引

代码基线：`8b0e326`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

签名和直接调用来自当前源文件 AST；数据前置条件、坐标与结果含义由对应组件正文定义。内部辅助函数保留具体名称以定位计算。

## backend.py

### `_hold_polynomial_report(trajectory, start_s)`

Compute complete suffix position/rate extrema of the saved polynomial.

[定义](../../manipulation/preference_planner/backend.py#L38)。直接调用：`PPoly`, `derivative.roots`, `dict`, `enumerate`, `float`, `int`, `np.abs`, `np.argmax`, `np.isfinite`, `np.unique`, `position`, `position.derivative`, `velocity`, `velocity.derivative`。

### `PreferencePlanner.__init__(self, *args, incoming_terminal_contract=None, **kwargs)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/backend.py#L79)。直接调用：`kwargs.get`, `super`, `super().__init__`。

### `PreferencePlanner.rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None)`

Reset request-local computations through the original rebinding API.

[定义](../../manipulation/preference_planner/backend.py#L84)。直接调用：`super`, `super().rebind`。

### `PreferencePlanner._bind_computation(self, extra_parameters=())`

Bind one original full calculation and its explicit reduced problem.

[定义](../../manipulation/preference_planner/backend.py#L90)。直接调用：`GeometryConstraints`, `JointComputation`, `TerminalHold.from_clock`, `dict`, `enumerate`, `pose_objective`, `super`, `super()._bind_computation`, `terminal_clock`。

### `PreferencePlanner._controls_curve(self, actions)`

Keep the full coefficients associated with this exact exported curve.

[定义](../../manipulation/preference_planner/backend.py#L113)。直接调用：`actions.detach`, `actions.detach().cpu`, `actions.detach().cpu().numpy`, `actions.detach().cpu().numpy().copy`, `isinstance`, `np.asarray`, `np.asarray(actions, float).copy`, `super`, `super()._controls_curve`。

### `PreferencePlanner.reference_seeds(self, joint_reference=None)`

Project the original numerical initializer into the declared subspace.

[定义](../../manipulation/preference_planner/backend.py#L121)。直接调用：`float`, `np.abs`, `np.max`, `original[0].tolist`, `projected[0].tolist`, `seeds.detach`, `seeds.detach().cpu`, `seeds.detach().cpu().numpy`, `self._controls_curve`, `self.terminal.expand`, `self.terminal.reduce`, `self.torch.as_tensor`, `super`, `super().reference_seeds`。

### `PreferencePlanner._candidate_metrics(self, trajectory, candidate_scene, *, record_work)`

Consume the same completed entity evidence for necessary-row updates.

[定义](../../manipulation/preference_planner/backend.py#L137)。直接调用：`self.geometry_constraints.add_evidence`, `super`, `super()._candidate_metrics`。

### `PreferencePlanner._candidate_domain_validation(self, trajectory)`

Qualify terminal motion on the same saved curve, including initializers.

[定义](../../manipulation/preference_planner/backend.py#L147)。直接调用：`_ProgressMap`, `_hold_polynomial_report`, `_phase_aligned_parameters`, `float`, `max`, `np.array_equal`, `np.asarray`, `super`, `super()._candidate_domain_validation`, `trajectory.residuals.update`。

### `PreferencePlanner._candidate_objective(self, trajectory, candidate_scene)`

Score the saved curve with the solver's fixed physical computation.

[定义](../../manipulation/preference_planner/backend.py#L170)。直接调用：`_ProgressMap`, `_phase_aligned_parameters`, `components.values`, `dict`, `float`, `np.array_equal`, `self._candidate_robot_state`, `self.computation.progress.numpy`, `self.computation.times.numpy`, `sum`, `tensors.items`, `torch.as_tensor`, `torch.no_grad`, `trajectory_objective_components`。

### `PreferencePlanner._candidate_geometry(self, trajectory, candidate_scene, *, initial_states, remaining_points, record_work)`

Reuse bounded interval queries and collect their necessary witnesses.

[定义](../../manipulation/preference_planner/backend.py#L212)。直接调用：`candidate_scene.clone_for_rollout`, `self.geometry_constraints.add_evidence`, `super`, `super()._candidate_geometry`。

### `PreferencePlanner._candidate_geometry.collect(*args, geometry_evidence_out, **kwargs)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/backend.py#L233)。直接调用：`batches.append`, `evaluate`。

### `PreferencePlanner.evaluate_candidate(self, trajectory, *, record_work, candidate_scene=None, objective=None)`

Keep original task/domain qualification and capture its row discoveries.

[定义](../../manipulation/preference_planner/backend.py#L249)。直接调用：`completed_evidence`, `deepcopy`, `dict`, `self._current_full_controls.tolist`, `self.candidate_history.append`, `super`, `super().evaluate_candidate`。

### `PreferencePlanner.optimize(self, seed, *, iterations=None)`

Solve reduced variables and exchange entity witnesses within the ledger.

[定义](../../manipulation/preference_planner/backend.py#L284)。直接调用：`ConstrainedSolution`, `_planning_workspace_bytes`, `deepcopy`, `dict`, `enumerate`, `geometry_workspace.values`, `len`, `next`, `np.asarray`, `np.asarray(full).copy`, `np.isfinite`, `np.isfinite(snapshot['controls']).all`, `row.detach`, `row.detach().cpu`, `row.detach().cpu().numpy`, `runtime.get`, `runtime.update`, `self._controls_curve`, `self._expand_snapshots`, `self.check_request`, `self.geometry_constraints.workspace_bytes`, `self.last_optimizer_diagnostics.append`, `self.last_optimizer_runtime.append`, `self.request_work_baseline.get`, `self.seed_strategies.append`, `self.solve_history.append`, `self.terminal.expand`, `self.terminal.reduce`, `solve_trajectory`, `str`, `sum`, `tuple`, `type`。

### `PreferencePlanner._expand_snapshots(self, runtime)`

Convert diagnostic coordinates once before original curve export.

[定义](../../manipulation/preference_planner/backend.py#L371)。直接调用：`np.asarray`, `runtime.get`, `self.terminal.expand`, `self.terminal.expand(reduced).tolist`。

### `PreferencePlanner.diagnostic_snapshot(self)`

Return owned CPU evidence for this request, after its planning verdict.

[定义](../../manipulation/preference_planner/backend.py#L382)。直接调用：`deepcopy`, `dict`, `self.geometry_constraints.describe`。

### `PreferenceMotionBackend.__init__(self, robot, *, robot_validator, budget=None, incoming_terminal_contract=None, device='cuda:0')`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/backend.py#L401)。直接调用：`super`, `super().__init__`。

### `PreferenceMotionBackend.bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None)`

Retain original ownership/capacity rules and choose the local class.

[定义](../../manipulation/preference_planner/backend.py#L406)。直接调用：`MotionPlanningError`, `PreferencePlanner`, `get_ident`, `getattr`, `self._planner.rebind`。

### `PreferenceMotionBackend.plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None, cancelled=None, deadline=None, joint_reference=None)`

Call this planner's lifecycle while retaining resident native models.

[定义](../../manipulation/preference_planner/backend.py#L426)。直接调用：`plan_trajectory`。

## lifecycle.py

### `plan_trajectory(problem, scene, robot, *, robot_validator, budget=None, device='cuda:0', continuation=None, request_ledger=None, rebind=None, backend=None, cancelled=None, deadline=None, joint_reference=None)`

Return the first fully checked candidate in the existing bounded search.

[定义](../../manipulation/preference_planner/lifecycle.py#L27)。直接调用：`ContinuationBindingError`, `MotionPlanningError`, `PlanningBudget`, `PlanningResult`, `PreferenceMotionBackend`, `ValueError`, `_bind_motion_work_ledger`, `_check_work_budget`, `_fixed_spline_grid`, `_obligation_contract`, `_prepare_continuation`, `_reference_clock_issues`, `_request_work_ledger`, `_restore_original_direction_origins`, `_unknown_policy_failure`, `_validate_bound_event_times`, `_verified_dependency_outcomes`, `any`, `asdict`, `backend._claim`, `backend._release`, `backend.bind_request`, `backend.close`, `bind_candidate`, `binding.update`, `bool`, `bound_scene.describe`, `cancelled`, `candidate.times.tolist`, `candidate_report[name].values`, `check`, `consumed.items`, `continuation.get`, `continuation['original_binding'].get`, `curve_snapshot`, `deepcopy`, `diagnostics.get`, `diagnostics.pop`, `diagnostics.update`, `diagnostics['attempts'].append`, `enumerate`, `event_occurrence_key`, `float`, `geometry_failed`, `getattr`, `getattr(limits, name).detach`, `getattr(limits, name).detach().cpu`, `getattr(limits, name).detach().cpu().numpy`, `getattr(limits, name).detach().cpu().numpy().tolist`, `hasattr`, `isinstance`, `ledger['binding_records'].append`, `ledger['timing_events'].append`, `len`, `list`, `mark_wallclock`, `max`, `min`, `motion_baseline.get`, `name.startswith`, `np.isfinite`, `outcome.execution_binding.get`, `plain`, `planner._verification_native.get_cost`, `planner.bind_reference_candidate`, `planner.diagnostic_snapshot`, `planner.evaluate_candidate`, `planner.evaluate_reference`, `planner.export`, `planner.initial_state_diagnostics`, `planner.ompl_seed`, `planner.optimize`, `planner.reference_seeds`, `planner.select_candidate`, `planner.torch.cat`, `problem.bound_motion_block`, `qualify_candidate`, `qualify_native`, `range`, `rebind.get`, `recovered.to`, `reference_attempt.get`, `reference_attempt.get('scene_components', {}).items`, `replace`, `report.get`, `scene.clone_for_rollout`, `scene.describe`, `search_initializers`, `select_candidate`, `set`, `sorted`, `str`, `time.monotonic`, `time.time_ns`, `traceback.format_exc`, `tuple`, `type`。

## native_setup.py

### `rank_pose_ik_candidates(candidates, reports, previous, *, count, joint_tolerance)`

Order finite distinct proposals by original evidence, then preference.

[定义](../../manipulation/preference_planner/native_setup.py#L37)。直接调用：`ValueError`, `_static_qualification`, `_validate_reports`, `float`, `int`, `len`, `np.abs`, `np.any`, `np.asarray`, `np.flatnonzero`, `np.isfinite`, `np.isfinite(prior).all`, `np.isfinite(values).all`, `np.linalg.norm`, `np.linalg.norm(prior - values[index], axis=-1).min`, `np.max`, `selected.append`, `sorted`, `type`。

### `_ik_manager_type(problem, budget, *, clock, seed_parameter)`

Bind original hard event metrics and explicit soft pose preferences.

[定义](../../manipulation/preference_planner/native_setup.py#L70)。直接调用：`IKCostManagerCfg`, `RobotCostManagerCfg.create`, `ValueError`, `_native_gradient_layout_type`, `_orientation_vjp_type`, `collection.add`, `data_dict.get`, `deepcopy`, `fields`, `getattr`, `goals.square`, `goals.square().sum`, `goals.sum`, `len`, `name.startswith`, `native_layout.apply`, `orientation_vjp.apply`, `pose_prior_cost`, `problem.goal_residual`, `problem.residuals`, `problem.transform_tool_pose`, `process.square`, `q.new_full`, `seed_parameter.to`, `seed_parameter.to(q).expand`, `self._add_event_costs`, `state.cuda_robot_model_state.tool_jacobians[..., 3:, :].detach`, `staticmethod`, `sum`, `super`, `super().__post_init__`, `super().compute_costs`, `super().initialize_from_config`, `terms.items`, `tool.quaternion.detach`, `torch.zeros_like`, `transition_model.robot_model.tool_frames.index`, `tuple`。

### `NativeSetup._initialize(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/native_setup.py#L171)。直接调用：`BoundToolReference`, `DeviceCfg`, `ET.parse`, `ET.parse(native_urdf).getroot`, `ET.tostring`, `InverseKinematics`, `InverseKinematicsCfg.create`, `JointState.from_numpy`, `Kinematics`, `Path`, `RuntimeError`, `ValueError`, `_ProgressMap`, `_anchor_capacity`, `_bind_event_clock`, `_bounded_robot_configuration`, `_continuation_timing`, `_fixed_spline_grid`, `_ik_manager_type`, `_initial_timing`, `_native_gradient_layout_type`, `_orientation_vjp_type`, `_phase_aligned_parameters`, `_planning_workspace_bytes`, `_reference_entrance`, `_require_planning_host_memory`, `_require_planning_memory`, `any`, `checker._frame_id`, `configuration`, `constraint.pop`, `data['rollout'].setdefault`, `deepcopy`, `dict`, `float`, `get_assets_path`, `get_task_configs_path`, `hand.bind_clock`, `int`, `join_path`, `kinematics.get`, `len`, `list`, `load_yaml`, `locks.items`, `np.any`, `np.asarray`, `np.concatenate`, `np.float32`, `np.isfinite`, `np.isfinite(anchors).all`, `np.unique`, `pose_cost.update`, `problem.joint_names.index`, `reference_optimizer['rollout']['constraint_cfg'].pop`, `reference_progress_clock`, `require_device_memory`, `robot_validator.validate_planning_binding`, `scene.clone_for_rollout`, `scene.validate_problem`, `self._bind_online_robot`, `self._bind_physical_domains`, `self._initialize_verification_constraints`, `self.device_cfg.to_device`, `set`, `str`, `tuple`, `zip`。

### `NativeSetup._rank_ik_candidates(self, candidates, success, previous, u, goal_ids)`

Rank original static evidence, then prior cost from that same FK.

[定义](../../manipulation/preference_planner/native_setup.py#L368)。直接调用：`(self.budget.constraint_weight * prior.sum()).item`, `_ompl_terms_accepted`, `_scene_boundary_evaluate`, `bool`, `dict`, `enumerate`, `failures`, `flat[index].item`, `float`, `list`, `maxima.items`, `name.startswith`, `np.asarray`, `np.flatnonzero`, `np.flatnonzero(~np.isfinite(row)).tolist`, `np.interp`, `np.isfinite`, `np.isfinite(row).all`, `pose_prior_cost`, `prior.sum`, `q.new_full`, `rank_pose_ik_candidates`, `report.update`, `reports.append`, `self.JointState`, `self._RobotState`, `self._verification_fk.compute_kinematics`, `self._verification_native.compute_costs`, `self.device_cfg.to_device`, `self.device_cfg.to_device(row[None, None]).contiguous`, `self.problem.residuals`, `self.problem.transform_robot_spheres`, `self.problem.transform_tool_pose`, `self.torch.full_like`, `self.torch.isfinite`, `self.torch.no_grad`, `self.torch.where`, `self.torch.where(self.torch.isfinite(flat), flat, self.torch.full_like(flat, float('inf'))).argmax`, `self.torch.zeros_like`, `state.tool_poses.tool_frames.index`, `task.items`, `terms.items`, `value.detach`, `value.detach().reshape`, `zip`。

## objective.py

### `pose_prior_cost(problem, position, quaternion, progress, times, *, goal_ids=None)`

Return dimensionless declared prior cost on the supplied event layout.

[定义](../../manipulation/preference_planner/objective.py#L19)。直接调用：`(errors[name] / scale).square`, `_first_event_query`, `enumerate`, `pose_objective`, `pose_objective_components`, `pose_objective_scales`, `scales.items`, `sum`, `torch.where`, `torch.zeros_like`。

### `trajectory_objective_components(*, problem, scene, states, robot_state, progress, times, ordinary_indices, bounds, motion_configuration, native_dt, constraint_weight, include_geometry=True, geometry_report=None)`

Evaluate the four original-coordinate preference terms once.

[定义](../../manipulation/preference_planner/objective.py#L48)。直接调用：`motion_cost`, `pose_prior_cost`, `pose_prior_cost(problem, *tool, progress, times).sum`, `preferences.sum`, `problem.optimizer_preferences`, `scene.optimization_objective`, `states[0].new_tensor`, `time_mean_weights`, `times.detach`, `times.detach().cpu`, `times.detach().cpu().numpy`, `tuple`。

## pose_roles.py

### `pose_objective(event)`

Read the sole role/scale declaration from a typed event's source.

[定义](../../manipulation/preference_planner/pose_roles.py#L19)。直接调用：。

### `pose_objective_scales(event)`

Return explicit preference scales in metres or radians per component.

[定义](../../manipulation/preference_planner/pose_roles.py#L29)。直接调用：`pose_objective`。

### `pose_objective_components(event, position, quaternion)`

Return the original physical errors selected by the declared scales.

[定义](../../manipulation/preference_planner/pose_roles.py#L34)。直接调用：`event.error_components`, `pose_objective_scales`。

### `ObjectivePoseEvent.component_tolerances(self)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/pose_roles.py#L56)。直接调用：`pose_objective`, `super`。

### `ObjectivePoseEvent.evaluate(self, position, quaternion)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/pose_roles.py#L61)。直接调用：`pose_objective`, `super`, `super().evaluate`。

### `_objective_declaration(event, row)`

Validate the declared data once at the complete-request boundary.

[定义](../../manipulation/preference_planner/pose_roles.py#L69)。直接调用：`PoseEventRequirement.component_tolerances.fget`, `ValueError`, `float`, `isinstance`, `np.isfinite`, `row['scale_source'].strip`, `set`。

### `bind_pose_objectives(problem, declarations)`

Copy a complete problem and bind all original pose-event identities.

[定义](../../manipulation/preference_planner/pose_roles.py#L95)。直接调用：`ObjectivePoseEvent`, `ValueError`, `_objective_declaration`, `deepcopy`, `dict`, `enumerate`, `events.append`, `fields`, `getattr`, `isinstance`, `len`, `range`, `row.get`, `set`, `tuple`, `type`。

### `prior_goal_ids(binding)`

Derive result-only pose identities from the same formal bound roles.

[定义](../../manipulation/preference_planner/pose_roles.py#L138)。直接调用：`ValueError`, `len`, `zip`。

## terminal.py

### `TerminalHold.from_clock(cls, curve_map, clock_u, clock_z, *, source)`

Bind the z=1 suffix of a source-time schedule.

[定义](../../manipulation/preference_planner/terminal.py#L31)。直接调用：`ValueError`, `cls`, `float`, `int`, `len`, `np.any`, `np.arange`, `np.asarray`, `np.diff`, `np.flatnonzero`, `np.linalg.lstsq`, `np.searchsorted`, `null_space`, `slice`, `str`。

### `TerminalHold.expand(self, reduced)`

Return original coefficients, retaining Torch autograd if supplied.

[定义](../../manipulation/preference_planner/terminal.py#L96)。直接调用：`isinstance`, `np.asarray`, `np.einsum`, `reduced.new_tensor`。

### `TerminalHold.reduce(self, full)`

Project an initial coefficient estimate onto this affine subspace.

[定义](../../manipulation/preference_planner/terminal.py#L108)。直接调用：`np.asarray`, `np.einsum`。

### `TerminalHold.pullback(self, jacobian)`

Map original C-order coefficient Jacobian columns to reduced ones.

[定义](../../manipulation/preference_planner/terminal.py#L119)。直接调用：`jacobian.reshape`, `len`, `np.asarray`, `np.einsum`, `np.einsum('rnj,nk->rkj', values, self.matrix).reshape`。

## termination_clock.py

### `terminal_clock(*, problem, budget, progress_map, reference_time_map, duration, continuation, incoming_terminal_contract)`

Return normalized time, terminal parameter and their explicit source.

[定义](../../manipulation/preference_planner/termination_clock.py#L16)。直接调用：`_initial_timing`, `_phase_aligned_parameters`, `float`, `np.array`, `np.unique`, `reference_progress_clock`。

## geometry.py

### `_world(point, pose)`

Place one fixed material coordinate in the calibrated world frame.

[定义](../../manipulation/preference_planner/geometry.py#L33)。直接调用：`pose[0].new_tensor`, `quaternion_rotate`。

### `_local(world, pose)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/geometry.py#L38)。直接调用：`quaternion_conjugate`, `quaternion_rotate`。

### `native_point_value(geometry, local)`

Original signed point distance and current closest-feature direction.

[定义](../../manipulation/preference_planner/geometry.py#L42)。直接调用：`((local - local.detach()) * gradient).sum`, `_transform`, `coal.DistanceRequest`, `coal.DistanceResult`, `coal.Sphere`, `coal.Transform3s`, `coal.distance`, `local.detach`, `local.detach().cpu`, `local.detach().cpu().numpy`, `local.new_tensor`, `np.asarray`, `np.eye`。

### `field_point_value(field, local, *, excluded_cells=())`

Original supported trilinear scalar, or an explicitly inactive row.

[定义](../../manipulation/preference_planner/geometry.py#L63)。直接调用：`((local - local.detach()) * derivative).sum`, `bool`, `field.sample`, `local.detach`, `local.detach().cpu`, `local.detach().cpu().numpy`, `local.new_tensor`, `local.sum`, `np.any`。

### `occupied_material(field, local, *, excluded_cells=())`

Check that a point belongs to the closure of represented volume.

[定义](../../manipulation/preference_planner/geometry.py#L79)。直接调用：`CORNERS.astype`, `bool`, `field.sample`, `int`, `len`, `np.all`, `np.any`, `np.asarray`, `np.flatnonzero`, `np.floor`, `np.floor(grid).astype`, `set`, `tuple`。

### `GeometryConstraints.__init__(self, computation)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/geometry.py#L142)。直接调用：`self.scene.robot_model.geometry_depending_on`, `set`。

### `GeometryConstraints.row_count(self)`

Fixed solver dimension until the next explicit evidence update.

[定义](../../manipulation/preference_planner/geometry.py#L154)。直接调用：`len`。

### `GeometryConstraints.workspace_bytes(self, *, record_count=None)`

Bound this component's added arrays at its actual query dimensions.

[定义](../../manipulation/preference_planner/geometry.py#L158)。直接调用：`int`, `len`, `max`, `sum`。

### `GeometryConstraints._context(self, controls, records)`

Reuse full FK and original relation producers at exact record times.

[定义](../../manipulation/preference_planner/geometry.py#L202)。直接调用：`Rotation.from_matrix`, `Rotation.from_matrix(local[:3, :3]).as_quat`, `_matrices`, `base.domain._curve_map.query`, `base.progress.detach`, `base.progress.detach().numpy`, `base.times.detach`, `base.times.detach().numpy`, `controls.new_tensor`, `dict`, `len`, `poses.items`, `position.new_tensor`, `quaternion.new_tensor`, `quaternion_multiply`, `quaternion_rotate`, `require_host_memory`, `robot_tool_states`, `self.scene._owner_poses`, `self.scene._permission`, `self.scene.gripper.state`, `self.scene.robot_model._geometries.items`, `self.workspace_bytes`, `self.workspace_bytes(record_count=len(records)).values`, `sum`, `torch.zeros_like`。

### `GeometryConstraints._permission(self, context, record, column)`

Recreate the existing pair domain from current poses and declarations.

[定义](../../manipulation/preference_planner/geometry.py#L255)。直接调用：`PairPermission`, `any`, `bool`, `first.removeprefix`, `first.startswith`, `model.permission`, `name.removeprefix`, `name.startswith`, `np.asarray`, `policy['bounds'][0, column].detach`, `policy['bounds'][0, column].detach().numpy`, `policy['normals'][0, column].detach`, `policy['normals'][0, column].detach().numpy`, `regions.append`, `relation.active`, `second.removeprefix`, `second.startswith`, `self.scene._initial_support_models.values`。

### `GeometryConstraints._native_interior_witness(self, native, native_matrix, field, field_matrix, world, permission, excluded)`

Find actual observed material strictly inside the unchanged native solid.

[定义](../../manipulation/preference_planner/geometry.py#L293)。直接调用：`_allowed`, `bool`, `float`, `id`, `inverse_points`, `native.computeCOM`, `native_point_value`, `np.array_equal`, `np.asarray`, `occupied_material`, `self.scene.check_request`, `torch.as_tensor`。

### `GeometryConstraints.add_evidence(self, full_controls, states=None, *, pair_batch=None)`

Add necessary rows from a completed original full-curve query.

[定义](../../manipulation/preference_planner/geometry.py#L327)。直接调用：`WitnessRow`, `enumerate`, `first.startswith`, `inverse_points`, `len`, `map`, `np.asarray`, `occupied_material`, `pending.append`, `record.get`, `result['skipped'].append`, `second.startswith`, `selected_witness_records`, `self._context`, `self._identities.add`, `self._native_interior_witness`, `self._permission`, `self.discoveries.append`, `self.witnesses.append`, `torch.as_tensor`, `torch.no_grad`, `tuple`。

### `GeometryConstraints.rows(self, full_controls)`

Return fixed-dimensional necessary inequalities g <= 0.

[定义](../../manipulation/preference_planner/geometry.py#L398)。直接调用：`(relative * relative.new_tensor(np.asarray(result.normal))).sum`, `_allowed`, `_local`, `_transform`, `_world`, `asdict`, `bool`, `coal.DistanceRequest`, `coal.DistanceResult`, `coal.distance`, `derivative.detach`, `derivative.new_tensor`, `enumerate`, `field_point_value`, `float`, `full_controls.reshape`, `full_controls.sum`, `inverse_points`, `isinstance`, `native_point_value`, `np.any`, `np.asarray`, `occupied_material`, `pose`, `relative.new_tensor`, `reports.append`, `result.getNearestPoint1`, `result.getNearestPoint2`, `self._context`, `self._permission`, `self.scene.check_request`, `sum`, `torch.stack`, `tuple`, `value.detach`, `values.append`, `world.detach`, `world.detach().numpy`。

### `GeometryConstraints.describe(self)`

Return row provenance and actual query states for the planner report.

[定义](../../manipulation/preference_planner/geometry.py#L469)。直接调用：`asdict`。

## joint_computation.py

### `ReducedJointDomain.__init__(self, full_domain, terminal)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/joint_computation.py#L23)。直接调用：。

### `ReducedJointDomain.signed_constraints(self, controls, *, with_witnesses=False)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/joint_computation.py#L30)。直接调用：`self.full_domain.signed_constraints`, `self.terminal.expand`, `self.terminal.pullback`。

### `JointComputation.__init__(self, *, base, terminal, geometry)`

内部计算辅助；输入输出见对应组件的调用定义。

[定义](../../manipulation/preference_planner/joint_computation.py#L46)。直接调用：`ReducedJointDomain`, `base.progress.numpy`, `super`, `super().__init__`。

### `JointComputation.states(self, controls)`

Evaluate q and its first three derivatives from the same full C.

[定义](../../manipulation/preference_planner/joint_computation.py#L58)。直接调用：`self.terminal.expand`, `super`, `super().states`。

### `JointComputation.task(self, controls, *, solver_rows_out=None)`

Append fixed necessary geometry rows to the original task rows.

[定义](../../manipulation/preference_planner/joint_computation.py#L62)。直接调用：`self.geometry.rows`, `self.terminal.expand`, `super`, `super().task`, `torch.cat`, `torch.ones_like`。

### `JointComputation.objective_components(self, controls, *, include_geometry=True)`

Compute all preferences once on the same robot states.

[定义](../../manipulation/preference_planner/joint_computation.py#L77)。直接调用：`costs.items`, `float`, `len`, `self.robot_states`, `self.states`, `self.terminal.expand`, `self.terminal.expand(controls).detach`, `self.terminal.expand(controls).detach().cpu`, `self.terminal.expand(controls).detach().cpu().numpy`, `self.terminal.expand(controls).detach().cpu().numpy().tolist`, `self.trial_history.append`, `trajectory_objective_components`, `value.detach`。

## solver.py

### `solve_trajectory(computation, initial, *, iterations, check_request, pending_geometry_bytes, report_out)`

Submit the full problem with batched task and geometry derivatives.

[定义](../../manipulation/preference_planner/solver.py#L22)。直接调用：`ConstrainedSolution`, `abs`, `batch_derivative_storage_bytes`, `bool`, `cache.clear`, `check_request`, `components.values`, `computation.domain.signed_constraints`, `computation.geometry.workspace_bytes`, `computation.objective_components`, `computation.task`, `constant.tolist`, `constraints.append`, `deepcopy`, `derivative_memory.values`, `evaluate`, `feasible.all`, `finite.tolist`, `fixed_constraints.append`, `float`, `geometry_workspace.get`, `gradient.detach`, `gradient.detach().numpy`, `gradient.detach().numpy().ravel`, `group.items`, `group.values`, `indices.tolist`, `initial.copy`, `initial.ravel`, `initial.ravel().copy`, `initial_task[0][computation.fixed_equality_rows].detach`, `initial_task[0][computation.fixed_equality_rows].detach().numpy`, `int`, `layout.numel`, `len`, `list`, `matched.get`, `max`, `memory.pop`, `memory.update`, `memory.values`, `minimize`, `np.abs`, `np.abs(initial_gradient).max`, `np.array_equal`, `np.asarray`, `np.asarray(snapshot['controls']).ravel`, `np.isfinite`, `np.max`, `perf_counter`, `report.update`, `require_host_memory`, `required.issubset`, `result.x.reshape`, `result.x.reshape(shape).copy`, `set`, `snapshot.update`, `snapshot['completed_components'].append`, `snapshot_for`, `solver_storage_bytes`, `str`, `sum`, `task_jacobian`, `torch.autograd.grad`, `torch.cat`, `torch.nonzero`, `torch.nonzero(~selected[offset:offset + size]).flatten`, `torch.tensor`, `torch.zeros`, `tuple`, `type`, `value.detach`, `value.numel`, `value.reshape`, `value.sum`, `values.detach`, `values.detach().numpy`, `values[offset:offset + size][indices].detach`, `values[offset:offset + size][indices].detach().numpy`, `x.copy`, `x.reshape`, `x.reshape(shape).tolist`, `zip`。

## derivatives.py

### `task_jacobian(values, controls, *, check_request, admit_work)`

Differentiate the supplied Torch graph in its original column layout.

[定义](../../manipulation/preference_planner/derivatives.py#L13)。直接调用：`admit_work`, `check_request`, `controls.numel`, `controls.sum`, `derivative.detach`, `derivative.detach().numpy`, `derivative.detach().numpy().reshape`, `torch.autograd.grad`, `torch.eye`, `values.numel`。

### `batch_derivative_storage_bytes(*, rows, variables, samples, frames, joints, geometry_workspace)`

Return batch storage in addition to the original pending geometry.

[定义](../../manipulation/preference_planner/derivatives.py#L36)。直接调用：`map`, `max`。

