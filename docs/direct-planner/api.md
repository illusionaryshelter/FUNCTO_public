# 直接规划器 API 与实现索引

实际代码基线与完整数学定义见 [组件架构](README.md)。本页的函数签名和包内依赖由当前源码生成；模块契约说明数据输入、计算职责及返回内容。

公开包入口为 `DirectPreferenceBackend`、`DirectPreferencePlanner`、`ReferenceObjective`、`plan_trajectory` 与原 `bind_pose_objectives`。内部模块通过包内相对导入组合，实验文件读取由调用方负责。

## direct_preference：公开后端与数值规划对象

[源码](../../manipulation/direct_planner/direct_preference.py)。输入：PlanningProblem、OnlinePlanningScene、官方机器人配置、验证器、预算、ReferenceObjective 与初始化声明。

计算和输出：DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。

直接包内依赖：`candidate_result`, `direct_computation`, `direct_lifecycle`, `direct_setup`, `geometry_constraints`, `nominal_hand_geometry`, `reference_initialization`, `reference_objective`, `reference_projection`, `shared_solver`, `terminal_binding`。包内调用组件：包公开入口或调用方。

| API 签名 | 调用作用 |
|---|---|
| `DirectPreferencePlanner.__init__(self, *args, reference_objective, initialization, incoming_terminal_contract=None, **kwargs)` | DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。 |
| `DirectPreferencePlanner.rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None)` | DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。 |
| `DirectPreferencePlanner._controls_curve(self, controls)` | Associate the original Hermite export with its exact coefficient array. |
| `DirectPreferencePlanner._candidate_domain_validation(self, trajectory)` | Read original continuous joint bounds and the bound terminal suffix. |
| `DirectPreferencePlanner._bind_computation(self, extra_parameters=())` | Share calibrated FK and one original curve across all consumers. |
| `DirectPreferencePlanner._candidate_objective(self, trajectory, candidate_scene)` | Align original query identities and bind the candidate's actual clock. |
| `DirectPreferencePlanner.diagnostic_snapshot(self)` | Attach the actual reference weighting to the original owned records. |
| `DirectPreferencePlanner.evaluate_candidate(self, trajectory, *, record_work, objective)` | Read the retained solver point through the local candidate contract. |
| `DirectPreferencePlanner.reference_seeds(self, joint_reference=None)` | Rank measured-entry proposals with the same problem used by the NLP. |
| `DirectPreferencePlanner.optimize(self, seed, *, iterations=None)` | Solve each explicitly supplied initial coefficient array once. |
| `DirectPreferencePlanner._expand_snapshots(self, runtime)` | DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。 |
| `DirectPreferencePlanner._retain_diagnostic_trial(self, runtime, solve_index)` | Keep an already evaluated curve when further work is interrupted. |
| `DirectPreferenceBackend.__init__(self, *args, reference_objective, initialization, **kwargs)` | DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。 |
| `DirectPreferenceBackend.bind_request(self, problem, scene, *, budget, continuation=None, goal_identities=None)` | Bind the local computation while retaining resident resource checks. |
| `DirectPreferenceBackend.plan(self, problem, scene, *, continuation=None, request_ledger=None, rebind=None, cancelled=None, deadline=None, joint_reference=None)` | Use the local initialization/optimization lifecycle with original inputs. |

## direct_setup：时钟、官方配置与原物理域

[源码](../../manipulation/direct_planner/direct_setup.py)。输入：当前具名 q/dq、任务事件、现场手指模型、机器人配置及原进度/续接声明。

计算和输出：绑定参考时钟、原配置的关节位置/速度/加速度/jerk 界限、CubicJointDomain 与共享运动学；候选域检查输出连续极值报告。

直接包内依赖：`differentiable_kinematics`。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `DirectSetup.__init__(self, problem, scene, robot, budget, device, *, robot_validator, continuation=None, goal_identities=None)` | 绑定参考时钟、原配置的关节位置/速度/加速度/jerk 界限、CubicJointDomain 与共享运动学；候选域检查输出连续极值报告。 |
| `DirectSetup._bind_request_clock(self, problem, scene, budget, continuation, goal_identities)` | Bind original task/reference/hand clocks from the measured entrance. |
| `DirectSetup._bind_robot_configuration(self, robot)` | Decode the same official joint order, limits and motion-cost values. |
| `DirectSetup._bind_request_computation(self)` | Bind one original reference and physical domain, then local components. |
| `DirectSetup.rebind(self, problem, scene, budget, *, continuation=None, goal_identities=None)` | Replace request data while retaining the fixed official joint configuration. |
| `DirectSetup.close(self)` | Restore the caller's arithmetic policy when this request owner closes. |
| `DirectSetup._candidate_domain_validation(self, trajectory)` | Evaluate original continuous polynomial limits from their direct owner. |

## direct_lifecycle：累计请求与候选交付

[源码](../../manipulation/direct_planner/direct_lifecycle.py)。输入：任务、现场、调用方持有的后端和预算，以及可选请求账本、续接、取消、截止时刻和关节参考。

计算和输出：完成输入绑定、初始化、求解、候选资格及原执行绑定，返回包含曲线或失败原因的 PlanningResult 和累计账本。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `plan_trajectory(problem, scene, *, backend, budget, continuation=None, request_ledger=None, rebind=None, cancelled=None, deadline=None, joint_reference=None)` | Optimize a bounded request and deliver its first qualified solver curve. |

## direct_computation：固定数值问题与同点共享

[源码](../../manipulation/direct_planner/direct_computation.py)。输入：原 TrajectoryComputation、末段仿射映射、几何提供器、完整参考目标及自由曲线系数。

计算和输出：DirectComputation/BoundComputation 建立固定行列结构；ComputationPoint 保存目标、条件、稀疏导数、空间见证和导出候选所需的同点状态。

直接包内依赖：`coefficient_coordinates`, `constraint_envelope`, `material_hull`, `reference_objective`, `sparse_physical`, `sparse_rows`, `sparse_task`, `support_linearization`。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `DirectComputation.__init__(self, *, base, terminal, geometry, reference_objective)` | DirectComputation/BoundComputation 建立固定行列结构；ComputationPoint 保存目标、条件、稀疏导数、空间见证和导出候选所需的同点状态。 |
| `DirectComputation.shared_context(self, controls)` | Evaluate original states and named frame Jacobians once at this x. |
| `DirectComputation.task(self, controls, *, solver_rows_out=None)` | Return the original complete declared task rows for inspection. |
| `DirectComputation.task_linearization(self, controls, *, solver_rows_out=None, context=None)` | Return values and factored analytic blocks in the original row order. |
| `DirectComputation.objective_components(self, controls, *, context=None, producer=None)` | Use one FK result for original costs and the complete TCP reference. |
| `DirectComputation.bind_solver(self, initial, *, check_request)` | Create one fixed task/geometry/physical sparse numerical problem. |
| `BoundComputation.__init__(self, computation, initial, *, check_request)` | DirectComputation/BoundComputation 建立固定行列结构；ComputationPoint 保存目标、条件、稀疏导数、空间见证和导出候选所需的同点状态。 |
| `BoundComputation.produce(self, name, function)` | Charge actual completed/failed producer work to this fixed problem. |
| `BoundComputation.solver_coordinates(self, point)` | Keep coefficient conditioning separate from physical value producers. |
| `BoundComputation.evaluate(self, controls)` | Build an exact-x point; same-point caching belongs to the solver. |
| `ComputationPoint.__init__(self, bound, controls, context, task_values, task_blocks)` | DirectComputation/BoundComputation 建立固定行列结构；ComputationPoint 保存目标、条件、稀疏导数、空间见证和导出候选所需的同点状态。 |
| `ComputationPoint.objective_gradient(self)` | Use the shared native first-order graph once for the complete cost. |
| `ComputationPoint.jacobian_values(self)` | Concatenate producer-owned sparse values, retaining no dense matrix. |

## reference_objective：完整参考与统一目标

[源码](../../manipulation/direct_planner/reference_objective.py)。输入：显式误差尺度和权重、BoundToolReference、物理查询时刻及共享工具位姿/速度。

计算和输出：冻结原参考样本和归一化物理积分权重，返回完整参考、角运动、关节运动及原声明偏好的分项标量和可微图。

直接包内依赖：`scene_relationships`。包内调用组件：`direct_computation`, `direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `ReferenceObjective.__post_init__(self)` | 冻结原参考样本和归一化物理积分权重，返回完整参考、角运动、关节运动及原声明偏好的分项标量和可微图。 |
| `ReferenceObjective.bind(self, bound_reference, physical_times)` | Freeze one reference on ordinary times in seconds from entry. |
| `ReferenceObjective.bind_clock(self, source_reference, *, problem, progress_u, progress_s, duration, physical_times)` | Bind a supplied candidate clock through the original reference API. |
| `BoundReferenceObjective.cost(self, position, quaternion)` | Return relative_weight times the dimensionless TCP error mean. |
| `BoundReferenceObjective.report(self, constraint_weight)` | Describe the actual declared weighting and frozen physical grid. |
| `BoundReferenceObjective.angular_motion_cost(self, quaternion)` | Integrate squared finite angular speed over the same physical clock. |
| `trajectory_objective_components(*, reference_objective, material_samples, problem, scene, states, robot_state, progress, times, ordinary_indices, bounds, motion_configuration, native_dt, constraint_weight, scene_relation_report=None, producer=None)` | Compose original task costs, declared scene relations and TCP reference. |

## reference_initialization：官方运动链与末端提案

[源码](../../manipulation/direct_planner/reference_initialization.py)。输入：Pinocchio 机器人模型、具名关节、工具和基座标定、原参考、入口 q/dq、关节区间及显式提案预算。

计算和输出：SymbolicKinematics 提供符号 FK、位姿 Jacobian 和连续四元数；有界拟合及原物理时钟连接生成全部末端分支提案和证据。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`coefficient_coordinates`, `direct_preference`, `reference_projection`。

| API 签名 | 调用作用 |
|---|---|
| `_quaternion_product(first, second)` | Compose wxyz unit quaternions without choosing a new matrix sign. |
| `_fixed_quaternion(transform)` | SymbolicKinematics 提供符号 FK、位姿 Jacobian 和连续四元数；有界拟合及原物理时钟连接生成全部末端分支提案和证据。 |
| `SymbolicKinematics.__init__(self, model, *, joint_names, fixed_positions, frame, reference_from_base, frame_from_tool)` | SymbolicKinematics 提供符号 FK、位姿 Jacobian 和连续四元数；有界拟合及原物理时钟连接生成全部末端分支提案和证据。 |
| `SymbolicKinematics.sample(self, positions)` | Return [sample,4,4] poses in the calibrated reference frame. |
| `bounded_pose_fit(kinematics, pose, seed, bounds, *, position_scale, rotation_scale, maximum_evaluations)` | Return an ordinary bounded numerical proposal and its actual error. |
| `event_reference_samples(kinematics, *, times, reference, start_q, start_dq, position_bounds, velocity_bounds, event_times, position_scale, rotation_scale, random_seed, proposal_count, maximum_evaluations, check_request=lambda : None)` | Propose bounded terminal states and connect from the measured q/dq. |

## reference_projection：完整参考初值拟合

[源码](../../manipulation/direct_planner/reference_projection.py)。输入：原曲线与末段映射、同点机器人声明、完整参考、required 事件以及末端连接系数。

计算和输出：编译稀疏残差和 Jacobian，在原系数位置盒内执行 SciPy 最小二乘，返回拟合系数、目标分项、次数及停止状态；原完整问题负责后续排序。

直接包内依赖：`reference_initialization`。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `_conjugate(q)` | 编译稀疏残差和 Jacobian，在原系数位置盒内执行 SciPy 最小二乘，返回拟合系数、目标分项、次数及停止状态；原完整问题负责后续排序。 |
| `_rotate(q, v)` | 编译稀疏残差和 Jacobian，在原系数位置盒内执行 SciPy 最小二乘，返回拟合系数、目标分项、次数及停止状态；原完整问题负责后续排序。 |
| `_rotation_vector(q)` | Use the original principal rotation-vector identity limit. |
| `_state_residuals(states, bounds, configuration, native_dt, weights, multiplier)` | Factor the existing joint STATE cost, including its physical clock. |
| `coefficient_position_box(curve, terminal, bounds)` | Bound this initializer's complete cubic coefficients in the joint box. |
| `ReferenceProjection.__init__(self, computation, context, *, check_request)` | 编译稀疏残差和 Jacobian，在原系数位置盒内执行 SciPy 最小二乘，返回拟合系数、目标分项、次数及停止状态；原完整问题负责后续排序。 |
| `ReferenceProjection.fit(self, initial, *, maximum_evaluations)` | Return one bounded coefficient proposal with complete fit provenance. |

## terminal_binding：末段静止参数化

[源码](../../manipulation/direct_planner/terminal_binding.py)。输入：原三次曲线基、参考进度时钟与终端来源。

计算和输出：返回 TerminalHold 的仿射系数映射；expand/reduce/pullback 分别恢复完整系数、求自由坐标和传递导数。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `bind_terminal_hold(curve_map, clock_u, clock_z, *, source)` | Return the original TerminalHold contract with orthonormal sparse columns. |

## coefficient_coordinates：固定求解变量坐标

[源码](../../manipulation/direct_planner/coefficient_coordinates.py)。输入：完整参考尺度、原时间积分、初值关节状态、关节范围与原样条映射。

计算和输出：按关节块形成度量并 Cholesky 分解，返回固定线性变量映射及条件数；原目标和约束在恢复后的原系数上计算。

直接包内依赖：`reference_initialization`。包内调用组件：`direct_computation`。

| API 签名 | 调用作用 |
|---|---|
| `coefficient_coordinates(computation, point)` | Return S for C = C_initial + S z and its numerical definition. |

## differentiable_kinematics：共享原生运动学与反向接口

[源码](../../manipulation/direct_planner/differentiable_kinematics.py)。输入：同一官方 Pinocchio 模型、请求参考系标定、具名关节 q/dq 和参考连杆集合。

计算和输出：一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_setup`。

| API 签名 | 调用作用 |
|---|---|
| `_PinPoses.forward(ctx, position, kinematics)` | 一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。 |
| `_PinPoses.backward(ctx, gradient)` | 一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。 |
| `_PinStates.forward(ctx, position, velocity, kinematics)` | 一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。 |
| `_PinStates.backward(ctx, pose_gradient, twist_gradient, geometric_gradient)` | 一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。 |
| `DifferentiablePinKinematics.poses(self, position, joint_names, *, fk_session=None)` | Return the inherited frame mapping using this module's pose node. |
| `DifferentiablePinKinematics.states(self, position, velocity, joint_names, *, fk_session=None)` | Return poses and reference-frame twists with the original layout. |
| `DifferentiablePinKinematics.states_with_jacobians(self, position, velocity, joint_names)` | Return poses, twists and the same native geometric frame Jacobians. |
| `DifferentiablePinKinematics._state_outputs(self, position, velocity, joint_names, *, with_jacobians, fk_session)` | Evaluate one named state batch and expose only requested products. |

## support_linearization：任务事件单位和支撑解析因子

[源码](../../manipulation/direct_planner/support_linearization.py)。输入：原任务声明、完整具名机器人状态、连杆 Jacobian、曲线系数和技能材料点索引。

计算和输出：返回等式/不等式向量、原行掩码及支撑体积/足迹的解析因子；required 位姿集合采用等价的归一化平方残差，其余任务使用原计算。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_computation`, `geometry_constraints`。

| API 签名 | 调用作用 |
|---|---|
| `normalized_pose_event(requirement, position, quaternion)` | Return dimensionless inequalities for the original pose-event set. |
| `SupportLinearization.row_values(self, row, columns)` | Fill one declared sparse row in [reduced control,named joint] order. |
| `linearized_robot_state(computation, q, dq)` | Return original robot_tool_states products plus same-call frame J. |
| `_support_factors(computation, requirement, robot_state, jacobians, components, material_indices)` | Pull one original event's rigid material points through frame J and W E. |
| `task_linearization(computation, controls, *, support_material_indices, solver_rows_out=None, robot_state=None, jacobians=None)` | Return task rows, normalized event sets and factored analytic support blocks. |

## sparse_task：任务稀疏系数导数

[源码](../../manipulation/direct_planner/sparse_task.py)。输入：原条件掩码、支撑因子、事件时刻与曲线局部支撑结构。

计算和输出：选择实际求解行并建立每行结构列；解析支撑与剩余自动微分共同填写同一稀疏数值槽。

直接包内依赖：`sparse_rows`。包内调用组件：`direct_computation`。

| API 签名 | 调用作用 |
|---|---|
| `SparseTaskRows.__init__(self, computation, values, blocks, declarations, context)` | 选择实际求解行并建立每行结构列；解析支撑与剩余自动微分共同填写同一稀疏数值槽。 |
| `SparseTaskRows._event_column(self, progress)` | 选择实际求解行并建立每行结构列；解析支撑与剩余自动微分共同填写同一稀疏数值槽。 |
| `SparseTaskRows._columns(self, name, shape)` | Resolve all current generic PlanningProblem scalar row producers. |
| `SparseTaskRows.select_values(self, values)` | Take the same fixed scalar selection from this point's task vector. |
| `SparseTaskRows.jacobian_values(self, values, controls, blocks, *, check_request)` | Write local analytic slots or one scalar VJP directly per row. |

## sparse_physical：连续关节条件与导数

[源码](../../manipulation/direct_planner/sparse_physical.py)。输入：原 CubicJointDomain、末段映射和完整关节曲线系数。

计算和输出：返回各多项式段的位置/速度极值、加速度端值、jerk 有符号条件和局部系数导数，保留原物理秒单位。

直接包内依赖：`sparse_rows`。包内调用组件：`direct_computation`。

| API 签名 | 调用作用 |
|---|---|
| `PhysicalPoint.jacobian_values(self)` | 返回各多项式段的位置/速度极值、加速度端值、jerk 有符号条件和局部系数导数，保留原物理秒单位。 |
| `SparsePhysicalRows.__init__(self, full_domain, terminal)` | 返回各多项式段的位置/速度极值、加速度端值、jerk 有符号条件和局部系数导数，保留原物理秒单位。 |
| `SparsePhysicalRows.evaluate(self, full_controls, *, check_request)` | Reuse native extrema and fill only each row's declared numeric slots. |

## sparse_rows：固定稀疏行列布局

[源码](../../manipulation/direct_planner/sparse_rows.py)。输入：各条件行实际依赖的系数列和总变量数。

计算和输出：生成按行存储、CCS 交付次序和局部槽索引；任务、几何、关节域和求解器共用这一明确结构。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`constraint_envelope`, `direct_computation`, `geometry_constraints`, `native_interval_distance`, `sparse_physical`, `sparse_task`。

| API 签名 | 调用作用 |
|---|---|
| `SparseRows.__init__(self, columns_by_row, variables)` | 生成按行存储、CCS 交付次序和局部槽索引；任务、几何、关节域和求解器共用这一明确结构。 |
| `SparseRows.storage_bytes(self)` | Resident index arrays; numeric values are owned by a current point. |
| `coefficient_columns(coefficient_rows, joint_columns, joints)` | Return the structural product of fixed cubic rows and named joints. |

## constraint_envelope：同实体条件的最大值表示

[源码](../../manipulation/direct_planner/constraint_envelope.py)。输入：完整几何行、原行身份与结构列。

计算和输出：按同一注册实体对及条件取最大值，返回等价的整体不等式和实际胜出行；稀疏导数使用该行的原数值。

直接包内依赖：`sparse_rows`。包内调用组件：`direct_computation`。

| API 签名 | 调用作用 |
|---|---|
| `ConstraintEnvelope.__init__(self, source_pattern, group_keys)` | 按同一注册实体对及条件取最大值，返回等价的整体不等式和实际胜出行；稀疏导数使用该行的原数值。 |
| `ConstraintEnvelope.select(self, original)` | Return exact maxima and the first original maximizing row at ties. |
| `ConstraintEnvelope.jacobian_values(self, original, winners)` | Map the active branch into its fixed union of coefficient columns. |

## geometry_constraints：原模型的几何条件与空间见证

[源码](../../manipulation/direct_planner/geometry_constraints.py)。输入：原任务查询时刻、场景实体/关系、官方机器人模型、名义手指和完整曲线系数。

计算和输出：固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。

直接包内依赖：`batch_distance`, `native_interval_distance`, `sparse_rows`, `support_linearization`。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `pair_identity(record)` | Keep all physical, relation, permission and declared pair identities. |
| `_query_records(base, progress_map)` | Use the original exact/left/right scene-boundary clock declarations. |
| `_pair_records(scene)` | Enumerate the same registered pairs as the original scene objective. |
| `_state_layout(base, records)` | Retain original base ordering and share exact added state identities. |
| `GeometryConstraints.__init__(self, computation, *, progress_map)` | 固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。 |
| `GeometryConstraints._prepare_materials(self)` | Prepare original field/native arrays once for this observed scene. |
| `GeometryConstraints.row_count(self)` | 固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。 |
| `GeometryConstraints.workspace_bytes(self)` | Count fixed row arrays; runtime process/device limits own total use. |
| `GeometryConstraints._context(self, full_controls, bound, *, shared_context=None)` | Share base FK/J and compute only added physical/relation query states. |
| `GeometryConstraints.bind_solver(self, terminal)` | Bind one immutable sparse layout to the same scene query arrays. |
| `GeometryConstraints.describe(self)` | Expose actual query cost and the scene location of its worst row. |
| `BoundGeometry.__init__(self, owner, terminal)` | 固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。 |
| `BoundGeometry._source(self, name, column)` | Resolve original pose composition to a named frame/query or constant. |
| `BoundGeometry.evaluate(self, full_controls, *, shared_context=None)` | Evaluate one coefficient point with grouped numerical library calls. |
| `GeometryPoint.__init__(self, bound, values, factors, witnesses, unknown_counts, query_counts)` | 固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。 |
| `GeometryPoint.jacobian_values(self)` | 固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。 |

## batch_distance：GPU 场查询与 CPU 原生计算的组合

[源码](../../manipulation/direct_planner/batch_distance.py)。输入：明确注册的几何对象、机器人及声明运动实体源、原行对应的位姿/许可。

计算和输出：建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。

直接包内依赖：`gpu_field`。包内调用组件：`geometry_constraints`。

| API 签名 | 调用作用 |
|---|---|
| `PointInventory.__init__(self, samples, excluded=(), *, native_points=None)` | 建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。 |
| `native_volume_points(geometry, spacing, *, device='cpu')` | Sample a registered robot solid on a fixed local Cartesian lattice. |
| `NativePointQuery.__init__(self, geometry, *, device)` | 建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。 |
| `NativePointQuery.sample(self, points)` | Return signed distance and its local-coordinate first derivative. |
| `BatchDistance.__init__(self, geometries, *, query_sources, device='cuda:0')` | 建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。 |
| `BatchDistance.inventory(self, name, excluded)` | Prepare a declared original-cell source subset before its first use. |
| `BatchDistance.field(self, name, excluded)` | Bind an original target's explicit cell exclusion on both devices. |
| `BatchDistance.submit(self, *, records, columns, matrices, permissions, active, check_request)` | Enqueue independent field queries and return their pending result. |
| `BatchDistance.fields_by_name(self)` | 建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。 |
| `PendingDistance.__init__(self, owner, count, started)` | 建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。 |
| `PendingDistance.finish(self)` | Wait once, then compute derivatives at the original winning points. |

## gpu_field：原标量场和官方机器人实体的设备查询

[源码](../../manipulation/direct_planner/gpu_field.py)。输入：float64 点/刚体位姿、现场 TSDF 数值/支持、官方机器人模型资源与有限接触许可。

计算和输出：Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`batch_distance`。

| API 签名 | 调用作用 |
|---|---|
| `_field_value(point: wp.vec3d, values: wp.array(dtype=wp.float32, ndim=3), valid: wp.array(dtype=wp.bool, ndim=3), origin: wp.vec3d, h: wp.float64)` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_native_value(p: wp.vec3d, mesh: wp.uint64, kind: wp.int32, shape: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d, unit: wp.float64, search_distance: wp.float32)` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_permitted(point: wp.vec3d, row: wp.int32, row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32), normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64))` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_transform(point: wp.vec3d, transform: wp.mat44d)` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_field_values(points: wp.array(dtype=wp.vec3d), values: wp.array(dtype=wp.float32, ndim=3), valid: wp.array(dtype=wp.bool, ndim=3), origin: wp.vec3d, h: wp.float64, result: wp.array(dtype=wp.float64), supported: wp.array(dtype=wp.bool))` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_store_tile(row: wp.int32, block: wp.int32, lane: wp.int32, column: wp.int32, value: wp.float64, queried: wp.int32, unknown: wp.int32, maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2), counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2))` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_field_pair_tiles(block_width: wp.int32, points: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d), row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32), normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64), values: wp.array(dtype=wp.float32, ndim=3), valid: wp.array(dtype=wp.bool, ndim=3), origin: wp.vec3d, h: wp.float64, scale: wp.float64, maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2), counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2))` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_native_pair_tiles(block_width: wp.int32, points: wp.array(dtype=wp.vec3d), poses: wp.array(dtype=wp.mat44d), row_regions: wp.array(dtype=wp.int32), region_planes: wp.array(dtype=wp.int32), normals: wp.array(dtype=wp.vec3d), bounds: wp.array(dtype=wp.float64), mesh: wp.uint64, kind: wp.int32, shape: wp.vec3d, lower: wp.vec3d, upper: wp.vec3d, scale: wp.float64, distance: wp.float32, maxima: wp.array(dtype=wp.float64, ndim=2), winners: wp.array(dtype=wp.int32, ndim=2), counts: wp.array(dtype=wp.int32, ndim=2), unknowns: wp.array(dtype=wp.int32, ndim=2))` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `_pair_inputs(points, source_poses, target_poses, permissions)` | Compose small pose/plane arrays before querying every fixed local point. |
| `submit_pair_maxima(target, points, source_poses, target_poses, permissions, scale)` | Enqueue finite maxima, winning indices and support counts on the device. |
| `DeviceField.__init__(self, field, device)` | Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。 |
| `DeviceField.sample(self, points)` | Return tensor values and support for an arbitrary leading layout. |

## native_interval_distance：机器人实体距离的区间驻点

[源码](../../manipulation/direct_planner/native_interval_distance.py)。输入：官方机器人实体对、原端点距离、曲线多项式、关节状态、连杆 Jacobian 和名义手指。

计算和输出：原连续速度范围排除常量尾区间，其余区间以距离时间导数括根并调用 Brent，返回胜出时刻、距离、最近点及关节导数。

直接包内依赖：`native_gradient`, `sparse_rows`。包内调用组件：`geometry_constraints`。

| API 签名 | 调用作用 |
|---|---|
| `NativeIntervalDistance.__init__(self, bound)` | 原连续速度范围排除常量尾区间，其余区间以距离时间导数括根并调用 Brent，返回胜出时刻、距离、最近点及关节导数。 |
| `NativeIntervalDistance._joint_gradient(self, pair, numeric, origins, jacobians)` | Pull original nearest-point covectors into complete named joints. |
| `NativeIntervalDistance._at(self, pair, moment, polynomial, complete_entry)` | Query one original pair at an explicit physical curve time. |
| `NativeIntervalDistance.evaluate(self, controls, context, native_numeric, active)` | Return right-endpoint or stationary interval winners and evidence. |

## native_gradient：最近点到关节的编译导数

[源码](../../manipulation/direct_planner/native_gradient.py)。输入：12 项原生距离记录、两参考连杆原点、两组 6×N Jacobian 和共同祖先列掩码。

计算和输出：Numba float64 CPU ufunc 输出 N 列距离导数，支持标量、批次及 strided 数组；共同刚体运动对应列为零。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`native_interval_distance`。

| API 签名 | 调用作用 |
|---|---|
| `native_joint_gradient(numeric, first_origin, second_origin, first_jacobian, second_jacobian, common_ancestor, result)` | Pull the same rigid covectors through the two original frame Jacobians. |

## nominal_hand_geometry：声明的位置控制手指模型

[源码](../../manipulation/direct_planner/nominal_hand_geometry.py)。输入：原手指命令多项式、任务条件、事件时钟及声明进度范围。

计算和输出：返回同一时钟上的名义手指位置、速度和条件未知区间；模型由命令、声明关系及关节反馈建立。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `bind_position_controlled_finger_geometry(reference, conditions, event_clock, *, progress_interval, source)` | Bind the declared held-to-free model using existing occurrence identities. |

## scene_relationships：显式关系的目标项

[源码](../../manipulation/direct_planner/scene_relationships.py)。输入：原现场接触/初始支撑声明、固定材料点、工具位姿、物理时间和技能进度。

计算和输出：调用原关系目标计算，返回同一图中的关系偏好标量及普通证据；实体非穿透由几何条件提供。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`reference_objective`。

| API 签名 | 调用作用 |
|---|---|
| `scene_relationship_objective(scene, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, material_samples, report_out=None)` | Return the original required-contact and initial-support scalar sum. |

## material_hull：技能材料点的线性约束索引

[源码](../../manipulation/direct_planner/material_hull.py)。输入：技能表示中的完整材料点数组。

计算和输出：返回半空间约束所需的原点索引，按实际仿射维数选择线性极值点；原技能点数组和软目标保持完整。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_computation`。

| API 签名 | 调用作用 |
|---|---|
| `material_hull_indices(points)` | Return original indexes retaining every linear halfspace extremum. |

## candidate_result：同点候选条件与正式验证元数据

[源码](../../manipulation/direct_planner/candidate_result.py)。输入：求解器保留的 ComputationPoint、其原曲线、任务/场景声明、连续关节域和统一目标。

计算和输出：消费同点原任务与几何值，补齐连续域、末段及声明关系信息。位姿事件的原超限、实际误差和容差进入独立 `pose_accuracy` 事后警告；其余条件组成接受/拒绝结果和执行绑定的违反量。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `_maximum(value)` | 消费同点原任务与几何值，补齐原连续域、末段及声明关系信息，形成原绑定接口消费的接受/拒绝结果及具体证据。 |
| `scene_conditions(point)` | Evaluate declared relationship and unknown policy on retained states. |
| `evaluate_candidate(planner, trajectory, *, record_work, objective)` | 输出本次曲线的物理/任务条件判定、独立位姿精度警告和执行绑定元数据；名义优化约束仍由 PlanningProblem 原接口提供。 |

## shared_solver：CasADi/IPOPT 与稀疏回调

[源码](../../manipulation/direct_planner/shared_solver.py)。输入：已绑定的同一数值问题、自由初值、实际预算/中断接口和固定数值坐标。

计算和输出：精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。

直接包内依赖：原 FUNCTO / 数值库接口。包内调用组件：`direct_preference`。

| API 签名 | 调用作用 |
|---|---|
| `_Jacobian.__init__(self, name, owner, input_names, output_names)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_n_in(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_n_out(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_name_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_name_out(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_sparsity_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.get_sparsity_out(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Jacobian.eval(self, arguments)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.__init__(self, bound, check_request)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_n_in(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_n_out(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_name_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_name_out(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_sparsity_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_sparsity_out(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.has_jac_sparsity(self, output, input_)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_jac_sparsity(self, output, input_, symmetric)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.has_jacobian(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.get_jacobian(self, name, input_names, output_names, options)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.record_abort(self, error)` | Preserve the first Python cause before CasADi translates the error. |
| `_Oracle.guard(self, function)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.point(self, flat)` | Create only complete same-problem cache entries, never partial ones. |
| `_Oracle.eval(self, arguments)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Oracle.snapshot(record)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.__init__(self, owner, coordinate_map)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.get_n_in(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.get_n_out(self)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.get_name_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.get_sparsity_in(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.get_sparsity_out(self, index)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `_Iteration.eval(self, arguments)` | 精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。 |
| `sparse_solver_storage_bytes(*, variables, constraints, nonzeros)` | Count owned numeric buffers and a sparse solver work estimate. |
| `solve_trajectory(bound, initial, *, iterations, check_request, remaining_seconds, on_solver_start, pending_geometry_bytes, report_out, geometry_workspace=None)` | Solve one fixed problem and return the original ConstrainedSolution. |
