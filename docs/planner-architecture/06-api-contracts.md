# API 契约与调用关系

代码基线：`__NATIVE_MODEL_COMMIT__`。本目录按运行阶段组织接口的作用、必要输入、输出和调用者。完整签名、dataclass 字段、返回表达式和内部调用符号见[逐符号实现索引](07-implementation-index.md)。同一对象的方法共享其构造时绑定的数据；表中省略的 `self` 指向该已构造实例。

批量张量记为 `B` 个候选、`H` 个时间点和 `J` 个关节。常用布局为关节 `[B,H,J]`、位姿位置 `[B,H,3]`、四元数 `[B,H,4]`。实体场保存自身 XYZ 栅格，候选仅改变其刚体位姿。NumPy 保存数组通常移除候选维。位姿四元数采用 `wxyz`，齐次变换将右侧坐标中的点映到左侧坐标。物理时间以秒计，平移以米计，角度以弧度计。

## 1. 参考表示与任务编译

以下接口对应[输入组件](01-inputs-and-task-model.md)。示教表示和当前任务关系提供参考，机器人反馈提供本次入口；最终 `PlanningProblem` 是数值求解统一读取的任务对象。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `representation.build_representation` | 建立源示教的相对功能运动；各位姿须共用源世界坐标 | 帧/时间、源物体/工具位姿、功能局部变换、事件 → 相对关系与来源记录；`generate_reference` |
| `sample_poses` | 在源时间轴上插值位姿 | 有序时刻及位姿、查询时刻 → 刚体变换序列；参考适配 |
| `generate_reference` | 将源功能运动绑定到当前功能坐标与入口 | 表示、当前功能位姿、坐标对应 `J`、附着 `A`、当前入口 → 物体/工具连续路径、数组、边界检查与缺失状态；任务适配器 |
| `program.ground_program` | 按技能种类绑定当前几何；持物声明与入口须代数一致 | `program,scene` → 工具参考、事件及几何诊断；`build_planning_problem` |
| `ground_object_program` | 生成当前物体参考 | 物体程序、当前关系与功能绑定 → 物体路径；持物参考调用者 |
| `function_motion.optimize_trajectory/transfer_function_motion` | 对已有参考施加明确的入口和功能关键帧 | 位姿序列、静态/当前入口、功能目标 → 适配位姿；`program` |
| `motion.MotionModel.poses` | 计算给定机构坐标对应的刚体位姿 | 已绑定轴/原点/类型、标量或向量坐标 → 位姿；机构参考与要求 |
| `MotionModel.coordinates_of` | 计算位姿对应坐标及模型相容性 | 机构模型、位姿 → 机构坐标；参考与机构诊断 |
| `build_planning_problem` | 编译一个技能、原事件、目标与过程要求 | `skills`、具名 `start_q/start_dq`、工具/基座标定、数值容差 → `PlanningProblem`；`plan_trajectory` |
| `PosePath.sample/sample_numpy` | 在给定进度读取已绑定路径 | 进度 → 位置与四元数；任务、物体运动 |
| `TransferredPosePath.sample` | 对参考施加固定入口与目标适配 | 已绑定适配数据、进度 → 位置与四元数；任务参考 |
| `reference_phase` | 将具名参考组织为阶段 | 参考、技能身份、来源 → `MotionPhase`；问题构造 |
| `event_occurrence_key` | 获取一次命令/事件的完整身份 | 事件记录 → `(skill_id,event_id,occurrence)`；规划、手指与执行 |
| `resolve_event_window` | 解析绝对或相对物理时间窗 | `EventTimeConstraint`、实际历史、可用预测原点 → 上下时间界；时钟绑定与执行 |
| `PoseProcessRequirement.error` | 计算过程关系偏差 | 候选位置/姿态/进度 → 指定过程残差；问题残差 |
| `PoseEventRequirement.errors/evaluate` | 计算确切功能目标误差与超容差量 | 候选工具位姿、已绑定材料/目标 → 位置/角度误差或正部和；硬验收 |
| `SupportRequirement.raw_inequalities` | 计算有限支撑四条件 | 候选工具位姿、物体点/材料、目标平面/边界 → gap/body/footprint/normal；硬条件与归一化 |
| `SupportRequirement.normalized_inequalities/evaluate` | 分别给出优化尺度和硬残差 | 四类原始条件与数值尺度 → 有符号归一化条件或正部和；AL 与验收 |
| `InteriorRequirement.evaluate` | 计算物体点在指定凸区域中的包含条件 | 候选位姿、点集、半空间、数值容差 → 非负越界；过程要求 |
| `MechanismRequirement.coordinates/errors` | 将候选位姿解释为机构坐标和流形偏差 | 已绑定机构、附着、候选位姿 → 坐标或模型误差；机构约束 |
| `MechanismRequirement.project_residual/history_complete` | 提供过程投影及转角历史完整性 | 位姿/进度 → 投影残差或历史完整状态；OMPL、任务验证 |
| `MechanismRequirement.coordinate_rates/direction_residuals` | 计算坐标变化率及方向要求 | 姿态、工具线/角速度、时间、原坐标起点 → 坐标速率/方向残差；优化与验收 |
| `CoordinateEventRequirement.absolute_target/error/evaluate` | 解析绝对/增量机构目标并检查候选 | 机构、入口坐标、目标与候选 → 绝对目标、误差或超容差量；事件验收 |
| `PlanningProblem.sample_reference/terminal_pose_target` | 提供进度参考或终端目标 | 已编译阶段、进度 → 工具参考或目标记录；IK 种子 |
| `source_time_map/source_time_to_progress/progress_to_source_time` | 在源时间和技能进度之间换算 | 阶段时间、查询值 → 映射数组或转换值；时钟与报告 |
| `transform_tool_pose/reference_to_link_pose` | 连接原生 FK 连杆与任务工具坐标 | 固定基座/工具标定、位姿 → 对应坐标中的位姿；FK 与 IK |
| `transform_tool_velocity` | 转换工具速度 | 线/角速度及连杆姿态 → 任务坐标中的工具速度；方向要求 |
| `transform_robot_spheres` | 转换原生状态的空球数组 | 当前纯运动学配置的空数组 → 同布局空数组；原生状态接口 |
| `residuals/evaluate` | 对同一候选状态计算全部已编译硬条件 | 位姿、进度、具名关节/速度、物理时间、确切目标身份 → 分项或聚合残差；原生 metrics |
| `functional_constraint_keys/functional_inequalities` | 给功能条件确定键与有符号尺度 | 确切事件样本、位姿/时间 → 条件键和归一化不等式；AL |
| `optimizer_preferences` | 计算过程/方向等优化偏好 | 候选状态、时间及优化阶段 → 偏好张量；代价管理器 |
| `measurements` | 保存可解释的任务原量 | 候选位姿与时间 → 测量字典；诊断 |
| `anchors/stop_progress` | 枚举确切任务时刻和停止要求 | 编译后任务 → 进度集合；固定时钟与查询布局 |
| `projection_requirements/project_residual` | 选择当前阶段的过程投影条件 | 进度及候选状态 → 要求集合或残差；OMPL 投影 |
| `goal_requirements_at/goal_residual` | 选择并计算原目标身份对应条件 | 进度、目标身份、候选 → 目标集合/残差；IK 排序与搜索终点 |
| `bound_motion_block` | 根据已满足独立依赖划分可执行运动块 | 原任务、目标身份、已满足依赖 → 复制问题与区间记录；请求绑定 |
| `requirement_consumption/verification_issues/remaining_obligations` | 解释要求的消费与剩余范围 | 原要求、查询进度、已执行身份 → 消费/缺项/剩余义务记录；绑定、报告与续规划 |

## 2. RGB-D 与已有实体模型

| API | 输入与前置数据 | 输出及实际消费者 |
|---|---|---|
| `build_scene_from_rgbd` | 必填 `robot_model`（同一原生机器人检查器）与 `fixed_geometry_names`（明确固定安装实体）；新建实体的 owner_frames、mask/深度/采集标定，或 observed_owners 中已有模型；可传已有 background/coverage | 同一 `OnlinePlanningScene`，并独立记录本次原实体 mask、原背景候选和实际积分背景；运行器及通用场景构造 |
| `save_observation_partitions` | 上述场景分区记录与输出目录 | 每视图的原实体文件、背景候选文件、实际积分文件及身份/路径/职责报告；后续建图复用前两者 |
| `integrate_entity/freeze_mapper` | 明确实体像素、原深度、K、T_entity_camera 和资源范围 | 原始样本、权重、支持构成的 `FrozenObservedField`；建图 |
| `FrozenObservedField.record/from_record/save/load` | 固定场或结构化样本记录 | 同一样本及坐标的场/记录；保存与复用 |
| `ObservedField.sample/cell_value` | 原场支持内的局部点或具体单元 | 值、有效性、梯度、数值误差；距离、过滤与接触 |
| `query_distance` | 两个原场、刚体变换、间距和有限许可 | 距离上下界、clear/margin_insufficient/intersection/unresolved及见证；优化与验收 |
| `pair_residual` | 原生实体—场或场—场、可微位姿和同值矩阵 | 几何分类及局部可微残差；场景分项 |
| `query_region_intersection` | 原场、位姿、有限半空间 | 受支持零表面 contact/missing_contact/unresolved；required接触 |
| `query_minimum_height` | 原场、位姿与声明平面 | 整体最低高度区间及见证；初始支撑 |
| `robot_self_filter.classify_depth` | 原深度、采集时FK放置的原生组件、明确shadow padding | SELF/AMBIGUOUS_SHADOW/ENVIRONMENT、原自由射线及来源；建图 |
| `ObservedSpace/query_coverage` | 原自由射线格、运动实体场及条件区域 | covered/unknown_witness/unresolved；场景按明确策略解释条件区域及未知 |
| `NativeSolid/NativePrimitive` | 原凸组件顶点或解析尺寸、所属连杆、固定变换和来源 | 原生机器人实体；共享检查器 |
| `native_field_distance.query_native_field` | Coal原实体、原任务场、两个位姿、间距、精度、点对许可 | 原集合距离区间及clear/margin_insufficient/intersection/unresolved；场景 |
| `RobotGeometryChecker.model_record` | 已消费的原模型 | URDF和原顶点/解析尺寸/局部变换的独立副本；输入绑定 |
| `DifferentiableRobotKinematics.poses` | 纯运动学配置和完整具名关节 | 可微世界连杆位姿；场景再合成原实体局部变换 |
| `RobotGeometryChecker.query` | 具名关节、基座标定、成对策略和所需帧 | 限位、独立FK、原生有符号距离/见证/法向及分离/穿透/数值边界；`RobotStateValidator` |
| `OnlinePlanningScene.bind_online_robot` | 同一原生检查器、纯运动学配置、验证器、请求检查 | 完整FK和对策略绑定；后端初始化 |
| `OnlinePlanningScene.components/evaluate` | 同时刻具名关节、工具位姿、进度、物理时间及 query_role | 分项/硬和与见证报告；IK、优化、验收 |
| `describe/to_execution_binding` | 场景已有输入及关系 | 描述与 `native_robot_observed_scene_v1` 快照；结果绑定 |
| `clone_for_rollout/with_release_history` | 同一原始场及已选搜索前缀 | 私有关系状态、共享不可变场；搜索与优化 |
| `estimated_workspace_allocation` | batch/horizon和实体数 | 共享场与逐状态临时存储估计；资源准入 |

## 3. 关系、任务接触与手指

| API | 输入与前置数据 | 输出及实际消费者 |
|---|---|---|
| `CollisionOwner.pose/predicted_pose` | 原位姿路径、固定关系、当前工具位姿及释放记录 | 原场在世界中的预测位姿；成对查询 |
| `ContactRule/OnlinePlanningScene._permission` | 源任务特征、有限目标区域、活动区间和间隙/法向条件 | 当前有限半空间许可及法向残差；同一原始场查询 |
| `bind_grasp_contact_rules/validate_grasp_contact_arrival` | 原名义目标、close事件和明确规则身份 | 条件绑定/一致性验证；任务及场景 |
| `build_material_contact_rules` | 原任务特征、robot_body和明确目标许可 | ContactRule记录；任务接触绑定 |
| `check_grasp_region_compatibility` | 源特征、当前工具/目标坐标和连续手指区间 | compatible/contradicted/unresolved及假设手指见证；抓取候选任务域 |
| `bind_holding_contacts/validate_holding_conditions` | robot_body材料身份、实体及原附着条件 | 指定实体对在声明区间内的内部许可；场景 |
| `PreparedInitialSupport.components/permission/record` | 原始作用场、支撑场、入口关系、平面及当前位姿 | 原单元许可、高度违反/未决和记录；场景 |
| `FixedFingerTargets.from_mapping/bind` | 具名手指目标、原事件和设备范围 | 时间命令；规划与执行 |
| `FingerTrajectory.build/sample/sample_torch` | 入口q/dq、命令、时长与设备限制 | 同一物理时钟的指令曲线及样本；设备与名义几何 |
| `FingerTrajectory.describe/from_mapping/validate_execution` | 曲线、原事件与设备声明 | 曲线记录、恢复或一致性检查；执行 |
| `bind_nominal_finger_geometry` | 手指指令、入口反馈、声明关系 | 名义手指几何及有效区间；场景 |
| `PrismaticGripper.bind_clock/state` | 具名关节、控制能力、事件和当前时刻 | 手指坐标、条件未知及关节残差；完整机器人FK |

## 4. 数值求解与诊断

本组接口的数学含义和控制流程见[求解组件](04-solver-and-search.md)。原生动作是固定三次基的自由位置系数，正式输出是具名关节多项式。

`PlanningBudget.interpolation_dt` 参与候选的时间采样，`validation_joint_step` 参与 OMPL 连接检查。当前 `validation_spatial_step_m` 的代码行为是保存配置值并校验其为正数；生产候选验收的采样规则见[验收组件](05-acceptance-and-execution.md)。关节列序由各对象的 `joint_names` 明确给出，原生与保存边界按名称转换。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `PlanningBudget` | 规定本次维数、工作和资源范围 | 显式预算或默认值 → 不可变预算；后端与账本 |
| `MotionBackend.bind_request/plan/close` | 管理复用求解器、请求与资源 | 任务、场景、预算、可选续规划 → 后端绑定/规划结果/释放；调用者 |
| `plan_trajectory` | 组织参考验收、优化、有限恢复和交付 | 任务/场景/机器人/验证器、预算、参考、续规划与取消接口 → `PlanningResult` |
| `CubicBasis.weights/weights_at_knot_times` | 查询公共三次基导数权重 | 归一化时间或结点坐标、阶数 → 动作/入口速度权重；CPU 域与 Torch 转移 |
| `CubicTrajectoryStep.bind_clock/update_dt/update_batch_size` | 绑定固定物理步长与缓冲布局 | 请求时钟、batch/horizon → 更新局部状态；原生初始化 |
| `CubicTrajectoryStep.sample/forward` | 将系数变为同一曲线的原生状态 | 动作、入口、索引、输出容量/时钟 → `q,dq,ddq,jerk,dt`；rollout/导出 |
| `JointPositionDomain.feasible/validate/limit_step/limit_step_tensor` | 约束 IK 候选与线搜索点的关节域 | 关节状态、方向和倍率 → 可行掩码、检查或缩短方向；原生优化 |
| `CubicJointDomain.evaluate/coefficients/extrema` | 计算完整样条与连续位置极值 | 系数、进度/阶数 → 状态、系数或极值；域判断 |
| `CubicJointDomain.initialize/fit_reference/recovery_step` | 构造物理域内的初值与更新方向 | 提议系数、参考/速度界或当前梯度 → 动作系数/方向；参考与数值恢复 |
| `CubicAffineMap.query/evaluate/extrema_times/velocity_extrema` | CPU 上查询同一基与导数极值 | 系数、进度、关节和阶数 → 仿射行、状态或极值时刻；QP |
| `fit_cubic_reference/recover_position_step` | 求受位置/速度域约束的拟合或更新 | 物理域、仿射基、目标或梯度/步长 → 合法系数或步方向；域方法 |
| `bind_solver_components` | 将物理域扩展装入一个原生实例 | cuRobo solver、domain、graph 设置 → 已绑定实例；后端初始化 |
| `PhysicalGradientCore.optimize/reinitialize/bind_domain` | 管理原生迭代、候选保留与请求域 | 合法种子、重置掩码或新域 → 优化动作/更新状态；IK 与轨迹 solver |
| `PhysicalGradientCore.solve_diagnostics/extra_workspace_bytes` | 导出本次数值恢复与存储量 | 当前优化核心 → 报告/字节；请求诊断/准入 |
| `FunctionalAugmentedLagrangian.bind/reset/terms/update` | 分种子保存并更新功能乘子 | 条件键、种子布局、有符号残差 → 优化项或更新掩码；代价与块间更新 |
| `FunctionalAugmentedLagrangian.snapshot/report` | 导出功能约束数值状态 | 已完成求解状态 → 乘子/罚参数/更新记录；诊断 |
| `rank_ik_candidates/select_ik_chains` | 排序并连接有限 IK 状态 | 原生结果、任务/场景残差、入口与阶段 → 排序记录/关节链；参考种子 |
| `_CuroboPlanner.reference_seeds/ompl_seed` | 生成固定基参考或搜索种子 | 原参考/可选外部曲线、场景、当前候选 → 系数及来源/搜索报告；请求主流程 |
| `_CuroboPlanner.evaluate_reference/optimize` | 运行完整参考 metrics 或原生优化 | 动作种子 → 原生结果；候选选择 |
| `_CuroboPlanner.select_candidate/native_diagnostics` | 按原始约束选择并解释已有候选 | `select_candidate` 返回 `(candidate, trajectory, report)`；`native_diagnostics` 返回指定种子的指标报告；主流程 |
| `_CuroboPlanner.evaluate_candidate/export` | 验收原多项式并输出完整曲线 | `evaluate_candidate` 返回 `(trajectory, attempt, points)`；`export` 复用同身份报告后返回 `(trajectory, points)`；请求入口随后生成执行绑定 |
| `AttemptWitnesses.add_native/add_validation/set_current` | 收集本次实际失败时刻并记录当前候选 | 完成的 native/export metrics → 有限参数集合；诊断采样 |
| `AttemptWitnesses.select_parameters` | 在固定容量内合并必须点与可选见证 | 固定/普通参数和最大点数 → 诊断查询参数；报告生成 |
| `capture_component_failures/scene_component_failures` | 保存具体失败状态和来源 | 分项、时刻、进度、关节/工具状态和许可 → 可解释失败记录；报告 |
| `bind_trace_to_native_seed` | 将诊断跟踪与实际种子身份相连 | 跟踪、原生状态、种子索引、关节名 → 已绑定跟踪；诊断输出 |

## 5. 验收与设备执行

这些接口的控制职责见[执行组件](05-acceptance-and-execution.md)。机器人反馈和独立效果观察有各自来源；仿真几何结果进入事后记录。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `JointTrajectory.sample` | 查询保存的原多项式 | 有限原时间范围内的时刻 → `q,dq,ddq,s`；验收与控制 |
| `cubic_position_report/cubic_derivative_report` | 检查连续位置或加速度极值 | 原 Hermite 曲线、具名边界、可选前缀 → 最大越界及见证；候选/执行 preflight |
| `RobotGeometryChecker` | 读取关节树及已知原生实体 | URDF、基座与 NativeSolid/NativePrimitive → Pinocchio FK及Coal实体对查询；RobotStateValidator |
| `RobotStateValidator.check_planning_self` | 使用名义完整关节状态检查原生机器人模型 | 机械臂、手指名义状态及时刻 → 完整具名状态、限位及实体对检查记录；规划绑定 |
| `RobotStateValidator.check_state/preflight` | 检查实际机器人与执行入口 | 完整具名反馈、原曲线、执行范围/限制 → 检查报告；执行器 |
| `curve_snapshot/bind_candidate/check_result` | 保存候选、绑定完成证据并核对一致性 | 原曲线/任务/场景/报告或正式结果 → 快照/绑定/绑定副本；执行交接 |
| `rebind_execution_result` | 明确设备控制条件与新绑定身份 | 成功结果、控制策略、评价目标身份、指令与限制 → 新绑定结果；设备适配器 |
| `relax_pose_goal_tolerances` | 继承原证据到显式扩大的任务允许集合 | 未执行成功结果、目标身份与新容差、来源 → 新任务绑定；明确任务修订 |
| `ExecutionLimits.from_mapping/describe/bind` | 恢复并核对完整具名设备限制 | 来源、关节分区、跟踪/停止/手指范围、观察策略 → 限制对象/描述/事件绑定；执行 |
| `TrajectoryExecutor.execute` | 按原时钟执行可执行前缀并记录 | 正式结果、初始反馈、限制/手指目标、观察与记录接口 → 执行报告及续规划数据；设备入口 |
| `RobosuiteExecutionBackend.apply_control/step/proprioception` | 下发机器人指令、推进设备、读取机器人反馈 | `q,dq,ddq`、手指位置 → 控制/物理步/具名反馈；执行器 |
| `reference_tool_pose/abort_hold/last_step_accounting` | 提供指令 FK、有限停止及物理步记账 | 指令或实际反馈/停止阈值 → 工具位姿、停止结果、步数；执行器 |
| `pose_goal_schedule` | 为事后目标评价确定原时刻 | 曲线、目标进度、设备步长 → 名义时刻与应观测步；评价器 |
| `continuation_execution_inputs` | 将执行历史连接到后续控制数据 | continuation、原手指目标/限制、明确 rebind → 续执行输入；调用者 |

## 6. 保存、空间查询、资源与实验适配

`collision_records.encode_array/decode_array` 保存 dtype、形状和压缩数组；消费者据其真实尺寸检查资源，再恢复同一数组。`slice_layer_selection` 从分层证据中选取指定样本。`tensor_cpu_snapshot` 将完成的张量复制到 CPU 报告，`runtime_profiling` 记录请求阶段的时间及资源。这些数据接口服务可复现记录。

`implicit_distance` 对同一原场完成距离上下界与分类，`observed_collision` 为这些结果提供局部可微方向与任务接触查询，`observed_coverage` 解释实体是否处于原自由射线或明确条件区域。`observed_robot` 将完整具名关节映射为机器人连杆位姿。`native_field_distance` 连接已知本体几何与原任务场，双方保留各自的表示及共同的世界坐标。

`resource_admission.require_host_memory/require_device_memory` 输入新增存储需求、当前物理占用与明确上限，返回准入记录或类型化 `ResourceLimitError`。`optimizer_device_checks.DeviceChecks` 在设备计算期间保留有限性和域错误，`require_tensor` 在当前检查作用域内消费布尔张量。预算、诊断和数值错误共享同一请求生命周期。

实验运行器位于 `tests/planning`。`run_recorded_skill.run_prepared` 接收已编译问题，读取已有实体，再为本次需要建图的实体读取 mask、深度与采集时反馈。`robot_validation_fixture` 构造已知机器人验证器和图像过滤组件；`observed_robot_fixture` 恢复已有任务实体。`recorded_task_planning.plan_and_record` 调用后端并保存曲线、结果和 `native_robot_observed_scene_v1` 来源快照。`load_saved_trajectory` 恢复正式执行输入；该操作不重建可查询场景。具体命令与数据字段见[记录入口说明](../../tests/planning/README.md)。

跨阶段读写时，调用者需要同时保留几何记录、其坐标变换、对象身份、任务区域、原事件与机器人入口。各接口按上述结构解释数据；原生模型恢复与本次回放见[阶段对照](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。
