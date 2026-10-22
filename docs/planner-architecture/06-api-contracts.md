# API 契约与调用关系

代码基线：`a2b3f686b866aea18af47c302bafb62e44212e22`（文档引用此实现提交；后续文档提交保持代码内容一致）。本目录按运行阶段组织接口的作用、必要输入、输出和调用者。完整签名、dataclass 字段、返回表达式和内部调用符号见[逐符号实现索引](07-implementation-index.md)。同一对象的方法共享其构造时绑定的数据；表中省略的 `self` 指向该已构造实例。

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
| `PoseEventRequirement.error_components/errors/errors_numpy` | 计算材料目标位置与完整旋转或倾斜/yaw | 工具位姿、固定材料关系、目标及显式目标局部轴 → 分量字典；`errors` 仅返回位置和旋转；准备、诊断、规划与事后评价 |
| `PoseEventRequirement.component_tolerances/evaluate` | 选择并验收声明分量 | 位置/旋转容差及可选 yaw 容差 → 受约束分量与超差正部和；π 时省略 yaw |
| `program._material_pose_requirement` | 在准备阶段绑定同一材料姿态集合 | 当前功能帧、目标、位置/旋转/可选 yaw 容差 → `PoseEventRequirement`；入口与功能点检查 |
| `quaternion_axis_angle` | 有限旋转的主值绕轴分解 | 同一坐标系的相对四元数及单位轴 → `[-π,π]` 内绕轴角；姿态事件与机构误差 |
| `SupportRequirement.raw_inequalities` | 计算有限支撑四条件 | 候选工具位姿、物体点/材料、目标平面/边界 → gap/body/footprint/normal 原始有符号值；优化、验收选择与分析 |
| `SupportRequirement.normalized_inequalities` | 缩放完整支持条件，保留各项零集合 | 四类原始条件与数值尺度 → 四项有符号归一化条件；硬分量供功能 AL，补集供 `quality` 阶段任务偏好 |
| `SupportRequirement.hard_components/evaluate` | 声明支持硬分量并计算相应残差 | 原几何、数值容差、四个名称的显式子集（默认 body/footprint）→ 硬分量正部和；任务残差、事件 IK、搜索与验收 |
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
| `PlanningProblem.residuals` | 计算同一声明的硬条件 | 位姿、进度、具名关节/速度、时间、目标身份与 point/trajectory 范围 → 非负分项；所有消费者使用同一硬集合，完整路径另有历史和速度数据 |
| `PlanningProblem.evaluate` | 汇总已声明硬条件 | 与 residuals 相同的候选状态及任务输入 → 分项之和；直接任务评价 |
| `functional_constraint_keys/functional_inequalities` | 给硬功能条件确定键与有符号尺度 | 确切事件样本、位姿/时间及支持 hard_components → 条件键、归一化不等式及事件掩码；AL 与同次硬证据 |
| `optimizer_preferences` | 计算支持软分量、过程等式/方向及可选目标中心偏好 | 原几何和阈值、候选状态、时间及优化阶段 → 偏好张量；代价管理器 |
| `measurements` | 保存可解释的任务原量 | 候选位姿与时间 → 测量字典；诊断 |
| `anchors/stop_progress` | 枚举确切任务时刻和停止要求 | 编译后任务 → 进度集合；固定时钟与查询布局 |
| `projection_requirements/project_residual` | 选择当前阶段的过程投影条件 | 进度及候选状态 → 要求集合或残差；OMPL 投影 |
| `goal_requirements_at/goal_residual` | 选择并计算原目标身份对应允许集合 | 进度、目标身份、候选及可选 normalized → 目标集合/超差正部；事件 IK 的归一化目标与搜索终点 |
| `bound_motion_block` | 根据已满足独立依赖划分可执行运动块 | 原任务、目标身份、已满足依赖 → 复制问题与区间记录；请求绑定 |
| `requirement_consumption/verification_issues/remaining_obligations` | 解释要求的消费与剩余范围 | 原要求、查询进度、已执行身份 → 消费/缺项/剩余义务记录；绑定、报告与续规划 |

## 2. RGB-D 与已有实体模型

| API | 输入与前置数据 | 输出及实际消费者 |
|---|---|---|
| `build_scene_from_rgbd` | 必填 `robot_model`（同一原生机器人检查器）、`fixed_geometry_names`（固定安装实体）与 `tool_geometry_names`（完整工具成员）；新建实体的 owner_frames、mask/深度/采集标定，或 observed_owners 中已有模型；可传已有 background/coverage | 同一 `OnlinePlanningScene`，并独立记录本次原实体 mask、原背景候选和实际积分背景；运行器及通用场景构造 |
| `save_observation_partitions` | 上述场景分区记录与输出目录 | 每视图的原实体文件、背景候选文件、实际积分文件及身份/路径/职责报告；后续建图复用前两者 |
| `integrate_entity/freeze_mapper` | 明确实体像素、原深度、K、T_entity_camera 和资源范围 | 原始样本、权重、支持构成的 `FrozenObservedField`；建图 |
| `FrozenObservedField.record/from_record/save/load` | 固定场或结构化样本记录 | 同一样本及坐标的场/记录；保存与复用 |
| `ObservedField.sample/cell_value` | 原场支持内的局部点或具体单元 | 值、有效性、梯度、数值误差；距离、过滤与接触 |
| `ObservedField._initial_occupied_points/occupied_points` | 不可变原始叶节点或当前动态子节点的范围与原场样本 | 原叶节点复用只读局部点集，动态节点计算本域点集；占据距离查询与存储计数 |
| `ObservedField.local_zero` | 查询点及原场中完整支持、非恒定且包含零的单元索引 | 最多八个局部单元中的零点、单位梯度方向及距离，或 None；优化块模型捕获 |
| `query_distance` | 两个原场、刚体变换、数值精度和有限许可 | clear/intersection/unresolved、当前未许可域的 lower_m、完整实体的 full_entity_lower_m 及见证；静态分类与区间几何分别消费 |
| `pair_residual` | 原生实体—场或场—场、可微位姿、同值矩阵、精度、许可及可选共享零值/原生界 | 原查询证据与分类残差（intersection 为1，其余0）；硬分项，分类梯度为零 |
| `pair_local_model` | 原实体、实际基准位姿及已完成交叠证据 | 原材料点和原场零面形成的 LocalPairModel，或 None；只在优化块起点调用 |
| `LocalPairModel.value` | 固定局部点、零点、单位法向、场间距及当前两个实体位姿 | 有量纲 softplus 标量，含点与法向刚体导数；固定块目标 |
| `LocalContactModel.value` | 固定目标零点及本次相同坐标系中的半空间系数 | 正部半空间超差范数；required contact 的连续优化目标 |
| `LocalGeometryModels.begin_capture/state/finish_capture/abort_capture` | seed 数、particle 数、物理时间和关系/许可进度 | 冻结或替换模型、按 seed 的变更掩码；初始和常规 rollout 共享，取消保留前一完整记录 |
| `LocalGeometryModels.repeated_base_evaluation` | 明确按 seed 重复的实际基准状态布局 | 在上下文内共享数值几何查询，逐粒子独立计算可微标量；块边界的梯度重算 |
| `query_region_intersection` | 原场、位姿、有限半空间 | 受支持零表面 contact/missing_contact/unresolved；required接触 |
| `query_minimum_height` | 原场、位姿与声明平面 | 整体最低高度区间及见证；初始支撑 |
| `robot_self_filter.classify_depth` | 原深度、采集时FK放置的原生组件、明确shadow padding | SELF/AMBIGUOUS_SHADOW/ENVIRONMENT、原自由射线及来源；建图 |
| `ObservedSpace/query_coverage` | 原自由射线格、运动实体场及条件区域 | covered/unknown_witness/unresolved；场景按明确策略解释条件区域及未知 |
| `NativeSolid/NativePrimitive` | 原凸组件顶点或解析尺寸、所属连杆、固定变换和来源 | 原生机器人实体；共享检查器 |
| `native_field_distance.query_native_field` | Coal 原实体、原任务场、两个位姿、精度、许可及可选同实体 local/world bounds | 当前未许可域的状态与 lower_m、完整实体的 full_entity_lower_m、可行点区间；场景与区间检查消费各自域的证据 |
| `native_pair_lower_bound` | 两个官方机器人原实体、当前变换及候选分离轴 | 完整实体支持投影的分离下界；区间几何证据，原生容差不作为误差界 |
| `RobotGeometryChecker.model_record` | 已消费的原模型 | URDF和原顶点/解析尺寸/局部变换的独立副本；输入绑定 |
| `DifferentiableRobotKinematics.poses` | 纯运动学配置和完整具名关节 | 可微世界连杆位姿；场景再合成原实体局部变换 |
| `RobotGeometryChecker.query` | 具名关节、基座标定、成对策略和所需帧 | 限位、独立FK、原生有符号距离/见证/法向及分离/穿透/数值边界；`RobotStateValidator` |
| `OnlinePlanningScene.bind_online_robot` | 同一原生检查器、纯运动学配置、验证器、请求检查 | 完整FK和对策略绑定；后端初始化 |
| `OnlinePlanningScene.components/evaluate` | 同时刻关节、工具位姿、进度、时间、query_role；optimization 另需 local_models | components 返回实际硬分项，另输出连续目标；evaluate 按角色汇总硬项或目标，原证据独立保留；IK、优化、验收共用非穿透规则 |
| `OnlinePlanningScene.components(..., geometry_evidence_out=...)` | 同一实际场景查询及显式证据容器 | 全量逐实体对的状态、当前距离下界、许可与内部关系；geometry_entity_key 以 robot/、owner/ 或 background 区分实体类别，供区间检查 |
| `describe/to_execution_binding` | 场景已有输入、工具成员及实体附着 | `collision_rule='nonpenetration'`、`tool_geometry_names` 描述与 `native_robot_observed_scene_v1` 快照；结果绑定 |
| `clone_for_rollout/with_release_history` | 同一原始场及已选搜索前缀，或固定候选的释放锚点 FK | 私有关系状态、共享不可变场；搜索前缀与候选区间几何复用预测释放位姿 |
| `estimated_workspace_allocation` | batch/horizon和实体数 | 共享场与逐状态临时存储估计；资源准入 |

## 3. 关系、任务接触与手指

| API | 输入与前置数据 | 输出及实际消费者 |
|---|---|---|
| `Attachment.active` | 原生效区间、可选释放条件与实际进度 | 有释放时左闭右开、无释放时自起点持续的成员布尔值；实体位姿和组合体内部配对 |
| `CollisionOwner.pose/predicted_pose` | 原位姿路径、固定关系、当前工具位姿及释放记录 | 原场在世界中的预测位姿；成对查询 |
| `ContactRule/OnlinePlanningScene._permission` | 源任务特征、有限目标区域、活动区间和间隙/法向条件 | 当前有限半空间许可及法向残差；同一原始场查询 |
| `bind_grasp_contact_rules/validate_grasp_contact_arrival` | 原名义目标、close事件和明确规则身份 | 条件绑定/一致性验证；任务及场景 |
| `build_material_contact_rules` | 原任务特征、robot_body和明确目标许可 | ContactRule记录；任务接触绑定 |
| `check_grasp_region_compatibility` | 源特征、当前工具/目标坐标和连续手指区间 | compatible/contradicted/unresolved及假设手指见证；抓取候选任务域 |
| `PreparedInitialSupport.components/permission/record` | 原始作用场、支撑场、入口关系、平面及当前位姿 | 原单元许可、高度违反/未决和记录；场景 |
| `FixedFingerTargets.from_mapping/bind` | 具名手指目标、原事件和设备范围 | 时间命令；规划与执行 |
| `FingerTrajectory.build/sample/sample_torch` | 入口q/dq、命令、时长与设备限制 | 同一物理时钟的指令曲线及样本；设备与名义几何 |
| `FingerTrajectory.describe/from_mapping/validate_execution` | 曲线、原事件与设备声明 | 曲线记录、恢复或一致性检查；执行 |
| `bind_nominal_finger_geometry` | 手指指令、入口反馈、声明关系 | 名义手指几何及有效区间；场景 |
| `PrismaticGripper.bind_clock/state` | 具名关节、控制能力、事件和当前时刻 | 手指坐标、条件未知及关节残差；完整机器人FK |

## 4. 数值求解与诊断

本组接口的数学含义和控制流程见[求解组件](04-solver-and-search.md)。原生动作是固定三次基的自由位置系数，正式输出是具名关节多项式。

`PlanningBudget.interpolation_dt` 是请求的最大采样间隔；`_derived_spline_grid` 选择精确实际时钟，容量检查消费其真实样本数，绑定后的预算保存实际间隔。`validation_joint_step` 参与 OMPL 连接检查。候选验收的具体时间与空间范围见[验收组件](05-acceptance-and-execution.md)。关节列序由各对象的 `joint_names` 明确给出，原生与保存边界按名称转换。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `PlanningBudget` | 规定本次维数、工作和资源范围 | 显式预算或默认值 → 不可变预算；后端与账本 |
| `MotionBackend.bind_request/plan/close` | 管理复用求解器、请求与资源 | 任务、场景、预算、可选续规划 → 后端绑定/规划结果/释放；调用者 |
| `plan_trajectory` | 组织参考完整资格、持续改善、有限恢复和最佳候选交付 | 任务/场景/机器人/验证器、预算、参考、续规划与取消接口 → `PlanningResult`，独立保存候选绑定与最终累计工作 |
| `CubicBasis.weights/weights_at_knot_times` | 查询公共三次基导数权重 | 归一化时间或结点坐标、阶数 → 动作/入口速度权重；CPU 域与 Torch 转移 |
| `CubicTrajectoryStep.bind_clock/update_dt/update_batch_size` | 绑定固定物理步长与缓冲布局 | 请求时钟、batch/horizon → 更新局部状态；原生初始化 |
| `CubicTrajectoryStep.sample/forward` | 将系数变为同一曲线的原生状态 | 动作、入口、索引、输出容量/时钟 → `q,dq,ddq,jerk,dt`；rollout/导出 |
| `JointPositionDomain.feasible/validate/limit_step/limit_step_tensor` | 约束 IK 候选与线搜索点的关节域 | 关节状态、方向和倍率 → 可行掩码、检查或缩短方向；原生优化 |
| `CubicJointDomain.evaluate/coefficients/extrema` | 计算完整样条与连续位置极值 | 系数、进度/阶数 → 状态、系数或极值；域判断 |
| `CubicJointDomain.initialize/fit_reference/recovery_step` | 构造物理域内初值、事件插值曲线及更新方向 | 提议系数；软参考/速度界与可选 event_progress/event_positions；或当前梯度 → 系数/方向和报告；等式仅用于初始化 |
| `CubicAffineMap.query/evaluate/extrema_times/velocity_extrema` | CPU 上查询同一基与导数极值 | 系数、进度、关节和阶数 → 仿射行、状态或极值时刻；QP |
| `fit_cubic_reference/_fit_anchor` | 以同一事件等式完成拟合和原生 dtype 恢复 | 原曲线、参考、原位置/速度界及事件线性行 → 系数/锚点、连续界与实际事件误差报告；fit_reference |
| `recover_position_step` | 在当前完整曲线位置域中构造下降方向 | 物理域、仿射基、梯度与原生步长范围 → 步方向及可行性报告；主优化恢复 |
| `bind_solver_components` | 将物理域扩展装入一个原生实例 | cuRobo solver、domain、graph 设置 → 已绑定实例；后端初始化 |
| `PhysicalGradientCore.optimize/reinitialize/bind_domain` | 管理原生迭代、候选保留与请求域 | 合法种子、重置掩码或新域 → 优化动作/更新状态；IK 与轨迹 solver |
| `PhysicalGradientCore._refresh_local_geometry` | 在当前实际基准动作上同时捕获局部几何与原硬证据 | seed 动作、常规 rollout、共享 LocalGeometryModels → `(changed, trajectories)`；块边界读取同次残差和固定目标，初始与常规梯度入口共用模型 |
| `PhysicalGradientCore._retain_sampled_candidate` | 在每个种子的固定槽中保留采样硬条件合格候选 | 当前动作、同次 feasible 与固定排序目标 → 按目标较小值更新动作；最终曲线验收继续决定交付 |
| `PhysicalGradientCore.solve_diagnostics/extra_workspace_bytes` | 导出本次数值恢复与存储量 | 当前优化核心 → 报告/字节；请求诊断/准入 |
| `ProcessCostManager.compute_costs` | 按配置分别计算普通参考成本或任务/场景约束 | reference_objective 消费普通时钟并返回加权原生关节项与 continuous_reference；trajectory_optimizer/metrics 消费有效列和确切锚点，输出原硬证据与求解项 |
| `ProcessCostManager.optimizer_evidence` | 以 AL/场景张量身份核对同次原始硬证据，合并其余原生约束 | trajectories → `(feasible, signed, fixed_objective)`；固定目标只汇总普通成本，优化核心跨块保留采样合格候选 |
| `physical_time_mean_weights` | 实际物理时间 [A,H] 与各行有效前缀长度 [A] | 归一化梯形积分权重，padding 为零；参考与原生普通成本共用 |
| `NativeReferenceCost.bind/row_indices/time_weights/forward` | 完整问题、原配置位姿权重、候选时长/进度映射、普通网格与已有原生工具 FK | 预采样目标、显式行映射、积分权重及 [B,H] 连续参考成本；普通优化与固定候选评分 |
| `NativeRigidToolTransform.apply` | 原生连杆位姿与固定 TCP 平移/旋转标定 | TCP 位姿及符合原生 FK 反向协议的导数；NativeReferenceCost |
| `freeze_cspace_time_weights` | 原普通 STATE 成本配置的副本、本请求 native_dt | 一次固化时间系数并启用外部梯度权重的配置；请求绑定时构造关节成本 |
| `FixedTimeStateCSpaceCost.setup_batch_tensors/forward` | 上述配置、固定系数时间尺度与真实物理关节状态 | 原生 STATE 成本；仅成本用固定 dt 视图，轨迹与积分时钟保持各自物理含义 |
| `ProcessCostManagerCfg.bind_ik_goals` | 为一次 IK 绑定同刻原目标身份 | pose/coordinate/support 的 exact-progress 身份集合，或空集合 → 事件允许集合/普通参考计算上下文；事件候选生成 |
| `FunctionalAugmentedLagrangian.bind/reset/terms/update` | 分种子保存并更新功能乘子 | 条件键、种子布局、有符号残差 → 优化项或更新掩码；代价与块间更新 |
| `FunctionalAugmentedLagrangian.snapshot/report` | 导出功能约束数值状态 | 已完成求解状态 → 乘子/罚参数/更新记录；诊断 |
| `rank_ik_candidates/select_ik_chains` | 排序并连接参考或事件 IK 状态 | 结果、任务/场景报告、入口、实际时间、有符号速度界、query_kind 及可选 reference_positions → 有限关节链；参考一致性在短名单截断前参与可加排名 |
| `_CuroboPlanner._build_goal_ik_candidates/_fit_event_seeds` | 求同刻连续任务允许集合，为每个参考种子构造并比较事件等式初值 | 原要求、时钟、参考种子、原 IK 预算 → 原事件池、带 reference_seed_index 的有限短名单、实际 QP 结果及被选 fit；主求解消费原任务允许集合 |
| `_CuroboPlanner.reference_seeds/ompl_seed` | 生成固定基参考或搜索种子 | 原参考/可选外部曲线、场景、当前候选 → 系数及来源/搜索报告；请求主流程 |
| `_CuroboPlanner._native_request` | 将类型化任务要求与完整路径普通成本绑定为统一 constrained 请求 | 具名种子布局 → 原生 SolveState 与空末端位姿目标；参考验收和主求解 |
| `_CuroboPlanner.evaluate_reference/optimize` | 运行完整参考 metrics 或原生优化 | 动作种子 → 原生结果；候选选择 |
| `_CuroboPlanner._candidate_objective` | 在请求固定归一化网格上评价一条候选的普通成本 | 原多项式、自身时长与进度映射 → fixed_objective/objective_components；跨种子和跨求解候选排序 |
| `_CuroboPlanner.select_candidate/native_diagnostics` | 按原硬指标、固定评分和完整区间验收选择候选 | `select_candidate` 返回 `(candidate, trajectory, report)`；`native_diagnostics` 返回指定种子的硬指标和各实际网格 ordinary_costs；统一候选排名另读 fixed_objective |
| `_CuroboPlanner.evaluate_candidate/export` | 验收原多项式并输出完整曲线 | `evaluate_candidate` 返回 `(trajectory, attempt, points)`；`export` 复用同身份报告后返回 `(trajectory, points)`；请求入口随后生成执行绑定 |
| `SceneMotionBounds.from_scene/partition_times/interval` | 完整机器人树、原实体局部范围、声明关系、工具标定及原曲线 | 固定连续模型的时间分区、按 geometry_entity_key 保存的逐实体点速度界和缺失运动状态；区间验收 |
| `check_interval_geometry` | 原曲线、运动界、指定时刻的几何查询回调、剩余点数及最大位移尺度 | 已证明区间、交叠见证、未决区间、实际查询数与时刻；候选报告 |
| `_CuroboPlanner._candidate_geometry` | 通过任务/关节检查的曲线、同一场景及剩余查询容量 | 首次查询绑定该候选释放锚点，后续新增时刻复用；返回区间证明/未决与原实际硬残差；候选选择前验收 |
| `plan_trajectory.qualify_candidate/qualify_native/deliver_incumbent` | 完整合格候选、同身份验收、场景与固定评分 | 在预算内构造独立执行绑定及曲线记录，按较低评分替换，最终恢复选中记录；请求交付与改善预算停止 |
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

`implicit_distance` 对同一原场完成非穿透分类并保留有范围定义的距离界。`observed_collision` 返回原分类，并在块起点从实际证据构造局部模型；`local_geometry_model` 保存固定记录并计算其连续标量。`observed_coverage` 解释实体是否处于原自由射线或明确条件区域。`observed_robot` 将完整具名关节映射为机器人连杆位姿。`native_field_distance` 连接固定机器人官方实体与原任务场，双方保留各自表示及共同世界坐标。场配对未决保存在查询证据中，原生自身数值边界、任务接触、初始支撑和覆盖各有独立硬条件。

`resource_admission.require_host_memory/require_device_memory` 输入新增存储需求、当前物理占用与明确上限，返回准入记录或类型化 `ResourceLimitError`。`optimizer_device_checks.DeviceChecks` 在设备计算期间保留有限性和域错误，`require_tensor` 在当前检查作用域内消费布尔张量。预算、诊断和数值错误共享同一请求生命周期。

实验运行器位于 `tests/planning`。`run_recorded_skill.run_prepared` 接收已编译问题，读取已有实体，再为本次需要建图的实体读取 mask、深度与采集时反馈。`robot_validation_fixture` 构造已知机器人验证器和图像过滤组件；其 `tool_geometry_names(fixture)` 从声明手指机构的公共父子树返回工具成员元组，写入运行报告并传给场景。`observed_robot_fixture` 恢复已有任务实体。`load_contact_bindings` 读取必填 `contacts/source` 和可选 `grasp_activations/initial_support_relations`，分别提供原有限任务接触与初始支撑记录。`recorded_task_planning.plan_and_record` 调用后端并保存曲线、结果和 `native_robot_observed_scene_v1` 来源快照。`load_saved_trajectory` 恢复正式执行输入；该操作不重建可查询场景。具体命令与数据字段见[记录入口说明](../../tests/planning/README.md)。

跨阶段读写时，调用者需要同时保留几何记录、其坐标变换、对象身份、任务区域、原事件与机器人入口。各接口按上述结构解释数据；原生模型恢复与本次回放见[阶段对照](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。
