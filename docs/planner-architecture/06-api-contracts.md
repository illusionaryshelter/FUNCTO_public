# API 契约与调用关系

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。本目录按运行阶段组织接口的作用、必要输入、输出和调用者。完整签名、dataclass 字段、返回表达式和内部调用符号见[逐符号实现索引](07-implementation-index.md)。同一对象的方法共享其构造时绑定的数据；表中省略的 `self` 指向该已构造实例。

批量张量记为 `B` 个候选、`H` 个时间点、`J` 个关节和 `N` 个球。常用布局为关节 `[B,H,J]`、位姿位置 `[B,H,3]`、四元数 `[B,H,4]`、球 `[B,H,N,4]`，球最后一维为中心与半径。NumPy 保存数组通常移除候选维。位姿四元数采用 `wxyz`，齐次变换将右侧坐标中的点映到左侧坐标。物理时间以秒计，平移与球半径以米计，角度以弧度计。

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
| `transform_robot_spheres/transform_tool_velocity` | 转换几何球和工具速度 | 机器人球或线/角速度及连杆姿态 → 任务坐标中的球/速度；场景与方向要求 |
| `residuals/evaluate` | 对同一候选状态计算全部已编译硬条件 | 位姿、进度、具名关节/速度、物理时间、确切目标身份 → 分项或聚合残差；原生 metrics |
| `functional_constraint_keys/functional_inequalities` | 给功能条件确定键与有符号尺度 | 确切事件样本、位姿/时间 → 条件键和归一化不等式；AL |
| `optimizer_preferences` | 计算过程/方向等优化偏好 | 候选状态、时间及优化阶段 → 偏好张量；代价管理器 |
| `measurements` | 保存可解释的任务原量 | 候选位姿与时间 → 测量字典；诊断 |
| `anchors/stop_progress` | 枚举确切任务时刻和停止要求 | 编译后任务 → 进度集合；固定时钟与查询布局 |
| `projection_requirements/project_residual` | 选择当前阶段的过程投影条件 | 进度及候选状态 → 要求集合或残差；OMPL 投影 |
| `goal_requirements_at/goal_residual` | 选择并计算原目标身份对应条件 | 进度、目标身份、候选 → 目标集合/残差；IK 排序与搜索终点 |
| `bound_motion_block` | 根据已满足独立依赖划分可执行运动块 | 原任务、目标身份、已满足依赖 → 复制问题与区间记录；请求绑定 |
| `requirement_consumption/verification_issues/remaining_obligations` | 解释要求的消费与剩余范围 | 原要求、查询进度、已执行身份 → 消费/缺项/剩余义务记录；绑定、报告与续规划 |

## 2. RGB-D、TSDF 与场景构造

本阶段消费经过机器人自遮挡处理的 RGB-D 视图和明确对象分区。生产场景从 TSDF 源组件建立固定候选距离场；审计入口另存其细粒度来源和分层查询记录，见[观测组件](02-observation-and-collision.md)。

生产 `views[observation_id]` 的必要字段为 `image:uint8[H,W,3]`、米制 `depth:[H,W]`、零斜切 `K:[3,3]`、`T_output_camera:[4,4]`、`reference_frame`、各 owner 的布尔 `owner_masks`，以及三态 `self_filter`。完整像素归属和原生 Mapper 主点加 `0.5` 的坐标转换见该组件第一节。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `robot_self_filter.classify_depth` | 按已知机器人实体及实际 FK 解释深度射线 | 深度、K、相机位姿、机器人实体/姿态、误差带 → ENVIRONMENT/SELF/AMBIGUOUS_SHADOW 标签及自由射线；场景建图 |
| `validate_self_filter` | 核对标签和自由射线与深度尺寸 | 自过滤记录、深度 → 规范记录；观测构造 |
| `scene_task_owner_ids` | 识别任务实际引用的观测对象 | 场景任务关系 → owner 身份集合；分辨率选择 |
| `observation_resolutions` | 为任务对象与背景确定观测分辨率 | owner 集合、任务 owner、环境体素大小 → 各分区分辨率；Mapper 构造 |
| `build_scene_from_rgbd` | 融合独立观测分区并冻结生产碰撞输入 | 视图、owner 坐标、Mapper 配置、深度误差、任务身份、手部/接触/附着/完整模型 → `OnlinePlanningScene`，报告位于其 `observation_report`；记录入口 |
| `build_audit_scene_from_rgbd` | 构建来源审计使用的分层场景 | 对应审计观测与模型记录、细层请求 → `PlanningScene`，报告位于其 `observation_report`；审计工具 |
| `observed_audit_space_from_rgbd` | 将射线和已知模型投到明确审计网格 | 视图、原点/形状/体素尺寸、已知实体 → 可观察空间记录；审计场景 |
| `build_tsdf_geometry` | 读取 Mapper 的 TSDF 值、观测权重和坐标 | 已融合 Mapper、owner 名称、来源 → TSDF 定义的表面/负体积组件记录；冻结几何 |
| `tsdf_geometry_from_samples` | 从规则采样构造相同 TSDF 组件 | `sdf`、observed 掩码、原点、体素尺寸 → 源组件；数值构造与检查 |
| `coal_component` | 按组件维数构造 Coal 计算对象 | 有限组件顶点 → 原生几何及相应变换；源几何复核 |
| `iter_source_components` | 按记录遍历源组件 | 组件数组记录 → 组件迭代器；原生几何准备 |
| `FrozenObservedGeometry` | 持有只读 TSDF 源数组与空间索引 | 源记录、主机预算 → 冻结几何；`FrozenESDF` |
| `prepare_rules/component_indices` | 建立接触许可对应的源组件集合并查询邻域 | 规则或 AABB/活动规则身份 → 固定许可集合或组件索引；字段准备与复核 |
| `collision_component/rule_region/surface_points` | 取得源实体及某规则的表面见证 | 组件身份/规则身份 → Coal 组件、区域、表面点；接触与实体复核 |
| `distance_field` | 将保留组件映射到指定候选网格 | 网格原点/形状/尺寸、活动规则、排除组件 → float32 保守候选场；普通/接触/初始支撑字段 |
| `FrozenESDF.from_source` | 把源几何与固定查询网格组合 | 源组件、网格、owner/来源 → `FrozenESDF`；生产建图 |
| `signed_distance/__call__` | 插值点候选值或计算球残差 | 局部点或局部球 → 距离候选或非负碰撞候选；场景 |
| `supported/supports_centres/intersects_geometry` | 分别查询球范围、球心数值支持及源包围盒相交可能 | 局部球 → 布尔数组；场景查询范围判断 |
| `prepare_contacts/contact_query/required_contact` | 准备并消费有限接触字段 | 规则、球半径、当前策略、局部球 → 接触查询与所需接触分项；场景 |
| `record/from_record/clone_for_rollout` | 保存、恢复及共享只读数据 | 冻结对象或记录 → 对应记录/对象；交接及候选隔离 |
| `FrozenPrimitives` | 提供已知完整基本体的解析局部查询 | 盒/球/圆柱等已声明几何、身份和来源 → 基本体 query；明确模型入口 |
| `CollisionOwner.pose/predicted_pose` | 按候选工具、附着区间和释放记录确定 owner 坐标 | 工具位姿、进度、释放位姿 → owner 位姿；场景成对查询 |
| `OnlinePlanningScene.bind_online_robot` | 连接原生球、实体及自碰撞对 | 具名机器人球几何、自碰撞对及 padding、来源与机器人模型 → 更新请求场景绑定；后端初始化 |
| `validate_problem` | 核对任务对象、附着、接触和目标身份 | `PlanningProblem`、原目标索引 → 完成绑定检查；后端初始化 |
| `components/evaluate` | 计算同一时刻全部场景分项及聚合值 | 机器人球、工具位姿、进度、手指/关节/时间、可选实体复核 → 字典或逐状态残差；原生代价与验收 |
| `query_state` | 以单状态接口提供模型报告 | 球、工具位姿、进度、具名关节、时间、许可及可选复核器 → 状态及分项；显式调用与接口回归 |
| `native_collision_components/native_world_collision_components` | 提供本场景中的原生源实体 | 已绑定几何 → 具名组件；独立机器人检查器 |
| `native_world_collision_pairs` | 枚举实际登记的运动实体与目标关系 | 机器人、完整运动对象、场景目标 → 成对注册；实体复核 |
| `estimated_workspace_allocation` | 估算指定查询布局所需场景临时存储 | batch、horizon、覆盖样本及路径模式 → 分项字节估计；资源准入 |
| `describe/to_execution_binding/from_binding` | 描述、保存与恢复同一数学场景 | 场景或结构化记录 → 描述/绑定/场景；执行交接与复现 |

## 3. 接触、附着与手指

本组 API 的几何关系及公式见[接触组件](03-attachment-contact-and-hand.md)。`Attachment` 定义名义刚体运动，`ContactRule` 定义指定有限区域的模型许可，`HoldingContact` 连接已知机器人材料与声明附着。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `build_material_contact_rules` | 把材料几何与许可声明合并 | 材料绑定、规则声明 → `ContactRule` 元组；场景构造 |
| `bind_grasp_contact_rules/validate_grasp_contact_arrival` | 将名义抓取目标和 close 身份连接到规则 | 任务、规则、激活声明 → 已绑定规则或一致性检查；场景 |
| `bind_native_owner_geometry/check_native_owner_material` | 核对完整基本体及其有限材料 | 原生几何、覆盖球/实体身份、材料 → 注册字典及对应关系/检查；完整 owner 接触 |
| `PreparedContactFields.prepare` | 为实际规则子集准备固定候选字段 | 目标 query、源几何、规则、最大球半径 → 字段与见证对象；query 初始化 |
| `PreparedContactFields.query/required_contact` | 查询激活许可及所需接触条件 | 球、当前规则策略、普通候选与范围状态 → 残差/表示缺口；场景 |
| `native_material_conditions` | 计算材料在目标有限区域中的几何条件 | 规则、变换后的材料、法线、预备字段 → 支持/间隙/法向分项；接触许可 |
| `ContactCellGeometry/ObservedContactGeometry/LayeredContactCellGeometry` | 为基本体、TSDF 或已选分层源提供接触计算 | 相应源几何及区域 → 区域查询对象；接触/审计 |
| `make_contact_geometry` | 根据实际源类型创建接触计算对象 | query 或源记录 → 上述类型；场景准备 |
| `bind_holding_contacts` | 将机器人材料与声明附着区间关联 | 问题、材料绑定、持物关系声明 → `HoldingContact`；场景 |
| `validate_holding_materials/validate_holding_conditions` | 核对材料球身份与附着身份/变换/区间 | 持物接触、机器人材料或任务/owners → 一致性检查；场景构造 |
| `PreparedInitialSupport` | 保存入口支撑层组件及最低高度基准 | 完整作用物体、静态观测支撑、关系记录 → 支撑查询对象；场景 |
| `PreparedInitialSupport.components/collision_values/record` | 计算向下越界、活动区间碰撞字段并保存 | 当前两物体位姿、进度、球 → 支撑分项/候选场/记录；场景 |
| `FixedFingerTargets.from_mapping/bind` | 恢复明确手指目标并匹配原事件 | 具名目标、轨迹、关节顺序和控制范围 → timed commands；执行 |
| `FingerTrajectory.build` | 在共享物理时钟构造满足范围的 C1 指令 | 手指 `q0,dq0`、命令、时长、前缀、步长、位置/速度/加速度界 → 手指多项式；规划/执行 |
| `FingerTrajectory.sample/sample_torch` | 查询同一手指命令 | 物理时刻 → 位置与速度；设备控制/名义几何 |
| `FingerTrajectory.describe/from_mapping/validate_execution` | 保存、恢复并核对指令契约 | 曲线记录、入口与事件/设备数据 → 曲线或核对结果；执行入口 |
| `bind_nominal_finger_geometry` | 选择自由运动或声明持物的名义几何 | 手指曲线、入口反馈、附着/开闭区间 → `NominalFingerGeometry`；场景时钟绑定 |
| `NominalFingerGeometry.valid_mask/sample` | 查询有效前缀与名义手指位置 | 物理时刻 → 掩码和位置；规划碰撞 |
| `PrismaticGripper.bind_clock/state/spheres` | 绑定指令时钟，解释手指状态并计算球包络 | 设备/事件、名义或测量坐标、工具位姿 → 状态及球；场景 |
| `PrismaticSelfCollision` | 以明确手指实体计算手部自碰撞 | 手指运动关系及当前坐标 → 碰撞分项；场景 |
| `build_post_grasp_robot_config` | 从具名反馈建立持物阶段机器人配置 | 原机器人配置、完整具名反馈 → 含 `robot_config`、`planning_feedback`、`full_feedback`、`hand_binding` 的字典；阶段准备 |

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
| `NativeCollisionConfirmation.confirm` | 以登记实体和 TSDF 源组件澄清候选冲突 | 原生机器人状态、候选对、owner 位姿、激活规则 → 复核分项与几何证据；候选验收 |
| `OnlineCollisionConfirmation.confirm_self/__call__` | 将原生复核接入生产分项 | 自碰撞残差/关节或代价/同次证据 → 修订碰撞残差；初始/搜索/候选查询 |
| `RobotGeometryChecker` | 读取已知机器人运动学与实体模型 | URDF、基座和几何注册 → Pinocchio/Coal 检查器；`RobotStateValidator` |
| `RobotStateValidator.check_planning_self/check_planning_world` | 使用名义完整关节状态执行实体检查 | 机械臂、手指名义状态、时刻及场景 → 模型检查；规划复核 |
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

`tsdf_spatial_query` 与 `scene_surface_batch` 被 `RobotGeometryChecker.query_scene` 所组织的源几何查询路径消费，使用现有 `_coal_scene_batch` 扩展调度有限组件与表面距离。`native_voxel_sources.export_source_bits` 被审计建图的 `_map_partition` 调用，利用原生 Mapper 数据导出最近站点、观测符号与负值来源位。`voxel_layers` 按已经绑定的粗/细层记录选择当前层并保存来源。这组接口服务分层源几何审计；生产固定 TSDF 场景使用 `FrozenObservedGeometry` 与 `NativeCollisionConfirmation` 的局部源组件查询。

`robot_coverage` 提供机器人球覆盖、材料与实体索引关系。`geometry_motion_bound` 提供 `SphereBody`、`GeometryMotionBound`、`build_geometry_motion_bound` 和 `relative_speed_bound`，按运动学系数及明确速度范围计算几何速度上界。当前仓库的调用者为 `tests/test_geometry_motion_bound.py`，其逐函数参数、返回式和直接调用见实现索引。

`resource_admission.require_host_memory/require_device_memory` 输入新增存储需求、当前物理占用与明确上限，返回准入记录或类型化 `ResourceLimitError`。`optimizer_device_checks.DeviceChecks` 在设备计算期间保留有限性和域错误，`require_tensor` 在当前检查作用域内消费布尔张量。预算、诊断和数值错误共享同一请求生命周期。

记录脚本位于 `tests/planning`。`prepare_recorded_task.py` 将所选记录与已知机器人 fixture 组织为候选输入；`run_recorded_skill.py` 连接任务编译、`recorded_observation.py` 的观测输入和生产场景建图；`recorded_task_planning.plan_and_record` 调用规划后端并保存请求、结果与可执行曲线，场景通过库内 `to_execution_binding/from_binding` 保存和恢复。`recorded_owner_geometry.py` 读取明确完整基本体输入。`run_recorded_simulation.py` 读取正式结果，经 `recorded_simulation.py` 连接设备执行，并使用 `recorded_skill_evaluation.NativeSkillPhysicsEvaluator` 独立记录与评估真实仿真结果。`robot_fixture.py` 提供具名机器人模型、碰撞材料、标定和设备范围。具体命令与参数见[记录入口说明](../../tests/planning/README.md)。

跨阶段读写时，调用者需要同时保留几何记录、其坐标变换、对象身份、任务区域、原事件与机器人入口。各接口按上述结构解释数据；当前持物计划对这条交接的具体补充见[阶段对照](../../../sim-version-plans/2026-10-19-planner-architecture-audit/plan-comparison.md)。
