# API 契约与调用关系

代码基线：`774e121`；正文描述该基线上本次已经实现的工作区代码，统一实现提交后更新为对应提交。阶段实验与计划对照位于项目外 `sim-version-plans`。

批量张量记为 `B` 个候选、`H` 个时间点和 `J` 个关节。常用布局为关节 `[B,H,J]`、位姿位置 `[B,H,3]`、四元数 `[B,H,4]`。实体场保存自身 XYZ 栅格，候选仅改变其刚体位姿。NumPy 保存数组通常移除候选维。位姿四元数采用 `wxyz`，齐次变换将右侧坐标中的点映到左侧坐标。物理时间以秒计，平移以米计，角度以弧度计。

初始化接口另以 `K=trajectory_seeds` 表示固定轨迹种子槽数，以 `S=ik_seeds` 表示每次原生 IK 的候选容量，以 `M` 表示三次基自由系数数，以 `D` 表示活动关节数。

## 1. 参考表示与任务编译

以下接口对应[输入组件](01-inputs-and-task-model.md)。示教表示和当前任务关系提供参考，机器人反馈提供本次入口；最终 `PlanningProblem` 是数值求解统一读取的任务对象。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `representation.build_representation` | 建立源示教的相对功能运动；各位姿须共用源世界坐标 | 帧/时间、源物体/工具位姿、功能局部变换、事件 → 相对关系与来源记录；`generate_reference` |
| `sample_poses` | 在源时间轴上插值位姿 | 有序时刻及位姿、查询时刻 → 刚体变换序列；参考适配 |
| `generate_reference` | 将源功能运动组合到当前功能坐标，并声明工具/物体入口和名义事件 | 表示、当前功能位姿、坐标对应 `J`、附着 `A`、当前入口 → 完整名义组合路径、采样数组、显式 `*_boundaries` 及缺失状态；任务适配器与 reference_phase |
| `program.ground_program` | 按程序声明绑定当前几何；持物声明与入口须代数一致 | `program,scene` → 工具参考、事件及几何诊断；`build_planning_problem` |
| `ground_object_program` | 生成当前物体参考 | 物体程序、当前关系与功能绑定 → 物体路径；持物参考调用者 |
| `function_motion.optimize_trajectory/transfer_function_motion` | 对已有参考施加明确的入口和功能关键帧 | 位姿序列、静态/当前入口、功能目标 → 适配位姿；`program` |
| `motion.MotionModel.poses` | 计算给定机构坐标对应的刚体位姿 | 已绑定轴/原点/类型、标量或向量坐标 → 位姿；机构参考与要求 |
| `MotionModel.coordinates_of` | 计算位姿对应坐标及模型相容性 | 机构模型、位姿 → 机构坐标；参考与机构诊断 |
| `build_planning_problem` | 编译一个技能、原事件、目标与过程要求 | `skills`、具名 `start_q/start_dq`、工具/基座标定、数值容差 → `PlanningProblem`；`plan_trajectory` |
| `PosePath.sample/sample_numpy` | 在给定进度读取完整组合路径及指定单侧导数 | 进度与 `side=left/right` → 位置/四元数或齐次变换；任务、物体运动与名义边界求导 |
| `reference_phase` | 将具名名义参考及其显式边界组织为阶段 | generate_reference 的 eef_path/eef_boundaries、技能身份、来源 → 保留组合路径与 reference_boundaries 元数据的 MotionPhase；问题构造 |
| `BoundToolReference.__init__` | 把实际物理时钟及机器人入口连接到已声明参考边界 | PlanningProblem、progress_u/s、duration、入口 T_W_E 与 `[v_W,omega_W]` → 名义阶段、物理边界、一阶修正多项式及诊断；规划器绑定、重绑与候选自身时钟 |
| `BoundToolReference.sample/sample_numpy` | 在物理秒上求同一完整工具参考 | 任意形状时刻 Tensor/NumPy → 同形状前缀的 W 系位置和 wxyz 四元数，或 4×4 齐次变换；参考 IK、保存与网页 |
| `write_reference_html` | 在真实源/目标图像中展示 grounding 几何输出 | 技能、RGB 视图、区域记录、名义路径及 eef_boundaries → 可缩放本地 HTML 与区域图，标记声明入口、名义首点与全部事件目标；config.ground_skill |
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
| `PlanningProblem.sample_reference/terminal_pose_target` | 提供完整名义进度路径或终端目标 | 已编译阶段、进度 → 名义工具参考或目标记录；任务表示与边界数据消费者，物理参考由 BoundToolReference 单独绑定 |
| `source_time_map/source_time_to_progress/progress_to_source_time` | 在源时间和技能进度之间换算 | 阶段时间、查询值 → 映射数组或转换值；时钟与报告 |
| `transform_tool_pose/reference_to_link_pose` | 连接原生 FK 连杆与任务工具坐标 | 固定基座/工具标定、位姿 → 对应坐标中的位姿；FK 与 IK |
| `transform_tool_velocity` | 转换工具速度 | 线/角速度及连杆姿态 → 任务坐标中的工具速度；方向要求 |
| `transform_robot_spheres` | 转换原生状态的空球数组 | 当前纯运动学配置的空数组 → 同布局空数组；原生状态接口 |
| `PlanningProblem.residuals` | 计算同一声明的硬条件 | 位姿、进度、具名关节/速度、时间、目标身份与 point/trajectory 范围 → 非负分项；所有消费者使用同一硬集合，完整路径另有历史和速度数据 |
| `PlanningProblem.evaluate` | 汇总已声明硬条件 | 与 residuals 相同的候选状态及任务输入 → 分项之和；直接任务评价 |
| `constraint_components` | 完整有序工具轨迹、进度、物理时间、具名关节及速度 | 原单位逐分量等式与有符号不等式；统一约束求解 |
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
| `query_distance` | 两个原场、刚体变换、数值精度和有限许可 | clear/intersection/unresolved、当前未许可域的 lower_m、完整实体的 full_entity_lower_m 及见证；静态分类与区间几何分别消费 |
| `pair_residual` | 原生实体—场或场—场、可微位姿、同值矩阵、精度、许可及可选共享零值/原生界 | 原查询证据与分类残差（intersection 为1，其余0）；硬分项，分类梯度为零 |
| `MaterialSamples` | 原场完整支持单元及官方机器人实体 | 不变材料点、原单元身份、接触零点与模型长度尺度；有限几何目标 |
| `pair_objective/native_pair_objective` | 两原实体、同刻可导位姿和有限许可 | 实际负场值/原生负距离的归一化平方及其位姿一阶导数；统一约束求解器 |
| `contact_objective` | 原场固定零点和当前声明区域半空间 | 零点到区域的最小平方超差；required contact 的有限偏好 |
| `query_region_intersection` | 原场、位姿、有限半空间 | 受支持零表面 contact/missing_contact/unresolved；required接触 |
| `query_minimum_height` | 原场、位姿与声明平面 | 整体最低高度区间及见证；初始支撑 |
| `robot_self_filter.classify_depth` | 原深度、采集时FK放置的原生组件、明确shadow padding | SELF/AMBIGUOUS_SHADOW/ENVIRONMENT、原自由射线及来源；建图 |
| `ObservedSpace/query_coverage` | 原自由射线格、运动实体场及条件区域 | covered/unknown_witness/unresolved；场景按明确策略解释条件区域及未知 |
| `NativeSolid/NativePrimitive` | 原凸组件顶点或解析尺寸、所属连杆、固定变换和来源 | 原生机器人实体；共享检查器 |
| `native_field_distance.query_native_field` | Coal 原实体、原任务场、两个位姿、精度、许可及可选同实体 local/world bounds | 当前未许可域的状态与 lower_m、完整实体的 full_entity_lower_m、可行点区间；场景与区间检查消费各自域的证据 |
| `native_pair_lower_bound` | 两个官方机器人原实体、当前变换及候选分离轴 | 完整实体支持投影的分离下界；区间几何证据，原生容差不作为误差界 |
| `RobotGeometryChecker.model_record` | 已消费的原模型 | URDF和原顶点/解析尺寸/局部变换的独立副本；输入绑定 |
| `DifferentiableRobotKinematics.poses` | 验证器现有 Pinocchio 模型、基座标定和完整具名关节 | 同一 CPU float64 FK，输出传回输入设备与精度并保留梯度；场景再合成原实体局部变换 |
| `RobotGeometryChecker.query` | 完整具名关节值、基座标定、冻结 PairQuery 和所需帧；同一模型中复用最近一次已验证的关节列、官方限位、frame IDs 与完整配对规则 | 每次状态的限位、独立 FK、原生有符号距离/见证/法向及分离/穿透/数值边界；`RobotStateValidator` 与场景共享机器人检查 |
| `RobotGeometryChecker._native_pair_distances` | 原有序配对与当前各实体刚体位姿；最近一组配对按每块最多 128 对保存原生模型及工作数据 | 每次先恢复原始求解器、写入当前变换并调用 pin.computeDistances，按原顺序返回全部距离结果；query 构造逐对证据 |
| `OnlinePlanningScene.bind_online_robot` | 同一验证器及请求期限回调 | 直接绑定检查器现有 Pinocchio 完整模型与标定；后端初始化 |
| `OnlinePlanningScene._robot_self_components` | 已准备的完整具名关节、可微机器人位姿及同值矩阵、手部状态、原 required_pairs | 官方机器人自身穿透与数值边界硬分项、原见证导数及查询报告；robot/full 两种 components 范围共用 |
| `OnlinePlanningScene.components/evaluate` | 同时刻关节、工具位姿、进度、时间与 geometry_scope | robot 范围计算官方机器人自身与手部，full 范围继续计算任务场；components 返回原分项，evaluate 汇总非 conditional 项，原证据独立保留 |
| `OnlinePlanningScene.optimization_objective` | 轨迹位置、单位四元数、进度、具名关节、物理时刻与必填 robot_link_poses | 固定材料点几何、原生自身、required contact 和初始支撑的标量目标；SLSQP 消费一阶导数，report_out 解释有限目标范围 |
| `OnlinePlanningScene.components(..., geometry_evidence_out=...)` | 同一实际场景查询及显式证据容器 | 全量逐实体对的状态、当前距离下界、许可与内部关系；geometry_entity_key 以 robot/、owner/ 或 background 区分实体类别，供区间检查 |
| `describe/to_execution_binding` | 场景已有输入、工具成员及实体附着 | `collision_rule='nonpenetration'`、`tool_geometry_names` 描述与 `native_robot_observed_scene_v1` 快照；结果绑定 |
| `clone_for_rollout/with_release_history` | 同一原始场及已选搜索前缀，或固定候选的释放锚点 FK | 私有关系状态、共享不可变场；搜索前缀与候选区间几何复用预测释放位姿 |
| `estimated_workspace_allocation` | batch/horizon、原场规模、实体数与样本／原生配对缓存构造状态 | CPU 已驻留、待分配共享数组、单次临时峰值三项；GPU 具名关节/连杆位姿及梯度传送；求解前将待分配与单次临时量和求解器合并准入 |

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

本组接口的计算定义见[求解组件](04-solver-and-search.md)。自由变量为固定入口、固定物理时钟的三次位置系数；输出为同一多项式的具名关节曲线。归一化时间、技能进度和关节列序在接口中明确传递。

| API | 必要输入与计算 | 输出及实际消费者 |
|---|---|---|
| `PlanningBudget` | 请求维数、IK/求解/搜索/验证额度和资源上限 | 固定请求预算；IK与主求解各完整消费自身迭代额度 |
| `MotionBackend.bind_request/plan/close` | 任务、场景、机器人模型和预算 | 创建或重绑IK及CPU计算对象，规划并释放；调用者 |
| `plan_trajectory` | 完整任务与原场景、独立验证器及可选原关节曲线 | `PlanningResult`、累计工作和诊断；执行交接 |
| `CubicBasis.weights/weights_at_knot_times/hermite_nodes` | 自由系数、原入口q/dq、段长和时间 | 同一曲线系数、导数及Hermite节点；域计算和保存 |
| `CubicAffineMap.query/evaluate/fit_samples` | 固定基、样本时间及目标关节状态 | 仿射矩阵、状态、最小二乘系数与误差；初值构造 |
| `CubicJointDomain.signed_constraints` | 原四阶关节界和实际自由系数 | 每段十类有符号行及Jacobians、可选极值见证；SLSQP |
| `PinKinematics.poses/states` | 已加载完整Pin模型、参考系标定、具名q/dq | 各帧位姿与速度及可微关系；场景和任务 |
| `robot_tool_states` | 活动关节、名义手指、物理时刻、进度和TCP标定 | 完整关节、帧位姿、TCP位姿与速度；求解与保存曲线验收 |
| `TrajectoryComputation.__init__/states/task/objective_components` | 固定完整查询、原任务、场景、仿射域和成本配置 | 原任务EQ/INEQ及运动、任务质量、几何标量；统一求解 |
| `time_mean_weights/motion_cost` | 普通物理网格、原STATE配置、物理导数和原界 | 时间平均运动成本；优化和固定评分 |
| `solver_storage_bytes` | 自由变量数及任务/物理约束行数 | 原SLSQP工作区及共享数值数组字节；分配前准入 |
| `task_jacobian` | 原任务值及自由系数的计算图、期限检查、主机准入回调 | 在独立权向量的反向图中逐列计算完整一阶任务导数；SLSQP任务约束 |
| `solve_trajectory` | 一条初始系数、完整计算对象、迭代额度、期限检查、场景待分配材料与单次几何临时字节 `pending_geometry_bytes` | 独立 `ConstrainedSolution(controls,report)`；联合内存准入与候选选择 |
| `_ik_manager_type` | 原生IK配置及可选事件允许集合 | 参考位姿或同刻声明允许集合的代价/指标管理器；原生IK |
| `rank_ik_candidates/select_ik_tracking_candidate/select_ik_chains` | 原候选、原成功与模型证据、真实时间、前态和界 | 成功及连续性优先的有限状态与完整链；参考样本 |
| `_reference_entrance` | 原具名q/dq、官方模型、手部入口与标定 | 参考系TCP位姿与线/角速度；BoundToolReference |
| `_ConstrainedPlanner.reference_seeds` | 完整参考或原关节曲线、固定时钟和IK预算 | CPU系数 `[1,K,M,J]`、查询成功数与拟合记录；初值评价和主求解 |
| `_ConstrainedPlanner._reference_ik_query/_select_reference_chains` | 实际时刻、明确seed来源及有限候选池 | 原查询节点、选择链和生产者身份；参考种子 |
| `_ConstrainedPlanner._build_goal_ik_candidates/ompl_seed` | 原功能允许集合、场景、固定时钟和搜索预算 | 事件候选及有限路径拟合系数；预算内恢复 |
| `_ConstrainedPlanner.evaluate_reference/optimize` | CPU系数及可选迭代数 | 直接初值记录或串行SLSQP结果列表；候选选择 |
| `_ConstrainedPlanner._controls_curve` | `[M,J]`系数、入口、原时钟、事件与手指曲线 | `JointTrajectory`；验收、归档和执行 |
| `_ConstrainedPlanner._candidate_objective` | 实际保存曲线及其场景 | 固定网格运动成本和声明任务质量；跨候选排序 |
| `_ConstrainedPlanner._candidate_metrics/_candidate_domain_validation` | 保存曲线、场景和记账接口、原关节界 | 原任务及场景采样报告、连续关节极值和入口残差；统一验收 |
| `_ConstrainedPlanner._candidate_geometry` | 保存曲线、候选场景、剩余点数与记账接口 | 分离证明、交叠见证、未决区间及硬分项；正式验收 |
| `SceneMotionBounds.from_scene/partition_times/interval` | 完整关节树、官方实体、TSDF、关系和曲线 | 时间分区与逐实体速度界；区间验收 |
| `check_interval_geometry` | 原曲线、运动界、几何查询回调和预算 | 已证明、交叠、未决及实际查询时刻；候选报告 |
| `_ConstrainedPlanner.select_candidate/evaluate_candidate/export` | 独立结果、同一候选评分及原模型验收 | 选中结果、曲线和报告；export交付已评价对象 |
| `plan_trajectory.record_validation_work/qualify_candidate/qualify_native/deliver_incumbent` | 已开始的验证与完成查询、已合格曲线和固定评分 | 即时工作账本与独立最佳绑定；请求返回 |

## 5. 验收与设备执行

这些接口的控制职责见[执行组件](05-acceptance-and-execution.md)。机器人反馈和独立效果观察有各自来源；仿真几何结果进入事后记录。

| API | 作用与前置数据 | 输入 → 输出；实际消费者 |
|---|---|---|
| `JointTrajectory.sample` | 查询保存的原多项式 | 有限原时间范围内的时刻 → `q,dq,ddq,s`；验收与控制 |
| `cubic_position_report/cubic_derivative_report` | 检查原多项式范围及加速度单侧/跳变 | 多项式或 Hermite 曲线、具名边界和可选前缀 → 连续极值、超界见证或导数报告；候选域与执行 preflight |
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

`implicit_distance` 对同一原场完成非穿透分类并保留有范围定义的距离界。`observed_collision` 返回原分类、有限接触和最低高度证据。`sampled_geometry_objective` 保存原材料点，在每次试探计算实际原场值或原生有符号距离及同一标量的一阶导数；当原点范围与原负值范围的坐标变换区间能证明两个方向均为零时，返回相同零值和零导数。报告单列此类配对数，支持与未知计数仅覆盖实际查询点。`observed_coverage` 解释实体是否处于原自由射线或明确条件区域。`observed_robot` 将完整具名关节映射为机器人连杆位姿。`native_field_distance` 连接固定机器人官方实体与原任务场，双方保留各自表示及共同世界坐标。场配对未决保存在查询证据中，原生自身数值边界、任务接触、初始支撑和覆盖各有独立硬条件。

`resource_admission.require_host_memory/require_device_memory` 输入新增存储需求、当前物理占用与明确上限，返回准入记录或类型化 `ResourceLimitError`。`optimizer_device_checks.require_tensor` 同步消费标量布尔张量，条件为真时返回 `None`，否则抛出声明的异常。手指曲线、任务定义与物理域在读取对应输入时使用该接口。`constant_tensor` 把声明的模型常量直接转换到消费者的 dtype/device，保留张量原有求导关系。预算、诊断和数值错误共享同一 eager 请求生命周期。

实验运行器位于 `tests/planning`。`run_recorded_skill.run_prepared` 接收已编译问题，读取已有实体，再为本次需要建图的实体读取 mask、深度与采集时反馈。`robot_validation_fixture` 构造已知机器人验证器和图像过滤组件；其 `tool_geometry_names(fixture)` 从声明手指机构的公共父子树返回工具成员元组，写入运行报告并传给场景。`observed_robot_fixture` 恢复已有任务实体。`load_contact_bindings` 读取必填 `contacts/source` 和可选 `grasp_activations/initial_support_relations`，分别提供原有限任务接触与初始支撑记录。`recorded_task_planning.plan_and_record` 调用后端并保存曲线、结果和 `native_robot_observed_scene_v1` 来源快照。`load_saved_trajectory` 恢复正式执行输入；该操作不重建可查询场景。具体命令与数据字段见[记录入口说明](../../tests/planning/README.md)。

跨阶段读写时，调用者需要同时保留几何记录、其坐标变换、对象身份、任务区域、原事件与机器人入口。各接口按上述结构解释数据；原生模型恢复与本次回放见[阶段对照](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。
