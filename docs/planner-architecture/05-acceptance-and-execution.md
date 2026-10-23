# 候选验收、执行交接与续规划

代码基线：`39880e5`；本次文档提交同步该实现的基线标记与索引，规划器代码内容与该代码提交相同。阶段实验与计划对照位于项目外 `sim-version-plans`。

## 1. 候选的完整验收

初值由参考 IK 或原关节曲线的物理时间样本经最小二乘取得。`reference_initialization.fits` 保存拟合结果；其系数、入口和时钟定义一条可评价曲线。`evaluate_reference` 为每个初值建立 `ConstrainedSolution` 记录，`select_candidate` 将初值和求解输出统一交给保存曲线验收。

`_ConstrainedPlanner._candidate_metrics` 消费实际 `JointTrajectory` 和候选场景，沿保存多项式计算 `q/dq/ddq/jerk`，通过共享 Pin 模型计算完整名义机器人、TCP 及速度。`_candidate_sample_times` 保留原多项式区间端点，并按 `interpolation_dt` 分割；任务锚点按候选自身进度映射精确加入。`PlanningProblem.residuals` 计算原任务允许集合，`_scene_boundary_evaluate` 计算相同物理时刻的原场景分项。报告保存具体分项、查询时刻及覆盖范围。

`SupportRequirement` 按实例 `hard_components` 验收，默认硬项为整体高度和有限投影范围；其他分量按原阈值提供求解目标中的质量偏好。位姿事件按材料原点距离、完整旋转或倾斜及声明 yaw 范围验收。完整轨迹按实际时间和速度评价机构方向历史、关节耦合和显式停止。

普通任务指标与连续关节域通过后，`evaluate_candidate` 调用 `_candidate_geometry`，以同一原曲线继续检查几何时间区间。`SceneMotionBounds.from_scene` 绑定完整机器人关节树、官方实体局部范围、原 owner TSDF 范围、工具标定和声明附着。`geometry_motion_bound` 中的局部包围球仅把关节转动上界转为实体点运动上界；实际碰撞查询仍使用原形状。机械臂速度由保存的 Hermite 多项式导数的 Bernstein 包络给界，名义手指曲线在其有效时间范围内提供手指速度，独立 owner 路径由原插值或机构坐标速率提供界。释放后的实体按该候选释放位姿保持静止。

`check_interval_geometry` 以进度映射折点、实体路径折点、关系事件及手指模型有效边界划分时间。端点证据的 `geometry_entity_key` 将机器人、任务实体和背景分别标为 `robot/<ID>`、`owner/<ID>`、`background`；速度表和配对缓存使用相同键。对区间长度 `Δt`，记两个实体的点速度上界为 `v_A,v_B`，令 `L=v_A+v_B`；端点完整实体距离下界为 `d_0,d_1`。`max(d_0,d_1)>LΔt` 或 `min(d_0,d_1)>LΔt/2` 可证明本区间分离。声明刚性组合体的内部对和空已表示集合分别保存其依据。

存在有限接触或初始支撑许可的端点使用独立的 `full_entity_lower_m`，该界由许可裁剪前的完整实体计算；无许可端点使用覆盖完整实体对的 `lower_m`。因此，完整实体范围已经足够分离的区间可以沿用上述证明。其余具有变化许可域的实体对保留 `dynamic_permission`，受许可裁剪的静态距离仅描述本次姿态下的未许可点对域。

未取得上述证明的区间按实体最大位移继续二分，直到位移上界不超过 `validation_spatial_step_m`、时间无法再分或共享查询容量用尽，保留为 `unresolved_intervals`。新增查询调用原多项式、完整 FK 和相同场景，并计算同一物理时刻的关系左右侧；进度平台只在入口使用左侧、出口使用右侧。第一次查询同时计算该曲线所需的释放锚点，后续细分从候选私有场景读取相同预测释放位姿。查询发现实际交叠后，保存见证并将其余待查区间记为 `not_checked_after_intersection`。`proved_intervals`、`confirmed_intersections`、`unresolved_intervals`、实际查询数量与时刻共同构成报告。

报告范围 `continuous_declared_geometry_model` 指输入数值所描述的连续运动与实体模型。该模型界的含义独立于设备浮点 FK 舍入和实际控制跟踪误差。空间细分截止、未知运动或容量不足均保留未决状态；它们不被解释为连续分离，也不新增碰撞拒绝。新增实际场景正残差以 `interval:process_scene` 进入原验收条件和恢复路径。`validation_joint_step` 继续用于 OMPL 连接检查。

候选使用固定机器人官方实体与原观测任务场。场景接口聚合原查询的硬分项；固定材料点几何目标为独立数值搜索目标，其零值只表示所采样材料点的结果。已确认体积交叠产生分类残差 1，场配对未决保留原状态。原生自身配对按原结构策略与数值边界检查，任务接触、初始支撑和覆盖条件分别计算。非 conditional 正分项表示其对应硬条件未满足。

每个验收样本以实际进度确定附着。工具与附着实体、同一 TCP 上共同附着实体的内部配对不进入碰撞查询；各成员原几何继续完成对外配对。`internal_assembly_pair_count` 保存内部配对样本数，工具成员由快照中的 `tool_geometry_names` 明确给出。

`_candidate_domain_validation` 检查保存曲线的入口 `q0/dq0` 及连续位置及一至三阶导数范围。`cubic_position_report` 对位置及其导数多项式求解析极值；`cubic_derivative_report` 另外记录加速度端值、内部节点两侧极限与跳变。`continuous_jerk` 读取每个三次段的常量三阶导数，包含断点两侧。通用 C1 Hermite 输入的位置与速度连续，加速度跳变按原报告保留，分段 jerk 描述其各段内及单侧值。`JointTrajectory.sample` 返回 `q,dq,ddq,s`，内部节点取右段导数，最终端点取左极限。

`evaluate_candidate` 对保存曲线统一完成任务与场景采样、连续关节界和区间几何报告。`optimization_diagnostics` 保存 SLSQP 的实际状态；正式报告来自保存曲线的原始条件。所有正式硬残差须有限且不大于零。几何区间报告保留已证明、交叠和未决状态。`select_candidate` 按固定质量评分依次评价，恢复类型读取该保存曲线的碰撞和覆盖分项。`export` 返回已完成资格的同一曲线与零新增查询。

任务过程的有限查询范围由 `test_task_sampling_scope.py` 中的显式反例界定。令目标坐标系中工具的平移为 `x(t)=0.08t(1−t)` 米，时间 `t∈[0,1]` 秒；其余位姿分量恒定，声明的 x 允许范围为 ±0.01 米。端点位置和速度唯一确定同一 Hermite 多项式。只查询两个端点时，任务有符号条件与原残差都合格；在 0.5 秒处同一曲线达到 0.02 米，原过程条件出现正违例。因而有限任务查询合格只证明所查询时刻，不扩展为相邻时刻之间的过程条件证明；连续关节极值和几何区间证据仍具有各自独立的范围。

## 2. 结果结构与输入身份

`PlanningResult` 保存 `success`、`trajectory`、`diagnostics`、`execution_binding`、`request_ledger`。成功结果的 `trajectory` 为 `JointTrajectory`；其具名关节、时间、位置、速度、进度映射、原事件、条件、手指曲线及已满足依赖共同定义可执行内容。失败候选以 `diagnostics.diagnostic_trajectory` 保存，用于解释残差。

`diagnostics.reference_initialization` 保存实际样本拟合及策略，参考 IK 查询与完整链保留原始候选来源。`last_optimizer_runtime` 保存每个已经完成的约束求解报告，包括状态、迭代、求值次数、原目标、归一化尺度与系数更新。异常路径保存已经完成的诊断，尚未完成的计算保持缺失。

`execution_binding.bind_candidate` 接受实际候选、任务、场景、连续验收和正式采样报告，生成 `candidate_id`，保存任务输入、`scene.to_execution_binding()`、`curve_snapshot` 与同身份验收。绑定参数 `native_report` 和记录 `validation.native` 保存该曲线的正式任务及场景评价报告，其生产者为 `_candidate_metrics`。`curve_snapshot` 保存原 Hermite 节点、实际系数、可执行终点和首个显式效果依赖边界；`plain` 将 NumPy 及公开 dataclass 字段转为普通记录，无界区间使用具名 JSON 表达。

`plan_trajectory` 将已完成执行绑定的曲线保存为独立最佳候选记录，包含原多项式副本、固定评分与分项、验收、候选场景、事件时间检查及资格完成时间。调用者的原关节曲线可具有自己的物理时长、进度平台和手指时钟；统一求解改善候选使用已绑定的三次基时钟。每条记录中的曲线、关系、命令和验收共同定义该候选。新候选完成这些计算且固定评分更低时才替换记录，后续求解器暂存值与记录分别保存。

`record_validation_work` 在每次正式候选评价前检查并登记验证尝试，在每批采样或区间查询完成后立即记录实际点数，再检查期限与预算。验证次数上限为原求解次数乘种子数；入口检查、查询点数和完整求解额度各按既有账本计量。每次实际调用统一求解器前登记轨迹求解额度；样本拟合属于初值构造。候选绑定记录完整构造后再检查预算，只有在预算内完成的记录才能成为最佳候选。

已有完整合格记录时，改善遇到 `PlanningWorkExhausted`，结果从该记录恢复成功曲线与绑定，`diagnostics.improvement_end` 保存停止阶段和原因。尚无合格记录时，同一异常产生预算失败。取消、外部截止时间和其他异常保持各自类型。所有返回路径在 `finally` 保存累计工作，候选资格完成时间与请求结束时间分别记录。

`check_result` 要求正式成功结果及完成的验收，读取 `validation.native.accepted` 和导出报告中的有限非正残差，然后核对候选身份、曲线、原始输入、时间和执行范围的一致性。这里消费轨迹验收已完成的结论；完整名义条件继续作为任务输入保存。可选 `execution_limits` 对照已绑定设备限制。返回独立字典副本，供执行入口消费。

`rebind_execution_result` 在明确控制策略、手指目标和设备限制下建立新的执行绑定，保留被验收曲线及数学输入。`validate_controller_policy` 核对按具名关节下发 `q,dq,ddq` 的控制解释、采样相位和加速度结点侧。`relax_pose_goal_tolerances` 是调用者显式扩大既有位姿目标允许集合的接口：输入具体目标身份、新容差及来源，验证集合包含关系后产生新任务绑定，保存原较严格条件下的已完成证据。此操作作用于尚未执行的结果。

## 3. 执行入口的数据契约

`TrajectoryExecutor(backend, *, geometry_checker)` 组合设备控制接口与独立 `RobotStateValidator`。`execute` 输入正式 `PlanningResult`、`FixedFingerTargets`、初始机器人反馈和 `ExecutionLimits`，以及按需要提供的独立效果观察器、记录回调和输出路径。

初始及后续反馈具有完整具名 `q/dq`、反馈身份、试验身份、时间、参考坐标、工具标定和工具位姿。机械臂与手指共同构成完整机器人关节顺序；设备原生 `qpos/qvel` 经适配器转换为这份具名反馈。`ExecutionLimits` 提供机械臂位置/速度跟踪带、工具中心点（TCP）相对指令轨迹的跟踪带、机械臂和手指停止速度、实际手指位置区间、事件限制与观察节拍。`FixedFingerTargets.bind` 把开闭目标与原轨迹事件匹配；实际手指允许区间和指令目标分别用于反馈检查与控制。

`RobotStateValidator.validate_planning_binding` 核对规划机器人与独立检查器的模型、关节和工具关系。`preflight` 检查执行入口、原曲线前缀和共享设备范围。`check_state` 处理实际完整关节状态、关节速度、原生机器人实体碰撞以及 FK/工具反馈一致性。`check_planning_self` 为规划阶段提供同一已知机器人模型的具名状态检查。

`check_state.distances_m` 按 `describe().pair_order` 输出原实体对距离，供执行器读取与记录；最小必检间隙单独汇总。数值边界与原模型穿透分别保留。未来手指状态未建模时的条件状态，与机器人模型缺失不是同一含义。

执行入口将任务中的每个 `declared_attachment` 与场景中对应实体的 `Attachment` 对照，核对条件身份、当前轨迹入口所处区间及固定变换。`declared_attachment_bindings` 记录这些已接受的模型关系，并保存 `tool_geometry_names`、释放身份和原来源。有限任务接触继续按原 `grasp_contact_activation` 与 close 事件绑定；附着模型记录与任务接触特征分别保留。

## 4. 控制时钟与每步操作

执行器保留轨迹原时间轴。每个设备步从保存多项式取得 `q,dq,ddq`，从 `FingerTrajectory` 取得手指指令，并交给 `backend.apply_control`。设备 `step` 推进一步后，`proprioception` 返回新的机器人反馈；执行器据此记录实际时间、跟踪误差、事件应用和物理步数。

`RobosuiteExecutionBackend` 实现仿真中的该设备接口。`describe` 给出关节、执行器、控制范围及标定；`apply_control` 实现已声明的关节轨迹控制；`reference_tool_pose` 用机器人 FK 计算指令工具位姿；`last_step_accounting` 记录实际推进的步数；`abort_hold` 在明确停止条件与有限步数内执行停止动作。

`TrajectoryExecutor._execution_violations` 对照实际机器人反馈与指令关节/工具状态，检查原设备约束及跟踪范围。工具跟踪项的目标是机器人应执行的指令位姿。名义功能目标的末端相对位置/姿态误差由事后评价计算；其数据来源和控制职责在[总架构](../../PLANNER_ARCHITECTURE.md)中定义。

当前 `execute` 中的碰撞检查覆盖机器人模型实体及其自碰撞。执行报告以 `world_collision_checked=False` 表达本次实际反馈检查的范围。规划场景快照继续保存在绑定中，用于解释原候选的模型验收。

## 5. 事件、独立效果与事后记录

原命令在其绑定物理时刻对应的首个设备步应用，`_activation` 以整数步和实际乘积确定该时刻。每个应用记录携带原技能、事件及 occurrence 身份。`_timing_checks` 计算实际命令或独立效果的时间要求，`resolve_event_window` 根据原事件窗口和实际历史求出允许范围。

`observation_callback` 提供独立任务效果的观察记录，状态使用 `T/F/U` 分别表示成立、不成立和未知；来源、关联条件及时间随记录保存。显式 `requires_evidence` 规定当前运动块结束后继续执行所需的效果条件，执行在该边界产生后续绑定所需状态。原任务声明规定效果记录与执行依赖的对应关系。

`step_callback` 为初始状态及每个已完成正常执行步提供事后记录数据，其返回值被忽略。首次回调异常写入报告并停用该记录接收方，不拒绝命令或触发停止。仿真评价器在这一通道读取仿真实际运行环境中产生的真值，包括当前物体位姿、接触与任务效果，计算目标误差和结果分数。`_posthoc_geometry_timing` 将依赖几何目标评价的时间项识别为事后记录范围；`pose_goal_schedule` 为评价器提供原目标时刻及对应设备步。`recorded_skill_evaluation._actual_pose_goal` 将评价时的材料位姿和目标位姿送入同一 `PoseEventRequirement.errors_numpy`，按各受约束分量评分，并另存完整转角与带符号 yaw。其输出仅写入事后报告。源技能中的轨迹和事件仍属于完整保留的技能表示，按原接口参与参考与任务编译。

## 6. 续规划与累计工作

执行报告包含实际反馈、已应用命令、事件与时间历史、已完成目标和独立效果、停止状态及原始执行绑定。`continuation_execution_inputs` 将其与已有手指目标、执行限制连接，并在明确重绑定声明下形成下一次执行所需参数。

规划器 `_prepare_continuation` 核对实际入口和原未完成义务，依据有效独立效果决定已满足依赖，保留原事件身份和物理时间来源。`_bind_motion_work_ledger` 将当前运动块与累计工作相连；一次只重试规划的请求记录为 `planning_retry`，执行过物理步后的继续请求记录为 `continuation`，明确模型或预算变更则保留其 `rebind` 来源。

账本保存 `consumed_planning`、绑定记录、原任务义务、事件及效果历史。`finally` 路径在成功、失败、取消和超时后均记录实际消耗，并释放请求占用；同一后端的长期缓冲由所有者在 `close` 时释放。

## 7. 失败分类与诊断消费

| 结果或原因 | 产生条件 | 调用者可读取的证据 |
|---|---|---|
| `needs_observation` | 当前未知空间策略要求取得相关新观察，且完成的分项证据支持该分类 | 覆盖与条件未知分项、涉及对象和样本 |
| `no_feasible_solution_found` | 有限候选与有限求解工作结束后，所选候选仍不满足声明条件 | 分项残差、种子来源、诊断曲线、搜索记录 |
| `cancelled` / `deadline_exceeded` | 调用者取消或外部请求期限到达 | 失败阶段、已消耗工作、最后完成候选 |
| 成功且 `improvement_end.reason='cumulative_planning_budget_exhausted'` | 改善预算耗尽前已有完成验收和执行绑定的候选 | 返回候选的评分、完整资格时间、改善停止阶段及最终累计工作 |
| `cumulative_planning_budget_exhausted` | 累计数值工作或请求时间达到预算，尚无完整合格候选 | 原预算、累计计数、当前运动块消耗 |
| `continuation_binding_gap` | 续规划数据与原事件/义务/反馈契约无法对应 | 具体绑定错误及原历史 |
| `event_time_requirement_unsatisfied` | 原事件物理时间要求无法成立 | 时间窗口、发生次数、相关原点 |
| `physical_domain_initializer_not_constructed` | 在固定入口和物理关节域内未构造出本次需要的初值 | 关节域、入口和构造错误 |
| `planning_stage_failed` | 阶段输入、资源或数值调用产生其他异常 | `failed_stage`、`reason`、`error_type`、原异常栈 |

`reference_initialization.status` 描述数值提议收集、工具曲线优化、完成、没有可分配初始化块以及中断。它与请求结果、快照中的 `sampled_scope_feasible` 和完整验收结论具有不同含义。`reference_candidate.qualification` 在省略初值完整扫描时记录明确功能违约的调度依据；已完成的普通完整验收继续保存实际候选报告。已有查询、完整快照、原始失败及缺失状态共同定位请求的实际停止阶段。

状态说明所完成计算的结论。结果消费者通过具体残差、输入来源及失败阶段决定下一项研究或操作。场景支持范围、未知区域、接触表示范围和实体相交各有独立分项，详见[观测与碰撞](02-observation-and-collision.md)。

场景快照类型为 `native_robot_observed_scene_v1`，保存原始任务场、位姿路径、附着、工具成员、原生机器人模型及其标定/配对/运动学、有限任务接触、支撑与自由格。单个任务场由 `FrozenObservedField.from_record` 恢复；场景快照用于输入保存与身份检查。
