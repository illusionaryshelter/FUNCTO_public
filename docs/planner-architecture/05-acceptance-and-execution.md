# 候选验收、执行交接与续规划

代码基线：`__NATIVE_MODEL_COMMIT__`。本组件将一个给定关节多项式及其模型验收结果转化为可执行请求，并记录实际控制、独立效果证据和续规划状态。规划验收的研究对象是固定任务与固定场景中的名义曲线；执行检查的研究对象是具名机器人反馈及原控制时钟。

## 1. 候选的完整验收

`_CuroboPlanner._candidate_metrics` 消费 `JointTrajectory` 和本候选场景，沿原多项式计算机器人 FK、任务残差、同一观测场的场景分项。`_candidate_sample_times` 保留原多项式各区间端点，并按 `interpolation_dt` 分割区间；原生候选直接保留此前普通与密集计算的实际硬残差。确切任务锚点和已加入的碰撞见证单独采样，并计入同一查询容量。调用方提供的原曲线使用自己的区间、时间采样与任务锚点，没有先前原生网格。

当前 `validation_joint_step` 用于 OMPL 连接检查；`validation_spatial_step_m` 虽属于 `PlanningBudget` 并校验为正数，但没有生产消费者。`geometry_motion_bound.py` 提供独立运动界计算及其单元回归，当前候选验收没有调用该模块。本次验收的几何计算范围由上述有限时间网格和确切锚点定义。

候选使用同一已知机器人与观测任务实体。任务场景配对在 `query_role='acceptance'` 下使用 2 mm 间距，原生自身配对使用原结构策略与数值边界。所有非 conditional 正分项都表示当前硬条件未满足；未决状态单独报告。普通与密集样本共享同一场，不存在另一套碰撞实体替换阶段。

`_candidate_domain_validation` 独立检查保存曲线的入口 `q0,dq0` 及连续关节范围。`cubic_position_report` 对每个标量三次段求位置极值；速度范围来自导数极值；`cubic_derivative_report` 检查每段两端的加速度，并分别包含内部结点左右极限。`JointTrajectory.sample` 通过保存的 `CubicHermiteSpline` 返回 `q,dq,ddq,s`，在内部结点采用右段导数，在最终端点采用左极限。

`evaluate_candidate` 汇总模型条件、原始任务条件和连续关节域。已有完整 `native_report` 时使用其同一曲线的已完成几何证据，并继续检查保存多项式及入口；返回报告记录本次是否发生新查询及其数量。所有验收分项要求有限数且残差不大于零。有限采样几何验收的含义由采样规则、源几何和未知策略共同确定；连续关节极值检查具有独立的解析范围。

## 2. 结果结构与输入身份

`PlanningResult` 保存 `success`、`trajectory`、`diagnostics`、`execution_binding`、`request_ledger`。成功结果的 `trajectory` 为 `JointTrajectory`；其具名关节、时间、位置、速度、进度映射、原事件、条件、手指曲线及已满足依赖共同定义可执行内容。失败候选以 `diagnostics.diagnostic_trajectory` 保存，用于解释残差。

`execution_binding.bind_candidate` 接受候选、任务、场景、导出验收和原生报告，生成 `candidate_id`，并保存任务输入、`scene.to_execution_binding()`、`curve_snapshot` 和同身份验收记录。`curve_snapshot` 还保存实际多项式系数、可执行终点和首个显式效果依赖对应的边界。`plain` 将 NumPy 与公开 dataclass 字段转换为明确的普通数据；无界区间使用具名 JSON 表达。

`check_result` 要求正式成功结果及完成的验收，然后核对候选身份、曲线、原始输入、时间和执行范围的一致性。可选 `execution_limits` 对照已绑定设备限制。返回独立字典副本，供执行入口消费。

`rebind_execution_result` 在明确控制策略、手指目标和设备限制下建立新的执行绑定，保留被验收曲线及数学输入。`validate_controller_policy` 核对按具名关节下发 `q,dq,ddq` 的控制解释、采样相位和加速度结点侧。`relax_pose_goal_tolerances` 是调用者显式扩大既有位姿目标允许集合的接口：输入具体目标身份、新容差及来源，验证集合包含关系后产生新任务绑定，保存原较严格条件下的已完成证据。此操作作用于尚未执行的结果。

## 3. 执行入口的数据契约

`TrajectoryExecutor(backend, *, geometry_checker)` 组合设备控制接口与独立 `RobotStateValidator`。`execute` 输入正式 `PlanningResult`、`FixedFingerTargets`、初始机器人反馈和 `ExecutionLimits`，以及按需要提供的独立效果观察器、记录回调和输出路径。

初始及后续反馈具有完整具名 `q/dq`、反馈身份、试验身份、时间、参考坐标、工具标定和工具位姿。机械臂与手指共同构成完整机器人关节顺序；设备原生 `qpos/qvel` 经适配器转换为这份具名反馈。`ExecutionLimits` 提供机械臂位置/速度跟踪带、工具中心点（TCP）相对指令轨迹的跟踪带、机械臂和手指停止速度、实际手指位置区间、事件限制与观察节拍。`FixedFingerTargets.bind` 把开闭目标与原轨迹事件匹配；实际手指允许区间和指令目标分别用于反馈检查与控制。

`RobotStateValidator.validate_planning_binding` 核对规划机器人与独立检查器的模型、关节和工具关系。`preflight` 检查执行入口、原曲线前缀和共享设备范围。`check_state` 处理实际完整关节状态、关节速度、原生机器人实体碰撞以及 FK/工具反馈一致性。`check_planning_self` 为规划阶段提供同一已知机器人模型的具名状态检查。

`check_state.distances_m` 按 `describe().pair_order` 输出原实体对距离，供执行器读取与记录；最小必检间隙单独汇总。数值边界与原模型穿透分别保留。未来手指状态未建模时的条件状态，与机器人模型缺失不是同一含义。

## 4. 控制时钟与每步操作

执行器保留轨迹原时间轴。每个设备步从保存多项式取得 `q,dq,ddq`，从 `FingerTrajectory` 取得手指指令，并交给 `backend.apply_control`。设备 `step` 推进一步后，`proprioception` 返回新的机器人反馈；执行器据此记录实际时间、跟踪误差、事件应用和物理步数。

`RobosuiteExecutionBackend` 实现仿真中的该设备接口。`describe` 给出关节、执行器、控制范围及标定；`apply_control` 实现已声明的关节轨迹控制；`reference_tool_pose` 用机器人 FK 计算指令工具位姿；`last_step_accounting` 记录实际推进的步数；`abort_hold` 在明确停止条件与有限步数内执行停止动作。

`TrajectoryExecutor._execution_violations` 对照实际机器人反馈与指令关节/工具状态，检查原设备约束及跟踪范围。工具跟踪项的目标是机器人应执行的指令位姿。名义功能目标的末端相对位置/姿态误差由事后评价计算；其数据来源和控制职责在[总架构](../../PLANNER_ARCHITECTURE.md)中定义。

当前 `execute` 中的碰撞检查覆盖机器人模型实体及其自碰撞。执行报告以 `world_collision_checked=False` 表达本次实际反馈检查的范围。规划场景快照继续保存在绑定中，用于解释原候选的模型验收。

## 5. 事件、独立效果与事后记录

原命令在其绑定物理时刻对应的首个设备步应用，`_activation` 以整数步和实际乘积确定该时刻。每个应用记录携带原技能、事件及 occurrence 身份。`_timing_checks` 计算实际命令或独立效果的时间要求，`resolve_event_window` 根据原事件窗口和实际历史求出允许范围。

`observation_callback` 提供独立任务效果的观察记录，状态使用 `T/F/U` 分别表示成立、不成立和未知；来源、关联条件及时间随记录保存。显式 `requires_evidence` 规定当前运动块结束后继续执行所需的效果条件，执行在该边界产生后续绑定所需状态。原任务声明规定效果记录与执行依赖的对应关系。

`step_callback` 为初始状态及每个已完成正常执行步提供事后记录数据，其返回值被忽略。首次回调异常写入报告并停用该记录接收方，不拒绝命令或触发停止。仿真评价器在这一通道读取仿真实际运行环境中产生的真值，包括当前物体位姿、接触与任务效果，计算目标误差和结果分数。`_posthoc_geometry_timing` 将依赖几何目标评价的时间项识别为事后记录范围；`pose_goal_schedule` 为评价器提供原目标时刻及对应设备步。源技能中的轨迹和事件仍属于完整保留的技能表示，按原接口参与参考与任务编译。

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
| `cumulative_planning_budget_exhausted` | 累计数值工作或请求时间达到预算 | 原预算、累计计数、当前运动块消耗 |
| `continuation_binding_gap` | 续规划数据与原事件/义务/反馈契约无法对应 | 具体绑定错误及原历史 |
| `event_time_requirement_unsatisfied` | 原事件物理时间要求无法成立 | 时间窗口、发生次数、相关原点 |
| `physical_domain_initializer_not_constructed` | 在固定入口和物理关节域内未构造出本次需要的初值 | 关节域、入口和构造错误 |
| `planning_stage_failed` | 阶段输入、资源或数值调用产生其他异常 | `failed_stage`、`reason`、`error_type`、原异常栈 |

状态说明所完成计算的结论。结果消费者通过具体残差、输入来源及失败阶段决定下一项研究或操作。场景支持范围、未知区域、接触表示范围和实体相交各有独立分项，详见[观测与碰撞](02-observation-and-collision.md)。

场景快照类型为 `native_robot_observed_scene_v1`，保存原始任务场、位姿路径、附着、原生机器人模型及其标定/配对/运动学、接触、支撑与自由格。单个任务场由 `FrozenObservedField.from_record` 恢复；场景快照用于输入保存与身份检查。
