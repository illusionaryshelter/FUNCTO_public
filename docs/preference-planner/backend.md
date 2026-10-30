# 驻留后端、联合计算与候选证据

代码基线：`8b0e326`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

`backend.py` 接收已经绑定显式角色的 `PlanningProblem`、观测场景、固定官方机器人模型、机器人验证接口和 `PlanningBudget`。输出为原 `JointTrajectory`、同曲线验收记录及 CPU 诊断。原 `MotionBackend` 继续持有机器人模型、容量、设备、锁和原生资源。

`PreferenceMotionBackend.bind_request` 要求请求拥有后端，按原接口核对容量。首次构造 `PreferencePlanner`，随后调用其 `rebind`。`plan` 显式调用本包的生命周期；继承的 `close` 释放原生资源。独立入口自动创建和关闭后端，驻留入口由调用者管理寿命。普通续接的终止合同来自之前正式绑定 `validation.export.reference_termination`，并在后端构造时作为 `incoming_terminal_contract` 传入。

`PreferencePlanner` 的继承顺序为 `NativeSetup, _ConstrainedPlanner`。原构造器设置精度和资源，通过正常方法分派执行事件成本初始化；异常清理继续使用原 `close`。`_bind_computation` 先调用原方法得到完整变量的 `TrajectoryComputation`，再由 `terminal_clock` 和 `TerminalHold` 建立终止映射，创建 `GeometryConstraints`，最终以 `JointComputation` 提供减少变量的计算接口。

`JointComputation.states(Z)` 在同一物理时刻计算 C 展开后的 q/dq/ddq/dddq。`task(Z)` 保留原任务行并附加几何必要行；`ReducedJointDomain` 对原连续关节条件保持原值，只将 Jacobian 映回 Z。`objective_components(Z)` 在同一次 FK 上计算共同四项成本并保存完整 C 的数值试探；普通积分索引、事件查询和几何锚点属于同一固定布局。

`reference_seeds` 调用原顺序参考 IK 和拟合，再进行 `expand(reduce(C))`。报告同时保留投影前后系数。`_controls_curve` 保存当前控制量与其精确导出曲线的身份。`_candidate_objective` 用保存多项式、同一查询时刻和共同目标评分；不同显式时钟使用原相位对齐，保持查询身份。评分的 no_grad 计算仅记录成本，SLSQP 中的同一公式负责必要修复的梯度。

`_candidate_domain_validation` 调用原连续关节检查，并对导出 Hermite 多项式的静止后缀求位置和速度极值。`_hold_polynomial_report` 查询断点、后缀端点及导数实根，报告具体关节与时间见证。要求起点采用原声明时间，三次表示提前形成常值段的时间另行记录。

`_candidate_metrics` 和 `_candidate_geometry` 复用原完整实体查询，收集同一曲线的已有 `EvidenceState` 和 `PairBatch` 几何证据；各对象与共用 FK 的所有权见[共享验收组件](../planner-architecture/10-shared-candidate-validation.md)。后者只装饰候选场景副本的 evaluate 调用，区间算法仍拥有采样、计费和未决记录。`evaluate_candidate` 在原验收完成后将完整 `continuous_geometry` 转为 `CompletedEvidence`，让候选报告与 `last_validation` 共享同一只读记录；原数值、实体和区间证据全部保留。它继续保存本次角色、终止映射、必要行数量、成本及接受状态，报告键为 `preference_joint_problem`。数据所有权、正式转换及文件接口见[完成证据与归档](../planner-architecture/09-completed-evidence-and-archives.md)。

`optimize` 将各原 seed 转为 Z，调用 `solve_trajectory` 并恢复完整 C，再将候选交给原选择器。只有拒绝曲线产生了新的必要几何行且原预算允许，才追加求解。无自由变量时直接提交唯一曲线。`_expand_snapshots` 将数值失败或中断记录从 Z 转为 C，并保留原 Z，便于按同一曲线追溯首次异常。

`diagnostic_snapshot` 深拷贝终止、角色、几何、求解、候选与目标试探。所有记录在产生处已转换到 CPU；非有限试探值保留为诊断。生命周期把快照交给结果对象，共享 `recorded_task_planning.plan_and_record` 选择 JSON 文件路径。该方法不执行模型查询或候选验收。
