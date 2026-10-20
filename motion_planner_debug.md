# 当前规划器验证与故障分析

本文件引用代码基线 `85f67b53ffaa13b474c98061a0acfc51dea91c32`，说明如何解释当前实现的诊断和组织必要验证。组件职责由 [PLANNER_ARCHITECTURE.md](PLANNER_ARCHITECTURE.md)定义，运行契约见[当前运动规划器运行契约](motion_planner_prd.md)。实验耗时和阶段方案保存在工作区的阶段文档中。

## 从输入到结果核对同一请求

故障分析首先确认读取的是同一次请求的 `PlanningProblem`、固定场景、机器人配置、源时钟和累计工作记录。原关节参考、参考 IK 种子、主求解候选、恢复候选和最终导出曲线具有不同身份。某个中间候选的失败不能被当作最终曲线的判断，另一个请求的成功也不能覆盖当前请求缺失的模型输入。

`recorded_task_planning.plan_and_record` 保存返回结果和诊断曲线，不再次求解或重新选择。`status=completed` 表示记录流程完成，规划是否通过应查看对应 success 字段和具体 failure；只有已接受曲线携带有效执行绑定。保存的拒绝候选用于定位问题，不能交给执行器。

坐标检查应先确认 `T_A_B` 的变换方向、共同参考坐标、机器人具名关节顺序和 `wxyz` 四元数约定，再核对源帧身份、技能进度与物理时间。位姿数组的尺寸相同不能证明坐标相同；事件进度相同也不能证明物理时刻相同。相关 API 和前置数据见[输入组件](docs/planner-architecture/01-inputs-and-task-model.md)。

## 区分几何冲突与表示缺口

当前场景分别计算原生机器人自身配对、任务场非穿透、有限接触、初始支撑和未知空间覆盖。诊断保留 `OnlinePlanningScene.components` 的原分项及其出现时刻，`process_scene` 是这些条件的聚合结果。

| 已有证据 | 可得结论及后续分析对象 |
|---|---|
| 任务场配对 `intersection` | 原模型中存在共同占据的开邻域；保留原实体、支持单元及区间见证 |
| 任务场配对 `unresolved` | 尚有域在当前精度或许可边界下未判定；仅记录，不产生碰撞正残差，也不记为 clear |
| 任务场配对 `clear` | 已表示的未许可查询域完成排除；任务接触、支撑及未知空间条件独立计算 |
| 原生自身 `penetrating/numerical_boundary` | 分别表示原生实体穿透与查询数值边界；由机器人自身的独立碰撞或未决分项处理 |
| `internal_assembly_pair_count` | 工具与当前附着实体、共同附着实体之间排除的配对样本数；检查工具成员及实际附着进度 |
| `contact/...` | 当前材料、有限区域、法向、间隙或必需接触条件不满足；检查同一规则输入和活动区间 |
| `unknown/forbidden/...` | 几何进入当前策略禁止的未知覆盖；不是观察到碰撞 |
| `conditional/...` | 保留允许尝试区域或条件手指几何的不确定性；通过不构成已观察自由空间 |
| `initial_support/...` 或 `unresolved/initial_support/...` | 分别表示已确认的相对入口支撑下降违反量，或当前高度关系尚未判定 |

对于 RGB-D 模型，核对原实体 mask、背景候选与实际积分分区、米制深度、采集标定、自过滤和实体坐标。场查询直接消费原始 TSDF、观测支持及三线性单元，空间框只用于安排计算。见证中点是显示位置，体积交叠结论来自保留数值误差的原表示查询；分类残差不等于米制穿透深度。详细字段与判断权限见[观测与碰撞组件](docs/planner-architecture/02-observation-and-collision.md)。

`unknown_policy` 的三种区域外策略具有不同结果。`require_observation` 隔离出的缺观察失败可直接请求新观察，`forbid` 仍可在原预算内寻找绕行，`allow_attempt` 保留条件未知记录。任何策略都不改写已存几何、接触许可或原观测自由证据。

## 求解、曲线与恢复的诊断

当前数值主链是 cuRobo L-BFGS 与项目接入的物理关节域、功能约束计算。分析 IK 时，区分名义位姿收敛、静态状态资格和整条定时曲线的资格；有限 IK 返回集合不是全部逆解。分析优化时，保留实际使用的自由参数、入口位置和速度、固定时钟、功能残差及当前求解状态。低目标值不能补偿硬要求失败，未找到候选也不证明固定目标全局无解。

实际交付曲线由 `JointTrajectory` 保存的时间、位置和速度构造三次 Hermite 多项式。位置、速度和加速度的连续检查应作用于该多项式，并保留分段连接处的单侧导数。场景检查仍是有限的普通网格、密集网格和明确锚点，不能由通过这些查询推导连续无碰撞或真实接触稳定。

恢复只能消费已完成候选给出的明确证据。碰撞见证补充和 OMPL 种子搜索是最多一次恢复选择中的不同路径；它们不赋予新预算，也不修改目标、材料许可或场景表示。OMPL 产生的离散路径还需形成同一曲线并完成原要求检查。主求解与恢复的具体调用见[数值求解组件](docs/planner-architecture/04-solver-and-search.md)。

运行记录应同时保留成功与失败、冷启动和常驻请求、场景准备、IK、优化、候选确认、搜索、保存、主机与设备占用。没有测量的数据标为缺失，不从日志文本长度或架构分层推导耗时。`--profile-runtime` 的 `cProfile` 包含分析器自身开销，该开销仍消耗原预算；它不能独立给出异步 GPU kernel 的实际持续时间。

## 执行与事后评价的诊断

执行中的机器人检查面向当前具名关节反馈、模型 FK、登记机器人/固定安装件几何、速度、指令跟踪和原设备时钟。当前执行器不恢复或查询冻结世界场，报告中的 `world_collision_checked=False` 应保留。手指命令、名义几何和实际反馈属于不同数据；持物名义几何可以冻结入口测得位置，但这不改变执行指令曲线，也不证明真实 Holding。相关时段和记录详见[附着、接触与夹爪](docs/planner-architecture/03-attachment-contact-and-hand.md)。

末端相对预测目标的误差、缺失目标位姿和目标到达时间窗都不能成为闭爪或继续执行的拒绝理由。目标身份可保留为规划和事后评价元数据。声明附着的使用不需要额外的成功抓持证明；独立效果条件只有在任务明确声明依赖时，才按其本来的观察接口处理。不能借同名 Holding 标签恢复已取消的模型使用门槛。

仿真实际运行环境中产生的真值，包括本次对象位姿、接触和任务效果，由独立评价器读取。评价器初始化、记录或评分失败保留为评价缺失或错误，不决定规划、执行或下一动作。源技能中的轨迹、区域、关系和事件属于完整保留的 skill representation，继续用于 grounding、参考与任务计算。曲线完成、独立效果成立和物理任务成功分别报告。实际接口与保存契约见[验收、保存与执行](docs/planner-architecture/05-acceptance-and-execution.md)。

## 现有回归与独立审计

以下文件是已经存在的验证入口，表中说明其覆盖主题，不代表本次文档更新已经重新运行它们。

| 覆盖主题 | 现有入口 |
|---|---|
| 任务表达、事件与原时间 | [test_manipulation_planning_problem.py](tests/test_manipulation_planning_problem.py)、[test_planning_time_requirements.py](tests/test_planning_time_requirements.py) |
| 物理关节域、局部 QP 与原生优化适配 | [test_physical_joint_domain.py](tests/test_physical_joint_domain.py)、[test_physical_joint_qp.py](tests/test_physical_joint_qp.py)、[test_curobo_solver_components.py](tests/test_curobo_solver_components.py) |
| 原始观测场、原生实体与覆盖 | [test_implicit_distance.py](tests/test_implicit_distance.py)、[test_native_field_distance.py](tests/test_native_field_distance.py)、[test_observed_contact_coverage.py](tests/test_observed_contact_coverage.py) |
| 附着组合体、任务特征与手指几何 | [test_attached_tool_collision.py](tests/test_attached_tool_collision.py)、[test_grasp_region.py](tests/test_grasp_region.py)、[test_gripper_trajectory.py](tests/test_gripper_trajectory.py) |
| 候选、绑定与累计预算 | [test_planning_binding.py](tests/test_planning_binding.py)、[test_planning_request_budget.py](tests/test_planning_request_budget.py)、[test_recorded_task_planning.py](tests/test_recorded_task_planning.py) |
| 声明附着和执行职责 | [test_declared_attachment_execution.py](tests/test_declared_attachment_execution.py)、[test_execution_responsibilities.py](tests/test_execution_responsibilities.py)、[test_execution_wallclock.py](tests/test_execution_wallclock.py) |

`OnlinePlanningScene.last_query_report` 保存当前原模型的分类、实体身份与有限见证；`RobotGeometryChecker.query` 提供机器人原生配对证据。独立事后审计消费保存的输入及结果解释具体失败，其仿真真值不回流规划或控制。

验证沿用现有 `recovery` 环境和 `PYTHONNOUSERSITE=1`。GPU 规划、感知与仿真应显式运行，遵守资源上限并串行安排；文档更新本身不需要重新启动这些实验。实验结果应交付可直接打开的本地 HTML，优先展示真实仿真图像中的源示教区域、迁移目标区域与采样点，并保留失败、警告和缺失状态。Markdown 与原始数组作为补充，指标图表不能代替迁移区域可视化。
