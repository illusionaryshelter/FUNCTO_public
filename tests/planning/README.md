# 记录式规划、执行与诊断入口

本文对应 FUNCTO_public 代码基线 `bc79a78071e464c35ed9839d62595f735ad85465`。规划器完整架构见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，逐 API 的作用、输入、输出和前置数据见[接口契约](../../docs/planner-architecture/06-api-contracts.md)。本目录负责把明确的记录数据连接到通用规划与执行库，通用 `manipulation` 模块不按数据集或技能名称派发。

## 运行环境与入口

`run_recorded_skill.py` 执行当前记录任务的输入绑定、场景准备和规划，`run_recorded_simulation.py` 读取已接受曲线并连接仿真控制及独立评价。使用现有 `recovery` 环境，命令显式设置 `PYTHONNOUSERSITE=1`。GPU、感知与物理仿真脚本应显式运行，按给定主机/显存额度串行安排；不因阅读本说明安装或调整环境。

两个入口的完整参数由各自 `--help` 给出。调用方须提供明确的 manifest、机器人 fixture、具名反馈、手指目标、执行限制和输出位置；规划入口另需当前任务、观测和 `--unknown-policy`。完整物体基本体、接触绑定和其它可选数据只通过各自现有参数提交，不从历史实验目录隐式补入。

`robot-fixture` 包含已知机器人模型、标定、碰撞实体及球所属关系。`feedback` 提供关节名称、位置、速度、设备时间与模型 FK 信息。所有 skill representation 完整保留，包括由仿真示教提取的源区域、轨迹、相对关系、任务几何和交互/释放事件。禁止求解额外引入的是仿真实际运行环境中产生的真值，包括本次环境的真实物体位姿、真实分割、接触和效果评分；这类数据只进入独立事后评价，不经输入文件、场景冻结或观察器进入规划及执行授权。各类文件的具体职责见[外部文件与数据职责](../../docs/planner-architecture/08-external-files-and-data-roles.md)。

## 规划阶段

`run_recorded_skill.run_prepared` 将 `PlanningProblem`、机器人配置和现场观测交给库。`planning_observation.build_scene_from_rgbd` 返回 `OnlinePlanningScene`：Mapper 融合结果先形成固定 TSDF 源组件，再准备保守候选距离场。物体 masks 与机器人自过滤决定端点归属，自由射线证据单独保留。详细数据流见[观测与碰撞模型](../../docs/planner-architecture/02-observation-and-collision.md)。

`--unknown-policy` 必须明确给出有限条件区域及区域外策略。`forbid`、`require_observation`、`allow_attempt` 分别决定未知覆盖的处理，不能把所有未知状态当成碰撞或自由空间。该策略不能删除已观测几何。

`--contact-bindings` 提交材料、目标有限区域、来源和对应的抓取或声明附着关系。任务接触由 `ContactRule` 表达，持物内部材料关系由 `HoldingContact` 表达。原 close 身份和名义规划区间仍被核对，但实际目标到达误差不控制闭爪。声明附着不会生成新的物体测量或成功抓持证明。当前表示限制和手指时钟见[附着、接触与夹爪](../../docs/planner-architecture/03-attachment-contact-and-hand.md)。

入口在请求生命周期内拥有 `MotionBackend`；库调用者可以跨请求复用常驻后端。主数值过程使用 cuRobo L-BFGS、项目物理关节域与功能约束计算。合格原关节参考或参考种子可直接交付；需要调整时进行主求解，并至多选择一次碰撞时刻补充或 OMPL 种子恢复。所有工作共享原预算，没有额外质量优化或场景重建循环。具体调用和候选条件见[数值求解与搜索](../../docs/planner-architecture/04-solver-and-search.md)。

## 结果保存与执行

`recorded_task_planning.plan_and_record` 保存后端一次返回的结果。已接受曲线通过 `save_curve` 携带精确多项式、原事件时钟、问题/场景记录、机器人身份和可执行范围。被拒绝曲线保存为 `rejected_diagnostic_not_executable`，没有正式执行绑定。记录器不重新求解、重排候选或调用独立源审计。`status=completed` 只表示记录流程完成，应另外读取规划 success 和 failure。

`TrajectoryExecutor` 验证已接受结果的绑定，并对实际具名关节反馈、速度、指令跟踪、设备时钟及已知机器人/固定安装件几何执行检查。当前执行器不恢复或查询冻结世界 TSDF。场景随曲线保存用于模型身份和来源记录，不能据此声称执行时检查了世界碰撞。

`--finger-targets` 给出原开、闭或保持事件对应的具名位置目标；`--execution-limits` 给出执行所用的跟踪、设备和停止范围。它们不建立第二条事件时钟。指令位置、名义手指几何和真实关节反馈分别记录，不能用命令冒充已测得手指位置或物理抓持。

显式独立效果依赖在原事件边界处理，需要续接时保存原身份、已经发生的事件与累计工作，并通过新请求绑定当前反馈。名义附着不要求额外 Holding 证明。末端相对预测目标的误差、缺失目标位姿及目标到达时间窗只用于事后评价，不能拒绝闭爪、继续执行、停止或重规划。

`run_recorded_simulation` 的独立评价器读取仿真实际运行环境中产生的真值，输出实际任务误差和效果；它的初始化或评分失败记录为评价缺失/错误，不具有拒绝运动的权限。在线 `observation_callback` 供另行声明的可获得观察使用，不转交这类运行环境特权状态。保存和执行契约见[验收、保存与执行](../../docs/planner-architecture/05-acceptance-and-execution.md)。

## 诊断工具及其用途

| 文件 | 当前职责 |
|---|---|
| `check_recorded_task_inputs.py`、`prepare_recorded_task.py` | 验证明确任务输入并构造当前问题 |
| `robot_fixture.py`、`check_recorded_robot_binding.py` | 核对具名机器人、标定和碰撞注册 |
| `recorded_observation.py`、`detect_recorded_owners.py`、`recorded_owner_geometry.py` | 连接标定观测、对象 masks 和显式提交的物体几何 |
| `recorded_simulation.py`、`recorded_skill_evaluation.py` | 控制适配与独立任务评价，二者的判断权限分别保留 |
| `visualize_recorded_task.py`、`visualize_recorded_failure.py`、`visualize_recorded_scene_failure.py` | 展示输入、保存曲线、实际反馈与失败位置 |
| `check_recorded_robot_geometry.py`、`audit_recorded_collision.py`、`robot_coverage_fixture.py`、`refine_robot_fixture.py` | 人工显式调用的机器人及源几何审计 |
| `probe_finger_control.py`、`probe_controller_takeover.py`、`probe_mounted_execution.py` | 显式控制和跟踪诊断 |

生产候选的局部源几何复核与独立审计具有不同入口。前者属于当前模型的必要判断，后者用于完整模型差异和覆盖分析，不由生产失败自动调用。独立审计使用 `build_audit_scene_from_rgbd` 等接口，不替代已返回曲线的正式绑定。

两个主入口均支持 `--profile-runtime`。其 `cProfile` 结果包含分析器开销，仍消耗原时间预算，不能当作独立 GPU kernel 计时。仿真入口的 `--result-only-goal` 与 `--diagnostic-goal` 调整独立评价的分类；它们不提供在线反馈，也不能改变机器人约束或原事件身份，更不会恢复目标到达门槛。

实验交付采用可直接打开的本地 HTML，优先展示真实仿真图像上的源示教区域、迁移目标区域和采样点，并保留接受、警告、拒绝、失败和缺失状态。阶段结果、计划及用户运行命令放在工作区 `sim-version-plans`，不写成库的隐式运行默认值。回归主题与失败分类见[验证与故障分析](../../motion_planner_debug.md)。
