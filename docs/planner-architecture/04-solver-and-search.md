# 固定时钟的统一约束求解与有限搜索

求解算法基线为 `be44b54`；同一请求中候选交付的当前实现见 [planning API](api/planning.md) 所列代码提交。组件正文描述当前实现；阶段验证位于项目外 `sim-version-plans`。

时间预算配置对应实际代码提交 `9602ebb`，本次文档回填保持该实现内容；当前默认值为 `PlanningBudget.planning_seconds=60.0`。

## 1. 请求、变量与依赖

`plan_trajectory` 接受 `PlanningProblem`、`OnlinePlanningScene`、机器人配置、`RobotStateValidator`、工作预算和可选原关节曲线，返回 `PlanningResult`。`MotionBackend` 管理可复用 cuRobo IK、请求占用及释放。每次请求私有化任务和场景，绑定事件、名义手指和附着释放时刻。

优化变量 `a` 是 `[M,J]` 的 binary64 自由位置系数，J 为具名活动关节数，M 为 `budget.knots+5`。`CubicBasis` 定义 `M-1` 个时长相等的三次段；`CubicAffineMap` 将系数映到关节位置及各阶导数。入口位置 `q0` 和速度 `dq0` 使用原反馈数值固定。每段物理时长为 `h`，总时长为 `(M-1)h`；终点速度由声明条件决定。`_derived_spline_grid` 按请求采样间隔选择可表达的固定时钟。`reference_progress_clock` 以源时间—进度节点、显式停止事件和场景锚点计算独立标量参考时钟：归一化物理时间 `u=t/T` 经 `z(u)` 映到原参考时间参数，再得到技能进度 `s(z(u))`。该固定节奏保留原参考末段区间，线性化误差界为 1e-6 无量纲参考参数；机械臂系数和终端导数仍由上述曲线及任务条件决定。`_bind_event_clock` 将指令事件量化到设备周期，`_ProgressMap` 保存完整事件身份；初次绑定、新请求和显式重新绑定共用该计算，普通续规划保留冻结时钟。原关节参考通过 `source_progress_map/reference_time_map` 保留暂停区间内的运动相位，只用于一次初值采样。

生产依赖已有 SciPy 的 B 样条、分段多项式、最小二乘及 SLSQP，Pinocchio 的运动学及导数，cuRobo 的 IK 配置与采样器，以及 OMPL 的状态空间搜索。组件内数据接口和机器人注册模型明确限定各依赖的职责。

## 2. 参考与初值

`_reference_entrance` 从具名反馈、同一官方机器人和 TCP 标定取得当前参考坐标系的工具位姿及线/角速度。`BoundToolReference` 保留源完整工具路径，通过显式边界适配产生物理时间参考。调用者提供的完整关节曲线先保留其原多项式及物理时钟，完成全部候选验收和执行绑定，合格即交付。其未通过且允许继续搜索时，`reference_seeds` 才采样该曲线；未提供关节曲线时，该方法在既定正时间样本处调用 cuRobo IK。

后端长期持有两个独立 cuRobo `InverseKinematics` 实例。`reference_ik` 使用原生普通位姿管理器和固定一个 seed；`ik` 使用 `_ik_manager_type` 表达功能事件允许集合，容量为 `budget.ik_seeds`。两者的 `inner_iters=1`，原生外层迭代完整消费 `ik_iterations`，保留原数值状态。配置在交给原生构造器前独立复制，原生构造过程中产生的配置对象属于各自实例。部分构造失败时释放已经创建的实例；`close` 逐一解除所有权并释放工作区，恢复调用者精度设置，重复释放不重复调用原生资源。

普通参考查询按严格递增物理时刻执行。第一次 `current_state` 的 q/dq 来自具名反馈；之后 q 来自上一原生返回值，dq 是相邻返回位置差除以实际时间差。每次 dt 是本次与前次样本的物理秒差，每点新建 `JointState` 后明确传入原生 `solve_pose`，并克隆返回位置以隔离下一次原生缓冲写入；唯一 warm seed 是前次 q。`optimization_dt=None` 使传入 dt 沿原生状态接口更新，速度由上述差分明确计算；原生速度和加速度正则的权重分别为 0.01。两实例均关闭 LM seed 和提前退出，功能实例的入口 current_state 使用显式 dt=0。完整位姿权重保留六轴，旋转使用 Lie group 定义并沿用原平方尺度换算。原生成功标志和实际位姿误差用于记录；有限返回位置进入下一查询及样条拟合，非有限位置不能形成数值状态，记录 `NONFINITE_STATE` 并结束初始化。功能事件候选由 `planning_ik.rank_ik_candidates` 按明确任务/模型失败、尚未建立的资格、与前态的关节向量距离及原始索引排序；其当前搜索分支的场景资格由 OMPL 实际节点查询决定。

`CubicBasis.reference_parameters` 为 M 次参考查询选取 M−2 个内部非初始分段边界，以及末段中点和终点。入口位置与速度已经由表示固定；内部位置观测后剩余的两个系数方向由末段内两次独立位置观测确定。该接口使用同一组样条节点与自由末端语义。

`CubicJointDomain.fit_samples` 在样本对应的实际时间上调用 `numpy.linalg.lstsq`，得到固定入口样条的自由系数。欠定情况用已有系数保留零空间分量。拟合报告保存原样本误差及同次 SVD 产生的奇异值、条件数，分别描述样本拟合程度与观测矩阵的数值条件。初值可以不满足关节或任务约束，统一求解器接受这类初值。`initial_seed_strategies` 记录 `local_reference_ik` 或 `recorded_joint_samples`。`reference_seeds` 返回 `[1,1,M,J]`，第二维为实际产生的一条初值；`trajectory_seeds` 是后续每批候选的数量上限。

## 3. 一次轨迹计算

`TrajectoryComputation` 固定普通积分网格、任务锚点和场景关系锚点。每个查询具有归一化时间、物理秒和原技能进度；同刻事件保留原进度数值。`verification_issues` 在构造时检查所需事件和历史查询是否齐全。普通积分网格的索引独立保存，增加事件锚点保持原运动成本积分权重。

`progress_clock.progress_anchor_parameters` 从同一固定进度表计算 binary64 锚点，暂停区间保留入口与出口。`_ProgressMap` 对原生张量查询另外检查 float32 可表达性。`query_parameters` 以锚点原归一化时间乘总时长得到物理查询键；遇到同一键时直接恢复原归一化时间和进度，避免乘除反算改变事件身份。连续 CPU 几何的原物体路径分段直接使用 binary64 锚点。

`robot_tool_states` 从仿射样条取得 `q,dq`，加入场景声明的名义手指坐标与速度，调用 `PinKinematics.states`。Pin 模型复用验证器的完整机器人。位姿导数使用帧 Jacobian；速度导数使用 Pin 的 LOCAL 帧原生导数，再经可微旋转表达在参考系中。工具固定平移贡献线速度杆臂项，固定旋转组成 TCP 姿态。`_candidate_robot_state` 对任意保存曲线调用同一接口。

`PinKinematics.poses/states` 按实际 Torch 梯度需求选择原生计算。启用 autograd 且输入需要梯度时，`_evaluate(..., with_derivatives=True)` 计算原帧 Jacobian 和速度导数。纯前向评分、验收及不需要梯度的调用使用 `forwardKinematics`、`updateFramePlacements` 和相同的 LOCAL 速度读取；位姿、标定、四元数和参考系旋转公式保持一致，导数缓冲为空。坐标、接口和验证见[运动学 API](api/pin_kinematics.md)。

`PlanningProblem.constraint_components` 返回等式字典和逐项有符号不等式字典。约定不等式 `g(a)<=0`，各行保留原物理单位、原任务点、半空间和事件身份。过程关系、位姿及机构坐标事件、方向历史、关节耦合和声明停止均从同一曲线状态计算。条件生产函数可同时写入 `solver_rows_out=(equality_masks, inequality_masks)`：与完整条件具有同一键、顺序和形状的布尔布局。固定进度界、声明区间外零行、方向历史中的固定首项及零宽固定入口偏航均由声明和曲线结构识别。布局只在初始任务求值中绑定，先核对被省去行的原值有限且满足等式为零或不等式非正，再按同一布局向 SLSQP 提交剩余行。`fixed_task_constraints` 保存原键、组内行索引、值和判定。固定行违反条件时直接返回原初值及不可变条件失败记录。完整 `task` 与最终验收保留所有原值；某次数值导数为零的变量行仍参与求解。

## 4. 连续关节约束

`CubicJointDomain.signed_constraints` 返回 `(g,J)`：`g` 为所有段和关节的有符号越界量，`J` 为对自由系数的导数。每段每关节共有十行：位置和速度的下、上界，加速度两个端点的下、上界，以及常量 jerk 的下、上界。位置和速度极值由 SciPy `PPoly` 的端点及导数实根确定。导数使用对应极值时刻的仿射基行；并列极值按确定次序取有效分支导数，分支切换处为分段光滑结构。

各行从原机器人关节限制取得边界，位置以弧度或米表示，各阶导数按物理秒计算。最终 `cubic_position_report` 对保存 Hermite 多项式独立计算原界和见证，求解器停止状态与该验收分别记录。

## 5. 目标函数与库求解

`motion_cost` 读取安装 cuRobo `trajopt/lbfgs_bspline_trajopt.yml` 的 STATE 配置，计算关节边界激活成本及速度、加速度、jerk 平方正则。原时间系数以请求 `native_dt` 一次绑定。普通时刻采用归一化物理梯形积分；当前状态只含运动学量，力矩项对应未计算状态。

`TrajectoryComputation.objective_components` 使用 `PlanningProblem.optimizer_preferences(..., phase='feasibility')`：显式过程等式与方向命令保留原偏好，支持要求的非硬分量和位姿/坐标事件中心偏好由独立质量记录承担。全部硬任务行仍由 `constraint_components` 产生。`OnlinePlanningScene.optimization_objective` 在原 TSDF 的固定局部材料点、原官方机器人形状与现有许可上计算有限几何目标；具体定义见观测组件。搜索总代价为运动成本加 `constraint_weight` 倍过程/方向偏好及几何目标，报告分项分别为 `motion_quality`、`task_preferences`、`observed_geometry`。

`_candidate_objective` 在固定查询上另外使用 `phase='quality'`，记录运动质量与完整声明任务质量之和 `fixed_objective`，分项为 `motion_quality` 和 `task_quality`。该分数必须有限，继续作为完整候选资格条件及结果记录；它不再比较首个合格候选与尚未计算的后续候选。普通参考路径只初始化曲线，不自动生成任务中心代价或必须通过的 waypoint。

`solve_trajectory` 直接调用 `scipy.optimize.minimize(method='SLSQP', jac=True)`。原任务和连续关节条件作为独立约束行提交。Torch 反向传播计算目标梯度，`task_jacobian` 计算完整任务 Jacobian，连续关节 Jacobian 由仿射多项式计算。设自由曲线系数为 `x∈Rⁿ`、原任务向量为 `c(x)∈Rᵐ`、其导数为 `J(x)∈Rᵐˣⁿ`。m 是结构消元后实际提交的任务行数。m<n 时，对每一标量任务值调用 `autograd.grad` 得到一行。m≥n 时，先以独立权向量 `v∈Rᵐ` 建立 `p(x,v)=J(x)ᵀv` 的反向图，再逐个计算 `∂p_i/∂v=J[:,i]`。第二次求导的变量是 v，机器人 FK 的一阶导数在此作为系数。每条路径分别需要 m 或 n 次标量反向查询，保留相同有符号条件及原单位。被保留的常量导数通过显式零依赖表示。

一次正比例目标归一化固定于初值，使数值尺度适配库算法；原目标、约束与候选评分保存原单位。正比例变换保留精确最优解集合；SLSQP 的固定 `ftol` 同时作用于目标、梯度和约束判据，因此有限迭代路径和停止时刻随尺度变化。运动、过程/方向偏好和几何项的相对权重表达实际优化偏好。库负责步长、子问题和停止判定。报告保存库状态、迭代数、实际求值数、系数更新、原目标、`task_jacobian_method` 及内存估计。`_planning_workspace_bytes` 分别保留场景驻留量、未分配共享数组和单次场景／Pinocchio 临时需求。每个串行候选求解前按当前样本构造状态重算 `pending_cpu_bytes`，通过 `solve_trajectory(pending_geometry_bytes=...)` 与实际约束行数确定的 SLSQP 工作区、Jacobian、独立权向量和额外反向缓冲合并。第一次主机准入位于 SLSQP 工作区、密集任务 Jacobian 分配及标量反向求导之前；初始任务求值已经建立的原生 Pinocchio Jacobian 和速度导数由 RSS 计入。列方向在第一次反向图建立后、行方向在首行反向前，由 `task_jacobian` 的 `admit_work` 回调检查当前 RSS 与即将新增的密集矩阵、一个返回行或列及 Pinocchio 求导临时缓冲之和。行路径的任务矩阵加单行临时向量为 `8(mn+n)` 字节，列路径的任务矩阵加单列临时向量为 `8(mn+m)` 字节。随后分配最终密集矩阵，每次只保留一个返回向量。此时原任务值、SLSQP 工作区、场景数组和早先结果由实测 RSS 计入；列路径的独立权向量和第一次反向图也已驻留，行路径未建立这两项。

Pinocchio 临时增量为 `8[S(6FNᵣ+12F+Nᵣ)+2]` 字节，其中 `S` 是任务物理时间查询数，`F` 是完整机器人计算帧数，`Nᵣ` 是含名义手指的机器人关节数。对应同时存活的数据为一份 `[S,F,6,Nᵣ]` 乘积、两份 `[S,F,6]` 空间权向量、一份 `[S,Nᵣ]` 关节累积量及两个标量；每个元素为 float64。该数量计入当前求导新增工作区，原生运动学导数与第一次反向图由 RSS 计入。

实际种子逐条调用同一求解函数，以生成器返回每个已完成结果；调用方在请求下一结果前完成当前结果的验收和合格记录绑定。`_ConstrainedPlanner.optimize` 在调用前把该种子的独立 `report_out` 字典登记到 `last_optimizer_runtime`，正常返回的 `ConstrainedSolution.report` 也引用这份记录。求解内在实际回调及任务导数行/列之间检查请求期限；退出时记录实际墙钟耗时与各类开始求值次数，`evaluation_seconds` 分别累计 objective、task、physical 的实际计算时间，包括任务初始布局查询及被中断的计算，缓存命中不计入；`completed_evaluations` 另记已返回值与导数的次数，其中任务初始行数查询计为一次任务求值。异常记录保留原类型和原因，并继续向请求调用方传播。初始目标归一化完成后，正比例尺度和原目标数值立即写入同一记录。

报告只保留三个有界状态：`latest_trial` 保存最近实际开始计算的系数、递增试算编号和已完成分项；`latest_complete_trial` 保存目标、连续关节和所需任务值及导数均已完成的最近系数；`last_callback_iterate` 保存 SLSQP 回调给出的最近迭代系数及其与已计算试算的对应编号。目标值及其 `geometry_objective` 在同一次目标和梯度计算完成时一起复制到所属试算，几何记录的含义仍为有限优化目标的计算范围。回调保存已有系数，不增加求值。正常返回的顶层几何记录仅引用与返回系数一致的试算。

预算中断时，`optimize` 依次选取最近完整求值、库回调迭代、最近已开始试算，使用 `_controls_curve` 导出独立的 `diagnostic_optimizer_candidate`。`diagnostic_optimizer_provenance` 保存系数、种子和试算身份、完成分项以及已有目标证据，其 `validation_status` 为 `not_completed`。若有限系数在 Hermite 速度导出中溢出，`diagnostic_curve_export_error` 保存导出错误，原求解异常仍向调用方传播。`plan_trajectory` 单独保存已完成参考验收的 `reference_candidate`，其中 `curve` 是原多项式快照、`source` 为 `validated_reference_seed`、`accepted` 保留原判定。失败结果通过 `diagnostic_candidate_source` 区分 `diagnostic_optimizer_trial`、`last_solver_iterate`、`validated_reference_seed` 和 `validated_optimizer_candidate`。后两者的完整验收结果与曲线身份对应，保存在 `diagnostic_candidate_report`；`accepted` 保留原判定。归档消费者保留该来源与验收状态。SLSQP 终止描述数值算法状态，候选交付继续经过原条件验收。

## 6. 搜索、验收与预算

`PlanningBudget.planning_seconds` 给出一次规划请求的累计墙钟上限，默认 60 s。`plan_trajectory` 的原计时器覆盖后端绑定、参考初始化、搜索、优化、候选验收和执行绑定；原请求重试继续累计已有消耗。`check_request` 消费同一账本，因此阶段切换与后续候选不重置时间。记录式 CLI 的省略值直接读取该字段；显式预算仍按原 API 传入和校验。输入场景准备及物理执行保留各自计时边界。`trajectory_iterations=100`、最多两个求解批次及其余原预算字段保持独立语义。

初值首先通过完整候选评价；合格并完成执行绑定时立即交付。若 require_observation 策略的完整分项证据将失败隔离为仅覆盖不足，返回 `needs_observation`；否则已有正碰撞或覆盖残差触发一次 `ompl_seed`，再进入主求解。forbid 的覆盖违例可由搜索处理，require_observation 与其他失败混合的证据沿同一几何调度；allow_attempt 的条件未知保持原含义。功能事件 IK 给出允许集合中的边界提议；搜索将每个提议放到当前前缀私有场景中，按该分支已建立的释放锚点和关系计算任务及几何资格。预计算时缺少释放锚点的静态判定只作为记录。

搜索路径按同一固定物理时间和最小二乘接口产生一个初值。搜索成功时该初值优先，候选上限 K≥2 时再加入原普通初值；搜索失败时主求解继续使用原初值，仍只消耗该次优化批。初次验收无几何违例、主求解随后发现几何违例且尚无合格候选时，可在尚有批次数额度的条件下使用这一次未消耗的搜索机会。搜索调用的 `finally` 汇总实际开始的状态查询和已耗时间，取消、超时及原生异常同样计入累计账本。

`optimize` 依原始种子顺序逐条产生 `ConstrainedSolution`。`select_candidate` 立即将每个结果转为保存曲线，运行完整 `evaluate_candidate`，以必填 `on_candidate(result, report)` 把当前曲线和验收送给 `plan_trajectory.qualify_native`。该回调在推进生成器之前完成执行绑定并冻结合格记录。回调完成执行绑定和原请求检查后，第一个合格结果立即返回，选择器不再推进求解生成器。原始 seed 索引、来源和验收保持一致；若所有结果均被拒绝，则按原残差严重度及原索引选取失败诊断。`plan_trajectory` 对原关节曲线、参考初值、主求解及恢复求解采用同一完整资格边界，首次绑定完成后直接交付。取消、外部期限、累计预算和原生异常在资格完成之前按各自类型结束请求。`finally` 释放后端所有权并写入实际消耗；交付记录完成后的清理耗时继续记账。

`PlanningBudget` 规定每个运动块最多 M 次绑定和 M 批优化，每批至多 K 条实际候选，每次验收至多 P 个查询。令 B 为已获准初始化的绑定数、S 为已开始优化批数、V 为已开始正式验收数，则 `V≤2B+KS≤M(K+2)`；其中每次绑定最多验收一条原关节曲线和一条拟合参考。入口场景查询另计一次，所以累计正式查询量 `Q≤PV+B≤PM(K+2)+M`。验证在开始时计次，查询在完成每批时计数，中断部分保留其实际消耗。K 与功能 IK 的 seed 容量各自定义上限。

CPU 工作区包含固定 FK 导数、实际提交的任务 Jacobian、连续关节 Jacobian 和 SLSQP 工作区；`solver_storage_bytes` 在分配前按实际行列数计算。场景工作区按固定材料点和活动查询估算。双 IK 实例构造前使用原资源准入，构造后根据真实主机与设备占用核对同一请求上限。后端重绑定更新请求任务、时钟和计算对象，继续复用这两个固定形状实例；所有者 `close` 释放它们。

`_candidate_evidence_workspace_bytes` 将候选验收保存的完整实体对记录、同次查询的重组字典及后续区间证明记录计入主机峰值。记录数量由注册实体对、物理查询上限、关系侧锚点和原物体路径分段确定；它与单批场景查询和 Pinocchio 临时量一起进入 `_planning_workspace_bytes`。同一候选已经完成的物理查询在区间检查中复用，批量查询只计入新请求时刻；记录的距离界、见证和关系侧保持原场景求值语义。
