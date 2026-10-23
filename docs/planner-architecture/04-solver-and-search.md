# 固定时钟的统一约束求解与有限搜索

代码基线：`ae2603f`；本次文档提交同步该实现的基线标记与索引，规划器代码内容与该代码提交相同。阶段实验与计划对照位于项目外 `sim-version-plans`。

## 1. 请求、变量与依赖

`plan_trajectory` 接受 `PlanningProblem`、`OnlinePlanningScene`、机器人配置、`RobotStateValidator`、工作预算和可选原关节曲线，返回 `PlanningResult`。`MotionBackend` 管理可复用 cuRobo IK、请求占用及释放。每次请求私有化任务和场景，绑定事件、名义手指和附着释放时刻。

优化变量 `a` 是 `[M,J]` 的 binary64 自由位置系数，J 为具名活动关节数，M 为 `budget.knots+5`。`CubicBasis` 定义 `M-1` 个时长相等的三次段；`CubicAffineMap` 将系数映到关节位置及各阶导数。入口位置 `q0` 和速度 `dq0` 使用原反馈数值固定。每段物理时长为 `h`，总时长为 `(M-1)h`；终点速度由声明条件决定。`_derived_spline_grid` 按请求采样间隔选择可表达的固定时钟。`reference_progress_clock` 以源时间—进度节点、显式停止事件和场景锚点计算独立标量参考时钟：归一化物理时间 `u=t/T` 经 `z(u)` 映到原参考时间参数，再得到技能进度 `s(z(u))`。该固定节奏保留原参考末段区间，线性化误差界为 1e-6 无量纲参考参数；机械臂系数和终端导数仍由上述曲线及任务条件决定。`_bind_event_clock` 将指令事件量化到设备周期，`_ProgressMap` 保存完整事件身份；初次绑定、新请求和显式重新绑定共用该计算，普通续规划保留冻结时钟。原关节参考通过 `source_progress_map/reference_time_map` 保留暂停区间内的运动相位，只用于一次初值采样。

生产依赖已有 SciPy 的 B 样条、分段多项式、最小二乘及 SLSQP，Pinocchio 的运动学及导数，cuRobo 的 IK 配置与采样器，以及 OMPL 的状态空间搜索。组件内数据接口和机器人注册模型明确限定各依赖的职责。

## 2. 参考与初值

`_reference_entrance` 从具名反馈、同一官方机器人和 TCP 标定取得当前参考坐标系的工具位姿及线/角速度。`BoundToolReference` 保留源完整工具路径，通过显式边界适配产生物理时间参考。`reference_seeds` 在既定正时间样本处调用 cuRobo IK，或者采样调用者原关节曲线。

原生 IK 的 `inner_iters=1`，原库外层迭代完整消费用户指定的任意正整数 `ik_iterations`，共享其原数值状态。普通参考 IK 保留完整位姿目标，功能事件 IK 通过 `_ik_manager_type` 使用声明允许集合。`planning_ik` 先比较原生成功和归一化关节连续性，再比较位姿能量；完整链保存每个查询节点的生产者和 `SOLVED/APPROXIMATE_PROPOSAL/INTERPOLATED_PROPOSAL` 身份。功能事件候选同时保留任务和静态模型资格。

`CubicBasis.reference_parameters` 为 M 次参考查询选取 M−2 个内部非初始分段边界，以及末段中点和终点。入口位置与速度已经由表示固定；内部位置观测后剩余的两个系数方向由末段内两次独立位置观测确定。该接口使用同一组样条节点与自由末端语义。

`CubicJointDomain.fit_samples` 在样本对应的实际时间上调用 `numpy.linalg.lstsq`，得到固定入口样条的自由系数。欠定情况用已有系数保留零空间分量。拟合报告保存原样本误差及同次 SVD 产生的奇异值、条件数，分别描述样本拟合程度与观测矩阵的数值条件。初值可以不满足关节或任务约束，统一求解器接受这类初值。`initial_seed_strategies` 记录 `reference_ik_samples/recorded_joint_samples/measured_joint_samples` 及拟合来源。

## 3. 一次轨迹计算

`TrajectoryComputation` 固定普通积分网格、任务锚点和场景关系锚点。每个查询具有归一化时间、物理秒和原技能进度；同刻事件保留原进度数值。`verification_issues` 在构造时检查所需事件和历史查询是否齐全。普通积分网格的索引独立保存，增加事件锚点保持原运动成本积分权重。

`robot_tool_states` 从仿射样条取得 `q,dq`，加入场景声明的名义手指坐标与速度，调用 `PinKinematics.states`。Pin 模型复用验证器的完整机器人。位姿导数使用帧 Jacobian；速度导数使用 Pin 的 LOCAL 帧原生导数，再经可微旋转表达在参考系中。工具固定平移贡献线速度杆臂项，固定旋转组成 TCP 姿态。`_candidate_robot_state` 对任意保存曲线调用同一接口。

`PlanningProblem.constraint_components` 返回等式字典和逐项有符号不等式字典。约定不等式 `g(a)<=0`，各行保留原物理单位、原任务点、半空间和事件身份。过程关系、位姿及机构坐标事件、方向历史、关节耦合和声明停止均从同一曲线状态计算。零偏航等式位于固定入口时，其值由 `q0` 决定；已满足常量行在求解接口结构消元，违反常量等式直接记录不可改变的入口失败。未来变量等式始终保留原行。

## 4. 连续关节约束

`CubicJointDomain.signed_constraints` 返回 `(g,J)`：`g` 为所有段和关节的有符号越界量，`J` 为对自由系数的导数。每段每关节共有十行：位置和速度的下、上界，加速度两个端点的下、上界，以及常量 jerk 的下、上界。位置和速度极值由 SciPy `PPoly` 的端点及导数实根确定。导数使用对应极值时刻的仿射基行；并列极值按确定次序取有效分支导数，分支切换处为分段光滑结构。

各行从原机器人关节限制取得边界，位置以弧度或米表示，各阶导数按物理秒计算。最终 `cubic_position_report` 对保存 Hermite 多项式独立计算原界和见证，求解器停止状态与该验收分别记录。

## 5. 目标函数与库求解

`motion_cost` 读取安装 cuRobo `trajopt/lbfgs_bspline_trajopt.yml` 的 STATE 配置，计算关节边界激活成本及速度、加速度、jerk 平方正则。原时间系数以请求 `native_dt` 一次绑定。普通时刻采用归一化物理梯形积分；当前状态只含运动学量，力矩项对应未计算状态。

`PlanningProblem.optimizer_preferences(..., phase='quality')` 提供声明的任务质量项，包括支持要求中非硬分量的原容差偏好。`OnlinePlanningScene.optimization_objective` 在原 TSDF 的固定局部材料点、原官方机器人形状与现有许可上计算有限几何目标；具体定义见观测组件。总代价为运动成本加 `constraint_weight` 倍任务质量与几何目标。最终合格候选用运动成本和任务质量计算固定比较分数。

`solve_trajectory` 直接调用 `scipy.optimize.minimize(method='SLSQP', jac=True)`。原任务和连续关节条件作为独立约束行提交。Torch 反向传播计算目标梯度，`task_jacobian` 计算完整任务 Jacobian，连续关节 Jacobian 由仿射多项式计算。设自由曲线系数为 `x∈Rⁿ`、原任务向量为 `c(x)∈Rᵐ`、其导数为 `J(x)∈Rᵐˣⁿ`。函数先以独立权向量 `v∈Rᵐ` 建立 `p(x,v)=J(x)ᵀv` 的反向图，再使用普通标量 `autograd.grad` 逐个计算 `∂p_i/∂v=J[:,i]`。第二次求导的变量是 `v`，机器人 FK 的一阶导数在这一计算中是系数。常量任务行通过显式零依赖得到零导数；所有原行按原顺序进入矩阵。

一次正比例目标归一化固定于初值，使数值尺度适配库算法；原目标、约束与候选评分保存原单位。库负责步长、子问题和停止判定。报告保存库状态、迭代数、实际求值数、系数更新、原目标、`task_jacobian_method` 及内存估计。`_planning_workspace_bytes` 分别保留场景驻留量、未分配共享数组和单次场景／Pinocchio 临时需求。每个串行候选求解前按当前样本构造状态重算 `pending_cpu_bytes`，通过 `solve_trajectory(pending_geometry_bytes=...)` 与实际约束行数确定的 SLSQP 工作区、Jacobian、独立权向量和额外反向缓冲合并。第一次主机准入位于任何求导之前。第一次反向图建立后，由 `task_jacobian` 的 `admit_work` 回调检查当前 RSS 与即将新增的 `8m(n+1)` 字节任务矩阵／单列缓冲、Pinocchio 求导临时缓冲之和。随后分配最终密集矩阵，每次仅保留一个返回列。此时原任务值、独立权向量、已建立反向图、SLSQP 工作区、场景数组和早先结果由实测 RSS 统一计入。

Pinocchio 临时增量为 `8[S(6FNᵣ+12F+Nᵣ)+2]` 字节，其中 `S` 是任务物理时间查询数，`F` 是完整机器人计算帧数，`Nᵣ` 是含名义手指的机器人关节数。对应同时存活的数据为一份 `[S,F,6,Nᵣ]` 乘积、两份 `[S,F,6]` 空间权向量、一份 `[S,Nᵣ]` 关节累积量及两个标量；每个元素为 float64。该数量计入当前求导新增工作区，原生运动学导数与第一次反向图由 RSS 计入。

多个种子串行调用同一求解函数。`_ConstrainedPlanner.optimize` 在调用前把该种子的独立 `report_out` 字典登记到 `last_optimizer_runtime`，正常返回的 `ConstrainedSolution.report` 也引用这份记录。求解内在实际回调及任务导数列间检查请求期限；退出时记录实际墙钟耗时与各类开始求值次数，`completed_evaluations` 另记已返回值与导数的次数，其中任务初始行数查询计为一次任务求值。异常记录保留原类型和原因，并继续向请求调用方传播。初始目标归一化完成后，正比例尺度和原目标数值立即写入同一记录。

报告只保留三个有界状态：`latest_trial` 保存最近实际开始计算的系数、递增试算编号和已完成分项；`latest_complete_trial` 保存目标、连续关节和所需任务值及导数均已完成的最近系数；`last_callback_iterate` 保存 SLSQP 回调给出的最近迭代系数及其与已计算试算的对应编号。目标值及其 `geometry_objective` 在同一次目标和梯度计算完成时一起复制到所属试算，几何记录的含义仍为有限优化目标的计算范围。回调保存已有系数，不增加求值。正常返回的顶层几何记录仅引用与返回系数一致的试算。

预算中断时，`optimize` 依次选取最近完整求值、库回调迭代、最近已开始试算，使用 `_controls_curve` 导出独立的 `diagnostic_optimizer_candidate`。`diagnostic_optimizer_provenance` 保存系数、种子和试算身份、完成分项以及已有目标证据，其 `validation_status` 为 `not_completed`。若有限系数在 Hermite 速度导出中溢出，`diagnostic_curve_export_error` 保存导出错误，原求解异常仍向调用方传播。`plan_trajectory` 单独保存已完成参考验收的 `reference_candidate`，其中 `curve` 是原多项式快照、`source` 为 `validated_reference_seed`、`accepted` 保留原判定。失败结果通过 `diagnostic_candidate_source` 区分 `diagnostic_optimizer_trial`、`last_solver_iterate`、`validated_reference_seed` 和 `validated_optimizer_candidate`。后两者的完整验收结果与曲线身份对应，保存在 `diagnostic_candidate_report`；`accepted` 保留原判定。归档消费者保留该来源与验收状态。SLSQP 终止描述数值算法状态，候选交付继续经过原条件验收。

## 6. 搜索、验收与预算

初值首先通过完整候选评价。主求解使用完整 `trajectory_iterations` 额度，在已有曲线基础上改善。原场碰撞或覆盖残差且搜索预算仍可用时，`ompl_seed` 在相同任务、场景和物理时钟上进行有限搜索，功能事件 IK 产生原允许集合的有限目标提议。搜索路径按实际时刻经相同最小二乘接口形成系数，再进入同一约束求解器。

`select_candidate` 按固定质量评分排序有限系数曲线，然后逐条调用 `evaluate_candidate`。保存曲线的任务、连续关节和原几何条件决定资格；完整合格且评分更低的候选更新请求记录。改善耗尽时交付已在预算内完成绑定的最佳记录。取消及外部期限按各自类型结束请求。

`PlanningBudget` 限定样条、IK 种子、轨迹种子、数值迭代、搜索时长、验证次数与查询量。CPU 工作区包含固定 FK 导数、任务 Jacobian、连续约束 Jacobian及原 SLSQP Fortran 工作区；`solver_storage_bytes` 在分配前按实际行列数计算。场景工作区按固定材料点和活动查询估算。主机与设备准入沿用调用者上限；原生设备资源用于 IK 与 OMPL 点查询。所有正式验证按实际完成批次计账。
