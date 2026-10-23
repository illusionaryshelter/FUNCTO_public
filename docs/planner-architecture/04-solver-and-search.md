# 固定时钟轨迹求解与有限搜索

代码基线：`2498027`。本次文档引用更新由随后文档提交保存，组件代码内容与该实现基线一致。本章讨论 `planning.py` 如何将 `PlanningProblem`、`OnlinePlanningScene` 和机器人模型联合转化为关节曲线。`q` 表示具名机械臂关节坐标，`dq` 表示其每秒变化率。问题入口按 `problem.joint_names` 排列，原生求解按 `_CuroboPlanner.joint_names` 排列，保存曲线按其 `trajectory.joint_names` 排列；各边界通过名称建立列映射。`s` 表示技能进度，`u=t/T` 表示总时长为 `T` 的轨迹归一化时间。每条候选的技能进度 `s` 与归一化时间 `u` 通过其绑定的 `_ProgressMap` 联系。原生优化在请求的一套固定时钟上运行；调用者提供的关节曲线保留自己的物理时长与进度映射。

## 1. 请求生命周期与资源分配

本章的 `rollout` 指从候选动作计算关节轨迹、机器人 FK、成本和约束的原生计算对象。`seed` 指一次求解中的数值起始候选，`particle` 指同一候选在一次线搜索中按不同倍率生成的试探。功能增广拉格朗日简称 AL，其乘子和罚参数在第 6 节定义。

`plan_trajectory(problem, scene, robot, *, robot_validator, ...)` 是一次请求的完整入口。输入任务已经具有参考、事件身份、目标与机器人入口状态；场景已经具有观测几何、物体身份、手部模型及接触关系；`robot_validator` 连接独立机器人模型检查；`robot` 指向 cuRobo 配置。返回 `PlanningResult`，同时保存分阶段耗时、累计规划工作记录、候选残差与类型化失败原因。

`MotionBackend` 在多次请求之间复用原生求解器与固定容量缓冲。`bind_request` 创建或重绑定 `_CuroboPlanner`；`plan` 将本实例传给同一 `plan_trajectory`；`_claim/_release` 保证一个实例同一时刻由一个请求使用；`close` 释放所持资源。直接调用 `plan_trajectory` 时由该函数创建并关闭本次拥有的后端。请求为任务和场景建立私有副本，事件时钟与候选释放位姿在该副本中绑定。

`PlanningBudget` 同时规定数值维数、工作次数和资源范围。默认 IK 种子为 8、轨迹种子为 2、`knots=8`、原生细分为 4、IK 迭代为 40、轨迹迭代为 100、最多完整轨迹求解为 2、搜索预算为 3 s、请求预算为 120 s。各项含义不同：种子数控制并行候选，`max_task_anchors` 控制确切事件与补充见证容量，`max_validation_points` 控制完整验收的有限查询数量，主机 RSS 与设备占用限制控制实际资源。默认硬件预算延续本地半内存、半显存约束。

`_solve_ik` 统一提交普通参考和功能事件 IK。调用者通过 `seed_config` 提供 `[batch,supplied_seed,joint]` 数值起点，原生 `SeedManager` 在同一固定 `ik_seeds` 容量内补齐其余种子。每次提交前按原生批次数乘 `ik_seeds` 增加同一 `request_work["ik_queries"]`；原生失败或异常仍计入已提交工作。累计上限为 `max_validation_points × ik_seeds × maximum_trajectory_solves`，达到上限时产生累计工作耗尽结果。全部调用继续检查请求期限。

`trajectory_iterations=N` 为一次轨迹求解额度中的原生更新数，必须是原生内迭代块长度 `b` 的整数倍。首次额度在 `reference_seeds` 之前登记：初始化分得 `floor((N/b)/2)·b` 次，第一次完整场景优化分得剩余次数。总数只有一个内迭代块时，初始化保存同一数值初值的前后记录，状态为 `no_initialization_iteration_block`，该块全部交给完整场景优化。后续碰撞或搜索恢复使用其原有完整求解额度。初始化、完整求解和诊断均使用同一请求截止时间、取消检查及累计工作账本。

`_planning_workspace_bytes` 从机器人自由度、连杆数、样本数和场景工作区推算本次分配需求；`_require_planning_host_memory` 与 `_require_planning_memory` 在分配前检查存储；`resource_admission` 对源几何与保存记录的实际分配继续承担相应检查。`_request_work_ledger` 和 `_check_work_budget` 记录并检查请求累计时间及求解次数。续规划从原账本继续记账，失败诊断保留已完成的工作量。

## 2. 任务进度、事件时间与可执行区间

`_initial_timing` 使用源技能时间建立初始进度映射，`_progress_for_cubic_boundaries` 将阶段、确切锚点、停止进度与三次曲线的物理区间对应。`PlanningBudget.interpolation_dt` 在请求入口表示最大采样间隔。`_derived_spline_grid` 按原结点间隔选择满足该上限的二次幂细分数，并用 `_fixed_spline_grid` 核对原生 float32 与保存 binary64 时钟的结点间隔、总时长完全一致。容量检查使用派生出的真实样本数；绑定后的预算保存实际间隔，后续普通采样、密集采样、手指和导出共同消费该时钟。`_bind_event_clock` 生成事件物理时间、发生次数及命令记录；`_validate_bound_event_times` 检查这些时间与原事件约束、已有实际历史的一致性。

`PlanningProblem.bound_motion_block` 依据原始依赖关系划定当前运动块。这里运动块指由一次规划和执行绑定管理的轨迹区间。函数复制问题，将 `end_progress`、事件、点目标与源时间映射截取到最早尚未满足的显式效果依赖；后续义务保存在原请求账本中。`JointTrajectory.next_verification` 返回当前曲线中的首个未满足依赖事件，`executable_until` 给出对应可执行终点。普通事件和参考阶段边界不自动添加停止。当前求解曲线末尾的停止区间由本章三次基产生，取得相应效果证据后的下一段通过续规划契约重新绑定。

`_prepare_continuation` 消费原请求的事件身份、执行进度、实际入口反馈、未完成义务、已执行事件和工作账本；`_continuation_timing` 保留原请求时钟中的剩余关系。`_obligation_contract` 保存尚需满足的任务义务，`_restore_original_direction_origins` 恢复机构方向要求的原始坐标起点。每次事件由 `(skill_id,event_id,occurrence)` 标识；同名重试具有不同的 occurrence。

## 3. 共享三次曲线与物理关节域

`CubicBasis(controls, knot_dt)` 使用 SciPy 三次 B 样条建立自由系数到关节位置及一至三阶时间导数的线性映射。关节坐标为 `q`，物理时间为秒制 `t`，`jerk` 指 `d³q/dt³`。转动关节这四类量的单位依次为 rad、rad/s、rad/s²、rad/s³。令自由位置系数数量为 `M`、原段间隔为 `h`，完整系数含 `M+5` 行、有效曲线含 `M+2` 个区间；规划器采用 `M=budget.knots+2`，所以区间数为 `budget.knots+4`。

入口位置 `q0` 和速度 `dq0` 是测得边界。若第一个自由系数为 `x0`，前两个完整系数由 `c0=x0−2h·dq0`、`c1=1.5q0+0.5h·dq0−0.5x0` 消元；末端重复最后一个自由系数，形成整个末区间的停止状态。实际计算使用相对系数 `x−q0`，位置为 `q=q0+W(x−q0)+Vdq0`。`weights` 和 `weights_at_knot_times` 返回指定阶数的 `W,V`，导数按同一物理秒缩放。

`CubicTrajectoryStep.sample` 消费 `[batch,M,dof]` 动作、入口索引和原生状态缓冲，生成普通或密集原生指标需要的 `q,dq,ddq,jerk,dt`；`batch` 是候选数，`dof` 是活动关节数。基权重和相对运算使用 float64，结果转为原生 dtype；Torch 对实际前向算术求导。`forward` 实现 cuRobo 状态转移签名，核对固定时钟及自由终端声明；`cubic_transition_config_type` 把该计算装配到原生 FK、状态和成本管理中。

保存曲线由 `CubicBasis.hermite_nodes(actions,start_q,start_dq)` 生成。它在整数原断点 `0…M+2` 上以 binary64 计算位置、速度，物理时刻为整数乘 `h`。每段三次多项式由两端位置与一阶导数确定，因而这些原节点可直接构造现有 `JointTrajectory` 的 Hermite 表示。保存定义与密集原生指标网格各自保留职责；正式评价读取实际保存曲线。

`JointPositionDomain` 管理单次 IK 状态的具名关节位置闭区间。`CubicJointDomain(bounds,start_q,start_dq,knot_dt,knots)` 管理完整曲线；`bounds[r,side,j]` 按 `r=0,1,2,3`、下界/上界、具名关节顺序提供同一份位置至 jerk 原界。`_bind_physical_domains` 从原 `JointLimits` 按活动关节名收集这些边界。入口 `q0/dq0` 固定、时钟固定后，`q(t;c)=B(t)c+d(t)` 对自由系数 `c` 仿射。每个时刻的位置及一至三阶导数边界均为系数线性不等式，所有时刻的交集构成同一凸域。

`all_extrema(action)` 返回尾维为 `[4,dof]` 的连续最小值和最大值，`extrema(action,derivative)` 选择其中一阶。位置检查速度二次式的根及端点；速度检查加速度线性式的根及端点；加速度检查每段两端；jerk 检查每段常量，包含断点两侧。`feasible/validate` 对实际系数的位置及一至三阶导数的全部极值及入口条件判断。入口位置、速度须在原界内；位置正处边界而速度向外时，有限加速度曲线会立即越界，该代数必要条件单独保留。

`CubicAffineMap` 在 CPU 上为同一基提供仿射行。`query` 使用归一化物理时间，`query_knot_times` 使用原段坐标；两者输出导数量的物理单位相同。实例绑定 `q0/dq0` 与时钟时，保存位置、速度、加速度在原断点处的固定行，以及 jerk 在各段中点处的固定行。每次求解复用这些数值数组，并为起始曲线的额外极值生成仿射行。`solve_joint` 将待求量表示为相对 `origin` 的系数增量，每个关节串行调用已有 quadprog；四类边界共同形成上下不等式，`step_limit` 约束增量幅度。解析极值交换最多进行 24 轮，每轮只为此前未装配的实际越界极值新增行，已有行和原边界保持数值不变。

`initialize`、`descent_step` 与 OMPL 路径拟合共用上述完整域。`fit_cubic_reference` 对搜索返回的软关节路径使用最小二乘目标，任务事件由原任务接口评价。`_construct_anchor` 先检查测得 `q0` 重复形成的自由系数曲线；入口速度非零时，该曲线仍保留测得 `dq0`。需要构造新的域内点时，`construct_cubic_anchor` 调用现有 SciPy `linprog(method="highs")`。

该线性规划逐关节处理全部自由系数。设单条原边界行为 `aᵀc≤b`，`c` 为该关节自由位置系数；引入同关节共享的系数空间半径 `σ≥0`，最大化 `σ`，并对已有行施加 `aᵀc+‖a‖₁σ≤b`。半径与自由系数具有相同关节单位，表示当前已装配约束中的系数扰动余量；`σ=0` 保留原始边界可行集合。实际候选转为动作 dtype 后，由完整连续极值检查中心曲线；发现新的越界极值则增添对应原边界行，最多 24 轮。返回记录中的 `control_radius` 描述有限装配问题的半径，完整曲线是否可用由原 `feasible` 判定。

QP 答案按照实际动作 dtype 的增量乘加形成候选，连续极值随后按原界复算。原生方向经相同域的 limiter 处理实际倍率；搜索路径拟合若发生舍入越界，则沿同域可行锚点到拟合系数的线段缩回。`fit_contraction_scale`、按导数阶数分列的 `maximum_bound_excess_before_contraction` 及 `continuous_max_abs_derivatives` 记录实际变化。`finite_constraints_inconsistent`、`rounded_boundary_repeated` 和 `extrema_exchange_limit` 分别说明有限子问题与交换过程的数值状态；锚点线性规划另保存实际 LP 状态及 `rounded_anchor_outside_domain`。原物理边界与入口定义保持为同一计算依据。

`projected_descent_step` 同时服务单状态位置域与完整曲线域。单状态子问题使用原位置盒；曲线子问题使用位置及一至三阶导数的同一仿射行。它按关节串行投影完整负梯度，原步长参数约束最远试探的系数增量。每个关节的完整曲线增量先经原生 dtype 的共同 limiter 校正，再合成方向并检查全部实际线搜索倍率。记录包含梯度与方向内积、每关节舍入缩放、QP 轮数和实际数值状态；真实成本下降由线搜索计算。

## 4. cuRobo 装配、参考 IK 与工具曲线初始化

`_CuroboPlanner._initialize` 核对机器人身份、工具、活动关节和场景，建立时钟、手指曲线及工作区，再创建 IK 与轨迹求解器。配置读取 `ik/lbfgs_ik.yml`、`trajopt/lbfgs_bspline_trajopt.yml` 和 `metrics_base.yml` 中的原目标、维数及迭代参数。`bind_solver_components` 在首次求解前把原阶段替换为 `PhysicalGradientCore`，复用原 rollout、目标和种子管理器；这些配置中的目标、维数与迭代参数由共享核心消费。`_bind_optimizer_constraint_config` 将原指标的 `cspace_cfg` 深拷贝到优化约束，采用相同原界、权重和零激活距离，并装配相同普通成本供固定评分。生产请求使用 eager 张量计算，方向子问题在 CPU 上调用 quadprog。

`_native_request` 统一构造 `constrained` 轨迹请求：`PlanningProblem` 中的类型化目标负责名义硬要求，`reference_objective` 负责完整路径的软跟踪。原生任务目标缓冲保留求解器所需的批次和种子布局，轨迹请求的原生末端目标位姿跟踪与关节目标位置跟踪均处于关闭状态。

普通 IK 工厂在原 `tool_pose_cfg` 上使用 `use_lie_group=True`、六轴权重全 1，并将旋转权重乘 4。Lie 核使用半角向量，该配置恢复原全转角二次能量及其关节导数。独立的 metrics/convergence 配置保留 axis-angle，继续以真实转角和原阈值判断原生收敛。普通参考目标由 `reference_to_link_pose` 转为原生工具连杆目标；`nominal_position_error_m` 与 `nominal_rotation_error_rad` 记录该原生目标的误差。

`reference_seeds` 首先为固定三次基产生自由系数提议。设自由系数数为 `M`，第 `j` 个查询的归一化时间为 `u_j=(j+1)/(M+2)`，`j=0,…,M−1`；这是 `CubicBasis.reference_parameters` 给出的 Greville 参数，即对应自由基函数的结点平均参数。查询位姿来自同一 `BoundToolReference`，查询物理时间为 `u_j T`。每个时刻执行一次原生完整姿态 IK，保存全部返回数值状态、原生成功标志、原位姿能量与静态任务/模型证据。

`native_pose_ik_success`、`task_satisfied` 和 `model_valid` 分别说明原生参考收敛、任务点检查和模型检查；`static_model_query` 记录静态查询是否实际运行。有限、处于原关节位置域内且具有有限位姿能量的状态属于数值提议。静态证据的 `_static_qualification` 依次区分明确任务失败、明确模型失败、任务结论未建立和模型结论未建立。明确失败来自已完成查询中的有限正违反量；未知区域、数值边界和未运行查询保留原来源。功能事件候选先比较这些原允许集合与模型结论，再比较关节连续性。

参考候选的数值选择先比较原生收敛，再比较连续性。`rank_ik_candidates(...,query_kind="reference")` 在成功标志相同的候选之间先比较到前态集合的最小关节距离，再以静态证据、原几何质量和索引处理并列。`select_ik_tracking_candidate` 每次只选一个下一查询的显式 seed：成功行依次比较关节距离和位姿能量；失败行依次比较位姿能量和关节距离，静态证据位于这些数值量之后。位姿能量为 `(w_p/2)e_p²+w_R e_R²`，`e_p,e_R` 分别为原生位置误差和真实转角，`w_p,w_R` 为补偿前原权重。其余种子由原生 `SeedManager` 在原容量内生成；没有合法提议时继续使用此前数值状态及其原来源。

`select_ik_chains` 从按物理时间排序的合法节点选择完整数值链。动态规划依次最小化未收敛节点数、失败节点位姿能量 `E_f`、归一化关节速度能量 `J_v` 和成功节点位姿能量 `E_s`，累计静态证据只处理数值指标相同的路径。`J_v` 为节点连线速度平方的物理时间积分，以原速度限值和查询总时间/关节数归一化。每个末节点保留请求所需的历史数，稳定索引处理并列；原位置域限定节点，真实连续速度及更高阶边界由曲线域处理。

链中已收敛状态记为 `SOLVED`，合法未收敛状态记为 `APPROXIMATE_PROPOSAL`；没有合法节点的时刻由入口与已有节点插值，记为 `INTERPOLATED_PROPOSAL`。`_select_reference_chains` 保留每行的生产查询及候选索引。所有链状态共同提出自由系数，查询状态不构成必须插值通过的关节目标。未填满的轨迹种子槽以测得 `q0` 重复作为数值提议，`CubicJointDomain.initialize` 将这些提议放入固定 `q0/dq0`、时钟及原四类边界定义的物理域。

`_refine_reference_seeds` 对该数值初值执行一次完整工具目标优化。它先调用原生 `reset_cuda_graph`，再经 `_prepare_goal_buffer/_update_rollout_params` 绑定同一请求、入口、显式时钟和自由终端声明。随后重置功能乘子，将几何范围设为 `robot`，保存数值初值快照 `coarse`。目标包含完整工具参考、原关节成本、过程关系、完整有序轨迹上的原功能允许集合，以及共享官方机器人自身与名义手部条件；任务观测场状态为 `task_scene_constraints="not_evaluated"`。`reinitialize → optimize` 使用第 1 节分配的更新额度，返回真实优化系数并保存 `refined`。

`_initialization_snapshot` 调用该 optimizer rollout 的 `compute_metrics_from_action`，输入为 `[K,M,D]` 的实际动作，`K` 是轨迹种子数，`D` 是活动关节数。`metrics_constraint_manager.optimizer_evidence` 取得同次任务、关节及机器人范围证据；`fixed_objective` 汇总普通成本，`optimization_objective` 保存全部优化成本与约束罚项之和。`reference_times_s/reference_point_costs` 保存普通网格的实际时间与加权参考贡献。每行系数经 `_controls_curve` 导出原断点多项式，并完成位置、速度、加速度和分段 jerk 的连续检查。该 metrics 入口按种子组织入口索引，线搜索的 `evaluate_action` 则按种子与粒子展开。

初始化结束时恢复 `full` 几何范围和总迭代配置，清理原生图记录。完整主求解通过 `_solve_impl` 重新绑定目标，建立该目标的数值状态、功能乘子和候选保存槽。`status="completed"` 表示初始化计算完成，完整交付资格由第 8 节的验收产生。调用者提供 `JointTrajectory` 时，原曲线先独立完成完整验收；改善路径按对应技能进度采样原关节曲线，形成自由系数提议，经同一物理域初始化后进入完整工具目标优化。原曲线及其时钟与该数值改善路径各自保存。

`_build_goal_ik_candidates` 由 `ompl_seed` 调用，为搜索构造事件边界的数值提议。它将当前窗口内相同精确进度的 `pose_events`、`coordinate_events` 和 `supports` 合为一次事件查询。`bind_ik_goals` 绑定其原目标身份，调用 cuRobo 的 `disable_tool_pose_tracking`，令该次 IK 使用 `PlanningProblem.goal_residual(normalized=True)` 的连续允许集合。`ProcessCostManager` 同时加入点上可评价的过程条件，排除需要实际速度的停止项；优化模式计算归一化超差平方，指标模式保留原硬超差。原生 `GoalToolPose` 缓冲仍接收名义位姿，实际目标由已绑定要求计算。每次事件求解结束后，`finally` 清除事件身份并恢复参考 IK 的完整姿态跟踪。

名义材料目标到原生连杆目标的换算仍由 `_functional_goal` 完成：`T_W_E=T_W_C A⁻¹`，`T_B_L=T_W_B⁻¹ T_W_E T_L_E⁻¹`。其中 W 是规划参考坐标，E 是 TCP，C 是材料坐标，B 是机器人基座，L 是原生工具连杆，`T_L_E` 是工具标定。事件 IK 直接在关节状态上优化任务允许集合，完整旋转、倾斜及有界或自由 yaw 均由同一误差接口定义。

`_rank_ik_candidates` 对返回的事件状态计算 FK、`residuals(direction_mode="point", point_goal_ids=...)`、实际场景与原关节条件。停止项在点筛选中排除，完整路径继续评价其真实速度。报告将事件求解标为 `candidate_source="event_allowed_set_ik"`，以 `allowed_set_ik_success` 保存该次原生结果，以独立重算的 `task_satisfied/model_valid` 保存任务和场景结论。参考 IK 的失败行保留原失败状态，其静态查询记录为 `not_run_native_reference_ik_failed`。

`initial_state_diagnostics` 使用已绑定机器人状态、同一任务和场景计算请求入口的具体残差。输出区分任务不满足、机器人模型碰撞、任务场已确认体积交叠及移动实体覆盖条件，并保留数值未决报告。任务场配对的未决不产生碰撞正残差；原生自身数值边界、任务接触、初始支撑及覆盖仍按各自条件计算。入口是否继续由实际硬分项决定，报告中的每个非 clear 状态并不自动成为拒绝理由。

## 5. 同一任务声明下的分阶段约束证据

`_process_manager_type` 为请求创建 `ProcessCostManager` 及对应配置类型。`bind_request_data` 绑定任务、场景、进度映射和功能约束；`set_dense_counts` 区分普通与密集样本的有效数量；`bind_scene_evidence` 连接同一次前向的场景证据。原生 `cost_cfg` 使用 `reference_objective` 模式，计算连续参考与原生关节正则项；`constraint_cfg` 中的原生 `cspace` 检查实际状态的关节界，`trajectory_optimizer` 模式计算任务和功能增广项，并按 `set_geometry_scope("robot"/"full")` 计算同次 `process_scene`。初始及普通梯度入口共享该范围：`robot` 计算官方机器人自身与手部，`full` 继续计算任务观测场并使用块内局部几何模型。`trajectory_metrics` 始终计算完整原硬要求，并配置相同普通成本供固定评分调用。

### 5.1 连续参考与固定时间平均

`reference_tracking.NativeReferenceCost` 读取该候选实际时钟上的 `BoundToolReference`，沿普通有效物理时刻采样完整工具参考。设 B 为机器人基坐标系，E 为标定 TCP，`p_E(q(t_i))` 为候选在时刻 `t_i` 的 TCP 原点，`p_E*(t_i)` 为相同时刻的绑定参考；二者均由原标定表达在 B 中。绑定对象已持有该候选的进度映射、时长和机器人入口边界，其构造见[参考组件](01-inputs-and-task-model.md#21-实际物理时钟上的工具参考)。`θ_i` 为两个 TCP 姿态之间的 SO(3) 最小转角。参考目标的离散时间平均为

```text
J_reference = Σ_i α_i [ (w_p/2) ‖p_E(q(t_i)) − p_E*(t_i)‖² + w_R θ_i² ]
```

`physical_time_mean_weights` 用真实物理时间的梯形公式计算 `α_i`，在有效前缀上归一化为和 1；填充样本权重为零。确切事件与碰撞见证保留各自的硬要求查询作用。`w_p,w_R` 来自原轨迹配置的 `tool_pose_cfg.weight`。`ToolPoseCost` 采用等权 Lie 旋转三轴、零收敛带和 `use_grad_input=True`；其旋转核输出 `θ²/4`，所以内部整体旋转权重取 `4w_R`，得到上述前向尺度。

`bind(sample_times,valid_count,bound_reference)` 在请求或网格变化时预先生成参考目标和积分权重；普通优化调用复用这些设备记录。参考 IK 的 `_goal` 使用同一个 `bound_reference.sample(times_s)`，再将 TCP 目标转换为原生连杆目标。保存候选的 `_reference_for_candidate` 按其自身物理时钟构造同规则绑定，供普通评分和完整指标上下文使用。原生成本调用将候选和时间展平成局部批次，显式目标索引保持每个时刻与其参考一一对应。`forward` 直接消费已有原生 FK 工具连杆位姿。`NativeRigidToolTransform` 将固定 `T_L_E` 合成到该连杆 L，位置导数包含 TCP 偏置产生的力矩项；姿态代价对基座系中无穷小转角的三维导数，经 cuRobo 的四元数导数编码返回同一 FK。该编码是已安装原生核的反向协议，坐标和固定工具标定与任务模型一致。

`freeze_cspace_time_weights` 从原普通成本模板的副本出发，在本次 `native_dt` 上一次吸收原生 STATE 成本的时间缩放因子，并设置 `use_grad_input=True`。`FixedTimeStateCSpaceCost` 继续调用原生前向，只为成本建立固定 `dt` 的 `JointState` 视图，因为原生能量项仍显式包含 `dt`。位置、物理速度、加速度、jerk、力矩和实际查询时间保持原值。请求重绑从原模板重新构造成本。普通关节成本与参考成本共同乘以 `α_i`，所以优化网格、密集网格和最终评分使用同一组成本系数。

### 5.2 任务与几何的同刻硬证据

轨迹状态先通过 cuRobo 纯运动学 FK 得到工具位姿。普通时刻与确切功能锚点按物理时间排列，`PlanningProblem.residuals` 与 `functional_inequalities` 消费相同有序状态、进度和时间，保留机构绕转历史、过程方向、停止条件及原事件允许集合。任务功能坐标、材料关系和工具标定从这些计算开始参与。

在两种几何范围中，`DifferentiableRobotKinematics` 都使用本时刻完整具名关节与名义手指坐标求各连杆位姿；`full` 范围继续由 `CollisionOwner.pose` 求任务实体位姿。`OnlinePlanningScene.components` 返回相应范围的硬分项。`_scene_boundary_evaluate` 将 exact 样本及同一物理状态的关系左右侧一起查询，保留释放见证。`query_role` 决定聚合返回硬分项还是连续优化项；两种角色同时产生同一范围的实际硬证据。

候选硬条件的有效位来自普通网格的实际样本数与确切锚点的有效标记。两组标记拼接后，按轨迹状态使用的同一时间排序重排为 `active`；硬条件归约只保留对应的有效样本。固定容量中的填充位置仍保留其无效身份。

场景计算以 `active.any(0)` 选取整批候选中至少有一个有效样本的时间列，再将这些列上的位姿、关节、进度和物理时间交给 `_scene_boundary_evaluate`。选择保持原时间顺序，因而各候选的确切事件和释放状态仍在同一矩形批次中求值。返回的场景残差和候选证据按原列索引回填；各行仍由自身 `active` 掩码归约。不同候选的有效样本数不同时，为其他候选保留的列仍可能包含本行无效位置。

`PlanningProblem.optimizer_preferences` 在 `feasibility/quality` 两种目标配置中均汇总过程等式坐标和机构方向偏好；`quality` 阶段另加入支持声明中非硬分量的归一化超差平方，以及功能位姿、机构坐标目标中心偏好。当前主求解使用 `feasibility` 阶段，其支持计算由声明的硬分量进入功能增广拉格朗日；连续示教参考在相同阶段由普通成本持续优化，`measurements` 提供其解释性误差。场景优化项由本块冻结的 `LocalPairModel/LocalContactModel`、原生机器人距离、初始支撑高度和已有连续控制项组成；原始分类与未决状态另存为硬证据和查询报告。

任务计算按原声明确定硬条件。`trajectory_optimizer` 与 `trajectory_metrics` 读取同一 `PlanningProblem.residuals`；硬功能条件由增广拉格朗日管理，其余过程项进入原过程代价。原生 `cspace` 约束包含位置、速度、加速度、jerk 及状态中已提供的力矩值。`diagnostics.joint_state_computation.effort` 根据实际 transition 的 `compute_inverse_dynamics` 分别记录 `inverse_dynamics` 或 `not_computed`。当前机器人配置关闭动力学计算，零力矩缓冲对应未计算通道。原生成功标志和普通/密集约束描述其各次实际采样，保存在 `optimization_diagnostics`；保存曲线的正式资格由第 8 节统一入口形成。

`optimization_constraints` 返回功能条件管理器，`query_anchor_parameters` 返回当前确切查询参数，`add_collision_witnesses` 将密集查询发现的有限冲突时刻加入同一锚点容量。`query_diagnostics` 保存本次实际输入布局和场景查询信息。同一次计算的张量身份将候选保存依据与其产生的代价连接。

## 6. 功能约束、线搜索与候选保留

`FunctionalAugmentedLagrangian` 管理按种子独立保存的乘子 `λ≥0` 与罚参数 `ρ>0`。输入 `g_k` 为功能要求的有符号不等式，`g_k≤0` 表示满足。`terms` 为每项返回 `relu(λ_k+ρg_k)²/(2ρ)`。`update` 在既有内迭代块之间使用该种子当前实际探索点的残差更新乘子；初始 `ρ=2`，必要时乘 10，上限为 `10^6`，改善率判据为 0.5。`snapshot/report` 导出每个种子的乘子、罚参数、更新次数与残差度量。

`ProcessCostManager.optimizer_evidence` 返回 `(feasible, signed, fixed_objective)`。`feasible` 合并同次计算的原过程残差、全部硬功能不等式、其他原生硬约束及当前几何范围的原始硬分项。每项须为有限数且不大于零。`signed` 保存各硬功能条件的实际归一化值。`fixed_objective` 汇总普通成本集合中的连续参考与原生关节项。功能增广项、局部场景目标以及约束管理器中的任务和控制偏好各自参与求解，其值不进入该固定排名。AL 和几何张量身份将这些证据与同一次前向计算绑定；初始化的 `geometry_scope="robot"` 明确其几何范围。

`PhysicalGradientCore._cost_and_gradient_with_guard` 执行原生前向、代价归约和反向传播，并在实际边界检查请求期限。`_begin_functional_solve` 按本次种子布局重置乘子和保留候选；完整场景阶段另调用 `_refresh_local_geometry`。该接口把每个实际起始动作按 seed→particle 顺序复制，通过常规 rollout 捕获一次时间、关系与许可布局。cuRobo 从同一优化配置构造初始和常规 rollout，FUNCTO 为它们绑定相同三次曲线时钟与过程管理器，因此两种梯度入口读取同一 `LocalGeometryModels`。捕获值只建立记录，随后正式梯度读取冻结模型。

`_finish_functional_block` 在已有块之间评价当前 `exploration_action`，从同一次轨迹计算读取硬证据、固定目标和功能残差。完整场景同时通过 `_refresh_local_geometry` 捕获观测场局部模型；初始化计算同一任务、关节及官方机器人自身和手部条件。几何与功能乘子变化的种子掩码取并集。`_restart_changed_objective` 为这些种子更新动作、成本、梯度、原生最佳状态和物理方向，保留累计迭代数。存在局部几何时，梯度重算经 `repeated_base_evaluation()` 共享数值几何记录，各粒子仍从自身张量计算导数。`abort_capture` 在中断时撤销未完成替换，异常进入请求生命周期。

`PhysicalGradientCore` 继承现成 `GradientOptCore`。设当前自由系数为 `c`，完整当前目标为 `F(c)`，其梯度为 `g=∇F(c)`；目标包括本块固定功能乘子和局部几何。设最大原生线搜索倍率为 `α_max>0`，原生增量范围为逐系数正值 `L`，原连续物理域为 `D`。普通方向 `d` 来自单位度量投影：

```text
minimize_d   1/2 ‖d + g‖²
subject to   c + α_max d ∈ D,
             |α_max d_i| ≤ L_i.
```

`D` 保持测得入口、物理时钟、位置、速度、加速度和 jerk 原边界。目标与这些边界按关节可分，各关节的全部自由系数共同组成一个串行 QP。`_physical_step_direction` 消费实际探索点和完整梯度；`_run_initial_evaluation` 在原生首次评价后设置同一方向，普通回调和目标更新后的重启也调用该接口。每次 QP 前后与设备前向/反向的实际边界检查同一请求期限。

`PhysicalLineSearch._prepare_search_points` 使用物理方向和原倍率生成实际 dtype 的试探。原生 `scale_action` 对整体方向的幅度归一化由方向子问题中的原位移范围取代，试探生成前仍用同一物理域 limiter 处理实际乘加舍入。`search` 调用现成 `ArmijoLineSearchStrategy`，对尚未找到下降的种子将方向缩半并再次使用相同选择器；已经选中的种子保留匹配的动作、成本和梯度。AL 参数及局部几何在整次搜索中固定。原 dtype 中所有正步均与基点相同时，该种子保留本次基点。

一次缩半属于同一迭代内的目标评价，原生最佳状态在该次搜索结束后更新一次。`_opt_step` 记录迭代、有效位移下降、线搜索批次数与缩半次数；`solve_diagnostics` 按种子保存方向调用数、实际 `qp_calls/lp_calls`、极值计算、约束装配、原生 QP/LP 和数组传输时段，以及最后方向报告。每个原生 QP/LP 调用前后检查同一请求，调用次数和求解时段通过调用者持有的记录保留，随后期限检查中止时仍可读取已经执行的工作。有限方向构造或线搜索结果描述本次数值计算，完整任务资格仍由原候选验收确定。每次完成的前向计算继续维护采样合格候选，未被选为探索点的试探也可凭同次原硬证据更新候选槽。

每个种子记录中的 `shared_batch_work` 是同一个 optimizer 的共享批量计算，汇总时每个 optimizer 取一次。其 `cost_gradient_calls/geometry_capture_calls/base_evidence_calls` 分别计数代价与梯度、局部几何捕获和基点硬证据计算；相应主机时段与后续显式设备流等待时段分别记录。主机调用时段包含调用内部已经发生的同步工作，显式等待只测量调用返回后的等待，两者不重叠；这些时段表示实际调用的墙钟时间。初始化在 `reference_initialization.optimizer_runtime` 保存各核心记录，主求解在 `last_optimizer_runtime` 保存最后一次调用记录；两者均在 `finally` 读取已完成工作。

`_retain_sampled_candidate` 为每个种子维护一个候选槽。动作与排序目标为有限数且本阶段的 `feasible=True` 时，按 `fixed_objective` 选择较小值。由于该值独立于块内局部几何和功能乘子，来自同一阶段不同块的候选可以直接比较。未保留到这类候选的种子返回原生优化结果，其残差由实际输出报告说明。完整主求解重新建立候选槽；普通与密集轨迹指标、连续关节域及最终候选验收继续决定交付。

## 7. 有限恢复与 OMPL 搜索

`plan_trajectory` 先完整验收调用者提供的原关节曲线。对 `reference_seeds` 的实际返回系数，`known_initialization_failures` 比较终态 `refined` 中的系数、具名关节、测得入口 `q0/dq0`、原多项式时刻、时长及进度映射。仅当 `joint_reference` 为空，且每个返回种子均与该终态记录相同并存在至少一个有限正的原始功能不等式时，省略初始化候选的完整扫描，直接进入主优化；`reference_candidate.qualification="skipped_known_functional_failure"` 保存对应种子和原违约值。其余候选通过 `evaluate_reference → select_candidate` 完成原完整评价。

合格初值完成执行绑定后进入独立最佳候选记录，随后使用首次额度的剩余更新数进行完整主求解。显式未知空间策略要求补充观察时保留相应结论。新候选完成全部验收、执行绑定且固定评分更低后替换已有合格记录。初始化调度依据与完整候选验收记录各自保存。

主求解后仍无合格候选时，恢复类型由所选保存曲线的正式碰撞或观测覆盖残差决定。对应原生候选的密集碰撞见证可由 `promote_collision_witnesses` 增加下一次求解的查询；这些见证保留原生采样身份。没有可提升见证且存在适用条件及剩余搜索预算时，`ompl_seed` 提供一次搜索种子。两个恢复来源互斥，求解次数、验证次数、查询数和时间各受原额度限制。改善工作耗尽预算时，已有在预算内完成资格的最佳记录仍可交付；取消、外部截止时间及其他异常保持各自类型。

`ompl_seed` 按功能目标、过程边界、接触切换及停止边界组织阶段。`_ompl_stage` 的状态为 `(q,u)`；`u` 单调前进并给出固定物理时间。过程等式通过 `_independent_projection` 选择局部独立行，在需要时交给 OMPL `ProjectedStateSpace`；允许区域中的不等式继续作为有效性条件。投影改变关节坐标，保留时间坐标。

`ompl_seed` 先以 `_build_goal_ik_candidates(..., check_search=...)` 生成实际搜索会读取的事件候选；这些 IK 调用前后检查同一搜索期限与请求期限。搜索检查直接连接后使用 RRTConnect。`checked_edge` 检查实际连接路径中的中间状态和跨过的事件边界，并调用同一观测场景和任务条件。当前阶段最多允许 `min(max_validation_points,8192)` 个状态查询，候选生成、投影和状态查询共同消耗搜索期限。功能终点候选由满足相关任务与模型条件的 IK 行提供；释放对象的姿态来自同一候选 FK。返回路径拟合到原有三次基，继续经过完整验收。

## 8. 从求解结果到候选曲线

`_observed_native_result` 对实际动作执行普通与密集原生 metrics，`evaluate_reference` 形成初始化返回动作的原生结果，`optimize` 形成完整场景优化结果。`native_diagnostics` 从这些原结果读取成功标志、约束、普通成本及碰撞/观测分类。它们描述原生网格上的实际计算；正式资格的对象是随后保存的 `JointTrajectory`。

`_trajectory_candidate` 通过原生 gather 取得指定 seed 的动作、状态、时钟及有限恢复见证，单 seed 与多 seed 均拥有独立必要张量。完整状态可含锁定关节，动作系数仅含活动关节，二者分别按自身具名布局保存。`select_candidate` 在任何保存曲线评分或指标调用之前，先取得全部原生候选与诊断，使后续复用常驻缓冲时这些输入保持完整。

`_native_curve` 读取所选结果的实际 `solution`，交给 `_controls_curve`；后者调用 `CubicBasis.hermite_nodes`，以原断点 binary64 节点构造 `JointTrajectory`，复制进度、事件、条件、手指曲线和已满足依赖。`_initialization_snapshot` 从每个种子的实际动作进入同一导出函数。`reference_initialization.coarse/refined` 分别记录物理域内的数值初值与完整工具优化输出的系数、多项式及同次证据；初始化几何范围为 `robot`，任务场状态为 `not_evaluated`。中断时保留已完成记录；完整候选尚未产生时，最后完成初始化阶段的第一条曲线可成为诊断轨迹。

`_candidate_objective` 在请求固定的 `(knots+4)×interpolation_steps+1` 个归一化节点上计算普通成本。每条曲线使用自身时长、进度映射、工具标定及物理导数，`fixed_objective/objective_components` 因而具有共同排名定义。有限原生曲线按该评分和稳定索引排列，已完成的评分直接传入正式评价。

`evaluate_candidate(trajectory,record_work=...,objective=...)` 对每条保存曲线统一调用 `_candidate_metrics`，再检查 `_candidate_domain_validation` 和 `_candidate_geometry`。实际样本由 `_candidate_sample_times` 在每个原多项式区间内按最大间隔生成，确切任务锚点从本候选物理时钟计算；位置、速度、加速度及 jerk 均来自保存多项式。临时 `_candidate_query` 向常驻指标管理器提供实际时间、场景、进度及锚点位置/速度，`finally` 清空该上下文和参考网格缓存。后续原生求解通过原 `_solve_impl` 重新绑定自身目标。

正式报告中的 `native_success/native_metrics` 表示原生指标 API 对该保存曲线的采样结果；`optimization_diagnostics` 单独保存提议产生时的原普通/密集结果。正式残差由实际保存曲线采样、连续关节界及区间场景查询组成。`select_candidate` 逐条评价，返回固定评分最小的完整合格曲线；全部已评价曲线未通过时，按正式残差严重度和稳定索引选取诊断结果。报告分别保存 `validation_query_points` 和 `evaluated_candidate_count`。

`record_validation_work` 是 `plan_trajectory` 中现有请求账本的回调。每次正式评价开始前检查剩余验证次数，再登记本次尝试；每批原生样本和锚点、每批区间几何查询完成后立即累计实际点数，再检查请求期限。后续候选中断时此前完成的工作已在账本中。验证次数上限为 `maximum_trajectory_solves × trajectory_seeds`，总查询额度来自同一预算；每条曲线的区间查询只使用其 `max_validation_points` 中扣除实际采样后的余额。

`export` 返回已验收的 `delivery_candidate`，新增查询数为零。`qualify_native` 随后通过 `qualify_candidate → execution_binding.bind_candidate` 完成曲线、问题、场景与正式验收的绑定。独立最佳记录包含曲线副本、评分、验收、事件时间检查及资格完成时间；记录完成后的预算检查通过，它才可替换已有记录。`deliver_incumbent` 恢复选中记录，最终账本另保存整次请求消耗。完整交接见[验收组件](05-acceptance-and-execution.md)。

`reinitialize` 完成当前目标的模型捕获、初始前向、功能参数初始化和候选保存后，`optimize` 消费同一个原生迭代状态。块结束时在实际探索点同时捕获下一块模型与原硬证据，候选排名读取固定普通目标；模型或乘子改变后重新计算梯度、物理方向和该目标的数值状态。
