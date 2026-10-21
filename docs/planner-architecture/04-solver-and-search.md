# 固定时钟轨迹求解与有限搜索

代码基线：`IMPLEMENTATION_COMMIT_PENDING`（当前工作区实现，提交时替换为实际代码提交）。本章讨论 `planning.py` 如何将 `PlanningProblem`、`OnlinePlanningScene` 和机器人模型联合转化为关节曲线。`q` 表示具名机械臂关节坐标，`dq` 表示其每秒变化率。问题入口按 `problem.joint_names` 排列，原生求解按 `_CuroboPlanner.joint_names` 排列，保存曲线按其 `trajectory.joint_names` 排列；各边界通过名称建立列映射。`s` 表示技能进度，`u=t/T` 表示总时长为 `T` 的轨迹归一化时间。技能进度 `s` 与归一化时间 `u` 通过请求内固定的 `_ProgressMap` 联系。

## 1. 请求生命周期与资源分配

`plan_trajectory(problem, scene, robot, *, robot_validator, ...)` 是一次请求的完整入口。输入任务已经具有参考、事件身份、目标与机器人入口状态；场景已经具有观测几何、物体身份、手部模型及接触关系；`robot_validator` 连接独立机器人模型检查；`robot` 指向 cuRobo 配置。返回 `PlanningResult`，同时保存分阶段耗时、累计规划工作记录、候选残差与类型化失败原因。

`MotionBackend` 在多次请求之间复用原生求解器与固定容量缓冲。`bind_request` 创建或重绑定 `_CuroboPlanner`；`plan` 将本实例传给同一 `plan_trajectory`；`_claim/_release` 保证一个实例同一时刻由一个请求使用；`close` 释放所持资源。直接调用 `plan_trajectory` 时由该函数创建并关闭本次拥有的后端。请求为任务和场景建立私有副本，事件时钟与候选释放位姿在该副本中绑定。

`PlanningBudget` 同时规定数值维数、工作次数和资源范围。默认 IK 种子为 8、轨迹种子为 2、`knots=8`、原生细分为 4、IK 迭代为 40、轨迹迭代为 100、最多完整轨迹求解为 2、搜索预算为 3 s、请求预算为 120 s。各项含义不同：种子数控制并行候选，`max_task_anchors` 控制确切事件与补充见证容量，`max_validation_points` 控制完整验收的有限查询数量，主机 RSS 与设备占用限制控制实际资源。默认硬件预算延续本地半内存、半显存约束。

`_planning_workspace_bytes` 从机器人自由度、连杆数、样本数和场景工作区推算本次分配需求；`_require_planning_host_memory` 与 `_require_planning_memory` 在分配前检查存储；`resource_admission` 对源几何与保存记录的实际分配继续承担相应检查。`_request_work_ledger` 和 `_check_work_budget` 记录并检查请求累计时间及求解次数。续规划从原账本继续记账，失败诊断保留已完成的工作量。

## 2. 任务进度、事件时间与可执行区间

`_initial_timing` 使用源技能时间建立初始进度映射，`_progress_for_cubic_boundaries` 将阶段、确切锚点、停止进度与三次曲线的物理区间对应。`PlanningBudget.interpolation_dt` 在请求入口表示最大采样间隔。`_derived_spline_grid` 按原结点间隔选择满足该上限的二次幂细分数，并用 `_fixed_spline_grid` 核对原生 float32 与保存 binary64 时钟的结点间隔、总时长完全一致。容量检查使用派生出的真实样本数；绑定后的预算保存实际间隔，后续普通采样、密集采样、手指和导出共同消费该时钟。`_bind_event_clock` 生成事件物理时间、发生次数及命令记录；`_validate_bound_event_times` 检查这些时间与原事件约束、已有实际历史的一致性。

`PlanningProblem.bound_motion_block` 依据原始依赖关系划定当前运动块。这里运动块指由一次规划和执行绑定管理的轨迹区间。函数复制问题，将 `end_progress`、事件、点目标与源时间映射截取到最早尚未满足的显式效果依赖；后续义务保存在原请求账本中。`JointTrajectory.next_verification` 返回当前曲线中的首个未满足依赖事件，`executable_until` 给出对应可执行终点。普通事件和参考阶段边界不自动添加停止。当前求解曲线末尾的停止区间由本章三次基产生，取得相应效果证据后的下一段通过续规划契约重新绑定。

`_prepare_continuation` 消费原请求的事件身份、执行进度、实际入口反馈、未完成义务、已执行事件和工作账本；`_continuation_timing` 保留原请求时钟中的剩余关系。`_obligation_contract` 保存尚需满足的任务义务，`_restore_original_direction_origins` 恢复机构方向要求的原始坐标起点。每次事件由 `(skill_id,event_id,occurrence)` 标识；同名重试具有不同的 occurrence。

## 3. 共享三次曲线与物理关节域

`CubicBasis(controls, knot_dt)` 使用 SciPy 三次 B 样条建立动作系数到位置、速度、加速度和加速度变化率 `jerk` 的线性映射；`jerk` 是关节位置对物理时间的三阶导数。令自由控制系数数量为 `M`、结点间隔为 `h`，完整系数含 `M+5` 行、有效曲线含 `M+2` 个区间。当前规划器把 `M=budget.knots+2` 交给该基，因而物理区间数为 `budget.knots+4`。

入口位置 `q0` 和速度 `dq0` 作为实测边界消去两个系数。若第一个自由位置系数为 `x0`，则前两个系数为 `c0=x0−2h·dq0`、`c1=1.5q0+0.5h·dq0−0.5x0`；末端重复最后一个自由系数形成停止区间。实现以 `x−q0` 做相对运算，位置查询为 `q=q0+W(x−q0)+Vdq0`。`weights` 和 `weights_at_knot_times` 返回相应阶数的 `W,V`，导数按物理秒缩放。

`CubicTrajectoryStep.sample` 消费 `[batch,M,dof]` 的动作、每个候选的入口索引、原生状态缓冲和有效样本数量，返回同一曲线的 `q,dq,ddq,jerk,dt`；其中 `batch` 是候选数，`dof` 是活动关节数。本文中的 rollout 指在给定时刻计算候选状态、运动学和代价的前向计算。普通 rollout 与密集导出共用此函数，Torch 对实际前向算术求导。`forward` 实现 cuRobo 的状态转移签名，核对固定时钟及自由终端声明。`cubic_transition_config_type` 通过实例配置工厂把该状态转移装配到 cuRobo 的 FK、状态缓冲和代价计算中。

`JointPositionDomain` 表示单个 IK 状态的具名关节闭区间。`feasible/validate` 检查有限数及位置边界，`limit_step_tensor` 为一组线搜索倍率限制步长，保证被评估的关节状态处于此域。`CubicJointDomain` 则对完整三次曲线建立位置域：`coefficients` 形成实际多项式系数，`extrema` 求各段内部驻点与端点，`feasible` 判断整条曲线的位置范围；`initialize` 构造域内初值，`fit_reference` 把离散关节目标拟合到相同基，`recovery_step` 求停滞时仍处在域内的候选方向。

`physical_joint_qp.py` 的 `CubicAffineMap` 为上述计算提供 CPU 仿射表达 `q(u)=B(u)c+b(u)`；`c` 是自由系数，`b` 由入口与固定时钟确定。`fit_reference(..., event_progress, event_positions)` 保留全部参考样本为最小二乘目标，把所选事件状态 `q_e` 写为 `B(u_e)c=q_e−b(u_e)`。`fit_cubic_reference` 使用已有 quadprog 的 `meq` 等式行，逐关节联合求解这些等式和位置、速度不等式；解析极值交换向 QP 增加实际违反的连续约束。

输出转换为原生浮点 dtype 后再次检查完整位置和速度域。若舍入产生越界，`_fit_anchor` 构造满足同一事件等式的可行锚点，再沿二者之间的系数线段恢复原生 dtype 可行性；锚点的数值内移只作用于事件等式仍允许改变的方向。报告保存实际事件关节插值误差，后续 FK 与任务检查消费原允许集合。事件等式只属于初始化，`CubicJointDomain` 的主优化域仍只包含入口和原关节位置约束。`recover_position_step` 在当前动作、梯度、步长范围及固定基上构造优化恢复方向。

## 4. cuRobo 装配、参考 IK 与事件 IK

`_CuroboPlanner._initialize` 先核对机器人模型身份、工具、活动关节及场景要求，然后建立时钟、绑定手指曲线、检查工作区并创建 IK 与轨迹求解器。原生配置来自 `ik/lbfgs_ik.yml` 和 `trajopt/lbfgs_bspline_trajopt.yml`。轨迹优化采用 L-BFGS，关节状态由本章三次基产生；主生产请求使用 eager 执行，即按当前程序调用计算张量运算。`bind_solver_components` 将本项目的物理域线搜索、优化核心与目标缓冲管理装配到对应求解器实例。

`reference_seeds` 在三次基参考参数处查询完整工具参考，调用原生完整姿态 IK，再经 `_rank_ik_candidates` 排序。报告分别记录 `native_pose_ik_success`、`task_satisfied` 和 `model_valid`。`select_ik_chains(query_kind="reference")` 把原生 IK 成功状态作为参考锚点，以测得入口和这些锚点补齐缺失参考样本，再拟合初始曲线；参考样本属于软目标。全部参考 IK 不成功时，现有测得入口状态提供独立初值。

`_build_goal_ik_candidates` 将当前窗口内相同精确进度的 `pose_events`、`coordinate_events` 和 `supports` 合为一次事件查询。`bind_ik_goals` 绑定其原目标身份，调用 cuRobo 的 `disable_tool_pose_tracking`，令该次 IK 使用 `PlanningProblem.goal_residual(normalized=True)` 的连续允许集合。`ProcessCostManager` 同时加入点上可评价的过程条件，排除需要实际速度的停止项；优化模式计算归一化超差平方，指标模式保留原硬超差。原生 `GoalToolPose` 缓冲仍接收名义位姿，实际目标由已绑定要求计算。每次事件求解结束后，`finally` 清除事件身份并恢复参考 IK 的完整姿态跟踪。

名义材料目标到原生连杆目标的换算仍由 `_functional_goal` 完成：`T_W_E=T_W_C A⁻¹`，`T_B_L=T_W_B⁻¹ T_W_E T_L_E⁻¹`。其中 W 是规划参考坐标，E 是 TCP，C 是材料坐标，B 是机器人基座，L 是原生工具连杆，`T_L_E` 是工具标定。事件 IK 直接在关节状态上优化任务允许集合，完整旋转、倾斜及有界或自由 yaw 均由同一误差接口定义。

`_rank_ik_candidates` 对返回的事件状态计算 FK、`residuals(direction_mode="point", point_goal_ids=...)`、实际场景与原关节条件。停止项在点筛选中排除，完整路径继续评价其真实速度。报告将事件求解标为 `candidate_source="event_allowed_set_ik"`，以 `allowed_set_ik_success` 保存该次原生结果，以独立重算的 `task_satisfied/model_valid` 保存任务和场景结论。参考 IK 的失败行保留原失败状态，其静态查询记录为 `not_run_native_reference_ik_failed`。

`_fit_event_seeds` 使用 `select_ik_chains(query_kind="functional")` 在每个事件的任务合格候选中选取完整序列。对相邻状态 `q_i,q_j` 和物理时间差 `Δt>0`，各关节满足 `v_min Δt≤q_j−q_i≤v_max Δt` 才能形成图边；这是固定时钟下的必要速度条件。动态规划按场景失败数量、归一化关节速度能量及既有质量数据排序，保留有限个序列。静态场景失败可保留为初始化提议，完整曲线的场景验收仍决定可执行性。

选定序列以事件等式进入本章第 3 节的 QP，全部参考样本仍作软拟合目标。缺少完整事件序列或等式拟合失败时，报告具体构造结果，主求解仍可从已经构造的参考曲线开始。有限池未能提供初值表示该次初始化覆盖不足。调用者传入的 `JointTrajectory` 首先按原多项式验收，随后才在需要优化时转为固定基初值。

`initial_state_diagnostics` 使用已绑定机器人状态、同一任务和场景计算请求入口的具体残差。输出区分任务不满足、机器人模型碰撞、任务场已确认体积交叠及移动实体覆盖条件，并保留数值未决报告。任务场配对的未决不产生碰撞正残差；原生自身数值边界、任务接触、初始支撑及覆盖仍按各自条件计算。入口是否继续由实际硬分项决定，报告中的每个非 clear 状态并不自动成为拒绝理由。

## 5. 同一前向计算中的任务与碰撞约束

`_process_manager_type` 为请求创建 `ProcessCostManager` 及对应配置类型。`bind_request_data` 绑定任务、场景、进度映射和功能约束；`set_dense_counts` 区分普通与密集样本的有效数量；`bind_scene_evidence` 连接同一次前向的场景证据；`compute_costs` 计算原生机器人项、任务项与场景项；`optimizer_evidence` 读取这一前向产生的候选分级依据。

轨迹状态先通过 cuRobo 纯运动学 FK 得到工具位姿。场景的 `DifferentiableRobotKinematics` 使用本时刻完整具名关节与名义手指坐标求各连杆位姿，`CollisionOwner.pose` 求场景实体位姿。普通时刻与确切功能锚点按物理时间排列，`PlanningProblem.residuals` 与 `OnlinePlanningScene.components` 分别产生任务和几何硬分项。`_scene_boundary_evaluate` 将 exact 样本及同一物理状态的关系左右侧一起查询，保留释放见证。`query_role` 决定场景聚合返回硬分项还是固定局部优化项；两种调用同时产生相同的实际硬证据。

候选硬条件的有效位来自普通网格的实际样本数与确切锚点的有效标记。两组标记拼接后，按轨迹状态使用的同一时间排序重排为 `active`；硬条件归约只保留对应的有效样本。固定容量中的填充位置仍保留其无效身份。

场景计算以 `active.any(0)` 选取整批候选中至少有一个有效样本的时间列，再将这些列上的位姿、关节、进度和物理时间交给 `_scene_boundary_evaluate`。选择保持原时间顺序，因而各候选的确切事件和释放状态仍在同一矩形批次中求值。返回的场景残差和候选证据按原列索引回填；各行仍由自身 `active` 掩码归约。不同候选的有效样本数不同时，为其他候选保留的列仍可能包含本行无效位置。

`PlanningProblem.optimizer_preferences` 在两阶段均汇总过程等式坐标和机构方向偏好；`quality` 阶段另加入支持声明中非硬分量的归一化超差平方，以及功能位姿、机构坐标目标中心偏好。当前主求解使用 `feasibility` 阶段，其支持计算由声明的硬分量进入功能增广拉格朗日。普通示教参考的偏离由 `measurements` 报告。场景优化项由本块冻结的 `LocalPairModel/LocalContactModel`、原生机器人距离、初始支撑高度和已有连续控制项组成；原始分类与未决状态另存为硬证据和查询报告。

任务计算按要求自身的声明确定硬条件。`trajectory_optimizer` 与 `trajectory_metrics` 均读取同一 `PlanningProblem.residuals`。硬功能条件由增广拉格朗日管理，其余过程项进入原过程代价；支持默认硬项为 body/footprint，gap/normal 按原阈值提供 `quality` 阶段偏好。原生指标的 `metrics.costs_and_constraints.constraints` 保存硬条件张量。`select_candidate` 首次按普通与密集结果判断时，要求该种子的 `result.success` 成立，且两套 constraints 的各项最大值均为有限数、不大于零。`convergence` 中的离散标识作为原生诊断保存。

`optimization_constraints` 返回功能条件管理器，`query_anchor_parameters` 返回当前确切查询参数，`add_collision_witnesses` 将密集查询发现的有限冲突时刻加入同一锚点容量。`query_diagnostics` 保存本次实际输入布局和场景查询信息。同一次计算的张量身份将候选保存依据与其产生的代价连接。

## 6. 功能约束、线搜索与候选保留

`FunctionalAugmentedLagrangian` 管理按种子独立保存的乘子 `λ≥0` 与罚参数 `ρ>0`。输入 `g_k` 为功能要求的有符号不等式，`g_k≤0` 表示满足。`terms` 为每项返回 `relu(λ_k+ρg_k)²/(2ρ)`。`update` 在既有内迭代块之间使用该种子当前实际探索点的残差更新乘子；初始 `ρ=2`，必要时乘 10，上限为 `10^6`，改善率判据为 0.5。`snapshot/report` 导出每个种子的乘子、罚参数、更新次数与残差度量。

`ProcessCostManager.optimizer_evidence` 返回 `(feasible, signed, fixed_objective)`。`feasible` 合并同次计算的原过程残差、全部硬功能不等式、原始场景硬分项及其他原生硬约束；每项须为有限数且不大于零。`signed` 保存各硬功能条件的实际归一化值。`fixed_objective` 从本次代价中移除功能增广项和整个 `process_scene` 局部几何目标，保留跨块定义不变的任务偏好及原生代价。AL 与场景张量身份将三者绑定到同一次前向计算。

`PhysicalGradientCore._cost_and_gradient_with_guard` 执行原生前向、代价归约和反向传播，并在实际边界检查请求期限。`_begin_functional_solve` 按本次种子布局重置乘子和保留候选，再调用 `_refresh_local_geometry`。该接口把每个实际起始动作按 seed→particle 顺序复制，通过常规 rollout 捕获一次时间、关系与许可布局。cuRobo 从同一优化配置构造初始和常规 rollout，FUNCTO 为它们绑定相同三次曲线时钟与过程管理器，因此两种梯度入口读取同一 `LocalGeometryModels`。捕获值只建立记录，随后正式梯度读取冻结模型。

`_finish_functional_block` 在已有块之间重算当前 `exploration_action` 的硬证据，刷新局部几何记录并更新功能乘子。两类变化的种子掩码取并集；`_restart_changed_objective` 为这些种子重算目标和梯度、清理 L-BFGS 历史并保留累计迭代数。无功能条件的请求仍可因几何模型变化重置历史。`abort_capture` 在中断时撤销未完成的替换记录，异常继续交给请求生命周期处理。完整求解次数由 `PlanningBudget` 管理。

`PhysicalLineSearch._prepare_search_points` 在前向查询前对原生步方向施加物理域限制，随后使用原生 Approximate Wolfe 条件选择步长。`_guarded_step_direction` 根据实际系数位移和梯度变化维护有限、正曲率的 L-BFGS 历史；`_guarded_opt_step` 记录每个种子的有效更新和连续停滞步数。停滞达到一个原生内迭代块时，`_recover_stalled` 使用物理域二次规划构造方向，进行最多八轮候选步评估，未通过时将方向减半，寻找严格 Armijo 下降。该种子仍未取得下降时，`_exhausted[index]` 标记其在当前目标下停止更新，并保留其候选与累计计数；功能乘子改变目标时，`_restart_changed_objective` 可清除相应标记并重建局部历史。

`_retain_sampled_candidate` 为每个种子维护一个候选槽。动作与排序目标为有限数且 `feasible=True` 时，按 `fixed_objective` 选择较小值。由于该值独立于块内局部几何和功能乘子，来自不同块的候选可以直接比较。普通与密集轨迹指标、连续关节域及最终候选验收继续决定交付。

## 7. 有限恢复与 OMPL 搜索

`plan_trajectory` 首先验收参考种子。参考满足要求时直接导出；需要优化时执行一次主求解。若密集样本发现可加入锚点的碰撞见证，则 `promote_collision_witnesses` 将其加入下一次求解；否则在存在适用的碰撞/禁止未知条件且仍有预算时，`ompl_seed` 提供一次搜索种子。整个请求最多使用预算允许的两次完整轨迹求解。

`ompl_seed` 按功能目标、过程边界、接触切换及停止边界组织阶段。`_ompl_stage` 的状态为 `(q,u)`；`u` 单调前进并给出固定物理时间。过程等式通过 `_independent_projection` 选择局部独立行，在需要时交给 OMPL `ProjectedStateSpace`；允许区域中的不等式继续作为有效性条件。投影改变关节坐标，保留时间坐标。

搜索先检查直接连接，再使用 RRTConnect。`checked_edge` 检查实际连接路径中的中间状态和跨过的事件边界，并调用同一观测场景和任务条件。当前阶段最多允许 `min(max_validation_points,8192)` 个状态查询，初始化、投影和状态查询共同消耗搜索期限。功能终点候选由满足相关任务与模型条件的 IK 行提供；释放对象的姿态来自同一候选 FK。返回路径拟合到原有三次基，继续经过完整验收。

## 8. 从求解结果到候选曲线

`_observed_native_result` 对给定动作执行普通与密集原生 metrics；`evaluate_reference` 验收初始动作；`optimize` 返回原生优化结果与计算诊断。`select_candidate` 优先选择已满足轨迹验收条件的种子，对剩余候选按验收残差和稳定索引排序；当代码重新计算已有候选时，仍使用相同任务验收定义、原始观测模型和非穿透规则。

`_native_curve` 从原生位置、速度和原始时钟导出 `JointTrajectory`。`select_candidate` 对原生任务指标合格候选调用 `evaluate_candidate`，先检查连续关节域，再以原曲线 FK、原场景距离下界和实体运动界完成区间几何计算。新增实际场景违反进入原恢复决策，未决区间保留其范围和原因。已完成的指标继续复用同一曲线和时钟身份。`export` 返回已经验收的 `delivery_candidate`，新增查询数为零。随后 `qualify_candidate` 调用 `execution_binding.bind_candidate`，把曲线、问题、场景和验收连接为正式结果，并附加运动块与累计工作记录。完整契约见[下一章](05-acceptance-and-execution.md)。

`reinitialize` 完成模型捕获、初始前向、功能参数初始化和候选保存后，`optimize` 消费同一个未消耗的原生迭代状态。块结束时先在旧模型下以 `torch.no_grad()` 计算实际探索点证据并保留候选；局部模型或乘子改变后，再计算新目标梯度及重建相应种子的历史。
