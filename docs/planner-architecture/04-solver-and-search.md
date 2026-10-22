# 固定时钟轨迹求解与有限搜索

代码基线：`e0262a5`（实际实现提交；后续仅文档提交更新引用，代码内容与此提交一致）。本章讨论 `planning.py` 如何将 `PlanningProblem`、`OnlinePlanningScene` 和机器人模型联合转化为关节曲线。`q` 表示具名机械臂关节坐标，`dq` 表示其每秒变化率。问题入口按 `problem.joint_names` 排列，原生求解按 `_CuroboPlanner.joint_names` 排列，保存曲线按其 `trajectory.joint_names` 排列；各边界通过名称建立列映射。`s` 表示技能进度，`u=t/T` 表示总时长为 `T` 的轨迹归一化时间。每条候选的技能进度 `s` 与归一化时间 `u` 通过其绑定的 `_ProgressMap` 联系。原生优化在请求的一套固定时钟上运行；调用者提供的关节曲线保留自己的物理时长与进度映射。

## 1. 请求生命周期与资源分配

`plan_trajectory(problem, scene, robot, *, robot_validator, ...)` 是一次请求的完整入口。输入任务已经具有参考、事件身份、目标与机器人入口状态；场景已经具有观测几何、物体身份、手部模型及接触关系；`robot_validator` 连接独立机器人模型检查；`robot` 指向 cuRobo 配置。返回 `PlanningResult`，同时保存分阶段耗时、累计规划工作记录、候选残差与类型化失败原因。

`MotionBackend` 在多次请求之间复用原生求解器与固定容量缓冲。`bind_request` 创建或重绑定 `_CuroboPlanner`；`plan` 将本实例传给同一 `plan_trajectory`；`_claim/_release` 保证一个实例同一时刻由一个请求使用；`close` 释放所持资源。直接调用 `plan_trajectory` 时由该函数创建并关闭本次拥有的后端。请求为任务和场景建立私有副本，事件时钟与候选释放位姿在该副本中绑定。

`PlanningBudget` 同时规定数值维数、工作次数和资源范围。默认 IK 种子为 8、轨迹种子为 2、`knots=8`、原生细分为 4、IK 迭代为 40、轨迹迭代为 100、最多完整轨迹求解为 2、搜索预算为 3 s、请求预算为 120 s。各项含义不同：种子数控制并行候选，`max_task_anchors` 控制确切事件与补充见证容量，`max_validation_points` 控制完整验收的有限查询数量，主机 RSS 与设备占用限制控制实际资源。默认硬件预算延续本地半内存、半显存约束。

`trajectory_iterations=N` 为一次轨迹求解额度中的原生更新数，必须是原生内迭代块长度 `b` 的整数倍。首次额度在 `reference_seeds` 之前登记：初始化分得 `floor((N/b)/2)·b` 次，第一次完整场景优化分得剩余次数。总数只有一个内迭代块时，初始化保存同一粗曲线的前后记录，状态为 `no_initialization_iteration_block`，该块全部交给完整场景优化。后续碰撞或搜索恢复使用其原有完整求解额度。初始化、完整求解和诊断均使用同一请求截止时间、取消检查及累计工作账本。

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

`physical_joint_qp.py` 的 `CubicAffineMap` 为上述计算提供 CPU 仿射表达 `q(u)=B(u)c+b(u)`；`c` 是自由系数，`b` 由入口与固定时钟确定。生产初始化调用 `fit_reference`，以粗关节样本为最小二乘目标，联合原位置与速度边界求解。`fit_cubic_reference` 使用已有 quadprog，解析极值交换向 QP 增加实际违反的连续约束。工具事件通过后续任务允许集合计算。

输出转换为原生浮点 dtype 后再次检查完整位置和速度域。若舍入产生越界，`_fit_anchor` 在同一连续位置和速度边界内构造可行锚点，再沿系数线段恢复原生 dtype 可行性。QP 的速度条件属于这次粗拟合；随后 `CubicJointDomain` 的迭代域保持测得入口与连续关节位置约束，速度和加速度由原生约束及保存曲线的连续导数报告另行评价。`recover_position_step` 在当前动作、梯度、步长范围及固定基上构造优化恢复方向。

## 4. cuRobo 装配、参考 IK 与工具曲线初始化

`_CuroboPlanner._initialize` 先核对机器人模型身份、工具、活动关节及场景要求，然后建立时钟、绑定手指曲线、检查工作区并创建 IK 与轨迹求解器。原生配置来自 `ik/lbfgs_ik.yml` 和 `trajopt/lbfgs_bspline_trajopt.yml`。轨迹优化采用 L-BFGS，关节状态由本章三次基产生；主生产请求使用 eager 执行，即按当前程序调用计算张量运算。`bind_solver_components` 将本项目的物理域线搜索、优化核心与目标缓冲管理装配到对应求解器实例。

`_native_request` 统一构造 `constrained` 轨迹请求：`PlanningProblem` 中的类型化目标负责名义硬要求，`reference_objective` 负责完整路径的软跟踪。原生任务目标缓冲保留求解器所需的批次和种子布局，轨迹请求的原生末端目标位姿跟踪与关节目标位置跟踪均处于关闭状态。

普通 IK 工厂在原 `tool_pose_cfg` 上使用 `use_lie_group=True`、六轴权重全 1，并将旋转权重乘 4。Lie 核使用半角向量，该配置恢复原全转角二次能量及其关节导数。独立的 metrics/convergence 配置保留 axis-angle，继续以真实转角和原阈值判断原生收敛。普通参考目标由 `reference_to_link_pose` 转为原生工具连杆目标；`nominal_position_error_m` 与 `nominal_rotation_error_rad` 记录该原生目标的误差。

`reference_seeds` 在三次基参考参数处查询完整工具参考，调用原生完整姿态 IK，再经 `_rank_ik_candidates` 记录静态证据。`native_pose_ik_success`、`task_satisfied` 和 `model_valid` 分别说明原生收敛、任务点检查和模型检查；`static_model_query` 记录静态检查是否实际运行。有限且位于原关节位置域内、具有有限位姿能量的状态可作为数值提议。其原生失败标记与真实残差随原始候选保留。

`select_ik_tracking_candidate` 在这些提议中按未截断的位姿能量、相对前一跟踪状态的欧氏关节距离及原索引选择至多一个状态。能量为 `(w_p/2)·e_p²+w_R·e_R²`，其中 `e_p,e_R` 来自上述原生位置和真实转角指标，`w_p,w_R` 为补偿前的原成本权重。原生 `SeedManager` 保留这一热启动，并生成固定总槽位内的其余种子；没有本次合法提议时，既有跟踪状态保持其来源。查询记录保存实际热启动数量、原生补充数量、跟踪来源以及所选索引。

`select_ik_chains` 消费具名关节状态、原生报告、相同顺序的位姿能量、实测入口、查询物理时刻和原位置/速度界。节点首先通过有限性和位置域条件；动态规划依次按未收敛节点数、位姿能量和归一化关节速度能量选择固定数量的粗链，静态证据用于后续并列排序。速度界在此给出连续性成本的单位尺度。原生收敛节点记为 `SOLVED`，合法但原生未收敛的节点记为 `APPROXIMATE_PROPOSAL`。缺少数值节点的时刻在入口与已有节点间插值，末段保持最后节点，这类补充点记为 `INTERPOLATED_PROPOSAL`。关节链和 QP 共同提供粗系数；不足的分支或未构造成功的粗拟合使用实测入口建立的位置域内提议，并保存对应构造结果。

`_refine_reference_seeds` 负责最终初始化输出。它先对原生优化阶段调用 `reset_cuda_graph`，再通过 `_prepare_goal_buffer/_update_rollout_params` 绑定同一请求、入口、显式时钟、自由终端声明和候选布局。该顺序对应原生 reset 同时清空常规 rollout 目标注册表的行为。随后关闭本阶段的场景约束计算，先记录粗系数，再使用第 1 节分得的更新额度调用 `reinitialize` 与 `optimize`。该阶段直接评价完整工具参考、原关节成本、过程关系和完整有序轨迹上的功能事件允许集合。`_initialization_snapshot` 在优化前后对实际系数计算同一任务/关节证据，导出原生多项式并评价连续位置、速度及加速度。保存后的最终系数直接作为返回值；完整场景评价从该值开始。

快照调用该 optimizer rollout 的 `compute_metrics_from_action`，输入为每个种子一行的 `[S,M,D]` 动作，并由其 `metrics_constraint_manager.optimizer_evidence` 读取同次任务/关节证据；`S` 是种子数，`M` 是自由系数数，`D` 是关节数。该 metrics 入口与优化入口使用相同成本及约束配置，但入口状态索引按种子组织。线搜索入口 `evaluate_action` 的索引则按种子与粒子展开。完整验收另由 `trajopt.metrics_rollout` 的 `trajectory_metrics` 模式完成。

每次切换目标范围都使旧场景证据失效。初始化结束时恢复场景配置与总迭代配置，再清理原生图记录；后续完整主求解由原生 `_solve_impl` 重新绑定目标，并建立新目标的历史、功能乘子和候选保存状态。初始化有限更新完成时可仍有任务或导数残差，报告保留这些实际数值，返回位置域内数值候选。`status="completed"` 表示该计算完成，完整可执行资格由第 8 节的候选验收产生。调用者提供的 `JointTrajectory` 先按原曲线完整验收，转换为固定基粗系数后也进入相同细化接口。

`_build_goal_ik_candidates` 由 `ompl_seed` 调用，为搜索构造事件边界的数值提议。它将当前窗口内相同精确进度的 `pose_events`、`coordinate_events` 和 `supports` 合为一次事件查询。`bind_ik_goals` 绑定其原目标身份，调用 cuRobo 的 `disable_tool_pose_tracking`，令该次 IK 使用 `PlanningProblem.goal_residual(normalized=True)` 的连续允许集合。`ProcessCostManager` 同时加入点上可评价的过程条件，排除需要实际速度的停止项；优化模式计算归一化超差平方，指标模式保留原硬超差。原生 `GoalToolPose` 缓冲仍接收名义位姿，实际目标由已绑定要求计算。每次事件求解结束后，`finally` 清除事件身份并恢复参考 IK 的完整姿态跟踪。

名义材料目标到原生连杆目标的换算仍由 `_functional_goal` 完成：`T_W_E=T_W_C A⁻¹`，`T_B_L=T_W_B⁻¹ T_W_E T_L_E⁻¹`。其中 W 是规划参考坐标，E 是 TCP，C 是材料坐标，B 是机器人基座，L 是原生工具连杆，`T_L_E` 是工具标定。事件 IK 直接在关节状态上优化任务允许集合，完整旋转、倾斜及有界或自由 yaw 均由同一误差接口定义。

`_rank_ik_candidates` 对返回的事件状态计算 FK、`residuals(direction_mode="point", point_goal_ids=...)`、实际场景与原关节条件。停止项在点筛选中排除，完整路径继续评价其真实速度。报告将事件求解标为 `candidate_source="event_allowed_set_ik"`，以 `allowed_set_ik_success` 保存该次原生结果，以独立重算的 `task_satisfied/model_valid` 保存任务和场景结论。参考 IK 的失败行保留原失败状态，其静态查询记录为 `not_run_native_reference_ik_failed`。

`initial_state_diagnostics` 使用已绑定机器人状态、同一任务和场景计算请求入口的具体残差。输出区分任务不满足、机器人模型碰撞、任务场已确认体积交叠及移动实体覆盖条件，并保留数值未决报告。任务场配对的未决不产生碰撞正残差；原生自身数值边界、任务接触、初始支撑及覆盖仍按各自条件计算。入口是否继续由实际硬分项决定，报告中的每个非 clear 状态并不自动成为拒绝理由。

## 5. 同一任务声明下的分阶段约束证据

`_process_manager_type` 为请求创建 `ProcessCostManager` 及对应配置类型。`bind_request_data` 绑定任务、场景、进度映射和功能约束；`set_dense_counts` 区分普通与密集样本的有效数量；`bind_scene_evidence` 连接同一次前向的场景证据。原生 `cost_cfg` 使用 `reference_objective` 模式，计算连续参考与原生关节正则项；`constraint_cfg` 中的 `trajectory_optimizer` 模式计算任务和功能增广项，并根据当前阶段计算场景项。`set_scene_constraints(False)` 使初始化的初始/普通梯度入口共同省略场景查询及局部几何模型，证据中相应缺少 `process_scene`。重新启用场景约束后，完整主优化恢复原场景计算。`trajectory_metrics` 始终计算完整原硬要求，并配置相同普通成本供固定评分调用。

### 5.1 连续参考与固定时间平均

`reference_tracking.NativeReferenceCost` 读取 `PlanningProblem.sample_reference`，沿当前候选已绑定的物理时间采样完整工具参考。设 B 为机器人基坐标系，E 为标定 TCP，`P(u)` 为该候选的进度映射，`T` 为其物理时长。在普通有效采样时刻 `t_i`，候选 TCP 原点为 `p_E(q(t_i))`，参考为 `p_E*(P(t_i/T))`；二者均在 B 系表达。`θ_i` 为这两个 TCP 姿态之间的 SO(3) 最小转角。参考目标的离散时间平均为

```text
J_reference = Σ_i α_i [ (w_p/2) ‖p_E(q(t_i)) − p_E*(P(t_i/T))‖² + w_R θ_i² ]
```

`physical_time_mean_weights` 用真实物理时间的梯形公式计算 `α_i`，在有效前缀上归一化为和 1；填充样本权重为零。确切事件与碰撞见证保留各自的硬要求查询作用。`w_p,w_R` 来自原轨迹配置的 `tool_pose_cfg.weight`。`ToolPoseCost` 采用等权 Lie 旋转三轴、零收敛带和 `use_grad_input=True`；其旋转核输出 `θ²/4`，所以内部整体旋转权重取 `4w_R`，得到上述前向尺度。

`bind(sample_times,valid_count,progress_map,duration)` 在请求或网格变化时预先生成参考目标和积分权重；普通优化调用复用这些设备记录。原生成本调用将候选和时间展平成局部批次，显式目标索引保持每个时刻与其参考一一对应。`forward` 直接消费已有原生 FK 工具连杆位姿。`NativeRigidToolTransform` 将固定 `T_L_E` 合成到该连杆 L，位置导数包含 TCP 偏置产生的力矩项；姿态代价对基座系中无穷小转角的三维导数，经 cuRobo 的四元数导数编码返回同一 FK。该编码是已安装原生核的反向协议，坐标和固定工具标定与任务模型一致。

`freeze_cspace_time_weights` 从原普通成本模板的副本出发，在本次 `native_dt` 上一次吸收原生 STATE 成本的时间缩放因子，并设置 `use_grad_input=True`。`FixedTimeStateCSpaceCost` 继续调用原生前向，只为成本建立固定 `dt` 的 `JointState` 视图，因为原生能量项仍显式包含 `dt`。位置、物理速度、加速度、jerk、力矩和实际查询时间保持原值。请求重绑从原模板重新构造成本。普通关节成本与参考成本共同乘以 `α_i`，所以优化网格、密集网格和最终评分使用同一组成本系数。

### 5.2 任务与几何的同刻硬证据

轨迹状态先通过 cuRobo 纯运动学 FK 得到工具位姿。普通时刻与确切功能锚点按物理时间排列，`PlanningProblem.residuals` 与 `functional_inequalities` 消费相同有序状态、进度和时间，保留机构绕转历史、过程方向、停止条件及原事件允许集合。任务功能坐标、材料关系和工具标定从这些计算开始参与。

场景约束启用时，`DifferentiableRobotKinematics` 使用本时刻完整具名关节与名义手指坐标求各连杆位姿，`CollisionOwner.pose` 求场景实体位姿，`OnlinePlanningScene.components` 产生几何硬分项。`_scene_boundary_evaluate` 将 exact 样本及同一物理状态的关系左右侧一起查询，保留释放见证。`query_role` 决定场景聚合返回硬分项还是固定局部优化项；两种调用同时产生相同的实际硬证据。

候选硬条件的有效位来自普通网格的实际样本数与确切锚点的有效标记。两组标记拼接后，按轨迹状态使用的同一时间排序重排为 `active`；硬条件归约只保留对应的有效样本。固定容量中的填充位置仍保留其无效身份。

场景计算以 `active.any(0)` 选取整批候选中至少有一个有效样本的时间列，再将这些列上的位姿、关节、进度和物理时间交给 `_scene_boundary_evaluate`。选择保持原时间顺序，因而各候选的确切事件和释放状态仍在同一矩形批次中求值。返回的场景残差和候选证据按原列索引回填；各行仍由自身 `active` 掩码归约。不同候选的有效样本数不同时，为其他候选保留的列仍可能包含本行无效位置。

`PlanningProblem.optimizer_preferences` 在 `feasibility/quality` 两种目标配置中均汇总过程等式坐标和机构方向偏好；`quality` 阶段另加入支持声明中非硬分量的归一化超差平方，以及功能位姿、机构坐标目标中心偏好。当前主求解使用 `feasibility` 阶段，其支持计算由声明的硬分量进入功能增广拉格朗日；连续示教参考在相同阶段由普通成本持续优化，`measurements` 提供其解释性误差。场景优化项由本块冻结的 `LocalPairModel/LocalContactModel`、原生机器人距离、初始支撑高度和已有连续控制项组成；原始分类与未决状态另存为硬证据和查询报告。

任务计算按要求自身的声明确定硬条件。`trajectory_optimizer` 与 `trajectory_metrics` 均读取同一 `PlanningProblem.residuals`。硬功能条件由增广拉格朗日管理，其余过程项进入原过程代价；支持默认硬项为 body/footprint，gap/normal 按原阈值提供 `quality` 阶段偏好。原生指标的 `metrics.costs_and_constraints.constraints` 保存硬条件张量。`select_candidate` 首次按普通与密集结果判断时，要求该种子的 `result.success` 成立，且两套 constraints 的各项最大值均为有限数、不大于零。`convergence` 中的离散标识作为原生诊断保存。

`optimization_constraints` 返回功能条件管理器，`query_anchor_parameters` 返回当前确切查询参数，`add_collision_witnesses` 将密集查询发现的有限冲突时刻加入同一锚点容量。`query_diagnostics` 保存本次实际输入布局和场景查询信息。同一次计算的张量身份将候选保存依据与其产生的代价连接。

## 6. 功能约束、线搜索与候选保留

`FunctionalAugmentedLagrangian` 管理按种子独立保存的乘子 `λ≥0` 与罚参数 `ρ>0`。输入 `g_k` 为功能要求的有符号不等式，`g_k≤0` 表示满足。`terms` 为每项返回 `relu(λ_k+ρg_k)²/(2ρ)`。`update` 在既有内迭代块之间使用该种子当前实际探索点的残差更新乘子；初始 `ρ=2`，必要时乘 10，上限为 `10^6`，改善率判据为 0.5。`snapshot/report` 导出每个种子的乘子、罚参数、更新次数与残差度量。

`ProcessCostManager.optimizer_evidence` 返回 `(feasible, signed, fixed_objective)`。`feasible` 合并同次计算的原过程残差、全部硬功能不等式及其他原生硬约束；场景约束启用时，再合并原始场景硬分项。每项须为有限数且不大于零。`signed` 保存各硬功能条件的实际归一化值。`fixed_objective` 汇总普通成本集合中的连续参考与原生关节项。功能增广项、局部场景目标以及约束管理器中的任务和控制偏好各自参与求解，其值不进入该固定排名。AL 张量身份以及实际存在时的场景张量身份，将这些证据与同一次前向计算绑定；场景未求值时保留其缺失范围。

`PhysicalGradientCore._cost_and_gradient_with_guard` 执行原生前向、代价归约和反向传播，并在实际边界检查请求期限。`_begin_functional_solve` 按本次种子布局重置乘子和保留候选；完整场景阶段另调用 `_refresh_local_geometry`。该接口把每个实际起始动作按 seed→particle 顺序复制，通过常规 rollout 捕获一次时间、关系与许可布局。cuRobo 从同一优化配置构造初始和常规 rollout，FUNCTO 为它们绑定相同三次曲线时钟与过程管理器，因此两种梯度入口读取同一 `LocalGeometryModels`。捕获值只建立记录，随后正式梯度读取冻结模型。

`_finish_functional_block` 在已有块之间评价当前 `exploration_action`，从同一次轨迹计算读取硬证据、固定目标和功能残差。完整场景阶段同时通过 `_refresh_local_geometry` 捕获模型；初始化只进行原任务和关节前向。几何变化与功能乘子变化的种子掩码取并集；`_restart_changed_objective` 为这些种子重算目标和梯度、清理 L-BFGS 历史并保留累计迭代数。存在局部几何模型时，梯度重算通过 `LocalGeometryModels.repeated_base_evaluation()` 声明重复基准布局，场景共享数值几何记录，各粒子仍从自己的位姿张量计算导数。无功能条件的请求仍可因几何模型变化重置历史。`abort_capture` 在中断时撤销未完成的替换记录，异常继续交给请求生命周期处理。完整求解次数由 `PlanningBudget` 管理。

`PhysicalLineSearch._prepare_search_points` 在前向查询前对原生步方向施加物理域限制，随后使用原生 Approximate Wolfe 条件选择步长。`_guarded_step_direction` 根据实际系数位移和梯度变化维护有限、正曲率的 L-BFGS 历史；`_guarded_opt_step` 记录每个种子的有效更新和连续停滞步数。停滞达到一个原生内迭代块时，`_recover_stalled` 使用物理域二次规划构造方向，进行最多八轮候选步评估，未通过时将方向减半，寻找严格 Armijo 下降。该种子仍未取得下降时，`_exhausted[index]` 标记其在当前目标下停止更新，并保留其候选与累计计数；功能乘子改变目标时，`_restart_changed_objective` 可清除相应标记并重建局部历史。

`_retain_sampled_candidate` 为每个种子维护一个候选槽。动作与排序目标为有限数且本阶段的 `feasible=True` 时，按 `fixed_objective` 选择较小值。由于该值独立于块内局部几何和功能乘子，来自同一阶段不同块的候选可以直接比较。未保留到这类候选的种子返回原生优化结果，其残差由实际输出报告说明。完整主求解重新建立候选槽；普通与密集轨迹指标、连续关节域及最终候选验收继续决定交付。

## 7. 有限恢复与 OMPL 搜索

`plan_trajectory` 完整验收调用者提供的原关节曲线；`reference_seeds` 完成任务空间细化后，`evaluate_reference` 再完整评价其实际返回系数。合格初值完成执行绑定后进入独立的最佳候选记录；随后在首次求解额度的剩余更新数内执行完整主求解，持续改善普通参考目标。显式未知空间策略要求补充观察时保留相应结论。新候选只有完成全部验收、执行绑定且固定评分更低时，才替换已有合格记录。

主求解后仍无合格候选时，密集样本发现的碰撞见证可由 `promote_collision_witnesses` 加入下一次求解；否则在存在适用的碰撞或禁止未知条件且仍有预算时，`ompl_seed` 提供一次搜索种子。这两个恢复来源互斥，整个请求最多使用预算允许的两次完整轨迹求解。改善过程中累计工作预算耗尽时，交付此前已在预算内完成资格的最佳记录。取消、外部截止时间和其他计算异常继续产生对应失败；实际工作消耗统一在请求结束时保存。

`ompl_seed` 按功能目标、过程边界、接触切换及停止边界组织阶段。`_ompl_stage` 的状态为 `(q,u)`；`u` 单调前进并给出固定物理时间。过程等式通过 `_independent_projection` 选择局部独立行，在需要时交给 OMPL `ProjectedStateSpace`；允许区域中的不等式继续作为有效性条件。投影改变关节坐标，保留时间坐标。

`ompl_seed` 先以 `_build_goal_ik_candidates(..., check_search=...)` 生成实际搜索会读取的事件候选；这些 IK 调用前后检查同一搜索期限与请求期限。搜索检查直接连接后使用 RRTConnect。`checked_edge` 检查实际连接路径中的中间状态和跨过的事件边界，并调用同一观测场景和任务条件。当前阶段最多允许 `min(max_validation_points,8192)` 个状态查询，候选生成、投影和状态查询共同消耗搜索期限。功能终点候选由满足相关任务与模型条件的 IK 行提供；释放对象的姿态来自同一候选 FK。返回路径拟合到原有三次基，继续经过完整验收。

## 8. 从求解结果到候选曲线

`_observed_native_result` 对给定动作执行普通与密集原生 metrics；`evaluate_reference` 验收初始动作；`optimize` 返回原生优化结果与计算诊断。`_candidate_objective` 以请求固定的 `(knots+4)×interpolation_steps+1` 个归一化节点计算普通成本。每条曲线按自身 `T` 转为物理时刻，使用自身 `P` 采样同一工具参考，并保留自身物理关节导数。该有限积分定义统一候选比较的网格与系数，`objective_components` 保存各项值。

`native_diagnostics` 分别在 `rollout/interpolated` 中记录本次实际网格的 `ordinary_costs`，对已有成本张量按项求和。两种网格的积分值描述各自已完成的前向计算；候选排序使用上述统一网格的 `fixed_objective`。

`reference_initialization` 独立保存初始化的 `status`、实际 `iteration_allocation` 和场景 `not_evaluated` 范围。`coarse/refined.seeds` 中每项记录实际 `seed_index`、自由系数、标准 `curve_snapshot`、普通成本分项、固定普通目标、功能有符号残差、原生约束和连续关节域报告。`sampled_task_joint_feasible` 来自同次优化前向；连续导数检查与完整场景资格分别由相应报告表达。`refined` 保存的是实际返回动作，包含原生候选保留后的选择。

`_joint_state_curve` 将同一原生密集位置、速度和原步长复制成具名 `JointTrajectory`，并保留进度映射、原事件、手指曲线和已满足依赖；它同时供初始化记录和完整结果导出使用。中断时，已完成的阶段记录在 `plan_trajectory.finally` 复制到 diagnostics。完整候选尚未产生时，最后完成的初始化阶段第一条曲线可作为诊断轨迹保存；它的执行资格仍由完整验收定义。

`select_candidate` 先按普通和密集硬指标区分资格，再按 `fixed_objective` 和稳定索引排序可验收候选，依次进行完整关节与区间几何检查，返回其中固定评分最小的完整合格候选。已拒绝候选按验收残差和稳定索引提供诊断。当代码重新计算已有候选时，仍使用相同任务验收定义、原始观测模型和非穿透规则。

`_native_curve` 从原生位置、速度和原始时钟导出 `JointTrajectory`。`select_candidate` 对原生任务指标合格候选调用 `evaluate_candidate`，先检查连续关节域，再以原曲线 FK、原场景距离下界和实体运动界完成区间几何计算。新增实际场景违反进入原恢复决策，未决区间保留其范围和原因。已完成的指标继续复用同一曲线和时钟身份。`export` 返回已经验收的 `delivery_candidate`，新增查询数为零。

`qualify_native` 在后续原生查询改写暂存结果前调用 `export`，再由 `qualify_candidate` 调用 `execution_binding.bind_candidate`，完成曲线、问题、候选场景与验收的正式绑定。完整记录包含曲线副本、评分及分项、验收、事件时间检查、场景描述、时钟与资格完成时间。构造记录后的预算检查通过，记录才成为新的最佳候选。`deliver_incumbent` 恢复选中候选的这些字段，最终账本另保存整次请求的实际消耗。完整契约见[下一章](05-acceptance-and-execution.md)。

`reinitialize` 完成模型捕获、初始前向、功能参数初始化和候选保存后，`optimize` 消费同一个未消耗的原生迭代状态。块结束时以 `torch.no_grad()` 对实际探索点同时捕获下一块模型与原硬证据；候选排名读取不含局部模型的固定目标。局部模型或乘子改变后，再计算新目标梯度及重建相应种子的历史。
