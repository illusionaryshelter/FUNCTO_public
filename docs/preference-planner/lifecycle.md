# 规划请求与首个完整合格候选的交付

## 研究对象与实现基线

本组件管理一个规划请求从问题绑定到正式候选交付的过程。输入包括完整 `PlanningProblem`、现场观测建立的场景、固定机器人配置、具名关节反馈、累计工作账本，以及可选的实际执行续接记录。完成规则统一作用于所有任务及显式位姿角色。

代码基线：`1323d81`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

组件返回原 `PlanningResult`。成功结果包含首条完成原完整验收、正式执行绑定及请求检查的曲线，以及属于该曲线的绑定、报告和累计账本。数值优化器产生曲线后，原验收接口仍须判断它是否满足任务、关节、场景及连续域条件。失败结果保留原类型化原因和对应候选的诊断证据。

## 输入语义与调用关系

`plan_trajectory(problem, scene, robot, *, robot_validator, budget, device, continuation, request_ledger, rebind, backend, cancelled, deadline, joint_reference)` 是本组件的入口。调用者可以提供 `PreferenceMotionBackend`，其 `bind_request` 返回执行本地目标计算、终止约束与几何条件交换的 `PreferencePlanner`。未提供后端时入口创建 PreferenceMotionBackend，并从正式续接绑定取得终止合同。参数和返回类型沿用原规划入口。

`problem` 保存任务声明、技能表示、参考坐标及显式位姿角色；`scene` 保存观测几何及原未知空间策略；`robot` 与 `robot_validator` 提供固定机器人模型及其原检查接口。`joint_reference` 表示调用者提供的完整关节曲线。参考位姿继续使用 `PlanningProblem` 声明的 world 坐标和工具参考连杆，关节数组使用曲线中的具名关节顺序；本组件传递这些数据和坐标声明，实际位姿计算由后端完成。

入口调用 `scene.clone_for_rollout()` 获得本请求的可变场景，调用 `_request_work_ledger` 建立或恢复累计预算、原义务和历史。实际执行续接由 `_prepare_continuation` 核对同步机器人反馈、已发生事件、累计物理步数与冻结时钟；通过后才把执行事实写入账本。`_verified_dependency_outcomes` 提供已验证的依赖完成证据，`PlanningProblem.bound_motion_block` 选择当前允许处理的区间，`_bind_motion_work_ledger` 恢复该区间对应的计费基线。

后端接收完整问题对象。位姿先验的显式数据由 `pose_roles` 解释，事件 IK 的目标与排序由 `native_setup` 处理，轨迹共同目标由 `objective` 和 `joint_computation` 计算。生命周期中的完成判断消费原验收结果和正式绑定状态。各目标项仍完整参与候选生成、必要修复和候选报告，具体公式见 [目标与后端架构](README.md) 及 [原生初始化架构](native-setup.md)。

## 候选生成与完成规则

`backend.bind_request` 绑定曲线时钟、问题和场景后，入口调用 `initial_state_diagnostics` 检查当前机器人入口。后续候选依次按以下调用关系处理。

1. 如果提供 `joint_reference`，先核对原曲线的事件时钟，再由 `bind_reference_candidate` 和 `evaluate_candidate` 完成该曲线的绑定与验收。合格曲线经过 `qualify_candidate` 后立即返回。需要观察或缺少声明模型支持时，保留对应诊断；可修复的拒绝继续进入原修复路径。
2. `reference_seeds` 生成初始化，`evaluate_reference` 将其交给原 `planner.select_candidate`。每条有限曲线由 `_candidate_objective` 计算共同成本，再由 `evaluate_candidate` 完成原验收。首条合格曲线完成回调后结束请求。
3. 参考初始化未合格时，入口按原条件调用 `optimize`。明确的碰撞或禁止覆盖残差可以使用原有限 `ompl_seed` 搜索；如果几何失败首次出现于主求解之后，仍可使用原剩余恢复机会。首条完成原验收及绑定的修复曲线立即返回。

原 `planner.select_candidate` 逐个消费候选生成器。它在合格候选的回调完成后返回，因此后续种子不再被请求。不合格候选继续原筛选流程；全部不合格时，原选择器保留已完成候选中违反程度最小者及其证据。`fixed_objective` 和 `objective_components` 是对应候选的共同成本报告，首个完整交付结果的完成条件不再依赖跨阶段成本比较。

## 完整绑定与结果所有权

内部 `qualify_native(result, attempt)` 对合格数值候选调用 `planner.export`，在该候选仍持有验证缓冲区时交给 `qualify_candidate`。调用者提供的完整关节曲线直接进入同一 `qualify_candidate` 边界。

`qualify_candidate(trajectory, validation, attempt, binding_scene)` 要求原报告为合格且验证记录完整。它调用原 `bind_candidate`，将曲线、场景和同一候选的验收记录转换为正式执行绑定，再补齐当前区间、目标身份、事件时钟、独立效果记录、物理预算和历史账本。正式绑定中的 `validation.export` 保存属于该曲线的验证数据。

完成上述绑定后，组件复制曲线及报告，执行原请求 `check()`，随后将它们发布到 `PlanningResult`，设置 `qualified_candidate_ready` 时间记录。返回结果持有独立曲线，`returned_candidate` 和正式绑定各自保存报告数据；后端复用验证数组或曲线缓冲区不改变已交付结果。参考候选报告与实际返回报告各自保留所属曲线的来源和验证状态。

完整供给曲线、参考初始化、主优化和几何恢复的调用点均在成功发布后立即返回。生命周期因此只需维护当前交付结果，完成之后不再建立用于比较后续候选的状态。

## 中断、续接与累计工作

`check()` 处理显式取消、外部截止时间和原工作预算。`record_validation_work()` 记录开始的验收次数与已完成查询；规划器内部查询继续使用同一 `consumed_planning` 字典。搜索时间和查询数在原搜索 `finally` 中计费，即使搜索中断也保留已发生的工作。

`PlanningWorkExhausted`、取消、截止时间以及程序或模型 API 异常均保留原失败类型、发生阶段和原因。验收合格但正式绑定或请求检查尚未完成时，结果仍不具备执行授权。中断优化保留当时的诊断曲线、来源和未完成验证状态；已完成的拒绝候选保留自己的验收报告。

`request_ledger` 用于继续同一失败规划请求的累计工作；实际执行后的 `continuation` 另外提供原正式绑定、同步关节反馈、已发生事件与物理步数。普通续接保留原冻结时钟，显式 `rebind` 使用原剩余工作或重试发生项接口。首个合格剩余曲线仍按同一完成规则交付，并保留续接后的原时间起点和目标身份。

入口的最终 `finally` 写回本轮耗时、绑定记录与完整累计账本，并释放后端所有权。成功结果的正式绑定取得最终计费后的账本。`qualified_candidate_elapsed_seconds` 表示完整交付候选的准备时间，`elapsed_seconds` 包括后续记录与释放；二者使用同一个请求的单调时钟。

## 证据返回与验证

`PreferencePlanner.diagnostic_snapshot()` 返回本请求的 CPU 证据副本，包括终止映射、角色声明、几何行来源、已完成候选、数值求解及逐次目标试探。`plan_trajectory` 在释放原生资源前将其放入 `PlanningResult.diagnostics['preference_planner']`，因此独立入口与驻留入口均保留相同证据。非有限数值试探保持诊断含义，文件格式和路径由调用者选择。

`tests/preference_planner/test_lifecycle.py` 覆盖首条合格即返回、必要修复、搜索、失败、截止、取消、正式绑定中断、请求重试及实际续接的账本语义。`test_public_entry.py` 检查直接入口和驻留入口均调用本组件、正确释放资源并返回独立诊断。组件测试不构成真实任务成功率验证。
