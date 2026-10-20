# 已知机器人与观测任务场景

代码基线：`TASK_NONPENETRATION_CODE_COMMIT`。

`OnlinePlanningScene` 的构造输入为 `background`、`owners`、`coverage`、`reference_frame`、共享 `robot_model` 和明确 `fixed_geometry_names`，以及接触、附着、手指、初始支撑和未知空间策略。background/owner 保留原观测场；robot_model 是与验证器相同的 `RobotGeometryChecker`。固定角色来自模型清单。

| API | 计算、前置数据及输出 |
|---|---|
| `bind_online_robot(*, configuration, validator, device, check_request=None)` | 绑定同一检查器、完整可微关节树、原配对和请求回调，保存 robot_configuration |
| `components(..., joint_position, joint_names, *, trajectory_time, ..., query_role)` | 同一状态的完整 q、任务位姿、进度和手指时钟 → 硬/conditional分项及 last_query_report；query_role 只记录优化/验收阶段 |
| `evaluate(*args, **kwargs)` | 聚合非 conditional 分项；可把原分项写入 components_out |
| `clone_for_rollout()` | 共享原场和原模型，复制关系/手指的候选状态 |
| `with_release_history(position, quaternion, progress)` | 固定已选搜索前缀中的确切释放 FK；返回候选场景 |
| `validate_problem(problem, *, goal_identities=None)` | 核对原任务、附着/释放条件、接触激活及命令身份 |
| `optimizer_preferences(progress, joint_position, joint_names)` | 位置跟踪指令的优化偏好 |
| `project_residual(progress, joint_position, joint_names)` | 手指驱动中独立于已给耦合方程的等式，供 OMPL 投影 |
| `estimated_workspace_allocation(batch, horizon)` | 原场和原生查询共享字节、逐状态临时字节 → 资源准入 |
| `describe()` | 原模型身份、固定实体、`collision_rule='nonpenetration'` 和未知策略 |
| `to_execution_binding()` / `binding_record()` | `native_robot_observed_scene_v1`：原机器人几何、基座/标定、完整配对、纯运动学配置、原任务场、接触/关系、手指和自由格 |

`components` 使用完整 FK 乘实体固定局部变换。必检机器人自身对直接消费 checker 的原生 relation：分离通过，穿透产生原距离残差，数值边界进入独立的 `unresolved/robot/self` 硬分项。机械许可及排除对保留原策略。

机器人—观测场、运动实体—背景及运动实体参与的场—场配对共用非穿透查询，优化与验收不改变几何规则。已确认体积交叠产生前向分类值 1，并附同一观测场局部法向构成的刚体导数；该值不是米制穿透深度。局部方向不可用不抹去已确认的分类值。场配对 `unresolved` 的碰撞残差为零，状态仍保留在 `status_counts` 及有限 `nonclear_pairs` 记录中，不转换为 `clear`。`evaluate` 仍对实际非 conditional 分项求和；任务接触、初始支撑及覆盖保留自己的硬或条件分项。

固定安装体与运动 owner 进行同一非穿透查询；活动机器人与固定安装体由原生配对覆盖。机器人基座连杆即使 FK 恒定仍保留原机器人身份。机器人图像覆盖不形成拒绝分项；运动 owner 继续按显式策略查询已测自由空间。

机器人材料使用确切 geom ID。`_permission` 把原有限特征和目标 patch、任务间隙、法向、区间转为点对许可；HoldingContact 在原附着区间把指定材料实体与 owner 作为内部对。其他原机器人实体继续接受非穿透查询。零距离不单独构成体积交叠证据；任务要求的有限区域表面接触由独立查询判断。

`_release_pose` 获取原释放锚点，`_owner_poses` 取得同一候选的 owner 位姿并记录缺失释放；`_matrices` 保存可微位姿的同值矩阵给数值查询，`_plain` 序列化已完成记录。释放后的原场保持该候选释放位姿。几何分类、优化导数和执行反馈检查各自保留职责。
