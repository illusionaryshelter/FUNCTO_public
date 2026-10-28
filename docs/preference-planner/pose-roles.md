# 位姿事件角色与现有绑定合同

本文说明 `pose_roles.py` 的实际接口及其在现有 FUNCTO 数据结构中的职责。

代码基线：`PREFERENCE_IMPLEMENTATION_COMMIT`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

研究对象是一个完整 `PlanningProblem` 中已经编译的 `pose_events`。每条事件保留原注册表索引、进度、工具到功能坐标的变换、目标变换、位置与旋转诊断范围、目标坐标中的可选 yaw 轴，以及 `actual_relation` 所声明的对象与功能坐标关系。`bind_pose_objectives` 在请求账本建立和运动块筛选之前接收完整问题及覆盖全部位姿事件的显式声明，返回原类型问题的副本。

位置输入使用问题的参考坐标系，单位为米；姿态输入为同一坐标系中的 `wxyz` 四元数。原 `PoseEventRequirement.error_components` 将工具位姿与 `tool_from_function` 组合为功能坐标位姿，并与目标比较。位置误差是功能点间距离。未声明 yaw 轴时，旋转误差为完整相对旋转的角度；声明轴时，旋转误差为移除绕该目标局部轴的主值 twist 后的 swing 角度，`yaw` 为该 twist 角度的绝对值。原 `signed_yaw` 与 `full_rotation` 仍是可读取的诊断量。

## 单一声明与数据保存

每条输入声明包含以下字段。

| 字段 | 数据含义 |
|---|---|
| `goal_id` | `['pose_events', 原索引]`，对应完整问题中的原事件身份。 |
| `role` | `prior` 表示优化偏好；`required` 表示保留原几何允许集合。 |
| `scales` | 偏好成本的正数尺度，以组件名索引。`position` 使用米，`rotation` 和有界 `yaw` 使用弧度。 |
| `scale_source` | 说明各数值的实际来源。角色对照可明确沿用原质量归一化数值。 |

每条事件都必须给出角色和尺度，包括 `required`。尺度的键必须与基类原有的实际几何组件一致。完整自由 yaw 不含 `yaw` 键；零宽度 yaw 仍有该组件，因此成本尺度必须另外声明为正数。位置与姿态的原诊断范围保持原值，尺度独立保存在声明中；数值相同只表示这次对照选择了相同归一化尺度。尺度表示优化取舍，不表示成功概率或经过估计的误差分布。

绑定后，唯一运行时角色声明是各事件 `source['pose_objective']`。输入列表中的 `goal_id` 用于一次性关联，角色列表不会附加到问题或形成第二份运行时配置。结构化 source 保留全部原键，包括接触身份所用的 `kind`、`carrier_id`、`activation_event`；原 source 若为字符串等非映射数据，其原值保存在 `original_source`。角色始终来自明确的声明，不由 source 中其他内容推断。

## 实现与公开 API

| API | 实际作用与返回值 |
|---|---|
| `bind_pose_objectives(problem, declarations)` | 验证完整事件覆盖，深拷贝问题，把每条 pose 事件重建为 `ObjectivePoseEvent`，返回同类型 `PlanningProblem`。 |
| `pose_objective(event)` | 读取同一 source 中的角色、尺度和尺度来源。原基类根据正式绑定重建的事件也可调用。 |
| `pose_objective_scales(event)` | 返回显式组件尺度。 |
| `pose_objective_components(event, position, quaternion)` | 调用原 `error_components`，返回尺度所选择的物理误差组件，保留 Torch 梯度。 |
| `prior_goal_ids(binding)` | 从正式绑定的各事件声明与 `goal_identities` 派生 prior 的原身份，供现有事后诊断聚合接口使用。 |

`ObjectivePoseEvent` 继承原 dataclass，不增加字段。它只覆盖两个硬条件接口。`component_tolerances` 对 prior 返回空组件集合，对 required 返回基类原集合。`evaluate` 对 prior 返回 `position[..., 0] * 0`，保持批次、时间维度、数据类型、设备与零导数；对 required 直接调用原方法。这样可以避免空生成器求和产生 Python 整数，同时保持所有几何测量方法原样可用。

原 `PlanningProblem.constraint_components` 遍历 `component_tolerances` 产生位姿等式或不等式。原 `residuals` 调用 `evaluate`；原 `goal_residual` 遍历同一组件集合构建点目标允许集合。三者因此都保留 required 的原计算，而 prior 不产生位姿资格行。问题中的进度界、过程关系及其他要求仍由原方法计算。原 `TrajectoryComputation.__init__` 通过 `component_tolerances.get('yaw') == 0` 选择固定入口 yaw 等式，所以其行布局直接遵循同一角色。

原 `PlanningProblem.measurements` 调用 `error_components`，不依赖覆盖后的允许集合。位置、swing、yaw、完整旋转和参考偏差均继续输出。`anchors` 与事件身份也保持原样。`bound_motion_block` 深拷贝并筛选原事件，保留子类、声明、原身份、时钟、起始关节位置与速度；本模块不新增停止事件。

## 正式绑定、重建和继续请求

`execution_binding.plain` 遍历原 dataclass 字段，递归保存 source。由于子类不新增字段，`execution.py` 与 `recorded_skill_evaluation.py` 中原有的 `PoseEventRequirement(**row)` 可以直接重建正式绑定，并完整保留 `pose_objective`。执行入口对有限接触的身份校验读取原来源字段与事件关系，既有 `arrival_goal_id` 不变。

事后评价重建的是原基类，因此仍按原位置、旋转与 yaw 范围产生诊断 T/F/U 和误差。`prior_goal_ids` 同时从该正式绑定取出角色及原索引，交给已有的 diagnostic-goal 聚合接口。这样允许诊断位姿结果与独立效果证据分别报告，不会将 prior 的测量自动改成成功。

原 `_obligation_contract` 已完整保存并比较 source，因此角色、尺度或尺度来源的变化都会表现为同一请求合同的变化。该合同还保存运行时类型名，所以第一次请求及后续完整问题都必须在账本建立前经过同一绑定函数；所有角色统一使用 `ObjectivePoseEvent`。声明的对象变换和实际关系保留原有可重新绑定规则。

采用 source 的理由是现有合同能够保真保存明确输入，并已贯穿执行和评价重建。本组件定义其 `pose_objective` 子记录的语义；共享组件无需读取该键来完成既有身份校验。原组件、序列化、事件时钟及继续请求实现直接复用，避免复制 `PlanningProblem` 或执行大函数。

## 检查与集成范围

`tests/preference_planner/test_pose_roles.py` 实际调用原硬行、点残差、测量、固定 yaw 行布局、plain 序列化、基类重建、请求合同和接触身份接口。它们覆盖同一时刻混合角色、自由 yaw、完整 source 与 `actual_relation` 保留、原身份派生、独立成本尺度，以及缺失或不一致声明的输入拒绝。

本模块向共同目标与事件 IK 提供原物理误差及尺度 API。实际目标接入、候选选择生命周期和评价入口调用由同包内模块及调用者的评价适配器负责，并在其架构文档及计划对照中记录。
