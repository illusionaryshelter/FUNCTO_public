# 共同位姿先验与轨迹成本

## 实现与输入

代码为 `objective.py`、`JointComputation.objective_components` 和 `PreferencePlanner._candidate_objective`，角色数据来自 `pose_roles.py`。代码基线：`PREFERENCE_IMPLEMENTATION_COMMIT`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。输入为同一请求的显式任务声明、原曲线状态q/dq/ddq/dddq、具名完整机器人状态、标定参考系中的TCP位置与wxyz单位四元数、物理时间、事件进度以及现场观测场。

## 事件成本API

`pose_prior_cost(problem, position, quaternion, progress, times, goal_ids=None)` 返回与batch/time布局一致的逐查询成本。每个活动prior在其第一次准确事件出现处计入一次，平台后续点不重复计费。`goal_ids` 使用当前problem中的局部事件索引，适用于运动块内的事件IK；正式原身份由原绑定表维护。

令e_k为原 `PoseEventRequirement.error_components` 中被声明选中的分量，s_k为独立的正成本尺度。单事件成本为sum_k (e_k/s_k)^2。位置分量单位为米，旋转与有限twist分量单位为弧度。无自由轴时使用原完整旋转误差；有自由轴时使用原目标坐标中的swing和有限twist定义，完全自由twist不产生分量。required事件维持原允许集合，不增加prior中心成本。

尺度来自 `source.pose_objective.scales`，允许集合来自原事件容差字段。两者在数据职责上独立。尺度由调用者显式声明，表示数值优化取舍。调用者将原 `constraint_weight` 乘一次；数值权重保留在输出中，实际取舍须根据各分量记录分析。

## 完整轨迹成本API

`trajectory_objective_components` 接收problem、scene、states、robot_state、progress、times、ordinary_indices、bounds、motion_configuration、native_dt和constraint_weight。`robot_state` 使用原 `robot_tool_states` 返回的完整关节、机器人连杆位姿、TCP位姿和速度。输出四个标量Tensor，保留一阶导数：

- `motion_quality`：原 `motion_cost`，只用普通物理时间点及其归一化积分权重。
- `task_preferences`：原feasibility阶段的过程偏好，乘原constraint_weight。
- `pose_prior`：上述显式prior成本之和，乘同一constraint_weight。
- `observed_geometry`：原现场几何方向成本，使用相同查询和机器人连杆位姿，乘同一constraint_weight。

`include_geometry=False` 供明确排除现场场成本的离线诊断；生产求解及候选评分使用完整四项。`geometry_report` 接收原几何成本已有的诊断输出。额外事件与几何查询不改变普通运动积分权重。

求解器调用共同函数后记录完整C、未缩放分量及必要几何行数。候选评分对相同时钟直接使用同一精确数组和普通索引；显式不同候选时钟使用原相位对齐接口映射同一组查询，保留事件进度及普通积分索引；运动权重根据映射后的物理时间计算，不追加第二组普通网格。候选报告使用四项未缩放和，SLSQP局部正缩放只服务本次数值求解。请求在首个候选完成原验收及绑定后返回；共同成本的存在不触发额外寻优。

## 参考表示与验证

完整参考保留在 PlanningProblem、普通参考 IK、时间绑定和结果中。事件偏好仅在声明事件出现处计费；它不是全程参考形态锁。`test_objective.py` 验证事件单次计费、IK 批次、尺度与硬界独立、旋转自由度及梯度；`test_candidate_objective.py` 验证导出曲线与求解器使用同一未缩放四项成本和查询布局。实际保存场景的数学审查属于实验记录。
