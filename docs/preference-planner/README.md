# FUNCTO 声明位姿偏好规划器

代码基线：`c289890`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

本组件将完整技能表示、当前 RGB-D 观测场、标定、具名关节位置和速度，以及固定机器人官方模型，计算为通过声明条件验收的关节轨迹。`PreferencePlanner` 中的 preference 指事件位姿可以被明确声明为 `required` 或 `prior`。前者保留原几何硬条件；后者按显式物理尺度进入成本。其他任务条件、关节域及观测几何继续由各自接口处理。

## 两个独立入口

| 入口 | 后端与计算对象 | 完成规则 |
|---|---|---|
| `manipulation.planning.plan_trajectory` | `MotionBackend` → `_ConstrainedPlanner` | 原完整条件验收及执行绑定后交付 |
| `manipulation.preference_planner.plan_trajectory` | `PreferenceMotionBackend` → `PreferencePlanner` | 显式角色下的完整条件、末段静止及执行绑定完成后交付 |

新类通过正常继承复用原构造器、普通参考 IK、搜索、运动学与验收方法。`NativeSetup` 直接拥有普通参考 IK 的数值运行配置，并定义事件 IK 的偏好成本和静态排序；`PreferencePlanner` 定义终止消元、共同轨迹成本及几何必要条件。两路候选生成和默认入口保持独立，保存曲线验收通过共同父方法和数值证据接口复用；该共享组件代码基线为 `8b0e326`，见[共享验收架构](../planner-architecture/10-shared-candidate-validation.md)。当前每次调用只运行选定的一路规划器；多个规划器的时间截止与比较调度属于另行讨论的方案。

```python
from manipulation.preference_planner import bind_pose_objectives, plan_trajectory

bound = bind_pose_objectives(problem, pose_objective_declarations)
result = plan_trajectory(bound, scene, robot, robot_validator=validator, budget=budget)
```

`pose_objective_declarations` 覆盖完整问题中的全部位姿事件，每项包含原 `goal_id`、`role`、各误差组件的正 `scales` 及 `scale_source`。角色绑定发生在账本建立和运动块筛选之前。需复用原生工作区时，构造 `PreferenceMotionBackend(robot, robot_validator=..., budget=...)`，调用其 `plan` 并由调用者最终 `close`。继续请求携带完整冻结绑定、实际机器人反馈和已发生事件；驻留后端的 `incoming_terminal_contract` 明确提供原验证的终止合同。

## 坐标、时间及变量

`T_A_B` 把 B 系坐标映射到 A 系。工具位姿为 `T_reference_base @ T_base_link(q) @ T_link_tool`；位置单位为米，角度为弧度，时间为秒，Torch 四元数采用 wxyz。功能坐标和目标姿态来自技能表示及 grounding 后的显式关系。仿真实际运行环境中产生的目标位姿、真实分割、接触和任务效果只进入事后评价。

归一化物理时间 u、技能进度 s 和源参考时间参数 z 分别由原绑定时钟表达。完整三次关节系数 C 固定测量入口 q0/dq0；末段静止条件产生 C=offset+matrix@Z，其中 Z 为自由系数。原参考 IK 先产生关节样本，原拟合接口得到 C，再投影为符合终止等式的初值。该初值仍须完整验收。

机器人及固定末端配置使用官方本体碰撞模型。其形状和固定关系不随任务改变，位姿由标定、关节反馈及运动学确定。任务实体使用现场 RGB-D 的 TSDF 及观测支持域；声明刚性关系使用关系建立处的机器人参考连杆固定锚点预测同一表示的运动。

## 计算顺序和结果

请求先建立累计账本、复制可变场景并绑定完整问题与时钟。调用者提供的关节曲线或普通 IK 初始化先进入共同评分和完整验收。首条完成验收、执行绑定及请求检查的曲线立即返回。初值未合格时，既有搜索和 SLSQP 承担必要修复；每次求解使用共同成本以及当前固定的任务、关节与几何行。完整实体查询发现新的必要几何见证后，可在原剩余预算内再次求解。

软成本包括运动质量、任务过程偏好、事件位姿 prior 和观测几何方向项。其存在不额外触发合格后的寻优。有限普通 IK 解可作为近似提案，正式资格由完整曲线检查决定。功能接触是否 required 仍由原任务声明决定；数值验收与实际任务效果分别记录。

`PlanningResult` 保存正式曲线、验收、执行绑定和累计账本。`diagnostics['preference_planner']` 保存独立 CPU 计算记录，字段为 `joint_problem` 和 `objective_trials`；文件写入由调用者负责。拒绝候选、未完成的优化试探、非有限数值和失败类型均保留各自含义。执行使用同一已绑定曲线和手指时钟；预测目标位姿误差用于事后评价。

完成的区间几何证据采用递归只读记录，在同候选诊断和正式绑定之间共享；可变报告、曲线和输入快照保持各自所有权。JSON 归档保存一份完整证据文件，各报告通过相对引用连接，完整读取接口恢复同一记录。纯前向评分和验收调用 Pinocchio FK 与速度计算，需要梯度的求解调用原解析导数。相关共享组件见[完成证据与归档](../planner-architecture/09-completed-evidence-and-archives.md)与[运动学 API](../planner-architecture/api/pin_kinematics.md)。

## 组件文档

- [后端与联合计算](backend.md)：资源所有权、公共调用、系数投影、共同评分和候选证据。
- [请求生命周期](lifecycle.md)：初始化、搜索、必要修复、首条交付、失败及续接。
- [位姿角色](pose-roles.md)与[共同目标](objective.md)：显式声明、物理误差、尺度、硬行和梯度。
- [原生初始化](native-setup.md)：普通参考 IK、事件 IK 和静态候选排序。
- [终止区间](terminal.md)：时钟语义、仿射消元及保存多项式检查。
- [几何条件](geometry.md)与[数值求解](solver.md)：原实体证据、必要行、导数和资源计费。
- [API 索引](api-index.md)：实际源文件定义、签名、说明及调用关系。

维护组件测试位于 `tests/preference_planner`，保存场景重算与仿真结果位于项目外 `sim-version-plans`。当前依赖为既有 NumPy、SciPy、Torch、cuRobo、Pinocchio、Coal 与 FUNCTO 观测/运动学组件。

记录型请求通过 [通用运行适配](../planner-architecture/12-shared-runtime-adapters.md) 调用同一后端。固定机器人资源由 `config/robots/fixed_franka` 提供；任务、观测、源录制和独立评价记录分别作为显式输入。
