# 事件 IK 的位姿偏好与静态候选排序

本文对应 `native_setup.py`。共享基础代码来自 FUNCTO `d3100d0` 的 `_ik_manager_type`、`_ConstrainedPlanner._initialize` 和 `_ConstrainedPlanner._rank_ik_candidates`，其余运动学、时钟、资源管理与场景查询函数通过原接口调用。

代码基线：`1323d81`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

本组件处理一个已绑定运动块的事件 IK。IK 在指定事件时刻提出机器人关节构型，作为连接搜索和整条轨迹求解的数值输入。输入包括 `PlanningProblem`、现场观测建立的场景、测得的入口关节状态、机器人标定和固定官方本体模型、原 `PlanningBudget`，以及已有事件时钟。输出包括驻留的原生 IK 对象、事件候选构型、每个候选的原静态判定证据及位姿偏好成本。

事件 `goal_ids` 使用当前运动块中各类型集合的局部索引。`bound_motion_block` 产生的正式原身份继续保存在原绑定 `goal_identities`，角色和尺度保存在事件 `source['pose_objective']`。选择局部事件时不重新解释 source 的来源字段。工具位置使用问题参考坐标系，四元数采用 `wxyz` 顺序；误差组件及目标局部自由轴由 `pose_roles.py` 复用原几何定义。

## 原生初始化与类组合

后端使用以下继承顺序。

```python
from manipulation.preference_planner.native_setup import NativeSetup

class PreferencePlanner(NativeSetup, _ConstrainedPlanner):
    ...
```

`NativeSetup` 不定义构造函数。原 `_ConstrainedPlanner.__init__` 因而继续设置精度策略、拥有两个原生工作空间，并在初始化异常时调用原 `close`。它通过正常方法分派调用 `NativeSetup._initialize`。该方法保留原初始化顺序：校验当前问题和场景的具名关系，建立源进度与事件时钟，冻结手指命令，校验官方机器人运动学树，建立物理时间上的工具参考，进行原资源额度检查，构造事件 IK 与普通参考 IK，绑定原生和独立验证所需的运动学及物理域。

局部初始化使用本模块的 `_ik_manager_type` 创建事件管理器。事件 IK 保持原 `ik_seeds`、迭代数和原生配置；普通参考 IK 继续使用原库成本管理器和一个数值种子。普通参考是初始化提案，其近似位姿误差不构成新的任务要求。

原 `rebind` 继续通过管理器配置类型上的 `bind_request` 更新问题和预算，并清空先前选择的事件索引。时钟字典与种子时间张量按原接口更新，已有原生实例继续复用。当前模块不需要额外的常驻重绑定分支。

## 事件成本与原生指标

`_ik_manager_type(problem, budget, clock, seed_parameter)` 返回原 `RobotCostManagerCfg` 的局部子类。它保留 `create`、`bind_request` 与 `bind_ik_goals` 接口以及原 `RobotCostManager` 生命周期。一个 IK 查询只能选择同一进度的事件。

`compute_costs` 先调用父类计算既有原生成本，再把同一 `CostCollection` 交给 `_add_event_costs`。后者使用原工具位置梯度布局和 `WorldQuaternionVJP`，将原生几何雅可比对应到普通 Torch 位姿成本梯度，接着调用原 `PlanningProblem.residuals` 与 `goal_residual`。required 的几何允许集合、过程关系与原单位保持原值。

| 原生模式 | 实际追加的事件项 |
|---|---|
| `seed_optimizer` | 原 `event_allowed_set`，以及 `objective.pose_prior_cost` 产生的 `pose_prior`。两项各乘原 `constraint_weight` 一次。 |
| `seed_metrics` | 原 `event_allowed_set`，保留原未平方的硬条件指标。 |
| 普通参考或未选择事件 | 直接返回父类已有 collection。 |

对于选中的 prior，设误差组件为 (d_{i,c}(q))，其中 (q) 是候选关节构型，(i) 是事件，(c) 是位置、旋转或声明的有限 yaw 组件；(s_{i,c}) 是该组件的显式偏好尺度。追加项为原约束权重乘以 Σ_i Σ_c [d_{i,c}(q)/s_{i,c}]²，事件的实际求值时刻和自由轴规则由共同 `pose_prior_cost` 负责。位置量及尺度采用米，角度量及尺度采用弧度。尺度仅定义优化偏好。

`_add_event_costs` 明确分开原生流同步与任务张量计算。生产入口总是先完成父类调用，再追加事件项；CPU 小检查可以直接运行这个实际追加方法、原姿态 VJP 与原 `CostCollection`，而无需构造 CUDA 求解器。

## 静态候选证据与排序

`NativeSetup._rank_ik_candidates` 对每个有限候选执行一次原 FK，随后使用该状态完成原任务、场景与原生静态查询。同一次 FK 的工具位姿还传入共同 `pose_prior_cost`，记录乘以原 `constraint_weight` 的 `pose_prior_cost`。数值原生成功标志继续保存在 `allowed_set_ik_success`，不取代原任务和模型证据。

`rank_pose_ik_candidates` 按以下有序量选择候选。

1. 原 `_static_qualification` 的四个量：确定的任务失败、确定的模型失败、未建立的任务判定、未建立的模型判定。
2. 同一加权位姿偏好成本。
3. 与全部给定数值前驱中最近构型的关节距离。
4. 原候选索引。

这保持原证据优先级。已注册实体的确定碰撞、观测关系未知及原生数值判定分别保留其原证据与排序含义。prior 偏差只影响相同静态资格之间的次序。重复构型仍按原 `problem.numerical.joint` 删除，返回数量仍受原 `budget.ik_seeds` 限制。非有限关节输入保留无效列记录，不执行 FK；其余被排序但未入选的候选仍有对应 report。连接搜索与整条轨迹资格由已有后续调用承担。

## 检查范围

`tests/preference_planner/test_native_setup.py` 使用实际安装的 cuRobo 配置类、管理器、成本容器及原几何雅可比到四元数梯度的映射；CPU 直接进入实际 `_add_event_costs`。它们覆盖 prior 的非零位置和姿态梯度、硬指标与偏好的分离、required 原归一化值、同一时刻混合角色、自由 yaw、运动块局部索引与重新绑定、已有成本容器保留，以及原构造函数的异常清理和精度恢复。

静态排序检查通过原 `_scene_boundary_evaluate` 处理解析场景分量，确认每个有限候选各一次 FK、场景与原生查询；并检查原资格优先、重复合并、数量限制、数值成功标志的记录职责和同成本下的原连续性排序。完整场景结果由阶段运行记录说明。
