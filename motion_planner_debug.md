我建议把方案收敛成：

> **演示提供抓持目标和运动初始解；cuRobo 负责求解与时间处理；需要时 OMPL 提供另一条路径初始解；Pinocchio＋Coal 负责一致的几何验证。全部使用固定的后端配置，不按 skill 调权重、距离或接入点。**

这里有真实可调用的接口，不需要另写一个新的轨迹优化算法。

------

## 1. 先划清“不调参”的含义

**可以做到：新增 `grasp_mug`、`grasp_bowl` 等技能时，不修改后端参数、不新增动作名称分支。**

不能诚实地承诺：数值系统完全没有容差、采样分辨率、规划预算和机器人速度限制。它们是机器人与求解器配置，不是技能模板。

两者应当严格分离：

| 固定在机器人／后端配置中        | 随技能调用传入的数据                 |
| ------------------------------- | ------------------------------------ |
| 机器人模型、TCP、关节顺序与限位 | 当前机器人状态                       |
| 夹爪几何、开口范围、控制接口    | 演示来源的目标交互位姿               |
| 碰撞模型与验证精度              | 当前场景与对象绑定                   |
| 求解器默认配置、计算预算        | 演示运动参考                         |
| 执行与停止条件                  | 已有的接触／开合事件、明确的过程要求 |

cuRoboV2 的 `MotionPlannerCfg.create()` 文档也明确建议：一般集成代码传入机器人、场景和容量配置，其余任务 YAML 保留默认设置。**可以冻结一份后端配置使用，而不是见一个技能调一遍。** 默认值是否覆盖某次请求，由求解结果决定；失败不是自动修改参数的理由。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.motion_planner.html)

我们的要求可以落实为：

```
同一 robot profile
同一 planner profile
同一后端代码

新增 skill → 只新增 SkillRecord 中的数据
```

这和 TaskFlux 已确定的“不按动作名字新增 generator”一致。   

------

## 2. 一个具体发现：前面把 cuRobo 的接口说得太笼统了

**“目标”“参考轨迹”“优化初始解”不能放进同一个参数。**

我核对到，当前 cuRoboV2 的 `TrajectoryOptimizer.solve_pose()` 有独立入口：

```
goal_tool_poses   # 要到达的工具位姿current_state    # 当前关节状态seed_config      # 用于生成初始轨迹的关节配置seed_traj        # 已有的轨迹初始解goal_state       # 显式目标关节状态
```

这些都是实际源码中的参数。[GitHub](https://raw.githubusercontent.com/NVlabs/curobo/main/curobo/_src/solver/solver_trajopt.py)

而且发现了两处值得特别注意的实现细节。

**第一，不能只看 `GoalToolPose` 的类型说明，就认为整条 Cartesian 参考已经接通。** 文档称它支持 `horizon > 1` 的逐时刻目标，但我检查的公开 `main` 调用链中，`ToolPoseCost.forward()` 对目标执行 `squeeze(1)`，下层又要求目标是四维张量，并按静态目标索引。多时刻目标在这条路径上存在明显的不一致。**我没有检查我们本机版本，不能说这就是我们当前卡住的原因，但不能再给我们一段“整条轨迹直接塞进去”的伪可用代码。**[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.types.html)

**第二，`seed_traj` 的实际输入需要包含种子维度。** 当前 `SeedManager` 明确检查：

```
[batch, num_seeds, action_horizon, dof]
```

而不是只传一个 `[时间点数, 关节数]` 数组。这里 `action_horizon` 来自求解器，不是视频帧数。[GitHub](https://raw.githubusercontent.com/NVlabs/curobo/main/curobo/_src/solver/manager_seed.py)

所以，现在最明确、能对接已有实现的入口是：

> **静态交互目标走 `goal_tool_poses`；演示转换得到的关节路径走 `seed_traj`。**

不先要求完成一个新的、持续追踪演示的代价函数。

------

## 3. 我建议的具体 runtime：固定目标，多种初始解，同一个验证出口

以下处理的是我们现在给出的这种请求：

> **参考运动＋目标交互位姿＋机器人模型＋当前场景。**

演示参考可以帮助求解，但不是“所有中间帧都必须经过”的硬性路径。SkillRecord 已明确要求保持的过程约束仍然要检查，不能默默删除。

### 第一步：编译一个运动请求，不让求解器看到技能名称

建议后端实际收到的是：

```
MotionRequest
    current_joint_state
    goal_tool_pose
    reference_tool_path
    scene_snapshot
    gripper_state_before_interaction
    interaction_events
    explicit_path_requirements
```

`grasp_mug`、`grasp_bowl` 不进入规划算法的判断条件。它们只影响 Executor 已经实例化出来的对象、目标和事件数据。

这一接口正对应文档中 Executor 的职责：把技能内容实例化为候选末端运动、开合时序和过程要求，再交给共享后端。   

### 第二步：检查目标本身，而不是先要求整条参考可执行

先求目标交互位姿的 IK 候选，检查关节限位、机器人自身碰撞和当前阶段不允许的环境碰撞。

这里要特别注意：

**用于到达的夹爪状态，与抓持完成后的夹爪状态不是同一个东西。**

对于我们讨论的抓取，应当用演示／夹爪适配已经给出的到达前开口状态检查运动，再执行闭合事件。不能拿“已经闭合并接触物体”的几何状态，去要求整个接近过程完全无接触。

如果终点都不合法，应返回具体原因，而不是继续在 reference weight 或 approach distance 上寻找答案。

### 第三步：把演示转成一个关节空间初始解

这里不再要求它是“已经可执行的演示后缀”。这里的演示就是FUNCTO原先生成的参考轨迹。

处理方式可以固定为：

```
迁移后的 Cartesian 参考
    ↓
按求解器所需数量重采样
    ↓
由终端 IK 配置向前回溯，使用邻近解作 IK seed
    ↓
得到关节路径初始解
    ↓
把边界配置设置为真实当前状态、已验证终端状态
```

这里的“向前回溯”指从演示末端向更早的采样求解。

**这条初始解允许暂时碰撞，也允许不平滑。** 因为它只是优化器的起点，不能直接执行。若某条演示无法产生基本可用的关节初始解，丢弃的是这个初始解，不是宣布这个 skill 永远无法执行。

关键区别是：

> 我们不再先花力气证明每一个演示 pose 都能严格跟踪，再开始规划。

当前机器人起点与演示起点不同，可能使这个初始解质量变差，但这不构成循环依赖：真实起点始终由 `current_state` 提供。

cuRobo 自己的图规划支路也是把几何路径插值到 `action_horizon`，然后作为 `seed_traj` 交给轨迹优化；并不把插值路径直接当作最终执行命令。[GitHub](https://raw.githubusercontent.com/NVlabs/curobo/main/curobo/_src/motion/motion_planner.py)

### 第四步：先尝试演示初始解，失败再使用路径搜索初始解

固定流程即可：

```
演示初始解 → cuRobo 轨迹优化
                    │
               验证通过？
                /       \
              是         否
              ↓          ↓
            候选     cuRobo 图规划／OMPL
                           ↓
                    路径初始解
                           ↓
                    同一个轨迹优化器
                           ↓
                        同一个验证器
```

**这不是按 skill 分支，而是按求解结果分支。**

cuRoboV2 原生 `MotionPlanner` 已经采用了“IK＋轨迹优化，必要时图规划提供 seed”的流程；我们不需要重新实现这一套搜索逻辑。OMPL 可以在既定共享后端里承担额外的路径搜索，不需要每次都把两套完整规划串行跑一遍。[GitHub](https://raw.githubusercontent.com/NVlabs/curobo/main/curobo/_src/motion/motion_planner.py)

也不需要引入新的参考权重来控制这套流程。可以固定采用：

> 优先接受通过验证的演示初始解结果；没有结果时，接受同一目标、同一明确约束下的其他合法规划结果。

**代价必须说清楚：演示作为初始解，不保证输出完整保留演示形状。** 这是该接口的确定含义，不是等待解决的研究问题。若某段运动已经被 SkillRecord 标为必须保持，就按要求验证；不能仅因为终点到了就放行。

我们已经接受执行可能失败，那么就不需要再把“自动证明所有未表述的过程语义”设成发出第一条命令的前提。

------

## 4. cuRobo、OMPL、Pinocchio、Coal 各自只做自己的事

我不建议再加 MoveIt、Drake 或另一个优化框架。我们现有组合可以这样分工：

| 组件          | 固定职责                                                     |
| ------------- | ------------------------------------------------------------ |
| **cuRoboV2**  | IK、轨迹优化、已有图规划、时间处理，产生关节轨迹候选         |
| **OMPL**      | 必要时搜索从当前关节状态到目标关节状态的几何路径，提供替代初始解 |
| **Pinocchio** | 统一正运动学、TCP 与关节顺序转换，更新机器人碰撞几何位置     |
| **Coal**      | 场景和机器人几何的碰撞／距离查询，服务 OMPL 和最终验证       |

Pinocchio 有直接更新几何位姿及计算碰撞的接口，Coal 提供碰撞与距离运算；OMPL 则要求使用方提供状态／运动有效性检查。它们本来就是可以这样组合的组件，不需要它们先理解 `grasp_mug`。[Gepetto Web](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_c_collisions.html?utm_source=chatgpt.com)

最容易让集成“什么都规划不出来”的，反而是下面这些接口不一致：

| 需要核对的项目                             | 可能导致的结果               |
| ------------------------------------------ | ---------------------------- |
| 世界坐标系与机器人基座坐标系不一致         | IK 目标被放错位置            |
| TCP 实际是法兰，目标却按夹爪中心定义       | 到达姿态系统性错误           |
| 四元数 `wxyz`／`xyzw` 混用                 | 姿态错误或大量 IK 失败       |
| 关节排序、mimic joints、夹爪锁定状态不一致 | 两个库看到不同的机器人       |
| cuRobo 球近似与 Coal 网格不一致            | 一方认为可行，另一方全部拒绝 |
| 抓持目标接触被当成禁止碰撞                 | 目标配置天然不可达           |
| 把参考点数当成优化器 action horizon        | 张量尺寸或轨迹表示错误       |

这些是**需要逐项检查的候选原因，不是我对我们本机故障的诊断**。但应先把它们记录出来，而不是在没有错误归因的情况下调规划参数。

### 接触不能靠“关掉手指全部碰撞”解决

预期接触应按**接触部件—目标对象—执行阶段**处理，其他碰撞仍要检查。Pinocchio 的碰撞对可以单独激活／停用，能够承载这种统一策略。[Gepetto Web](https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/structpinocchio_1_1GeometryData.html?utm_source=chatgpt.com)

cuRobo 的公开 `plan_grasp()` 会使用固定接近偏移，并可以禁用指定 link 的碰撞。因此，它并不适合未经适配直接作为我们这里的统一 grasp 实现。尤其不能为了让规划成功，永久取消手指对所有障碍的检测。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.motion_planner.html)

------

## 5. 真正需要接上的 cuRobo 调用，长这样

下面展示的是**现有 API 的核心接线参考代码**，不是声称已经在我们本机运行过的完整后端，并且只保留了核心调用，仅供参考。前提是 `planner` 已按我们的机器人和场景创建，`q_goal` 是已核验的目标 IK，`seed_bank` 是已转换的关节初始解。

```
import torch

from curobo.types import GoalToolPose, JointState


def solve_from_demo_seeds(
    planner,
    q_current: torch.Tensor,       # [1, dof]
    q_goal: torch.Tensor,          # [1, dof]
    goal_position: torch.Tensor,  # [3]，规划基座坐标系
    goal_quaternion: torch.Tensor,# [4]，wxyz
    seed_bank: torch.Tensor,       # [1, seeds, action_horizon, dof]
):
    solver = planner.trajopt_solver

    if len(planner.tool_frames) != 1:
        raise ValueError("此适配函数要求一个已配置的 TCP。")

    expected_tail = (solver.action_horizon, solver.action_dim)
    if seed_bank.ndim != 4 or tuple(seed_bank.shape[-2:]) != expected_tail:
        raise ValueError(
            f"seed_bank 应为 [1, seeds, {expected_tail[0]}, "
            f"{expected_tail[1]}]，实际为 {tuple(seed_bank.shape)}"
        )

    current = JointState.from_position(
        q_current, joint_names=planner.joint_names
    )
    terminal = JointState.from_position(
        q_goal, joint_names=planner.joint_names
    )

    goal = GoalToolPose(
        tool_frames=planner.tool_frames,
        position=goal_position.reshape(1, 1, 1, 1, 3),
        quaternion=goal_quaternion.reshape(1, 1, 1, 1, 4),
    )

    # 使用固定 solver profile 的种子数量，不随 skill 名称修改。
    count = solver.config.num_seeds
    terminal_seeds = q_goal[:, None, :].expand(
        -1, count, -1
    ).contiguous()

    result = solver.solve_pose(
        goal_tool_poses=goal,
        current_state=current,
        goal_state=terminal,
        seed_config=terminal_seeds,
        seed_traj=seed_bank,
    )

    # 此处仅返回候选；不能在这里直接认定“抓持成功”或下发硬件。
    return result
```

`solve_pose()` 和 `seed_traj` 路径在源码中真实存在；上面的四维种子布局按 `SeedManager` 的实际检查写，而不是只照抄上层 docstring。[GitHub](https://raw.githubusercontent.com/NVlabs/curobo/main/curobo/_src/solver/solver_trajopt.py)

**还需要写的是适配代码，不是新的算法：**

- 把当前关节状态、目标 TCP、场景转换到同一约定。
- 把演示 IK 路径或 OMPL 路径重采样成种子。
- 读取返回轨迹，统一验证并交给控制器。

不能直接把离线视频中的 `[N,4,4]` 位姿数组填进 `seed_traj`；那是 Cartesian 轨迹，不是关节初始解。

------

## 6. 什么时候允许开始执行？

这里应该采用**可检查的放行条件**，而不是“对技能语义有足够信心”这种无法落地的条件。

对返回候选，检查：

**实际起点正确；终端 TCP 满足演示抓持目标；关节与运动限制满足；整段路径没有禁止碰撞；已明确声明的过程要求通过；夹爪事件可执行。**

碰撞不能只检查优化器的稀疏节点。OMPL 文档明确指出，离散运动检查过粗会漏掉节点之间的碰撞；检查精度属于共享验证配置。[Open Motion Planning Library](https://ompl.kavrakilab.org/stateValidation.html?utm_source=chatgpt.com)

通过后，后端输出：

```
CANDIDATE_READY
    timed_joint_trajectory
    gripper_events
    validation_report
    scene_version
```

Executor 执行；执行中目标移动、跟踪失效或控制异常则中断／回报。终端由观测确认是否真的建立抓持，不把规划成功直接写成 `Holding = TRUE`。这正是 TaskFlux 文档要求的“候选与实际后果分开”。   

如果失败，报告也应当具体：

```
INVALID_START
NO_VALID_GOAL_IK
NO_PATH_WITHIN_BUDGET
COLLISION_VALIDATION_FAILED
EXPLICIT_CONSTRAINT_VIOLATION
EXECUTION_FAILED
```

**“有限预算内没找到路径”不是“这个技能理论上不成立”，也不是“现在必须增加一种新表示”。** TaskFlux 已要求区分这些情况。





源码核查发现三个关键问题：

- 当前编译器把演示后缀默认锁为逐帧位姿约束；取消硬锁时，必须同时保留真正的目标消费。
- 固定cuRobo的dt仍不够：现有OMPL分支会改变progress，进而移动夹爪事件。
- 接触候选即使获准尝试，仍可能被下游旧筛选条件再次拒绝；独立场景验证和执行交接也尚未贯通。