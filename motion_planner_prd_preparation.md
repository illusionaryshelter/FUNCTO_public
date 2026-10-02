我们此前的问题几乎都可以映射到“某个需求没有写清楚”。我们现在做的基本就是**机器人 manipulation motion stack 的标准工程问题，我们要停止继续自己发明通用 manipulation infrastructure**。

以后遇到问题先看：

```
MoveIt / MTC 是怎么定义这个问题的？
Isaac ROS cuMotion 是怎么实现的？
这些库里的issue是否已经有人帮我们踩雷了？
```

然后把成熟的概念和行为移植进现有系统。

比如：

```
Stage
PlanningScene diff
Allowed Collision
Attach/Detach
Fallback
Cartesian Connect
```

这些概念直接借就行。

比如：

- “夹爪 open/close”没有规定究竟是二值语义，还是实际双指状态；
- “自碰撞”没有规定 native geometry、sphere proxy 谁是真值；
- “unknown space”没有规定当前机器人已经占据但相机看不到的体积怎么算；
- “轨迹合法”没有规定 planner 内部合法、独立验证合法、物理执行合法三者是否都要通过；
- “pre-grasp”没有规定来自演示阶段边界，还是从 grasp pose 人工后退固定距离；
- “允许碰撞”没有区分合法指垫接触和真正自碰撞；
- “position tolerance”没有区分 numerical tolerance 和 skill-level `pose_bounds`；
- “finger trajectory”没有规定 nominal reference 和 actual state 必须分开；
- “碰撞安全”没有提前说明必须在生成阶段考虑，而不仅是生成后 reject。

这些都不是科研理论问题，而是**系统语义没有冻结**。

所以我认为现在补一份完整规格文档，价值非常高。

但我不会把它写成传统互联网 PRD。我们真正需要的文档结构应该长这样。

------

## 1. System boundary

第一段就明确系统负责什么、不负责什么。

例如：

```
Input:
- demonstration-derived interaction representation
- current RGB-D scene
- robot state
- robot model

Output:
- executable synchronized arm + gripper trajectory

System must:
- preserve demonstrated interaction semantics
- satisfy robot limits
- avoid forbidden collision
- handle unknown-space conservatively
- execute through robosuite/MuJoCo
- independently validate execution

System does NOT:
- independently invent a new grasp strategy
- infer unobserved space as free
- treat planner command as actual robot state
```

这部分能直接防止我们前面讨论时出现的“又加 grasp candidate generator”这种方向漂移。

------

# 2. 明确所有 state 的语义

这可能是最重要的一节。

统一定义：

\[ q= [q_1,\dots,q_7,q_{f1},q_{f2}] \]

其中：

- \(q_1\dots q_7\)：机械臂实际关节位置；
- \(q_{f1},q_{f2}\)：两个独立 finger 实际位置。

然后明确区分：

```
q_ref(t)      planned/reference state
q_cmd(t)      controller command
q_actual(t)   simulator/robot feedback
```

禁止任何模块默认：

```
q_cmd == q_actual
```

我们前面花不少时间才把这件事彻底拆开。

如果 PRD 一开始就有这条 invariant：

> **任何安全验收只能使用 actual state；规划检查可以使用 nominal state，但不得混用。**

很多问题会提前消失。

------

# 3. Geometry source of truth

明确写：

### 仿真阶段

```
MuJoCo native geometry
    = execution geometry truth
```

### 独立检查

```
Pinocchio + Coal
    = independent exact geometry validator
```

### 快速规划

```
cuRobo spheres
    = conservative planning proxy
```

并且明确：

> sphere overlap 不等价于 native geometry collision。

这一条如果早写，43.9 mm 假 penetration 根本不会成为一个长期模糊问题。

------

# 4. Collision semantics

把 collision pair 分类固定下来：

```
FORBIDDEN
ALLOWED_CONTACT
CERTIFIED_IGNORE
```

比如：

```
finger_pad ↔ finger_pad
    ALLOWED_CONTACT

link0 ↔ link1
    CERTIFIED_IGNORE

forearm ↔ mug
    FORBIDDEN
```

并定义：

- 哪些作用于 planning；
- 哪些作用于 execution；
- 哪些由 model 静态声明；
- 哪些由 phase 改变。

这样就不会后来才发现：

> “允许指垫接触”跟“忽略整个夹爪 self collision”完全不是一回事。

------

# 5. Unknown-space semantics

我们这次踩的坑特别适合直接写进规格。

明确：

\[ V_{\text{new}}(q)=V(q)\setminus V_0 \]

其中：

- \(V_0\)：实测起点机器人实体占据；
- \(V(q)\)：未来状态机器人实体占据；
- \(V_{\text{new}}\)：未来新增占据体积。

要求：

> only newly occupied volume requires free-space evidence.

同时明确：

```
unknown != obstacle
unknown != free
```

以及：

```
current robot occupancy does not need to be re-proven by depth
```

这会直接避免“416 个 sphere 全部 unknown → 起点非法”的设计。

------

# 6. Motion phase contract

不要写 task-specific 规则，而写阶段语义。

例如 grasp：

```
FREE_CONNECT
    current → pre-grasp

LOCAL_APPROACH
    pre-grasp → contact/grasp pose

GRIPPER_CLOSE
    arm reference approximately fixed
    fingers follow synchronized trajectory

POST_GRASP
    object attached / coupled
```

同时明确：

- free segment 可以自由规划；
- local approach 受 demo-derived reference / pose_bounds 限制；
- close phase finger state 是真实轨迹；
- post-grasp 使用实际闭合后的 finger state。

这样就不会以后又争论：

> “free-plan 到 grasp pose 到底是不是合理？”

因为系统已经定义了 free 段边界。

------

# 7. Planning contract

这里特别关键。

不要只写：

> “使用 cuRobo + OMPL”。

而写：

### planner 输入必须包括

```
robot state
scene geometry
joint limits
collision rules
unknown-space rule
interaction reference
pose_bounds
finger reference when relevant
```

### planner 输出必须包括

```
trajectory
success/failure status
constraint residuals
planner provenance
```

### 明确规定

> 返回 trajectory object 不等于 planning success。

只有：

```
planner success
+
independent validation success
```

才允许执行。

我们现在 view_a 出 5.16 rad 这种情况，就能马上判断到底是：

- solver failure diagnostic candidate；
- 还是 planner 错误宣称 success。

而不是之后再猜。

------

# 8. Execution contract

明确：

```
robosuite controller
→ MuJoCo
```

负责真实物理执行。

每个 physics step：

```
read q_actual
read qdot_actual
read torque
read contacts
run Pinocchio/Coal validation
```

失败立即停止。

并明确：

> execution validity is based on actual state, not planned state.

我们现在已经做到这一点了，应该把它冻结成要求，不要以后被重构破坏。

------

# 9. Tolerance hierarchy

这个非常值得写清楚。

至少区分：

### numerical tolerance

比如：

```
optimizer convergence
float comparison
collision epsilon
```

不能携带 task semantics。

### skill-level bound

```
pose_bounds
```

表达：

> 这个局部动作真正允许偏多少。

### physical acceptance

例如：

```
joint limits
force limits
penetration tolerance
tracking tolerance
```

这三类不能共用一个 `epsilon`。

我们这次 0.1 mm → 1.3 mm 的问题就是一个非常好的案例。

------

# 10. Acceptance ladder

我认为这一节会极大减少我们以后“到底算不算成功”的争论。

定义固定层级：

```
L0 Model Validity
L1 Static State Validity
L2 IK Feasibility
L3 Planned Trajectory Validity
L4 Independent Geometric Validation
L5 Physical Execution
L6 Interaction Success
```

比如当前 `grasp_mug`：

```
L0 ✓
L1 ✓
L2 ✓
L3 ✗
L4 未进入完整成功路径
L5 未执行
L6 无结论
```

这样就不会出现：

> “抓杯失败，是不是 controller 的问题？”

因为根本没到 L5。

这个非常工程化，而且非常有用。

------

# 11. 每个 requirement 必须带 acceptance test

这是区别“好 PRD”和“漂亮废话”的关键。

例如不要写：

> Gripper shall be collision-safe.

而写：

> 在完整允许的 open→close reference 下，每个 physics step 使用实际 finger state 检查 native geometry；仅模型声明的 pad-pad contact 可发生；其他 self-collision 均导致停止。

对应测试：

```
empty_gripper_close
2s / 1000 physics steps
expected:
    pass
```

你现在已经有这个测试了。

再比如：

> Free-space planning must not enter unknown newly occupied volume.

对应：

```
known-start / unknown-future-region test
```

不是一句抽象要求。

------

# 12. 建议增加一个“禁止事项”章节

这反而是我们这种项目特别需要的。

例如明确写：

```
MUST NOT:
- generate new grasp poses unless explicitly enabled
- infer free DOF from missing observation
- treat sphere overlap as exact collision truth
- treat unknown as occupied at the current robot volume
- use full finger travel envelope as current hand shape
- use controller command as measured robot state
- execute a trajectory that only passed planner-internal checks
- globally relax numerical tolerance to satisfy one skill
```

这几乎就是我们过去几轮踩坑的反面总结。

而且它能防止未来代码重构把老问题带回来。

------

## 所以，之前那些失败是不是“PRD没写好”？

**有相当一部分是。**

更准确地说：

> 缺的不是产品功能列表，而是**系统行为规格**。

有些 bug 即使 PRD 很好也不可能提前消灭，比如：

- sphere 拟合具体会差 43.9 mm；
- robosuite 第一个 2 ms 会出现 16.86 N·m；
- finger tracking 会落后 16.4 mm；
- 某个 optimizer 会在折角留下 1.325 mm residual。

这些必须实际跑才会发现。

但是好的规格会让你看到这些现象以后，很快知道：

> 它违反了哪一条约定，应该改哪个模块。

而不是像前面那样需要重新讨论：

> “是不是 FUNCTO 思路错了？”
> “是不是要换 Crocoddyl？”
> “是不是需要新的 grasp generator？”
> “是不是要放宽所有 tolerance？”

这就是 PRD 真正能节省的时间。

------

我现在反而建议我们**暂停新增功能一小段时间，把这几轮已经确定下来的规则写成一份冻结版 System Requirements / Planning & Execution Contract。**

不必写成几十页。

如果写得够具体，大概 **10–15 页正文 + acceptance test 表格** 就已经非常有价值。

而且现在是写它的好时机：**太早写我们还不知道真实问题是什么；现在我们已经踩过足够多的坑，但系统还没有复杂到无法收敛。**

如果你愿意，我下一步可以直接基于我们这整个讨论，给你整理成一份**可直接交给工程实现的 PRD/系统规格草案**，不是泛泛的项目说明，而是包含 interface、invariant、failure semantics、acceptance tests 和当前已知状态的版本。