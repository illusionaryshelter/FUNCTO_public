# FUNCTO 9a17cd3：规划耗时、几何目标与交付调度审计

## 结论与修改范围

本轮工作针对 `segment_002 / put_mug_on_shelf`。已执行 `git pull --ff-only`，实际审计提交为 **`9a17cd35ff38675c430205b0b4266b0a18d9ab5c`**。使用本轮附件的 `execution_binding.problem/scene/curve`；此前手动物理轨迹只进入事后比较脚本，不进入场景重建、目标、种子、筛选或优化。

**当前定位到三类主要成本，其中目标求值和交付调度已有局部修改证据；尚不支持删库重写或再换一套规划器。**

1. 已按当前库的资格策略接纳并冻结了一个初值，却仍强制等待主优化；最后预算耗尽才返回同一初值。本次需求是得到满足声明的执行轨迹，这段等待不是必需步骤。
2. 几何目标在实体对内部反复扫描大型观测实体的全部材料点；其后又为已确定为零的原生机器人碰撞项建立反向传播图。两处都可以在**不改变原目标及一阶梯度**的前提下消除。
3. 初值的原有候选验收本身仍慢。独立重放总耗时 **56.94 s**，其中区间几何 **49.11 s**。因此，本报告不把“一次目标求值缩短到约 2.25 s”写成“整条规划已经低频实时”。

最新代码已经修复上轮的部分任务行冗余、逐 seed 延迟冻结和初始几何失败后才搜索的问题。本文不把这些旧问题重复归罪于当前提交。

生产文件没有修改；对照原型、调度补丁和所有结果在独立审计目录。生产合并应修改现有函数，删除被替换的全量扫描路径和多余图构造，**不保留审计用的双后端、兼容开关或第二套碰撞模型**。

---

## 1. 本次究竟测了什么

### 1.1 输入与证据身份

原始附件 JSON 的 SHA-256：

```text
ec5c83f801102ea9e7fba64311af6428edb4a514fb7fc81db2aa146b9aca5cd1
```

这个 JSON 约 407.8 MB，包含重复序列化的观测、模型和验收记录。它并非只有画图轨迹：其中有原始加权 TSDF、observed 掩码、free-ray coverage、两个观测 owner、原机器人模型的 16 个碰撞实体、机械 pair 策略、12 个接触声明和手指名义模型。无损提取后的观测数组压缩包约 **1.18 MB**，实际数值没有降采样或更换几何表达。来源清单见 `audit9_inputs/binding/extraction.json`。

当前入口直接取 `problem.start_q/start_dq`：

```text
q0 = [-0.412119495692506,  0.7051348412484442, -0.2867163369855215,
      -2.6345177383635767, 1.6903017585012015,  1.380879129124128,
       0.6315245701121296]
dq0 = [ 1.1690560991128032e-5,  6.294265991466382e-6,
       -1.3346866776402164e-5, -2.9508627994097865e-5,
       -4.616021532575278e-5,  -6.62881057457884e-5,
       -6.714916645979932e-5]
```

此前手动记录声明 `grasp_mug` 已完成；它不构成自动规划的新反馈或新碰撞地图。本次重建使用的是当前附件中的声明与反馈。

### 1.2 重建核对

`restore_case.py` 用当前生产类恢复场景、问题和曲线；它没有运行手动操作轨迹，也没有把场景任务物体转成 CAD/mesh。官方机器人自带的原生凸体和解析形状来自附件固定模型。

| 核对项 | 结果 |
|---|---:|
| 主优化查询时刻 | 84 |
| 运动积分普通时刻 | 81 |
| 自由控制量 | 21 × 7 |
| 曲线物理时长 | 10.4040002823 s |
| 保存曲线重建最大关节差 | 6.412 × 10⁻¹⁵ rad |
| 入口 q、dq 差 | 均为 0 |
| 重建 motion quality | 238.545047392245 |
| 重建 task quality | 224,360,148.482717 |

随后 `replay_acceptance.py` 直接复用当前 `_ConstrainedPlanner.evaluate_candidate` 的 CPU 路径和实际关节限制。重放得到的 `accepted`、所有残差、总目标值、893 个新增区间查询、852 个 proved 区间和 1,156 个 unresolved 区间，均与附件吻合。没有运行 CUDA IK 或真实控制器，因此计时范围明确是保存曲线的目标/导数与候选验收。

### 1.3 证据不能混用

用户提供的 **87.29 s 主优化、81.80 s 目标及梯度、99.56% 几何前向占比、三个未验收试探超差**，作为本轮原运行日志事实引用。这份轨迹附件保存的却是另一个对象：

```text
classification = planner_accepted
native.accepted = true
seed_provenance = local_reference_ik
optimizer.source = reference_initialization
unchanged_from_seed = true
maximum_control_update = 0
```

最后未验收试探的 2.761 mm、1.021°、3.812 mm，不能套到这个已保存初值上。独立复算保存初值：

| 声明指标 | 保存初值 | 任务上限 |
|---|---:|---:|
| 交互材料原点位置误差 | 0.287043 mm | 2 mm |
| 去除自由 yaw 后的 swing | 0.07967° | 1° |
| 释放材料原点位置误差 | 0.000230568 mm | 2 mm |

具体证据：`audit9_solver/attachment_evidence.json`、`support_preference_results.json`。

---

## 2. 代码层面的主瓶颈

当前主优化实际使用 **SciPy SLSQP + CPU Pinocchio + Torch 自动微分 + Coal/原 TSDF**。`planning.py:3691` 调用 `unified_solver.solve_trajectory`；后者声明并实现 CPU 轨迹目标。cuRobo 的 IK 使用并不意味着这个自定义几何目标自动获得 GPU 批执行。

主要代码定位：

| 位置（固定提交） | 当前职责及问题 |
|---|---|
| [`online_geometry.py:800`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/online_geometry.py#L800) | 按实体对、逐状态组织几何目标，材料对象已有缓存 |
| [`sampled_geometry_objective.py:64`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/sampled_geometry_objective.py#L64) | 固定材料样本及其原 cell 身份 |
| [`sampled_geometry_objective.py:169`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/sampled_geometry_objective.py#L169) | `_field_direction` 变换并查询全部源材料点后才选择负贡献 |
| [`sampled_geometry_objective.py:199`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/sampled_geometry_objective.py#L199) | `_native_direction` 先变换全部源材料点，再筛原生实体 AABB，再逐点调用 Coal |
| [`sampled_geometry_objective.py:298`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/sampled_geometry_objective.py#L298) | 原生距离非负时仍构造最近点、四元数与位姿反向图 |

**不能说代码完全没有 broadphase 或缓存。** 当前确实有整对包围盒筛选；实际一次目标调用的 3,194 个 field 相关 pair-state 中，2,855 个已被整对筛选省掉。缺口是：剩余 pair 内，大型观测实体的包围盒覆盖了小机器人实体附近，便继续扫描整片观测材料。

实际固定点数：

| 观测实体 | 原材料点数 |
|---|---:|
| 支撑物 `observed_support_owner` | 578,201 |
| 被持物 `observed_held_carrier` | 109,300 |
| 背景 `background` | 176,110 |

热态分解中，field→support 约 **6.60 s**，native→background 约 **3.14 s**。因此最大热点是大型观测实体的全量点扫描，不能只归因于“背景地图太大”。一次求值的重复源点坐标变换约 **78,894,229 次**；裁剪后约 **947,240 次**，保留约 **1.20%** 的处理量。

另有 **7,434 次**机器人原生实体对查询。保存曲线中这些距离均非负、几何目标为零，但原代码仍建立几千条零贡献的自动微分路径。它们的存在解释了为什么几何目标值为零，反向仍需要数秒。

---

## 3. 修改一：在变换之前查询原材料点的空间索引

### 3.1 最小实现

在现有 `MaterialSamples` 中为原观测 cell 的连续样本范围保存一次 `start/stop` 索引，复用 **已有的 `ObservedField.tree`**：

1. 把目标实体可能产生负值的包围盒，以当前相对位姿保守变换到源字段坐标；
2. 从原 cell 空间树查出与这个盒相交的源 cell；
3. 按原顺序取出这些 cell 对应的原材料点；
4. 执行原 TSDF/Coal 查询、原 observed/permission/cell-exclusion 逻辑；
5. 仍除以原完整材料点数。

实际额外索引存储 **1,155,808 bytes**，一次构造约 **0.011 s**。这是原有材料预处理的一部分，不需要另建地图、地图更新协议或跨请求缓存框架。新场景已有新的 immutable field/material 对象，索引随它们生灭。

审计原型为了对照复制了部分函数，生产中应把索引和选择逻辑放回现有类/函数，删除原型包装层。

### 3.2 等价证明

一个方向的现有有限样本代价可写为：

\[
J(C)=\frac1N\sum_{i=1}^{N}m_i(C)\left[\max\left(0,-\frac{d_i(C)}{\ell}\right)\right]^2.
\]

其中，\(C\) 是关节曲线控制量，\(N\) 是原材料点总数，\(d_i\) 是当前位姿下原 TSDF 值或 Coal 有符号距离，\(\ell\) 是原代码长度尺度，\(m_i\) 表示原支持域、有限接触许可和 cell 排除规则是否使该点参与。

设保守空间查询保留索引集合 \(S(C)\)。对于任意被省掉的点，查询必须证明它不可能落入目标的负值区域；原点要么不受支持、要么 \(d_i\ge0\)。因此，在原支持/许可分支内：

\[
i\notin S(C)\Longrightarrow J_i(C)=0,\quad \nabla_CJ_i(C)=0.
\]

于是用 \(S(C)\) 求和仍得到原 \(J\) 和一阶梯度。**分母仍是 \(N\)，不能换成保留点数 \(|S(C)|\)**；后者会随位姿改变权重，已经是另一个优化问题。

对原 TSDF，三线性插值是有效 cell 顶点的凸组合；全为非负的有效 cell 内不可能出现负插值。对原生实体，负距离区域在原实体 AABB 内。索引使用既有外包围算术处理刚体变换舍入与格点闭边界；这不是新增物理裕量。

未知空间仍由原验收政策处理。这里跳过的是原优化公式中已为零的工作，不把未知体素宣告成自由空间。有限 permission、被排除 cell、释放 anchor、名义手指有效区间都保留原定义。

### 3.3 已测试的范围

- 最新原型 63 组原值/14 维 pose 梯度对照：最大差均为 0。
- 两个独立 agent 分别检查 320 组闭盒/格点及 384 组尺度、公共大平移、旋转边界；后者逐项检查 74,636 个潜在贡献点，未遗漏。
- 当前完整场景三次原/裁剪对照：值及全部控制量梯度差均为 0。
- 同一场景的确定性非执行扰动产生 **62,706 个负贡献**，几何目标约 **157.755242**、最大控制量梯度约 **30,321.251**；改动前后仍一致。这个实验避免了只比较两个全零目标的弱证据。

仅此修改，热态完整 `objective_components` 前向加几何项反向的中位数从 **15.5689 s → 5.5263 s，约 2.82 倍**。这项分解试验的反向只求几何梯度；第 4.1 节才是完整标量及全部控制量梯度的最终对照。数据见 `audit9_geometry/actual_scene_results.json`。

---

## 4. 修改二：原生距离非负后，不建立零贡献的微分图

当前函数先调用 Coal，再无条件读取 witness、转换 witness、旋转点并构造位姿图，最后计算：

\[
g(d)=\max(0,-d/\ell)^2.
\]

这里 \(d\) 是原 Coal 有符号距离，\(\ell>0\) 为原长度尺度。直接求导可得：

\[
g'(d)=\begin{cases}2d/\ell^2,&d<0,\\0,&d\ge0.\end{cases}
\]

在接触边界 \(d=0\) 也可导且导数为零。因此最小修改就是：**原 Coal 查询返回 \(d\ge0\) 后立即返回同 dtype/device 的零标量；仅 \(d<0\) 构造原 witness 微分图。**

这没有删掉任何 Coal 距离查询，也没有改变碰撞阈值。`OnlinePlanningScene.optimization_objective` 已有一次全局输入零依赖；不必在每一对里再建立 `pose.sum()*0`。

测试包括原生球体正间隙、负间隙、二进制精确接触，以及实际场景与同一个非零碰撞扰动，值和一阶梯度一致。若以后把代价改为在正间隙内也产生作用的 clearance 代价，零贡献分支须按那条新公式定义，不能沿用当前 \(d\ge0\) 条件。

### 4.1 两项组合后的完整目标实验

下面测的是生产 `objective_components` 的三项之和，以及对全部 147 个控制量的梯度。三个配对在同一进程、相同输入、相同缓存状态下交替执行；没有 pair 级计时包装。

| 热态配对 | 原代码，s | 空间筛选＋零图短路，s | 完整目标差 | 最大控制量梯度差 |
|---|---:|---:|---:|---:|
| 0 | 16.8472 | 2.0467 | 0 | 0 |
| 1 | 20.0478 | 2.2475 | 0 | 0 |
| 2 | 47.9986 | 2.5921 | 0 | 0 |
| 中位数 | **20.0478** | **2.2475** | **0** | **0** |

中位数之比约 **8.92**。组合版本首次包含材料创建的调用为 **3.9105 s**。原版第三次确实有离群：进程 CPU 时间约 47.508 s，不能无证归咎于别的进程抢 CPU，也没有将其 18.5 倍配对比率当成代表结果。

这些是当前 CPU 环境的三组测试，不是用户工作站的固定倍率。它们证明当前调用组织存在大幅可消除的成本；不证明整个规划请求只需 2.25 秒。

具体原始记录：`audit9_geometry/final_objective_results.json`。审计原型中未降低精度、点数归一化、接触条件或任务上限。

---

## 5. 修改三：已有接纳结果时，结束本次首结果请求

[`planning.py:4333–4447`](https://github.com/illusionaryshelter/FUNCTO_public/blob/9a17cd35ff38675c430205b0b4266b0a18d9ab5c/manipulation/planning.py#L4333) 的当前流程是：

1. 构建 local IK 初值；
2. 原流程验收并通过 `qualify_candidate` 冻结候选和 binding；
3. 即使 `incumbent` 已存在，仍调用主优化；
4. 预算耗尽后，异常分支调用 `deliver_incumbent()`。

对于本次“得到满足任务的执行轨迹”的请求，第 3 步是隐含质量搜索。最小补丁在完成初值资格与冻结之后调用已有交付函数：

```python
if incumbent is not None:
    return deliver_incumbent()
```

测试没有更改生产文件，而是复制当前 `plan_trajectory` 到独立命名空间，只插入这两行，并运行仓库的 `MotionBackend.plan` 合同夹具。

| 项目 | 当前分支 | 两行补丁 |
|---|---|---|
| 实际调用 | seed → reference → export → solve | seed → reference → export |
| 主求解次数 | 1 | 0 |
| 初值验收次数 | 1 | 1 |
| 曲线所有数组 | 基线 | 与基线相同 |
| 冻结候选及核验字段 | 基线 | 与基线相同 |

账本中的实际消耗及独立请求 ID 正确地不同，不能宣称整个 JSON 位级相同。小补丁见 `audit9_solver/first_feasible_dispatch.patch`，测试见 `first_feasible_patch_results.json`。另有按完整文件定位的 `audit9_root/first_feasible_9a17cd3.patch`，已在固定提交上通过 `git apply --check`；实际生产文件保持未修改。

工程合并时，这个合同也应一致应用于现成 joint reference 和后续搜索/优化得到的首个接纳候选；不再无条件跑下一条 seed。现有候选冻结、清理、账本和交付函数都可复用，无需新增“快规划器”。

**本轮能确定的是省掉主优化这段额外等待；不能直接把 121.94−87.29 秒当成精确的首结果时刻。** 附件没有保存完整 `qualified_candidate_ready` wallclock；已有代码的该时间字段可在实际工作站报告中直接使用。

---

## 6. 法向偏好为什么不值得继续无条件优化

当前 `hard_components=['body','footprint']`。法向和 gap 是 quality 软偏好，不是本轮拒绝条件。当前 `unified_solver.py:173–179` 无条件请求 `phase='quality'`，而 `PlanningProblem.optimizer_preferences` 已有 `phase='feasibility'`。

独立复算：

| 项目 | 代价 |
|---|---:|
| 法向 normal | 223,991,444.9811 |
| gap | 366,008.9864 |
| 所有 task quality | 224,360,148.4827 |
| motion quality | 238.5450 |
| normal / 保存总目标 | **99.83556%** |

归一化分母不是单独的 `normal_tolerance=1e-4 rad`。代码取它与 `numerical.rotation_rad=1e-3 rad` 对应 chord 的较大值，实际约 **0.001**。忽略这点会把相对权重又夸大一百倍。

### 6.1 倾角下界的必要前提与证明

设 \(R_*\) 为声明的目标材料姿态，\(n_b\) 为杯底朝外材料法向，\(n_s\) 为支撑面法向，\(a\) 为目标坐标系中的自由 yaw 轴，\(\beta\) 为允许 swing。由于两个接触面朝外法向相反，准确地定义参考倾角 \(\alpha=\angle(-R_*n_b,n_s)\)，实际倾角 \(\theta=\angle(-Rn_b,n_s)\)，其中 \(R\) 是实际材料姿态。本例实际满足：

- 支撑与功能事件使用同一 `tool_from_carrier/tool_from_function` 材料关系；
- \(a=R_*^Tn_s\)，向量最大差 \(4.44\times10^{-16}\)。

因此自由 yaw 对所比较法向相对支撑法向的倾角不产生改善；允许 swing 至多改变这个夹角 \(\beta\)。由球面角距离三角不等式：

\[
\theta\ge\max(0,\alpha-\beta)
=2.6380175064^\circ-1^\circ
=1.6380175064^\circ.
\]

这里 \(\theta\) 是允许姿态集合内的剩余倾角。令 \(\epsilon=0.0001\) rad 为法向软阈值，\(s=\max(2\sin(\epsilon/2),2\sin(0.001/2))\) 为实际归一化 chord，\(w=10^5\) 为已有权重，则：

\[
J_n=w\left[\frac{\max(2\sin(\theta/2)-2\sin(\epsilon/2),0)}{s}\right]^2
\ge81,155,624.72.
\]

所以这个软目标在原姿态允许集合里无法归零。在一个仅向支撑面转动的姿态切片中，法向加 swing 中心偏好的最优点就在 1° 边界。这解释了向硬边界搜索的动机；它不能证明最后每一步超差都由此单独造成，也不表示硬约束整体无解。

### 6.2 对本次需求的处理

已有候选按原规则接纳时直接交付。未接纳时，首结果搜索使用已有 **`phase='feasibility'`**，保留原运动代价、几何引导、硬事件、硬支撑与真正过程关系。不要在这个请求中无条件启动 normal/gap/目标中心的 quality 改善。

这不是按 mug 特判删法向，也不是永久删除 gap 或技能表示。那些量仍可测量、记录，确有质量需求的调用再使用。普通搬运参考仍用于初值；不因为想降低 RMS 就把每个普通 waypoint 升级成必须经过的任务约束。

---

## 7. 还没有被前两项优化解决的验收成本与语义

`replay_acceptance.py` 使用相同当前场景、曲线和生产验收方法，未运行 IK/优化/控制器。一次 CPU 重放：

| 阶段 | 时间 |
|---|---:|
| 固定曲线 task/scene queries | 7.776 s |
| 连续关节多项式限制 | 0.004 s |
| 区间几何 | **49.105 s** |
| 原候选评分 | 0.052 s |
| 合计 | **56.939 s** |

对应 `accepted=true`、所有记录残差为 0，`fixed_objective=224360387.0277673`，与原附件一致。这是本机的一次候选重放，不能替代用户整机耗时；但足以证明优化目标提速之后，验收仍需单独处理。

原 native 记录与重放均包含：

| 连续几何记录 | 数量/范围 |
|---|---:|
| 新增查询时刻 | 893 |
| proved 区间 | 852，覆盖 4.103099 s |
| unresolved 区间 | 1,156，覆盖 6.300901 s |
| 因 sampling capacity 未完成 | 1,118 |
| 因 spatial resolution 未完成 | 38 |
| 已确认相交 | 0 |

**“按当前资格策略接纳”不是“所有连续时刻均已证明无碰撞”。** 当前 `evaluate_candidate` 汇总已查询位置的 `hard_scene_residual`，不将 unresolved 数量直接变成拒绝项。反过来，unresolved 也不等于已经发生碰撞，不能因本报告就给用户增加 1,156 个新拒绝条件。

本轮直接交付补丁返回的仍是原代码在预算耗尽后会返回的同一候选，资格规则没有放松；它不依赖重新定义 unresolved。

已发现的多余工作：

- `_candidate_metrics` 已查询的几何结果没有交给区间过程；`check_interval_geometry` 已有 `initial_states` 接口，调用方却没用。可按同一不可变曲线、同一时刻/关系侧/释放 anchor 复用原结果。首次查询必须仍只计费一次。
- `SceneMotionBounds.partition_times` 把 1,898 个参考进度时钟节点都作为几何分区。对当前静态/附着实体，连续几何并不随每个这样的节点改变。一般外部运动 owner 的原式乘以区间时长后也能界住总运动量，因此数学上可以跨普通时钟折点；不过改变分区会改变查询调度，不能与目标求值的代数等价混为一谈。

区间阶段的进一步分解与数学审查见第 11 节。现有实测没有证明 1 Hz、5 Hz 或任何用户工作站上的端到端指标。

---

## 8. 为什么这些修改有成熟实现依据

本报告借鉴的是实际执行组织，而不是拿另一个库的宣传耗时作为本场景承诺。

| 一手实现/实机材料 | 可对应的工程方法 | 在本仓库的具体使用 |
|---|---|---|
| [FCL 官方 broadphase](https://github.com/flexible-collision-library/fcl#interfaces) | 先用空间结构排除不可能的实体，再精查 | 用既有 cell tree 先选原材料点，保留原精查公式 |
| [cuRobo V2 固定源码 `wp_autograd.py`](https://github.com/NVlabs/curobo/blob/78fd485fa82d9b9a063fb4985e371814587e666a/curobo/_src/geom/collision/wp_autograd.py#L70) 与同提交 `checker_collision.py` | batch×horizon×实体的批工作组织、结果/梯度缓冲 | 当前先消除无效 work 和零图；仍有瓶颈时再把相同剩余查询批量下沉 |
| [OMPL/VAMP 官方教程](https://ompl.kavrakilab.org/vampPlanningTutorial.html) | 整条连接的检查在一次原生调用里完成 | 避免把已经成批的状态重新拆成大量解释器调用 |
| [MoveIt ACM 官方说明](https://moveit.picknik.ai/humble/doc/concepts/kinematics.html#allowed-collision-matrix-acm) | 从显式机械关系预先定义需要检查的 pair | 本仓库已有完整机械策略，继续复用，不臆造豁免 |
| [ROS-Industrial：Tesseract/TrajOpt 实机项目](https://rosindustrial.org/news/2022/2/21/using-tesseract-and-trajopt-for-a-real-time-application) | 连续碰撞与真实物理控制器、取得结果后交付 | 明确首结果与继续改善的请求合同，不把总预算当强制等待时长 |

FCL 的默认最小距离 manager 返回全局最近 pair，不能直接替换这里对全部负残差的求和。VAMP 的原生例子使用几何 primitives，不能直接替换此处的原 TSDF/support/permission。cuRobo 也不能仅靠换入口就获得当前自定义几何语义。

ROS-Industrial 项目确有物理控制器验证，并明确说未强制硬实时期限。**这些一手资料支持架构原则；本轮局部代码原型的证据是实际输入等价重放，尚没有真机部署测试。** 两者的证据等级没有混为一谈。

---

## 9. 建议合并顺序与明确不做的改动

### 第一批：已经有闭环证据的修改

1. **交付调度**：本次首结果请求在现有完整资格与冻结之后立即调用 `deliver_incumbent`；普通 joint reference 和后续候选同一合同。
2. **几何计算**：在现有 `MaterialSamples` 和 `_field_direction/_native_direction` 中合并原 cell 索引筛选；原分母、点身份、许可、精度、释放关系不变。
3. **原生零项**：`native_pair_objective` 在原 Coal 查询确定距离非负后直接返回零，不建立无贡献图。
4. **目标阶段**：尚未得到首个接纳候选时使用现有 feasibility 偏好；不要自动把本例的法向质量取舍塞进必须等到结束的搜索。

第 1–3 项已有对应控制流和实际输入测试。第 4 项是任务合同修正，不应包装成目标数值等价：它有意不执行未经请求的质量改善，原硬允许集合不变。

### 下一批：区间验收

按补充 profile 处理现有区间过程的主要成本。已有输入和原数组查询 API 足够使用，无需增加另一套验证器。第 11 节的实测已经否定“只删参考时钟分区就能提速”的猜测，所以本轮不把它列入立即合并的性能修改。后续批处理应保留原事件/关系侧、释放 anchor、点预算和分离判据，并分别记录查询调度变化。不能为了跑得快扩大物理容差。

### 本轮没有证据支持

- 增加曲率、挠率门槛来解决速度；
- 继续增大优化预算或把 `ftol` 当成本次终止根因；
- 批量降低点数、减小验证范围、把未知当自由来制造提速；
- 再改 IK 候选链以解释本轮已接纳初值后的等待；
- 整库迁移到另一个优化器、以新 GPU 后端重写所有几何；
- 保留两个数值不同的几何目标，通过兼容模式选择。

`ftol` 控制 SLSQP 的数值停止条件，而不改变单次几何工作量；本次终止仍是累计预算。当前同一 `x` 缓存也有效，不存在每次 fun/jac 都完整重复一次前向的旧式问题。小夹具中有 `nfev=10/njev=9` 却做 10 次 backward 的次要浪费，但不足以解释主要前向成本，优先级低于上面的实测热点。[SciPy 官方说明](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html)

---

## 10. 复现文件、测试边界与独立审查

主要机器可读证据：

- `audit9_inputs/binding/extraction.json`：原附件 hash 和无损观测提取；
- `audit9_root/restoration_results.json`：场景/问题/曲线复原核对；
- `audit9_geometry/actual_scene_results.json`：实际场景裁剪对照与非零扰动；
- `audit9_geometry/native_shortcut_results.json`：原生零图短路、接触两侧对照；
- `audit9_geometry/final_objective_results.json`：最终同进程完整目标/梯度配对；
- `audit9_solver/support_preference_results.json`：法向、gap、事件误差及下界；
- `audit9_solver/first_feasible_patch_results.json`：两行调度补丁的隔离实验；
- `audit9_solver/interval_acceptance_evidence.json`：原记录的区间证据；
- `audit9_root/acceptance_replay_results.json`：原候选验收的独立实测。
- `audit9_geometry/recorded_batch_results_initial.json`、`recorded_batch_results.json`：固定原查询点的批量重放、原日志数值差异及同环境定位。

复现说明见 `audit9_root/README_REPLAY.md`。交付压缩包包含代码原型、机器可读结果、原观测的紧凑提取和事后比较所需文件，未重复打包原大型 JSON。`AUDIT_MANIFEST.json` 记录逐文件 SHA-256。

环境使用 Python 3.12、NumPy 2.3.5、SciPy 1.17.0、Torch 2.14.1 CPU、Pinocchio 4.1.0、Coal 3.0.3。性能对照设置 Torch/BLAS 单线程，三次配对完整保留。cuRobo 固定源码基线为 `78fd485fa82d9b9a063fb4985e371814587e666a`；本机没有运行 CUDA IK。机器人原生批查询扩展源码在本次拉取中未变。

三名独立 agent 分工检查几何热路径、求解/目标合同、成熟实现与数学边界，并交叉审阅：具体纠正了“没有 broadphase”“全是背景热点”“accepted 等于连续全时刻证明”“法向是硬约束冲突”等过度结论。根 agent 独立恢复原输入并重放完整候选验收。各自的 `READ_SCOPE.md/CROSS_REVIEW.md` 记录了读取区间、测试和失败后修正，没有宣称读过未读文件或完成未运行的真机实验。

## 11. 区间验收进一步剖析：否定了一项看似直接的提速办法

### 11.1 慢在大量单点场景回调

使用高层包含时间和自身时间分解，再次运行原区间流程，没有改变时刻、网格、判据和预算。计数、证明范围和接纳结果与原文件相同。该次区间耗时 29.766 s，与前一轮 49.105 s 存在明显运行间波动；它是 profile，不作为性能改动后的收益。

| 原区间流程中的工作 | 次数 | 此次包含时间 |
|---|---:|---:|
| 场景边界回调 | **890** | **24.015 s，约 80.7%** |
| 其中单物理时刻回调 | **889** | 包含在上项 |
| 区间运动界 | 2,119 | 1.892 s，约 6.4% |
| 接触 permission 构造 | 10,680 | 5.621 s |
| 候选机器人状态构造 | 890 | 2.516 s |
| native-field 原查询 | 33,093 | 2.069 s |
| field-field 原查询 | 1,790 | 1.436 s |

包含时间有嵌套，**不能将这些行相加**。首个回调查 4 个物理时刻，关系侧展开为 6 个状态；之后 889 次都只有一个物理时刻。当前 `query_geometry(times_s, ...)` 本来就接受数组，批态场景 API 也已经存在，但调度把它使用成了单点接口。

证据：`audit9_geometry/interval_profile_results.json`。这使下一处应重构的职责变得具体：**同一候选的区间查询组织**，而非换掉 Pinocchio、Coal 或再添加验收层。

### 11.2 只删普通参考时钟分区，未减少主要计算

独立数学审查后，做了一个单变量反事实：只从原 `partition_times` 删除全量 `trajectory.progress_u`，不添加关节节点，不改变其他几何/关系边界、预算、接受政策或原距离判据。

| 项目 | 原流程 | 只删 clock 分区 |
|---|---:|---:|
| 初始分区节点 | 1,898 | 88 |
| 新增物理查询 | 893 | 893 |
| 场景回调 | 890 | 890 |
| 已证明时间范围 | 4.103099 s | 8.667039 s |
| unresolved 时间范围 | 6.300901 s | 1.736961 s |
| 接纳结果/任务残差/固定目标 | 基线 | 同本例基线 |

该次区间计时 59.061 s，与原 profile 不是配对计时；不拿它算回归或加速倍数。**确定的事实是回调和实际查询数量没有下降，所以不能把这一删除当成本轮速度修复。** 它改善了相同点预算的覆盖分配，仍未消除反复进入场景的成本。这个建议被实际测试筛掉，没有交给用户试错。

### 11.3 跨 clock 折点的证明为何成立，以及它不证明什么

设原单调参考进度映射为 \(s(t)\)，owner 的归一化路径参数为

\[
r(t)=\operatorname{clip}\bigl((s(t)-a)/(b-a),0,1\bigr).
\]

其中 \([a,b]\) 是 owner 的进度区间，\(t\) 是物理时间。令 \(K\) 为所跨路径段中任一材料点对 \(r\) 的导数范数上界。单调性给出该点整个区间的运动长度界：

\[
C_{\rm owner}\le K\,[r(t_1)-r(t_0)].
\]

原 `_owner_speed` 的数值是 \(K\Delta r/\Delta t\)，而当前区间消费者只使用它乘以 \(\Delta t\)。因此得到的总运动界仍有效，不要求瞬时进度速率不变。改用最大 \(ds/dt\) 也会给合法界，却可能在长等待、短快动时无谓放大界，增加细分；本轮没有这么改。

对两个实体的合计运动长度界 \(C=C_A+C_B\)，假设中途某时刻接触。同一对接触材料点在前后两段运动长度分别约束端点距离，于是端点几何距离下界 \(d_0,d_1\) 必须满足 \(d_0+d_1\le C\)。原来的 \(\max(d_0,d_1)>C\) 或 \(\min(d_0,d_1)>C/2\) 仍足以排除中途接触。它不需要假设物理中点前后各移动一半。

这个证明依赖保留模式/关系切换、原 owner 路径、名义手指有效边界及有限许可的 **full-entity** 距离界。若将这条整理合入生产，应把外部 owner 的量准确描述为区间运动界，不继续泛称为跨折点的瞬时速度上界。完整推导见 `audit9_industrial/CLOCK_PARTITION_REVIEW.md`。

**证明有效，不代表改变查询调度后所有输入的接纳决定相同。** 当前库允许 unresolved；旧时钟节点可能恰好查到一个薄障碍，新分区可能在原空间分辨率处以 unresolved 结束，反过来新的预算分配也可能发现原调度未查询的碰撞。独立审查给出了总运动 0.5 mm、小于原 1 mm 空间终止尺度的明确一维反例。不能把这一调度变化包装成与第 3、4 节相同的目标/梯度等价改动。

### 11.4 下一处重构的最小边界

使用原 `query_geometry` 数组入口，将同一待处理区间集合中已知的缺失端点/待查点成批提交，复用现有 scene 方法。保持原距离、原关系侧与释放处理，不建立第二个几何后端。原先已查询的同一候选证据可经现有 `initial_states` 进入同一过程。

这是查询调度重构，会改变预算下的深度优先顺序；合入前应看实际曲线的查询/证明范围/接纳结果，而不是只看 callback 数量。它也不等价于预先知道未来的所有自适应查询点。本轮已有确认的交付与几何目标修改不依赖这个后续改动。

### 11.5 已完成的固定原查询点批量重放

进一步使用原 native 日志的 **893 个查询时刻**，从生产 `_candidate_geometry` 获取其真实 `query` 闭包，一次性传入原数组 API。原曲线、观测、关系侧、release anchor 和所有几何方法不变，临时截获的只是外层调度入口；这次没有执行自适应二分。

| 项目 | 结果 |
|---|---:|
| 场景边界调用 | 1 次 |
| 物理查询时刻 / 关系侧状态 | 893 / 895 |
| 首次 / 定位重放 callback 耗时 | 10.336 s / 9.613 s |
| 原分离与刚性关系证据核对 | 117,576 项 |
| 证据缺失 / 原严格判据不再成立 | 0 / 0 |
| 原 scene 硬残差及四类残差 | 全为 0 |

这里实际重新核对了原 852 个 proved 区间的各实体对端点界，以及原 `max(distance)>change` 或 `min(distance)>change/2` 谓词，不是只看总残差为零。

原日志的端点数值并非全部复现到同一浮点数：226,632 个比较中，56,899 项不完全相同，最大差 **4.370119 × 10⁻⁶ m**。首次 `checks_passed=false` 的原结果保存在 `recorded_batch_results_initial.json`，没有通过放宽原 `1e-12` 比较阈值把它改成成功。

随后在同一恢复场景和已经固定的 release history 中，对最大差异所在的 **4 个物理端点、688 个距离界**逐点复查：batch 与 serial 的最大差为 **0**，实体状态及 hard 残差相同。这些端点相对原运行日志的差异也出现在当前环境的 serial 结果中，因此不能将它归因于 batch。当前证据没有进一步区分原运行环境、数值库或记录恢复对这些微小差异的贡献，也没有把 4 点复查扩展声称为全部端点的逐位一致。

这项实验表明，现有数组接口能处理同样的原查询点，并保持这批原证据；但 **9.6–10.3 秒仍是离线已知时刻的一次接口成本**。原区间 profile 与它不是同进程配对，不据此宣称整体验收加速多少倍，更不能称为已经实现在线自适应批调度。下一步应在现有待处理区间中实现并验证在线批调度，已有证据不支持再造碰撞后端。

## 12. 当前轨迹与此前文件的事后比较

当前保存曲线的 TCP 路径长 **0.867915 m**、关节向量路径长 **6.002491 rad**。R9 分别为 **1.579343 m**、**8.609257 rad**；上轮诊断曲线分别为 **3.189201 m**、**17.292150 rad**。当前初值的几何形状已明显改善，本轮耗时不能再归因于早先那条发散初始化。

必须说明参考身份：本次 `execution_binding.problem` 重建的连续参考，与此前上传的 `reference-bound.npz` 并不完全一致；两者物理时钟逐项一致，但最大位置差 **13.176 mm**、最大完整姿态差 **0.102468 rad**。因此保留两个独立计算口径，不混报 RMS：

| 同一当前曲线的比较基准 | 位置 RMS | 完整姿态 RMS |
|---|---:|---:|
| 本次 binding 重新构造的参考 | 8.8168 mm | 0.065288 rad |
| 此前上传的冻结参考 | 2.8703 mm | 0.007514 rad |

本次性能对照不依赖这两条参考相同：它们使用同一实际关节曲线、同一 task、同一原场景进行直接目标/梯度比较。

手动物理记录长 **30.5 s**，包括松爪、撤离和静置；TCP 路径长 **0.891621 m**，臂关节向量路径长 **4.374336 rad**。其入口臂 q/dq 与本次反馈的最大差均为 0。这里只比较事后几何量，不按不同执行时钟计算逐时刻 RMS，也未把手动路径送进任何规划调用。

所有数值及来源见 `audit9_root/trajectory_metrics.json`，脚本明确在规划输入、当前参考与当前曲线计算完成后才打开手动记录。
