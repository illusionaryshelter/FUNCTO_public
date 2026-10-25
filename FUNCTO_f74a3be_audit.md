# FUNCTO f74a3be：抽屉把手抓取规划失败审计

审计日期：2026-10-25。固定源码：`f74a3bea18715618138a6e45f5945c2e11eb6f5f`。

任务是 `segment_003` 的 grasp 类抽屉把手抓取。工作范围是源码审计、附件复算、独立数值实验和一个局部修复建议；没有执行机器人。附件 `satisfied_dependencies` 为空，未提供前置动作完成记录，不能把空记录解释为某个前置动作已经完成。入口采用本次保存的具名 q、dq，之后又与同提交的 `scene/drawer_original/robot_feedback.json` 逐元素核对一致。

## 1. 决策：只推荐已证明的局部修复，不再用重构代替根因证据

**当前可以确认并修复的问题：恢复搜索选中关节越界的功能 IK 提案，而接收提案的投影函数要求输入已经在同一关节范围内。** 这使一个下游必然拒绝的状态消耗一次有限尝试，并可能遮住已有的有界提案。独立两行补丁、五个直接反事实和应用补丁后的原测试已完成。

**这个修复不是“本次完整抓取已修好”。** 用户报告的主 SLSQP 失败在恢复搜索之前发生。当前附件没有完整运行报告、原始初始化记录、实际导数界限、原运行 cuRobo 配置和冻结几何场；本审计没有重放原第 10 次迭代的整个非线性问题，也没有交付通过原碰撞验收的完整新轨迹。

本次还主动检验了两个修改方向，结果不足以支持生产改动：

- 以输出帧数改变导致静态收敛不同为由重写 PinKinematics：本轮实际机器人上的同输入 FK、任务值和 Jacobian 不支持这个判断。
- 直接把静态求解换成有界 TRF，或者给 TRF 改用同代价的向量残差：在固定三个入口、原 2 mm／1°和固定函数评估额度下，六次试验全部失败，不推荐生产化。

因此，没有足够证据删除整个规划器、Pinocchio 桥、三次曲线或 SLSQP。也没有足够证据宣布主优化层没有问题。可靠的结论是：一个局部接口矛盾已经证明，主数值失败的具体机制仍未完成复现；二者必须分开。

## 2. 输入、版本与执行状态

### 2.1 原始附件

| 附件 | SHA-256 | 本次用途 |
|---|---|---|
| grasp_drawer的参考轨迹.zip | `b806c71c423d007270d71611224892e16d2b177f851e42557454860957736a3e` | 642 个同时间参考、失败 FK、已报告机器人几何位姿 |
| f74a3be的轨迹(1).zip | `c319ac5b2db06502a9cfaade1867f9db29ee027da8eaf5f728098ba69cf28ec9` | 21 个关节 Hermite 节点及速度、事件与诊断身份 |

位姿 JSON 和 NPZ 的全部七组数组逐元素相同。所有比较使用标定世界坐标、米和秒，旋转误差为完整 SO(3) 旋转角。没有用物理执行轨迹提供初值、目标或参数。

失败曲线分类为 `rejected_diagnostic_not_executable`，`execution_binding=None`，`physical_execution=not_started`。`execution_limits` 中的 0.01 rad 等值是执行跟踪误差要求，**不是机器人关节位置上下界**，没有把它们拿来约束规划。

入口七个臂关节，按 `robot0_joint1` 至 `robot0_joint7`：

```text
q0 = [-0.021132560062632418, -0.16105143326781862,
       0.020916729056392257, -2.444592209444811,
       0.004115624511780582,  2.2267522245222815,
       0.7823010517437409]

dq0 = [-0.015280709418220215, -0.00003584760855043446,
        0.015138836379431635,  0.000007825972533672833,
        0.0030319586784685283, -0.0000016434880011452871,
       -0.0023209330870032126]
```

手指状态也通过同名反馈核对。总曲线时长为 9.3000000715 s；闭爪激活时刻为 7.742 s；终端事件为 8.8350000679 s。三者不能混为同一个“终点”。

### 2.2 独立团队和交叉审核

三名代理分别负责运动学／帧布局、完整数值层、恢复／预算／诊断。运动学代理又审查了数值测试的适用范围和向量残差推导；数值代理审查了恢复补丁；恢复代理核对关节顺序、上下界来源和原预算控制流。根代理复算附件、核对当前源码、复核一手资料并融合结论。没有把不同代理重复的判断当成独立的执行成功。

## 3. 失败指标复现，以及一个必须分清的目标语义

用附件的 `candidate_fk` 直接复算，结果如下：

| 比较 | 位置误差 | 完整旋转角误差 |
|---|---:|---:|
| 闭爪 7.742 s：候选对共同抓取目标 | 77.084591 mm | 16.644734° |
| 闭爪 7.742 s：候选对同刻普通参考 | 82.513284 mm | 13.946325° |
| 终端 8.8350000679 s：候选对目标 | 24.317317 mm | 12.572063° |

共同抓取目标用保存参考在终端进度处的位姿复算，精确复现用户报告到给定精度。原始 grounded PlanningProblem 未包含在两个附件中，因此这里没有宣称读到了它的独立目标矩阵。

源码 `planning_problem.py:2309` 与 `:2350` 分别为终端与 close 事件使用同一个 `grounded["goal_pose"]`。普通参考则来自整段 `grounded["eef_poses"]`。因此闭爪时刻存在“对同刻参考误差”和“对正式交互目标误差”两种不同指标，**不是附件矛盾，也不能用普通参考误差替代任务资格判定**。

闭爪位置差的世界轴分量为：

```text
[-75.523482, -12.499416, -9.055522] mm
```

全程同刻普通参考的时间加权位置 RMS 为 38.843278 mm，旋转角 RMS 为 0.137651171 rad；参考路径长 0.714166 m，候选路径长 1.139735 m。这些只是轨迹描述。普通 waypoint 不是任务必须经过的点，不能把这些统计转换成新硬约束或用来证明方案不可行。

保存曲线在 9.3 s 的位置偏差达到 91.482045 mm、姿态偏差 22.032681°。这是已经拒绝的曲线在尾部的表现，不是发生过的真实 Holding 失效。若任务确实要求某一段保持相对位姿，应由原过程声明表达；本审计不擅自把整个 close→terminal 区间新增为保持约束。

对 8.210 s 的掌部相交，本次保留用户的原观测／原验收结论。附件仅含几何位姿，没有冻结观测场和实体模型的完整验收输入；**没有自行重新确认那一次实体相交**。这一缺口不能靠摆出一个 TCP 点或一个没有场景的 IK 解填补。

复算入口：`audit13_metrics/recompute.py`；数据：`trajectory_metrics.json`；图：`trajectory_errors.png`。

## 4. 确定的代码错误：提案集合超出了投影入口定义域

### 4.1 原代码的实际控制流

| 位置 | 实际职责 | 本次确认 |
|---|---|---|
| `planning.py:3186–3189` | 从功能 IK 池选择提案 | 只要求 `task_satisfied` 和有限 q，没有检查 joint box |
| `planning.py:3238–3254` | 消耗边界尝试并调用投影 | 越界提案同样占用有限尝试 |
| `planning.py:1917–1918` | 投影入口 | q 越界直接返回，尚未进入投影迭代 |
| `planning.py:2811–2815` | 构造物理位置域 | 按同一具名关节轴从官方 native limits 取值 |

设任务允许集合为 A，关节位置域为

\[
D=\{q:q_{\min}\le q\le q_{\max}\}.
\]

当前候选选择允许有限的 `q∈A`，消费者只在 D 上定义。对任意 `q∈A\D`，源码控制流确定返回失败。增加迭代、延长期限或改 Jacobian 都无法让尚未开始的投影成功。这就是可独立证明的接口矛盾。

修复只将优先功能 IK 提案限制到 `A∩D`。上下界直接使用现有 `position_domain`，关节名顺序与 `self.ik.get_active_js(...)` 一致；不新增容差、不裁剪关节、不改变正式任务集合。

```diff
 qualified = ([] if entry is None else [(index, row)
     for index, row in enumerate(entry["candidates"])
     if row["report"]["task_satisfied"]
-    and np.isfinite(row["joint_position"]).all()])
+    and np.isfinite(row["joint_position"]).all()
+    and np.all(np.asarray(row["joint_position"]) >= lower)
+    and np.all(np.asarray(row["joint_position"]) <= upper)])
```

这是修正已有筛选谓词，不是增加一套验证器。工作量是已有少量候选的关节向量比较，没有新 FK、几何查询或求解器调用。

### 4.2 为什么不按其他标志一刀切

`model_valid` 还包含预计算时的场景、附着和释放条件。恢复树的当前前缀可能建立新的场景关系，该分支已有的几何检查才拥有当前判断。因此不能把整个 `model_valid=False` 当成前缀无关的拒绝理由。

原生 `success` 也不等于本项目的任务允许集合。cuRobo V2 明确区分 constraint feasibility 与名义位姿 convergence；在允许自由方向、容差带等任务里，不能拿“没有精确收敛到名义中心”覆盖真正的任务谓词。[S2]

两行补丁只筛投影入口必拒的关节越界状态。全池越界时仍走原 `candidate_curve`；普通参考初始化和首个完整 SLSQP 完全不变。它没有把静态成功作为完整规划的条件。

### 4.3 反事实与集成验证

使用实际 AST 提取的生产提案函数、原 `_project_allowed_boundary` 和原生 OMPL 2.0.1，五个反事实通过：

1. 首项任务合格但 q=1.01 超出 [-1,1]，次项 q=0.2 合格：原版选越界项并被入口拒绝；补丁选次项。
2. 功能 IK 项全越界，原曲线 q=0.25：补丁保留原曲线回落。
3. 界内且任务合格、预先 `model_valid=False`：仍保留为分支提案。
4. 界内且任务合格、原生 success=False：仍保留。
5. 无任务合格项：原回落行为不变。

“入口拒绝前零次残差”只指投影内部；调用者此前可能已经查询 process_count，不能声称整个恢复分支没有计算成本。

随后临时应用补丁，在完整 patched 模块上运行原四文件：

```text
tests/test_planning_ik.py
tests/test_planning_request_budget.py
tests/test_motion_backend_contract.py
tests/test_recorded_task_planning.py

102 passed, 5 subtests passed
```

没有修改断言。测试结束在 finally 反向应用补丁；源码 SHA-256 与原始一致，tracked diff 为空。独立 patch 保留供审阅。这些回归证明所测预算、记录和接口未退化，**不证明原 CUDA IK 或完整抽屉避碰求解成功**。

原始测试曾被环境阻断：两项缺本地 Coal 扩展，两项 `psutil.Process()` 无法识别容器 PID。后续用原 C++ 源码和原 IEEE rounding 编译标志建立 Coal 扩展，并仅在离线测试进程中禁用 `unified_solver.require_host_memory`。原始失败与适配后的结果均保留；不声称真实主机内存准入通过。

## 5. 实际机器人运动学与帧布局：目前不支持重写

最初仅在官方示例模型验证，随后从指定提交的 `scene/drawer_original/scene.xml` 取出**固定机器人 `robot0_base` 子树**。调用仓库原 `_robot_model(..., collision_names=[])`，在编译前排除全部非机器人 body、geom 和 mesh。编译模型 `ngeom=0`、`nmesh=0`；只做 FK，没有物理积分、接触或环境查询。

源 XML blob 为 `7ed99c985810d51540d9548d6b75f9ebc3e03df8`。基座、固定连杆、关节轴和 TCP 均来自该树，没有根据结果拟合标定。MuJoCo 编译后的固定／标量关节变换按现有 `robot_fixture.py` 的规则交给 Pinocchio 官方模型构造器。

| 642 个共同时间点 | 最大位置差 | 最大旋转角差 |
|---|---:|---:|
| 原 robot-only MuJoCo FK 对附件 FK | 1.3172e-15 m | 1.2395e-15 rad |
| 构造的 Pin 模型对附件 FK | 1.0679e-15 m | 8.5967e-16 rad |

因此可以用这棵已核对的运动学树检查本例数值接口。它仍不是原运行 URDF 字节或完整 native 规划配置的恢复；速度、加速度、jerk、力矩界限没有从 XML 猜造。

在本轮 0、6.510、7.742、8.835 s 的实际构型，比较仅工具帧、完整机器人帧、逆序帧和工具首位：

- 前向位姿、速度、原始 Pin Jacobian 和局部速度导数逐元素一致。
- 生产 Torch 位姿／速度反传对中心差分最大差 2.518e-10。
- 包含零项的全帧反传与工具帧反传最多差 2.22e-16，符合求和分组产生的末位差异。[S4]

更直接地，在本次失败曲线的 **147 个控制变量**上，取实际闭爪与终端的四个有符号任务行：

```text
close position    +0.07508459147538088 m
close rotation    +0.2730521144237802 rad
terminal position +0.02231731681772428 m
terminal rotation +0.20197060931337815 rad
```

这四项按原“误差减容差”的形式重建，没有增加任务松弛；目标沿用第 3 节的终端参考重建，未读取原 raw grounded 矩阵。单工具、37 个非 universe 帧、反序 37 帧的任务值与 4×147 Jacobian 逐元素相同。12 方向中心差分最大差 5.065e-10。该四行矩阵的奇异值为 1.17317、1.02058、0.69548、0.23147。

所以在这个实际保存点，没有发现“帧多了导致错误 FK／导数”“任务梯度消失”或四行任务 Jacobian 自身秩亏。**这不等于原失败 QP 的活动约束系统、BFGS 近似矩阵也良好**；后者包含尚未复现的信息。

用户提供的“工具帧静态求解 1.76 mm／0.69°，完整帧变成 44.39 mm／14.18°”仍是必须保留的失败证据，但两个最终优化结果不同，不能倒推出同输入 FK 错误。要定位那次分岔，需要原迭代的同点残差、Jacobian、目标和求解设置。

报告保留了一个被主动弃用的实验路线：直接使用 Pin 的 MJCF parser 没有通过这份 XML 的绑定核对，因此没有使用它进行 IK 结论。实际采用上述已逐点匹配的模型。这不是当前生产 URDF 路线的缺陷证据。

## 6. 三次曲线和完整优化：确认的事实与未定位的困难

### 6.1 普通参考的角色没有改变

`reference_seeds` 每个时刻以前一个有限状态继续 IK，再做一次关节样本拟合。失败但有限的 IK 结果是数值近似，不是任务已满足的证据。这种允许不精确普通 waypoint 的设计本身不违背任务需求。用户原记录显示该继续构型接近限位；本审计确认了单一继续的源码路径，尚未独立分离初始化对完整失败的贡献，不能由此认定普通 waypoint 必须全部求解成功。

同样，不能为此把整段工具参考变成硬跟踪条件，或重新叠加曲率／挠率限制。本任务真正必须满足的是原事件允许集合、真实过程约束、机器人物理界限和场景验收。

### 6.2 保存曲线的连续域导数通过

从实际 Hermite 数据恢复当前三次控制矩阵，最大重建误差为 7.3275e-15 rad。21×7 控制量对应 147 变量、1400 个连续物理不等式行；重建采样矩阵秩为 21，条件数为 20.5135。这不表示 21 次原始 IK 都成功。

12 方向、三组中心差分步长的最大误差为 5.024e-8、1.910e-8、2.471e-7。该实验用任意常数构造 bounds 偏置，只测试导数；没有把这些数当成真实硬件界限。

对 `q(t,x)=A(t)x+b(t)` 的唯一内部极值 `t*(x)`，

\[
\frac{d}{dx}q(t^*(x),x)
= A(t^*)+\frac{\partial q}{\partial t}\frac{dt^*}{dx}
=A(t^*).
\]

极值处时间导数为零，当前活动极值 Jacobian 有数学依据。多个最大值切换会产生非光滑性，但本次保存点没有发现精确 tie，不能仅凭状态 3 把它指定为根因。

确实发现 266 个结构重复的加速度约束：C2 曲线的 19 个内部节点、7 个关节、上下两侧，由相邻区间各提交一次。删除重复副本可以证明不改变数学可行域；**没有实测它能修好本次 mode 3，也没有性能收益证据，本次不推荐为此扩大修改范围**。位置／速度当前零梯度行则可能随极值分支改变，不能按某次零梯度删除。

### 6.3 SLSQP 状态 3 的正确解释

SciPy 文档将状态 3 定义为内部 LSQ 子问题迭代过多，状态 9 才是外层通常的迭代上限。[S1] 因此增加 `trajectory_iterations` 不是已证明的修复；状态 3 也不是“目标全局不可达”的证书。

当前代码已按首次目标值和梯度取固定正数尺度，整个主问题显式使用 float64；不能再次把“添加目标缩放”当作新修复。约束函数与 Jacobian 同时取负后交给 SciPy，符号与接口一致。

主 SLSQP 的硬行是任务条件和连续关节物理界限。几何通过目标函数引导，原实体／连续区间查询决定最终接受。**不要写成全部碰撞已经作为硬不等式交给 SLSQP。** 罚函数优化加独立验收是一种可用的工程结构，但有限局部求解失败时不会给出可行保证。

当前 `pair_residual` 的分类值为 1 时导数为零，优化使用另一个固定材料点标量及其一阶导数。历史“值恒为 1 而传非零梯度”的问题不能沿用为 f74a3be 的已知根因。没有冻结场景，仍不能独立认证本次几何导数正确。

## 7. 真实机器人上的负结果：有界求解器和等代价残差都没有提供修复证据

为避免再次推荐未经检验的数值替换，本次实际测试 SciPy 的有界 Trust Region Reflective 法。它是维护中的通用库方法，其迭代遵守变量边界；这与无界 Newton 后再裁剪不同。[S5]

试验保持本轮目标、机器人源模型的位置范围、原 2 mm／1°和完整机器人帧。固定三个入口：反馈 q0、失败曲线的闭爪 q、终端 q。每次 `max_nfev=40`，默认 ftol、xtol、gtol、x_scale 不改，没有随机搜索。函数评估次数不等同于 native LBFGS 迭代的时间预算；这里只是明确固定的离线对照。

| 入口 | 原两个标量允许集合残差 | 同代价六维向量残差 | 原任务集合 |
|---|---:|---:|---|
| q0 | 61.463 mm／0.981018 rad | 4.670 mm／0.091386 rad | 均失败 |
| 保存的闭爪 q | 6.262 mm／0.286455 rad | 4.860 mm／0.219104 rad | 均失败 |
| 保存的终端 q | 5.441 mm／0.240093 rad | 5.202 mm／0.241361 rad | 均失败 |

六次都保持在原模型位置范围内，但都耗尽 40 次评估，并且都不满足原任务。单次 0.37–0.52 s 是审计环境静态计时，不能外推完整规划速度。没有追加参数或预算直到得到一条好结果。

向量表示的数学也独立核对了。对三维位置误差或完整旋转向量 e，令 `n=||e||`、原阈值 τ>0：

\[
r_s=\frac{(n-\tau)_+}{\tau},\qquad
r_v=\frac{(n-\tau)_+}{\tau\max(n,\tau)}e.
\]

有 `||r_v||²=r_s²`，目标值和精确一阶梯度不变，任务集合不变。在 n>τ 处，令 u=e/n，有

\[
J_{r_v}^{T}J_{r_v}-J_{r_s}^{T}J_{r_s}
=J_e^T[(1/\tau-1/n)^2(I-uu^T)]J_e\succeq0.
\]

所以六维形式保留了 Gauss–Newton 的切向曲率。实测六个确定点的代价差不超过 3.638e-12、梯度差不超过 7.276e-12，残差 Jacobian 秩从 2 变成 6。但这只改变 GN 近似，不改善原 LBFGS 的精确一阶梯度，**更不证明原 SLSQP 的失败原因**。实际仍失败，所以不把这一数学上合理的变换当成可交付修复。

这也没有反驳用户“12 个初值中 7 个静态解成功”的证据：这里测试的是固定三个入口、固定评估额度及特定残差，而非全局可达性证明。

## 8. 成熟实现提供的依据，以及没有提供的保证

| 一手实现／资料 | 可以借鉴的工程事实 | 不应推导出的结论 |
|---|---|---|
| cuRobo 论文与真实机器人实验 [S6] | IK、多个优化初值、几何路径与轨迹优化协同；研究包含真实机器人跟踪与部署 | 本项目一个单初值 SLSQP 应自动达到其速度／成功率，或照搬流水线即可满足当前全部事件语义 |
| MoveIt CHOMP 官方文档 [S7] | 局部轨迹优化会受坏初值、局部极小影响；可以消费 OMPL 提供的初始路径 | 用某个静态 IK 成功替代完整路径证明，或给所有任务增加静态准入 |
| OMPL constrained planning [S3] | 约束表达、解析 Jacobian、投影／流形插值有明确适用前提 | 一次投影失败等于目标无解，或任意越界输入都能被投影器修好 |
| Pinocchio 官方帧 Jacobian [S8] | 明确作用点和坐标轴约定，复用解析导数 | 因优化终点不同就改 FK 公式 |
| SciPy SLSQP／TRF [S1][S5] | 明确退出状态、约束符号、变量边界与停止准则 | 换库名、调 maxiter、看到 cost 下降就已经得到可执行解 |

本次没有找到支持“删库重写”的严格证据。保持普通参考初始化→完整受约束优化→原验收，以及已有的几何恢复路径，是比再次改掉准入／预算职责更有依据的选择。改动数值层必须针对原失败问题做单变量对照；现有局部负结果不能包装成工程成功。

## 9. 推荐的修改边界与下一步必要输入

**唯一建议现在合入的代码变化，是第 4 节的关节域一致性补丁。** 文件为 `audit13_recovery/proposal_domain.patch`。它可以消除一个确定无效的优先提案路径，不能保证本例恢复成功，也不能改善它之前已经失败的主 SLSQP。

以下职责保持现有 f74a3be 行为：普通参考仍可包含有限近似；基础初始化仍在原请求预算内；附加搜索失败不阻断原完整求解；已完成候选仍交给记录器；最终容差、接触许可与原几何验收不变。补丁没有新后端、兼容层、在线验证器或 skill-specific 分支。

定位主优化的下一步，需要原失败非线性问题的可复算证据。优先提供已有的小记录即可，不需要先上传全部原始 RGB-D：

1. **本次 `planning-report.json` 和 `request-ledger.json`。** 源码已经归档 `reference_ik_queries`、`reference_initialization`、`functional_ik_candidates`、`last_optimizer_runtime`、`verification_joint_limits` 等字段。它们能确定原近似 IK 节点、实际四阶界限、初始拟合、试算身份和预算，不必新增在线监控。
2. **本次请求的任务声明／grounded problem，以及实际加载的 cuRobo 版本和 STATE 配置。** 例如现有请求 JSON 和实际 `trajopt/lbfgs_bspline_trajopt.yml` 中的 `cspace_cfg`。不能用当前网上 main 版本冒充原运行权重。

拿到这些后，可以先在已核对机器人上重放原任务与物理约束，并隔离几何目标的影响。若要宣布完整场景修复，最终仍必须由原冻结场景执行原实体与连续区间验收；仅有上述小记录或新的单点 IK，不能作这个承诺。

不建议当前扩大成下列修改：事件 IK 成功前置、精确事件关节拟合前置、重新分配主优化／搜索预算、按结果删掉困难候选、放宽 2 mm／1°、全局夹紧关节曲线、曲率／挠率限制、或为此 grasp 特设权重。这些都缺少本轮完整成功对照。

## 10. 测试与复现目录

| 目录 | 证据 |
|---|---|
| `audit13_metrics/` | 原附件指标复算、误差序列与图 |
| `audit13_robot/` | 真实机器人源树绑定、642 点对照、实际构型帧／导数测试 |
| `audit13_kinematics/` | 初步官方示例、生产完整任务接口布局核对、数学推导 |
| `audit13_solver/` | 实际曲线连续物理域、实际机器人四行任务 Jacobian、静态负结果、既有数值测试 |
| `audit13_recovery/` | 两行 patch、五个反事实、基线与 patched 完整模块回归、环境适配范围 |

每个代理附有 `READ_SCOPE.md`；根代理另附 `AUDIT_READ_SCOPE.md`。目录内脚本直接调用固定提交的生产函数。它们是离线审计工具，不接入运行时。与外部仓库同步的源码没有被复制成另一套生产后端。

审计环境：Python 3.12、NumPy 2.3.5、SciPy 1.17.0、Torch 2.14.1+cpu、Pinocchio 4.1.0、Coal 3.0.3、OMPL 2.0.1、MuJoCo 3.17.0。没有 CUDA，本次没有运行原 cuRobo GPU IK。依赖装在隔离目录，不修改用户远端环境。

独立数值测试中四个原测试通过，但其中一个是接口 mock，只有一个实际执行合成三自由度、无几何场景的 SLSQP。它们没有被计算成四次完整抓取成功。真实机器人上的静态六次试验是失败结果，完整保留。

复现前在同一目录放置固定版本仓库 `FUNCTO_public`，安装对应依赖；脚本中的 `site.addsitedir` 仅指向本轮隔离环境，在依赖已安装的常规 Python 环境中可直接运行。代表命令：

```bash
git -C FUNCTO_public show f74a3be:scene/drawer_original/scene.xml > audit13_robot/source_scene.xml
git -C FUNCTO_public show f74a3be:scene/drawer_original/robot_feedback.json > audit13_robot/robot_feedback.json
python audit13_metrics/recompute.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit13_robot/probe_robot_binding.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit13_robot/probe_actual_robot_frames.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit13_solver/probe_cubic_derivatives.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python audit13_solver/probe_actual_robot_task.py --feedback audit13_robot/robot_feedback.json --model-evidence audit13_robot/robot_binding_results.json
python audit13_recovery/probe_recovery_contract.py
```

两份 Git 跟踪的源文件不在交付包中重复保存；上面两条命令从已固定的仓库提取。所有物体和环境节点在机器人模型编译之前排除。`robot_only.xml` 是本次生成的纯机器人审计产物。

代理的具体测试脚本和结果给出其各自执行命令。`run_patched_regression.py` 会临时应用补丁并在 finally 恢复；如要复现，使用干净的独立 checkout。交付包不含依赖二进制、原私有碰撞场或可执行机器人轨迹。

## 一手资料

- [S1] SciPy，SLSQP 退出码与接口：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_slsqp.html>；<https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html>。
- [S2] NVIDIA cuRobo V2，`IKSolverCfg.success_requires_convergence`：<https://github.com/NVlabs/curobo/blob/main/curobo/_src/solver/solver_ik_cfg.py>。用于库语义参考，不作为用户已安装版本的替身。
- [S3] OMPL，Constrained Planning：<https://ompl.kavrakilab.org/constrainedPlanning.html>。
- [S4] PyTorch，Numerical accuracy：<https://docs.pytorch.org/docs/2.14/notes/numerical_accuracy.html>。
- [S5] SciPy，least_squares/TRF：<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html>。
- [S6] Sundaralingam 等，cuRobo 论文，尤其 §6.3、§6.4 与多初值分析：<https://arxiv.org/html/2310.17274v2>。这是 v1 研究中的真机与架构证据，不是当前自定义 V2 集成的性能保证。
- [S7] MoveIt，Using CHOMP Planner：<https://moveit.picknik.ai/main/doc/how_to_guides/chomp_planner/chomp_planner_tutorial.html>。
- [S8] Pinocchio，帧 Jacobian 定义：<https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/devel/doxygen-html/frames_8hpp_source.html>。
- 固定项目源码：<https://github.com/illusionaryshelter/FUNCTO_public/tree/f74a3bea18715618138a6e45f5945c2e11eb6f5f>。

这里的第三方资料支持算法职责和实现选择；本轮成功／失败的判断来自上述实际实验。完整抓取成功没有在本环境得到证明。
