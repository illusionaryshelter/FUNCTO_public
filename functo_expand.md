| 想处理的运动                 | 合适的配置表示                                     | 独立自由度 |
| ---------------------------- | -------------------------------------------------- | ---------- |
| 三维空间中自由运动的刚体     | \(SE(3)\)                                          | 6          |
| 在固定平面内平移和转动的刚体 | \(SE(2)\)，可用 \((x,y,\psi)\) 表示                | 3          |
| 在固定平面内移动，但方向固定 | \(\mathbb R^2\)，可用 \((x,y)\) 表示               | 2          |
| 绕固定轴转动的单关节         | 理想情况下为 \(SO(2)\)；有限位时通常用角度区间表示 | 1          |
| 沿固定轴滑动的单关节         | 位移区间，属于 \(\mathbb R\)                       | 1          |

**FUNCTO 原论文没有引入刚体动力学或接触动力学约束。你这次的五类扩展也不要求引入 Drake。**

我建议把实现确定为：

> **保留 FUNCTO 的功能对应、功能关键帧和演示轨迹迁移；把轨迹生成改成“在当前实例的运动学配置空间内进行功能约束优化”。**

工具栈上，采用 **Pinocchio + Coal + Aligator 作为核心**；Crocoddyl 可以保留为已有模型的适配入口，但不要同时维护两套问题建模和两轮优化。

下面按源码、组件边界、数据结构、五类运动建模、grounding、求解实现和开发计划展开。这里是基于论文与公开源码审阅的方案，**不是已经在你的环境编译验证过的版本组合**。

------

## 一、先明确 FUNCTO 原版究竟做了什么

### 1. 原论文是运动学轨迹优化，不是动力学优化

FUNCTO 优化工具的位姿轨迹，跟踪演示中的功能点位置与姿态，并约束初始位姿和功能关键帧。附录进一步说明使用 **CasADi + IPOPT**，加入平移/角速度限制和工具—障碍物包围盒之间的距离约束。没有引入质量、惯量、驱动力矩、摩擦力或接触动力学方程。[arXiv](https://arxiv.org/html/2502.11744v1)

因此，三个概念不能混淆：

| 约束                                     | 性质           | 这次扩展是否需要       |
| ---------------------------------------- | -------------- | ---------------------- |
| 门板只能绕铰链转动、物体只能在平面内移动 | 运动学配置约束 | **需要**               |
| 轨迹速度、角速度、平滑性、几何避碰       | 运动学轨迹约束 | **需要**               |
| 力矩、摩擦、接触力、惯性和重力平衡       | 物理动力学约束 | **不属于当前组件职责** |

尤其不要把“低速执行”叫作已经建模了“准静态动力学”：没有力平衡模型，就仍然只是运动学规划。

### 2. 公开代码不能不加区分地当成论文公式

官方项目页当前链接到 [mkt1412/FUNCTO_public](https://github.com/mkt1412/FUNCTO_public?utm_source=chatgpt.com)。这个仓库目前的 README 使用 **MimicFunc** 名称；复现时应该记录代码快照，而不是默认当前代码与 2025 年 FUNCTO v1 完全一致。[Google Sites](https://sites.google.com/view/functo)

最值得注意的是 [`tool_traj_transfer.py`](https://github.com/mkt1412/FUNCTO_public/blob/main/tool_traj_transfer.py)：

| 项目         | 我核对到的公开实现                                          | 对扩展的影响                                   |
| ------------ | ----------------------------------------------------------- | ---------------------------------------------- |
| 姿态优化     | 使用欧拉角及 wrapped angle difference                       | 应改为流形上的姿态差，而不是继续扩大欧拉角实现 |
| “速度”限制   | 约束相邻帧差，没有显式时间步长                              | 必须恢复时间单位，不能让结果依赖采样帧数       |
| 碰撞         | 该文件的 `optimize_trajectory()` 中没有论文附录所述避碰约束 | 新组件需要显式实现，不能认为继承代码就已具备   |
| 参考轨迹对齐 | 包含绕目标 z 轴旋转的处理                                   | 不能直接用于任意铰链轴、导轨或平面             |

这些是该公开函数的实现观察，不等于声称作者所有实验代码都没有相关能力。[raw.githubusercontent.com](https://raw.githubusercontent.com/mkt1412/FUNCTO_public/main/tool_traj_transfer.py)

**因此，忠于 FUNCTO 应该忠于它的方法结构，而不是保留上述实现细节。**

------

## 二、工具栈怎么分工：不需要 Drake，也不要强行把四个库串起来

我的推荐分工如下。

| 库            | 在这个组件中的职责                                  | 不承担的职责                         |
| ------------- | --------------------------------------------------- | ------------------------------------ |
| **Pinocchio** | 五类运动学模型、流形积分、位姿、点和坐标系 Jacobian | 不必使用 ABA、RNEA 或质量模型        |
| **Coal**      | 几何距离、碰撞查询、最近点和法向                    | 不负责摩擦、接触力和物理仿真         |
| **Aligator**  | 带等式/不等式约束的轨迹优化                         | 不要求问题必须是力矩控制             |
| **Crocoddyl** | 已有 action/residual 模型的可选适配来源             | 不作为 Aligator 前后额外的一轮求解器 |

Pinocchio 的配置流形操作可以直接被 Aligator 的 `MultibodyConfiguration` 和 `MultibodyPhaseSpace` 使用；Aligator 的显式状态转移接口接受一般的 \(x_{k+1}=f(x_k,u_k)\)，并不强制你使用物理动力学。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/include/aligator/modelling/spaces/multibody.hpp)

Aligator 也有官方 Crocoddyl 兼容层，包括 `convertCrocoddylProblem`。不过，**没有现成 Crocoddyl 模型要复用时，直接用 Aligator 原生接口更清楚**；已有模型则应针对锁定版本验证约束是否完整转换，而不是默认转换后完全等价。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/include/aligator/compat/crocoddyl/problem-wrap.hpp)

Drake 本身也有运动学轨迹优化工具，并非只能做动力学；但它没有为这五种情况提供一个你现有栈缺失的必要能力。引入它反而会多出一套模型、几何和求解接口。[Drake](https://drake.mit.edu/doxygen_cxx/classdrake_1_1planning_1_1trajectory__optimization_1_1_kinematic_trajectory_optimization.html)

**另外，这五类扩展并不迫使你更换 CasADi/IPOPT。**选择 Aligator 是为了与你现有栈一致、复用流形和约束建模，不是因为 FUNCTO 原来的优化器不能表达这些约束。

------

## 三、组件边界：只有“编译 skill”和“实例化 skill”两项核心操作

我建议整个组件围绕下面的接口组织：

```
# 建议的组件接口，不是现有 FUNCTO API。compile_skill(    demo: PreprocessedDemo,    spec: SkillSpec,) -> SkillTemplateground_skill(    template: SkillTemplate,    instance: InstanceBinding,    limits: MotionLimits,) -> GroundingResult
```

对应的数据流：

```
预处理演示 + 已声明的运动模式/任务关系
                    │
                    ▼
              compile_skill
                    │
                    ▼
              SkillTemplate
                    │
       新实例功能对应、机构参数、
       当前状态、目标关系、几何
                    │
                    ▼
               ground_skill
                    │
                    ▼
       参数化物体轨迹 + 事件 + 验证报告
```

**这不是完整机器人系统。**组件不负责视频分割服务、在线目标检测、抓取搜索、机器人控制、接触模式搜索，也不负责长程任务规划。

但这些能力不在组件内部，**不等于它们的输入假设可以不写清楚**。

### 1. 三种数据对象必须分开

#### `PreprocessedDemo`：演示中实际观测到了什么

至少包括：

- 时间戳、功能事件索引、目标/参考坐标系；
- 被操纵刚性部件的位姿轨迹，或足以恢复它的三维跟踪数据；
- function、grasp、center 等语义点及其置信信息；
- 已知或已估计的演示机构参数，以及估计来源和残差。

原仓库已有可接入的离线文件，例如 `tool_pose_track_output/tool_track.json` 和 `key_point_track_output/keyframe_idx.json`。应先做 adapter，而不是重新实现原仓库的整个 `main.py`。[GitHub](https://raw.githubusercontent.com/mkt1412/FUNCTO_public/main/tool_traj_transfer.py)

#### `SkillTemplate`：跨实例应该保留什么

建议保存：

| 内容               | 示例                                                         |
| ------------------ | ------------------------------------------------------------ |
| 功能角色与参考证据 | 哪个点是 function、哪个区域用于 grasp；对应参考图像/描述符资产 |
| 功能坐标定义       | 功能轴如何构造、哪些姿态分量有意义                           |
| 演示运动形状       | 目标相对轨迹、约束空间中的运动过程、阶段性速度形状           |
| 功能事件           | 功能开始、到达关键状态、任务结束                             |
| 目标关系           | 功能点到目标点、轴对齐、完整位姿、到达某个关节状态           |
| 可迁移规则         | 允许的对称变换、是否允许重定时                               |
| 前提与容差         | 稳定抓持、刚性部件、距离/角度容差、数据有效范围              |

**不要把演示柜门的世界坐标铰链轴，作为模板中永远不变的参数。**它可以作为演示证据保存，但新实例必须重新绑定自己的轴。

同样，不能把演示末帧的某个数值角度直接解释成所有柜子的“关闭角度”。

#### `InstanceBinding`：当前实例具体是什么

包括：

- 新实例上的功能点和功能坐标关系；
- 当前机构的轴、平面、零位变换和限位；
- 当前配置 \(q_{\text{start}}\)；
- 当前目标位置或目标关节状态；
- 固定环境与运动部件的碰撞几何；
- 可选的实际抓握变换。

**一个 grasp point 不等于一个六自由度 grasp pose。**没有实际抓握变换时，组件只能输出物体/功能坐标系轨迹，不能凭空输出确定的末端轨迹。

### 2. 单段演示无法自动提供的知识，要作为显式输入

这里最重要的两个例子：

**运动模式。**一条直线轨迹不能证明物体只能沿直线运动。`mode` 应来自已声明的机构信息或有证据的估计，而不是从轨迹维数直接下结论。

**目标语义。**看见门停在某处，不一定能判断它是“完全关闭”还是“关到一半”。`JointBoundary("closed")` 必须在新实例中绑定到明确的状态/几何关系。

这不是为了省事而额外假设，而是组件必须诚实表达的信息边界。新增这些输入时，也应在实验中明确：**受约束扩展获得了哪些原版没有使用的机构信息。**

### 3. 序列化保持简单

建议使用：

```
skill/
  manifest.json       # 类型、角色、坐标约定、事件、目标关系、版本
  reference.npz       # 时间、轨迹、关键点、拟合参数
  assets/             # 必要的参考图像/描述符，或外部资产引用
```

不要序列化 solver、Pinocchio `Data` 或 Python pickle 对象。求解器和当前实例的几何缓存应在 grounding 时建立。

------

## 四、五类运动统一使用一个“小型物体运动学模型”

### 1. 推荐直接复用 Pinocchio，而不是自己再造一套流形实现

这里可以创建一个很小的 **虚拟物体运动学模型**：

- 没有整台机器人；
- 只包含固定参考、允许的虚拟关节和运动部件；
- 仅使用运动学计算；
- 不为了使用优化器而编造质量、惯量或驱动力。

五种模式分别这样实现：

| 模式             | Pinocchio 建模方式                 | 配置存储维度 / 切空间维度 |
| ---------------- | ---------------------------------- | ------------------------- |
| 三维自由刚体     | Free-flyer                         | 7 / 6                     |
| 平面平移与转动   | Planar joint                       | 4 / 3                     |
| 平面固定姿态平移 | 两个正交移动关节，最终部件姿态固定 | 2 / 2                     |
| 固定轴转动       | 有限位的标量 revolute joint        | 1 / 1                     |
| 固定轴滑动       | Prismatic joint                    | 1 / 1                     |

Free-flyer 使用位置加四元数；Planar 使用二维位置和旋转的正余弦表示。因此，**配置数组长度和优化增量维数不同**，不能把状态 Jacobian 写成统一的“对原始数组求导”。[GitHub](https://raw.githubusercontent.com/stack-of-tasks/pinocchio/v3.4.0/include/pinocchio/multibody/joint/joint-free-flyer.hpp)

有限位的柜门建议使用明确的标量角度和角度范围，而不是把角度无条件 wrap 到 \([-\pi,\pi]\)。否则会丢失限位、绕行分支或演示中的转动方向信息。

### 2. 坐标系关系必须一次定义清楚

约定 \({}^{X}T_Y\) 把 \(Y\) 坐标转换到 \(X\)。

使用：

- \(W\)：世界坐标系；
- \(B\)：固定底座/参考部件；
- \(L\)：被操纵的刚性部件；
- \(A\)：任务目标坐标系；
- \(F\)：功能坐标系。

统一表达式：

\[ {}^{A}T_F(q) = ({}^{W}T_A)^{-1} {}^{W}T_B \,F_{\text{mode}}(q;\eta)\, {}^{L}T_F . \]

这里，\(F_{\text{mode}}\) 输出 \({}^{B}T_L\)，\(\eta\) 是当前实例的机构参数，\({}^{L}T_F\) 来自**新实例**的功能对应。

受约束模式再写成：

\[ {}^{B}T_L(q) = {}^{B}T_J\, J_{\text{mode}}(q)\, {}^{J}T_{L,0}. \]

其中：

\[ \begin{aligned} J_{\text{planar}}(q)&=\operatorname{Trans}(x,y,0)\operatorname{Rot}_z(\psi),\\ J_{\text{translation2}}(q)&=\operatorname{Trans}(x,y,0),\\ J_{\text{revolute}}(q)&=\operatorname{Rot}_z(\theta),\\ J_{\text{prismatic}}(q)&=\operatorname{Trans}(0,0,d). \end{aligned} \]

**任意方向的轴和平面通过固定变换表示，而不是要求它们与世界 z 轴一致。**

实际代码中，这套公式应由 Pinocchio joint placement、frame placement 和 FK 实现；不要同时维护一套手写 FK 和一套 Pinocchio FK，形成两个真相来源。

### 3. 运动模式不等于控制模式

Aligator 的 SE(2) 小车示例可以借鉴流形和求解接口，但**不能复制其车辆运动方程**。你的平面刚体允许独立的横向移动，不应被错误地加上小车的非完整约束。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/examples/se2-car.hpp)

另外，平面约束可能来自机构，也可能来自任务要求。建议记录：

```
constraint_origin = mechanism | task
```

这能避免把“本次要求沿平面操作”误写成“这个物体物理上只有平面自由度”。

------

## 五、离线编译：保留 FUNCTO 的核心，补上必要的数据校验

### 1. 从原代码中拆出什么

优先复用三个部分：

**功能对应结果。**保留原方法的功能角色、参考图像和对应点证据。新的核心可以读取已经缓存好的对应结果，不必在求解过程中调用 VLM。

**功能坐标和关键帧构造。**从 [`tool_pose_transfer.py`](https://github.com/mkt1412/FUNCTO_public/blob/main/tool_pose_transfer.py) 拆出几何逻辑，与图像加载、渲染、VLM 调用分离。这个文件目前混合了这些职责，也包含按具体任务名处理的高度等启发式规则，不宜整体继承为通用组件。[GitHub](https://raw.githubusercontent.com/mkt1412/FUNCTO_public/main/tool_pose_transfer.py)

**参考轨迹迁移。**保留“目标相对的功能轨迹 + 功能事件约束”的思想，替换其姿态表示、时间处理和运动可行性部分。

这里要坚持一条界线：

> **新增机构信息是为了限制可行运动，不是为了绕过演示、直接写一个与 FUNCTO 无关的柜门控制器。**

### 2. 必须处理功能坐标退化

原实现从 function、center、grasp 三点构造坐标轴和平面。[GitHub](https://raw.githubusercontent.com/mkt1412/FUNCTO_public/main/tool_traj_transfer.py)

扩展到门板、抽屉或对称物体后，可能出现：

- function 与 grasp 是同一个部位；
- 三点接近共线；
- 两点距离过小；
- 关键点的语义正确，但不足以确定完整朝向。

这些情况不能通过“分母加一个 epsilon”解决。

我建议的规则是：

**有可靠部件位姿或额外表面方向时，用它补足坐标定义；只有一个可观察功能轴时，就保存轴约束，而不是虚构完整功能坐标系。**

如果任务确实要求完整姿态，而数据不能确定它，则编译结果应报 `INSUFFICIENT_ORIENTATION_EVIDENCE`。

### 3. 保持刚性一致性

如果输入已经给出部件位姿轨迹，应该估计一次部件局部关键点，再用刚体变换恢复各帧位置。

不要每一帧独立构造一个受噪声影响的三点坐标系，最后把跟踪抖动当作演示技能。

同理，对已声明的 revolute、prismatic 等模式，可以拟合机构参数并报告残差；但**拟合得好不等于证明该模式唯一正确**。

### 4. 功能事件与任务末帧分开

模板至少区分：

```
start
function_event
end
```

必要时可以有多个明确事件，但这仍是一个给定 skill 内部的事件序列，不需要引入任务图或自动技能分解器。

不能默认 FUNCTO 的功能关键帧就是最终停止状态；否则 pouring、cutting 等“进入功能姿态后继续运动”的技能会被截断。

------

## 六、grounding：不只是“转移点”，还要生成可行的功能目标

我建议固定为下面五步。

### 步骤 A：绑定当前实例

读取当前：

\[ \eta_{\text{new}},\quad q_{\text{start}},\quad {}^{L}T_F,\quad {}^{W}T_A, \]

以及目标关系、几何和运动限制。

先检查关节限位、坐标一致性、功能角色是否齐全。**输入不完整和求解失败应是不同的错误。**

### 步骤 B：在可行运动空间内确定功能状态

这里是扩展最重要的变化：

> **不要先生成一个任意 SE(3) 功能位姿，投影到铰链轨迹上之后，就认为功能目标已经满足。**

投影可以作为初始化，但投影后必须重新检查功能条件。

建议支持一组固定、明确的目标残差：

```
PointRelation       功能点与目标点/区域的关系
AxisAlignment       功能轴与目标方向的关系
FramePose           完整功能位姿
JointValue          到达指定关节状态
JointBoundary       到达当前实例绑定的某个语义边界
```

对于自由刚体，完整位姿约束经常可直接使用。

对于一自由度机构，完整位姿目标只有在与机构一致时才可达。不能为了让优化器收敛，偷偷把所有姿态约束降成弱代价。

例如，“关闭”可以使用 `JointBoundary("closed")`，但 `"closed"` 到底对应上限、下限还是一个中间状态，必须由当前实例绑定。**不能统一写成 `q=0`。**

### 步骤 C：根据当前状态生成参考运动

五种情况共享功能表示，但参考轨迹适配不必完全相同。

#### 自由刚体

保留 FUNCTO 的功能轨迹迁移和关键状态对齐。只有目标明确具有相关对称性时，才允许绕对称轴改变接近方向。

不应把原代码默认的 z 轴旋转推广成所有任务都允许的变换。

#### 两类平面运动

在实例的平面坐标中迁移平移轨迹；允许旋转的模式再迁移朝向过程。

对于固定姿态的二维移动，**姿态从模型中固定，而不是增加一个“大权重的姿态保持代价”**。

#### Revolute / Prismatic

对一个非零净位移的演示区间，可以保存归一化进度：

\[ \rho(s) = \frac{q_{\text{demo}}(s)-q_{\text{demo,start}}} {q_{\text{demo,end}}-q_{\text{demo,start}}}, \]

再生成当前实例的参考：

\[ q_{\text{ref}}(s) = q_{\text{start}} + \rho(s)\bigl(q_{\text{goal}}-q_{\text{start}}\bigr). \]

这样保留的是演示运动过程，而不是原实例的绝对角度或位移。

但不能只实现这个公式：往复动作、净位移为零和中途停顿，需要按已有事件区间表示，或者保存增量运动，不能遇到零分母就把轨迹改成线性插值。

**对一自由度机构，空间运动轨迹主要由机构决定。演示的价值在功能对应、目标关系、运动方向、事件和运动过程，而不一定在复杂的空间路径形状。**

### 步骤 D：执行统一的约束优化

五类模式仅替换配置空间、FK 和约束参数；不维护五个优化器。

### 步骤 E：重建并验证轨迹

输出物体轨迹。提供实际刚性抓握变换后，才进一步输出：

\[ {}^{W}T_E(t) = {}^{W}T_L(t)\,{}^{L}T_E . \]

这仍然不是机器人关节轨迹，也不自动证明机器人可以执行。

------

## 七、Aligator 中如何具体实现

### 1. 使用运动学状态转移，而不是伪造物理动力学

推荐状态：

\[ x_k=(q_k,v_k),\qquad u_k=\dot v_k. \]

这里 \(v_k\) 是配置空间的切向速度，\(u_k\) 是其变化率，不是力矩。

离散转移可用下面的一阶形式说明：

\[ q_{k+1} = \operatorname{integrate}(q_k,h_kv_k), \qquad v_{k+1}=v_k+h_ku_k. \]

实际实现要固定积分约定和误差控制，并用时间步长细化测试检查结果。**不能把四元数配置直接做加法，也不能把 \(u\) 无条件当作世界坐标系下的物理线加速度。**

Aligator 已有适合这一结构的 `MultibodyPhaseSpace`；自定义状态转移只需实现对应的 `forward()` 和 `dForward()`。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/include/aligator/modelling/spaces/multibody.hpp)

选择这种状态扩充，是为了方便表达平滑性和速度边界，**不是宣布我们引入了物理动力学模型**。

### 2. 保留 FUNCTO 风格的目标函数

以功能坐标系为例：

\[ r_{\text{func},k} = \begin{bmatrix} (p_{F,k}-p^{\text{ref}}_{F,k})/\sigma_p\\ \operatorname{Log}\!\left( (R^{\text{ref}}_{F,k})^\top R_{F,k} \right)/\sigma_R \end{bmatrix}. \]

建议总代价：

\[ \sum_k h_k \left[ w_k\|r_{\text{func},k}\|^2 + \|v_k-v_k^{\text{ref}}\|_{W_v}^2 + \|u_k\|_{W_u}^2 \right]. \]

其中：

- \(\sigma_p,\sigma_R\) 负责位置与角度尺度，避免直接把米和弧度相加；
- \(w_k\) 来自明确的功能阶段，不写死“前 40% 不重要”；
- 只有功能轴可观察时，用轴残差替代完整旋转残差；
- 关键功能条件由单独的硬约束或明确容差处理，不都塞进软代价。

### 3. 必须显式加入的约束

建议至少包括：

| 约束              | 实现方式                                           |
| ----------------- | -------------------------------------------------- |
| 当前状态          | 固定初始 \(q\)，初始速度由输入或明确的静止前提确定 |
| 功能事件          | 指定节点上的点、轴、位姿或关节状态约束             |
| 配置限位          | 作用于真实物理坐标，不是随意给四元数分量加上下限   |
| 速度限制          | 配置切向速度，以及关键部位的笛卡尔速度             |
| 平滑性/变化率限制 | 对应明确单位和坐标约定                             |
| 碰撞间距          | 按几何对和功能阶段指定                             |
| 停止条件          | 任务要求停止时，约束末端速度                       |

这里特别要加**关键部位的速度约束**。

相同角速度下，长门板把手的线速度比短门板更大。因此只限制 \(\dot\theta\)，不能表达跨尺寸迁移中的线速度限制。应通过 Jacobian 计算功能点或抓握点速度：

\[ v_{\text{point}}=J_{\text{point}}(q)\,v. \]

若接口提供“笛卡尔加速度上限”，就要计算相应的 \(J\dot v+\dot Jv\)，不能拿 \(\|u\|\) 代替。

### 4. 时间与相位必须分开

模板保存相位 \(s\)，grounding 绑定物理时间。

建议显式提供：

```
timing_policy:
  fixed
  stretch_allowed
```

以及阶段时长和允许范围。

不能因为新门更大、原速度约束不满足，就悄悄放松速度上限。允许重定时的技能可以延长时间；不允许的技能应报告不满足条件。

对于依赖冲击或材料作用的技能，运动学组件只能保留和检查指定运动条件，**不能因此认证功能一定成功**。

### 5. 可以直接借鉴的 Aligator 代码

[`examples/clqr.cpp`](https://github.com/Simple-Robotics/aligator/blob/main/examples/clqr.cpp) 展示了这套直接可用的结构：

```
StageModel
  + addConstraint(..., BoxConstraint / EqualityConstraint)

TrajOptProblem
  + terminal constraints

SolverProxDDP
  -> setup
  -> run
```

无需先创建完整的机器人最优控制问题，再把对象轨迹塞进去。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/examples/clqr.cpp)

### 6. 导数与数值验证

位置、姿态、运动学状态转移优先使用 Pinocchio 的解析导数，再用流形上的有限差分检查。

检查应使用：

\[ q^{+}=\operatorname{integrate}(q,\epsilon\,\delta), \]

而不是对四元数数组的某个元素直接加 \(\epsilon\)。

Pinocchio 官方 [`examples/inverse-kinematics.py`](https://github.com/stack-of-tasks/pinocchio/blob/devel/examples/inverse-kinematics.py) 可以借鉴位姿误差、`Jlog6` 和流形更新的配合方式，但不应把这个 IK 示例当作完整可执行性验证器。[GitHub](https://raw.githubusercontent.com/stack-of-tasks/pinocchio/devel/examples/inverse-kinematics.py)

------

## 八、Coal：复用已有距离残差，但把接触语义和验证边界写清楚

### 1. Aligator 已有可复用实现

可以优先审阅：

- [`frame-collision.hpp`](https://github.com/Simple-Robotics/aligator/blob/main/include/aligator/modelling/multibody/frame-collision.hpp)
- [`frame-collision.hxx`](https://github.com/Simple-Robotics/aligator/blob/main/include/aligator/modelling/multibody/frame-collision.hxx)

其中 `FrameCollisionResidualTpl` 基于 Pinocchio 几何模型查询距离，并用最近点、法向和 frame Jacobian 构造导数。你的状态是 \((q,v)\)，小型物体模型也使用 Pinocchio，因此这条接入路线很直接。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/include/aligator/modelling/multibody/frame-collision.hpp)

另一个值得参考的是 [agimus-project/colmpc](https://github.com/agimus-project/colmpc?utm_source=chatgpt.com) 的 `ResidualDistanceCollision`，尤其是最近点 Jacobian 的处理。这里借鉴残差和测试，不需要引入其整套 MPC。[GitHub](https://raw.githubusercontent.com/agimus-project/colmpc/main/include/colmpc/residual-distance-collision.hpp)

### 2. 不要把所有接触都当成碰撞

我建议将几何对分成：

```
avoid                 必须保持间距
intended_contact      指定阶段允许接触，但不允许无界穿透
structural_exclusion  机构连接处明确排除的几何对
```

例如关闭柜门时，门板与柜体局部最终接触是任务的一部分。

不能：

- 对整个门板—柜体施加始终为正的大间距；
- 也不能为了能关上门，整个过程忽略门板—柜体所有碰撞。

预期接触应绑定到具体几何区域和阶段。**这里表达的是几何接触容许关系，不是接触力约束。**

### 3. 几何近似不能破坏物体拓扑

柜体、盒子等中空物体不能随意用一个外部凸包代表，否则空腔会被填满，规划器可能把合法运动判断成碰撞。

建议按固定部件和运动部件建立几何，必要时进行凸分解；点云、网格或 CAD 的来源不必统一，但应记录几何近似精度。

### 4. 最近距离不是处处光滑

Coal 提供几何查询，不意味着整个距离函数在最近特征切换、接触或穿透处都光滑。[GitHub](https://github.com/coal-library/coal)

因此需要：

- 每个约束保留明确的几何对，不把所有几何对粗暴取一个全局 `min` 后求导；
- 对梯度在良态、分离的几何配置中做有限差分测试；
- 求解器之外，再独立检查最终路径；
- 并行评估时，不共享可变的 `Data` / 查询结果缓存。

### 5. 不只检查轨迹节点

**柜门两个合法姿态之间，线性插值其世界位置会离开铰链圆弧。**

输出曲线必须在 \(q\) 空间中重建，再做 FK。需要平滑导数时，应使用与流形相容的分段插值，并重新检查速度、限位和碰撞。

可以对输出曲线自适应加密采样，记录最小间距和采样分辨率；但不能把这种采样检查写成“已证明连续时间绝对无碰撞”。

------

## 九、结果对象要区分“求解成功”和“组件承诺成立”

建议返回：

```
GroundingResult
  status
  bound_parameters
  trajectory
    q / v / time / events
    moving_link_pose(t)
    function_pose(t)
    end_effector_pose(t)   # 仅在 grasp transform 已提供时
  report
    functional_residuals
    kinematic_constraint_residuals
    speed_limits
    sampled_clearance
    correspondence_provenance
    robot_feasibility_checked
```

尤其不要只有 `success: true/false`。

至少区分：

| 状态                    | 意义                                               |
| ----------------------- | -------------------------------------------------- |
| `INVALID_BINDING`       | 缺少功能对应、机构信息或坐标不一致                 |
| `UNOBSERVABLE_TEMPLATE` | 演示不能确定任务需要的功能信息                     |
| `GOAL_INCOMPATIBLE`     | 有明确证据表明目标不属于可行运动集合               |
| `SOLVER_NOT_CONVERGED`  | 局部优化没有找到满足条件的解，不等于已证明不存在解 |
| `VALIDATION_FAILED`     | 求解器返回了结果，但独立验证未通过                 |
| `ACCEPTED`              | 在报告声明的模型、容差和验证范围内通过             |

还应支持“已经在目标状态”的结果，不必生成无意义运动。

**默认承诺应是“物体运动学上、功能几何上通过验证”，而不是“机器人一定可执行、一定不会打滑、一定能产生所需作用力”。**

机器人可达性验证可以接一个外部 adapter；未调用时，报告明确标为未检查，不把整台机器人规划器吸收到当前组件中。

------

## 十、具体实施计划与代码组织

### 1. 建议的代码组织

不需要大量抽象层，下面这些文件已经足够容纳核心职责：

```
functo_skill/
  schema.py                 # Demo / Template / Binding / Result
  legacy_functo.py           # 原仓库离线数据适配
  functional_geometry.py    # 功能点、轴、坐标、几何目标
  motion_model.py            # 五类 Pinocchio 小模型
  compile.py                # 演示校验和模板编译
  grounding.py               # 实例绑定、事件目标、参考运动
  residuals.py               # 功能/速度等残差
  solver_aligator.py         # 唯一主求解路径
  collision.py               # 几何对策略和路径验证
  trajectory.py              # 参数化曲线、采样、导数

tests/
  fixtures/
  test_motion_models.py
  test_derivatives.py
  test_grounding.py
  test_collision.py
  test_legacy_regression.py
```

Python 负责数据和流程，Pinocchio、Coal、Aligator 承担数值核心。只有测量到自定义残差存在性能瓶颈时，再迁移相应部分到 C++；不必一开始维护两份完整实现。

### 2. 按交付物推进，而不是按“简化版功能”推进

| 工作包                        | 交付物                                        | 通过标准                                           |
| ----------------------------- | --------------------------------------------- | -------------------------------------------------- |
| **A. 固定依赖与原版基线**     | 锁定环境、FUNCTO 快照、离线 fixture、adapter  | 不调用云服务即可重建原版功能关键帧与参考轨迹       |
| **B. 统一数据和五种运动模型** | Schema、坐标规范、五类 FK/Jacobian/限位       | 任意轴和倾斜平面测试通过，配置维度与切空间维度正确 |
| **C. 模板编译**               | 刚性校验、功能坐标退化处理、事件与轨迹保存    | 模板序列化往返一致；缺失信息有明确错误             |
| **D. 实例 grounding**         | 目标关系绑定、约束内关键帧、状态条件化参考    | 新位置、形状、轴、初始状态分别改变时，绑定正确     |
| **E. 统一优化与碰撞**         | Aligator 残差、约束、Coal 策略、时间处理      | 五类均使用同一流程；导数与独立约束验证通过         |
| **F. 参数化输出与回归**       | 曲线求值、验证报告、失败分类、可选 grasp 映射 | 输出重采样后仍满足声明条件；不把失败隐藏成成功     |

这里没有把任何模式留到“以后再考虑”。**五类模式的数据接口、坐标语义和失败处理从开始就确定，开发顺序只是为了定位错误。**

### 3. 必须覆盖的测试，不只是五个成功演示

**坐标等变性测试。**对整个场景施加相同刚体变换，内部相对关系不应改变，输出世界轨迹应随之变换。这是检测坐标混乱最有效的测试之一。

**机构参数化等价测试。**同一铰链换轴方向符号、零角度约定，或者改变平面坐标基，只要物理解释等价，最终物理轨迹也应等价。

**泛化维度分离测试。**分别改变物体位置、朝向、功能点局部位置、机构尺寸、轴位置、当前状态和目标状态，不要把所有变化混在一次成功率里。

**失败用例测试。**包括共线关键点、越限目标、不兼容完整位姿、碰撞阻断、零净位移演示、重复动作、当前已完成、缺少抓握变换。

**时间一致性测试。**改变采样数量后，物理速度限制不能变化；改变允许时长后，轨迹与报告必须使用相同时间定义。

对于 revolute/prismatic，还要有一个很重要的测试：

> **当机构唯一允许的路径被障碍物阻断时，组件必须失败，而不是让优化器通过违反机构约束“绕过去”。**

### 4. 依赖版本与兼容性

不要直接组合四个库各自的最新版本。Aligator 的近期版本涉及 Pinocchio 支持和接口变更；应该以一套锁定依赖开始，再在 CI 中运行五类小模型、残差导数和碰撞 smoke tests。[GitHub](https://raw.githubusercontent.com/Simple-Robotics/aligator/main/CHANGELOG.md)

保留原版 CasADi 脚本作为**独立回归基线**即可，不需要再做一个长期维护的双后端框架。

同时，把两种实验分开报告：

- **给定正确功能对应时，轨迹 grounding 的能力；**
- **接入 FUNCTO 视觉对应后，整体的能力。**

这样才能知道改进来自机构建模，还是输入对应本身更准确。

------

## 最终建议

这个组件最合适的定位是：

> **一个 FUNCTO 式的参数化功能技能编译与实例化器：从离线演示保存功能对应和运动结构，在新实例的配置空间内生成满足功能关系的轨迹。**

真正需要新增的核心只有三块：**五类运动学模型、机构兼容的功能目标 grounding、统一的约束轨迹优化与验证。**

不需要 Drake，不需要接触动力学，也不需要完整机器人系统。与此同时，坐标系、目标语义、时间、功能坐标退化、几何接触策略和失败报告都必须现在设计清楚——这些才是会决定后续架构能否成立的部分。