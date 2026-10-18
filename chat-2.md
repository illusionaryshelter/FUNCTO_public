# v2 技术评审：离线提取、在线迁移与五个 Skill 校验

我重新阅读了你最新上传的 《离线提取-在线迁移-v2.md》全部102行。本次以这份文件为唯一的架构基准，不再沿用之前对话中与 v2 冲突的设计。

先给结论：

v2 的整体方向合理，而且在仿真 Teleoperation 条件下，工程可行性比此前的纯视频方案高很多。但它目前还不能直接实现完整的 Skill Transfer：Function Region 自动提取、MARCO 的实际使用方式、空间 Region 的定位、参考轨迹的几何适配和 Push 闭环控制，仍缺少明确算法。

我认可并保留你已经确定的设计：

- 已完成 Skill 事件分段。
- Grasp / Manipulation 分类，Manipulation 保留 master / slave 区分。
- 仿真 GT 只用于获取离线刚体与机器人末端位姿；Function Region 和 Functional Frame 仍然必须提取。
- 使用 MARCO、GRF、语义—几何联合对应进行新场景迁移。
- 运行期双固定 RGB-D，但不使用 MVTracker。
- 抓取前建立物体碰撞几何，抓取成功后绑定到末端并通过 FK 更新。
- Push Object 没有 master object、没有刚性抓取，使用 Point2Pose 独立跟踪被推物体。

这些都与你的 v2 主干一致。

## 一、先指出 v2 中需要解决的六个问题

| 优先级 | v2 当前表述                              | 审查结果                                                     |
| ------ | ---------------------------------------- | ------------------------------------------------------------ |
| P0     | 仿真中 Function Region / Frame 可跳过    | 只能跳过位姿估计，不能跳过功能区域和坐标系的提取             |
| P0     | MARCO 提取特征，再做点对应               | 方向可行，但需要修正成真实的 MARCO 特征与匹配接口            |
| P0     | 空间 Region 直接用语义特征相似度匹配     | 不充分：空闲空间不一定存在可提取视觉特征的实体表面           |
| P0     | Open Drawer/Cabinet 使用夹爪位姿作为参考 | 夹爪用于当前活动刚体状态估计；轨迹的 master/reference 仍应是静止柜体 |
| P1     | 运行期位姿跟踪通常由 Manipulation 决定   | 应由实际连接状态决定：刚性抓取以 FK 为主，Push 以 Point2Pose 为主 |

前四项不是措辞问题，而是如果不修正，五个校验 Skill 中至少有几个会在接口层面无法运行。

### 一个重要的现成组件核查：MARCO

这次我确认了 v2 中的 MARCO 是 [MARCO: Navigating the Unseen Space of Semantic Correspondence，CVPR 2026](https://github.com/visinf/MARCO)。

其公开版本使用 DINOv2 ViT-L/14，323M 参数，不是此前讨论的 DINOv3 ViT-S/16。

更重要的是，它的官方接口是：

```
model = torch.hub.load(    "visinf/MARCO",    "marco",    pretrained=True,    trust_repo=True,    device=device,)inputs = model.preprocess_data(    source_image,    target_image,    source_keypoints,    device=device,)target_keypoints = model(**inputs)
```

官方的 `forward()` 输入是图像对和源二维关键点，直接输出目标二维关键点；其内部另有 `forward_backbone()` 可以获得上采样特征图。([MARCO README](https://github.com/visinf/MARCO)、[模型源码](https://github.com/visinf/MARCO/blob/main/models/marco.py))

这决定了我们应该实现一个薄适配器：

```
离线：源图像 + 源关键点
        ↓
保存参考图像、关键点与源三维几何

在线：源图像 + 目标图像 + 源关键点
        ↓
MARCO 得到目标二维对应点
        ↓
目标 RGB-D 反投影
        ↓
3D 几何一致性验证
```

需要缓存源特征时，可以从 MARCO 的 `forward_backbone()` 提取，但必须自行包装、保持 crop 坐标映射一致，不能将其视为一个已经发布的、可独立调用的三维 descriptor API。

所以 v2 的 MARCO 选择可以保留，而且不用自己重新训练；但它解决的是二维语义对应候选，不是直接输出可靠的三维 Functional Frame。

为避免这次长篇分析再次丢失内容，我也生成了一份独立的完整技术规范，包含 v2 的逐节审查、五个 Skill 的输入输出走查、建议的自研算法和验收条件。没有修改你的原始 v2 文件。

下载完整技术审查与五技能算法规范.md

以 v2 为准的工程规格与可行性审查

## 二、离线：仿真 Teleoperation 究竟给我们省掉什么？

这里应当明确划出边界。

仿真提供的 GT 位姿，可以直接代替离线 Point2Pose 的刚体运动估计，但不应该代替你希望自主获得的 Function Region、Functional Frame 和交互语义。

离线读取：

```
DemoInput:
    rgb:             uint8[T, H, W, 3]
    depth_m:         float32[T, H, W]
    camera_K:        float32[3, 3]
    camera_extrinsic: float32[4, 4]
    timestamps:      float64[T]

    body_poses_gt:   dict[body_id, float32[T, 4, 4]]
    ee_poses_gt:     float32[T, 4, 4]
    gripper_states:  float32[T, ...]

    skill_segment:   EventSegment
```

这里有一个实施前提：GT 必须包含独立运动部件的位姿。对于柜门、抽屉，只有整个柜子的 root pose 不够，必须取得活动门或抽屉 link 的位姿。

机器人自身的几何和夹爪状态可以使用 Teleoperation 系统已经具备的机器人模型；新物体的功能区域仍从图像及 RGB-D 中提取，不直接从仿真 mesh 标签读取。

### 2.1 自研算法 A：从交互轨迹自动提取 Function Region

这是 v2 当前第一个真正缺失的算法。

我建议输入：

```
extract_function_region(
    event,
    ee_pose_trajectory,
    gripper_geometry,
    body_pose_trajectory,
    object_rgbd_cloud,
    region_candidates,
)
```

输出：

```
FunctionEvidence:
    contact_points_body: [N, 3]
    contact_weights:     [N]
    approach_direction:  [3]
    function_frame_candidates: [K, 4, 4]
    observation_quality: dict
```

这里不要求每个 Skill 必然有实际接触点，但对 Grasp 应当优先使用接触证据。

第一步：从演示中获得物体局部几何。

使用 SAM 对源物体或部件分割。在有效深度像素上反投影，再利用 GT 位姿将点转换到物体固定坐标系：

\[ p_i^B= \left({}^{W_D}T_B(t)\right)^{-1}p_i^{W_D}(t). \]

其中：

- \(p_i^B\)：物体局部坐标中的第 \(i\) 个表面点。
- \(p_i^{W_D}(t)\)：时刻 \(t\) 的世界坐标观测。
- \({}^{W_D}T_B(t)\)：仿真 GT 刚体位姿。

不同时刻看到的表面可以积累，但只能积累真实观测过的部分。

第二步：根据交互时刻与夹爪几何寻找接触区域。

利用已知机器人夹爪模型和 EE GT，计算物体表面点到夹爪接触面的距离。

对交互建立附近的有效帧累积：

\[ w_i=\sum_{t\in\mathcal T_c} v_i(t) \exp\left( -\frac{d_i(t)^2}{2\sigma^2} \right). \]

其中：

- \(\mathcal T_c\)：接触建立附近的时间窗口。
- \(d_i(t)\)：第 \(i\) 个物体表面点到夹爪接触面的距离。
- \(v_i(t)\)：该点当前观测是否有效。
- \(\sigma\)：用于控制距离权重衰减的尺度。

这不是 GT 接触标签，而是由可见几何和运动构造的接触候选评分。

如果接触时已经遮挡，就使用接触前看到的有效表面，再通过 GT 刚体运动传播到接触时刻。

对于刚性抓取，还可以用抓取后相对位姿是否近似恒定来验证事件：

\[ {}^ET_B(t)= \left({}^{W_D}T_E(t)\right)^{-1} {}^{W_D}T_B(t). \]

稳定抓取时，它在该操作阶段应当近似恒定。

第三步：从接触区域恢复 Function Frame。

功能原点可以来自高权重接触点的鲁棒中心，方向来自：

- 局部表面法向；
- 末端接近方向；
- 接触区域的空间分布；
- 必要时的相邻局部几何。

如果某个方向不能可靠确定，保留方向歧义，不应让 PCA 随意补出一个看似正确的旋转轴。

最后计算：

\[ {}^BT_F= \left({}^{W_D}T_B(t_c)\right)^{-1} {}^{W_D}T_F(t_c). \]

这就是 Function Region/Frame 的离线核心算法。 它不依赖 `open_drawer`、`grasp_mug` 等具体名称。

一个现实限制是：如果功能表面在整个演示中从未被观测，而且没有充分的几何证据，单靠 GT 刚体位姿也无法推导精确接触点。此时必须报告提取不确定，而不是把某个最近点当作真值。

### 2.2 对目标 Region，也需要保存实际视觉证据

v2 对 Function Region 的描述偏向单个交互部位，但对 Manipulation，我建议至少区分：

Slave 侧的功能区域：被操纵物体上参与运动或目标交互的部分。

Reference/Target 侧的功能区域：目标物体表面，或者某个空间区域。

这不是新 Skill 分类，而是同一条相对运动必须具备的两个几何端点。

对于 `put_bowl_in_drawer`，需要同时知道碗的作用区域以及抽屉内部的目标区域；只存抽屉内部是不够的。

对于 `open_drawer`，可以沿用此前 Grasp Skill 已提取的把手 frame，并从静止柜体恢复 reference frame。如果 Manipulation 片段一开始把手就已经被完全遮挡，不能要求它再从该片段独立完成把手分割。

## 三、在线：最关键的三个自研算法

### 3.1 自研算法 B：MARCO + GRF + RGB-D 的三维功能对应

我不建议将 v2 的五步理解成严格串行的：

```
MARCO → GRF → RANSAC → OAReg → Frame
```

原因是，MARCO 提供的可靠二维对应本身就可以产生三维刚体初值；GRF 应作为额外候选，而不是必须首先计算的唯一初始化。

建议的计算顺序：

```
Source Region
├── 原演示 RGB crop
├── 源二维 keypoints
└── 与 keypoints 对应的源三维点
                   │
                   ▼
               MARCO
                   ▲
                   │
Target RGB-D ── SAM 功能区域候选
                   │
                   ▼
         目标二维匹配 keypoints
                   │
            Depth lifting
                   │
                   ▼
         Source/Target 3D Pairs
                   │
        ┌──────────┴──────────┐
        ▼                     ▼
   3D RANSAC              GRF 候选
        │                     │
        └──────────┬──────────┘
                   ▼
     Partial-overlap geometric check
                   │
         Local shape refinement
                   │
                   ▼
         Target Functional Frame
```

参考接口：

```
match_function_region(
    source_rgb,
    source_keypoints_px,
    source_keypoints_xyz,
    target_rgbd_views,
    target_masks,
) -> CorrespondenceResult
```

内部操作：

① MARCO 二维点迁移。 将源功能区域上有实际 RGB-D 支持的三维锚点投影到源图像，保存源像素坐标。对每个新目标相机视角运行 MARCO，获得目标二维对应。

② RGB-D 反投影。 对目标 MARCO 返回的有效像素读取深度，得到世界坐标中的目标候选点。对落在 mask 外、深度边缘、未知空间或明显遮挡处的结果予以拒绝。

③ 三维匹配验证。 使用 RANSAC、源/目标点间距离、局部法向、对应覆盖率和空间分布检查候选。GRF 可以另外生成多组姿态初值；两种初值统一参与排序。

④ 局部形状适配。 当刚体配准不能充分解释源目标的功能部位形状差异时，可以借鉴 OAReg 进行局部非刚性对应；它只用于确定功能锚点的新位置，不代表机器人操作中的物体会非刚性变形。

⑤ 求 Functional Frame。 功能原点由映射后的锚点恢复，方向由具有足够空间分布的锚点通过加权刚体拟合恢复。对对称或退化结构保留多候选。

这套过程直接符合 v2 的 MARCO 与 GRF 设计，但不会把某个 2D matching score 错误地当成可靠的 SE(3) 位姿。

### 3.2 自研算法 C：空间目标 Region Grounding

这是目前 v2 中风险最大的地方。

v2 目前对没有明确物体实例的目标区域，写的是直接做语义特征相似度匹配。

但语义特征只能匹配可观测的图像结构，不能保证一块空闲空间具备可匹配的局部特征。

因此需要明确接口，但是仅供参考：

```
ground_target_region(
    source_target_evidence,
    target_semantic_prompt,
    rgbd_left,
    rgbd_right,
    scene_bindings,
) -> GroundedRegion
```

计算顺序是：

先定位区域的空间锚点。

可以是目标实体、可见的目标区域、桌面标记、周边稳定几何，或用户提供的图像位置。VLM 负责找候选 ROI，MARCO 负责在确实存在视觉表面对应时辅助匹配。

再获取实际三维几何。

对 ROI 内及其周边的有效 RGB-D 点提取局部表面。通过深度观测构建已知表面、已知空闲空间和未知空间。

最后生成候选目标 frame 或目标位姿集合。

如果目标具有明确边界，就可以得到较强的目标位置约束；如果目标是大面积均匀区域，则可能存在多个等价的放置位置。

统一输出：

```
GroundedRegion:
    reference_frame_candidates
    visible_geometry
    verified_free_space
    unknown_space
    valid_goal_candidates
```

不需要根据 Skill 名字判断是不是 shelf 或 drawer。

但有一个必须承认的可辨识性限制：

Push 的“指定区域”如果既没有可见标记、目标 ROI、空间语言参照，也没有额外坐标输入，那么仅靠视频和目标物体本身无法确定一个唯一的指定位置。

这不是 MARCO 不够强，而是缺少定义目标所必需的信息。

### 3.3 刚性抓取与 Push 在这里必须分开

对于刚性抓取后的 Manipulation：

\[ {}^{W_N}T_E^{seed}(s) = {}^{W_N}T_{F_N}^{seed}(s) \left({}^{E}T_{F_N}\right)^{-1}. \]

其中，\({}^{E}T_{F_N}\) 是新场景本次成功抓取后实际建立的夹爪—物体功能 frame 关系。

对于 Push，不能使用固定 \({}^{E}T_{F_N}\)。

离线必须额外保存时变的：

\[ C_D(s)= \left({}^{W_D}T_{F_D}(s)\right)^{-1} {}^{W_D}T_E(s). \]

运行期将其接触几何适配到新物体，生成末端运动初值，并用 Point2Pose 实时测量物体是否真的沿参考轨迹运动。

我建议 Push 采用短段执行—观测实际位移—重新优化剩余段的闭环，不必训练 policy。需要时可以从小幅接触运动与物体响应中在线估计局部运动响应，但应把它作为一个需要验证的控制器，而不是认为迁移末端轨迹即可保证推移结果。

## 四、五个校验 Skill：逐项跑通 v2

下面直接检查：每项离线究竟存什么、在线究竟 Ground 什么、执行时信什么、哪里可能失败。

| Skill         | 离线必须提取                                                 | 在线 Grounding                         | 执行状态与主要缺口                                           |
| ------------- | ------------------------------------------------------------ | -------------------------------------- | ------------------------------------------------------------ |
| Bowl → Drawer | Bowl 功能区域、抽屉内部目标 Region、两者相对轨迹、终点几何   | Bowl 功能 frame + 当前抽屉内部可见区域 | 刚性抓取后 Bowl 用 FK；必须检查碗的占据空间与抽屉入口，未观测内部不能当作空闲 |
| Mug → Coaster | Mug 抓取/放置区域、Coaster 交互区域、相对轨迹与末态          | Mug 与 Coaster 的功能对应              | Mug 用 FK；Coaster 若被外部移动，要独立更新目标位姿          |
| Open Cabinet  | 活动柜门 link GT、静止柜体 frame、把手功能区域、相对 SE(3) 轨迹 | 柜门功能 frame、静止柜体参考 frame     | 抓取后由 FK 推断门位姿；把手遮挡不成问题，但不同铰链位置可能让源轨迹不可行 |
| Open Drawer   | 活动抽屉 link GT、静止柜体 frame、把手区域、相对轨迹         | 抽屉功能 frame、静止柜体 frame         | 与 Open Cabinet 共用算法；滑脱后 FK 不能再当抽屉实际状态     |
| Push → Region | 被推物体 GT 轨迹、EE 时变接触轨迹、指定目标 Region           | 物体功能区域和目标 Region              | Point2Pose 跟踪真实物体；没有 master 不等于没有空间参考；不能依赖固定 attachment |

### 校验 1：Bowl 放进 Drawer

这里是 `target_kind=REGION` 最重要的测试。

离线至少需要得到：

```
slave = bowl
target_region = drawer interior
target_region.owner = moving drawer link

H_D(s) = bowl_function_pose relative to drawer_region
```

关键不是知道整个 drawer 的精确 CAD，而是在演示中确认目标区域的实际位置和可观测几何，并把它固定到活动抽屉的局部坐标系。

新场景里，如果抽屉已经在前一 Skill 打开，应当继承已确认的抽屉实例 ID 和内部几何。 不能因为进入下一 Skill 就重新当成一个完全未知实体。

终点迁移后，必须检查实际 Bowl 几何与抽屉入口的空间关系。若源或目标的内部始终不可见，系统无法保证没有隐藏障碍。

这项 Skill 会直接检验 `REGION Grounding` 是否真的实现，而不是只在 SAM 上换一个 prompt。

### 校验 2：Mug 放到 Coaster

这项更容易形成完整的跨实例对应。

Coaster 是可分割物体，源/目标交互区域通常具有明确边界，能够利用 MARCO 的图像关键点对应和 RGB-D 几何恢复目标 frame。

Mug 的放置作用区域则通过源交互几何与新实例局部对应迁移。

刚性抓取后，机器人末端轨迹由本次 `T_EF` 推导，而不是沿用源演示中的固定手—杯关系。

这里的独立状态风险来自 Coaster 被挪动。 Mug 的 FK 无法推出目标 Coaster 的新位姿，需要重新观测 Coaster，并重新实例化剩余轨迹。

### 校验 3、4：Open Cabinet / Open Drawer

这两项应该共享完全相同的离线与在线算法。

离线使用活动 link 和静止参考 body 的 GT 位姿，存储相对 SE(3) 运动。

如果前一个 Grasp Skill 已经提取把手功能 frame，Manipulation 可以沿用它，不应强迫 SAM 在把手被夹爪遮挡后重新识别。

运行期仍需保留静止柜体作为运动参考，而不是将 EE 当成 reference。

EE 的作用是根据已建立的固定抓取变换，预测活动 link 当前位姿：

\[ {}^WT_B(t) = {}^WT_E(t)\,{}^ET_B. \]

其中 \({}^ET_B\) 是本次抓取建立的固定变换。

但源演示中的目标仍然是：

\[ {}^RT_B(s), \]

即活动部件相对于静止参考结构的运动。

这里还应明确：仅仅知道新把手位置，并不能保证知道新柜门的铰链轴或新抽屉的全部行程。 当新机构几何差异足以改变可行运动时，源相对轨迹可能需要受约束的在线修正；若缺少可辨识的机构信息，就不能宣称一次静态 Grounding 已经正确解决。

### 校验 5：Push 到指定区域

这是 v2 中最需要补算法的校验 Skill。

离线至少应输出：

```
slave = pushed object
master = None
target_region = specified scene region

H_D(s) = object relative to target region
C_D(s) = EE relative to object, time-varying
```

`H_D(s)` 记录物体要如何移动，`C_D(s)` 记录演示末端如何与物体保持操作接触。

在线通过双 RGB-D 完成对象与目标区域的初始化后，选择一台能够可靠观测该物体的 RGB-D 接入 Point2Pose。

Point2Pose 是运行期真实物体状态来源，而不是 FK。

如果需要从左相机切到右相机，Point2Pose 官方并未提供现成的跨相机统一地图切换 API；仍需以相机外参、已有物体位姿、对象身份和新观测实现重定位。这是待集成的工程接口，而不是重新引入 MVTracker 的理由。

推移闭环应比较实际物体状态和参考物体状态，修正剩余接触运动，而不是仅比较 EE 是否到达规划 waypoint。

## 五、碰撞模型：v2 的设计合理，但还缺两项处理

我赞同你在 v2 第93行提出的原则：碰撞模型应在抓取前建立，抓取后绑定到末端，通过 FK 更新。

但是：

第一，双 RGB-D 融合出来的是部分观测几何，不自动等于完整封闭物体模型。 对杯子、碗这类具有不可见背面的物体，使用纯观测点云构建碰撞模型可能漏掉真实占据空间。需要使用带安全余量的保守代理，并将无法确认的部分标记为不确定，而不是宣称已知完整形状。

第二，抓取成功后需要正确更新 Planning Scene 中的实体状态。 物体不能既作为原位置的静态障碍，又作为附着在夹爪上的移动障碍重复存在；释放之后也必须根据实际观测，将其放回环境碰撞场景。

同时，FK 提供的是抓取稳定假设下的物体预测位姿。考虑到你最终研究 Recovery，重要状态事件仍需要独立观测或其他可靠证据确认，否则抓取滑脱后会出现“FK 认为目标已完成、物体实际停在半途”的问题。

但是这个功能可能作为模块插件，无需作为紧耦合的部分。

## 六、最终我建议 v2 补齐的最小自研模块

这次我不建议再引入一个庞大的新架构。保留 v2 的主干，只需要把五个尚未明确的接口真正实现。

| 自研模块                       | 输入                                               | 输出                                 | 核心难点                         |
| ------------------------------ | -------------------------------------------------- | ------------------------------------ | -------------------------------- |
| `FunctionRegionExtractor`      | 演示 GT 位姿、RGB-D、夹爪几何、事件                | 功能锚点、Function Frame、源图像证据 | 遮挡与实际接触位置               |
| `MARCO3DCorrespondence`        | 源图像/源2D-3D锚点、目标 RGB-D                     | 新实例三维对应、Frame 候选和残差     | 大视角、部分重叠、对称、形状变化 |
| `SpatialRegionGrounder`        | 源目标区域、prompt、当前双 RGB-D                   | 目标空间 frame/候选范围及可信几何    | 空间 Region 并非可直接分割的物体 |
| `RelativeTrajectoryRetargeter` | 源相对 waypoints、当前功能 frame、目标候选         | 从当前状态出发的轨迹初值             | 目标变化、形状变化、机构可行性   |
| `ContactOnlyExecutor`          | 推移物体参考轨迹、EE 接触轨迹、Point2Pose 实测状态 | 修正后的短段机器人操作轨迹           | 摩擦、接触改变、真实物体响应     |

前三个是 Skill Generalization 的核心，第四个把迁移结果变成可执行参考，第五个是 Push 在没有 rigid attachment 时不可省略的执行环节。

目前这份 v2 的合理性，我会这样评价：

架构层面：成立。 尤其是仿真 Teleoperation 直接提供离线 SE(3) 位姿、正常刚性抓取使用 FK、抓取前建立碰撞模型、Push 使用 Point2Pose，这些决策形成了一条相对清楚且不依赖新 policy 的路线。

算法层面：尚未完成。 MARCO 从二维到三维的可信对应、功能区域自动提取、空间 Region 定位及动态接触轨迹迁移，都是需要真正研究和验证的部分。

完整可行性：有条件成立。 在局部功能结构具有可对应几何、目标区域具有足够可观测信息、执行环境和接触状态可以验证的范围内，这条方案值得实现。但无法从一次演示和有限 RGB-D 观测保证任意隐藏机构、任意目标空间及任意推移接触都正确。

我认为下一步最值得固定下来的不是模型选型，而是这五个自研模块的输入输出契约、失败返回值和测试指标。把它们实现清楚之后，v2 的整体系统就不需要再因为新增一个 Skill 而重构架构。