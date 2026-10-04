我们现在这个新证据把问题又缩小了一层，而且方向非常明确：

> **当前 `Coal + 占据体素` 这条所谓“实体检查”并不是一个真正独立于 ESDF 表示的环境几何检查。机器人几何换成了来源实体，但环境几何仍是由 ESDF 的零/负值格经过“整格立方体化”得到的第二层代理。**

所以 `−0.03042025 mm` **不能解释成“2 mm fine map 也确认真实碰撞”**。更准确的解释是：

> 在 `scatter` 生成的 ESDF 零值 seed 上，把每个 2 mm cell 变成完整立方体后，Coal 返回 link6 与这些立方体的负距离。这个返回值不是退出整个体素并集所需的全局最小平移。

这和真实桌面仍有约 \(+0.642\) mm 间隙完全可以同时成立。

这些数字与体素表示造成额外占据厚度的假设相容，尚不能分离各项误差来源。表示对照及导出坐标检查的实测结果见[专项报告](../sim-version-plans/2026-10-04-tsdf-surface-review.md)；下文的环境表示迁移属于架构建议，尚未接入正式规划。

------

## 1. 先看体素表示的数量级关系

我们当前 Coal 环境代理实际上相当于：

\[ \mathcal O_h = \bigcup_{k:\phi_k\le0} \left( c_k+ [-h/2,h/2]^3 \right) \]

其中：

- \(\mathcal O_h\)：Coal 最终看到的环境占据集合；
- \(\phi_k\)：第 \(k\) 个 ESDF cell 的值；
- \(c_k\)：该 cell 中心；
- \(h\)：voxel 边长；
- 每个被选择的 cell 都被扩展成完整立方体。

对一个单位表面法向量

\[ n=(n_x,n_y,n_z), \]

轴对齐 voxel 在该法向上的“半厚度”是

\[ \rho_h(n) = \frac{h}{2} \left( |n_x|+|n_y|+|n_z| \right). \]

如果桌面基本和 grid 对齐，

\[ n\approx(0,0,1), \]

那么就是近似

\[ \rho_h\approx\frac h2. \]

于是：

### 15 mm coarse

\[ \frac{h}{2}=7.5\text{ mm}. \]

我们的真实桌面距离约：

\[ d_{\rm truth}=+0.642\text{ mm}, \]

而 Coal coarse：

\[ d_{\rm coarse}=-6.0051\text{ mm}. \]

两者差：

\[ 0.642-(-6.0051) = 6.6471\text{ mm}. \]

这就在 **7.5 mm 半 voxel** 的尺度以内。

### 2 mm fine/scatter

\[ \frac h2=1\text{ mm}. \]

Coal 得到：

\[ d_{\rm fine}=-0.03042\text{ mm}. \]

相对真实桌面的差：

\[ 0.642-(-0.03042) = 0.67242\text{ mm}, \]

同样在 **1 mm 半 voxel** 的尺度以内。

这里的半厚度只描述立方体相对格中心的支撑距离，不是 TSDF 重建误差、seed 偏移或 Coal 负距离的误差上界：

> **两组数值与体素离散化的尺度相容；由于范围、分辨率和 seeding 同时变化，不能据此认定距离差按 voxel 厚度缩小，也不能将两种不同几何的距离差全部归因于立方体化。**

因此不能仅凭这些数字把 link6 的主要问题归为“ESDF 本身错误”或某一种表示转换。

更准确的是：

> **ESDF → `field<=0` → full voxel cubes 把非正值样本对应的格子解释为完整占据立方体。ESDF 零样本不等同于连续 TSDF 零等值面。**

------

# 2. 而 `gather` 又会在这之前进一步加厚 surface seed

这次我们特别指出：

> 正式规划：`gather`
> 扩大 fine 诊断：`scatter`

这个区别非常重要。

cuRoboV2 官方现在已经把两者语义写得非常清楚：

### `scatter`

一个符合 seed 条件的 TSDF voxel 只映射到包含其中心的那个 ESDF cell。官方将这种映射描述为：

> exactly one ESDF voxel thick

并且官方直接说：

> **scatter faithfully matches the TSDF surface geometry**

### `gather`

每个 ESDF cell 不仅检查中心，还检查 6 个 face-center，因此可能捕获邻居 cell 中的 surface voxel，形成：

> **dilated seed band (~1.5 voxels thick)**

官方明确说，这是为了得到更高 collision recall，但代价是**不再精确匹配 TSDF surface**。[nvlabs.github.io](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

这些说明解释了两种 seed 方法的设计差异，不构成当前场景的严格厚度上界，也不能直接预测某个姿态的 Coal 距离变化。

本地 `builder_esdf.py` 的两种方法均使用 `abs(tsdf) <= 0.9*h` 或 `tsdf < -(truncation_distance-1.1*h)` 作为 seed 条件，其中 `h` 是 TSDF 格距。当前截断距离为 `2*h`，这两条条件合起来包含全部有效负 TSDF 样本及正侧表面带。因此 seed 不能一概解释为连续表面上的采样点。

所以我们正式路径的几何链实际上是：

```
RGB-D
 ↓
TSDF 有效样本及 seed 条件
 ↓
gather
 ↓
可能扩张的 ESDF seed 集合
 ↓
field <= 0
 ↓
每个被选中的格变成完整 voxel cube
 ↓
Coal collision geometry
```

这里包含**seed 邻域扩张与格子立方体化两个不同步骤**，不能把它们的厚度简单相加或预先归因到某个碰撞见证。

因此：

> 用这条 Coal 路径去证明“CuRobo 的 ESDF 是否误报”

存在一定程度的循环论证。

因为 Coal 的环境仍然源自同一个已经经过 gather 的 ESDF。

------

# 3. `scatter` 下 32782 个 cell 全部恰好等于 0，这个现象尤其关键

我们说：

> 负值格 = 0
> 所有 32782 个选中格都是 ESDF zero cell

这意味着在这次诊断里：

```
occupied = field <= 0
```

实际上等价于：

```
occupied = field == 0
```

这只说明所选格子的 ESDF 都为零；按照前述 seed 条件，其中仍可能包含原 TSDF 为负的样本，不能推出“没有内部样本”。

它看到的是：

> **ESDF 零值 seed cells 的立方体并集，未必是只有一格厚的连续表面外壳。**

下面仅示意把格中心变成完整立方体的操作，不代表实际 seed 数量、厚度或与真实表面的对齐关系：

```
真实连续 surface:

────────────────────────────


scatter ESDF zero centers:

   •   •   •   •   •   •


变成完整 2 mm cubes:

┌───┬───┬───┬───┬───┬───┐
│   │   │   │   │   │   │
└───┴───┴───┴───┴───┴───┘
────────────────────────────
        ↑
   形成有限厚度 shell
```

所以：

\[ d_{\rm Coal}<0 \]

只能说明：

> robot source geometry 与这些**占据立方体**相交。

不能推出：

> 来源机器人实体与 TSDF 零等值面也相交。

更不能推出：

\[ d_{\rm physical\ table}<0. \]

这一点现在应该成为我们报告里的明确证据边界。

------

# 4. 官方 API 实际上已经给我们准备了更合适的两种导出方式

我们发现的这两个接口非常重要，而且不是“我们重新发明一套检查方法”。

## A. `extract_mesh()`

当前 V2 的 `Mapper.extract_mesh()` 直接调用 TSDF integrator，通过 marching cubes 提取：

\[ \text{TSDF}=0 \]

的三角等值面。

源码明确写的是：

- `level=0.0`
- GPU marching cubes
- 从 `_tsdf_integrator.extract_mesh(...)` 导出

公开的 `Mapper.extract_mesh(refine_iterations=0, surface_only=True)` **不接受 `level` 参数**；`level=0.0` 是底层 integrator 的默认值。调用公开 API 时不要传入 `level=0`。

它**不经过 ESDF gather/scatter seeding**。[GitHub](https://github.com/NVlabs/curobo/blob/main/curobo/_src/perception/mapper/integrator_esdf.py)

官方 volumetric mapping 文档也把两件事明确分开：

- TSDF → mesh
- TSDF → ESDF → \(O(1)\) trilinear collision distance query

也就是说官方架构本身没有把“ESDF 零格”当 mesh 使用。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.examples.getting_started.volumetric_mapping.html?utm_source=chatgpt.com)

这适合用 Coal 的三角网格 BVH（包围体层次树，保留三角形并加速距离和相交查询）做环境表示对照，但仍来自同一份观测 TSDF，不是独立重建的环境真值。

另有一项已确认的本地导出坐标差异：`builder_coord.voxel_to_world` 将 TSDF 样本放在 `index+0.5`，而 `builder_mesh.generate_mesh_kernel` 将相同样本值放在 `index`。在 `refine_iterations=0` 时，每轴补回半个 TSDF 格距可恢复样本中心坐标。FUNCTO 的 [`extract_tsdf_surface(mapper)`](manipulation/mapper_surface.py) 固定使用零 refinement，返回修正后的 CPU 顶点和原三角索引；它不改第三方库、不补洞、不做凸包，也不增加碰撞余量。该函数是表面导出修正，**正式规划尚未使用它替换验收判据**。

这些顶点位于 Mapper 融合观测所用的坐标系，已含 `grid_center`；被分区目标物体（owner）的局部地图在查询时只应用一次该物体位姿。三角表面可能开放，正距离不能证明机器人处于障碍外部，也不能证明未知空间没有障碍。

------

## B. `extract_occupied_voxels()`

这个接口也不是从 `compute_esdf()` 的 `feature_tensor<=0` 提取。

它直接从 **TSDF integrator** 提取 occupied/surface voxels：

- `surface_only=True` 时提取 near-surface voxels；
- `sdf_threshold` 默认是底层 TSDF voxel size；
- 返回 voxel/subvoxel sample center。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

因此它同样绕开 `gather/scatter`。

但我会强调：

> **如果我们把 `extract_occupied_voxels()` 得到的中心再次变成完整边长 \(h\) 的立方体，那么半 voxel 的几何膨胀依然存在。**

它直接表达 TSDF 阈值选出的样本，适合与 ESDF seed 做来源对照；如果目标是查询 TSDF 零等值面的距离，**保留三角拓扑的 mesh 更直接**。两者都不能自动升级为完整障碍实体或真实几何。

------

# 5. 因此我现在建议把架构明确成“三种几何表示，三种职责”

这是碰撞环境表示与验收职责的架构建议，不涉及 skill 参数。当前 OMPL 状态验收、轨迹优化候选验收和稠密轨迹验收仍保留 sphere/ESDF 的硬拒绝；Coal 检查通过不会自动覆盖这些拒绝。要改变最终判据，需要先明确完整占据、表面距离、接触权限和未知空间各自的职责，再经架构决定统一修改相关入口。

现有“未知可尝试”政策应保持：不可见区域缺少几何证据，不应因表面导出而被宣称已证自由，也不能在这次修正中静默改为一律禁止尝试。地图层选择的有效支撑要求与未知空间政策分别处理。

## 第一层：CuRobo sphere + ESDF

保留现在的：

```
robot spheres
      ×
gather ESDF
```

用于：

- L-BFGS 梯度；
- 快速 cost；
- 远离障碍物；
- broad phase。

`gather` 为了提高 recall 而故意保守，这是官方设计，不必因为现在的问题就全面改成 scatter。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

------

## 第二层：source robot geometry + TSDF mesh + Coal

用于诊断时，proxy 判断危险后可比较另一种环境表示：

```
sphere / ESDF says collision
          ↓
source robot collision geometry
          ×
extract_tsdf_surface(mapper)
          ↓
Coal 三角表面距离/相交查询
```

直接用途是分析 sphere/ESDF 与三角表面查询的分歧。若要参与 OMPL 状态、轨迹优化候选及稠密轨迹的最终验收，仍须满足统一的占据、包含关系、接触与未知空间合同；**开放 BVH 表面的正距离不能单独放行状态或轨迹**。

当前 `robot_geometry_check._scene_shape()` 对 `record['kind'] == 'mesh'` 的分支将顶点构造成填充凸包，面向来源凸实体。TSDF 网格可能非凸且开放，不能直接送入这个入口；诊断应保留其 faces 构建 BVH。BVH 的表面相交和距离也不能替代完整实体的包含关系检查。

Coal 本身就是为 GJK/EPA collision 和 distance narrow phase 设计的，也支持安全 margin 和距离下界。[GitHub](https://github.com/coal-library/coal?utm_source=chatgpt.com)

可以借鉴 self-collision 中将优化梯度与来源实体验收分开的职责：

```
sphere provides optimization gradient
source geometry provides acceptance decision
```

但 self-collision 使用的是来源完整实体；world side 若从：

```
ESDF-derived cubes
```

改为：

```
TSDF zero-level mesh
```

还会失去原占据立方体明确的体积含义。因此这不是可直接等价替换的实现修改。

------

## 第三层：simulator native geometry

只在仿真开发阶段作为 oracle：

```
MuJoCo table / object collision geometry
```

用于回答：

> Mapper 和 narrow phase 相对“真实仿真世界”误差多少？

它不应该成为最终真机算法的一部分。

------

# 6. 这样仍然可以满足我们之前强调的“同一地图选择规则”

这个要求不需要放弃。

应该统一的是：

> **选哪个 map layer / 哪个 observation snapshot**

而不是强制：

> **所有模块必须使用同一种几何表示。**

在明确上述验收合同后，可以保留以下同层表示对照关系；图中的 Coal 项首先是表面查询证据：

```
MapSelector(q, link)
        ↓
    chooses FINE
        ↓
┌──────────────────────────┐
│ same fine mapper snapshot│
├──────────────────────────┤
│ ESDF → CuRobo gradient   │
│ TSDF mesh → Coal evidence│
└──────────────────────────┘
```

或者：

```
MapSelector(q, link)
        ↓
    chooses COARSE
        ↓
┌──────────────────────────┐
│same coarse mapper snapshot│
├───────────────────────────┤
│ ESDF → CuRobo gradient    │
│ TSDF mesh → Coal evidence │
└───────────────────────────┘
```

也就是：

\[ \text{map identity一致} \]

但

\[ \text{representation可以不同}. \]

这反而更标准。

------

# 7. 表示对照实验的比较原则

固定：

- 当前 RGB-D；
- 当前 TSDF；
- 同一个 link6 \(q\)；
- 同一个 13.378 s OMPL boundary；
- 同一来源机器人 collision geometry。

然后只换环境表示。

以下保留表示对照的原始比较项；问号对应的实测值见[专项报告](../sim-version-plans/2026-10-04-tsdf-surface-review.md)，不要求重复同一实验：

| 环境表示                               | seed                  | Coal link6 distance |
| -------------------------------------- | --------------------- | ------------------- |
| Simulator 原桌面                       | —                     | **+0.642 mm**       |
| TSDF `extract_mesh()`                  | 不涉及 gather/scatter | **?**               |
| TSDF `extract_occupied_voxels()`→cubes | 不涉及 gather/scatter | **?**               |
| 2 mm ESDF→`field<=0` cubes             | scatter               | **−0.030 mm**       |
| 2 mm ESDF→`field<=0` cubes             | gather                | **?**               |
| 15 mm ESDF→`field<=0` cubes            | gather                | **−6.005 mm**       |

这一次实验的信息量非常高。

### 如果得到：

\[ d_{\rm mesh}\approx +0.6\text{ mm} \]

则可以支持：

> **在这个姿态上，填充格与零等值面给出不同的几何关系，需分析体素表示的额外占据范围。**

### 如果：

> 保留三角拓扑的查询确认零等值面与机器人实体相交。

则说明：

> 当前导出坐标下的 TSDF 零等值面与 link6 相交；应继续区分导出坐标、TSDF 重建与原始深度的影响。

单独一个 BVH signed distance 返回值不足以建立闭合实体的内部/外部含义；即使表面不相交，也不能据此证明完整状态有效。

这个实验不会涉及任何 skill 参数。

------

# 8. gather/scatter 需要在同一 TSDF 上比较

我们这次正式与诊断现在不能直接比，就是因为同时变化了：

- extent；
- resolution；
- seeding method。

固定：

\[ h=2\text{ mm} \]

且 extent 完全相同，分别：

```
fine/gather
fine/scatter
```

然后比较：

1. zero cell 数量；
2. negative cell 数量；
3. `field<=0` cube 的 AABB；
4. Coal link6 distance；
5. CuRobo 原始 sphere→ESDF cost。

官方说明 gather 可能增加 seed，但不能据此推出每个姿态的 Coal 距离都会更负。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com) seed 数量、完整占据范围、具体实体距离和各球查询值应分别记录；同 TSDF 的实测结果见专项报告。

------

# 9. 不把 `field <= 0` 填充格称为独立环境真值

当前实现已将它定义为同一观测场的填充格近似。需要纠正的是对这种表示精度和独立性的解释；生产替换仍需前述架构决定。

条件符号符合当前负内正外的 ESDF 定义；不能简单改为 `field < 0`，否则会丢掉零值 seed 对应的障碍表示。它的局限是：

`compute_esdf()` 的目标是产生距离场供规划器查询。

官方教程直接称它为：

> dense Euclidean Signed Distance Field，供 robot collision spheres 做 \(O(1)\) trilinear distance query。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.examples.getting_started.volumetric_mapping.html?utm_source=chatgpt.com)

而不是：

> 对数组做 threshold，然后把每个 cell 填满成为几何真值。

因此：

```
occupied = field <= 0
```

更适合 debug 可视化/离散占据近似，而不适合成为我们所谓的“independent exact environment validator”。

------

# 10. 当前 link6 的真实状态应该重新表述

在现在的证据下，不应该再说：

> link6 与 2 mm fine map 也相交。

更准确的是：

> link6 对由 2 mm scatter ESDF 零值 cell 构成的填充格并集返回负距离；该数值不是真实桌面穿透深度。

同时：

\[ d_{\rm simulator\ table} \approx+0.642\text{ mm}. \]

TSDF 零等值面的原始导出、坐标修正及与原桌面的比较见专项报告。各表示的距离应分别保留，不能把 mesh 分离直接转为生产规划放行。

------

# 11. 另外，我们这个 0.642 mm 本身仍然值得警惕，但这是另一件事

即使最后：

\[ d_{\rm TSDFmesh}=+0.5\sim0.7\text{ mm}, \]

也只说明：

> 来源机器人实体与已导出的三角表面分离。

并不等于：

> 真机上应该允许 0.6 mm clearance。

在完整占据、包含关系、接触和未知空间合同已明确的前提下，可以讨论统一的间距要求：

\[ d_{\rm narrow} \ge m_{\rm safe}, \]

其中：

- \(d_{\rm narrow}\)：已明确几何语义的最小距离；仅有 TSDF 表面距离不能独自满足完整状态验收；
- \(m_{\rm safe}\)：统一的系统安全裕量，由 RGB-D、标定、机器人状态误差等决定。

这个 \(m_{\rm safe}\) 应该是**backend/system-level** 参数，不应该是：

```
grasp_mug = 0.4 mm
grasp_bowl = 0.9 mm
rotated mug = 0.2 mm
```

这样才不会进入我们最担心的 per-skill/per-generalization 调参地狱。

当前已有 `measurement_allowance_m` 参与球半径、图层选择及 Coal 间距检查。这里的公式不授权新增第二套 margin 参数，也不改变已有未知可尝试政策；应先明确现有余量覆盖哪些误差。

------

## 我现在对我们的问题排序已经发生了变化

之前：

1. coarse map 太粗；
2. sphere protrusion；
3. map selector。

现在增加这个证据以后，我会改成：

**P0：明确 `ESDF<=0 → full cubes` 的占据近似语义，并区分它与 TSDF 表面、完整实体及环境真值。**

**P1：正式 planning 使用 `gather`，它官方定义上就会 dilate ESDF seed band；必须与 scatter 诊断分开。**

**P2：robot sphere protrusion，已经确认在 link7 / palm 和 self-collision 上产生 false positive。**

**P3：15 mm coarse 的填充格不能提供亚毫米真实间距精度；负距离中各项表示与重建因素的贡献尚不能由半格尺度直接分解。**

**P4：最后才是这个特定 IK posture 的真实 clearance 是否过小。**

所以我现在**不会建议我们扩大 fine ROI、改 skill、增加 OMPL 时间或继续加 refine points**。

用于区分表示的关键检查是：

> **在完全相同的 Mapper TSDF snapshot 上调用公开 `extract_mesh()`，同时保留原始输出与 FUNCTO `extract_tsdf_surface(mapper)` 的坐标修正语义，让 link6 来源 collision geometry 对保留三角拓扑的 mesh 做 Coal 表面距离和相交查询。**

这项检查绕开 `gather/scatter` 的 ESDF seed 差异，但仍需检查导出坐标与重建误差，不能仅凭正距离确认完整环境无碰撞。官方 API 和源码支持提取这种观测表面。[GitHub](https://github.com/NVlabs/curobo/blob/main/curobo/_src/perception/mapper/integrator_esdf.py) 已有实测结果统一保留在专项报告，不以重复测试替代架构决定。

