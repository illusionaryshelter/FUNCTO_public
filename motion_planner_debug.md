我们现在这个新证据把问题又缩小了一层，而且方向非常明确：

> **此次诊断中的旧 `Coal + 占据体素` 路径并不是一个真正独立于 ESDF 表示的环境几何检查。机器人几何换成了来源实体，但当时的环境几何仍是由 ESDF 的零/负值格经过“整格立方体化”得到的第二层代理。**

所以 `−0.03042025 mm` **不能解释成“2 mm fine map 也确认真实碰撞”**。更准确的解释是：

> 在 `scatter` 生成的 ESDF 零值 seed 上，把每个 2 mm cell 变成完整立方体后，Coal 返回 link6 与这些立方体的负距离。这个返回值不是退出整个体素并集所需的全局最小平移。

这和真实桌面仍有约 \(+0.642\) mm 间隙完全可以同时成立。

这些数字与体素表示造成额外占据厚度的假设相容，尚不能分离各项误差来源。前四节保留这次表示对照的分析；它们描述的 ESDF 填充格路径已从生产验收中移除。已批准的 TSDF 零集合与同源负区重建，以及规划和执行的来源几何验收，均已接入当前代码，具体职责见第 5、6 节。表示对照、导出坐标检查和迁移后的最终验证结果统一记录在[专项报告](../sim-version-plans/2026-10-04-tsdf-surface-review.md)；代码接入本身不代表验证已通过。

------

## 1. 先看体素表示的数量级关系

此次诊断所针对的旧 Coal 环境代理实际上相当于：

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

所以此次诊断所针对的旧正式路径，其环境几何链实际上是：

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

因为这条旧 Coal 路径的环境仍然源自同一个已经经过 gather 的 ESDF。当前生产路径已按第 5 节改为从观测 TSDF 样本重建几何；上述循环依赖分析解释的是被移除路径的问题。

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

另有一项已确认的本地导出坐标差异：`builder_coord.voxel_to_world` 将 TSDF 样本放在 `index+0.5`，而 `builder_mesh.generate_mesh_kernel` 将相同样本值放在 `index`。在 `refine_iterations=0` 时，每轴补回半个 TSDF 格距可恢复样本中心坐标。FUNCTO 的 [`extract_tsdf_surface(mapper)`](manipulation/mapper_surface.py) 固定使用零 refinement，返回修正后的 CPU 顶点和原三角索引；它不改第三方库、不补洞、不做凸包，也不增加碰撞余量。该函数保留为**诊断用的公开 marching-cubes 导出坐标修正**。当前生产验收使用第 5 节的 `export_blocks()` 样本重建；不会把这里补回半格的三角表面直接作为唯一验收判据。

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

# 5. 已批准并接入的架构：“三种几何表示，三种职责”

这次修改统一了碰撞环境表示与验收职责，不涉及 skill 参数。当前代码由 sphere/ESDF 提供优化代价与梯度，由机器人来源实体及同一观测 TSDF 重建的完整有限分量作碰撞和接触验收。验收必须查询所需来源几何对，不以代理球代价是否为零决定是否查询，也不再让 sphere/ESDF 代价单独否决已通过来源几何检查的状态。

现有“未知可尝试”（`allow_attempt`）政策保持：不可见区域缺少几何证据，既不被宣称已证自由，也不因本次重建而一律禁止尝试。已观测的负 TSDF 分量仍属于碰撞证据；缺少必需来源几何适配、查询未完成等程序性缺口仍显式拒绝，不能冒充不可见区域。地图层选择的有效支撑要求与未知空间政策继续分别处理。

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
- 优化查询自身的空间筛选；它不决定来源几何验收是否执行。

`gather` 为了提高 recall 而故意保守，这是官方设计，不必因为现在的问题就全面改成 scatter。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

------

## 第二层：来源机器人实体 + 同源 TSDF 零集合和负区 + Coal

生产几何由 [`planning_observation.py`](manipulation/planning_observation.py) 绑定到每个地图快照；[`tsdf_geometry.build_tsdf_geometry()`](manipulation/tsdf_geometry.py) 调用 `mapper.tsdf.export_blocks()`，读取观测块中的加权 TSDF 累积值与权重，并按样本中心坐标还原有效样本。它不消费 ESDF 的正负号，也不调用公开 marching-cubes 表面导出作为验收几何。

```
同一 Mapper 观测快照
          ↓
export_blocks() → 有效 TSDF 样本值、权重和样本中心坐标
          ↓
相邻样本中心构成的单元，按固定六四面体划分
          ↓
每个完整已观测四面体的仿射 TSDF
          ↓
零集合 + 同一仿射函数的负区有限分量
          ×
机器人来源碰撞实体 / 已绑定物体来源实体
          ↓
Coal 距离、相交及完整接触区域检查
```

这里“仿射 TSDF”指一个四面体内由其四个顶点样本确定的一次函数，包含常数项；“零集合”是该函数等于零的点集，“负区”是同一个函数为负的部分及其零边界。固定六四面体划分保证相邻单元共享面的划分一致；每个四面体只有在四个样本均被观测到时才重建。八个角都严格为负的完整单元可以压缩保存为整数索引，其几何仍只占相邻样本中心之间的范围。孤立的非正样本保留为单点，不扩成半格邻域；四个样本全零的四面体保留整个零值区域。未观测空间不补洞、不外推为负区，也不通过整个场景的凸包填满。

[`ObservedTsdfGeometry`](manipulation/tsdf_geometry.py) 保留这些有限分量的并集。每个分量本身是点、线段、零截面多边形或凸负区，Coal 按其实际维数进行查询；完整并集可以非凸，也不要求形成封闭的障碍外表面。`robot_geometry_check._scene_shape()` 中原有 `kind='mesh'` 仍服务于明确的来源凸实体，观测 TSDF 使用独立的 `kind='tsdf'` 记录，不把开放三角网格整体凸包化。负区检查能发现实体落在已观测负区内部的情形，所以只有开放表面距离为正不再足以放行。

有限接触由 [`contact_geometry.py`](manipulation/contact_geometry.py) 和 [`RobotGeometryChecker.query_scene()`](manipulation/robot_geometry_check.py) 按同一目标几何检查。“允许接触”要求来源实体与目标分量的**完整交集**处于声明的材料面、目标接触片及间隙约束内，不能用少数见证点代替全交集。“必须接触”要求目标已观测零集合或显式来源实体表面与该有限区域相交，并满足原有方向等约束；负区在未知边界形成的截断面不能建立接触。测量余量继续沿用已有合同，不通过本次表示迁移增加另一套安全余量。

[`PlanningScene.bind_source_scene()`](manipulation/planning_scene.py) 为每次规划绑定一次来源几何，`source_components()` / `evaluate_source()` 保留任务约束和未知空间政策，并调用完整来源检查。以下四个入口已使用该路径：

- 起始状态检查：`planning.py` 的 `initial_state_diagnostics()`。
- OMPL 状态有效性检查：`ompl_seed()` 内的状态与边检查。
- 优化候选验收：`_process_manager_type()` 生成的轨迹指标处理器，供 `select_candidate()` 使用。
- 导出稠密轨迹验收：`export()` 对实际导出轨迹及约束切换状态进行检查。

[`RobotStateValidator.check_planning_world()`](manipulation/robot_validation.py) 将规划状态交给同一 `RobotGeometryChecker.query_scene()` 几何合同。执行前检查和执行时实际状态检查消费相同的序列化 TSDF 分量、来源实体和接触规则；规划工作区与执行工作区分开，规划候选不会重绑定执行状态。这里统一的是几何与权限语义，仍不能声称离散检查之间已获得连续无碰撞证明。

因此职责保持为：

```
sphere / ESDF → 优化代价与梯度
来源实体 × TSDF 零集合及同源负区 → 完整状态的碰撞与接触验收
```

这次迁移保留了原分析中“单独用零表面替换占据体积会丢失包含关系”的结论，并通过**同一仿射重建的负区**补足已观测内部的几何语义。公开 `extract_mesh()` 与坐标修正后的 `extract_tsdf_surface()` 继续用于表示对照；它们不承担生产环境的唯一验收职责。当前接入的验证结果与尚存限制见[专项报告](../sim-version-plans/2026-10-04-tsdf-surface-review.md)。

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

当前实现保留以下同层关系：PlanningScene 用既定规则为每个实体选择地图层，来源验收消费同一份层选择记录和观测快照；执行检查也验证该选择记录。图中的 Coal 项包含同源零集合与负区，不只是表面距离：

```
MapSelector(q, link)
        ↓
    chooses FINE
        ↓
┌─────────────────────────────┐
│ same fine mapper snapshot   │
├─────────────────────────────┤
│ ESDF → CuRobo gradient      │
│ TSDF zero + negative → Coal │
└─────────────────────────────┘
```

或者：

```
MapSelector(q, link)
        ↓
    chooses COARSE
        ↓
┌─────────────────────────────┐
│ same coarse mapper snapshot │
├─────────────────────────────┤
│ ESDF → CuRobo gradient      │
│ TSDF zero + negative → Coal │
└─────────────────────────────┘
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

该表示属于同一观测场的填充格近似，不是独立环境真值。当前生产代码已移除由 `ESDF<=0` 构造完整格子的验收分支；本节保留其几何局限，解释为什么不能仅改阈值符号来完成迁移。

旧条件符号符合负内正外的 ESDF 定义；如果只改为 `field < 0`，就会丢掉零值 seed 对应的障碍表示。旧填充格路径的局限是：

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

TSDF 零等值面的原始导出、坐标修正及与原桌面的比较见专项报告。各表示的距离分别保留；生产规划按第 5 节同时检查来源实体、同源零集合和负区及接触权限，不把诊断 mesh 的分离直接转为放行。

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

**P0：明确 `ESDF<=0 → full cubes` 的占据近似语义，并区分它与 TSDF 表面、完整实体及环境真值；生产验收现已移除这条填充格分支。**

**P1：正式 planning 使用 `gather`，它官方定义上就会 dilate ESDF seed band；必须与 scatter 诊断分开。**

**P2：robot sphere protrusion，已经确认在 link7 / palm 和 self-collision 上产生 false positive。**

**P3：15 mm coarse 的填充格不能提供亚毫米真实间距精度；负距离中各项表示与重建因素的贡献尚不能由半格尺度直接分解。**

**P4：最后才是这个特定 IK posture 的真实 clearance 是否过小。**

所以我现在**不会建议我们扩大 fine ROI、改 skill、增加 OMPL 时间或继续加 refine points**。

原先用于区分表示的关键诊断是：

> **在完全相同的 Mapper TSDF snapshot 上调用公开 `extract_mesh()`，同时保留原始输出与 FUNCTO `extract_tsdf_surface(mapper)` 的坐标修正语义，让 link6 来源 collision geometry 对保留三角拓扑的 mesh 做 Coal 表面距离和相交查询。**

这项诊断绕开 `gather/scatter` 的 ESDF seed 差异，但仍需检查导出坐标与重建误差，不能仅凭正距离确认完整环境无碰撞。官方 API 和源码支持提取这种观测表面。[GitHub](https://github.com/NVlabs/curobo/blob/main/curobo/_src/perception/mapper/integrator_esdf.py) 它的实测结果与当前 `export_blocks()` 同源零集合和负区路径的最终验证结果统一保留在[专项报告](../sim-version-plans/2026-10-04-tsdf-surface-review.md)。架构与代码已按第 5 节接入，不以重复诊断或坐标修正表面单独代替生产验收。

