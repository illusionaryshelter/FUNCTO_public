我们现在这个新证据把问题又缩小了一层，而且方向非常明确：

> **当前 `Coal + 占据体素` 这条所谓“实体检查”并不是一个真正独立于 ESDF 表示的环境几何检查。机器人几何换成了来源实体，但环境几何仍是由 ESDF 的零/负值格经过“整格立方体化”得到的第二层代理。**

所以 `−0.03042025 mm` **不能解释成“2 mm fine map 也确认真实碰撞”**。更准确的解释是：

> 在 `scatter` 生成的 ESDF 零值 seed 上，把每个 2 mm cell 变成完整立方体后，link6 与这个“加厚后的离散表面”发生了 0.03 mm 的相交。

这和真实桌面仍有约 \(+0.642\) mm 间隙完全可以同时成立。

而且我们的数字现在非常漂亮地支持这个解释。

------

## 1. 先看一个很有说服力的数量级关系

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

这不是证明，但已经是非常强的诊断证据：

> **15 mm → 2 mm 后，误差从约 6.65 mm 缩小到约 0.67 mm，几乎按 voxel 几何厚度同步缩小。**

所以我现在不会再把 link6 的主要问题描述成“ESDF 本身错误”。

更准确的是：

> **ESDF → `field<=0` → full voxel cubes 这个转换把一个距离场的零等值面变成了有有限厚度的实体层。**

------

# 2. 而 `gather` 又会在这之前进一步加厚 surface seed

这次我们特别指出：

> 正式规划：`gather`
> 扩大 fine 诊断：`scatter`

这个区别非常重要。

cuRoboV2 官方现在已经把两者语义写得非常清楚：

### `scatter`

一个 TSDF surface voxel 只映射到包含其中心的那个 ESDF cell，因此形成：

> exactly one ESDF voxel thick

并且官方直接说：

> **scatter faithfully matches the TSDF surface geometry**

### `gather`

每个 ESDF cell 不仅检查中心，还检查 6 个 face-center，因此可能捕获邻居 cell 中的 surface voxel，形成：

> **dilated seed band (~1.5 voxels thick)**

官方明确说，这是为了得到更高 collision recall，但代价是**不再精确匹配 TSDF surface**。[nvlabs.github.io](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

这几乎就是我们现在现象的官方解释。

所以我们正式路径的几何链实际上是：

```
RGB-D
 ↓
TSDF surface
 ↓
gather
 ↓
约 1.5 voxel 厚的 ESDF seed band
 ↓
field <= 0
 ↓
每个 seed 又变成完整 voxel cube
 ↓
Coal collision geometry
```

这里实际上发生了**两次空间加厚**。

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

也就是说 Coal 看到的根本不是“障碍物内部 voxel volume”。

它看到的是：

> **ESDF surface seed cells 的立方体并集。**

示意：

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

> robot source geometry 与这个 **voxelized shell** 相交。

不能推出：

\[ d_{\rm TSDF\ surface}<0. \]

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

它**不经过 ESDF gather/scatter seeding**。[GitHub](https://github.com/NVlabs/curobo/blob/main/curobo/_src/perception/mapper/integrator_esdf.py)

官方 volumetric mapping 文档也把两件事明确分开：

- TSDF → mesh
- TSDF → ESDF → \(O(1)\) trilinear collision distance query

也就是说官方架构本身没有把“ESDF 零格”当 mesh 使用。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.examples.getting_started.volumetric_mapping.html?utm_source=chatgpt.com)

这非常适合我们的 Coal 独立验证。

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

所以它比我们现在的 ESDF threshold 方法语义更正确，但如果目标是做高精度 narrow-phase distance，**mesh 更合适。**

------

# 5. 因此我现在建议把架构明确成“三种几何表示，三种职责”

这不是给 skill 加东西，而是修 collision backend。

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

当 proxy 判断危险时：

```
sphere / ESDF says collision
          ↓
source robot collision geometry
          ×
Mapper.extract_mesh()
          ↓
Coal narrow phase
```

用于：

- OMPL state validity 的最终裁决；
- TrajOpt candidate 最终裁决；
- dense trajectory validation；
- proxy false-positive 消歧。

Coal 本身就是为 GJK/EPA collision 和 distance narrow phase 设计的，也支持安全 margin 和距离下界。[GitHub](https://github.com/coal-library/coal?utm_source=chatgpt.com)

这跟我们已经对 self-collision 做的事情是一模一样的哲学：

```
sphere provides optimization gradient
source geometry provides acceptance decision
```

只是 world side 也要从：

```
ESDF-derived cubes
```

换成：

```
TSDF zero-level mesh
```

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

例如：

```
MapSelector(q, link)
        ↓
    chooses FINE
        ↓
┌──────────────────────────┐
│ same fine mapper snapshot│
├──────────────────────────┤
│ ESDF → CuRobo gradient   │
│ TSDF mesh → Coal verdict │
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
│ TSDF mesh → Coal verdict  │
└───────────────────────────┘
```

也就是：

\[ \text{map identity一致} \]

但

\[ \text{representation可以不同}. \]

这反而更标准。

------

# 7. 我们现在最值得做的不是继续规划，而是一个非常小的表示对照实验

固定：

- 当前 RGB-D；
- 当前 TSDF；
- 同一个 link6 \(q\)；
- 同一个 13.378 s OMPL boundary；
- 同一来源机器人 collision geometry。

然后只换环境表示。

建议直接得到这张表：

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

那么问题几乎确定：

> **我们的 `field<=0 → cubes` 路径正在制造 collision geometry inflation。**

### 如果：

\[ d_{\rm mesh}<0 \]

那才说明：

> TSDF 重建出来的 zero surface 本身已经侵入 link6。

这时才继续查 RGB-D integration / TSDF。

这个实验不会涉及任何 skill 参数。

------

# 8. 特别建议再做 gather/scatter 的 apples-to-apples 比较

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

官方已经告诉我们预期：

\[ \text{gather seed band} > \text{scatter seed band} \]

因为 gather 本来就会 over-seed。[NV Labs](https://nvlabs.github.io/curobo/latest/api/curobo.perception.html?utm_source=chatgpt.com)

如果实测：

```
scatter: −0.03 mm
gather : −1.x mm
```

就不会再奇怪。

------

# 9. 我会停止使用 `field <= 0` 作为“实体 occupied geometry”的定义

这可能是当前最值得直接修改的一项。

不是因为条件符号一定错——我们已经验证了具体数据——而是因为**语义不对**：

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

> link6 与由 2 mm scatter ESDF 的零值 cell 转换得到的填充 voxel shell 相交 0.0304 mm。

同时：

\[ d_{\rm simulator\ table} \approx+0.642\text{ mm}. \]

而 TSDF zero-level mesh 距离目前：

\[ d_{\rm TSDFmesh}=?. \]

**现在真正缺的就是这个问号。**

它会决定接下来走哪条路。

------

# 11. 另外，我们这个 0.642 mm 本身仍然值得警惕，但这是另一件事

即使最后：

\[ d_{\rm TSDFmesh}=+0.5\sim0.7\text{ mm}, \]

也只说明：

> 这个 configuration 几何上没有碰撞。

并不等于：

> 真机上应该允许 0.6 mm clearance。

最终硬判据应该是：

\[ d_{\rm narrow} \ge m_{\rm safe}, \]

其中：

- \(d_{\rm narrow}\)：source robot geometry 对 TSDF mesh 的最小距离；
- \(m_{\rm safe}\)：统一的系统安全裕量，由 RGB-D、标定、机器人状态误差等决定。

这个 \(m_{\rm safe}\) 应该是**backend/system-level** 参数，不应该是：

```
grasp_mug = 0.4 mm
grasp_bowl = 0.9 mm
rotated mug = 0.2 mm
```

这样才不会进入我们最担心的 per-skill/per-generalization 调参地狱。

------

## 我现在对我们的问题排序已经发生了变化

之前：

1. coarse map 太粗；
2. sphere protrusion；
3. map selector。

现在增加这个证据以后，我会改成：

**P0：`ESDF<=0 → full cubes` 作为 Coal world geometry 的语义错误/过保守转换。**

**P1：正式 planning 使用 `gather`，它官方定义上就会 dilate ESDF seed band；必须与 scatter 诊断分开。**

**P2：robot sphere protrusion，已经确认在 link7 / palm 和 self-collision 上产生 false positive。**

**P3：15 mm coarse 仍然不适合裁决亚毫米 clearance，但其 −6 mm 的很大一部分现在很可能可以由 full-voxel shell inflation 解释。**

**P4：最后才是这个特定 IK posture 的真实 clearance 是否过小。**

所以我现在**不会建议我们扩大 fine ROI、改 skill、增加 OMPL 时间或继续加 refine points**。

下一步最有决定性的结果只有一个：

> **在完全相同的 Mapper TSDF snapshot 上调用 `extract_mesh(level=0)`，让 link6 来源 collision geometry 对这个 mesh 做 Coal distance。**

如果这个值接近 simulator 的 \(+0.642\) mm，我们基本就找到了当前环境误拒链条里最核心的表示错误。官方 API 和源码都支持这种检查，而且它天然绕开 `gather/scatter` 的 ESDF seed 差异。[GitHub](https://github.com/NVlabs/curobo/blob/main/curobo/_src/perception/mapper/integrator_esdf.py)

