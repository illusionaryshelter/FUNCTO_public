> 本文保留用户提供的初始研究思路，不是当前执行规范。原 PoC 已经由独立 `transfer` 模块替换并删除，不保留旧入口。已实现的数学合同、依赖、命令与实测边界见 [模块说明](README.md) 和 [独立实验计划](../../sim-version-plans/2026-10-13-local-region-transfer/plan.md)；下文原脚本运行示例及未验证的性能建议不能当作当前结果。

在你现在明确的条件下，我建议不再设计新的神经描述子，也不做显式结构分解，而是直接用 MARCO + 三维几何关系构造一个 training-free 的交互区域迁移系统。

而且我基本认同你的两个假设，但建议把它们变成可以实验检验的边界条件：

1. Shape Similarity Prior：源物体和目标物体的可观测几何具有足够相似性，允许建立一批可靠的跨实例对应点。
2. Interaction Relation Consistency Prior：在对应的功能状态下，交互区域相对于物体整体几何的关系基本一致。

这两个假设不需要保证所有物体都能迁移。我们的目标应该是：满足这些条件时尽可能准确迁移，不满足时能够检测不确定性并拒绝输出。

## 一、先明确 MARCO 能做什么、缺什么

我查看了你使用的 [MARCO 官方代码](https://github.com/visinf/MARCO)，尤其是 [`models/marco.py`](https://github.com/visinf/MARCO/blob/main/models/marco.py) 和 [`util/visualization.py`](https://github.com/visinf/MARCO/blob/main/util/visualization.py)。

它已经提供了你需要的一大部分基础能力：

- 利用 DINOv2 backbone、adapter 和特征上采样产生密集二维视觉特征。
- 给定源图像关键点，在目标图像中寻找跨实例语义对应。
- 可以直接调用 `forward_backbone()` 提取特征图，或者使用 `sample_descriptors()` 获取关键点特征。

MARCO 的原始公开实现还是二维图像对应方法，而不是已经具备 SE(3) 不变性的三维描述子。你通过 RGB-D 把特征提升到三维，是一个合理的扩展方向。

真正缺少的不是视觉特征提取，而是下面这一步：

\[ \underbrace{\text{MARCO Feature}}_{\text{建立物体表面对应}} \quad+\quad \underbrace{\text{3D Relational Geometry}}_{\text{保存交互区域相对关系}} \quad\longrightarrow\quad \text{Region Transfer} \]

这里我建议不要把区域本身当成必须匹配的视觉对象。

区域可以没有视觉特征，但物体上其他可匹配的点，可以成为定位这个区域的参照。

这件事对 Shelf、Drawer、Cabinet 都成立。

## 二、三个场景首先要统一成“可观测功能状态”

- Shelf — 放置表面直接可见：直接观察承载面，获得目标物体点云，执行区域迁移。
- Cabinet — 门打开后内部可见：源、目标均在门打开且目标交互区域可观测的状态下进行迁移。不必识别柜门的铰链或建立运动学模型。
- Drawer — 抽屉拉出后内部可见：在抽屉已经拉出、内部底面能被 RGB-D 看到时迁移放置区域。拉出程度不同会带来遮挡和可见几何变化，首先使用相近的拉出状态。

这里有个必须划清的边界：如果只有关着门的 Cabinet 或未拉开的 Drawer 的单视角点云，内部区域完全不可见，那么我们不可能仅靠这些输入，唯一确定内部任意局部区域的位置。

因此，首轮验证不解决开门、拉抽屉的控制问题。把它们作为交互区域迁移的前置条件：

\[ \text{Observe Accessible State} \rightarrow \text{Transfer Region} \]

未来要形成完整操作流程，可以再接上：

\[ \text{Open / Pull} \rightarrow \text{Observe} \rightarrow \text{Transfer} \rightarrow \text{Place} \]

这样不会牺牲未来的通用性，也不会把整个机械操作链条强行压进区域描述子。

## 三、核心方案：用 Anchor Distance Signature 表示整个物体与交互区域的关系

我建议先验证一种非常简单、具有明确数学不变性的描述子，而不是直接上神经场。

设源物体点云为 \(P_s\)，已知交互区域点云为 \(\Omega_s\)，目标物体点云为 \(P_t\)。

### 1. 利用 MARCO 自动找到跨实例 Anchor

在源物体的整个可见点云上，用 Farthest Point Sampling（FPS）均匀采样约 300–500 个点，通过 MARCO 特征在目标点云中寻找对应。

注意不要只在交互区域附近采样。我们需要一些分布在整个物体上的点，否则无法区分 Shelf 上重复的层板。

推荐的匹配流程：

Source / Target RGB-D + SAM

获取可见点云与对应的 MARCO 特征

MARCO Feature Correspondences

特征最近邻、双向匹配、空间均匀采样

Geometric Consistency Filtering

Sim(3) RANSAC 剔除明显错误对应

K 对可靠的 3D Anchors

建议保留 16–32 对，要求空间分布尽量广

为什么使用 Sim(3) 而不是 SE(3)？因为不同实例可能存在整体尺度变化。Sim(3) 提供旋转、平移和统一缩放，用作鲁棒匹配的一致性检验，而不是假设所有 Shelf 都有完全相同的形状。

初始 RANSAC 内点阈值可以设为目标物体点云 RMS 半径的 5%–10%，再观察匹配是否稳定。

### 2. 用归一化距离构造 Interaction Descriptor

设源交互区域的中心为 \(\mathbf c_s\)，源物体上筛选出的锚点为 \(\mathbf a_i^s\)，共有 \(K\) 个锚点。

定义：

\[ L_s=\operatorname{median}_{i<j} \|\mathbf a_i^s-\mathbf a_j^s\|_2 \]

其中 \(L_s\) 是锚点间距离的中位数，代表一个由几何参照点确定的尺度。

构造源交互区域描述子：

\[ \boxed{ D_s(\mathbf c_s)= \left[ \frac{\|\mathbf c_s-\mathbf a_1^s\|_2}{L_s}, \ldots, \frac{\|\mathbf c_s-\mathbf a_K^s\|_2}{L_s} \right] } \]

这个 \(K\) 维向量有几个重要性质。

第一，它是显式尺度不变与 SE(3) 不变的。 只要锚点和交互区域一起经历相同的整体相似变换，描述子不变。

第二，它不依赖局部区域的纹理。 即使 Shelf 的放置区域是普通平板，也能用远处框架、边缘、其他层板的对应点对它进行定位。

第三，它不需要理解什么是 Shelf、Drawer 或 Cabinet。 只依赖现成的跨实例对应和整体相对几何。

在目标物体上，对任意候选表面点 \(\mathbf q\) 构造：

\[ D_t(\mathbf q)= \left[ \frac{\|\mathbf q-\mathbf a_1^t\|_2}{L_t}, \ldots, \frac{\|\mathbf q-\mathbf a_K^t\|_2}{L_t} \right] \]

其中 \(\mathbf a_i^t\) 是第 \(i\) 个目标锚点，\(L_t\) 是目标锚点间距离的中位数。

然后求解：

\[ \boxed{ \mathbf c_t^*= \arg\min_{\mathbf q\in P_t} \operatorname{TrimMean} \left( |D_t(\mathbf q)-D_s(\mathbf c_s)| \right) } \]

这里 \(\operatorname{TrimMean}\) 表示去掉少量最大的距离差后取平均，减少错误锚点和形状变化的影响。

直观地说：在目标物体表面寻找一个点，使得它到物体各个对应 Anchor 的相对距离，尽量接近源交互区域。

这就是最小版本的 Object-relative Interaction Descriptor。

### 3. 再加入两个很轻量的约束

仅依靠距离描述子，会有歧义。因此建议增加：

\[ E(\mathbf q)= E_{\mathrm{relation}}(\mathbf q) +\lambda E_{\mathrm{align}}(\mathbf q) +\mu E_{\mathrm{surface}}(\mathbf q) \]

其中：

- \(E_{\mathrm{relation}}\)：上述归一化距离匹配误差。
- \(E_{\mathrm{align}}\)：候选点到全局 Sim(3) 迁移预测位置的归一化距离，防止落到远处重复结构。
- \(E_{\mathrm{surface}}\)：候选点附近的局部接触平面与源区域变换后平面的法向差异或非平面程度。
- \(\lambda,\mu\)：两个约束的权重。

这两个约束只利用几何，不需要部件识别。

如果目标是 Drawer 内底板，那么候选应该具有与源接触面一致的局部几何；如果候选落到了侧壁，就应受到惩罚。

需要强调，Sim(3) 全局对齐只是弱先验。存在不同宽深比时，关系描述子可以比单纯 Sim(3) 更灵活，但其严格不变性不覆盖任意非均匀形变。

## 四、从区域中心恢复整片放置区域

中心定位以后，需要迁移整个源区域点云，而不是只输出一个点。

最简单的方法是先估计源区域局部平面，再利用全局配准的旋转和尺度，将区域相对于中心的点集迁移：

\[ \hat{\mathbf p}_t= \mathbf c_t^*+ sR_{\mathrm{local}} (\mathbf p_s-\mathbf c_s) \]

其中：

- \(\mathbf p_s\) 是源交互区域中的点。
- \(s\) 是跨实例尺度估计。
- \(R_{\mathrm{local}}\) 是根据全局旋转和目标局部接触平面法向修正得到的旋转矩阵。
- \(\hat{\mathbf p}_t\) 是迁移后的预测区域点。

之后将预测点投影到目标附近估计出的局部支撑平面上，再依据区域投影轮廓，从目标可见点云中选择对应点。

第一版可以直接使用区域的二维凸包。这样无需重建 mesh；对于凹形或多连通区域，再考虑保留原始轮廓。

如果区域中心能匹配，但新 Shelf 宽深比变化使得整个区域的形状迁移不理想，第二版可以加入局部仿射变换，用附近锚点的对应估计局部拉伸。它仍然是最小二乘问题，不需要训练新模型。

## 五、可直接运行的验证脚本

我按照上面的方法写了一份 NumPy + SciPy 的参考实现，直接接收你已经得到的 MARCO 点特征，不需要重新实现 MARCO，也不需要安装新的深度学习模型。

interaction_transfer_poc.py

包含特征对应、Sim(3) RANSAC、关系描述子、区域恢复和评测

下载脚本

输入 `pair.npz`：

| 字段              | 内容                                       |
| ----------------- | ------------------------------------------ |
| `src_xyz`         | 源整体点云，\(N\times3\)                   |
| `tgt_xyz`         | 目标整体点云，\(M\times3\)                 |
| `src_feat`        | 源点逐点对应的 MARCO 特征，\(N\times C\)   |
| `tgt_feat`        | 目标点逐点对应的 MARCO 特征，\(M\times C\) |
| `src_region_mask` | 源交互区域布尔 Mask，长度 \(N\)            |
| `tgt_gt_mask`     | 可选：目标区域人工真值，仅用于评测         |

这里 \(N,M\) 是两片点云的点数，\(C\) 是 MARCO 特征维度。

点云与特征必须逐点对齐。实际使用中，要保留 RGB-D 像素到点云的索引映射，并正确处理 MARCO 输入图像的 resize 和 padding。

运行：

```
pip install numpy scipy

python interaction_transfer_poc.py pair.npz \
    --out prediction.npz
```

输出包含 `tgt_region_mask`、`tgt_region_points`、`tgt_center`、对应锚点索引和评测统计。

我用合成的多层 Shelf 点云检查过代码，包括三维旋转、统一缩放和非均匀尺寸变化，能完成区域迁移。这只是程序正确性的初步测试，不代表真实 MARCO 特征、真实 RGB-D 遮挡条件下的成功率。 脚本中的置信度阈值也是实验初值，需要根据真实数据校准。

## 六、真正的可行性验证应该怎么安排？

我建议用一个很小的实验，先让 Shelf 成功，再验证 Drawer 和 Cabinet。

## 1

同一个 Shelf，不同位姿

先对同一个源点云施加已知三维变换和缩放，检查实现的数学一致性；然后实际重新拍摄同一个 Shelf 的另一个视角，检查单视角遮挡和 MARCO 对应的影响。

这两个测试不能混在一起：前者只验证几何不变性，后者才涉及真实观测变化。

## 2

不同 Shelf，相同结构布局

选约 3 个不同尺寸的 Shelf，保持层数和整体布局近似一致。在一个 Shelf 上示教不同层板、不同相对位置的局部放置区域，迁移到其他 Shelf。

重点看是否落到正确层板，以及在层板上是否保持相似的相对位置。这是验证你的结构关系假设最关键的一步。

## 3

Open Drawer / Open Cabinet

用约 2–3 组同类型但不同尺寸的 Drawer 或 Cabinet，拍摄打开、内部接触面可见的 RGB-D。

首先保持开合状态相近，验证能否恢复底板或内部层板上的局部区域。之后再增加不同拉出程度、门角度和遮挡程度。

评测中，目标区域的人工标注只作为 Ground Truth，不能参与算法推理。

### 必须做的 Baseline 对比

| 方法                                   | 作用                                               |
| -------------------------------------- | -------------------------------------------------- |
| MARCO Direct Region Matching           | 直接匹配源区域点，观察纯视觉特征的局限             |
| MARCO + Global Sim(3)                  | 验证仅靠整体形状配准是否足够                       |
| MARCO + Relational Descriptor          | 验证相对整体几何关系是否真正提升迁移               |
| Oracle Anchors + Relational Descriptor | 仅用于诊断：假如锚点对应正确，关系表示本身能否成功 |

Oracle 对应只需要在少量测试样本中标记几个对应点，不参与最终算法。脚本支持 `--oracle` 测试模式。

建议第一轮记录三个核心指标：正确承载区域命中率、归一化中心位置误差、目标区域点集 IoU。另外记录匹配内点比例和拒绝输出率。

可以先将 80% 正确承载区域命中率、中心误差小于目标物体 RMS 半径的 5%、区域 IoU 大于 0.5 作为探索性目标，而非已验证的性能标准。

## 七、实验结果如何决定下一步？

这是整个方案里我最关注的判断。

| 实验现象                                    | 说明                               | 下一步                               |
| ------------------------------------------- | ---------------------------------- | ------------------------------------ |
| Global Sim(3) 已经足够好                    | 结构相似假设很强                   | 直接使用简单配准，不必增加描述子     |
| Sim(3) 不行，Relational Descriptor 明显更好 | 全局形状变化下关系表征有价值       | 保留 Anchor 关系表示                 |
| Oracle 好，Automatic 差                     | 主要问题是 MARCO 对应点不可靠      | 改进匹配筛选和空间一致性             |
| Oracle 也差                                 | 锚点距离关系或结构一致性假设不足   | 考虑局部仿射、分段关系或收窄适用场景 |
| Shelf 好，Drawer 差                         | 可能受关节状态、遮挡、刚体混合影响 | 检查功能状态与有效可见点云           |

还有一点不能忽视：置信度不能保证检测出所有错误。 如果新 Shelf 存在两个几何上完全对称的区域，错误对应可能具有很高的一致性分数；如果关闭的柜门外表面恰好形成相似几何，也可能产生虚假的候选。这种情况下需要遵守可观测功能状态约束，或允许输出多个位置，而不是宣称仅凭点云必然可以判别。

## 八、我的最终建议

我觉得第一阶段只需要实现三个模块：

1. MARCO-guided 3D Anchors

寻找可靠的跨实例三维对应点

2. Scale / SE(3)-Invariant Relational Signature

编码交互区域相对于整物体锚点的归一化距离

3. Surface-constrained Region Recovery

在目标可见表面恢复交互区域，而不是只匹配一个局部特征点

不做显式层板识别、不做 CAD 重建、不做关节估计、不做新的 Learning。这些都不是当前证明可行性所必需的。

如果第一轮实验表明 MARCO + Sim(3) + Anchor Relational Signature 已经能稳定迁移 Shelf 的局部放置区域，那就说明你提出的核心假设值得继续研究。

真正值得进一步提炼成研究贡献的也不是一个新的点云 backbone，而是：

> 在已知跨实例形状相似、交互结构关系一致的条件下，利用少量自动建立的三维对应点，将没有显著视觉语义的局部交互区域表示为物体整体几何的相对关系，并在新的部分观测物体上恢复该区域。

这条路线既保留 model-free 的优势，也把难度控制在可以快速验证的范围内。