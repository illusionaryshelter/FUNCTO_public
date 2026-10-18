# GraspGen-X 最终集成方案：Demonstration-Conditioned Grasp Planner

GraspGen-X如何做到：指定允许产生抓取的目标区域；指定希望夹爪从哪个方向接近；设置允许的接近角度偏差；接收演示里迁移过来的交互位姿先验？

我想要的效果是：演示迁移得到的交互位姿是很好的初值，但是很多时候因为环境约束必须更改抓取的方式)

以下可以作为参考。

## 1. 核心结论

针对你的目标，我建议采用：

GraspGen-X + 演示先验局部采样 + Discriminator 统一评分 + 硬约束过滤 + 操作轨迹可行性检查。

不建议只采用“生成后按演示位姿过滤与重排序”。 那个方案虽然容易实现，但有一个严重缺陷：一旦演示位姿附近没有模型生成的候选，就可能找不到合理的抓取；反过来，如果只在演示位姿附近寻找，又会漏掉环境限制下必须采取的其他抓取方式。

你想要的是：演示位姿很好时尽量保留它；有障碍、不可达或不满足操作约束时，允许系统主动选择另一种抓取姿态。

最适合的是三路候选生成、统一评估，而不是一个严格的姿态邻域。

SAM 3.1 + 双 RGB-D + POIL / OAReg

完整目标点云、允许抓取区域、Target Functional Frame、演示位姿先验

Diffusion

学习型全局候选

OBB

几何候选

Demo Prior

先验周围的局部候选

GraspGen-X Discriminator

三路候选统一抓取质量评分

Task-Constrained Feasibility Filter

指定区域、接近方向、角度阈值、碰撞、IK、操作可行性

Demonstration-Aware Ranking

在可行解中尽量保留演示抓取意图，允许必要的姿态偏离

Final Grasp + Pre-grasp + Skill Execution Plan

### 我找到了实现它的关键源码接口

这次进一步查到了一个非常有用的现成函数：

[`graspgenx/samplers/graspmoe.py`](https://github.com/NVlabs/GraspGenX/blob/main/graspgenx/samplers/graspmoe.py) 中已经存在：

```
_score_grasps_with_discriminator(    grasps_world,    pc_centered,    pc_center,    grasp_sampler,    object_embedding=None,)
```

它能够对外部生成的任意一组 6D 抓取候选进行评分。 也就是说，你不需要重新训练网络，也不需要修改扩散模型的去噪过程，就可以把演示位姿及其局部扰动作为第三路候选送进 GraspGen-X 判别器。

这是官方 GraspMoE 自己用于 OBB 候选评分的实现。函数目前以下划线开头，属于内部接口，因此工程上建议将它封装为自己的稳定适配层，避免以后升级 GraspGen-X 时受影响。

![img](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com&sz=32)

GitHub

+1



这基本解决了你的核心诉求：演示位姿能够实际参与候选生成和模型质量评估，而不是只能在最后被动重排序。

## 2. 四个需求具体怎样实现

| 你的需求               | 最终实现方式                                                 | 约束性质      |
| ---------------------- | ------------------------------------------------------------ | ------------- |
| 指定允许抓取的目标区域 | SAM 3.1 生成抓取区域 → 3D 点云 → 夹爪扫掠体积内的区域一致性检查 → 真实手指接触验证 | 硬约束        |
| 指定希望接近的方向     | 用世界坐标系中的单位向量 \(\mathbf d\) 表示                  | 软偏好        |
| 设置角度偏差           | 设置偏好角 \(\theta_{\mathrm{pref}}\) 与绝对最大角 \(\theta_{\max}\) | 偏好 + 硬上限 |
| 接受演示迁移的抓取位姿 | 原始位姿 + 多尺度局部 SE(3) 扰动 + Discriminator 评分        | 软先验        |

### 方向约束不要只有一个阈值

你的需求里存在一个看似矛盾、实际上必须支持的特性：

既希望夹爪按演示方向接近，又希望有障碍时能改变抓取方式。

所以不能简单地把所有偏离 30° 的抓取直接删掉。

我建议默认配置为：

首选接近角度范围

30°

越接近演示方向，评分越有利；不是超出即淘汰

绝对允许最大偏差

110°

超过此角度的抓取一律拒绝，不会自动放宽

当前约束

优先选择偏差约 30° 以内的方向，但在几何和动作约束允许时，可考虑偏差至 110° 的其他抓取。

 复制参数

30° / 110° 是建议的初始实验参数，不是对所有技能都安全的角度。对抽屉和柜门，还必须约束抓取后的施力方向与运动轴。

如果你明确要求“接近方向最多只能差 30°”，那么应该把偏好角与硬上限都设为 30°。环境有障碍，也不应偷偷违反你明确指定的硬上限，而是返回无可行抓取。

### 区域约束必须检查接触，而不是只检查抓取中心

这里我想修正一种常见实现：

```
distance(grasp.translation, region_center) < 0.03
```

这个判定不够可靠，因为 GraspGen-X 的位姿原点位于夹爪坐标系，通常不是手指与物体的实际接触位置。

更合理的是将目标表面点转换到候选夹爪坐标系，检查：

- 可抓取区域是否确实进入两指之间的闭合扫掠体积。
- 进入扫掠体积的点是否主要来自允许的区域。
- 真实的左右指垫接触点是否落在允许区域。
- 夹爪其他部位有没有与物体不允许接触的部分发生碰撞。

GraspGen-X 的 `XGripperInfo` 已包含夹爪扫掠体积、碰撞网格和有关 TCP 的变换信息，可以作为几何检查的基础。

![img](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com&sz=32)

GitHub



前两项可以做成快速候选过滤。后两项需要基于真实夹爪几何和闭合运动进行验证，不能仅依靠 GraspGen-X 的置信度。

## 3. 最终优化目标：先保证可行，再尽量接近演示

不要单纯把 GraspGen-X 分数与演示位姿距离相加，然后直接选最高分。

正确的决策分成两步。

第一步：硬约束过滤。 只有满足允许接触区域、最大接近角、碰撞、机器人可达性，以及技能执行轨迹约束的抓取，才能进入可行集合 \(\mathcal G_{\mathrm{valid}}\)。

第二步：在可行集合中排序。

\[ g^*=\arg\max_{g\in\mathcal G_{\mathrm{valid}}} \left[ w_q q(g)+w_d s_{\mathrm{demo}}(g) +w_a s_{\mathrm{approach}}(g) \right] \]

其中：

- \(g\)：候选 6D 抓取位姿。
- \(q(g)\)：GraspGen-X Discriminator 的抓取质量评分。
- \(s_{\mathrm{demo}}(g)\)：与迁移演示位姿的相似度。
- \(s_{\mathrm{approach}}(g)\)：与期望接近方向的相似度。
- \(w_q,w_d,w_a\)：三个评分项的权重，必须根据实际数据调整。

演示相似度可以采用高斯形式：

\[ s_{\mathrm{demo}}(g)= \exp\left[ -\frac12\left( \frac{\Delta p^2}{\sigma_p^2} + \frac{\Delta R^2}{\sigma_R^2} \right)\right] \]

这里 \(\Delta p\) 是候选与演示抓取中心的距离（米），\(\Delta R\) 是两个抓取旋转之间的测地角距离（弧度或角度，需要与 \(\sigma_R\) 单位一致）；\(\sigma_p\)、\(\sigma_R\) 分别控制对位置和旋转偏离的容忍程度。

这使演示先验成为可偏离的偏好，而不是禁止不同抓取方式的限制。

还需要注意平行两指夹爪可能具有绕接近轴旋转 180° 的抓取对称性，旋转距离应结合实际夹爪配置判断，否则容易错误地惩罚几何等价的抓取。

## 4. 我实现了一套可集成的工程原型

Demonstration-Guided GraspGen-X

Python 实现、GraspGen-X 适配器、无 GUI 推理入口、单元测试和接入说明。

下载完整实现 ZIP

7 项纯 Python/Mock Backend 测试通过；真实 GraspGen-X GPU 推理尚未在你的 recovery 环境验证。

你也可以直接查看：

主规划器与 GraspGen-X Adapter · 离线推理脚本 · README

### 主要接口

```
from demo_guided_grasp import (    GraspGenXBackend,    DemoGraspPlanner,    GraspRequest,)backend = GraspGenXBackend(    checkpoints_dir="/home/shelter/recovery/models/graspgenx_checkpoints/release",    gripper_name="robotiq_2f_85",    assets_dir="/home/shelter/recovery/GraspGenX/assets",    num_grasps=200,)request = GraspRequest(    model_pc_W=model_pc,                # 用于生成的物体/局部点云    object_pc_W=object_pc,              # 完整目标物体点云    allowed_pc_W=grasp_region_pc,       # 允许抓取的表面点云    scene_without_target_pc_W=scene_pc, # 其他物体及障碍物    T_WG_prior=transferred_grasp,       # 演示迁移抓取先验    approach_W=desired_approach,        # 世界坐标系中的期望方向    preferred_approach_deg=30,    hard_max_approach_deg=110,)planner = DemoGraspPlanner(    backend,    scene_collision_free=backend.make_scene_collision_callback(),    # 执行级使用时，需要接入以下两个实际验证器：    # target_contact_valid=validate_contact,    # robot_skill_feasible=validate_skill,)result = planner.plan(request)print(result.status)print(result.source)print(result.T_WG)
```

上述示例可以生成与筛选候选，但由于尚未接入真实手指接触验证和机器人运动规划，返回的是 `CANDIDATE_ONLY`，不能直接用于控制真机。

### 在 recovery 环境中做离线测试

将 ZIP 解压到 `~/recovery/demo_graspgenx`，准备包含目标点云、抓取区域、场景障碍和可选先验的 NPZ 文件后执行：

```
conda activate recovery
cd ~/recovery/demo_graspgenx

export GRASPGENX_CHECKPOINT_DIR="$HOME/recovery/models/graspgenx_checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="$HOME/recovery/models/gripper_descriptions"

python run_offline.py \
  --input-npz scene_for_grasp.npz \
  --checkpoints "$GRASPGENX_CHECKPOINT_DIR/release" \
  --assets-dir "$HOME/recovery/GraspGenX/assets" \
  --gripper robotiq_2f_85 \
  --preferred-approach-deg 30 \
  --hard-max-approach-deg 110 \
  --env-collision-check \
  --output-npz grasp_plan_result.npz
```

NPZ 的字段及坐标系要求在 README 中列出。整个实现直接调用 Python API，不需要 ZMQ、额外训练或新的模型权重。

## 5. 真正执行机器人操作时，还必须补齐什么？

我建议把这三个检查明确分离。

| 检查器                    | 作用                                       | 当前工程情况              |
| ------------------------- | ------------------------------------------ | ------------------------- |
| Environment Collision     | 检查夹爪与桌面、柜体、其他物体的碰撞       | 已接入官方快速过滤接口    |
| Target Contact Validation | 检查指垫实际接触、闭合扫掠、非允许区域碰撞 | 需要实际夹爪几何/接触验证 |
| Robot Skill Feasibility   | 检查 IK、连续进场轨迹、抓取后技能轨迹      | 需要对接你的机器人规划器  |

官方 `filter_colliding_grasps()` 用采样后的夹爪表面点与环境点云计算距离。它的 `torch.cdist` 中间矩阵可能占较大显存，所以我在适配器里使用较小的碰撞检测批量作为起点。这个检测是快速近似，不能取代连续路径碰撞检测。

![img](https://www.google.com/s2/favicons?domain=https://raw.githubusercontent.com&sz=32)

GitHub



对于你的场景，Robot Skill Feasibility 尤其重要：

- 开抽屉： 不仅抓住把手，还要确认机器人能沿滑动方向持续运动。
- 开柜门： 要考虑铰链运动轨迹、把手方向和机械臂的连续可达性。
- Tool-use： 必须保证抓取后工具工作端仍能到达目标接触位姿。
- 搬移物体： 要验证预抓取、闭合、抬升和搬运路径。
- SE(2) Push： 这是接触推动，不应默认送入 GraspGen-X；应使用相应的推压接触规划器。

GraspGen-X 官方的 end-to-end 示例也采用了将抓取位姿转换为机器人 tool frame，再进行碰撞与运动规划的分层设计。

![img](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

GitHub



## 6. 对你最重要的两个工程决定

第一：GraspGen-X 接收的应该是抓取先验，而不能未经转换就使用 Functional Frame。

当你确实拥有源演示的机器人抓取位姿时，可以计算：

\[ {}^WT_{G,\mathrm{prior}} = {}^WT_{F_t}\;{}^{F_s}T_{G_s} \]

其中 \(W\) 是目标世界坐标系，\(F_t\) 是目标功能坐标系，\(F_s\) 是源功能坐标系，\(G_s\) 是源机器人夹爪坐标系。\(^{F_s}T_{G_s}\) 表示源夹爪相对源功能坐标系的刚体变换。此式要求 source/target functional frame 的轴定义在任务语义上保持一致。

如果只有人手演示或者仅有功能作用坐标系，没有对应机器人夹爪姿态，就不能假设这个刚体先验已经存在。需要先完成 embodiment mapping，或者仅迁移接触区域与接近方向。

第二：离线版本默认保留全局候选，不要只依赖先验。

我建议的初始搜索策略是先生成多尺度局部候选，再调用 GraspMoE 的全局候选。如果存在经过完整验证、质量很高的先验候选，可以走快速路径；否则保留全局搜索与一次更密集的 OBB 回退。

这既利用演示提供的好初值，也保留了改变抓取方式的能力。

## 最终验收标准

不能只用“能产生抓取矩阵”作为成功标准。我建议对同一批目标场景分别测试无演示先验、只做生成后重排序，以及这套三路候选方案，重点记录：

- 允许区域的真实接触合规率；
- 接近方向最大误差是否严格满足硬约束；
- 演示姿态被遮挡时找到替代抓取的比例；
- IK 和完整操作轨迹的可行率；
- 最终抓取与技能执行成功率；
- RTX 5070 Ti 上的 P50/P95 延迟和峰值显存。

最终工程选择：保留 GraspGen-X 原模型不变，在外部增加演示局部采样、判别器评分和任务可行性验证。

这样改动范围最小，也最容易与未来的 GraspGen-X 权重和模型版本升级兼容。已经提供的代码完成了这一规划框架和离线候选测试接口；真正的接触动力学与机器人运动规划必须结合你的机械臂和夹爪实现后，才能把它认定为可执行系统。