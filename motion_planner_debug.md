- 这批信息已经足够让我把问题进一步收敛成 **3 个已证实问题 + 1 个高优先级待确认问题**。现在看起来，不是 skill 本身在“泛化后失效”，而是规划链里几种不同近似的边界条件叠加后，把本来有可行空间的问题变成了拒绝。

  ## 一、已经能确认：自碰撞存在“软优化、硬验证”的语义错配

  你的自碰撞例子现在非常清楚。

  优化阶段对同一个球对看到的是

  \[ D=9.669724\times 10^{-6}\ {\rm m}^2 \]

  以及

  \[ z_{\rm opt} = \frac12\cdot2500\cdot D \approx0.01209. \]

  其中：

  - \(D=R^2-d^2\)：最大球对平方差；
  - \(R=r_i+r_j\)：球半径和；
  - \(d\)：球心距离；
  - \(z_{\rm opt}\)：优化阶段看到的自碰撞代价。

  这对 L-BFGS 来说只是一个**有限的小代价**，完全可能为了改善更重的场景/任务项而接受它。

  但验证时，同一个 \(D>0\) 被映射为

  \[ C=1+\min(z,1) \]

  所以哪怕只重叠

  \[ p\approx0.0797\text{ mm}, \]

  都会从

  \[ D=0\quad\Rightarrow\quad C=0 \]

  直接跳到

  \[ D>0\quad\Rightarrow\quad C>1. \]

  也就是说这里有一个真正的**离散语义跳变**：

  > 优化器认为“这是一个小代价，可以权衡”；
  > validator 认为“只要符号变正，就是硬失败”。

  而且你已经证明这不是实际 source collision：

  \[ d_{\rm Coal}=+8.347276\text{ mm}. \]

  更关键的是，这次甚至没有 self-collision padding。误差完全来自球覆盖：

  \[ 6.836178+3.465190-10.221683 = 0.079685\text{ mm}. \]

  所以这里我现在会明确判断：

  **这个拒绝不是调大优化权重就能从根本上修好的。**

  调大权重只会逼 CuRobo 更认真地满足一个本身存在保守误报的代理约束。

  这里更合理的职责划分应该是：

  - 球：继续用于 GPU 优化梯度和 broad phase；
  - 当最终验证出现 \(D>0\) 时：
    - 再用 source geometry + Coal 做 narrow phase；
    - Coal 也碰撞 → 真正拒绝；
    - Coal 满足安全间隙 → 不因为球代理单独拒绝。

  这和你目前已有的 Pinocchio/Coal 链非常契合，甚至不需要重新设计 skill。

  当然，这只解决**最终误拒**。如果球误报已经严重到阻止优化器进入某个可行区域，那还得改善球拟合。但你这个具体 pair 只有约 \(0.08\) mm overlap，而 source geometry 有 8 mm 以上间隙，我会优先把它当成 narrow-phase 问题，而不是先重做整个机器人球模型。

  ------

  # 二、环境问题的主要矛盾已经不是机器人球，而是 15 mm 粗图

  这个结果更重要。

  两个 OMPL 目标里：

  \[ d_{\rm link6,coarse} = -6.005,\ -6.191\text{ mm} \]

  这是 **source collision geometry 对 coarse occupancy 本身就相交**。

  所以哪怕完全删除机器人 sphere approximation，两个 OMPL 目标仍然非法。

  但与此同时，对 simulator 原始桌面：

  \[ d_{\rm link6,table} = +0.642,\ +0.337\text{ mm}. \]

  候选 01 环境最大点也是：

  \[ d_{\rm link6,coarse}=-5.133\text{ mm}, \]

  而

  \[ d_{\rm link6,table}=+0.898\text{ mm}. \]

  因此现在可以非常明确地区分：

  > 这不是 link6 真撞上桌子了；
  > 是 link6 撞上了 **15 mm 离散地图所表示出来的桌子**。

  这对于 15 mm voxel 并不令人意外。

  如果一个 occupied voxel 被看成完整边长

  \[ h=15\text{ mm} \]

  的立方体，仅半个 voxel 就已经是

  \[ \frac h2=7.5\text{ mm}. \]

  沿三维对角线，半对角线甚至达到

  \[ \frac{\sqrt3}{2}h \approx12.99\text{ mm}. \]

  这里 \(h\) 是 voxel 边长。

  我不是说你的实现一定产生上述精确误差界——这取决于 voxel 怎么从 RGB-D/ESDF 填充——但 **“真实桌面还有约 0.3–0.9 mm，15 mm occupancy 已经侵入实体约 5–6 mm”这个数量级本身非常合理。**

  所以这里实际上暴露出了一个重要事实：

  > 当前规划允许机器人姿态进入“对真实几何仍安全，但 clearance 已经远小于 coarse-map 分辨能力”的区域。

  这也是为什么你觉得：

  > “场景空间余量明明很大。”

  全局确实可能很大。

  但**这两个具体关节状态的 link6 与桌面实际上只剩亚毫米 clearance**。

  这两个概念要分开：

  \[ \exists\ \text{高余量可行轨迹} \]

  并不等于

  \[ \text{当前 seed/goal configuration 本身也高余量}. \]

  现在你的证据反而说明：

  **泛化以后得到的某些 full-body configuration 过于贴近桌面。**

  这并不否定 interaction pose 是正确的。

  ------

  # 三、我现在非常怀疑 ProcessCostManager 存在“约束追赶”现象

  你这个时间关系非常值得注意。

  候选 00 的环境最大违例在：

  \[ t=13.300000\text{ s}. \]

  于是 refine 加了 16 个点：

  \[ 13.266667 \sim 13.337500\text{ s}. \]

  重新优化之后，候选 01 的最大环境违例跑到了：

  \[ t=13.434559\text{ s}. \]

  已经跑出了新增检查窗口约

  \[ 13.434559-13.337500 = 0.097059\text{ s}. \]

  而这一次：

  - 控制点确实重新优化；
  - 最大变化达到 \(0.545822\) rad；
  - 新增点只进入 ProcessCostManager；
  - 项目场景检查本身仍是离散的；
  - 最终 validator 又比优化时密得多。

  所以目前很像这种行为：

  ```
  candidate 00:
  
  ----------X----------------
           13.30
            ↑
         collision
  
  加入检查点：
  
  -------||||||||------------
  
  
  candidate 01:
  
  -------||||||||-----X------
                      ↑
                   13.4346
  ```

  也就是优化器把违例从被加强约束的区域**推到了旁边没有同等检查密度的位置**。

  注意我现在还不能断言就是这样，因为 00→01 连几何路径也改变了，而不只是时间平移。

  但这是非常值得验证的。

  如果成立，它实际上比“interpolation_dt 是否太大”更具体：

  > **最终密集验证使用的约束集合，比 TrajOpt 真正优化的 ProcessCostManager 场景约束集合更强。**

  于是你永远可能遇到：

  \[ \text{optimizer passes its sampled problem} \]

  但

  \[ \text{dense validator rejects another point}. \]

  refine 再把这个点加入以后，又可能出现下一个点。

  这就会退化成一种 constraint whack-a-mole。

  你前面提出的：

  > 按完整机器人几何空间位移界增加检查时刻

  如果目前主要用于**最终验证**，那我现在认为它真正应该发挥价值的地方，可能恰恰是：

  > **也用于决定优化阶段 ProcessCostManager 必须检查哪些时间。**

  不一定每轮都用 2 万多个点，那显然太贵。

  但可以自适应地把“无法由现有采样证明安全”的 segment 加入 optimizer constraint set。

  这和单纯在失败以后围绕最大点加固定 16 个时间，本质上不同。

  ------

  # 四、粗细地图切换现在也出现了一个明显的非光滑边界

  这个虽然还不是主因，但我认为会影响优化稳定性。

  候选 01 的 link7 只因为超出 fine 支撑：

  \[ x\approx0.131\text{ mm}, \qquad y\approx0.695\text{ mm}, \]

  整条 link 就从：

  \[ 2\text{ mm fine map} \]

  切换成：

  \[ 15\text{ mm coarse map}. \]

  而你又说：

  - 以“整个 link 的全部覆盖球”是否被 fine map 支撑作为选择条件；
  - 两张图不叠加；
  - 不是按 collision cost 选择。

  这个规则本身逻辑是干净的，但对于梯度优化存在一个问题：

  \[ M(q)= \begin{cases} M_{\rm fine},&q\in\Omega_{\rm fine}\\ M_{\rm coarse},&q\notin\Omega_{\rm fine} \end{cases} \]

  其中：

  - \(q\)：机器人配置；
  - \(M(q)\)：当前选择的地图；
  - \(\Omega_{\rm fine}\)：整个 link 被 fine map 完整支撑的配置集合。

  在边界上，地图查询函数可能出现离散跳变。

  L-BFGS 得到的是当前地图内部的梯度，但并不知道：

  > 再移动 0.2 mm，整个 link 会换成另一张 resolution 相差 7.5 倍的地图。

  这不是 cuRobo 梯度本身的问题，而是你外部 map selector 引入的非光滑性。

  对于候选 01 的 **link6 主失败**它不是根因，因为 link6 本来就在 coarse。

  但它很可能是后续泛化中的一个稳定性问题。

  ------

  # 现在我最想继续确认 4 件事

  ### ① 候选 00 与 01 的“环境最大值”和“自碰撞最大值”分别是多少？

  最好给我：

  |               | candidate 00 | candidate 01 |
  | ------------- | ------------ | ------------ |
  | env max       | ?            | 0.222158104  |
  | env max time  | 13.300000    | 13.434559    |
  | self max      | ?            | 1.006044     |
  | self max time | ?            | 6.614063     |

  特别想知道：

  **01 是否真的降低了原来的环境问题，只是把最大值转移了？**

  如果：

  \[ C_{\rm env}^{00}>C_{\rm env}^{01} \]

  说明 refine 有效果，只是没有形成全轨迹覆盖。

  如果反而：

  \[ C_{\rm env}^{01}\ge C_{\rm env}^{00}, \]

  那重新优化本身还有更大的问题。

  ------

  ### ② ProcessCostManager 原本到底在哪些时间检查环境？

  这现在是我最关心的信息之一。

  比如是：

  - 16 个 B-spline control times？
  - 641 个导出点中的子集？
  - 固定 \(N\) 个 progress samples？
  - 每个 phase 分配固定个数？
  - 根据 task events 插入？

  另外新增的 16 个时间加入后：

  > 原来的 grid 保留不变，只增加这 16 个？

  还是会整体重新生成？

  我想计算：

  \[ \Delta t_{\max} = \max_k(t_{k+1}-t_k) \]

  尤其是 \(13.3\sim13.45\) s 附近。

  ------

  ### ③ OMPL 的两个“目标配置”究竟是硬编码的完整 \(q\)，还是一个 goal manifold？

  这个问题现在优先级非常高。

  也就是 OMPL 实际要求：

  \[ q=q_{\rm goal} \]

  还是：

  \[ FK(q)=X_{\rm task}, \qquad q\in Q_{\rm valid} \]

  然后允许自己在冗余自由度里寻找不同的 \(q\)？

  如果目标就是失败候选里拿出的两个完整关节配置，而这两个配置的 link6 恰好只有

  \[ 0.337\sim0.642\text{ mm} \]

  真实桌面 clearance，那么 OMPL 在入口：

  ```
  goal invalid
  → motion_checks = 0
  ```

  其实非常自然。

  而且这样会出现一个架构上的限制：

  > **OMPL 被允许换路径，却没有被允许换 endpoint posture。**

  如果交互任务实际上允许不同 elbow posture，那么“空间明明够但 OMPL 连搜索都没开始”就有非常直接的解释。

  这一点现在我不会猜，希望你确认。

  ------

  ### ④ 15 mm coarse occupancy 到底怎么构造？

  我想知道从 RGB-D 到这个占据体的具体规则：

  - TSDF 后按什么阈值得到 occupancy？
  - 一个 voxel 只要被 surface 穿过就把整个 cell 填满？
  - Coal 看到的是完整 \(15^3\) mm cube？
  - ESDF 距离是到 voxel center、voxel surface，还是 reconstructed zero level set？
  - coarse/fine 两层是否来自同一 TSDF 后重新采样，还是独立积分？

  这会决定我们能不能给 coarse map 建一个明确的保守误差界

  \[ \epsilon_{\rm coarse}. \]

  一旦这个界能建立，就可以把 multiresolution 规则从“ROI 里用 fine”升级成更有意义的：

  \[ d_{\rm coarse}>\epsilon_{\rm coarse}+g_{\rm safe} \Rightarrow \text{coarse 已足够证明安全}, \]

  否则：

  \[ \text{必须进入 fine / exact refinement}. \]

  其中：

  - \(d_{\rm coarse}\)：粗图查询距离；
  - \(\epsilon_{\rm coarse}\)：粗图几何表示误差上界；
  - \(g_{\rm safe}\)：真正需要的安全间隙。

  这比“整个 link 是否完全位于一个预设 fine ROI”更接近你现在真正遇到的问题。

  ------

  目前我的排序已经比较明确：

  **P1：优化场景约束的采样集合与最终密集验证不一致。**
  **P2：15 mm coarse occupancy 无法支持亚毫米级 table clearance，却被 OMPL hard gate 当作最终合法性。**
  **P3：球代理 positive 被 validation binary conversion 直接升级为硬拒绝，而 source geometry 实际安全。**
  **P4：fine/coarse 硬切换给梯度优化引入非光滑边界。**

  这四个全都是**通用 planner/backend 问题**，目前没有一个要求你给 `grasp_mug` 单独调参数。