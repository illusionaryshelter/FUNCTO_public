你现在的判断已经比较完整了。**真正还没解决的核心不是“功能项权重不够大”，而是你仍在用一个单一标量目标去表达“功能必须满足、背景引导尽量好”这两种不同语义。**只要还是

\[ J(x)=J_{\text{bg}}(x)+J_{\text{func}}(x)+\cdots \]

就始终存在

\[ \Delta J_{\text{bg}}<0,\qquad \Delta J_{\text{func}}>0, \]

但总和仍下降的可能。你现在测到梯度余弦 \(-0.986\)，而且沿改善功能位姿方向背景代价几乎等量上升，正是这个机制的直接证据。

这里我先强调一个容易过度修正的点：

> **没必要要求每一次内部优化更新都始终留在 1 mm / 1° 内。**

那样容易把搜索限制得过死，尤其在非凸问题里可能导致新的停滞。真正应该保证的是：

1. 功能容差在数学上是约束，不是普通有限权重项；
2. 最终返回值必须满足它；
3. 低优先级背景代理不能通过降低自己的代价“买走”功能可行性。

这正是成熟非线性约束优化与 task-priority 方法在解决的问题，而不是继续调一个更大的 weight。Filter-SQP、增广拉格朗日以及严格/软任务优先级都明确区分“constraint feasibility”和“objective quality”。[DOI](https://doi.org/10.1137/s1052623403426556?utm_source=chatgpt.com)

## 我最建议的第一步：把两个功能要求改成真正的非线性约束

假设轨迹控制变量为 \(x\)，在功能时刻 \(t_k\) 定义：

\[ g_{p,k}(x) = \frac{e_{p,k}(x)}{\tau_p}-1 \le 0, \]\[ g_{R,k}(x) = \frac{e_{R,k}(x)}{\tau_R}-1 \le 0. \]

其中：

- \(e_{p,k}\)：第 \(k\) 个功能时刻的位置误差；
- \(e_{R,k}\)：第 \(k\) 个功能时刻的姿态误差；
- \(\tau_p=1\text{ mm}\)：位置容差；
- \(\tau_R=1^\circ\)：姿态容差；
- \(g\le0\)：表示满足功能要求。

然后优化问题应该表达成：

\[ \min_x J_{\text{soft}}(x) \]

subject to

\[ g_{p,k}(x)\le0,\qquad g_{R,k}(x)\le0. \]

这里

\[ J_{\text{soft}} \]

才包含：

- 背景球/距离场引导；
- 平滑；
- continuity；
- 其他“越小越好但不是必须为零”的项。

这和你现在的“功能优先”有本质区别：

> 目前是把功能 penalty 调得更重要；
> 这里是让功能条件退出 weighted-sum 竞争。

Ipopt 这类成熟 NLP solver 原生就是“objective + nonlinear constraints”的结构，而且其默认 globalization 就是 filter line search，而不是只依赖固定 penalty merit function。[GitHub](https://github.com/coin-or/IPOPT?utm_source=chatgpt.com)

------

## 如果不想重写 CuRobo/L-BFGS，最小侵入路线是 Augmented Lagrangian 外层

你目前已经有：

- 112 个 B-spline 控制变量；
- GPU objective/gradient；
- 自己的 L-BFGS；
- 功能误差梯度。

这种情况下，我不建议第一步直接把整个优化器换成 SQP/IPOPT。

更现实的是在现有 L-BFGS 外面加一个**标准增广拉格朗日（Augmented Lagrangian）外循环**。

概念上从当前：

\[ J_{\text{bg}} + w_f V_{\text{func}} \]

改成约束优化的增广形式：

\[ \mathcal L_\rho(x,\lambda) = J_{\text{soft}}(x) + \text{AL-term}\bigl(g(x),\lambda,\rho\bigr), \]

其中：

- \(g(x)\)：上述真实功能约束；
- \(\lambda\)：约束乘子；
- \(\rho\)：增广惩罚参数；
- \(\mathcal L_\rho\)：L-BFGS 实际优化的增广目标。

内层仍然跑你的 GPU L-BFGS。

外层则：

```
L-BFGS solve
   ↓
检查真实功能 constraint
   ↓
若仍违规
   更新 λ / ρ
   ↓
再次 solve
```

这和“把功能权重再乘 10”不是一回事。增广拉格朗日通过 multiplier update 逐渐逼迫约束成立，而不是永远允许固定比例交换。NLopt 就提供标准 `AUGLAG`，文档描述的正是“用任意本地优化器解决增广子问题，约束不满足则更新 penalty 后继续”的结构。[NLopt Documentation](https://nlopt.readthedocs.io/en/stable/NLopt_Algorithms/?utm_source=chatgpt.com)

### 这里有一个你项目中特别重要的防副作用措施

**每次更新 \(\lambda\) 或 \(\rho\) 后，重置 L-BFGS history。**

因为外层更新后，实际被优化的目标已经改变。如果保留前一个增广问题积累的 quasi-Newton 曲率历史，很容易再次出现你之前见过的异常方向尺度和停滞。

------

## 更严格的长期方案是 Filter-SQP，但不需要现在就上

如果以后 Augmented Lagrangian 仍然出现：

- 约束附近震荡；
- \(\rho\) 越来越大；
- conditioning 明显恶化；

那么下一步才值得考虑 SQP/filter。

Filter 方法本来就是为了解决：

> “我到底应该减少 objective，还是减少 constraint violation？”

而不把两者强行组合成

\[ f(x)+\rho V(x). \]

Wächter–Biegler 的经典 filter line-search 工作就是把“目标值”和“约束不可行度”作为两个独立维度判断 trial step，而不是靠一个固定 merit weight。[DOI](https://doi.org/10.1137/s1052623403426556?utm_source=chatgpt.com)

对你来说，这很契合当前现象：

```
背景代理大幅下降
但功能约束变坏
```

Filter/constraint solver 不会简单把两者换算成一个数字然后说“总和更小，所以很好”。

------

# 第二条：背景 proxy 目前应该保持“软引导”，不要升级成 hard constraint

你已经证明一个非常重要的事实：

\[ d_{\text{source}}\approx+4.77\text{ mm}, \]

但 proxy 却给出明显正碰撞代价。

所以现在不应该因为想做“严格优先级”，反过来把

\[ J_{\text{background-proxy}}=0 \]

变成硬条件。

那会制造新的误拒。

你的层次应该是：

```
真正功能要求                hard constraint
真实 joint/dynamic limits    hard constraint
最终来源实体碰撞验收          hard acceptance

sphere + ESDF background     soft guide
smoothness                  soft quality
reference continuity        soft quality
```

这样即使 proxy 仍然觉得：

> “再离背景远一点比较好”

它最多只能在**功能可行域里面**推动轨迹。

它不能再把：

\[ 0.069\text{ mm} \]

推到

\[ 1.018\text{ mm} \]

然后用 proxy cost 的巨大下降证明自己“更优”。

严格/软 task priority 在机器人里就是这么区分职责的：必须保证的条件进入高优先级 constraint，其他任务留在低层 objective。[ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0736584518303405?utm_source=chatgpt.com)

------

# 第三条：21 > 16 不应该再被当成“本轮必须停止”

这是目前另一个独立的算法性问题。

你现在实际上已经在做一种 collision constraint generation：

```
优化
 ↓
dense validation
 ↓
找到失败时间
 ↓
把失败时间加入下一轮
```

这不是特殊方法。Hauser 对连续轨迹碰撞约束的 semi-infinite programming 方法就是通过 oracle 找到最严重违例，再逐步把这些约束加入有限优化问题。[Sage Journals](https://journals.sagepub.com/doi/10.1177/0278364920983353?utm_source=chatgpt.com)

因此：

> 发现 21 个 witness，而单轮最多能加入 16 个

最合理的行为不是：

```
21 > 16
→ 整个 refinement 终止
```

而应该是 **batched exchange**：

```
发现 21 个
   ↓
选最严重的 16 个
   ↓
重新优化
   ↓
dense validate
   ↓
剩余或新出现的 witness
   ↓
下一批加入
```

这里有两个重要防副作用规则：

### 旧 witness 不应无条件丢掉

如果一个历史 witness 经过新优化仍然接近 active boundary，它应该继续留在 working set。

只有当它已经有足够 slack，并通过 dense validator，才可以淘汰。

否则会重新出现：

```
压住 A
→ B 出来

压住 B
→ A 又回来
```

### 不要简单把总容量从 16 改到 128

如果 16 是 GPU tensor / fixed shape 的设计限制，直接放大可能显著增加计算和内存。

更稳的是：

> 保持固定-size active working set + dense oracle 迭代更新。

这正是 exchange / active-set 方法的用途。[Sage Journals](https://journals.sagepub.com/doi/10.1177/0278364920983353?utm_source=chatgpt.com)

------

# 第四条：552 个正背景查询全部落在 coarse 层，说明 fine refinement 还没有打到真正的 active set

你前一次修改“在有观测支持区域使用 fine”确实有效——同一曲线下降约 7%。

但目前 active penalty 的关键 link6 仍然全部落在 15 mm coarse 层。

所以 fine-map 逻辑下一步应该从：

> “这个 link 恰好完整位于预定义 fine ROI 吗？”

逐步转向：

> **“这个 link/time 当前是不是 active collision witness？”**

也就是说，当某个 coarse proxy 真正成为优化中的主要 active constraint 时，再为它提供 fine support。

这可以是 planner-level 的：

```
coarse active witness
        ↓
link6 @ t
        ↓
申请/扩展局部 fine support
        ↓
下一 outer iteration 用 fine guide
```

而不是给某个 skill 特判。

但不要同时无限动态移动 fine ROI，否则 cost field 在迭代间会大变，L-BFGS history 会失效。**地图层发生重大切换后，也应该重置 quasi-Newton history。**

------

# 第五条：100 次 iteration 现在属于次要问题

你已经证明最终还有方向，使：

\[ J_{\text{func}}\downarrow \]

而且

\[ J_{\text{total}}\downarrow. \]

所以 100 次确实可能过早。

但现在我仍然**不建议第一步改成 300/500 次**。

因为当前存在两个更明确的算法性提前退出：

1. weighted sum 允许功能和 proxy 交换；
2. refinement 在 \(21>16\) 时直接终止。

先修掉这两个以后，再测：

- functional constraint residual；
- soft objective gradient；
- projected/KKT residual；
- witness active-set 是否稳定。

那时再决定迭代上限。

成熟 constrained solver 的停止条件也不会只看“迭代数”，而会同时看 constraint violation 和 stationarity。Ipopt 的问题形式本身就显式区分 objective 和 constraint bounds。[GitHub](https://github.com/coin-or/IPOPT?utm_source=chatgpt.com)

一句话总结这轮：

> 你现在应该从“功能优先的 weighted sum”真正跨到“功能是约束、背景是目标”的 constrained optimization 语义；同时把 21>16 的 refine 逻辑改成成熟的 exchange/working-set，而不是继续靠权重和固定容量硬撑。
