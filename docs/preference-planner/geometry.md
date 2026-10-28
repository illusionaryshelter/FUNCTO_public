# 声明偏好规划中的几何必要约束与有限区间查询

## 实现范围与代码基线

代码基线：`PREFERENCE_IMPLEMENTATION_COMMIT`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

## 研究对象、坐标与职责

令W为标定世界坐标系，A和B分别为两个注册实体自身的固定坐标系。`T_WA(t)`、`T_WB(t)`来自同一关节曲线的机器人正运动学，或原场景接口声明的实体运动关系。机器人实体继续使用已加载的官方本体和固定末端模型。因为机器人配置、部件形状与固定关系不随任务改变，该模型可持续复用。任务实体继续使用现场RGB-D建立的原三线性TSDF、八角点观测支持域及实体局部坐标。未知域没有被赋予新的距离值。

本组件承担两个相互衔接的计算。第一，原完整实体查询发现某个时刻的相交或近邻材料见证后，从该证据构造优化器能够消费的必要不等式。第二，调用 FUNCTO 原有限采样与自适应区间查询，在同一最终关节曲线上发现实际场景违约，并记录可以证明分离或尚未完成证明的时间区间。第一项提供数值更新方向，第二项按已有采样尺度和查询预算补充几何证据。轨迹验收消费原场景残差；有限材料点和有限时间查询的结论范围分别保留在记录中。

## 必要约束的数学含义

对于原观测实体B中固定的材料点 `b_B`，其世界坐标为 `T_WB(t)b_B`。机器人实体A原生模型的有符号点距离记为 `d_A(x_A)`，外部为正、内部为负。当前必要条件为

\[
g_b(C)=-d_A\bigl(T_{WA}(t)^{-1}T_{WB}(t)b_B\bigr)\leq0,
\]

其中C为当前同一三次关节曲线的完整控制系数。若该点属于原观测占据体积的闭包，并且位于A内部，则附近存在体积相交，因此任何原模型下的无相交配置都必须满足该必要条件。实际活动域由原有限许可与支持排除决定。

`occupied_material(field, local, excluded_cells=...)` 用原观测插值检查该材料来源：严格负的支持值，或完整非正的有效单元，构成体积材料；孤立零值面不构成体积证据。机器人—观测相交见证有时正好落在机器人边界。`_native_interior_witness` 使用现成 `native.computeCOM()` 取得同一满维凸实体的内部点，沿见证到该点的线段二分；仅在原支持、占据、有限许可及原机器人有符号查询同时确认后，保存一个实际观测材料点。这是方向见证构造，几何模型和碰撞判据不随之改变。

`native_point_value` 调用现成Coal有符号距离接口；零半径球表示数学查询点。返回的当前标量和当前近似最近特征法向一起接入Torch一阶方向传播。原Coal距离迭代决定标量与法向的数值精度；Torch传播该返回方向，并不对Coal内部有限次迭代算法求导，也不对最近特征分支提供二阶导数。

观测实体之间使用两方向材料点条件。`field_point_value` 查询原 `ObservedField.sample`，使用该三线性标量及其解析空间梯度。这个标量保持原TSDF含义，不改称欧氏穿透深度。目标点落在无观测支持域时，该必要行不活动，报告保留未知。常值占据单元可以没有空间梯度，报告明确记录方向缺失；完整实体查询仍可确认相交。

机器人自碰撞继续使用原注册实体对的Coal有符号距离，并将当前近似最近见证相对运动的法向投影作为一阶方向。机械接触或被排除的实体对没有被重新添加，因为行来源始终是原完整查询实际提交的实体对。

## `GeometryConstraints` 的接口与数据流

`GeometryConstraints(computation)` 接收原完整控制系数接口的 `TrajectoryComputation`。它依赖已经绑定的原场景、机器人运动学、三次曲线映射、参考进度和物理时钟。终止区间消元先把减少后的变量展开回完整C，再传入本组件。

`add_evidence(full_controls, states)` 接收对应同一C的实体查询记录。初始有限查询使用原 `_geometry_query_states` 输出；自适应查询使用原 `scene.evaluate(..., geometry_evidence_out=...)` 输出，并为每条pair记录提供其物理时间和进度。pair记录保留实体标识、实际关系侧的 `progress`、有限许可侧的 `permission_progress` 和见证。所有已确认相交保留其具体时刻；其他查询每对实体保留最接近的一个可用见证。缺少合格材料见证时，记录跳过原因。该选择控制必要行数量，完整实体查询继续负责寻找遗漏。输入必须对应当前三次基和时钟；外部曲线由原候选验收接口处理。

`rows(full_controls)` 返回固定维度的 `g <= 0` 张量。它在每次试探下重新计算当前实体位姿、原许可域、原支持状态、当前有符号标量与一阶方向。`row_count` 为本次优化调用的行数；只有两次求解之间显式调用 `add_evidence` 才改变该维度。`describe()` 输出材料点所属实体、局部坐标、时刻、原状态、发现记录及最后一次行求值状态。

`workspace_bytes()` 按原基础查询数加当前行数计算新增FK批次的数组规模，包括仿射查询、Pinocchio位姿/速度及三个Jacobian、实体变换、接触特征图、见证标量图、原观测支持排除的串行临时数组，以及一次反向计算相对于原基础批次的差额。`witness_scalar_graphs` 与 `witness_records` 分别记录张量和共享Python证据，便于反向批次只扩展张量。原场景模型和观测值共享，不重复计入。主求解器将这些额外字节与其密集求解缓冲一起准入；`_context` 在实际构造前再次使用原 `require_host_memory` 和同一请求内存上限，读取当前驻留状态而不创建新预算。

内部 `_context` 在原完整查询之后添加见证时刻。原查询中所有释放事件锚点继续存在，`scene._owner_poses` 因而对原释放时刻的机器人位姿求导；释放后的实体不会错误跟随之后的机器人位姿。`_permission` 复用原 `scene._permission` 和初始支撑许可生产接口，每次用当前位姿重建有限区域，区分用于运动关系的progress与用于接触许可的permission_progress。机器人手指运动未知、声明附着内部实体对和许可点对各自保留原状态。

## 批量约束导数与内存

`derivatives.task_jacobian(values, controls, check_request, admit_work)` 接收已经按原声明选择的任务与几何张量，以及优化器当前的减少后控制系数。令m为选中行数，n为控制系数数量。该接口在同一个Torch计算图上用 `autograd.grad(..., is_grads_batched=True)` 和m阶单位阵同时反向传播m个输出基，返回形状为 `m × n` 的CPU数组。原Pinocchio `backward` 已使用保留前导批次轴的解析张量收缩，因此无需修改生产库。常值行通过显式零依赖得到零导数；行序、物理单位、任务选择掩码及Coal当前方向保持相同。调用前执行原请求检查和内存准入，单次反向完成后再检查同一请求。

`batch_derivative_storage_bytes(rows, variables, samples, frames, joints, geometry_workspace)` 以实际m、n、原任务查询数、机器人frame数量和具名关节数计算批量反向新增数组。输出包括 `m × m` 基矩阵、`m × n` 返回矩阵、m个方向的原Pinocchio空间协向量和关节收缩、见证附加时刻的反向数组、实体变换/有限许可/标量方向数组。原geometry字典中的场样本、观测支持及Python记录保持共享。零任务行仍按一个方向计入目标函数的标量反向缓冲。

数值适配器 [solver.py](../../manipulation/preference_planner/solver.py) 在首次总准入和每次导数回调前使用同一字典。首次总准入从原 `pending_geometry_bytes` 中扣除 `additional_largest_pinocchio_reverse`，因为新字典已计入完整批量；原标量 `pinocchio_reverse_buffers` 也由新字典替换。每次导数回调时前向张量已在当前RSS中，只将返回字典作为待分配量准入。这样初始总量与回调增量使用同一维度依据。

Coal 最近特征方向是其默认数值距离模型提供的一阶方向；当前使用原默认距离精度。方向传播与内部迭代距离函数的有限差分一致性分别记录。相关保存场景数值审查位于实验目录。

## 原区间查询、证据接入与结果含义

`PreferencePlanner._candidate_geometry` 调用父类 `_ConstrainedPlanner._candidate_geometry`；父方法拥有原 `SceneMotionBounds`、同一三次多项式的导数上界、原完整实体距离下界、物理查询预算及所有场景残差。父方法内部调用 `manipulation.interval_geometry.check_interval_geometry`。`query_geometry(times_s, point_budget=...)` 查询同一保存曲线，并返回原实体证据；初始有限查询已经计入的 `initial_states` 继续复用。

本组件接入只对当前控制系数所生成的候选曲线采集优化见证。`candidate_scene.clone_for_rollout()` 创建本次查询专用场景，保留原候选的机器人模型、观测场、手指声明以及 `_fixed_release_poses` 中的释放锚点；实体和手指实例按原克隆接口处理。本组件仅装饰此实例的 `evaluate` 方法，并沿用父调用传入的 `geometry_evidence_out` 容器。每次原查询完成后保存pair记录的引用；父区间方法完成原计费和诊断后，按批次将这些记录交给 `GeometryConstraints.add_evidence`。因此原查询中的碰撞证据、关系侧、许可侧与优化见证具有共同来源。共享场景及库内函数保持原绑定，见证消费阶段不执行第二次场景查询。

令L为两个实体在世界坐标中的点速度上界之和，Δt为时间区间长度，`d_a,d_b` 为两端完整实体分离距离的下界。原条件 `max(d_a,d_b)>LΔt` 或 `min(d_a,d_b)>LΔt/2` 足以证明该区间完整实体分离。有限许可随实体运动时，许可裁剪后的点对距离属于变化的查询域，因此原接口使用单独的完整实体下界。声明刚性内部组合与空表示域沿用原规则，时间分区保留接触激活、附着、释放和手指模型有效边界。

对于尚不能证明分离的区间，原算法按请求中的 `validation_spatial_step_m` 限制单区间最大实体位移，按剩余 `max_validation_points` 限制实际查询数量，并记录浮点时间分辨率等终止原因。到达采样尺度表示本次自适应细分已完成其有限工作范围，不表示整段轨迹无碰撞。已查询状态中的实际相交仍直接进入原场景残差；观测支持、未知运动及允许尝试区域保留原场景声明。

返回的 `proved_intervals` 保存已经成立的区间证据，`confirmed_intersections` 保存实际查询到的相交位置与实体，`unresolved_intervals` 保存采样尺度、预算、缺失边界侧或运动模型未知等原因。条件未知继续通过原 `conditional_unknown_residuals` 等场景通道表达。验收使用原 `hard_scene_residual` 与原任务、关节域残差，未决区间本身仅说明证据范围。该结果表示当前声明模型和有限查询范围下的名义运动可行性；抓取是否持有、释放后是否放稳等物理效果由独立执行结果评价。

## 验证与计划对照

[test_geometry.py](../../tests/preference_planner/test_geometry.py) 与 [test_derivatives.py](../../../sim-version-plans/2026-10-26-twenty-second-planning/joint-planner-003/tests/test_derivatives.py) 使用原Coal、观测场、场景关系和Pinocchio接口，检查有符号值及有限差分方向、常值零占据与零值面区别、相交见证、维度稳定、释放锚点导数与当前有限许可。批量测试用解析机器人box与原TSDF材料行验证全部控制列，并用保存问题的完整 `98 × 126` 矩阵对照批量与逐行API。

[test_geometry_sampling.py](../../tests/preference_planner/test_geometry_sampling.py) 构造一个沿25毫米直线运动的Coal盒及100毫米观测占据立方体，两个实体全程位于同一个有限许可区域 `[-1,1]^3`。原实体查询在实际查询时刻均为clear，完整实体分离下界为零；原算法查询两个端点后保留 `spatial_resolution` 与 `dynamic_permission` 未决记录。另一个真实TSDF反例验证原采样尺度下的自适应中点查询仍能发现相交。该组构造分别检查已许可点对、分离证明不足和实际相交的含义。

[test_backend.py](../../tests/preference_planner/test_backend.py) 比较原父方法与本组件接入在真实非空场景中的完整报告和查询计费，使用原运动界、FK及Coal/TSDF查询，并验证带已冻结释放锚点的场景克隆、同一时刻左右关系侧和必要行更新。候选验收测试同时保留未决、条件未知和实际相交三类结果。相应测试通过实际 Coal/TSDF 查询检查上述接口；测试运行记录位于阶段计划。

完整请求中的可行路径、实际所需约束数量、自适应查询开销及物理执行效果由主实验记录提供。必要材料点约束在复杂常值观测体积之间可能缺少有效局部方向，现有连接搜索及完整原几何查询继续承担该计算。
