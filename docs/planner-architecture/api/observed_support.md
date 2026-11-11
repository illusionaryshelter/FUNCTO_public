# `manipulation/observed_support.py`：原观测占据体积的中心与有限支撑几何

代码基线：`CODE_BASELINE_PENDING`。

代码入口为 [observed_support.py](../../../manipulation/observed_support.py)，回归入口为 [test_observed_support.py](../../../tests/test_observed_support.py)。本组件依赖原[连续占据集查询](observed_collision.md)和[观测场表示](../../../manipulation/implicit_distance.py)。

## 职责与几何对象

本组件把同一现场 RGB-D TSDF 用于体积中心估计、旋转后的法向最低高度、有限承托域的边方向极值和固定旋转放置目标。组件不改变原始场的坐标、观测支持或材料集合。设场的固定局部坐标系为 B，格点间距为 h。当前持有物体工厂将同一坐标系记为 C；B 与 C 在该调用中指同一个 RGB-D 获取坐标系。`ObservedField` 将八个格点均已观测的单元组成支持域 D；每格内采用原值的三线性插值 f。实际被查询的材料集合是 Omega={p 属于 D 且 f(p)<=0}。D 外的位置继续为未知。

场由 `observed_mapping.freeze_mapper` 从现场 RGB-D 融合值、观测权重与观测支持生成，或由 `FrozenObservedField.from_record` 恢复同一原始值。观测权重用于融合与确定观测支持。体积积分对物理空间中的等体积子格使用相同体积权重。各 API 接受已有场实例；组件不创建替代曲面或材料模型。

体积中心 c_B 定义平移坐标变换 K=T_B_G=[I,c_B]。G 的坐标轴与 B 相同，原点为 c_B。原场、声明附着、源功能区域和材料坐标仍在 B 中。将中心参考换算为场参考时使用 T_W_B=T_W_G K^{-1}。W 表示调用方与目标支撑面共用的参考坐标系。

## `estimate_occupied_volume_center(field)`

输入为已构造的 `ObservedField` 或其 `FrozenObservedField` 子类。该 API 在 `field.cells` 枚举的候选格内进行固定求积；这些格已经满足八角均观测且至少一个角点非正。候选格本身只限定搜索区域，不能整格视为占据体积。

每个原格按每轴二分形成八个等体积子格，其归一化中点属于 {1/4,3/4}^3。调用现有 `implicit_distance.trilinear` 得到原插值值 f。设所有满足 f<=0 的中点为 p_k，其体积权重为 dV=(h/2)^3，则估计体积 V=dV N，一阶矩 M=dV sum_k p_k，中心 c_B=M/V。1024 个原格构成一个计算批次；该批次与原场已有的批处理尺度一致，细分格和中心计算不建立持久几何索引。

输出为可直接序列化的字典：

- `center_local`：B 中的三个中心坐标，单位 m；
- `volume_m3`：固定求积得到的占据体积，单位 m³；
- `quadrature_spacing_m`：h/2，单位 m；
- `occupied_sample_count`：参与求积的中点数 N。

这是已表示占据体积的数值估计。若固定求积没有占据中点，V=0，中心无定义，函数抛出说明零体积的 `ValueError`。后续方向极值仍查询完整连续 Omega，因此体积求积的边界误差不成为碰撞或支撑集合的修改。

## `support_extrema(field, pose, surface, *, precision_m)`

`pose` 为 T_W_B；`surface` 为既有 `goals.SupportSurface`。该对象提供 W 中的平面原点 o、单位法向 n、正交切向基 U（3×2）和有限凸多边形。多边形第 j 条边由原 SciPy ConvexHull 方程 a_j^T eta+b_j<=0 表示。`precision_m` 是连续查询的数值区间精度，输入到既有 `query_minimum_height`。

法向查询给出 min_{p in Omega} n^T(T_W_B p-o) 的下界 L 和上界 H。空间盒给下界，实际支持材料点给上界。法向结果 `normal` 原样保留连续查询的 `status`、`lower_m`、`upper_m`、`witness_local` 和存在时的 `visited_boxes`。见证位于 B 中，是给出上界的可行材料点，精确极小值不要求在该有限查询点处取得。

边方向为 d_j=U a_j。沿 -d_j 调用同一最小值查询，若其区间为 [L_j,H_j]，则边最大投影 max_{p in Omega} d_j^T(T_W_B p-o) 位于 [-H_j,-L_j]。这个见证仍位于 B 中，并给出最大值的下界。输出 `footprint` 为并行序列：

- `directions_world`：J×3 的 d_j，W 中的单位边方向；
- `offsets`：原边方程的 b_j，单位 m；
- `lower_m`、`upper_m`：J 个最大投影数值区间端点；
- `witnesses_local`：J 个 B 中可行材料见证；
- `statuses`、`visited_boxes`：每个方向的原查询状态和访问数。

所有投影都相对 o，边常数 b_j 未并入方向极值。`bounded`、`unresolved` 和 `missing_observation` 由原查询决定；缺失域的端点和见证保留为 `None`。调用方可以由原场坐标见证和刚性位姿计算其导数，同时保留连续查询的实际数值界与状态。

## `place_observed_on(preferred_center_pose, field, surface, *, center_local, margin, precision_m)`

输入 `preferred_center_pose` 为期望中心位姿 T_W_G；`center_local` 为 c_B；`margin` 为原任务声明的非负边界余量。最终旋转 R 完全取自输入位姿。当前调用方 `reference_transfer.observed_manipulation_targets` 先从源静态任务关系、原RPY声明与目标功能坐标确定R，再调用本组件。

首先构造 T_centered=[R,o-Rc_B]，使 G 位于支撑原点。`support_extrema(field,T_centered,surface,...)` 计算 R(p-c_B) 的法向最小区间 [L,H] 与各边最大区间 [l_j,u_j]。中心切向坐标 xi 必须满足 a_j^T xi<=-b_j-margin-u_j。组件调用现有 `goals._nearest_feasible`，将输入中心在切向基 U 中的坐标投影到该有限线性可行域。该求解器沿用库已有的 SciPy SLSQP 计算和结果条件。

以 z=-L 构造中心位置 x=o+U xi+z n，输出原场位姿 T_W_B=[R,x-Rc_B]。这使原集合的最低平面间隙处于 [L+z,H+z]，有限边约束处于 [a_j^T xi+b_j+margin+l_j, a_j^T xi+b_j+margin+u_j]。初始支撑面到中心的距离不进入此计算。相同最终旋转下，当前几何增高会改变法向目标；旋转改变时，最低材料位置由同一连续场重新计算。

返回值为 `(pose_B,evidence)`。`evidence` 包含 `geometry_source`、`precision_m`、`margin_m`、`center_local`、`center_world`、`center_xy`、中心化形状的 `normal` 查询、最终 `minimum_gap_interval_m`、原边极值及新增 `inequality_lower_m`/`inequality_upper_m`。`geometry_source` 保存 `surface.source`，即有限支撑面的观测来源；原场的观测来源保存在调用方持有的场记录中。中心、平面与方向均有明确坐标，因此结果可叠加到原场景中。

缺失已表示材料时无法构造放置目标，函数抛出 `ValueError`。有限但未解的方向区间仍用于构造候选并保留其原状态；该初始化 API 不决定规划候选的接受。有限可行域为空时，由现有平移求解器报告物体无法容纳。

## 调用关系与依赖

中心估计调用 `implicit_distance.trilinear`；方向极值调用 `observed_collision.query_minimum_height`；有限放置调用本组件的方向极值以及 `goals._nearest_feasible`。位姿结构使用已有 `goals._pose`。运行依赖都是环境中已有的 NumPy、SciPy 和几何查询依赖。

`reference_transfer.observed_manipulation_targets`在构造完整源轨迹之前，从原保留field调用中心估计与固定旋转放置。返回C目标、中心K及原方向证据。`generate_observed_reference`再调用原源运动生成与连续右组合，输出原工具路径及边界。`continue_held_motion.build_problem`使用原MotionPhase、BoundToolReference、PoseEventRequirement和SupportRequirement；几何目标计算属于参考组件。

## 已完成回归

`test_observed_support.py` 的 12 项回归已在 recovery 解释器下通过。独立预期覆盖均匀长方体解析中心与体积、半空间切分格的部分占据体积、未知格排除、零求积体积、长方体旋转90°后的最低高度、边方向极值的可行原场见证、较高当前几何的中心目标、有限区域投影与无法容纳、更换支撑切向坐标表达、原场局部平移换表达、缺失观测状态传递，以及有限未解查询状态不增加放置门槛。命令采用 `PYTHONNOUSERSITE=1`、单线程数值库以及 recovery 已有 pytest；当前结果为 `12 passed in 2.50s`。
