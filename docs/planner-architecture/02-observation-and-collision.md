# 观测与碰撞模型

观测场算法基线：`be44b54`；官方机器人距离批处理与资源估计的代码基线：`12e0ef5`；通用距离函数当前实现见 [implicit_distance API](api/implicit_distance.md) 的代码引用。组件正文描述当前实现；阶段验证位于项目外 `sim-version-plans`。

`implicit_distance.box_distance(a,b)` 接收同一查询坐标系内两个轴对齐包围盒的上下界，每个端点为长度三的 binary64 数组。函数计算逐轴间隔的欧氏范数，按原端点数值规模减去算术舍入余量，再向负无穷舍入，返回非负距离下界。端点规模由 `np.concatenate((a[0],a[1],b[0],b[1]))` 按原顺序连接后求绝对值最大值。该函数用于观测场、官方机器人实体与观测场、初始支撑的查询；包围盒下界描述空间范围的分离程度，实体碰撞结论仍来自各查询的实际表示与证据。

本次文档引用回填不改变代码；上述实现提交与本次文档提交的相关生产代码相同。

## 1. 两种实体来源

`build_scene_from_rgbd` 把新观测与已有模型组成一个 `OnlinePlanningScene`。`owner_frames` 仅包含本次按 RGB-D 建立的实体，给出局部坐标和来源；`observed_owners` 提供已有 `CollisionOwner`，保留其场对象、原样本、分辨率、观测支持及声明运动。同一身份选择一种来源。任务对象身份在两组实体的并集中解析。

构造器必填的 `robot_model` 是与机器人验证器共享的固定机器人官方模型检查器，`fixed_geometry_names` 声明固定安装体，`tool_geometry_names` 声明附着所使用的完整工具组件。后两者均为该检查器中已注册的几何实体 ID；其来源是机器人适配器提供的模型身份和机构拓扑。官方本体及官方固定末端执行器的 mesh 可持续复用，位姿由标定、关节反馈与 FK 确定；任务实体、环境和其他对象采用 RGB-D 场及其观测支持域。

新观测视图包含 `image:uint8[H,W,3]`、相机 z 向米制 `depth`、零斜切内参 `K`、`T_output_camera`、参考坐标、各新实体的布尔 `owner_masks` 和机器人过滤结果。`_partition_masks` 只验证新建实体的像素分区。已保存实体不进入可见深度计数。

静态背景可直接通过 `background` 传入，已测自由空间通过 `coverage` 传入。两者和所有实体均已给定时，视图可为空，构造过程不调用 Mapper。需要建立背景时，`background_mask` 明确选择该幅观测中的静态端点；普通首次建图也可用新实体 mask 的补集。在含有独立复用实体的观测中，静态分区由调用者明确提供。原观测中未映射的端点记录为 `unmapped_pixels`，不被解释为新增自由空间。

背景积分保留一圈像素身份不确定带。令 `E` 为有效 ENVIRONMENT 像素，`B` 为上述背景候选 mask；边缘带为 `E & B & binary_dilation(E & ~B, 3×3, 一次)`，最终背景为 `E & B` 排除该带。该规则同时覆盖新建实体与原静态分区所排除的已有实体，原实体 mask 不扩张。SELF、AMBIGUOUS_SHADOW 和无效深度不产生边缘种子。被排除端点计入现有 `unmapped_pixels`，自由空间仍由独立深度射线计算。这是一像素的分割边界保护，不是任意分割误差的几何上界。

`scene.observation_partition_masks[view]` 独立保留三种数据：`owner_masks` 是本次传入的新建实体原始语义分区，`background_candidate` 是过滤前的背景候选 `B`，`background` 是实际积分背景。前两者复制为只读数组，保留被机器人过滤排除的原语义像素。显式背景候选直接保存；首次新建背景而未给候选时，保存原实体 mask 的补集。复用背景时，实际积分背景为 `None`；此时仅保存明确提供的候选。全部模型复用且无需视图时，记录为空。

`save_observation_partitions` 按视图写出实体原始 mask、背景候选和实际积分背景，实体 ID 与路径对应关系存入 `partition-report.json`。再次建图时，`--owner-mask` 读取实体原始文件，`--background-mask` 读取 `background-candidate-mask.npy`，从而只执行一次边缘排除。已有实体继续复用自己的模型及原有记录。

## 2. 深度过滤与场的形成

`robot_self_filter.classify_depth` 使用图像采集时具名关节 FK 放置同一已知机器人组件。端点严格位于名义实体内部时标为 SELF；其余端点对应的相机至测量点有限线段若与明确 padding 后的实体相交或处于数值边界，标为 AMBIGUOUS_SHADOW；其余有效测量标为 ENVIRONMENT。只有 ENVIRONMENT 保留原测量深度作为自由射线。padding 定义过滤阴影域，规划碰撞实体保留原形状。

`integrate_entity` 接收 RGB、深度、实体 mask、K 和 `T_entity_camera`，逐实体串行融合。`_mapper_intrinsics` 将主点增加半像素，以适配 Mapper 的像素中心约定。`freeze_mapper` 读取加权 TSDF 与权重，保存 XYZ 排列的 float32 场值；`origin` 为首个样本中心。最小有效权重确定样本观测支持。

`FrozenObservedField.record/from_record` 保存和恢复原始值、权重、支持、原点、间距、身份与来源；空间索引由这些样本重建。`clone_for_rollout` 返回同一不可变场。任务实体新建时采用 2 mm 样本间距，环境采用明确给定的更粗采样间距；已有场保留原分辨率。这些长度定义 TSDF 网格。资源准入按串行 Mapper 峰值及当前已驻留数据计算。

## 3. 同一原始场上的非穿透查询

`ObservedField` 在八角点均有支持的闭单元内作三线性插值，以 `φ≤0` 定义占据。`query_distance` 输入两个原场、各自刚体位姿、数值精度 `precision_m` 和有限点对许可，检查未获许可部分的体积交叠。变换后的空间框产生距离下界，原占据集合中的实际点对产生上界；这些界用于排除、细分和诊断，不要求实体保有额外正间距。浮点误差进入边界计算。初始化按1024个单元分批计算同一边界收紧公式，单元内部仍依次处理三个坐标轴；标量细分与批量构造共享值及误差计算。

原始叶节点的局部点集由 `_initial_occupied_points` 在建树时按 1024 格分批计算，包含原角点、格内样本及边上的零点，并保持 `occupied_points` 的原顺序和算术。`_Node.points` 保存这份只读数组；后续刚体查询把同一局部点变换到相应实体姿态。动态细分节点继续按其当前范围计算点集。`storage_nbytes` 同时计入这些数组及其数组头，供场景资源准入使用。

占据树保留全部可能包含非正值的完整支持单元，包括全负内部格和常值零格。优化目标的材料点直接来自这些原叶；接触偏好的固定零点来自原顶点零值及符号变化边上的解析零根。三线性值和空间梯度均由本次试探位置所在的原支持单元计算。

体积交叠见证须严格位于双方原始支持单元的开内部，包含坐标变换误差后的范围也不能触及原格边界。在每一侧，该范围上的插值值须可靠地严格为负，或原单元八角点全部非正。后一条件使整个单元属于 `φ≤0` 占据，包括插值恒为零的满格。由此确认共同占据的开邻域，而非仅有一个公共边界点。这里的原格边界不是搜索时产生的子框边界。

| 状态 | 几何含义 |
|---|---|
| `clear` | 已表示的未许可查询域完成排除，没有留下体积交叠证据或未决域；空集合另带 `empty_set` |
| `intersection` | 原表示中已确认存在共同占据的开邻域，即体积交叠 |
| `unresolved` | 未发现上述交叠证据，但仍有域在当前数值精度或许可边界下无法判定 |

`PairPermission` 描述允许的点对域。完全许可的框对被排除，其余继续查询；混合许可框采用按进入顺序展开，明确未许可框优先细化。见证必须属于未许可域。该顺序不改变原场及许可范围。

`clear` 报告的 `lower_m` 取所有已严格分离分支下界的最小值。空占据集合返回 `lower_m=None`；若全部查询域均被许可移除，则返回有效下界 0。`distance_scope='unpermitted_point_pairs_at_query_pose'` 明确该距离针对本次姿态和本次许可域。若后续姿态或许可域改变，需要相应的运动与域关系才能解释两个时刻之间的距离。

`full_entity_lower_m` 另保存完整实体对的距离下界。有点对许可时，该值来自许可裁剪前的两个完整实体包围范围，因而与当前许可区域的位置独立；没有许可时，可使用完整查询得到的更紧下界。空占据集合的该字段为 `None`。几何时间区间检查读取这份完整域证据，把实体运动界与距离界结合；受许可裁剪的 `lower_m` 继续用于解释本次静态查询。

数值精度约束局部搜索的停止范围。除子域已细分到指定尺度外，还可在本域的实际占据点对距离上界不超过 `precision_m`、且双方空间框交集至少一个轴向厚度不超过该精度时停止细分。本域随即记为未决，查询继续处理其他域；不能用另一个域的近距离点对替代本域上界，也不能由松散框的薄交集直接推断实体接触。只有遍历结束且没有交叠证据时，才根据是否存在未决域返回 `unresolved` 或 `clear`。

`pair_residual` 为 `intersection` 返回分类残差 1，为 `clear` 和 `unresolved` 返回零碰撞残差，并独立返回原查询证据。分类量的导数为零；其单位是状态标记。未决身份保留在报告中。场景调用通过 `zero` 复用当前样本的可导零值，独立成对调用由两个位置张量构造该值。

统一约束求解器通过 `OnlinePlanningScene.optimization_objective` 调用[固定材料点目标](api/sampled_geometry_objective.md)。求解首次调用固定各原场叶材料点及原单元身份；官方机器人形状使用已有可行点构造得到固定材料坐标。此后每个试探在原三线性支持域内读取实际场值和空间梯度；反向将原场点查询到官方机器人原生形状，包含关系由原生有符号点距离给出。两个方向的负值平方按原观测间距归一化，再取均值。原生自身配对保留 Coal 有符号距离及其当前见证导数。

`MaterialSamples.point_bounds` 包围固定源材料坐标，`negative_bounds` 包围目标函数可能取负值的坐标。观测场的后一范围来自原占据树根所包含的完整支持单元：支持单元的三线性插值由非负权重组成，取负值要求至少一个原角点非正，因此这些完整单元覆盖全部可能负值位置。该范围计入原场格点查询的浮点运算误差。官方原生实体沿用已有支持投影范围。`_pair_separated` 用 `world_bounds` 和 `_local_interval` 将源材料范围变到目标坐标，并保留正变换及逆变换的数值区间；仅当两个方向均严格分离时返回零目标与两实体十四个位姿分量的零导数。许可只删除贡献，所以同一结论适用于原有限区域和原单元排除。范围相交或接触时继续计算可能贡献的原材料项。`zero_contribution_separated_pair_count` 记录此类零贡献配对。

<!-- BEGIN fixed material selection preserves the original finite objective. -->
每份观测材料另有不可变的 `cell_ranges[C,2]`：以原单元 ID 寻址，保存该单元在原材料数组中的半开范围 `[start,stop)`。构造顺序仍为原树叶遍历顺序，同一坐标在相邻单元中的记录分别保留。`cell_radius[3]` 保存实际材料点与原 `ObservedField.tree` 实际中心之间逐轴差的向外最大值，沿原材料数组按1024点一批求最大值；名义半网格间距不能替代这一数值界，因为浮点中心与端点的差可能略大于名义半径。

`_material_indices` 输入原源材料、目标材料及本次两实体位姿，输出原材料数组索引。以下使用列向量，`(A,a)` 和 `(B,b)` 分别表示源、目标局部坐标到同一参考坐标的数值旋转与平移。原点变换为 `w_hat=fl(Ax+a)`、`y_hat=fl(Bᵀ(w_hat-b))`。令 X 为源 `point_bounds`，D 为目标 `negative_bounds`，`E_w` 为 `transform_point_error` 对 X 逐轴最大绝对坐标计算的前向误差，`W=world_bounds(X,A,a)`。代码以现有 `_GAMMA=32eps/(1-32eps)` 形成 `E=γ|B|ᵀ(max_abs(W)+|b|)+|B|ᵀE_w` 的向外界；因此 `y_hat∈D` 时，精确 `Bᵀ(Ax+a-b)` 属于 `D_E=D+[-E,E]`。

SciPy 四元数转换给出数值旋转。代码计算 `δ≥||BᵀB-I||∞`，包含矩阵乘法舍入，且该旋转满足 `δ<1`。恒等式 `B⁻ᵀ-B=B[(BᵀB)⁻¹-I]` 给出 `η=||B||∞δ/(1-δ)`；令 `m=max_{y∈D_E}||y||∞`，先计算 `world_bounds(D_E,B,b)`，再将每轴向外扩大 `ηm`，即包围目标变换的精确逆解。现有源 `_local_interval` 最后完成参考坐标到源局部坐标的包围，包括 A 的逆向算术及正交缺陷。正量误差式的求值乘以 `1+γ` 并使用 `nextafter` 向外取整；全部误差均描述坐标计算，不改变物理容差。

`MaterialSamples.indices_in_box` 接收上述源局部闭盒，先按 `cell_radius` 扩张，再以已有中心树的 `query_ball_point(p=inf,eps=0,workers=1)` 查询包含它的立方体。树查询半径保留坐标减法的数值外包围；随后逐轴闭区间比较筛选原单元。候选单元按原 `start` 排序，再展开其连续索引。树 ID 的排序不能代替材料顺序。官方原生源材料没有观测单元，直接返回原固定点索引。这是一条查询规则，索引内只保存局部不变数据，每次姿态重新计算选择范围。

`_field_direction` 和 `_native_direction` 只对这些原记录执行原 TSDF 或 Coal 查询，保留原支持掩码、有限区域许可和单元排除。负值对应的原坐标重新连接到本次两实体的可导位姿；每个方向仍除以完整 `len(source.points)`。被省去的项在原数值查询下不能贡献负值平方，故局部标量及一阶导数保持原定义。`sample_count_scope='material_points_actually_queried'` 描述实际查询的支持、未知和活动点计数；`fixed_sample_counts` 保存完整材料数，查询数量可因空间筛选而减少。owner 释放后的姿态继续来自本候选的释放锚点，局部索引不保存跨时刻的位姿筛选结果。

`native_pair_objective` 始终执行原 Coal 有符号距离查询。距离 `d≥0` 时，平方负部 `max(0,-d/l)²` 的值与一阶导数均为零，包括 `d=0`；函数直接返回零标量。负距离继续读取原见证和法向并构造同一位姿导数。场景 `optimization_objective` 已有一次连接位置、四元数及完整关节控制量的全局零项，因此全零目标仍可求导，每个非负原生实体对不再建立重复的零值旋转图。
<!-- END the same original queries own collision and contact acceptance. -->

有限点优化目标具有明确计算范围。原支持域和许可边界定义分段分支，最近特征切换处可能不可微；常值零场和常值负场内部的原空间梯度为零；有限点之间的交叠可能未进入目标。`report_out` 保存这些范围及实际查询计数。原实体体积交叠、required contact、覆盖及连续时间区间仍由各自原查询给出证据。

`sample(excluded_cells=...)` 在目标查询时移除初始支撑声明的原第二实体单元；反向点集按同一单元 ID 删除对应材料。共享边界仍可由未删除邻格提供原值。有限接触仍由原 `_permission` 的实际半空间定义，两个重叠点端点分别测试。required contact 的偏好取固定原零点相对声明区域的最小超差；实际接触状态仍由 `query_region_intersection` 提供。

每个实体对的临时点梯度先归约到两实体十四个位姿分量，再连回调用者的 FK 和三次轨迹变量。轨迹图按实体对及状态保存位姿导数，不按材料点数量保存整段点图。`estimated_workspace_allocation` 将 CPU 需求分为三项：`cpu_resident` 记录已驻留的原场、当前原生配对表及已构造材料数组；`cpu_pending_shared` 记录尚未构造的固定材料与配对表上界；`cpu_live` 记录原生数值块、一次实体对计算的临时数据与保留的位姿图。材料对象按身份去重，构造后其实际数组字节转入驻留项；每个材料对象的两个三维上下界数组最多占 96 字节，同时进入构造前上界和构造后实际计数。每份观测材料的原单元范围与中心偏差新增 `16C+24` 字节，同样在构造前计待分配、构造后计驻留；场景 clone 共享该不可变对象。一次方向查询的原 `256N` 临时上界增加 `64N+96C`，覆盖展开索引及重复缓冲、所选坐标、活动索引映射，以及中心树结果转换、中心与逐轴掩码、范围排序和偏移数组；它与材料构造、接触零点的临时上界取最大值，不按轨迹长度重复计材料数组。原场和已分配样本由 RSS 统计，新增准入使用待分配项与临时项；GPU 项仅计点查询中具名关节、连杆位姿及其梯度的张量传送缓存。该目标是一阶接口，供同一 SLSQP 内核计算目标与雅可比。

## 4. 已知机器人与原任务场的组合

`RobotGeometryChecker` 用 URDF 建立 Pinocchio 关节树，以 `NativeSolid` 原凸组件顶点或 `NativePrimitive` 解析盒体/圆柱建立 Coal 碰撞实体。模型保留实体 ID、所属连杆、固定变换与官方来源。完整具名关节及基座标定确定各实体位姿。该模型对应当前固定机器人及其官方固定末端配置，可跨任务复用；任务物体、环境和其他对象的 mesh/CAD 不进入该许可范围。

`query` 返回原生有符号距离、最近点、法向、FK 和关节域。`required`、`excluded`、`mechanical_contact` 是调用者明确提供的原模型配对语义。自碰撞使用 `separated`、`penetrating`、`numerical_boundary` 区分原生查询结果，保留 Coal 默认数值边界；该数值尺度不等于严格误差界，数值边界单列未决。

同一检查器的 `_query_schema` 保存最近一次完整查询的结构：具名关节顺序及其模型列、对应官方位置界、所需帧到 Pinocchio frame ID 的映射，以及完整 `PairQuery` 元组。`PairQuery` 为冻结记录，其有向端点、策略和来源共同参与结构相等比较。查询结构改变时，`query` 重新验证完整关节覆盖、帧身份、配对端点与配对策略，验证完成后替换这一个记录；相同结构直接复用这些模型关系。每次实际关节值和基座标定仍在当前调用验证，当前关节限位、FK、几何位姿和距离见证由当前状态计算。

`_prepare_query` 供完整报告与数值批次共享输入验证和 float64 FK。`query` 服务 `RobotStateValidator` 与执行前检查，返回关节限位、帧和实体位姿以及逐对距离报告；`query_pair_batches` 服务规划场景，只交付各状态有序配对的十二个数值。两者经 `_pair_table` 使用同一个 `NativePairTable`；该表只持有官方形状强引用和有向配对索引。端点顺序改变时替换表，策略和来源仍随当前 `PairQuery` 进入报告。每次 Coal `distance` 创建独立冷求解器，不缓存见证点、距离或 GJK/EPA 历史。原有 Pinocchio 关节模型继续承担 FK。

`query_pair_batches` 按每块最多 4096 次配对组织连续状态；若单状态配对更多，该状态作为完整一块。准备状态、进入与离开原生调用、场景逐对消费时分别检查请求期限。原生块内错误使整块失败，错误记录块的状态范围以及块内状态、配对索引。形状与配对索引在固定表中持有；原生调用在释放 GIL 前复制变换、mask 和请求，返回独立 NumPy 数组。Python 批次接口将数组转成调用者持有的列表，用于已有 JSON 证据。具体坐标、字段和舍入推导见[原生配对接口](api/native_pair_table.md)。

资源估计将已存在的表字节 `native_query_resident_nbytes` 计入驻留项，将需要构造或替换的表上界 `native_query_storage_nbytes` 计入待分配项。`native_query_temporary_nbytes` 始终进入单次查询的临时项，涵盖相邻块短暂共存的矩阵、原生变换、输出数组和 Python 列表，以及一个冷距离求解器。固定表已存在时，这些临时数组仍需预留。

`RobotStateValidator.required_pairs` 是显式配对策略中所有 `required` 记录组成的元组。`OnlinePlanningScene._robot_self_components` 将该元组及连续完整关节状态传给 `query_pair_batches`；投影 mask 仅在需要区间下界且该配对运动已建模时为真。每个配对仍计算原始距离。`pair_evidence` 统一构造数值关系和具名见证，场景随后消费未知手部状态、穿透值与见证导数；完整 `robot_pairs` 继续用于模型描述与快照保存。

`DifferentiableRobotKinematics` 直接接收共享检查器现有的 Pinocchio 完整模型，包括本时刻名义手指坐标。`poses` 将输入显式传送为 CPU float64，调用同一 `PinKinematics`，再把位姿传回输入设备和精度；该传送保留梯度链。场景对解析实体乘固定局部变换；凸顶点已在连杆坐标中，局部变换为单位阵。`bind_online_robot` 只接收验证器和期限回调，模型与基座标定来自原验证器。

`components(..., geometry_scope="robot")` 和 `geometry_scope="full"` 先运行同一手指状态、完整具名关节 FK、机器人实体放置和 `_robot_self_components`。共享函数输出 `collision/robot/self` 与 `unresolved/robot/self`；确定穿透使用原生有符号距离及其见证导数，数值边界保留独立未决分项。缺少名义手指几何时，对依赖相应关节的原配对记录 `unmodelled_motion`。手指控制残差与 conditional 手部状态也由同次手部计算保留。`robot` 范围完成这些项后返回；`full` 范围继续构造任务实体位姿并查询接触、观测场、支撑和覆盖。初始化与完整场景由此共享同一官方机器人实体、配对和数值分类。

`native_field_distance.query_native_field` 直接查询原生实体与原任务场。原实体的支持投影及任务空间框产生距离下界；原实体内可行点及原场占据点产生上界。凸体内点由原面顶点和原顶点均值的非负组合构造，区间计算保留该实际点的局部与世界范围。原生实体保持其有体积的凸体或解析形状；若其实际可行点完整区间严格位于原支持单元的开内部，且该范围上三线性值可靠为负，或该原单元八角点全部非正，即有原场占据开邻域与原生实体发生体积交叠。未许可的这类见证产生 `intersection`。`witness_*_bounds` 保存点的数值区间，中点仅用于显示和局部导数。结论针对已知机器人模型与已表示任务场，不输出真实物理穿透深度。

通常仅细分任务场；点对许可跨越边界时，局部搜索还细分原生实体与搜索盒的交集。数值精度停止条件同时考虑当前许可域所需的细分尺度；薄域停止仍要求本域实际点对上界不超过精度，且空间框交集至少一个轴向厚度不超过精度。局部未决不结束其余域的检查。空间框交叠只用于安排计算，不能代替原实体归属或体积交叠证据。

`query_native_field(..., native_bounds=(local,world))` 可直接消费同一原生实体的局部与世界投影范围。场景构造时保存不可变形状的局部范围；`components` 在本次调用内按实体和状态计算一次世界范围，供不同场配对复用。空场无需构造对应世界范围。required contact 使用的作用实体矩阵也在整批转换一次，然后按状态读取。这些范围和矩阵只复用相同几何计算，实体查询仍使用原形状和原任务场。

完整范围的 `OnlinePlanningScene.components` 计算原生机器人自身、机器人与观测场、运动实体与背景、实体之间的配对，返回相同硬分项。主约束求解器通过独立 `optimization_objective` 取得固定有限目标，通过 `components` 与连续区间接口取得原实体证据，具体输入和作用见[场景接口](api/online_geometry.md)。`evaluate` 汇总所有非 conditional 原分项并保留独立硬证据。`describe` 和查询报告以 `collision_rule='nonpenetration'` 说明实际规则。`last_query_report` 保留 `geometry_scope`、分类计数和有限见证；`nonclear_pairs` 按硬分项各选取物理时间最早与配对残差最大的记录，若两者相同则合并为一条。每条记录同时保留 `state_index`、`time_s`、任务 `progress`、接触许可 `permission_progress` 和选择原因。残差最大只在同一分项内部比较：机器人自身配对使用原生穿透距离，实体场配对使用原有交叠分类值。该显示选择直接消费已有查询证据；连续区间验收的完整 `geometry_evidence_out` 仍覆盖每个状态的每个要求配对。场配对的未决只进入查询报告。原生自身的 `numerical_boundary`、required contact、初始支撑高度与覆盖按各自硬条件计算，conditional 分项只作为条件报告。

区间几何调用显式提供 `geometry_evidence_out`，接收每个状态的完整实体对记录，包含原实体 ID、硬状态、距离范围、许可种类、内部组合体和未建模运动；该记录不受显示报告的有限条数限制。`first/second` 保存显示名称，`geometry_entity_key` 的两个同名字段分别保存 `robot/<geometry ID>`、`owner/<owner ID>` 或 `background`。带类别的实体键连接距离证据、运动界、配对缓存和交叠去重，也用于筛选依赖未建模手指运动的机器人实体；同名机器人组件与任务实体保留独立身份。

原生机器人实体对的分离下界由 `NativePairTable.query` 在同一距离循环中对两个完整形状作支持投影，并采用外向区间运算。方向范数取上界，投影间隙和除法取下界；零方向给出零分离下界。Coal 查询法向只提供投影方向，原生求解容差不被解释为距离误差界。区间检查如何将端点状态的几何证据与运动界结合，见[验收组件](05-acceptance-and-execution.md)。

每次 `components` 调用从完整关节张量和各实体位置张量构造一次形状与进度一致的零值张量。条件未知跳过项、组合体内部跳过项及分类残差复用其中对应样本的标量。机器人自碰撞保留原生 signed distance 及刚体见证导数，初始支撑保留最小高度区间及其见证；这两个有量纲计算保持其原数值语义；新的有限优化接口另对原值与长度尺度构造确定的平方目标。

在每个实际进度样本，场景先以 `Attachment.active(progress)` 确定当前附着实体，再从任务场配对中排除两类内部对：`tool_geometry_names` 中任一工具实体与当前附着实体，以及同一 TCP 上两个当前附着实体。内部对不调用几何查询，`internal_assembly_pair_count` 记录排除的配对样本数。所有成员保留自己的原形状、位姿和观测支持，继续参与对背景、组合体外实体和其余机器人部件的查询。机器人自身仍由原生配对策略处理。有限接触的 `contact_permission_progress` 仅决定任务许可，不改变附着生效时刻。

`fixed_geometry_names` 明确声明安装体身份。安装体与运动 owner 进行同一非穿透查询；活动机器人与安装体由原生配对检查。安装体与静态背景/静态 owner 不增加运动配对。机器人首连杆即使 FK 恒定仍保留其机器人角色。机器人图像覆盖不形成拒绝分项，报告为 `not_queried`；运动 owner 继续使用显式未知空间策略。

## 5. 自由空间与移动实体覆盖

`ObservedSpace` 保存只读 `known[z,y,x]`、世界原点、间距和视图身份。生产 known 来自 `_known_free_voxels`：一个完整闭体素在有效测量射线之前时才计为已知自由。实体移动后，该格网仍在原世界坐标中。

`query_coverage` 检查移动实体已表示占据是否包含于已测自由格与声明条件区域的并集。实际占据见证可证明越出覆盖；框交叠仅用于细分。闭单元边界及多个区域接缝共同参与并集计算。未知区域按 `forbid`、`require_observation` 或 `allow_attempt` 解释。

## 6. 保存与调用关系

`build_scene_from_rgbd → integrate_entity → freeze_mapper` 建立本次新任务场；已有场直接进入 `OnlinePlanningScene`。已知机器人由 `RobotGeometryChecker` 注册，规划器绑定完整 FK 后，在原状态检查和验收中调用同一场景；统一约束求解器向有限目标显式传入相同标定的 `robot_link_poses`，由场景乘固定组件变换并预测所有者关系。

`to_execution_binding` 输出 `native_robot_observed_scene_v1`，保存机器人原形状、基座、标定、完整配对、固定角色、`tool_geometry_names` ，同时保存任务场样本、运动关系、有限任务接触与自由格。任务单场通过 `FrozenObservedField.from_record` 恢复；场景快照用于输入来源和身份交接，执行曲线恢复不等于重新构造可查询场景。


完整实体随曲线运动的区间界由 `interval_geometry.SceneMotionBounds` 从同一已注册 Pinocchio 模型、验证器基座标定、官方实体局部范围和观测 TSDF 范围建立。局部包围球在这里仅提供点速度系数，实体碰撞继续由原形状计算。固定 frame、完整手指和附着关系的输入输出与推导见[候选验收](05-acceptance-and-execution.md)。
