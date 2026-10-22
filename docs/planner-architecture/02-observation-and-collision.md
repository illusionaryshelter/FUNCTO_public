# 观测与碰撞模型

代码基线：`a56c3e4`（A–D 实际实现提交；本次仅文档更新提交引用，所述代码与该实现提交一致）。

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

同一场维护占据单元与局部零面候选两套索引。占据树保留全部可能包含非正值的完整支持单元；`_zero_tree` 使用现有 SciPy `cKDTree` 索引满足 `min≤0≤max` 且 `min<max` 的单元中心。`local_zero` 在最近的最多八个此类单元中，以原三线性零面为等式、单元为范围，求局部零点及非零梯度方向。全负内部格和常值零格继续参与占据计算，局部方向查询只由具有零面方向的原单元提供。返回的是有限局部候选中的方向记录。

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

优化块开始时，`pair_local_model` 从实际基准状态的交叠证据取得一个材料点，并在另一个原观测场中求局部零点与法向。令点 `x` 属于实体 A，零点 `z` 和单位法向 `n` 属于实体 B；两个实体当前位姿为 `(R_A,p_A)`、`(R_B,p_B)`。`LocalPairModel.value` 计算 `d=(R_Ax+p_A−R_Bz−p_B)·R_Bn` 和 `f=ℓ·softplus(−d/ℓ)`，其中 `ℓ>0` 是提供零面的场的样本间距，`softplus(a)=log(1+exp(a))`。材料点、零点、局部法向和尺度在该块内固定，所有试探位姿都对这个相同标量求导，包含法向随 B 旋转的导数。无法取得局部零面时，硬证据照常保留，记录方向不可用。

`LocalGeometryModels` 按种子、物理时刻、实体关系进度和接触许可进度保存上述记录。一个种子的全部线搜索粒子共享模型。试探时重新计算实际硬证据，局部模型在下一个实际基准状态才刷新；已分离试探仍可具有正 softplus 值，新出现的交叠仍具有原硬分类。`LocalContactModel` 另把一个固定目标零点相对声明有限区域的半空间超差范数作为 required contact 的连续目标。原 `query_region_intersection` 以支持零表面判断接触成立、缺失或未决。

模型捕获和 `repeated_base_evaluation()` 作用域中的输入按种子复制同一个实际基准状态。场景在单次调用内以查询列、`(seed,time,relation_progress,permission_progress)` 及实体对、接触或覆盖身份保存数值查询记录；重复粒子读取相同记录，分别通过自身位姿和关节张量构造可微标量。不同查询列保留独立身份。普通线搜索试探各自完成原几何查询。`PhysicalGradientCore._refresh_local_geometry` 返回本次捕获的轨迹计算与模型变化标记，块边界从这份计算读取硬证据、固定排名目标和功能残差；目标改变后重新计算该基准状态的梯度。报告中的 `pair_queries` 统计逻辑实体对证据数量，数值查询复用不改变该计数的含义。

## 4. 已知机器人与原任务场的组合

`RobotGeometryChecker` 用 URDF 建立 Pinocchio 关节树，以 `NativeSolid` 原凸组件顶点或 `NativePrimitive` 解析盒体/圆柱建立 Coal 碰撞实体。模型保留实体 ID、所属连杆、固定变换与官方来源。完整具名关节及基座标定确定各实体位姿。该模型对应当前固定机器人及其官方固定末端配置，可跨任务复用；任务物体、环境和其他对象的 mesh/CAD 不进入该许可范围。

`query` 返回原生有符号距离、最近点、法向、FK 和关节域。`required`、`excluded`、`mechanical_contact` 是调用者明确提供的原模型配对语义。自碰撞使用 `separated`、`penetrating`、`numerical_boundary` 区分原生查询结果，保留 Coal 默认数值边界；该数值尺度不等于严格误差界，数值边界单列未决。

`RobotStateValidator.required_pairs` 是原配对策略中所有 `required` 记录组成的元组。`OnlinePlanningScene.components` 将该元组直接传给 `RobotGeometryChecker.query`，取得当前自碰撞条件需要的实体对证据；完整 `robot_pairs` 继续用于模型描述与快照保存。实体身份、几何形状和必检配对集合由原机器人配置确定。

`DifferentiableRobotKinematics` 用 cuRobo 完整关节树求连杆位姿，包括本时刻名义手指坐标。场景对解析实体乘固定局部变换；凸顶点已在连杆坐标中，局部变换为单位阵。机器人模型与状态验证器共享同一个检查器。

`native_field_distance.query_native_field` 直接查询原生实体与原任务场。原实体的支持投影及任务空间框产生距离下界；原实体内可行点及原场占据点产生上界。凸体内点由原面顶点和原顶点均值的非负组合构造，区间计算保留该实际点的局部与世界范围。原生实体保持其有体积的凸体或解析形状；若其实际可行点完整区间严格位于原支持单元的开内部，且该范围上三线性值可靠为负，或该原单元八角点全部非正，即有原场占据开邻域与原生实体发生体积交叠。未许可的这类见证产生 `intersection`。`witness_*_bounds` 保存点的数值区间，中点仅用于显示和局部导数。结论针对已知机器人模型与已表示任务场，不输出真实物理穿透深度。

通常仅细分任务场；点对许可跨越边界时，局部搜索还细分原生实体与搜索盒的交集。数值精度停止条件同时考虑当前许可域所需的细分尺度；薄域停止仍要求本域实际点对上界不超过精度，且空间框交集至少一个轴向厚度不超过精度。局部未决不结束其余域的检查。空间框交叠只用于安排计算，不能代替原实体归属或体积交叠证据。

`query_native_field(..., native_bounds=(local,world))` 可直接消费同一原生实体的局部与世界投影范围。场景构造时保存不可变形状的局部范围；`components` 在本次调用内按实体和状态计算一次世界范围，供不同场配对复用。空场无需构造对应世界范围。required contact 使用的作用实体矩阵也在整批转换一次，然后按状态读取。这些范围和矩阵只复用相同几何计算，实体查询仍使用原形状和原任务场。

`OnlinePlanningScene.components` 计算原生机器人自身、机器人与观测场、运动实体与背景、实体之间的配对，返回相同硬分项。优化调用另外提供 `local_models`，由 `objectives_out` 返回固定局部模型和其他连续场景项。`evaluate` 在 acceptance 模式汇总硬分项，在 optimization 模式汇总优化项；两者均保留独立硬证据。`describe` 和查询报告以 `collision_rule='nonpenetration'` 说明实际规则。`last_query_report` 保留分类计数和有限见证；场配对的未决只进入该报告。原生自身的 `numerical_boundary`、required contact、初始支撑高度与覆盖按各自硬条件计算，conditional 分项只作为条件报告。

区间几何调用显式提供 `geometry_evidence_out`，接收每个状态的完整实体对记录，包含原实体 ID、硬状态、距离范围、许可种类、内部组合体和未建模运动；该记录不受显示报告的有限条数限制。`first/second` 保存显示名称，`geometry_entity_key` 的两个同名字段分别保存 `robot/<geometry ID>`、`owner/<owner ID>` 或 `background`。带类别的实体键连接距离证据、运动界、配对缓存和交叠去重，也用于筛选依赖未建模手指运动的机器人实体；同名机器人组件与任务实体保留独立身份。

原生机器人实体对的分离下界由 `native_pair_lower_bound` 对完整形状作支持投影，并保留数值范围。Coal 查询法向只提供投影方向，原生求解容差不被解释为距离误差界。区间检查如何将端点状态的几何证据与运动界结合，见[验收组件](05-acceptance-and-execution.md)。

每次 `components` 调用从完整关节张量和各实体位置张量构造一次形状与进度一致的零值张量。条件未知跳过项、组合体内部跳过项及分类残差复用其中对应样本的标量。机器人自碰撞保留原生 signed distance 及刚体见证导数，初始支撑保留最小高度区间及其见证；这两个有量纲计算与固定局部零面模型分别提供连续优化项。

在每个实际进度样本，场景先以 `Attachment.active(progress)` 确定当前附着实体，再从任务场配对中排除两类内部对：`tool_geometry_names` 中任一工具实体与当前附着实体，以及同一 TCP 上两个当前附着实体。内部对不调用几何查询，`internal_assembly_pair_count` 记录排除的配对样本数。所有成员保留自己的原形状、位姿和观测支持，继续参与对背景、组合体外实体和其余机器人部件的查询。机器人自身仍由原生配对策略处理。有限接触的 `contact_permission_progress` 仅决定任务许可，不改变附着生效时刻。

`fixed_geometry_names` 明确声明安装体身份。安装体与运动 owner 进行同一非穿透查询；活动机器人与安装体由原生配对检查。安装体与静态背景/静态 owner 不增加运动配对。机器人首连杆即使 FK 恒定仍保留其机器人角色。机器人图像覆盖不形成拒绝分项，报告为 `not_queried`；运动 owner 继续使用显式未知空间策略。

## 5. 自由空间与移动实体覆盖

`ObservedSpace` 保存只读 `known[z,y,x]`、世界原点、间距和视图身份。生产 known 来自 `_known_free_voxels`：一个完整闭体素在有效测量射线之前时才计为已知自由。实体移动后，该格网仍在原世界坐标中。

`query_coverage` 检查移动实体已表示占据是否包含于已测自由格与声明条件区域的并集。实际占据见证可证明越出覆盖；框交叠仅用于细分。闭单元边界及多个区域接缝共同参与并集计算。未知区域按 `forbid`、`require_observation` 或 `allow_attempt` 解释。

## 6. 保存与调用关系

`build_scene_from_rgbd → integrate_entity → freeze_mapper` 建立本次新任务场；已有场直接进入 `OnlinePlanningScene`。已知机器人由 `RobotGeometryChecker` 注册，规划器绑定完整 FK 后，在 IK、轨迹求解及验收中调用同一场景。

`to_execution_binding` 输出 `native_robot_observed_scene_v1`，保存机器人原形状、基座、标定、完整配对、固定角色、`tool_geometry_names` 和运动学配置，同时保存任务场样本、运动关系、有限任务接触与自由格。任务单场通过 `FrozenObservedField.from_record` 恢复；场景快照用于输入来源和身份交接，执行曲线恢复不等于重新构造可查询场景。
