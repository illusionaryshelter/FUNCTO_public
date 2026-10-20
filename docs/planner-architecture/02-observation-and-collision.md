# 观测与碰撞模型

代码基线：`__NATIVE_MODEL_COMMIT__`。

## 1. 两种实体来源

`build_scene_from_rgbd` 把新观测与已有模型组成一个 `OnlinePlanningScene`。`owner_frames` 仅包含本次按 RGB-D 建立的实体，给出局部坐标和来源；`observed_owners` 提供已有 `CollisionOwner`，保留其场对象、原样本、分辨率、观测支持及声明运动。同一身份选择一种来源。任务对象身份在两组实体的并集中解析。

新观测视图包含 `image:uint8[H,W,3]`、相机 z 向米制 `depth`、零斜切内参 `K`、`T_output_camera`、参考坐标、各新实体的布尔 `owner_masks` 和机器人过滤结果。`_partition_masks` 只验证新建实体的像素分区。已保存实体不进入可见深度计数。

静态背景可直接通过 `background` 传入，已测自由空间通过 `coverage` 传入。两者和所有实体均已给定时，视图可为空，构造过程不调用 Mapper。需要建立背景时，`background_mask` 明确选择该幅观测中的静态端点；普通首次建图也可用新实体 mask 的补集。在含有独立复用实体的观测中，静态分区由调用者明确提供。原观测中未映射的端点记录为 `unmapped_pixels`，不被解释为新增自由空间。

背景积分保留一圈像素身份不确定带。令 `E` 为有效 ENVIRONMENT 像素，`B` 为上述背景候选 mask；边缘带为 `E & B & binary_dilation(E & ~B, 3×3, 一次)`，最终背景为 `E & B` 排除该带。该规则同时覆盖新建实体与原静态分区所排除的已有实体，原实体 mask 不扩张。SELF、AMBIGUOUS_SHADOW 和无效深度不产生边缘种子。被排除端点计入现有 `unmapped_pixels`，自由空间仍由独立深度射线计算。这是一像素的分割边界保护，不是任意分割误差的几何上界。

`scene.observation_partition_masks[view]` 独立保留三种数据：`owner_masks` 是本次传入的新建实体原始语义分区，`background_candidate` 是过滤前的背景候选 `B`，`background` 是实际积分背景。前两者复制为只读数组，保留被机器人过滤排除的原语义像素。显式背景候选直接保存；首次新建背景而未给候选时，保存原实体 mask 的补集。复用背景时，实际积分背景为 `None`；此时仅保存明确提供的候选。全部模型复用且无需视图时，记录为空。

`save_observation_partitions` 按视图写出实体原始 mask、背景候选和实际积分背景，实体 ID 与路径对应关系存入 `partition-report.json`。再次建图时，`--owner-mask` 读取实体原始文件，`--background-mask` 读取 `background-candidate-mask.npy`，从而只执行一次边缘排除。已有实体继续复用自己的模型及原有记录。

## 2. 深度过滤与场的形成

`robot_self_filter.classify_depth` 使用图像采集时具名关节 FK 放置同一已知机器人组件。端点严格位于名义实体内部时标为 SELF；其余端点对应的相机至测量点有限线段若与明确 padding 后的实体相交或处于数值边界，标为 AMBIGUOUS_SHADOW；其余有效测量标为 ENVIRONMENT。只有 ENVIRONMENT 保留原测量深度作为自由射线。padding 定义过滤阴影域，规划碰撞实体保留原形状。

`integrate_entity` 接收 RGB、深度、实体 mask、K 和 `T_entity_camera`，逐实体串行融合。`_mapper_intrinsics` 将主点增加半像素，以适配 Mapper 的像素中心约定。`freeze_mapper` 读取加权 TSDF 与权重，保存 XYZ 排列的 float32 场值；`origin` 为首个样本中心。最小有效权重确定样本观测支持。

`FrozenObservedField.record/from_record` 保存和恢复原始值、权重、支持、原点、间距、身份与来源；空间索引由这些样本重建。`clone_for_rollout` 返回同一不可变场。任务实体新建时采用 2 mm 间距，环境采用明确给定的更粗间距；已有场保留原间距。资源准入按串行 Mapper 峰值及当前已驻留数据计算。

## 3. 同一原始场上的距离

`ObservedField` 在八角点均有支持的闭单元内作三线性插值，以 `φ≤0` 定义占据。`query_distance` 对两个场及其刚体变换进行分支细分。变换后的空间框产生距离下界；属于原占据集合的点对产生上界；公共的受支持严格负值点证明相交。浮点误差进入边界计算。初始化按1024个单元分批计算同一边界收紧公式，单元内部仍依次处理三个坐标轴；标量细分与批量构造共享值及误差计算。空间树与原始场保持一致。

| 状态 | 几何含义 |
|---|---|
| `clear` | 查询域中的占据集合满足要求间距；空集合另带 `empty_set` |
| `margin_insufficient` | 实际占据见证点对的距离上界小于要求间距 |
| `intersection` | 同一个受支持点位于双方严格负值内部 |
| `unresolved` | 在给定数值精度和支持条件下尚不能确定所需结论 |

`PairPermission` 描述允许的点对域。完全许可的框对被排除，其余继续查询；混合许可框采用按进入顺序展开，明确未许可框优先细化，以避免许可边界阻塞其他实体证据。该顺序不改变场、阈值或许可范围。

`pair_residual` 将上述状态转换为可微残差。分离见证点对给出刚体运动方向；严格相交时使用同一场的局部零截面法向给出优化方向。方向不改变几何分类，也不作为穿透深度估计。未决状态单列分项。

## 4. 已知机器人与原任务场的组合

`RobotGeometryChecker` 用 URDF 建立 Pinocchio 关节树，以 `NativeSolid` 原凸组件顶点或 `NativePrimitive` 解析盒体/圆柱建立 Coal 碰撞实体。模型保留实体 ID、所属连杆、固定变换与来源。完整具名关节及基座标定确定各实体位姿。官方随附、跨任务复用的机器人模型是已知本体输入；任务物体继续使用 RGB-D 原场，不要求逐个物体人工制作 mesh。

`query` 返回原生有符号距离、最近点、法向、FK 和关节域。`required`、`excluded`、`mechanical_contact` 是调用者明确提供的原模型配对语义。自碰撞使用 `separated`、`penetrating`、`numerical_boundary` 区分原生查询结果，保留 Coal 默认数值边界；该数值尺度不等于严格误差界，数值边界单列未决。

`DifferentiableRobotKinematics` 用 cuRobo 完整关节树求连杆位姿，包括本时刻名义手指坐标。场景对解析实体乘固定局部变换；凸顶点已在连杆坐标中，局部变换为单位阵。机器人模型与状态验证器共享同一个检查器。

`native_field_distance.query_native_field` 直接查询原生实体与原任务场。原实体的支持投影及任务空间框产生距离下界；原实体内可行点及原场占据点产生上界。凸体内点由原面顶点和原顶点均值的非负组合构造，区间计算保留该实际点的局部与世界范围。只有原生点完整区间位于原支持单元且三线性值严格为负，才确认两集合相交。见证中点仅用于显示和局部导数；`witness_*_bounds` 保存实际点的数值区间。相交结论不输出物理穿透深度。

普通距离仅细分任务场。点对许可跨越边界时，局部搜索同时细分原生实体与搜索盒的交集。达到指定空间精度，或距离上下界已在计算精度内夹住阈值时，保留 `unresolved`。空间框交叠只触发继续计算。

`OnlinePlanningScene.components` 计算原生机器人自身、机器人与观测场、运动实体与背景、实体之间的配对。任务场景查询在优化使用 1 mm，验收使用 2 mm；原生自身检查消费原结构配对和数值边界。`last_query_report` 保存每种分类及有限见证。`evaluate` 聚合硬分项，conditional 分项作为报告保留。

`fixed_geometry_names` 明确声明安装体身份。安装体与运动 owner 查询任务间距；活动机器人与安装体由原生配对检查。安装体与静态背景/静态 owner 不增加运动配对。机器人首连杆即使 FK 恒定仍保留其机器人角色。机器人图像覆盖不形成拒绝分项，报告为 `not_queried`；运动 owner 继续使用显式未知空间策略。

## 5. 自由空间与移动实体覆盖

`ObservedSpace` 保存只读 `known[z,y,x]`、世界原点、间距和视图身份。生产 known 来自 `_known_free_voxels`：一个完整闭体素在有效测量射线之前时才计为已知自由。实体移动后，该格网仍在原世界坐标中。

`query_coverage` 检查移动实体已表示占据是否包含于已测自由格与声明条件区域的并集。实际占据见证可证明越出覆盖；框交叠仅用于细分。闭单元边界及多个区域接缝共同参与并集计算。未知区域按 `forbid`、`require_observation` 或 `allow_attempt` 解释。

## 6. 保存与调用关系

`build_scene_from_rgbd → integrate_entity → freeze_mapper` 建立本次新任务场；已有场直接进入 `OnlinePlanningScene`。已知机器人由 `RobotGeometryChecker` 注册，规划器绑定完整 FK 后，在 IK、轨迹求解及验收中调用同一场景。

`to_execution_binding` 输出 `native_robot_observed_scene_v1`，保存机器人原形状、基座、标定、完整配对、固定角色和运动学配置，同时保存任务场样本、运动关系、接触与自由格。任务单场通过 `FrozenObservedField.from_record` 恢复；场景快照用于输入来源和身份交接，执行曲线恢复不等于重新构造可查询场景。
