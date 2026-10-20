# 观测与碰撞模型

代码基线：`__IMPLEMENTATION_COMMIT__`。

## 1. 两种实体来源

`build_scene_from_rgbd` 把新观测与已有模型组成一个 `OnlinePlanningScene`。`owner_frames` 仅包含本次按 RGB-D 建立的实体，给出局部坐标和来源；`observed_owners` 提供已有 `CollisionOwner`，保留其场对象、原样本、分辨率、观测支持及声明运动。同一身份选择一种来源。任务对象身份在两组实体的并集中解析。

新观测视图包含 `image:uint8[H,W,3]`、相机 z 向米制 `depth`、零斜切内参 `K`、`T_output_camera`、参考坐标、各新实体的布尔 `owner_masks` 和机器人过滤结果。`_partition_masks` 只验证新建实体的像素分区。已保存实体不进入可见深度计数。

静态背景可直接通过 `background` 传入，已测自由空间通过 `coverage` 传入。两者和所有实体均已给定时，视图可为空，构造过程不调用 Mapper。需要建立背景时，`background_mask` 明确选择该幅观测中的静态端点；普通首次建图也可用新实体 mask 的补集。在含有独立复用实体的观测中，静态分区由调用者明确提供。原观测中未映射的端点记录为 `unmapped_pixels`，不被解释为新增自由空间。

## 2. 深度过滤与场的形成

`robot_self_filter.classify_depth` 消费采集时完整机器人关节反馈的 FK、相同连杆观测场及其 TSDF 块。受支持的严格负值端点标为 SELF；原生 `Mapper.render` 的预测命中先经原场支持复核，遮挡射线标为 AMBIGUOUS_SHADOW；其余有效测量标为 ENVIRONMENT。只有 ENVIRONMENT 保留原测量深度作为自由射线。缺失连杆观测单独记录。

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

## 4. 机器人与场景查询

`RobotGeometryChecker` 从纯运动学 URDF 建立 Pinocchio 模型，查询关节范围、独立 FK 和注册连杆场的成对距离。`DifferentiableRobotKinematics` 使用 cuRobo 完整关节树，包含本时刻的手指名义坐标。`ObservedBody` 保留连杆身份、固定刚体分组、原场或明确缺失状态。

`OnlinePlanningScene.components` 取得机器人连杆位姿、各 `CollisionOwner.pose`、手指时钟状态和接触许可，再查询需要检查的机器人自身、机器人与场景、运动实体与背景以及实体之间的组合。`query_role='optimization'` 使用 1 mm；`acceptance` 使用 2 mm。`last_query_report` 保存分类计数和有限见证记录。`evaluate` 聚合硬分项，conditional 分项保留为报告。

机器人位置越出相机视野不产生拒绝项：`robot_unknown_policy='allow_attempt'`、`robot_view_coverage_status='not_queried'`。已有机器人观测场仍参与碰撞。缺失观测不被补成占据。

## 5. 自由空间与移动实体覆盖

`ObservedSpace` 保存只读 `known[z,y,x]`、世界原点、间距和视图身份。生产 known 来自 `_known_free_voxels`：一个完整闭体素在有效测量射线之前时才计为已知自由。实体移动后，该格网仍在原世界坐标中。

`query_coverage` 检查移动实体已表示占据是否包含于已测自由格与声明条件区域的并集。实际占据见证可证明越出覆盖；框交叠仅用于细分。闭单元边界及多个区域接缝共同参与并集计算。未知区域按 `forbid`、`require_observation` 或 `allow_attempt` 解释。

## 6. 保存与调用关系

`build_scene_from_rgbd → integrate_entity → freeze_mapper` 建立新场；已有场直接进入 `OnlinePlanningScene`。规划器绑定完整机器人 FK，并在 IK、轨迹求解及验收中调用同一场景。`to_execution_binding` 输出 `observed_scene_v1`，保存场样本、机器人模型、实体运动、任务接触、原自由格及来源。当前场景快照用于保存输入身份；单体恢复由 `FrozenObservedField.from_record` 和 `ObservedBody.from_record` 实现。
