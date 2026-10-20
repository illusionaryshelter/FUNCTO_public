# 观测与碰撞模型

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。本组件将经过标定和对象分区的 RGB-D 转为固定场景，并为候选机器人状态计算碰撞、接触、未知空间和查询范围分项。入口为 `planning_observation.build_scene_from_rgbd`，返回类型为 `online_geometry.OnlinePlanningScene`。

## 1. 观测输入与对象分区

`views` 是从观测身份到视图记录的映射，一次接受 1–16 个视图，总深度像素数不超过 8,000,000。各视图使用相同的任务参考坐标，字段契约如下。

| 字段 | 类型、形状与含义 |
|---|---|
| `image` | `uint8[H,W,3]`，与深度逐像素对齐的 RGB |
| `depth` | `[H,W]` 米制相机深度，`H,W≥3`；不在本次 Mapper 有效量程内的样本在规范副本中置零 |
| `K` | `[3,3]` 整流针孔相机内参；要求零斜切，输入像素中心为整数坐标 |
| `T_output_camera` | `[4,4]` 刚体变换，将相机坐标中的点变到共同参考坐标 |
| `reference_frame` | 与请求相同的非空参考坐标身份 |
| `owner_masks` | 每个已选择 owner 对应一个 `bool[H,W]`；空 mask 也须显式提供，任意两个 owner 的原 mask 不得重叠 |
| `self_filter.labels` | `uint8[H,W]` 三态分类 |
| `self_filter.free_ray_depth_m` | `[H,W]` 非负米制自由射线记录；环境像素保留原深度，其他像素为零 |
| `self_filter.source` | 非空分类来源 |

`self_filter.labels` 中，`ENVIRONMENT` 表示可用于环境几何的深度端点，`SELF` 表示机器人自身，`AMBIGUOUS_SHADOW` 表示机器人附近归属不确定的深度。生产接口要求三态 `self_filter`；历史审计入口使用的二元 `robot_mask` 是另一份输入契约。

`_mapper_intrinsics` 只在创建原生 `CameraObservation` 时复制 `K`，并对两个主点坐标各加 `0.5`。原生 Mapper 的积分射线从半整数像素中心发出，因此这一步保持输入整数像素中心所表示的物理射线。其他反投影与自由射线计算继续使用原 `K`；此变换不移动深度、不扩大材料，也不改变 RGB 特征的坐标约定。

`_self_filtered_views` 完成尺寸、坐标和有效深度范围的标准化。`_partition_masks` 根据明确的 owner masks 将每个有效环境端点分配给一个对象或背景。背景是该请求中固定于世界坐标的几何集合；移动对象的可见端点归入其对象分区。`save_observation_partitions` 保存实际进入建图的背景分区和统计，便于逐像素复查输入。

`owner_frames` 按对象身份提供 `T_output_owner` 与来源。每个对象在自己的固定表示坐标中融合深度；背景在共同参考坐标中融合。`task_owner_ids` 明确给出被操作物体和功能目标对象，`observation_resolutions` 据此分配任务对象 2 mm 的体素间距，其他对象使用配置中的环境间距。显式 `fine_regions` 可以为一个分区选择固定范围与间距：任务对象范围为 1–2 mm，环境分区保持比 2 mm 更粗。一次请求中，每个分区使用一个确定的场。

`complete_owners` 是另一个显式输入，包含已经建立的完整 `CollisionOwner`。建图函数保留这些对象的语义 masks 和射线证据，并直接采用其查询模型。当前旧放置适配器通过此参数传入长方体杯体；报告中的 `direct_complete_owners` 与 `mapped_partition_estimates_bytes` 分别列出直接采用的模型和实际分配 Mapper 的分区。

## 2. 从 Mapper 到 TSDF 几何

`_map_online_partition` 为每个需要观测建图的分区创建 cuRobo `Mapper`，按相机标定和对象坐标融合其深度端点。`build_tsdf_geometry` 从 Mapper 的动态 TSDF 块读取加权距离和权重，依据原生块坐标及有效权重规则得到 CPU 上的 `sdf` 与 `observed` 数组，然后调用 `tsdf_geometry_from_samples`。

`tsdf_geometry_from_samples(sdf, observed, origin, voxel_size, ...)` 以采样中心为格点。每个八角单元采用一致的六四面体划分，四面体的四个采样点均有观测时，线性插值定义其中的场值。场的零截面形成表面组件，负值部分形成凸体积组件；全负八角单元使用紧凑整数索引保存。孤立的非正观测点保留为点组件。输出是 `solids`、`query_gaps` 与 `field_summary`，其中每个 TSDF solid 保存：

| 字段 | 数据语义 |
|---|---|
| `origin`、`voxel_size`、`shape_xyz` | 原采样中心网格的位置、间距和三维尺寸 |
| `surface_vertices/surface_offsets` | 零截面顶点及压缩行式组件边界 |
| `negative_vertices/negative_offsets` | 仿射负值部分的顶点及组件边界 |
| `negative_cell_flat_indices` | 全负单元的原网格索引 |
| `pose`、`name`、`source` | 表示坐标、模型身份和来源 |

这一记录保存的是 TSDF 所定义的几何组件。`FrozenObservedGeometry` 将记录解码为只读数组，为各组件建立闭合轴对齐包围盒树 `ComponentAABBTree`，并按需缓存 Coal 形状。`component_indices(bounds, active_rule_ids)` 返回指定局部包围盒内、经当前接触许可筛选后保留的组件身份；`collision_component(id)` 返回该组件的类型、局部索引、原生形状、局部位姿和顶点。所有场景副本共享这些固定源数组与索引。

## 3. 保守距离场与球查询

`FrozenObservedGeometry.distance_field` 将保留组件的包围盒覆盖到一个指定规则网格，使用 SciPy 的欧氏距离变换计算候选场。源组件包围盒所覆盖单元之外的距离扣除单元与插值误差界，覆盖内部构造负候选值；输出为向下取整的 float32 数组。其作用是给球查询提供源距离的保守界，并快速定位需要进一步检查的几何对。

`FrozenESDF.from_source` 将上述场、源几何、网格坐标和 owner 身份组织成查询对象。虽然类名沿用 ESDF，当前 `distance` 字段的具体含义是由源组件包围盒生成的候选距离场。`signed_distance(points)` 使用 Torch 三线性插值，输入 `[...,3]` 局部点，输出 `[...]` 场值。`__call__(spheres)` 对 `[...,4]` 的局部球数组计算 `relu(radius−field(center))`，返回每个球的非负候选冲突量。

`intersects_geometry` 检查球是否可能与源几何整体包围盒相交；`supports_centres` 检查球心处是否具有数值查询支持；`supported` 检查整个球的影响范围。查询域与观测自由域分别保存。`FrozenESDF` 在支持范围外给出相应的范围状态，由场景的 `scene_update/...` 分项表达需要补充模型范围的位置。

`FrozenPrimitives` 提供同一查询接口，输入为明确基本体及局部位姿，支持盒、球、圆柱和胶囊等表示。`signed_distance` 使用相应解析几何，`prepare_contacts` 在明确接触区域建立查询数据。其 `record` 保存基本体参数；`FrozenESDF.record` 保存候选距离数组、TSDF 源组件、坐标和已准备的接触字段。`from_record` 根据保存的 schema 恢复对应对象。

## 4. 已观测自由空间与未知策略

`_known_free_voxels` 依据有效环境射线、深度误差和原背景网格确定自由体素。`ObservedSpace` 保存这些体素、原点、间距、观测身份和参考坐标。未知空间即当前自由证据尚未覆盖的空间，其状态独立于候选距离场的正负。

`unknown_policy` 包含有限的 `conditional_regions` 与 `outside_regions`。每个条件区域提供明确下界、上界和来源；区域外策略为 `forbid`、`require_observation` 或 `allow_attempt`。`_prepare_unknown_coverage` 预先组合可观察自由域与明确允许尝试区域，`_unknown` 对整个查询球计算禁止与条件尝试分项。场景保留原始 `known_free`，策略只决定该请求怎样使用未知区域。

`_background_query_config` 根据机器人球和夹爪运动包络准备足够的数值查询范围；`_validate_background_query_support` 检查生成结果。自由射线仍使用原背景覆盖网格。一个对象之后发生刚体运动时，其局部几何可按对象位姿变换；世界中的射线证据仍属于原观测位置。

## 5. 场景批量查询的数据流

`OnlinePlanningScene.components` 的主要输入为：

| 参数 | 形状与含义 |
|---|---|
| `robot_spheres` | `[B,H,S,4]`，同一候选批次和时间上的机器人球，最后维为中心与半径 |
| `position/quaternion` | `[B,H,3]`、`[B,H,4]`，任务参考坐标中的工具位姿 |
| `progress` | `[B,H]`，与状态对应的技能进度 |
| `joint_position/joint_names` | 具名关节位置及其列顺序，供手指与相关状态计算 |
| `trajectory_time` | `[B,H]`，候选实际物理时刻 |
| `contact_permission_progress` | 接触许可使用的进度；通常与主进度一致，边界检查可分别给出 |
| `active_contact_rule_ids` | 显式状态查询所启用的规则集合；默认规划使用声明区间 |
| `owner_poses` | 显式给定的对象表示坐标位姿映射；普通规划从附着/路径计算 |
| `native_confirmation` | 非梯度决策调用使用的实体复核器 |
| `collision_paths_out` | 调用者接收逐球路径数据的字典，供碰撞优化目标计算 |

函数首先根据 `PrismaticGripper.state` 放置手指球并分离已知与条件手指几何，再计算各对象的位姿。机器人始终是查询作用方；`moving=True` 的物体通过 `local_spheres` 形成另一个作用方。每个作用方分别查询世界背景和其他对象；相对于对象的球坐标由其位姿逆变换得到。

返回字典按语义命名，通常每项为 `[B,H]`：

| 分项族 | 具体含义 | 消费方式 |
|---|---|---|
| `collision/...` | 机器人自碰撞或某作用方与目标的候选冲突量 | 优化引导；具备登记关系时可进入实体复核 |
| `contact/...` | 指定材料区域的法向、侧边、背面、间隙与必需接触条件 | 任务接触约束 |
| `gripper/controller` | 手指状态与已绑定控制模型的关系 | 硬约束 |
| `unknown/forbidden/...` | 查询几何进入当前策略禁止的未知空间 | 资格判断与相应搜索恢复 |
| `conditional/...` | 明确允许尝试的未知空间或条件手指几何 | 独立报告通道 |
| `scene_update/...` | 与已有源几何可能相交但缺少数值查询支持 | 场景范围不足状态 |
| `unsupported/...` | 当前表示或查询所需结构不成立，例如不完整运动物体 | 当前能力与输入结构限制 |
| `initial_support/depth/...` | 名义物体相对初始支撑基准的向下越界 | 独立初始支撑条件 |

`evaluate` 汇总除 `conditional/` 外的硬分项。`query_state` 将一个状态整理为可读报告，保留各分项与判定。`clone_for_rollout` 创建请求或候选私有的可变查询状态，共享固定几何。`validate_problem` 将任务条件、接触声明、对象和事件身份相互绑定。`bind_online_robot` 补充具名球所属实体、机器人碰撞对、原生模型和固定覆盖半径。

## 6. 保守候选与源实体复核

`NativeCollisionConfirmation` 在构造时读取机器人已登记实体、完整运动基本体以及各目标的 TSDF 源组件或基本体。`confirm` 输入某一状态中的正候选球索引、已计算的对象位姿、有效接触规则和初始支撑关系，输出逐实体对记录，以及 `cleared_targets`、`collision_targets`、`unresolved_targets`。

对于 TSDF 目标，复核器先将作用方实体变换到目标坐标，再用实体包围盒选择源组件，最终通过 Coal 测量这些有限组件的有符号距离。对于普通基本体目标，使用其解析原生形状；启用有限接触时使用相应准备结果。接触许可过滤的是同一源组件身份。数值边界附近归入 `unresolved`，确定负间距归入 `collision`，全部相关实体对分离时归入 `clear`。

`OnlineCollisionConfirmation` 将以上结果作用到同一次候选查询的 `collision` 分项。`confirm_self` 用候选具名关节和已知机器人模型处理正自碰撞代理。其他接触、未知、查询域和输入结构分项继续由各自计算结果决定。

复核调用发生在初始状态诊断、功能/搜索状态检查和完整候选确认等非梯度边界。优化 rollout 消费可微球代理和候选距离场。`select_candidate` 对仅因可复核碰撞分项未通过的既有候选进行完整状态确认，再决定该候选资格。两种计算共享源几何和接触定义，分别承担梯度引导与离散实体判定。

## 7. 保存、恢复与独立审计

`OnlinePlanningScene.to_execution_binding` 保存 `online_scene_v3`，包括背景、owners、各 owner 的位姿路径和附着、球/材料身份、接触字段、未知策略、自由覆盖、夹爪名义几何、初始支撑准备数据与观测报告。`from_binding` 解码这些字段，恢复只读源组件并重新连接规则与机器人身份；`binding_record` 是保存函数的别名。

`collision_records.encode_array/decode_array` 用显式 dtype、shape 和有界二进制编码保存数组，解码结果为只读 NumPy 数据。`resource_admission` 在大数组、Mapper、索引和原生缓存分配前计算所需存储，并结合当前进程/设备占用判定资源容量。`estimated_workspace_allocation` 将批次、时间长度、对象球数、接触字段与可选路径目标的工作区分别计入。

独立审计路径由 `build_audit_scene_from_rgbd`、`planning_scene.PlanningScene`、`ObservedTsdfGeometry` 和 `RobotGeometryChecker.query_scene` 组织，提供源组件遍历、精细层选择与完整场景距离证据。`scene_surface_batch` 和 `tsdf_spatial_query` 对这些 CPU 查询进行有界调度。其结果由审计报告消费。生产建图与在线复核使用前述固定模型与候选局部复核过程。
