# 已知机器人实体与原生距离

代码基线：`TASK_NONPENETRATION_CODE_COMMIT`。

`RobotGeometryChecker` 使用 Pinocchio 解释具名关节树，使用 Coal 查询显式注册的凸实体、盒体和圆柱。输入模型是同一机器人跨任务复用的本体模型。实体形状不依赖当前相机可见范围。坐标变换 `T_A_B` 将 B 中的点映到 A，长度以米计。

`NativeSolid(name, frame, vertices_frame, source)` 的顶点已在连杆 frame 中，形状局部变换为单位阵。`NativePrimitive(name, frame, shape_kind, dimensions, T_frame_geometry, source)` 保留解析形状和固定局部变换；box 尺寸为三轴全长，cylinder 尺寸为半径和轴向全长。每个原实体分别注册。

`PairQuery(first, second, policy, source)` 指定两个实体及原模型策略。`required` 对必须分离；`mechanical_contact` 是这两个实体的显式机械许可；`excluded` 保存原生过滤；`diagnostic` 仅记录。所有策略均保留原查询证据。

| API | 输入与计算 | 输出及消费者 |
|---|---|---|
| `__init__(urdf_path, *, base_frame, geometries)` | URDF、基座身份及上述实体；编译 Pinocchio/Coal 模型 | 关节范围、单一 `urdf_xml` 快照、实体及来源；场景与验证器共享此实例 |
| `geometry_names / geometry_sources` | 已注册实体 | 有序身份及来源 |
| `model_record()` | 已消费的模型 | 独立副本，保存 URDF、原顶点或解析尺寸/局部变换；场景及规划绑定 |
| `geometry_depending_on(joint_names)` | 指定关节的原树依赖 | 受这些关节影响的实体身份；名义手指不确定状态 |
| `query(*, joint_names, joint_positions, T_reference_base, pairs, frames=(), check_request=None)` | 完整具名关节、基座标定、配对和额外帧；FK 后合成各实体位姿 | 限位、帧与实体位姿、原生有符号距离、见证点、法向和 relation；验证器与场景 |

`query` 的 relation 为 `separated`、`penetrating` 或 `numerical_boundary`，采用原 Coal GJK/EPA 数值尺度，且明确记录该尺度并非严格距离误差界。机器人自身的必检配对继续要求 `separated`；数值边界按该独立结构契约处理，没有并入任务场配对只报告未决的规则。非有限查询结果作为计算错误保留。

`_native_pair_distances` 按原配对顺序分块，每块最多128对；每次恢复冷求解状态。`_frame_id` 解析具名帧，`_convex_solid` 将一个原凸碰撞组件编译为 Coal 形状；`_name`、`_rigid_pose` 校验输入坐标与身份。`native_query_storage_nbytes` 为原生查询工作区提供资源估计。
