# 采集时机器人身份与遮挡过滤

代码基线：`b32dfbe5108be41ddb2d866c5813e6ff391e2918`。

`RobotCollisionShape(name, kind, T_world_geometry, source, dimensions=(), vertices=None)` 保存单个原生凸组件或解析盒体/圆柱在图像采集时的位姿。凸顶点在实体局部坐标中；解析尺寸约定与机器人检查器一致。缩放围绕该实体局部原点。

`classify_depth(*, observed_depth, K, T_world_camera, geometries, padding_scale, padding_offset_m, source)` 读取原相机 z 深度、内参与相机世界位姿，并查询显式实体。测量端点在名义本体实体内时为 `SELF`；相机到该端点的有限射线与带指定 padding 的模型相交时为 `AMBIGUOUS_SHADOW`；其余有效端点为 `ENVIRONMENT`。padding 只服务遮挡分类，名义 SELF 与碰撞模型保持原形状。无效深度保留不可用于建图的状态。

输出 `self_filter` 包含 uint8 标签、`free_ray_depth_m` 与来源，另有 provenance 说明模型和参数。只有 ENVIRONMENT 像素保留完全相同的原测量深度，其余自由射线深度为零。`validate_self_filter(value, depth_m)` 核对三态布局及该原深度契约，供场景构造读取。

`_shape` 用 Coal 构造名义或缩放后的原形状，`_world_bounds` 提供查询候选框，`_segment_rotations` 放置有限射线段。框用于减少原生查询，最终分类消费原形状距离。过滤描述的是已知碰撞模型与测量的关系；基座/相机标定及图像采集时关节值由调用者提供。
