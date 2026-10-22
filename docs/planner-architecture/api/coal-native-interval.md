# 原生几何区间接口

代码基线：`fe0a4f8`。

`manipulation/_coal_scene_batch.cpp` 的 `native_projection_bounds(geometry, transform, direction)` 返回原生实体在指定世界方向上的投影上下界。它复用已有形状支持计算与外向区间算术；`native_field_distance` 用其证明距离下界。

`native_feasible_point_bounds(geometry, transform, wanted_world)` 输出同一实际可行点的 `local_bounds` 与 `world_bounds`。wanted_world 只用于提出接近目标的点；原生凸实体由 `volume_convex_point` 从原面顶点与原顶点均值提出非负归一权重，再对该组合进行区间重算。解析盒体/圆柱复用原形状构造。已有三角面可行点仅在上界更小时替换，局部与世界区间共同更新。

`volume_convex_point` 的 Eigen 求解仅提出权重，实际点属于原形状的依据是原顶点的非负凸组合。`point_bounds` 将区间输出为坐标下界与上界。RoundingGuard 保留调用者舍入状态。这些接口只提供原实体的上下界和可行点，任务TSDF支持、点对许可及最终分类由Python混合查询消费。

本项目扩展由 `tests/planning/build_coal_scene_batch.py` 使用现有编译器和已安装库构建，输出项目内共享库；环境依赖保持原安装。
