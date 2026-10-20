# 已知凸实体与原始观测场的距离

代码基线：`__NATIVE_MODEL_COMMIT__`。

`query_native_field(geometry, native_pose, field, field_pose, *, separation_m, precision_m=1e-4, permission=None, check_request=None)` 输入一个已编译 Coal 原生实体、原 `ObservedField`、各自世界位姿、要求间距、计算精度和有限点对许可。机器人保持原凸实体或解析形状，任务对象保持原支持单元内的三线性非正集合。

空间框和原实体支持投影提供距离下界。Coal 的 native/box 查询仅建议投影方向；`native_projection_bounds` 对实际原实体计算外向投影区间。`_projection_gap` 结合任务子框的投影得到可靠分离下界，GJK 数值容差不充当误差上界。

`native_feasible_point_bounds` 构造原实体内的点，并保留其局部和世界区间。凸实体的候选来自原面顶点及原顶点均值的非负组合，Eigen 仅提出权重；区间重算证明该点的实际归属。解析形状使用对应的原形状构造。它与任务场中已确认的占据点给出距离上界。

若同一原生点的完整区间位于一个原支持单元内，且该区间上的原三线性值严格为负，则返回 `intersection`。这是已知机器人集合与已表示任务集合相交，不能单独解释为真实材料的穿透深度。见证中点仅供显示和局部导数，完整 `witness_*_bounds` 保留点的数值区间。

`PairPermission` 同时约束两个端点。完全许可的域排除；跨许可边界时局部细分原生实体与搜索盒的交集，以检查许可外部分。普通距离仅细分任务场。空间精度耗尽或距离上下界已在计算精度内夹住阈值时返回 `unresolved`。`clear` 表示所需间距得到证明，`margin_insufficient` 表示实际点对上界低于间距。所有分类由原集合决定。

`_transform` 构造 Coal 位姿，`_native_bounds` 使用三轴投影包围原实体，`_local_interval` 将世界点区间带误差地映回任务场局部坐标。请求回调沿遍历执行，查询采用串行 CPU 工作区。
