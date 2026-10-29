# 官方机器人原生形状与观测场查询接口

代码基线：`858d011`。本文的后续文档提交登记该代码基线，生产代码与该实现提交相同。实现为 [`manipulation/native_field_distance.py`](../../../manipulation/native_field_distance.py)。数据归属见[观测与碰撞](../02-observation-and-collision.md)及[共享验收](../10-shared-candidate-validation.md)。

## 1. 输入对象与职责

`query_native_field` 计算一个已注册 Coal 凸体或解析形状与一个原 `ObservedField` 三线性非正集合之间的非穿透关系。机器人形状来自固定机器人和其官方固定末端配置；形状及组件固定关系可跨任务复用，其位姿由标定、具名关节反馈和运动学计算。任务实体采用现场 RGB-D 的原场及观测支持域。

`native_pose`、`field_pose` 分别将各自局部坐标映到共同世界坐标 W。局部搜索框只约束计算域：原生侧的搜索域为原形状与该框的交集。框本身不替代注册实体材料。

## 2. 数值辅助 API

| API | 前置输入、输出与调用作用 |
|---|---|
| `_transform(matrix)` | 4×4 刚体矩阵转成 Coal `Transform3s` |
| `_native_bounds(geometry,transform)` | 通过完整原生形状在三条坐标轴上的支持投影，返回 `[2,3]` 范围 |
| `_prepared_world_bounds(lower_key,upper_key,pose_key)` | 每个键含 dtype、shape 和全部 bytes，恢复数组并计算 `implicit_distance.world_bounds`，冻结结果；LRU 上限 65536 项 |
| `world_bounds(lower,upper,pose)` | 以完整数值输入标识相同刚体范围变换；缓存的是纯范围变换 |
| `_projection_gap(geometry,transform,box)` | Coal 距离提供搜索方向，完整原生支持投影提供实体区间，共同数值核计算向外舍入分离下界 |
| `_local_interval(world,pose)` | 对世界范围八角作逆变换并计入逆变换误差，返回局部包围范围 |

`_projection_gap` 对有限非零方向调用 `norm3` 与 `projection_lower`。零方向给出零下界，非有限方向保留原错误语义。原生支持投影由 `_coal_scene_batch.native_projection_bounds` 计算；范围距离和半空间分类分别来自 `implicit_distance` 的共同接口。数学公式、Numba 首次编译与数组布局语义见[数值内核](geometry_kernels.md)。

## 3. 主查询 API

```python
query_native_field(geometry, native_pose, field, field_pose, *,
                   precision_m=1e-4, permission=None,
                   check_request=None, native_bounds=None)
```

`precision_m` 是正的局部数值停止尺度。`permission` 为本次已激活的 `PairPermission`，其有限半空间位于 W。`check_request` 每 64 次节点访问调用一次。`native_bounds=(local,world)` 可由场景提供同一注册实体在本次姿态下的完整支持范围；缺省时本接口计算这两个范围。空场直接返回空集合记录。

查询先保存原生完整范围与场根完整范围的下界。内部 `push` 按许可、范围距离、方向投影依次排除搜索域，未分离域进入优先队列。普通情形细分任务场；许可边界切分原生形状时，按原局部范围与场范围大小细分对应域。每个动态子框仍表示原形状的子集。

到达原支持单元后，`native_feasible_point_bounds` 将候选点映到原生形状的实际可行点范围，并保留局部、世界区间。该实际点的完整区间须严格位于原支持单元的开内部，且三线性值可靠为负或原八角全部非正，才形成体积交叠证据。有限许可同时检查两端点。可行点范围的中点用于显示和局部导数，区间本身用于成员和误差判断。

原占据点与原生实际可行点对给出距离上界。仅边界占据、达到数值精度或无法继续细分的域记录未决，查询仍继续处理其余域。最终未决与确定交叠分别返回。

## 4. 输出与调用关系

返回 `status` 为 `clear/intersection/unresolved`，并保存 `lower_m`、`upper_m`、`full_entity_lower_m`、`intersection`、`queries`、世界点对见证及 `scope='known_native_and_represented_observed_set'`。`distance_scope` 明确下界属于当前姿态的未许可点对域；`full_entity_lower_m` 另表示完整实体对。空集合以 `empty_set` 标记。

存在见证时，`witness_first_bounds/second_bounds` 保留世界区间，`witness_first/second` 保存中点，`witness_coordinates` 标识中点用途。这些字段随原 query 直接进入场景证据。

调用链为 `OnlinePlanningScene.components → observed_collision → query_native_field`。场景拥有官方机器人形状及本次状态的局部/世界范围，场拥有原 TSDF、支持域和空间树；本模块拥有单次搜索队列和见证。连续验收通过场景保存的完整实体键与距离证据分析时间区间，执行侧继续通过其既有机器人和观测约束接口调用。
