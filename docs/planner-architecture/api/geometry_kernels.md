# 三维几何数值内核

代码基线：本次通用几何内核实现，完成提交后登记实际代码提交。实现为 [`manipulation/_geometry_kernels.py`](../../../manipulation/_geometry_kernels.py)，调用者为 [`implicit_distance`](implicit_distance.md) 与 [`native_field_distance`](native_field_distance.md)。

## 1. 职责、输入与依赖

本组件计算包围范围和有限半空间的数值关系。输入范围端点为有限 float64 三维数组，满足逐坐标 lower≤upper；一组计算的坐标均属于调用者给定的同一笛卡尔坐标系。几何查询通常使用标定世界坐标 W。数组可以连续、非连续或只读，函数只读取输入。范围表示搜索域的包围，不表示新的实体材料。

Numba `njit(fastmath=False, cache=True)` 编译这些固定维数运算，NumPy 提供二进制浮点类型、平方根和 `nextafter`。首次实际调用按数组布局和只读属性形成所需编译签名，时间属于触发它的请求；之后复用编译代码。Numba 的缓存只保存代码，不保存实体判定、距离或查询历史。几何表示、空间树、预算回调与返回证据均由调用者持有。

`_EPS` 为 float64 机器精度，`_GAMMA=32*_EPS/(1-32*_EPS)` 为当前坐标计算的前向舍入余量。它与模型几何余量的职责不同：前者随算术量级形成误差界，后者是场景声明中的物理数据。

## 2. API 与实际调用关系

| API | 输入 | 输出与调用作用 |
|---|---|---|
| `_plane_sums(low, high, normal)` | 三维范围和一行法向 | 最小/最大投影及对应乘积绝对值之和；由分类和投影下界共同调用 |
| `classify_box(low, high, normals, bounds)` | 范围、`normals[N,3]`、`bounds[N]`，半空间为 n·x≤b | `1` 表示全部可靠地位于域内，`-1` 表示至少一行可靠地在域外，`0` 表示混合；`PairPermission._classify` 转为既有字符串 |
| `norm3(x,y,z)` | 三个分量 | `sqrt((x*x+y*y)+z*z)`；范围距离和原生形状投影使用同一三项运算次序 |
| `box_distance(low_a, high_a, low_b, high_b)` | 两个同坐标系范围 | 向外舍入的非负分离下界；`implicit_distance.box_distance` 保留既有范围对输入接口 |
| `projection_lower(low, high, direction, native_low, native_high, length)` | 范围、投影方向、原生实体完整投影区间、方向范数 | 原生实体与范围在该方向上的非负分离下界；由 `native_field_distance._projection_gap` 消费 |

调用链为 `OnlinePlanningScene` 的成对几何查询进入 `query_distance/query_native_field`，再调用范围距离及 `PairPermission` 分类；原生形状查询还通过 `_projection_gap` 调用 Coal 取得方向及原生投影区间，然后调用 `projection_lower`。`initial_support` 通过其原有 `implicit_distance.box_distance` 导入使用同一实现。

## 3. 运算及边界语义

对半空间每一行，按照法向符号逐轴选择最小和最大端点，按 x、y、z 次序累加乘积与乘积绝对值。最大投影加误差仍不超过边界时该行可靠地在内；最小投影减误差严格超过边界时该行可靠地在外。优先判断全部在内，再判断至少一行在外，其余为混合。零行半空间表示整个空间。

范围距离令 `g_i=max(0, a_low[i]-b_high[i], b_low[i]-a_high[i])`，`d=norm3(g)`，`m` 为十二个端点分量的最大绝对值，返回 `max(0,nextafter(d-GAMMA*(d+m),-inf))`。逐轴访问四个端点可直接消费不同只读属性和不同步长的数组。

投影下界先将范围投影向外扩展 `GAMMA` 误差，再与原生完整投影区间计算间隙，对误差扣减和除以方向长度的结果向负无穷舍入。`_projection_gap` 的 Coal 调用决定方向，原生支持投影决定实体区间。零方向在调用者处给出零下界。

显式三项范数保留当前 NumPy 一维 `sqrt(dot(v,v))` 的计算次序；Numba 通用 `linalg.norm` 使用的缩放范数具有不同的舍入、溢出与下溢行为，因此此接口采用三项表达式。所有核关闭快速数学重排。

## 4. 验证范围

`tests/test_geometry_numerics.py` 以原 NumPy 向量归约核对半空间边界及其相邻浮点、只读/非连续数组、范围距离和投影间隙。完整实体查询由 `test_implicit_distance.py`、`test_native_field_distance.py`、`test_geometry_query_bounds.py` 等测试覆盖；初始支撑和连续区间分别沿其现有 API 验证。实验复刻的数据、性能和物理结果位于项目外阶段记录。
