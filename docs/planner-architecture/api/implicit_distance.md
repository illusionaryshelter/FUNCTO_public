# 观测占据集合的几何查询接口

代码基线：本次通用几何内核实现，完成提交后登记实际代码提交。实现为 [`manipulation/implicit_distance.py`](../../../manipulation/implicit_distance.py)。组件语义见[观测与碰撞](../02-observation-and-collision.md)，共同数值计算见[三维数值内核](geometry_kernels.md)。

## 1. 研究对象与坐标

`ObservedField` 表示具有实际观测支持的三线性非正集合。`values[X,Y,Z]` 是现场 RGB-D 积分所得 TSDF 样本；`observed` 给出每个样本的实际支持；`origin[3]` 与正 `voxel_size` 将格点映到场局部坐标。只有八个顶点都被观测的单元参与插值，非正集合由该单元的原三线性函数定义。

查询给定每个场从局部坐标到共同世界坐标 W 的 4×4 刚体矩阵。场数据保持不可变，实体运动通过这些矩阵表达。未知样本、支持边界、常值零体积和只含零边界的单元分别保留各自语义。距离下界用于排除搜索域，交叠结论需要实际占据见证。

## 2. 坐标与数值 API

| API | 前置输入、输出和作用 |
|---|---|
| `transform_points(points, pose)` | 局部点与局部到 W 的刚体矩阵，输出世界点 |
| `transform_point_error(points, pose)` | 对同一正变换计算逐坐标前向误差 |
| `inverse_points(points, pose)` | 世界点及同一矩阵，输出局部点 |
| `inverse_point_error(points, pose)` | 同时计入逆变换运算和数值旋转的正交缺陷，输出局部误差界 |
| `box_corners(lower, upper)` | 两个三维端点，按固定八角点顺序返回范围角点 |
| `world_bounds(lower, upper, pose)` | 以中心和半径变换局部范围，再计入误差并向外舍入，输出世界范围 |
| `box_distance(a, b)` | 两个 `(lower[3],upper[3])` 的 float64 世界范围，直接调用共同编译内核，输出非负分离下界 |
| `trilinear(values, coordinates)` | 八角值及单元归一化坐标，输出插值、局部梯度和算术误差 |

`_EPS` 和 `_GAMMA` 来自共同数值模块。`box_distance` 是场—场、机器人—场和初始支撑共同消费的接口。原范围对接口及输入数组所有权保留；Numba 编译的详细算术和首次调用语义见[内核契约](geometry_kernels.md)。

## 3. 有限点对许可

`PairPermission(regions, excluded_second_cells=())` 持有有限许可的并集。每项为 `((normals_a,bounds_a),(normals_b,bounds_b))`，两端点必须同时属于同一项的两个半空间交集才获得许可。法向和边界均在 W 中；区间激活、方向和关系由场景先行计算。

`_classify(box,region)` 将半空间数组转换为 float64 后调用共同 `classify_box`，输出 `inside/outside/mixed`。`boxes(first,second)` 检查整个并集：任意同一项的两个范围都在内则 `permitted`；存在双方均未被证明在外的项则 `mixed`；其余为 `forbidden`。空半空间交集表示全空间，空并集表示没有许可。

`excludes_cells(first,second)` 依据声明的第二场原单元身份进行有限材料排除；原 TSDF 样本和相邻单元身份保持原值。

## 4. 观测场及其内部 API

`ObservedField(values, observed, *, origin, voxel_size)` 复制并冻结 float32 样本与 bool 支持，构造完整支持单元、可能含非正值的单元索引，以及 SciPy `cKDTree` 上的空间层级。`_Node` 保存局部范围、子节点或原单元 ID；原叶还持有只读局部材料点。`storage_nbytes` 计入保留数组、树节点及材料点，用于场景资源准入。

| 方法 | 输入、输出和实际作用 |
|---|---|
| `_initial_cell_bounds()` | 对全部原单元按 1024 格分批收紧可能占据范围；每个单元依次处理三个轴，输出每叶 lower/upper |
| `_initial_occupied_points(lower,upper)` | 在上述范围内批量生成原角点、已知非正样本和边零根，输出按原叶顺序的只读点数组 |
| `bounds()` | 返回根范围；空占据索引返回 `None` |
| `cell_value(index,points,point_error=0)` | 读取指定原单元，输出值、米制梯度和包含坐标误差的插值误差；支持批量索引 |
| `sample(points,excluded_cells=())` | 在局部点定位完整支持单元，输出值、有效标志、梯度和误差；闭单元边界检查邻接支持，未知值保留 NaN |
| `node_values(node)` | 返回该原单元在当前搜索范围八角的值 |
| `tighten(node)` | 依据可靠为正的面片去除空域，按原三线性边根收紧范围，返回新搜索节点 |
| `occupied_points(node)` | 原叶直接返回预计算材料点；细分节点按当前范围计算角点、原非正样本及向非正侧修正的边根 |
| `split(node)` | 内部节点返回原子节点；叶按最长轴二分，仅保留仍可能包含占据的子域并收紧；不可再分时返回空元组 |

构造与细分使用同一 `cell_value` 误差公式。支持单元中只有非负角值且至少一项正的非恒定单元，其非正部分只能位于边界，记为 `_boundary_only_cells`；八角均为零的常值单元保持有体积的占据。

## 5. 成对查询与返回记录

`query_distance(first, first_pose, second, second_pose, *, precision_m=1e-4, permission=None, check_request=None)` 输入两个原场、局部到 W 的矩阵、正数值精度和本次许可。它先保留完整实体根范围的下界，然后按原空间层级检查未许可点对。

每次 `push` 计算两个世界范围及许可分类。完整许可域被移出本次查询域；严格正的距离下界完成分离；其余域进入优先队列。混合许可域按进入顺序展开，普通域使用原直径排序。每 64 次实际节点访问调用一次请求预算回调。

两个原支持单元的严格内部共同占据点才能构成交叠见证。坐标误差范围须位于双方支持开内部；对应三线性值须可靠为负，或原单元八角全部非正。只含边界的单元及达到数值停止尺度的域保留未决；本域未决后继续检查其他域。距离上界来自原占据点对，不能以包围范围的交叠代替材料见证。

| 返回字段 | 含义 |
|---|---|
| `status` | `clear` 表示完成排除；`intersection` 表示存在体积交叠见证；`unresolved` 表示剩余数值未决域 |
| `lower_m`, `upper_m` | 本次未许可点对域的距离界；未知上界为 `None`，空集合界为 `None` |
| `full_entity_lower_m` | 许可裁剪前的完整实体下界；无许可时可使用完整搜索的更紧下界 |
| `intersection` | 确定交叠为真，完成排除为假，未决为 `None` |
| `queries` | 实际节点访问数 |
| `witness_first/second` | 本次保存的世界点对，或 `None` |
| `empty_set` | 空占据集合明确记录此标志 |
| `scope`, `distance_scope` | 原表示占据集合与本次姿态的未许可点对域 |

## 6. 依赖与消费者

`FrozenObservedField` 继承本场表示并负责归档；`observed_collision` 由场景选择场—场查询。`native_field_distance` 复用范围、坐标误差、点对许可和场单元计算，同时保留官方机器人原形状。`initial_support` 消费相同范围距离与原单元身份。`sampled_geometry_objective` 消费原材料点及实际插值/梯度；`interval_geometry` 消费场景交付的完整实体下界。组件之间通过上述表示和返回记录连接。
