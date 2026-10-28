# 两个规划入口共用的保存曲线验收

代码基线：`WORKING_IMPLEMENTATION`。本文描述该提交的生产代码；精确函数签名与直接调用见各组件 API 页。阶段运行数据位于项目外 `sim-version-plans`。

## 研究对象、输入与坐标

验收对象是一条已经保存为 `JointTrajectory` 的三次关节曲线。输入包含该曲线的具名关节位置与速度、多项式物理时间、技能进度、`PlanningProblem`、候选私有场景、同一请求预算以及已完成查询。场景中机器人使用固定配置的官方实体模型；其形状和组件固定关系可跨任务复用，位姿由标定、具名关节及正运动学得到。任务实体使用现场 RGB-D 建立的 TSDF、观测支持域和声明的运动关系。功能区域、相对关系、轨迹及事件属于完整技能表示。

世界坐标系 W 由标定定义，机器人模型基座为 B，连杆为 L，实体局部坐标为 A。`T_WB @ T_BL(q) @ T_LA` 把实体局部点映到 W。原生碰撞矩阵使用这个连乘顺序；Torch 场景位姿使用原 Pinocchio 四元数投影和局部位姿合成。两种消费路径共享原始正运动学结果，各自保留原浮点操作。位置和距离使用米，角度使用弧度，时间使用秒。`progress` 表示运动关系所在的技能进度，`permission_progress` 表示接触许可的进度，两者在同一物理时刻可以不同。

## 调用顺序与返回边界

`_ConstrainedPlanner._candidate_metrics` 在保存多项式上计算原固定时间样本、任务锚点和 q/dq/ddq/jerk。`_candidate_kinematics(scene)` 用场景已有 Pinocchio 模型及工具、基座、碰撞连杆的并集创建 `FKSession`。`robot_tool_states(..., fk_session=...)` 补齐原名义手指状态，再由 `PinKinematics.states` 计算完整连杆和 TCP 状态。`_scene_boundary_evaluate` 在关系边界扩展左右进度，保持相同关节值和物理时刻，调用 `OnlinePlanningScene.evaluate`，最后将扩展行映回原物理查询索引。

场景把这次完整查询写入一个 `PairBatch`。`_geometry_query_states` 生成共享该批次的 `EvidenceState` 序列。任务与连续关节域验收通过后，`_candidate_geometry` 把这些已计费状态交给 `check_interval_geometry`。该函数用实际实体运动界与完整实体距离下界检查各时间区间，并按原空间尺度和剩余查询点数细分；新增时间通过同一场景接口产生新的 `PairBatch`。一个区间检查调用中的所有新增批次共用其 `FKSession`；固定查询结束时已释放固定查询的 FK 存储，初始几何证据则继续持有。

`PreferencePlanner` 继承上述完整验收。其 `_candidate_metrics` 将初始 `EvidenceState` 交给 `GeometryConstraints.add_evidence`；其 `_candidate_geometry` 保存候选场景每次产出的 `PairBatch` 引用，父方法完成区间计费后，再按原批次顺序消费。这一步选择优化所需材料见证，并不再次查询场景。两个入口各自的候选生成、目标函数、请求生命周期和正式结果绑定由其原组件负责。

## 完整几何证据：`pair_evidence.py`

一个批次有 S 个扩展状态及 P 个声明实体对。`PairDescriptor` 保存显示名称、含类别的实体标识、分项、原生配对策略和来源；`key` 返回分项及两个含类别标识，`display()` 构造最终区间记录的身份字段。机器人、任务实体和背景分别使用 `robot/…`、`owner/…` 和 `background`，显示名称相同不合并实体。

`PairBatch()` 创建一次查询的所有者。`bind_clock(times, progress, permission_progress)` 复制二维批次时钟并建立扩展 `state_index`。`begin_native(pairs, boundary, include_bounds)` 注册原生列并分配 S×P_native×12 的 float64 数组；`retain_native_chunk(start, numeric, unmodelled)` 复制原块的全部十二列及未知运动标记。十二列依次为原 Coal 有符号距离、双方三维见证、三维法向、见证恒等式误差以及完整实体投影分离下界。原始数值在未知运动行中仍被保留，消费状态单独表示未知。

`append_observed(index, component, first, second, evidence, geometry_entity_key, permission_kind)` 持有原观测查询或声明关系的完整 payload 引用。现场 TSDF 的可变见证包围区间、支持单元、查询次数及其他模型字段保持原结构。已注册场景对每个实体对枚举全部状态，包含内部刚性关系与未建模运动状态；生产者完成后不再修改 payload。

`seal()` 按原阈值建立 S×P 的状态、许可、下界、完整实体下界、上界、空集合和见证存在索引。每个数值界具有独立有效位，区分字段缺失／None 与数值 NaN。`order` 保留原生行以状态优先、观测行以实体对优先的遍历次序；公共数值数组转为只读。`native_pair_count` 返回原生前缀列数。内部 `_register` 只分配声明列并保存其身份索引。

`set_query_layout(query_index, side)` 在同一次关系扩展查询完成后绑定原物理列和 −1/0/+1 关系侧。`materialize(state,column,query_layout=True)` 仅在诊断、输出或已选见证边界恢复完整原字典；原观测字段存在性和原生策略来源均保留。

`EvidenceState(time_s,side,progress,residual,batch,source_rows)` 表示一个精确物理时刻及关系侧。`tokens()` 保留原状态内的完整行顺序；`pair_columns` 和 `source` 为区间读取提供原字典最后一行的索引语义；`pair_record` 导出该完整行。`_geometry_query_states` 只分组索引，空实体对仍返回已完成物理状态。`selected_witness_records(states=… 或 batch=…)` 分别按原分组顺序或原生产顺序选取全部相交及每实体对最近可用见证，返回所选完整记录与源行总数；相同排名和共享 NaN 的原 Python 比较身份保持原选择结果。

## 原始运动学复用：`validation_kinematics.py`

`FKSession(model,frames)` 接收已加载的同一固定机器人模型和消费连杆集合，创建一个 Pinocchio Data。上下文退出释放 Data 与状态表；该实例由验收调用显式传递。`placements(model,q)` 按完整模型顺序的 q 字节返回自有 SE3 副本。`_state(model,q,dq,frame_ids)` 使用 Pinocchio 原 q 或 q/dq 重载，速度另按完整 dq 和请求 frame 序列区分。`evaluate(bridge,position,velocity)` 接收 `PinKinematics._ordered` 处理后的 CPU float64 状态，按该 bridge 的标定和 frame 次序返回原位姿与 LOCAL 速度数组。基座标定字节区分投影缓存；不同手指值、相邻浮点 q 和不同 dq 保持各自结果。`describe()` 返回计算次数及自有数值存储计数，`_charge` 更新这些计数。

`PinKinematics.poses/states/_evaluate`、`DifferentiableRobotKinematics.poses`、`robot_tool_states` 和 `RobotGeometryChecker._prepare_query` 显式接收 `fk_session`。只有无导数验收路径读取该对象；需要梯度时仍计算原 Pinocchio 解析导数和 Torch 反向关系。原生实体矩阵继续使用原 SE3 齐次矩阵，而不是将场景四元数还原为另一份矩阵。

## 距离与区间准备的范围

`RobotGeometryChecker.query_pair_batches` 返回有序独立 NumPy 数组，块上限保持每块 4096 次实体配对，单状态超出时保留完整状态。场景请求完整证据时使用 `parallel=True`，由四个标准库线程对当前块的连续状态分片执行原 `NativePairTable.query`，按输入次序连接全部十二列。形状和配对表只读共享，原生函数为每次调用复制变换、mask 和请求并使用独立求解状态。原逐状态、块前后及场景逐对的期限检查保留；异常记录整个块与分片的状态范围及原生内部索引。

`native_field_distance.world_bounds` 调用原外向舍入包围盒变换，按上下界和位姿的 dtype、shape 与全部字节复用结果。`_prepared_world_bounds` 使用进程内最多 65536 项的 LRU，结果数组只读；其输入键与结果均有界持有。实体距离、许可裁剪、观测支持和分类仍由当前实际查询计算。

`interval_geometry.prepared_polynomial` 使用完整 float64 系数、断点、形状与导数阶数作为键，最多保留32份原 SciPy PPoly 导数及 long-double 段系数。`_polynomial_ranges` 通过二分索引定位原相交段，在这些段上执行原 Bernstein 包络及舍入公式。`SceneMotionBounds.segment_bounds` 在该场景绑定实例内保存每条声明实体路径的原段几何位移界和进度倒数；每个时间区间仍计算自己的进度变化率和物理速度界。

`check_interval_geometry` 直接读取 `EvidenceState` 和 `PairBatch` 数值列，按两个端点批次共享实体对身份目录。已确认相交保留具体状态、实体与完整见证；已证明区间保存原端点下界与运动界；未决区间保存尺度、预算、缺失状态或模型未知原因。有限接触许可仍使用独立完整实体下界来证明全时段分离。最终报告的定义、未知状态、停止顺序和实际查询点计费保持原接口。

现有 `_candidate_evidence_workspace_bytes` 的候选分配上界继续覆盖完整观测 payload、数值块、共享状态索引、原模型计算和最终区间记录；主机准入读取当前 RSS，包含进程内已驻留准备项。新增 FK 存储受候选实际物理查询点数和 frame 数约束。测试核对完整值、状态侧、释放、许可、原生十二列、梯度、取消位置及分配范围；两个入口完整请求的复现记录位于阶段报告。
