# 完成证据的所有权与归档

本组件对应本轮实现提交，代码编号在实现提交后回填。修改前代码基线为 `4dbaef7`。本文说明 `completed_evidence.py`、`evidence_archive.py` 及其规划、记录和执行消费者的当前实现。

## 研究对象、输入与输出

研究对象是已经固定的观测数组编码，以及一次候选验收已经完成的区间几何记录。几何记录包含原场景实体身份、时间区间、实际查询、距离界、分离证据、交叠见证、未决区间与查询消耗。其物理坐标、时间和单位由原观测/验收接口定义，本组件只管理值的所有权及文件表示。仿真实际环境的对象真值与接触效果仍由独立评价接口产生。

`PreferencePlanner.evaluate_candidate` 在父接口结束所有任务、关节及几何计算后调用 `completed_evidence(report['continuous_geometry'])`。返回的 `CompletedEvidence` 由同候选报告与 `last_validation` 共同持有。原 `_ConstrainedPlanner` 的选择和验收实现继续使用原报告契约；共享模块能够接收普通数据树与本组件的只读记录。

## 完成记录与可变数据的边界

`CompletedEvidence(value, record_id=None)` 把 NumPy 数组转为 CPU 数值，把字典和列表递归转为只读字典与只读列表。数值不舍入，字段不裁剪，列表顺序和普通值比较保持不变。`record_id` 为记录身份，不根据内容合并不同候选。`completed_evidence(value)` 对既有完成记录直接返回同一对象，对普通 CPU 记录建立一次独立快照。

只读容器的公开写入操作抛出 `TypeError`；`copy` 和 `deepcopy` 返回同一不可变值。候选列表、初始化报告、返回报告和正式绑定因此可以共享区间几何证据。轻量报告头、残差字典、曲线数组、问题/场景输入快照及累计账本继续使用原有独立副本。后续请求生成新的区间几何记录，不修改前次结果。

`CompletedEvidence.formal()` 被 `execution_binding.plain` 调用。有限、字符串键的完成记录已符合正式数据树语义，直接返回自身；含无穷值的诊断记录按原 `plain` 规则转为明确的 unbounded 字段并保存一次转换结果。NaN 继续触发原正式绑定错误。原诊断值与正式转换分别保留。其余可变输入仍按 `plain` 的原规则复制；正式输入与验收输入之间的修改检测保持原定义。

`FrozenObservedField.record()` 对构造后固定的 `values/observed/weights` 分别缓存一次原 `encode_array` 结果，并以完成记录保存。这些值仍采用原 shape、dtype 和精确 base64 字节。`ObservedSpace.record()` 对固定已观测自由空间布尔网格使用相同接口。`OnlinePlanningScene.to_execution_binding()` 直接消费这些记录，所有调用仍各自生成可变的坐标、来源与关系头。字段 origin、任务和候选关系的修改检测继续通过原输入/验收副本完成。

## 文件表示与读写接口

`externalize_evidence(document, directory, write_record)` 遍历普通报告容器，在遇到完成记录时将其完整写入 `evidence/<record_id>.json`，并返回 `{"$ref": "相对文件路径"}`。同一记录保存后记住其文件位置，后续规划、执行重绑定、continuation 和执行报告均引用这份文件。归档目录之间以相对路径连接；搬移结果时须一并保留引用的文件及其相对目录关系。不同记录即使数值相同也各自保存。

`write_evidence_json(path, document, convert)` 使用调用者已有的数值编码规则写普通 JSON 与完成证据。执行侧传入 `_feedback_json`，规划诊断侧继续使用 `_write_diagnostic_json` 的诊断非有限值规则。文件写入发生在对应结果产生之后。

`read_evidence_json(path, max_bytes=None)` 读取主文档并解析本地完整文件引用。同一文件在一次读取中只解码一次，所有引用位置取得同一只读记录。设定字节上限时，主文件和每个唯一引用文件的字节数共同计入该上限。原始旁文件可直接用 JSON 工具查看，完整程序消费者通过此接口读取。

`referenced_evidence_bytes(document, directory)` 对已外置的文档统计唯一引用文件大小。曲线写入器将其与主 JSON 字节数相加，继续使用原 512 MiB 完整归档限制。

`recorded_task_planning.save_curve` 先通过 `check_result` 取得独立可变绑定，再补齐设备限制并交还原结果，使用该已有副本生成曲线元数据。传入关节界限与正式绑定界限相同时，完整导数报告直接取自 `validation.export.cubic_derivatives`；可执行区间等于完整曲线时，区间报告使用相同结果。实际区间或界限不同时，按该输入计算相应极值。

`recorded_simulation.load_curve_archive` 在原数组、时钟及手指契约检查前解析完整证据引用；`load_saved_trajectory` 继续进行原正式结果检查。`run_recorded_simulation.run(..., planning_result=None)` 可接收已返回或已加载的正式结果，或者从显式曲线文件取得结果，随后统一执行设备重绑定与实际入口检查。实验 `simulation.run_simulation` 将用于位姿角色检查的同一结果传入该接口，避免再次加载相同曲线。

`TrajectoryExecutor` 的 continuation 和执行报告通过 `_write_execution_json` 保存。该函数调用 `write_evidence_json`，引用原规划证据。控制、关节/机器人检查、事件时钟和独立效果观察仍由执行器原接口负责。

## 调用关系与验证

实际顺序为：候选验收 → `completed_evidence` → 原候选选择/正式绑定 → `PlanningResult` → 规划记录器 → 完整文件及引用 → 正式加载或内存交接 → 设备重绑定 → 执行器 → continuation/执行记录。读取和保存均消费已完成结果。

`test_completed_evidence.py` 验证原值保留、输入隔离、只读行为、重复副本共享、非有限值语义、单文件归档和完整读取限制。规划生命周期、原绑定及执行测试继续验证失败、续接、时钟和实际执行边界。实验对照进一步检查同一请求的正式曲线、完整验收证据、物理样本和独立效果。
