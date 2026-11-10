# 已接受初态与连续任务衔接

代码基线：`b91b0c8`。本次文档登记保持该提交的组件实现内容。本文对应 `online_geometry.py`、`planning_observation.py`、`pair_evidence.py`、两个规划器的几何消费接口及记录式场景入口。数值关系目标继续由源任务的有限接触声明定义。

## 研究对象与输入

连续任务将已接受的前置执行末态用作当前任务初态。`inherited_initial_state` 表达该初态有效、现场实体对无碰撞的规划前提。它是任务衔接声明，输入为具名关节位置 q、速度 dq、反馈身份、物理时间和前置记录来源。机器人反馈与完整技能表示分别提供当前机器人状态和任务几何；RGB-D、标定及已注册 TSDF 提供任务实体表示。

声明的 JSON 字段为 `joint_names/q/dq/feedback_id/timestamp_s/source`。`q/dq` 与 `joint_names` 对齐；旋转关节单位为 rad、rad/s，移动关节为 m、m/s，时间为秒。`source` 指向前置执行保存的机器人反馈。独立任务入口可令声明为 `None`，由现场实体查询计算其入口状态。

任务曲线中的 t 从本次入口起计，t=0 对应声明的前置记录时刻。进度及事件关系保留原任务时钟。左右事件侧在同一物理 t=0 查询时共同消费同一初态前提。

## 场景与数值计算

`OnlinePlanningScene.__init__` 保存声明和原观测实体。`validate_problem` 按当前 `PlanningProblem.joint_names` 选择声明列，核对固定的 `start_q/start_dq`；该操作建立同一机器人边界的数据身份。

`components` 仍计算官方机器人自身配对、手指模型、声明关系、有限接触和观测覆盖。对于机器人与观测场、运动实体与背景、观测实体之间的配对，精确 t=0 使用声明，并在完整配对证据中记录 `inherited_initial_state`。正时间继续调用原实体距离查询。机器人官方模型由原注册形状、固定组件关系和具名 FK 放置，跨任务复用；任务实体保持原 RGB-D TSDF 数值、权重、观测支持域及局部坐标。

`optimization_objective` 的有限材料点配对采用相同 t=0 语义。`direct_planner.GeometryConstraints` 保留固定行身份，在通用 `_permission` 中把初始观测配对标为 `inherited_initial_state`，原批量提供器为该行返回非正的常量尾值和零导数。该数值只表达不违反约束。后续配对、官方自身距离及其连续区间驻点保持原计算。

`scene_relationship_objective` 组合完整源任务声明的 required contact，按原查询状态数归一化。`candidate_result.scene_conditions` 计算相同的接触、手指、运动关系及观测覆盖条件；几何结果直接读取同一个数值点。接受的候选通过场景快照保存初态声明。

`PairBatch` 为每个原查询状态和实体对保留状态、实体身份、时间和进度。`inherited_initial_state` 是声明状态，其距离上下界为空；见证选择器按这一数据职责处理它。实际查询的 `clear/intersection/unresolved` 继续保存各自几何含义。

## API 与实际调用

| API | 前置数据与作用 | 输出及消费者 |
|---|---|---|
| `run_recorded_skill.parse_args` | `--inherited-initial-state` 的明确文件路径 | 场景入口读取已接受机器人边界 |
| `load_contact_bindings` | `contacts/source` 及可选 `grasp_activations` | 原有限材料规则与抓取激活声明字典 |
| `build_scene_from_rgbd` | 新观测、复用场、机器人模型和初态声明 | 同一 `OnlinePlanningScene`，供后端绑定与验收 |
| `OnlinePlanningScene.validate_problem` | 任务具名 q/dq、关系、释放和手指事件 | 当前任务与已注册场景的一致绑定 |
| `OnlinePlanningScene.components/evaluate` | 同刻关节、工具位姿、物理时间和进度 | 原硬分项及完整配对证据 |
| `OnlinePlanningScene.optimization_objective` | 同一场景与工具/关节轨迹 | 固定材料点及 required contact 标量 |
| `preference_planner.GeometryConstraints._permission` | 原实体身份、时刻、关系和有限材料域 | 数值配对的活动模式及有限许可 |
| `direct_planner.GeometryConstraints/BoundGeometry` | 原曲线基、物理查询、场景、官方机器人 | 固定稀疏几何行及原模型一阶导数 |
| `scene_relationship_objective` | 完整 required contact、工具/实体位姿和原时钟 | 声明接触的归一化标量及诊断 |
| `scene_conditions` | 选中数值点的原状态、几何和观测策略 | 同点候选现场条件与具体失败证据 |
| `to_execution_binding` | 已接受曲线对应的场景数据 | 含初态声明的原模型快照，供正式候选归档 |

实际调用为：前置机器人末态与完整源任务 → 记录式场景入口 → `build_scene_from_rgbd` → `OnlinePlanningScene.validate_problem` → 后端入口诊断 → 完整参考初始化及主优化 → 同点候选验收 → 原执行绑定。各步骤使用同一初态声明和原关节曲线。

## 执行与验证范围

设备入口使用 `RobotStateValidator.preflight/check_state` 检查实测机器人限位、速度、官方机器人自身几何及标定模型一致性。执行沿正式保存曲线和原命令时钟推进。仿真恢复完整原生与控制器边界，并由独立评价器记录实际物体状态、接触和任务效果；评价数据与后续动作之间保持明确的计算职责。

`test_inherited_initial_state` 通过实际场与机器人模型核对初态声明、任意正时间碰撞、官方自身碰撞、q/dq 身份和快照独立性。直接规划几何回归比较继承前后的所有原行与导数：只有精确初始观测配对成为非活动，其余行保持原值。记录入口、完整观测构建和机器人执行验证另沿现有测试覆盖。
