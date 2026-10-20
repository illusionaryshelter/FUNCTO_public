# 记录式规划、执行与诊断入口

本文对应 FUNCTO_public 代码基线 `__IMPLEMENTATION_COMMIT__`。规划器完整架构见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，逐 API 的作用、输入、输出和前置数据见[接口契约](../../docs/planner-architecture/06-api-contracts.md)。本目录负责把明确的记录数据连接到通用规划与执行库，通用 `manipulation` 模块不按数据集或技能名称派发。

## 运行环境与输入

`run_recorded_skill.run_prepared` 是接收已编译 `PlanningProblem` 的实验入口。使用现有 recovery 环境，运行命令设置 `PYTHONNOUSERSITE=1`。输入包含 manifest、具名机器人反馈、`observed_robot_fixture_v1` 机器人记录、手指目标、执行限制、未知区域策略和输出目录。manifest 的 `snapshot.feedback_path` 是图像采集时的机器人反馈，规划入口反馈独立保存。

`observed_robot_fixture.build_robot_validator` 读取纯运动学 URDF、各连杆的原始观测场、缺失状态和明确实体对策略。运行器另将纯运动学 YAML 交给 cuRobo。`--observed-owner` 读取已有实体场及其位姿和附着关系。`--owner-mask` 仅提供本次需要从 RGB-D 建图的实体；`--background-mask` 明确指定静态背景端点分区。复用实体无需当前图像中可见，也无需重新分割。

## 场景构造与求解

核心接口 `planning_observation.build_scene_from_rgbd` 接受两类实体输入：新实体的 mask、深度和标定，经 `integrate_entity` 构成原始 TSDF；已有实体由 `observed_owners` 直接传入。已有 `background/coverage` 也可直接传入，完整既有场景可以不提供图像。混合输入使用明确背景分区，防止已有实体再次进入背景。自过滤使用图像采集时的机器人 FK 和同源连杆观测场。

`--contact-bindings` 保存原任务特征、有限目标区域、材料身份、活动区间和固定关系；`ContactRule` 与 `HoldingContact` 分别解释任务接触和附着内部许可。固定关系沿用关系建立前的同一场。源任务点、轨迹、区域与事件完整保留。

`MotionBackend` 统一组织参考、IK、轨迹优化和有限搜索。碰撞计算使用同一观测场的距离分类；优化间距为 1 mm，验收间距为 2 mm。具体数值过程和工作预算见[求解组件](../../docs/planner-architecture/04-solver-and-search.md)。

## 输出与执行

`recorded_task_planning.plan_and_record` 保存一次后端结果。正式曲线保存原多项式、事件时钟、任务、场景快照及设备条件；拒绝候选标为 `rejected_diagnostic_not_executable`。`status=completed` 表示记录流程完成，规划结果以 success/failure 字段解释。

`TrajectoryExecutor` 检查实际具名关节、速度、指令跟踪、设备时钟和观测机器人模型。末端与冻结预测目标的差只供事后评价；指令按原事件执行。保存的场景快照用于来源与模型身份，曲线恢复不等于重建可查询场景。具体交接见[执行组件](../../docs/planner-architecture/05-acceptance-and-execution.md)。

`--profile-runtime` 启用已有 cProfile 记录，分析器开销计入请求。实验交付使用可直接打开的本地 HTML，优先展示真实仿真图像上的源示教区域、迁移目标区域和采样点，支持比较与放大，同时保存接受、警告、拒绝、失败和缺失状态。阶段数据放在工作区 `sim-version-plans`。
