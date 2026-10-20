# 记录式规划、执行与诊断入口

本文对应 FUNCTO_public 代码基线 `TASK_NONPENETRATION_CODE_COMMIT`。规划器完整架构见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，逐 API 的作用、输入、输出和前置数据见[接口契约](../../docs/planner-architecture/06-api-contracts.md)。本目录负责把明确的记录数据连接到通用规划与执行库，通用 `manipulation` 模块不按数据集或技能名称派发。

## 运行环境与输入

`run_recorded_skill.run_prepared` 是接收已编译 `PlanningProblem` 的实验入口。使用现有 recovery 环境，运行命令设置 `PYTHONNOUSERSITE=1`。输入包含 manifest、具名机器人反馈、已知机器人 fixture 记录、手指目标、执行限制、未知区域策略和输出目录。manifest 的 `snapshot.feedback_path` 是图像采集时的机器人反馈，规划入口反馈独立保存。

`robot_validation_fixture.build_robot_validator` 读取 robot-only 原模型、显式实体清单、纯运动学 URDF、原生配对策略及控制参数。运行器将同一纯运动学配置交给 cuRobo。`--self-padding-scale`、`--self-padding-offset-m` 明确图像阴影过滤参数；它们不改变规划碰撞实体。`--observed-owner` 读取已有实体场及其位姿和附着关系。`--owner-mask` 仅提供本次需要从 RGB-D 建图的实体；`--background-mask` 明确指定静态背景端点分区。复用实体无需当前图像中可见，也无需重新分割。

## 场景构造与求解

核心接口 `planning_observation.build_scene_from_rgbd` 接受两类实体输入：新实体的 mask、深度和标定，经 `integrate_entity` 构成原始 TSDF；已有实体由 `observed_owners` 直接传入。已有 `background/coverage` 也可直接传入，完整既有场景可以不提供图像。混合输入使用明确背景分区，防止已有实体再次进入背景。自过滤使用图像采集时的机器人 FK 和同一已知机器人模型中的原生几何实体。

新背景还会排除原分区边界外的一圈八邻域像素，覆盖原图中已复用实体的边界。实体 mask 和自由射线保持原值；边缘端点记录在 `unmapped_pixels`。分区回归见 `test_observation_boundary_protection.py`，公开构造入口与模型复用回归见 `test_observed_scene_inputs.py`。

`save_observation_partitions` 在 `observations/<view>/` 中独立保存本次新建实体的 `owner-*-semantic-mask.npy`、原始 `background-candidate-mask.npy` 和实际积分 `background-mask.npy`。`partition-report.json` 按实体 ID 指向文件并说明职责。后续 `--owner-mask` 读取原实体文件，`--background-mask` 读取原始候选文件；实际积分背景用于检查本次过滤结果。这样重复复用只排除同一圈边缘，也无需再次调用分割。复用背景模型时不会生成新的积分背景 mask。

`--contact-bindings` 保存原任务特征、有限目标区域、材料身份、活动区间和固定关系；`ContactRule` 与 `HoldingContact` 分别解释任务接触和附着内部许可。固定关系沿用关系建立前的同一场。源任务点、轨迹、区域与事件完整保留。

`MotionBackend` 统一组织参考、IK、轨迹优化和有限搜索。碰撞计算组合已知机器人实体与原观测任务场。任务场景配对在优化、IK 和验收中采用同一非穿透规则；`query_role` 仅记录阶段。已确认体积交叠的分类残差为 1，不是米制穿透深度；场配对未决保留诊断，不作为碰撞正残差。原生自碰撞使用原配对和数值边界，任务接触、支撑及覆盖保持各自条件。任务 TSDF 的 2 mm 样本间距仍定义其网格分辨率。具体数值过程和工作预算见[求解组件](../../docs/planner-architecture/04-solver-and-search.md)。

## 输出与执行

`recorded_task_planning.plan_and_record` 保存一次后端结果。正式曲线保存原多项式、事件时钟、任务、场景快照及设备条件；拒绝候选标为 `rejected_diagnostic_not_executable`。`status=completed` 表示记录流程完成，规划结果以 success/failure 字段解释。

`TrajectoryExecutor` 检查实际具名关节、速度、指令跟踪、设备时钟和已知原生机器人模型。末端与冻结预测目标的差只供事后评价；指令按原事件执行。保存的场景快照用于来源与模型身份，曲线恢复不等于重建可查询场景。具体交接见[执行组件](../../docs/planner-architecture/05-acceptance-and-execution.md)。

`--profile-runtime` 启用已有 cProfile 记录，分析器开销计入请求。实验交付使用可直接打开的本地 HTML，优先展示真实仿真图像上的源示教区域、迁移目标区域和采样点，支持比较与放大，同时保存接受、警告、拒绝、失败和缺失状态。阶段数据放在工作区 `sim-version-plans`。
