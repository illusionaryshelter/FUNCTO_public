# 记录式规划、执行与诊断入口

本目录原适配接口代码基线为 `6090527`，当前时间预算接入的代码引用在下一段单列。规划器完整架构见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，逐 API 的作用、输入、输出和前置数据见[接口契约](../../docs/planner-architecture/06-api-contracts.md)。本目录负责把明确的记录数据连接到通用规划与执行库，通用 `manipulation` 模块不按数据集或技能名称派发。

当前时间预算接入对应实际代码提交 `9602ebb`，本次文档回填保持该实现内容。`run_recorded_skill.parse_args` 的 `--planning-seconds` 默认直接取 `PlanningBudget.planning_seconds`，当前为 60 s；`probe_recorded_grasp_feasibility.observed_scene` 准备的预算同样读取这一字段。前者将值传入完整请求累计账本，后者保留其原规划边界截获职责。源记录中的显式预算是当时实验的数据；活动实验调用器在需要使用当前全局配置时明确覆盖并记录原始值与生效值。

## 运行环境与输入

`run_recorded_skill.run_prepared` 是接收已编译 `PlanningProblem` 的实验入口。使用现有 recovery 环境，运行命令设置 `PYTHONNOUSERSITE=1`。输入包含 manifest、具名机器人反馈、已知机器人 fixture 记录、手指目标、执行限制、未知区域策略和输出目录。manifest 的 `snapshot.feedback_path` 是图像采集时的机器人反馈，规划入口反馈独立保存。

`robot_validation_fixture.build_robot_validator` 读取 robot-only 原模型、显式实体清单、纯运动学 URDF、原生配对策略及控制参数。运行器将同一纯运动学配置交给 cuRobo。`--self-padding-scale`、`--self-padding-offset-m` 明确图像阴影过滤参数；它们不改变规划碰撞实体。`--observed-owner` 读取已有实体场及其位姿和附着关系。`--owner-mask` 仅提供本次需要从 RGB-D 建图的实体；`--background-mask` 明确指定静态背景端点分区。复用实体无需当前图像中可见，也无需重新分割。

## 场景构造与求解

核心接口 `planning_observation.build_scene_from_rgbd` 接受两类实体输入：新实体的 mask、深度和标定，经 `integrate_entity` 构成原始 TSDF；已有实体由 `observed_owners` 直接传入。已有 `background/coverage` 也可直接传入，完整既有场景可以不提供图像。混合输入使用明确背景分区，防止已有实体再次进入背景。自过滤使用图像采集时的机器人 FK 和同一已知机器人模型中的原生几何实体。

新背景还会排除原分区边界外的一圈八邻域像素，覆盖原图中已复用实体的边界。实体 mask 和自由射线保持原值；边缘端点记录在 `unmapped_pixels`。分区回归见 `test_observation_boundary_protection.py`，公开构造入口与模型复用回归见 `test_observed_scene_inputs.py`。

`save_observation_partitions` 在 `observations/<view>/` 中独立保存本次新建实体的 `owner-*-semantic-mask.npy`、原始 `background-candidate-mask.npy` 和实际积分 `background-mask.npy`。`partition-report.json` 按实体 ID 指向文件并说明职责。后续 `--owner-mask` 读取原实体文件，`--background-mask` 读取原始候选文件；实际积分背景用于检查本次过滤结果。这样重复复用只排除同一圈边缘，也无需再次调用分割。复用背景模型时不会生成新的积分背景 mask。

`--contact-bindings` 的必填字段为 `contacts/source`，可选字段为 `grasp_activations/initial_support_relations`。`ContactRule` 保留原任务特征、有限目标区域、材料身份及活动区间；原抓取激活和初始支撑分别绑定。源任务点、轨迹、区域与事件完整保留。

`robot_validation_fixture.tool_geometry_names` 根据已声明手指关节的机构公共父子树返回完整工具组件，运行器将其写入报告并传给场景。`CollisionOwner` 的有效 `Attachment` 把已有实体场随同一 TCP 刚性运动，并决定组合体内部配对；当前工具五个实体与附着对象、共同附着对象之间不作内部查询，各成员继续用原几何检查外部。无显式释放时，附着包含规划窗口终点。场景报告用 `internal_assembly_pair_count` 记录排除的配对样本数。

`MotionBackend` 组织参考 IK 样本、固定入口样条拟合、SciPy SLSQP 统一约束求解及预算内 OMPL 搜索。工具与场景运动学复用独立验证器的 Pinocchio 模型。固定材料点的原场求值提供几何搜索目标，候选验收读取官方机器人实体和原任务观测场的非穿透证据。已确认体积交叠的分类残差为 1，不是米制穿透深度；场配对未决保留诊断，不作为碰撞正残差。原生自碰撞使用原配对和数值边界，任务接触、支撑及覆盖保持各自条件。任务 TSDF 的 2 mm 样本间距仍定义其网格分辨率。具体数值过程和工作预算见[求解组件](../../docs/planner-architecture/04-solver-and-search.md)。

## 输出与执行

`recorded_task_planning.plan_and_record` 保存一次后端结果。正式曲线保存原多项式、事件时钟、任务、场景快照及设备条件；拒绝候选标为 `rejected_diagnostic_not_executable`。`status=completed` 表示记录流程完成，规划结果以 success/failure 字段解释。

`TrajectoryExecutor` 检查实际具名关节、速度、指令跟踪、设备时钟和已知原生机器人模型。末端与冻结预测目标的差只供事后评价；指令按原事件执行。保存的场景快照用于来源与模型身份，曲线恢复不等于重建可查询场景。具体交接见[执行组件](../../docs/planner-architecture/05-acceptance-and-execution.md)。

## 独立物理证据与评分

材料身份绑定的实际代码基线为 `5adbdd699846de4f98a50eb33e1f532347150142`；本次仅文档提交保持该实现内容。

`run_recorded_simulation` 将正式曲线交给既有执行器，并把原生仿真采样送入独立记录回调。`recorded_skill_evaluation.NativeSkillPhysicsEvaluator` 消费正式执行绑定、显式原生实体对应、任务声明和每一步采样；输入中的真实TCP位姿、物体位姿、接触只用于结果分析。世界位姿 `T_W_B` 将物体本体坐标B映射到标定世界W，固定 `T_B_C` 将技能/观测实体坐标C映射到B，支持任务的材料点按 `T_W_C=T_W_B T_B_C` 转入世界坐标。每个效果在其原声明进度对应的设备时刻计算，记录保留原始样本与独立时钟。

`_FunctionalGoalEvaluator` 的材料身份集合来自正式任务的位姿事件 `actual_relation`、支持要求的载体绑定，以及场景已注册实体。调用方以 `evaluation_owner_bindings` 给出材料身份、原生 body 名称和带来源文本的静态 `T_body_owner`。构造器核对材料身份属于该集合、目录与原生 body 身份一致、固定变换为刚体；`observe` 将本次原生 body 位姿与固定材料坐标组合。材料坐标可以属于任务关系而不另设碰撞字段，评价结果保持在独立报告中。


| API | 前置输入与实际作用 | 输出及消费者 |
|---|---|---|
| `NativeSkillPhysicsEvaluator.__init__` | 正式执行绑定、原生模型/实体清单、手指接触组与明确坐标对应；恢复原任务及效果声明 | 独立评估对象，由实验运行器持有 |
| `_FunctionalGoalEvaluator.__init__` | 正式功能/支持要求、场景身份、显式材料—原生 body 对应及静态坐标 | 材料测量绑定与原目标身份，由 `observe/finalize` 消费 |
| `observe` | 初态及每个正常物理步的原生状态、执行器采样时间 | 连续位姿、关节、接触与原任务测量；回调返回值不参与控制 |
| `_FunctionalGoalEvaluator._measure_support` | 保存的 `SupportRequirement`、原生物体位姿、材料坐标对应；逐点计算gap/body/footprint/normal | 四项原测量全部保留，`contact_face.valid`仍表示完整几何便利判定；要求的T/F只汇总 `hard_components` 中的项，空集表达全部为偏好 |
| `NativeSkillPhysicsEvaluator.finalize` | 完整采样序列及执行器的正常步账本 | `physics-evidence.npz`、`evaluation-report.json`；覆盖缺失保留U，原始目标时刻与末帧诊断分别保存 |
| `summarize_skill_evidence` | 已测量的目标/效果/过程/终点T/F/U、正式绑定和覆盖 | 聚合任务结果与诊断角色；不重新测量原始NPZ、不把缺失谓词变为通过 |

标准释放效果 `actual_release_from_current_feedback` 在原设备时刻截取接触前缀：同一目标此前有接触，且声明手指组均持续无接触达到已有释放窗口，才得到T。终点接触诊断和原事件时刻的效果证据分别记录。生产条件的 `effect_requirement` 选择这一独立评分计算；在线事件是否等待证据由它自身的 `requires_evidence` 指定。

`--profile-runtime` 启用已有 cProfile 记录，分析器开销计入请求。实验交付使用可直接打开的本地 HTML，优先展示真实仿真图像上的源示教区域、迁移目标区域和采样点，支持比较与放大，同时保存接受、警告、拒绝、失败和缺失状态。阶段数据放在工作区 `sim-version-plans`。
