# 外部文件与数据职责

库实现基线：`IMPLEMENTATION_COMMIT_PENDING`（本轮实现提交完成后填写；正文依据当前已实现的接口覆盖更新）。本文说明 FUNCTO 技能、观测、规划和执行接口的持久化数据。第 4.1 节的实验适配器位于项目外，其实现由所链接脚本和阶段计划记录。总入口见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，观测场与机器人输入的对应关系见[实现与运行对照](../../../sim-version-plans/2026-10-19-put-planner-audit/observed-implementation-comparison.md)。

## 1. 技能表示与运行环境状态

所有属于 skill representation 的内容完整保留，包括源演示区域、轨迹、物体与工具相对关系、功能坐标、任务几何、交互事件和释放事件。由仿真示教提取的这些数据也属于合法技能输入。Grounding 将这些源知识与当前图像、深度及明确坐标关系结合，产生当前功能区域和参考；任务编译把绑定结果、事件、任务条件及机器人入口状态组织为 `PlanningProblem`；规划器再联合机器人模型、场景碰撞模型与资源预算求解关节轨迹。

禁止求解额外使用的是**仿真实际运行环境中产生的真值**，即本次待执行或正在执行环境的真实物体/目标位姿、真实分割、接触和成功效果等特权状态。它们进入独立事后评价。当前 RGB-D、机器人编码器反馈、运动学和标定通过各自传感或声明接口参与计算。当前固定机器人的官方本体及官方固定末端执行器 mesh 可跨任务复用，其位姿由反馈与运动学确定；任务物体、环境和其他对象的 mesh/CAD 不属于该许可，任务实体使用现场 RGB-D TSDF 及其支持域。

数据职责由生产过程和实际消费者确定。源物体位姿表示示教运动；当前物体位姿可能是观测估计、名义预测，也可能是运行环境直接提供的特权状态，三者须分别说明。源接触事件用于表达技能时序，当前接触效果另由允许的观察或事后评价记录。JSON、NPZ、文件名中的 `report` 以及保存时间均不能单独决定用途。

## 2. 技能包与源演示

`manipulation.skill.load_grounding_skill(directory)` 读取同目录的清单、JSON 元数据、NPZ 数组和源图像，组合成 `metadata/source/representation`。`source` 中说明原录制位置的路径可以仅承担来源记录；读取该技能包不等于再次读取原录制环境。

| 文件 | 内容与含义 | 直接读取及计算 |
|---|---|---|
| `skill.json` | 技能包格式及各成员文件名 | `load_grounding_skill` 定位成员 |
| `metadata.json` | 技能身份、技能类别、对象角色、语义描述与源事件信息 | 技能选择、语义对象选择与 grounding |
| `source.json` + `source.npz` + `rgb.png` | 源图像、深度、相机标定、裁剪、源实体/功能区域、源点观测、特征和功能坐标 | `load_grounding_skill` 恢复源记录；`run_region_transfer` 使用源区域与特征提出并选择当前区域 |
| `representation.json` + `motion.npz` | 源帧与时间、源物体/工具轨迹、`L_G/L_R`、`C_D/H_g/H_m`、交互/释放时刻与事件 | `generate_reference` 适配源运动并生成连续物体/工具参考 |
| `program.json` | 另一任务入口使用的源示教、功能事件、开闭指令与任务关系 | `prepare_candidate` → `build_planning_problem` → `ground_program`；部分实验入口直接由 representation 构造问题 |
| 源 `episode.h5`、`trajectories.h5`、`masks.h5`、`regions.json`、`evidence.json` | 提取技能时使用的演示记录、区域及事件依据 | 技能导入/导出组件产生上述技能包；独立技能包加载按其成员文件读取 |

`C_D` 表示源工具相对源物体的变换，`H_g` 表示源工具相对抓取功能坐标的变换，`H_m` 表示源物体相对操作功能坐标的变换。它们是参考运动的内容。源 `release_time` 决定参考受控区间的结束时刻；源开爪命令保留自身时刻。两者共同保留，并分别表达源物理事件与源控制意图。

## 3. 当前感知与 grounding 文件

| 文件 | 内容与含义 | 直接读取及计算 |
|---|---|---|
| `rgb.png`、`depth_m.npy`、`calibration.json` | 当前彩色图、米制深度、相机内参 `K` 与 `T_world_camera_cv` | 回投影、区域迁移、自过滤与 TSDF 建图 |
| 检测目录、`selection.json`、`*-owner-mask.npy` | 当前图像检测结果、所选对象身份与像素归属 | `load_detections/prepare_selected_observation`；规划入口据此分配对象和背景 |
| `observations/<view>/owner-*-semantic-mask.npy` | 本次传入的新建实体原始语义 mask；包括过滤未接纳的原语义像素 | `save_observation_partitions` 写出；可作为以后 `--owner-mask` 输入 |
| `observations/<view>/background-candidate-mask.npy` | 原始背景候选，尚未应用机器人过滤与边缘保护 | `save_observation_partitions` 写出；可作为以后 `--background-mask` 输入 |
| `observations/<view>/background-mask.npy` | 本次经过机器人过滤与一次边缘排除的实际积分分区 | 建图决策诊断；复用背景模型时无新分区输出 |
| `observations/<view>/partition-report.json` | 实体 ID 对应的原 mask 路径、两种背景路径、文件职责及分区统计 | 指定后续复用文件；缺少本次输出的背景路径为 `null` |
| `*-observation.npz`、`*-feature.npz` | 回投影点、原像素、有效性与实体归属；MARCO 图像特征 | 区域候选、图像对应和评分；是感知计算结果 |
| `surface.npz` | 多视角表面元素、观测像素、视图身份、观测与元素对应 | 区域几何和各图的区域投影 |
| `region.npz` | 选定功能区域的表面元素掩码等结果 | 当前功能区域、支撑面选择与可视化 |
| grounding 的 `report.json.functional_frame` | 迁移所得当前功能坐标 `T_world_region` | 实验 `reference_inputs` 实际读取并传给 `generate_reference`；该字段是计算输入 |
| 关系 JSON 中的 `J`、`A` | `J` 为当前物体坐标到源物体坐标的明确对应；`A=T_eef_body` 为名义附着 | 参考生成、任务点坐标与附着模型 |
| `reference.npz` | 保存采样后的工具/物体参考、时间和相关数组 | 用于结果保存及指定入口的对照；是否重新读取由调用代码决定，不能替代连续路径对象的实际消费说明 |

感知入口还依赖现有 SAM、MARCO 等本地模型权重和运行配置；它们提供图像分割与特征计算能力。已保存特征属于某次图像计算的结果，权重属于算法资源，两者职责不同。

## 4. 规划文件

| 文件 | 内容与含义 | 直接读取及计算 |
|---|---|---|
| `input-manifest.json` | 工作根目录、技能段、候选及图像/场景路径 | 记录入口选择明确任务与文件；清单本身不是几何 |
| `scene.json` | 当前对象身份、功能绑定、入口、目标、名义附着、任务点、支撑几何与任务容差 | `prepare_candidate/build_planning_problem` 或实验 `build_problem`；形成完整参考、明确硬要求及软偏好 |
| `feedback.json`、`robot_feedback.json` | 具名机器人 `q/dq`、手指状态、设备时间和标定 FK | 固定规划入口、机器人/手指几何与执行续接 |
| `robot-fixture.json`、纯运动学 YAML/URDF | 关节树、坐标、标定、关节限制及明确实体对策略 | cuRobo FK/IK、Pinocchio FK 和设备绑定 |
| robot-only 模型及其引用的官方机器人组件 | 当前固定机器人的官方原形状、局部变换、原生逐实体配对、固定安装身份及固定手指机构拓扑 | `robot_validation_fixture` → `RobotGeometryChecker` / `RobotCollisionShape` 与 `tool_geometry_names`；跨任务复用，供规划、机器人检查、图像过滤及工具成员声明 |
| `contact-bindings.json` | 必填 `contacts/source`，可选 `grasp_activations/initial_support_relations`；完整源任务特征、有限目标区域、实体身份及活动区间 | `load_contact_bindings` → `ContactRule`、原抓取条件绑定及初始支撑关系；原任务特征和当前区域分别保留 |
| `observed-owner.json` 与所引用原始场 | 已保存实体模型、局部坐标、位姿路径与固定附着 | `load_observed_owner` 直接读取，后续技能复用相同场 |
| `unknown-policy.json`、`fine-regions` JSON（可选） | 未知空间的明确处理策略；指定局部细化请求 | 场景构造和覆盖查询；策略允许尝试不表示该区域已观测为空 |
| `finger-targets.json`、`execution-limits.json` | 原事件对应手指目标；设备、跟踪、停止及事件约束 | 手指曲线、候选检查和执行绑定 |
| 关节参考 NPZ（可选） | 具名时间/进度上的关节参考 | `load_joint_reference` 提供候选参考 |
| `continuation.json`、`rebind` JSON、`request-ledger.json`（可选） | 已发生事件、实际反馈、显式续接变化及累计工作 | 当前请求续接、绑定与预算计算 |

技能表示中的功能点和接触定义提供“要做什么”的几何表达，现场碰撞模型提供“本次模型中的实体占据”。将它们变换到共同坐标后分别消费，不以改变碰撞表示为由删除技能内容。

### 4.1 已生成参考到规划要求的实际文件路径

当前 [put_prepare.py](../../../sim-version-plans/2026-10-18-shared-scene-toward-base/put_prepare.py) 通过 `reference_inputs()` 调用 `generate_reference`，随后由 `build_problem()` 直接构造 `PlanningProblem`。该路线使用源技能包中的 `representation`，工具参考、物体参考及目标取样共用返回的连续路径对象。具体职责如下。

| 文件或任务声明 | 内容与含义 | 直接消费与输出 |
|---|---|---|
| `2026-10-18-shared-scene-toward-base/skills/put_mug_on_shelf` | 源运动、功能坐标和交互/释放时刻 | `load_grounding_skill` → `generate_reference`；交互时刻另用于位姿要求和支撑要求的进度 |
| `2026-10-18-shared-scene-toward-base/put-grounding-001/report.json` | 当前迁移得到的 `functional_frame.T_world_region` | `reference_inputs` → `generate_reference(T_world_function=...)` |
| `2026-10-19-put-planner-audit/prepared-inputs/relations.json` | 当前到源物体坐标对应 `J`、声明的 TCP 到物体变换 `A` | `generate_reference` 生成刚性一致的参考；`PoseEventRequirement.tool_from_function` 使用同一 `A` |
| 同一 `prepared-inputs/feedback.json` | 从已保存抓取末态接续的机器人 `q/dq`、标定 TCP 位姿及具名关节 | 参考入口、`PlanningProblem.start_q/start_dq` 与坐标标定 |
| 同一 `prepared-inputs/scene.json` | 支撑平面、有限边界、源任务点、接触面及其原容差 | `SupportRequirement` 与附着条件 |
| 同一 `prepared-inputs/scene.json` 的 `pose_event_tolerances` 及[声明来源](../../../sim-version-plans/2026-10-21-yaw-tolerance/plan.md) | 材料原点距离 2 mm、消去自由 yaw 后总倾斜 1°、yaw 容差 π | 两个 `PoseEventRequirement` 读取数值；yaw 轴为观测支撑法向在目标材料坐标中的表示，来源随任务保存 |

表中目录均相对工作区 `sim-version-plans`。两个位姿目标来自 `object_path.sample_numpy(s)`：一个在源交互进度，另一个在受控参考终点进度 1，当前对应释放边界。被比较的候选是物体表示坐标系 `C` 在规划世界 `W` 中的位姿 `T_W_E(q)A`，其中 `E` 是 TCP，`A=T_E_C`；C 的原点由实体表示坐标确定，坐标与误差定义见[输入与任务模型](01-inputs-and-task-model.md#41-目标坐标中的可设置-yaw-容差)。原 `SupportRequirement` 与这两个要求分别保存，`hard_components` 默认 body/footprint，gap/normal 按原阈值提供 `quality` 阶段偏好；当前生产调用固定使用 `feasibility` 阶段。

`prepared-inputs/reference.npz` 保存参考采样及来源记录；当前目标构造实际查询内存中的连续 `object_path`。`source.reference_scope` 分别记录 `interaction` 和 `controlled_reference_endpoint`，`source.target_source` 记录连续路径及事件进度这一计算来源。事件进度通过规划器的固定时钟映到精确查询时刻，受控参考终点与机器人稳定尾段最后样本分别记录。

## 5. 输出、执行与评价

`recorded_task_planning.save_curve` 生成 `trajectory.npz` 与同名 JSON。NPZ 保存关节名称、时间、位置、速度和进度等数组；JSON 保存分类、执行绑定、问题和场景、事件、验收与设备限制。`load_saved_trajectory` 读取二者恢复正式执行输入。`diagnostic-trajectory.*` 保存拒绝候选的诊断数据，其执行资格与正式曲线不同。

`planning-report.json`、场景快照 NPZ、自过滤与分区文件记录计算结果。场景绑定中的 owner query 保存用于复现的几何及坐标；某文件以后被显式加载时，它同时成为该后续阶段的输入。来源路径和实际文件读取须分别说明。

任务快照中的支持要求保存 `hard_components`，续规划义务记录保留同一字段。`reference_ik_queries` 保存普通参考 IK 的原成功标志、原生连杆位置与真实转角误差、位置域资格、二次位姿能量，以及下一查询的跟踪来源。`warm_start_count/generated_seed_count` 表达固定种子总槽位中的实际分配；`proposal_candidate_indices` 与 `qualified_candidate_indices` 分别表示数值提议和原生收敛候选。`reference_ik_chains` 保存实际粗链、被选候选索引及 `SOLVED/APPROXIMATE_PROPOSAL/INTERPOLATED_PROPOSAL` 状态；`reference_seed_fits` 保存各粗曲线的构造结果。

`reference_initialization` 的 `iteration_allocation` 保存总更新数、原生内迭代块长度以及初始化/主求解分配；`scene_constraints="not_evaluated"` 表达该阶段的场景计算范围。`coarse/refined.seeds` 按实际 `seed_index` 保存自由系数、`trajectory` 标准快照、普通目标与分项、原生约束、完整任务有符号不等式以及连续位置/速度/加速度报告。快照保留具名关节、物理时刻、位置/速度、多项式系数、进度映射、事件、条件与手指时钟；其中 `executable_until/next_verification` 描述事件依赖定义的条件区间，完整执行资格由候选验收与绑定记录提供。`refined` 对应实际返回并交给完整主求解的系数。中断后只保留已完成记录，原始失败与缺失状态随阶段记录保存。

事件 IK 由 OMPL 搜索入口实际调用时，`functional_ik_candidates` 保存原目标身份、物理时刻、候选关节状态及独立任务/场景结论；其用途是搜索边界提议。完整轨迹的事件约束由同一任务允许集合求值。

完成固定评分的候选报告以 `fixed_objective/objective_components` 保存统一网格上的普通参考和关节成本；`returned_candidate` 指向实际返回曲线的完整验收报告。`native_metrics.rollout/interpolated.ordinary_costs` 分别保存两种实际网格已完成的积分值。`qualified_candidate_elapsed_seconds` 表示该候选在请求内完成验收、复制和执行绑定的时刻，`elapsed_seconds` 表示整次请求耗时。若后续改善预算耗尽，`improvement_end` 保存停止阶段和原因，正式曲线继续使用此前已完成资格的绑定；最终 `request_ledger.consumed_planning` 保存全部实际工作。原始曲线与原生改善曲线的时长、进度映射和来源分别随各候选保存。

`NativeReferenceCost` 的目标缓存由完整技能参考、请求标定和候选时钟生成，供同次求解的重复成本计算使用。`LocalGeometryModels` 属于一次数值求解中的临时优化记录，来源是相同观测场和本块实际候选姿态；场景持久化仍保存原 TSDF 样本和关系声明。原叶节点点集与数值查询复用记录是这些已有模型的计算中间量。

场景快照中的 `tool_geometry_names` 保存机器人适配器给出的工具成员，各 owner 的 `attachments` 保存刚性关系及释放边界。运行报告记录同一工具清单；场景查询报告中的 `internal_assembly_pair_count` 是按实际状态排除的组合体内部配对数。执行报告的 `declared_attachment_bindings` 保存原条件与实体关系的模型对应，独立于实际接触与效果记录。

仿真器通过 `scene.xml`、`initial-state.npz` 或 `episode.h5` 中的积分状态恢复运行环境；控制适配器向执行器提供机器人反馈。构造环境所用的内部状态不等于规划器可读的输入。相同的 HDF5 可以同时保存机器人编码器和环境对象状态，判断以实际读取的数据列与传递对象为准。

`NativeSkillPhysicsEvaluator` 消费仿真实际运行环境中产生的真值，以及用于解释对象身份、源技能和评价条件的记录，产生物理效果报告。`step_callback` 的返回值不控制机器人指令，记录错误单独保存；外层运行器的 `physics_success` 用于事后输出。源 `evidence.json` 在技能提取中的事件职责与评价器在当前运行中的读取职责分别成立。

实验运行器的 `snapshot.feedback_path` 明确指向图像采集时的机器人反馈，规划入口的反馈单独绑定。`owner-mask` 仅用于本次建立的实体；`background-mask` 接收原观测的背景候选。保存的实际积分背景已经过边缘保护，再将它作为候选会多排除一圈，因此后续读取 `background-candidate-mask.npy`。已有实体复用时无需再次提供其 mask；背景世界坐标与原自由射线不随运动实体改变。
