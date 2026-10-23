# 输入、参考与任务模型

代码基线：`01a6609`。本次仅文档提交回填实际实现提交号，代码内容与该基线相同。阶段实验与计划对照位于项目外 `sim-version-plans`。

## 1. 输入的来源与相互依赖

`build_planning_problem` 接收恰好一个技能实例。`skills[0]` 至少提供 `skill_id`、`program` 与 `scene`。`program` 保存技能种类、示教和任务关系；`scene` 保存当前绑定的对象身份、坐标、功能关系和声明附着。函数另外接收具名机器人 `joint_names/start_q/start_dq`、`tool_link`、`reference_frame`、`base_pose`、`tool_transform`、数值容差和可选关节耦合关系。

具有连续参考及明确任务声明的输入适配器也可直接构造 `PlanningProblem`：`reference_phase` 将连续工具参考组织为 `MotionPhase`，适配器分别传入位姿目标、支撑条件和原事件。两条构造路径最终使用相同的要求类型与规划消费者；本文第 3 节和第 4 节前半部分描述 `build_planning_problem` 的编译路径，第 4.1 节描述直接构造路径。

示教的 `frame_indices` 表示原记录帧身份，`timestamps` 表示对应源时间。二者保持严格递增且相互对应。功能事件与夹爪事件保留原帧身份，`source` 和 `provenance` 保存其生产依据，供输入来源追踪。

所有 skill representation 均为需要保留的输入，包括由仿真示教提取的源轨迹、区域、坐标关系、任务几何和交互/释放事件。源事件说明示教中发生了什么，参与当前参考和事件编译；它不表示当前执行已经产生相同效果。禁止额外进入求解的是仿真实际运行环境中产生的真值。文件中的源表示、当前传感观测及当前运行环境状态按其生产与消费职责区分，完整文件目录见[外部文件与数据职责](08-external-files-and-data-roles.md)。

| 输入 | 具体含义与生产者 | 直接消费者 |
|---|---|---|
| `program.skill_kind` | 抓取或持物操作的已选技能类别 | `ground_program`、`build_planning_problem` |
| `program.demo` | 源位姿/坐标序列、时间、功能事件、开闭指令、释放边界 | 参考生成与事件编译 |
| `scene.carrier_id` | 当前被操作实体的明确身份 | 功能关系、附着、场景 owner 对应 |
| `scene.reference_id` | 当前功能目标对象身份；持物操作时与 carrier 不同 | 功能绑定与目标关系 |
| `scene.reference_frame` | 全部当前任务几何共同使用的坐标系名称 | 参考、规划问题与碰撞场景的一致性检查 |
| `scene.eef_pose` | 当前机器人关节反馈经标定 FK 得到的工具位姿 | 参考入口与声明附着的模型一致性 |
| `scene.carrier_pose` | 抓取参考所使用的当前物体表示坐标 | 抓取区域变换与独立抓取目标 |
| `scene.initial_pose` | 持物阶段的名义物体入口；必须与工具位姿和声明附着一致 | 持物参考、功能约束与 owner 初始坐标 |
| `scene.holding.T_carrier_eef` | 工具坐标到物体表示坐标的声明变换 | `ground_program` 的附着换算与持物任务编译 |
| `scene.grasp_goal` | 抓取的独立目标变换、位置/角度允许范围与来源 | `PoseEventRequirement` 编译抓取目标要求 |
| `scene.functional_bindings` | 每个功能事件所需的当前材料坐标、对象映射或支撑几何 | `_functional_events`、功能要求编译 |
| `scene.body_points` | 当前物体坐标中的任务几何采样点 | 支撑与内部区域约束 |
| `scene.support`、`on_geometry` | 有限支撑面、接触点、外法线、间隙与方向容差 | `SupportRequirement` |
| `scene.motion_model` | 当前机构类型、轴、原点、坐标和限制及其来源 | 机构参考与 `MechanismRequirement` |
| `execution_events` | 调用者明确指定的动作、停止或独立证据依赖 | `PlanningEvent` 编译相应事件 |
| `coordinate_directions` | 机构坐标在给定区间的方向、净变化或局部速度要求 | `CoordinateDirectionRequirement` |

`tests/planning` 中的实验运行器 `run_recorded_skill.run_prepared` 将编译后的问题与机器人记录连接。`robot_validation_fixture.build_robot_validator` 读取已知机器人组件、关节树、标定及原生实体对策略；`observed_robot_fixture.load_observed_owner` 读取已有任务实体场与固定关系。实验适配器与当前持物计划的对应关系见[架构与计划差异审查](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。

## 2. 固定坐标下的示教表示

`representation.build_representation` 的输入是源时间、帧身份、技能种类，以及可用的物体、参考实体和工具世界位姿，另有功能局部坐标 `L_G/L_R` 和事件时间。`carrier` 表示被抓取或被操作物体，`reference` 表示承载目标功能区域的实体，`eef` 表示机器人工具坐标。`L_G=T_carrier_FG` 将抓取功能坐标 `FG` 中的点变换到 carrier 坐标，`L_R=T_reference_FR` 将操作目标功能坐标 `FR` 中的点变换到 reference 坐标；二者均是固定局部变换。对于源时间 `t`，函数计算：

```text
C_D(t) = inv(T_world_carrier(t)) T_world_eef(t)
H_g(t) = inv(L_G) C_D(t)
H_m(t) = inv(L_R) inv(T_world_reference(t)) T_world_carrier(t)
```

其中，`C_D=T_carrier_eef` 表示源工具在被操作物体坐标中的位姿，`H_g=T_FG_eef` 表示源工具在抓取功能坐标中的位姿，`H_m=T_FR_carrier` 表示源物体在目标功能坐标中的位姿。抓取分支构造 `H_g`，持物操作分支构造 `H_m`；工具和物体位姿同时存在时构造 `C_D`。返回的几何表示保留源时间、原帧身份、原位姿、可用相对关系、事件及缺失字段，供参考生成调用。

`representation.sample_poses` 对明确时间上的刚体位姿插值，平移和旋转分别计算。`trajectory_summary` 和 `_sampled_summary` 在接近/交互时刻提取几何摘要。`_component_path` 将输入数组组织成连续的 `PosePath`；`_right_path` 对连续位姿函数施加固定右乘变换，因此保留旋转偏置引起的弧线。

`generate_reference` 接收以上表示及当前功能坐标 `T_world_function`、当前到示教的物体坐标对应 `J`、声明附着 `A`、当前工具/物体位姿，以及可选机构模型与坐标。该接口以 `body` 命名被操作物体的表示坐标。`A=T_eef_body` 是物体坐标在工具坐标中的位姿，满足 `T_world_body=T_world_eef A`；`J=T_demo_body_current_body` 将当前物体坐标中的点映到示教物体坐标。将该表示绑定到 `program` 接口的同一 `carrier` 坐标后，`A=inv(scene.holding.T_carrier_eef)`。

抓取名义路径由源抓取关系和当前功能坐标组合生成。对于自由持物路径，记源序列入口为 `t₀`、交互时刻为 `t_c`，函数首先构造目标功能坐标中的相对运动 `Q(t)=H_m(t)H_m(t₀)⁻¹`。由于同一物体坐标重定义会在这两个因子中相消，`Q` 保留相同的几何运动。随后定义附着在当前物体上的材料坐标 `M`：它在名义交互时刻与当前目标功能坐标重合，其局部变换为 `L_body_material=T_body_M=J⁻¹H_m(t₀)⁻¹Q(t_c)⁻¹`。函数依次组合当前功能路径、`Q(t)`、`Q(t_c)⁻¹` 和 `L_body_material⁻¹`，得到完整名义物体路径。给定声明附着 `A` 后，名义工具路径由该物体路径右乘 `A⁻¹` 唯一确定。父路径与固定右乘在每次求值时组合，旋转引起的 TCP 杆臂运动由同一函数保留。

对于转动或移动机构，输入提供源标量坐标与当前坐标。参考在标量坐标中保留源运动增量与累计转角，再通过当前机构模型的正运动学生成位姿。`contact_motion=True` 与固定附着 `A` 是互斥输入分支，分别对应时变接触关系与刚性附着关系；同时给定时产生 `ValueError`。

功能位姿参考可在技能表示中声明 `is_initial_relative_rotation_rpy`。该字段必须为三个布尔值，顺序为 roll、pitch、yaw，表示初始朝向的分量来源。旋转分量使用功能参考坐标 F 中的 XYZ 外禀约定，即 `R=Rz(yaw) Ry(pitch) Rx(roll)`。`true` 选择本次运行初态的对应分量，`false` 选择原功能参考迁移入口的对应分量。选择发生在入口；后续运动通过完整旋转矩阵保留相对于入口的旋转变化。标量机构参考的姿态由其正运动学定义，该声明适用于功能位姿参考。

对于物体运动，旋转对象是上述固定材料功能坐标 M。记原名义路径为 `T_W_M(s)`，功能路径为 `T_W_F(s)`，技能进度为 s。原初始旋转在 F 中为 `R_n0=R_W_F(0)^T R_W_M(0)`，当前初始旋转为 `R_c=R_W_F(0)^T R_W_body(0) R_body_M`。当前物体模型位姿由已有观测或 `T_W_E A` 给出。`_initial_rotation_offset` 将二者转换为上述 RPY，按布尔值选定入口 `R_a`，计算固定修正 `D=R_a R_n0^T`，同时返回选择依据与修正矩阵。抓取工具参考使用同一规则，移动坐标 M 取已标定的 E。

`_RotationOffsetPath` 保存原连续材料路径、连续功能路径与 D。每次查询时先求两个父路径，再计算 `R'_W_M(s)=R_W_F(s) D R_W_F(s)^T R_W_M(s)`，材料原点位置 `p_W_M(s)` 保持原函数。其 `sample` 返回保持 Torch 进度导数的位置和 wxyz 四元数；继承的 `sample_numpy` 返回同一位姿函数。该类型的 `poses` 为自身节点处的精确缓存，物体和末端变换通过普通子 `PosePath` 在每次查询时右乘。因而旋转偏移造成的物体原点与 TCP 杆臂位移仍属于同一连续函数。当前入口的三个分量全部选择运行初态时，材料入口朝向与本次模型初态一致；仅 yaw 选择运行初态时，D 是 F-z 轴的固定旋转，原参考的 roll/pitch 曲线保持原值。

`generate_reference` 在字段的实际消费处校验布尔类型与长度，返回的 `rotation_transfer` 保存布尔数组、RPY 约定、两种入口、选定入口和 D。未声明旋转选择的功能参考按原功能坐标组合。物体原始 `H_m`、工具原始关系、时间与事件仍是源表示；`bound_*`、生产路径、交互目标和释放终点共同使用生成后的名义旋转。位姿要求和执行条件分别由其原接口声明。

`config.ground_skill.ground_skill` 读取技能、当前 RGB-D 与标定、机器人反馈及明确的 J/A 关系，调用图像实体选择、特征提取和 `run_region_transfer` 获得 `T_world_region`，再传给 `generate_reference`。它将数组写入 `reference.npz`，将旋转选择依据写入 `report.json.reference_rotation_transfer`，将连续工具路径与边界传给 `reference_phase`，最后调用参考网页输出。这个报告字段是上述参考运算的可追溯结果，其坐标与计算入口与路径对象一致。

输出包括 `object_path/eef_path`、对应时间和名义位姿数组、`object_boundaries/eef_boundaries`、`boundary_checks`、`summaries`、`missing` 与 `status`。受控路径的终点为源释放时间与显式持物终点中的较早者；`bound_*` 另保存原名义绑定诊断。每份边界声明包含 `initial_pose`、带局部进度/位姿/来源的 `goals` 以及声明来源；`boundary_checks` 记录 `declared_for_physical_time_binding`。缺少交互时间、当前入口或必要坐标对应时，函数将相应字段列入 `missing`。

`reference_phase` 复用完整 `eef_path`，将 `eef_boundaries` 复制到 `MotionPhase.metadata["reference_boundaries"]`；源帧、源时间和原事件分别保存。`reference_visualization.write_reference_html` 在真实源/目标图像上叠加原功能区域、采样点和名义路径，直接从该边界声明投影工具入口与每个事件目标。这个页面的时间是源示教时间；物理时间连接在规划器绑定后产生。

## 2.1 实际物理时钟上的工具参考

`timed_reference.BoundToolReference` 的输入为名义阶段、归一化物理时间到技能进度的 `progress_u/progress_s`、实际时长 `duration`，以及当前机器人 TCP 位姿和世界速度。`planning._reference_entrance` 使用同一官方机器人模型和具名关节 `q/dq` 运行 Pinocchio FK，读取工具连杆原点的 `LOCAL_WORLD_ALIGNED` 速度，经基座标定旋转到 W，并将 `ω_W × r_W` 加到线速度；`r_W` 为连杆原点到标定 TCP 的世界杆臂。输出 `initial_twist=[v_W,ω_W]` 的单位分别为 m/s 与 rad/s。

设名义 TCP 位姿为 `(p_n(t),R_n(t))`，其中 `p_n` 是 E 原点在 W 中的位置，`R_n` 将 E 坐标映到 W。接口使用世界加性平移 `b(t)` 与左旋转 `E(t)=Exp([φ(t)]×)`，计算 `p=p_n+b`、`R=E R_n`。名义物体、机构坐标及父路径组合在实际 `s(t)` 上直接求值。名义线速度和世界角速度由 Torch 对完整组合函数的单侧时间导数获得，保留进度停留段、斜率变化和旋转杆臂。

入口的目标速度是测得机器人速度；每个声明事件选择右侧名义物理速度作为左右输出的共同速度。对每个边界，修正位置为目标位置减名义位置，修正旋转为 `R_target R_nᵀ`；其导数满足 `ḃ=v_target−v_n`、`φ̇=J_l(φ)⁻¹(ω_target−Eω_n)`。这里 `J_l` 是 SO(3) 指数映射的左雅可比，代码以 `pin.Jexp3(-φ)` 求值。各相邻边界之间使用 SciPy `CubicHermiteSpline` 构造修正的三次多项式，Torch 负责按物理秒求值。

事件时间取其声明进度的首次到达时刻。恢复名义后缀的拼接以右侧名义速度为共同边界，左段修正补偿原名义左右导数差；拼接处及其后直接返回原名义函数。位于整个请求终点的声明目标可以采用不同于名义终点的位姿，此时修正函数在终点仍生效；位于请求内部的最后一个拼接则须恢复名义位姿，以连接其后的名义路径。多个边界使用同一规则，修正旋转坐标在相邻边界间选择连续分支；原机构累计转角保留在名义坐标中。当前边界阶次为一阶，连续性结论覆盖实际声明的拼接处，其他源内部断点保留其原导数。`source_boundaries` 保留原完整声明，`boundaries/diagnostics` 保存实际使用的物理边界、左右速度和来源。

`sample(times_s)` 返回 W 中的位置 `[...,3]` 与单位 wxyz 四元数 `[...,4]`；`sample_numpy` 返回同一函数的 `[...,4,4]` 齐次变换。参考取样和规划结果记录消费该物理时间接口。保存候选使用自身的时长和进度映射构造同规则参考；没有显式边界声明的普通阶段继续提供其名义值。

续接区间位于最后一个声明拼接之后，或本区间唯一拼接位于零时刻时，返回原名义后缀，并以 `entrance_twist_bound=false` 记录当前区间的入口连接范围。存在未来声明拼接时，实际机器人入口连接到该拼接，并在已声明边界提供一阶连续性。原零时刻事件继续保留其技能表示和任务接口职责；参考对象提供该物理时钟上的名义位姿，当前 FK 与名义目标之差用于事后解释。

## 3. 当前 program 的几何绑定

`program.ground_program(program, scene)` 是技能编译所实际调用的入口。抓取分支调用 `grasp.ground_grasp`，得到当前末端参考、独立抓取目标和开闭事件。持物分支读取声明的 `T_carrier_eef`，检查 `initial_pose @ T_carrier_eef = eef_pose`，然后调用 `_object_reference` 生成物体参考，再通过 `attached_eef_reference` 得到工具参考。该等式检查输入所声明的物体坐标、工具坐标与附着变换之间的代数一致性。

`_object_reference` 根据任务选择相应数值过程。完整功能坐标任务使用事件变换；`on` 任务使用有限支撑与物体接触采样，允许支撑面内的自由度；关节坐标任务使用机构模型与明确标量目标。`_functional_events` 把每个源功能事件与 `scene.functional_bindings[event_id]` 对应；`_joint_reference` 处理机构坐标分支；`_event_transfer` 将参考入口与功能事件几何送入连续适配。`ground_object_program` 返回物体参考，工具参考由具有附着声明的接口计算。

`motion.MotionModel` 保存 `free/prismatic/revolute/planar` 等模型的几何参数和允许误差。`poses` 将模型坐标变为位姿，`coordinates_of` 将位姿映回模型坐标并检查模型误差，`fit_motion/analyze_motion` 用已有位姿数据估计或比较模型。拟合结果量化给定位姿数据与候选机构模型的相容性，并返回模型参数及误差。`free_transfer_parameters` 和 `transfer_motion` 提供自由运动适配；`function_motion.transfer_function_motion` 组织多个明确功能目标的适配计算。

## 4. 任务编译如何产生硬要求

`build_planning_problem` 先检查技能声明，再调用 `ground_program`。它将有效持物时间分成 `MotionPhase`，建立原帧到阶段进度的映射，同时生成目标、过程关系、事件和条件。

抓取任务生成一个终点 `PoseEventRequirement`，其目标来自 `scene.grasp_goal`。每次 close 还保留该指令进度上的名义抓取目标，用于规划几何约束，并生成 `grasp_contact_activation` 以连接指尖接触规则与原 close 身份。`candidate_attachment` 及末端效果事件保存抓取评价的对象身份与关联事件。

持物任务生成 `declared_attachment`，记录 `T_carrier_eef`、对象身份和模型来源。阶段元数据保存附着区间及释放条件身份。任务为 `on` 时生成 `SupportRequirement`；标量机构目标生成 `CoordinateEventRequirement`；其余完整功能坐标目标生成 `PoseEventRequirement`。需要物体始终处于指定凸区域时，生成 `InteriorRequirement`。开爪输入描述设备指令，释放边界描述物体运动阶段的终点，二者分别编译并保留各自事件身份。

| 要求类型 | 输入及计算对象 | 输出与使用位置 |
|---|---|---|
| `PoseEventRequirement` | 一个进度、功能坐标到工具坐标的变换、目标位姿、位置/旋转容差，以及成对声明的 `yaw_axis/yaw_tolerance_rad` | `error_components` 返回位置、完整旋转或剩余倾斜及 yaw；`component_tolerances` 选择受约束分量；`evaluate` 汇总各分量超差 |
| `SupportRequirement` | 进度、物体坐标在工具坐标中的变换、物体点、接触点、外法线、有限支撑面、容差和 `hard_components` | `inequality_components` 保留逐点与逐支撑边界的 gap/body/footprint/normal；`raw_inequalities` 给出各组最大值；显式硬分量进入硬条件，其补集进入 `quality` 偏好 |
| `InteriorRequirement` | 区间、物体点、凸区域半空间和内缩距离 | `raw_inequalities` 保留每个点与半空间的条件；`evaluate` 返回其最大正违反量 |
| `PoseProcessRequirement` | 区间、相对位姿及六维允许范围 | 过程中的平移和转动误差，只有明确受约束方向进入残差 |
| `MechanismRequirement` | 当前机构模型、附着、坐标限位、入口坐标及方向要求 | 流形偏离、坐标限位与方向条件；`direction_inequalities` 保留可行侧，`direction_residuals` 给出正违反量；完整路径保留角度展开历史 |
| `CoordinateEventRequirement` | 某机构、进度、绝对或入口增量目标、容差 | `signed_error` 给出有符号坐标差，`error` 给出绝对值；转动时区分局部姿态和累计转角 |
| `CoordinateDirectionRequirement` | 区间、坐标列、方向、最小变化或速度、容差与来源 | 净变化、单调或局部变化率等明确声明的方向条件 |
| `JointCoupling` | 具名关节、矩阵和目标 | `q_selected @ matrix.T - target` 的容差残差 |
| `PlanningEvent` | 原技能/事件/发生次数身份、进度、源帧、种类、可选停止与效果依赖 | 决定指令时刻，并按显式字段编译停止要求或下一段动作的独立依赖 |
| `EventTimeConstraint` | 检查对象、绝对或相对时间窗、可选起源事件 | 规划时检查名义时钟；执行只消费允许的指令/独立效果时间，几何目标到达时间仅事后评价 |

`PoseEventRequirement.error_components` 以候选工具位姿和固定变换求材料坐标 C 的位姿。位置分量为 C 原点到目标原点的欧氏距离。未声明 yaw 轴时，旋转分量为目标与候选之间的完整 SO(3) 最小转角。有轴时，旋转分量为去除绕轴转动后的剩余倾斜角；yaw 分量另行计算。`errors` 返回位置与该旋转分量；`errors_numpy` 为准备和诊断提供同一几何的 CPU 接口。`evaluate` 对 `component_tolerances` 中每个分量计算 `max(0,error−bound)` 后求和。`SupportRequirement.evaluate` 对 `hard_components` 指定的原始有符号条件取正部后求和；默认硬分量为 body、footprint。`constraint_components` 读取同一声明下的未归约硬分量；`normalized_inequalities` 按长度或法向弦长缩放四组最大值，`optimizer_preferences` 在 `quality` 阶段读取硬分量的补集。

`NumericalTolerance` 控制浮点计算比较，目标容差规定任务允许误差，支撑容差规定允许间隙；实体距离的模型余量由观测碰撞查询单独应用。各量由对应接口保存。

### 4.1 目标坐标中的可设置 yaw 容差

令 W 为规划世界系，E 为标定工具中心点 TCP，C 为受约束的材料坐标，C* 为该材料坐标的目标。固定关系 `A=T_E_C` 与机器人正运动学给出 `T_W_C(q)=T_W_E(q)A`。`target` 保存完整的 `T_W_C*`。`yaw_axis` 是 C* 系中的单位向量 a，`yaw_tolerance_rad=β` 是主值绕轴转角的绝对值上界，范围为 `[0,π]`。这两个字段共同声明；没有轴声明时，事件表达完整姿态精度。

相对旋转 `R=R_W_C*ᵀ R_W_C` 的单位四元数写为 `(w,v)`，其中 v 为 C* 系中的三维向量。令 `b=v·a`，绕轴转角为 `ψ=atan2(2wb,w²−b²)`，剩余旋转为 `R_s=R Rot(a,−ψ)`。`rotation_tolerance_rad` 约束 `R_s` 的最小转角 θ，`yaw_tolerance_rad` 独立约束 `|ψ|`。这是有限旋转分解。因而带轴声明中的 1°表示 θ≤1°，不是某个 RPY 欧拉角，也不是 roll、pitch 各有 1°余量。绕垂直于 a 的轴转过半周时，绕轴分解的 ψ 不唯一；实现取 ψ=0 并保留 θ=π。

β=π 覆盖整周，`component_tolerances` 只包含位置和 θ。功能约束、验收以及可选目标居中项均按此分量集合计算。β=0 要求绕轴偏差为零，`constraint_components` 使用 `signed_yaw=0` 表达该等式；β=π/6 给出 ±30°范围。`full_rotation` 提供完整转角的解释性数值，`signed_yaw` 同时保留绕轴误差的方向。源完整姿态和参考路径仍保存原值。

`_functional_events` 保留事件的轴与容差，`_material_pose_requirement` 用当前材料变换和功能目标构造上述要求。`_object_reference` 在入口以及参考功能点上使用同一分解；同刻事件的矛盾检查使用包含完整 yaw 范围的旋转球上界。`build_planning_problem` 从原事件复制轴与容差，保留 C* 的局部坐标语义。有限支撑声明由 `SupportRequirement` 独立表达；需要同时表达姿态集合时，调用者另行传入 `PoseEventRequirement`。

当前连续参考适配器 [put_prepare.py](../../../sim-version-plans/2026-10-18-shared-scene-toward-base/put_prepare.py) 的 `reference_inputs()` 读取完整技能表示、迁移功能坐标、声明关系 J/A 和冻结机器人反馈，调用 `generate_reference` 得到 `object_path/eef_path`。`build_problem()` 在交互进度和受控参考终点建立两个材料姿态要求。目标为 `object_path.sample_numpy(s)`，轴为 `R_W_C*ᵀ n_W`，其中 `n_W` 是观测支撑面的世界法向。三个容差直接读取 [scene.json](../../../sim-version-plans/2026-10-19-put-planner-audit/prepared-inputs/scene.json) 的 `pose_event_tolerances`：当前位置 0.002 m、剩余倾斜 π/180 rad、yaw π rad。该 1°衡量相对参考的倾斜；支撑面的法向与接触间距仍使用独立支撑声明。源时间、交互和释放事件继续决定查询进度。

## 5. PlanningProblem 的运行接口

### 5.1 候选状态、坐标和固定查询

`sample_reference(s)` 返回当前进度的完整名义工具路径；物理时间适配由第 2.1 节的 `BoundToolReference` 读取同一阶段及显式边界后完成。`source_time_to_progress` 和 `progress_to_source_time` 在保存的源时钟中转换。`transform_tool_pose`、`reference_to_link_pose` 和 `transform_tool_velocity` 将机器人计算转换到任务参考坐标，或将任务目标转换回机器人连杆坐标；这些变换依赖构造时固定的 `base_pose/tool_transform`。普通阶段的 `PosePath` 提供名义数据，`PoseProcessRequirement.reference` 则是显式任务关系中的目标函数；二者由所属声明确定计算职责。

完整候选查询包括工具位置、wxyz 姿态、技能进度、实际时间、具名关节位置与速度，以及声明坐标速率时需要的工具线速度和角速度。工具位姿与 twist 均在 `reference_frame` 中表达；twist 的线速度对应标定工具原点，时间单位为秒。局部机构速率使用实际机器人 Jacobian 与关节速度所得的工具 twist，保留工具和附着实体之间的旋转杆臂。

`anchors` 合并请求首尾、目标进度、事件、过程边界及显式 `PoseProcessRequirement.reference` 的节点。`verification_issues` 在查询布局建立时检查这些精确进度及角度机构的实际起点历史；随后同一次数值求解使用该固定布局。点事件采用其精确进度上的最早实际时间，重复记录同一时刻时只选一项。过程条件在所属区间的每个给定查询上计算。

### 5.2 完整有符号任务条件

`constraint_components(position, quaternion, progress, joint_position=None, joint_names=None, joint_velocity=None, *, trajectory_time, tool_linear_velocity=None, tool_angular_velocity=None)` 返回 `(equalities, inequalities)`。两者均为名称到 Torch 张量的字典，分别表达 `value=0` 和 `value<=0`。输入姿态分别为 `[...,3]` 和 `[...,4]`，进度和实际时间具有相同的批量/查询维度。给定声明与固定查询布局后，字典顺序和每组张量形状保持固定；对值展平并拼接即可构成数值求解器的条件向量，Torch 自动求导保留它们对候选状态的 Jacobian。

生产调用关系是 `planning._ConstrainedPlanner._bind_computation` → `unified_solver.TrajectoryComputation` → `PlanningProblem.constraint_components`。前者合并普通物理积分查询、任务锚点和场景锚点，按实际时间排序，随后一次校验 `verification_issues`。`TrajectoryComputation.states` 用同一固定物理时钟的三次曲线系数生成 `q/dq/ddq/jerk`。`robot_tool_states` 补齐场景声明的名义手指坐标，交给 `PinKinematics.states` 计算参考坐标中的工具连杆位姿和速度，再组合固定工具变换；工具线速度包含角速度与固定杆臂的叉积。`task` 将完整任务字典按插入顺序展平，`solve_trajectory` 调用 `task_jacobian`，在同一次任务求值图上逐列计算对自由曲线系数的导数，交给 SciPy SLSQP。令 `c(x)` 为原任务行向量、`x` 为自由系数、`J=∂c/∂x`，一次反向传播建立 `p(x,v)=Jᵀv`，其中 `v` 是与 `c` 等长的独立权向量。随后仅对 `v` 求每个 `p_i` 的导数，得到 `J` 的第 `i` 列；该计算使用机器人运动学的一阶导数。

`TrajectoryComputation.objective_components` 在同一候选上计算三部分：普通积分查询上的 STATE 关节运动质量、`optimizer_preferences(phase="quality")` 的声明任务偏好、场景已观测几何的搜索目标。STATE 的权重、激活距离与物理时间因子来自既有 cuRobo 配置，梯形积分按物理时长归一化；补入事件查询不会改变普通运动质量的积分权重。工具和物体的几何允许条件仍由对应原任务行及场景验收计算确定。候选保存后的复核调用同一 `robot_tool_states`，输入为保存曲线自身的时间、关节状态和技能进度。

数值求解器从完整任务值确定等式与不等式行数，然后根据 SciPy 密集工作区、约束 Jacobian、矩阵拼接和运动学反向传播的同时存活数组进行主机内存准入。调用方在每个串行候选求解前重新计算场景的待分配固定材料样本与一次几何/FK 临时图，作为 `solve_trajectory` 的必需参数 `pending_geometry_bytes`；求解器将该项与自身增量合并，在任何导数分配前检查主机预算。`task_jacobian` 建立第一次反向图后再次检查主机预算，此时新增量只有最终任务 Jacobian、单个返回列与 Pinocchio 求导临时缓冲；原任务值、独立权向量、第一次反向图以及已分配的求解器和场景数组由当前 RSS 计入。`solver_storage_bytes` 的入口总预算还计入原任务连接向量和自由系数空间中的反向积。每次候选系数改变时，目标、任务与关节约束缓存同步清空。求解器报告分别保留库终止原因、实际系数变化和各部分数组容量，最终接受由原约束与保存曲线的复核决定。

固定查询绑定同时确定等式对自由曲线系数的结构依赖。现有零宽偏航等式在声明事件首次物理时间为零、且该时刻三次曲线的系数行恒为零、仿射偏置恰为 `start_q` 时，属于固定入口条件。`fixed_equality_rows` 记录这些原行，`variable_equality_rows` 记录其余行。求解前按原零宽条件核对固定行；非零或非有限值返回 `solver_called=false` 的入口不可行记录。满足的固定行从提交 SLSQP 的向量中消去，完整 `task` 和保存候选验收仍返回原行。这一消元由事件时间和曲线表示确定，当前某行导数恰为零不构成消元依据。

求解器向 SLSQP 提交 `f(x)/α` 及 `∇f(x)/α`，其中 `x` 是同一曲线自由系数，`f` 是上述三部分原目标之和，`α=max(1,|f(x₀)|,‖∇f(x₀)‖∞)` 在该次求解开始时确定并固定。由于 `α>0`，任意两个可行候选的目标大小关系与原目标一致，最优解集合相同；所有任务和物理约束保持原值。`objective_scale`、`normalized_objective`、`objective` 分别记录该常数、归一化目标和原目标。SLSQP 的 `ftol=1e-9` 是归一化数值问题的停止设置，候选排序使用原目标，候选接受使用原任务和几何验收。库的成功终止及近零等式残差各自记录其数值含义。


各行保留原物理单位和原允许范围。过程位姿的每个有限上下界分别成为有符号行，平移分量使用米、旋转向量分量使用弧度，并保留 `NumericalTolerance` 的分量 allowance。机构流形继续使用原位置范数和旋转范数容差；坐标限位使用每个原生坐标的上下界。关节耦合对每条 `Aq-b` 方程保留原 `numerical.joint` 双边区间；停止事件对每个关节速度保留原 `numerical.velocity` 双边区间。

| 条件组 | 计算与保留的行 | 范围 |
| --- | --- | --- |
| 进度 | `start_progress-progress`、`progress-end_progress` | 全部查询 |
| 位姿过程关系 | 每个有限下界、上界及其原分量 allowance | 声明区间内的查询 |
| 机构流形和限位 | 位置/旋转误差减模型容差；解缠坐标的双边限位 | 声明区间内的查询 |
| 方向要求 | 净位移下界、相邻查询的单调条件、实际瞬时坐标速率下界 | 原实际时间顺序和区间 |
| 支撑 | `hard_components` 指定的逐接触点、逐物质点、逐支撑边界及法向条件 | 支撑事件首次到达 |
| 内部区域 | 每个物质点与每条半空间的原有不等式 | 声明区间内的查询 |
| 材料位姿事件 | 位置范数、完整旋转或剩余倾斜、受限偏航分别减原阈值 | 位姿事件首次到达 |
| 机构坐标事件 | 实际解缠坐标对绝对目标或原入口增量目标的双边条件 | 坐标事件首次到达 |
| 关节耦合 | 每条具名关节线性方程的双边数值 allowance | 全部查询 |
| 停止 | 各关节速度的双边数值 allowance | 停止事件首次到达 |

零宽度 yaw 是有符号等式 `signed_yaw=0`；整周 yaw 容差通过分量集合保留自由转动。位置和倾斜误差仍按同一材料坐标计算。支撑和内部区域的逐点行保留可行侧导数，事件级或报告级最大值由对应报告函数计算。未激活过程查询保留值与导数均为零的行，维持固定布局。

角度机构的 `coordinates` 按实际时间排序候选姿态，以声明的实际入口坐标确定角度分支，再累计相邻旋转增量。完整坐标事件直接使用此解缠值，因而保留累计转数。`direction_inequalities` 对净变化保留其显式起点或原入口来源，对单调性比较相邻实际时间查询，对 `local_rate` 使用 `coordinate_rates` 映射后的实际工具 twist。`direction_residuals` 对相同结果取正部，用于原验收和方向偏好。

上述过程结果具有有限查询范围。相邻查询的单调性与给定时刻的瞬时速率分别具有对应范围；连续关节多项式的界限检查由关节曲线组件提供，几何区间由场景组件单独报告。场景附着、释放及接触规则使用 `conditions` 和阶段元数据，事件时间窗使用已绑定时钟；这些声明各自进入对应计算接口。

### 5.3 验收、诊断与目标投影

`residuals` 返回同名非负硬违反量，`evaluate` 汇总这些量。`point_goal_ids` 选择当前局部查询所对应的原目标身份；`direction_mode="point"` 用于独立状态，`"trajectory"` 用于具备物理顺序的完整路径。完整曲线查询补齐历史、速度和停止条件。各消费者读取同一 `hard_components` 和任务容差。

`measurements` 输出物理误差等解释性原量。`projection_requirements`、`project_residual`、`goal_requirements_at` 和 `goal_residual` 为事件 IK 与搜索状态检查提供声明条件及投影量；`goal_residual(normalized=True)` 依据原容差缩放超差正部，保留相同允许集合。`stop_progress` 收集明确停止事件。`bound_motion_block` 以最早尚未满足的独立效果依赖界定当前可执行区间，保留原目标身份、机构坐标和阶段进度；`requirement_consumption` 与 `remaining_obligations` 提供各项职责和剩余记录。

### 5.4 质量偏好与共用几何

`optimizer_preferences(..., phase="quality")` 返回每个查询的软目标值，包括支持硬分量补集的归一化正违反量平方、位姿和坐标事件对原目标中心的偏好、显式零宽过程关系的等式误差平方，以及声明方向的偏好。`phase="feasibility"` 保留过程等式和方向偏好；两种配置均使用同一硬条件声明。支撑的软补集继续按原 gap/normal 等几何、阈值和归一化尺度计算。

支撑的 `inequality_components` 是逐点硬条件与组最大值的共用计算；`raw_inequalities` 对这些行归约，`normalized_inequalities` 对原组最大值缩放。内部区域的 `raw_inequalities` 保留逐点半空间行，`evaluate` 取其最大正违反量。材料事件的全部消费者使用 `error_components`；方向约束、方向残差和方向偏好使用同一 `direction_inequalities` 计算。候选机器人状态是优化变量，技能表示、功能目标、允许范围和声明运动关系保持各自输入语义。

## 6. 支撑任务的几何输入契约

`on` 分支接收 `body_points`、物体接触区域点及其外法线、目标支撑面的有限区域。物体点与法线在物体表示坐标中定义，支撑面在任务共同参考坐标中定义。`SupportRequirement.inequality_components` 将物体点和接触点经过候选工具位姿与固定物体变换映入共同参考坐标，再逐项计算接触间隙、物体点高度、在支撑面内的投影范围以及材料外法线方向。`raw_inequalities` 对相同行取组最大值。

具体地，记候选工具位姿与 `tool_from_carrier` 的乘积为 `(R,p)`，物体点为 `x_i`，接触点为 `c_k`，物体接触面的单位外法线为 `n_b`。变换后的点为 `x′_i=R x_i+p`、`c′_k=R c_k+p`。支撑面由原点 `o`、单位法线 `n` 和平面正交基矩阵 `U∈R³ˣ²` 定义；其有限凸区域的第 `j` 条二维半空间写为 `a_jᵀy+b_j≤0`。记间隙容差为 `ε_g`，法向夹角容差为 `ε_θ`，数值位置容差为 `ε_num`，支撑边界内缩量为 `m`，四项原始条件为：

```text
gap       = max_k |nᵀ(c′_k−o)| − ε_g
body      = −min_i nᵀ(x′_i−o) − ε_num
footprint = max_i,j [a_jᵀ Uᵀ(x′_i−o)+b_j] + m − ε_num
normal    = ‖R n_b+n‖₂ − 2 sin(ε_θ/2)
```

四项均以小于等于零为满足。前三项以米计；第四项是两个单位法线之间的弦长条件，表达物体外法线与支撑面法线反向对齐。`normalized_inequalities` 使用 `max(ε_g,ε_num)` 缩放前三项，使用声明角容差和数值角容差对应弦长的较大者缩放法向项。实例的 `hard_components` 是四个名称的无重复子集，可显式为空；默认 body/footprint 是硬条件，gap/normal 是 `quality` 阶段偏好。该声明随任务和续规划义务保存，所有硬消费者据此选取相同分量；四项原始几何及阈值继续完整保留。

完整有符号接口使用上述最大值之前的各行：gap 为每个 k 的正、负高度两侧，body 为每个 i 的高度下界，footprint 为每个 `(i,j)` 的二维半空间条件，normal 保留单一弦长条件。因此每个已声明物质点都具有独立的条件值和候选状态导数。

参考 API 接收显式物体坐标约定。若 TSDF 使用相机坐标 `C`、任务点使用物体坐标 `B`，则 `T_C_B` 将任务点转换至 TSDF 坐标，法线使用该变换的旋转部分。随后各计算通过同一物体表示坐标和共同参考坐标连接。技能表示中的任务点与功能区域保留其任务含义；当前物体的碰撞占据由场景模型表达。二者共享坐标，并不要求以碰撞模型替换技能表示。当前输入契约与持物改造计划的对应关系见[架构与计划差异审查](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。

## 7. 输入辅助模块的调用范围

`pipeline.py` 为数值任务入口提供 `motion_kind`、`run_stage`、`run_sequence` 和 `run_prepared`。`motion_kind` 从已选任务关系决定自由或机构运动类型；`run_stage` 将已绑定观察、目标关系和物体点交给运动适配与目标计算，返回该阶段的数值结果；`run_sequence` 沿明确的阶段顺序传递对象位姿；`run_prepared` 读取已准备输入并保存结果。`serializable/write_json` 将这些结果转为可保存的普通数据。`program` 和 `planning_problem` 复用其中的 `_surface/_interior`，把明确记录构造为有限支撑面或内部区域。

`observations.py` 定义相机记录 `Camera(K,width,height,depth_scale)`。`load_calibration` 读取针孔内参和深度单位，`load_rgbd` 返回对应图像、米制深度与标定；`backproject` 以 `K⁻¹[u,v,1]` 和深度恢复相机点。`rigid_registration` 从明确的三维对应估计刚体变换和误差；`poses_from_tracks/poses_from_models` 将已有点轨迹或模型姿态组织为具名时间上的物体位姿。`track_rgbd` 通过独立跟踪 worker 取得固定点身份的观测序列，供示教输入准备。`planning_observation` 与机器人自过滤直接复用 `_intrinsics`；观测、区域及支撑输入准备复用 `backproject`。

`grounding.py` 中 `infer_task` 从指令与源/当前图像产生结构化语义任务，`select_roles` 从检测候选选择对象角色，`propose_query_regions` 提出可在图像中核对的区域查询，`match_points` 在指定 masks 内计算对应像素。`selection.select_grounding_candidates` 将已有源语义、当前指令和明确视图候选交给选择器，返回所选候选或未决状态。它们为所选技能与几何绑定提供输入；数值规划从已经确定的任务与场景开始。各接口完整输入和直接返回记录见[实现索引](07-implementation-index.md)。

已有实体模型通过 `observed_owners` 直接进入场景；新实体可以由 mask 和深度建立。事件前表示坐标 C 与事件工具位姿 E_g 确定固定关系 A=E_g⁻¹C_pre，之后工具 FK 给出 C(t)=E(t)A。源任务坐标至 C 的变换 M_D 同时用于任务点重表达，参考生成消费 J=M_D⁻¹。
