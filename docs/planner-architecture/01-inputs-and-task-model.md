# 输入、参考与任务模型

代码基线：`d7bcafc`。本次文档引用更新由随后文档提交保存，组件代码内容与该实现基线一致。本组件将已有示教、当前任务关系及机器人入口状态组织为 `PlanningProblem`，供场景绑定与数值规划调用。第 4.1 节的实验输入适配器以所链接脚本和阶段计划记录其实现。总架构见 [PLANNER_ARCHITECTURE.md](../../PLANNER_ARCHITECTURE.md)，各函数的完整参数与调用关系见[接口契约目录](06-api-contracts.md)。

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

输出包括 `object_path/eef_path`、对应时间和名义位姿数组、`object_boundaries/eef_boundaries`、`boundary_checks`、`summaries`、`missing` 与 `status`。受控路径的终点为源释放时间与显式持物终点中的较早者；`bound_*` 另保存原名义绑定诊断。每份边界声明包含 `initial_pose`、带局部进度/位姿/来源的 `goals` 以及声明来源；`boundary_checks` 记录 `declared_for_physical_time_binding`。缺少交互时间、当前入口或必要坐标对应时，函数将相应字段列入 `missing`。

`reference_phase` 复用完整 `eef_path`，将 `eef_boundaries` 复制到 `MotionPhase.metadata["reference_boundaries"]`；源帧、源时间和原事件分别保存。`reference_visualization.write_reference_html` 在真实源/目标图像上叠加原功能区域、采样点和名义路径，直接从该边界声明投影工具入口与每个事件目标。这个页面的时间是源示教时间；物理时间连接在规划器绑定后产生。

## 2.1 实际物理时钟上的工具参考

`timed_reference.BoundToolReference` 的输入为名义阶段、归一化物理时间到技能进度的 `progress_u/progress_s`、实际时长 `duration`，以及当前机器人 TCP 位姿和世界速度。`planning._reference_entrance` 使用同一官方机器人模型和具名关节 `q/dq` 运行 Pinocchio FK，读取工具连杆原点的 `LOCAL_WORLD_ALIGNED` 速度，经基座标定旋转到 W，并将 `ω_W × r_W` 加到线速度；`r_W` 为连杆原点到标定 TCP 的世界杆臂。输出 `initial_twist=[v_W,ω_W]` 的单位分别为 m/s 与 rad/s。

设名义 TCP 位姿为 `(p_n(t),R_n(t))`，其中 `p_n` 是 E 原点在 W 中的位置，`R_n` 将 E 坐标映到 W。接口使用世界加性平移 `b(t)` 与左旋转 `E(t)=Exp([φ(t)]×)`，计算 `p=p_n+b`、`R=E R_n`。名义物体、机构坐标及父路径组合在实际 `s(t)` 上直接求值。名义线速度和世界角速度由 Torch 对完整组合函数的单侧时间导数获得，保留进度停留段、斜率变化和旋转杆臂。

入口的目标速度是测得机器人速度；每个声明事件选择右侧名义物理速度作为左右输出的共同速度。对每个边界，修正位置为目标位置减名义位置，修正旋转为 `R_target R_nᵀ`；其导数满足 `ḃ=v_target−v_n`、`φ̇=J_l(φ)⁻¹(ω_target−Eω_n)`。这里 `J_l` 是 SO(3) 指数映射的左雅可比，代码以 `pin.Jexp3(-φ)` 求值。各相邻边界之间使用 SciPy `CubicHermiteSpline` 构造修正的三次多项式，Torch 负责按物理秒求值。

事件时间取其声明进度的首次到达时刻。恢复名义后缀的拼接以右侧名义速度为共同边界，左段修正补偿原名义左右导数差；拼接处及其后直接返回原名义函数。位于整个请求终点的声明目标可以采用不同于名义终点的位姿，此时修正函数在终点仍生效；位于请求内部的最后一个拼接则须恢复名义位姿，以连接其后的名义路径。多个边界使用同一规则，修正旋转坐标在相邻边界间选择连续分支；原机构累计转角保留在名义坐标中。当前边界阶次为一阶，连续性结论覆盖实际声明的拼接处，其他源内部断点保留其原导数。`source_boundaries` 保留原完整声明，`boundaries/diagnostics` 保存实际使用的物理边界、左右速度和来源。

`sample(times_s)` 返回 W 中的位置 `[...,3]` 与单位 wxyz 四元数 `[...,4]`；`sample_numpy` 返回同一函数的 `[...,4,4]` 齐次变换。`_CuroboPlanner._goal`、`NativeReferenceCost.bind` 和规划结果记录均消费该物理时间接口。保存候选使用自身的时长和进度映射构造同规则参考；没有显式边界声明的普通阶段继续提供其名义值。

续接区间位于最后一个声明拼接之后，或本区间唯一拼接位于零时刻时，返回原名义后缀，并以 `entrance_twist_bound=false` 记录当前区间的入口连接范围。存在未来声明拼接时，实际机器人入口连接到该拼接，并在已声明边界提供一阶连续性。原零时刻事件继续保留其技能表示和任务接口职责；参考对象只承担软跟踪目标的构造，当前 FK 与名义目标之差用于事后解释。

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
| `SupportRequirement` | 进度、物体坐标在工具坐标中的变换、物体点、接触点、外法线、有限支撑面、容差和 `hard_components` | `raw_inequalities` 返回 gap/body/footprint/normal；显式硬分量进入全部硬条件消费者，其补集进入 `quality` 阶段偏好 |
| `InteriorRequirement` | 区间、物体点、凸区域半空间和内缩距离 | 所有指定点相对半空间的最大越界量，减数值容差后截为非负 |
| `PoseProcessRequirement` | 区间、相对位姿及六维允许范围 | 过程中的平移和转动误差，只有明确受约束方向进入残差 |
| `MechanismRequirement` | 当前机构模型、附着、坐标限位、入口坐标及方向要求 | 流形偏离、坐标限位与方向残差；完整路径保留角度展开历史 |
| `CoordinateEventRequirement` | 某机构、进度、绝对或入口增量目标、容差 | 该事件的标量坐标误差；转动时区分局部姿态和累计转角 |
| `CoordinateDirectionRequirement` | 区间、坐标列、方向、最小变化或速度、容差与来源 | 净变化、单调或局部变化率等明确声明的方向条件 |
| `JointCoupling` | 具名关节、矩阵和目标 | `q_selected @ matrix.T - target` 的容差残差 |
| `PlanningEvent` | 原技能/事件/发生次数身份、进度、源帧、种类、可选停止与效果依赖 | 决定指令时刻，并按显式字段编译停止要求或下一段动作的独立依赖 |
| `EventTimeConstraint` | 检查对象、绝对或相对时间窗、可选起源事件 | 规划时检查名义时钟；执行只消费允许的指令/独立效果时间，几何目标到达时间仅事后评价 |

`PoseEventRequirement.error_components` 以候选工具位姿和固定变换求材料坐标 C 的位姿。位置分量为 C 原点到目标原点的欧氏距离。未声明 yaw 轴时，旋转分量为目标与候选之间的完整 SO(3) 最小转角。有轴时，旋转分量为去除绕轴转动后的剩余倾斜角；yaw 分量另行计算。`errors` 返回位置与该旋转分量；`errors_numpy` 为准备和诊断提供同一几何的 CPU 接口。`evaluate` 对 `component_tolerances` 中每个分量计算 `max(0,error−bound)` 后求和。`SupportRequirement.evaluate` 对 `hard_components` 指定的原始有符号条件取正部后求和；默认硬分量为 body、footprint。`normalized_inequalities` 保留全部四项，按长度或法向弦长缩放；功能增广拉格朗日读取硬分量，`optimizer_preferences` 在 `quality` 阶段读取其补集。

`NumericalTolerance` 控制浮点计算比较，目标容差规定任务允许误差，支撑容差规定允许间隙；实体距离的模型余量由观测碰撞查询单独应用。各量由对应接口保存。

### 4.1 目标坐标中的可设置 yaw 容差

令 W 为规划世界系，E 为标定工具中心点 TCP，C 为受约束的材料坐标，C* 为该材料坐标的目标。固定关系 `A=T_E_C` 与机器人正运动学给出 `T_W_C(q)=T_W_E(q)A`。`target` 保存完整的 `T_W_C*`。`yaw_axis` 是 C* 系中的单位向量 a，`yaw_tolerance_rad=β` 是主值绕轴转角的绝对值上界，范围为 `[0,π]`。这两个字段共同声明；没有轴声明时，事件表达完整姿态精度。

相对旋转 `R=R_W_C*ᵀ R_W_C` 的单位四元数写为 `(w,v)`，其中 v 为 C* 系中的三维向量。令 `b=v·a`，绕轴转角为 `ψ=atan2(2wb,w²−b²)`，剩余旋转为 `R_s=R Rot(a,−ψ)`。`rotation_tolerance_rad` 约束 `R_s` 的最小转角 θ，`yaw_tolerance_rad` 独立约束 `|ψ|`。这是有限旋转分解。因而带轴声明中的 1°表示 θ≤1°，不是某个 RPY 欧拉角，也不是 roll、pitch 各有 1°余量。绕垂直于 a 的轴转过半周时，绕轴分解的 ψ 不唯一；实现取 ψ=0 并保留 θ=π。

β=π 覆盖整周，`component_tolerances` 只包含位置和 θ。功能约束、验收以及可选目标居中项均按此分量集合计算。β=0 要求绕轴偏差为零；β=π/6 给出 ±30°范围。`full_rotation` 与 `signed_yaw` 仅提供解释性数值。源完整姿态和参考路径仍保存原值。

`_functional_events` 保留事件的轴与容差，`_material_pose_requirement` 用当前材料变换和功能目标构造上述要求。`_object_reference` 在入口以及参考功能点上使用同一分解；同刻事件的矛盾检查使用包含完整 yaw 范围的旋转球上界。`build_planning_problem` 从原事件复制轴与容差，保留 C* 的局部坐标语义。有限支撑声明由 `SupportRequirement` 独立表达；需要同时表达姿态集合时，调用者另行传入 `PoseEventRequirement`。

当前连续参考适配器 [put_prepare.py](../../../sim-version-plans/2026-10-18-shared-scene-toward-base/put_prepare.py) 的 `reference_inputs()` 读取完整技能表示、迁移功能坐标、声明关系 J/A 和冻结机器人反馈，调用 `generate_reference` 得到 `object_path/eef_path`。`build_problem()` 在交互进度和受控参考终点建立两个材料姿态要求。目标为 `object_path.sample_numpy(s)`，轴为 `R_W_C*ᵀ n_W`，其中 `n_W` 是观测支撑面的世界法向。三个容差直接读取 [scene.json](../../../sim-version-plans/2026-10-19-put-planner-audit/prepared-inputs/scene.json) 的 `pose_event_tolerances`：当前位置 0.002 m、剩余倾斜 π/180 rad、yaw π rad。该 1°衡量相对参考的倾斜；支撑面的法向与接触间距仍使用独立支撑声明。源时间、交互和释放事件继续决定查询进度。

## 5. PlanningProblem 的运行接口

`sample_reference(s)` 返回当前进度的完整名义工具路径；物理时间适配由第 2.1 节的 `BoundToolReference` 读取同一阶段及显式边界后完成。`NativeReferenceCost` 同时供工具曲线初始化与完整主优化使用，预采样各候选实际物理时钟对应的绑定参考。`source_time_to_progress` 和 `progress_to_source_time` 在保存的源时钟中转换。`transform_tool_pose`、`reference_to_link_pose` 和 `transform_tool_velocity` 将原生机器人计算转换到任务参考坐标，或将任务目标转换回原生连杆坐标。变换依赖构造时固定的 `base_pose/tool_transform`。

`residuals(position, quaternion, progress, ...)` 接收相互对齐的批量候选状态，返回名字到非负硬残差张量的字典。姿态数组分别为 `[...,3]` 和 `[...,4]`；进度数组是相同的批量/时间维度。具名关节、速度、物理时间和工具速度在对应要求需要时使用。事件约束在其精确进度的首次物理到达样本上计算，`point_goal_ids` 将当前查询样本绑定到相应原目标身份。`direction_mode="point"` 用于独立候选状态，`"trajectory"` 用于具有物理顺序的完整路径；依赖路径历史的方向要求由后者评价。所有调用读取同一 `hard_components` 和同一任务容差。

`PlanningProblem.evaluate` 汇总 `residuals`。`functional_constraint_keys` 枚举当前请求区间的硬功能条件身份；`functional_inequalities` 消费完整有序工具轨迹、进度和物理时间，按这些身份返回有符号归一化条件与事件掩码，供功能约束增广拉格朗日计算。机构绕转历史与确切事件时刻由这一轨迹输入保存。正容差的分量使用 `error/tolerance−1`；零宽 yaw 使用 `error/numerical.rotation_rad`，数值缩放保持原零集合。支持事件只为 `hard_components` 生成约束键。缺少要求的精确事件样本时输出 NaN，表达该条件在当前样本集上的未求值状态。

`ProcessCostManager` 在工具曲线初始化中读取过程残差、全部硬功能不等式、原生关节约束和 `robot` 几何范围，构造同次候选保留资格；该几何范围计算共享官方机器人自身配对与名义手部条件。完整主优化使用 `full` 范围，继续加入任务观测场、接触、支撑和覆盖分项。初始化记录 `geometry_scope="robot"` 与 `task_scene_constraints="not_evaluated"`。事件 IK、搜索的目标投影和完整曲线验收都使用相同任务硬要求；独立点查询的范围由点上可求值的要求确定，曲线检查补齐历史、速度和停止条件。源功能坐标、材料关系及任务几何从首次任务求值开始进入各自计算。

`optimizer_preferences` 在 `feasibility/quality` 两种目标配置中均汇总过程等式和方向偏好；`phase="quality"` 另加入支持硬分量补集的归一化超差正部平方，以及位姿、坐标事件的原目标中心偏好。该方法的默认参数是 `quality`，当前生产求解则显式采用管理器的 `feasibility` 阶段；因此生产中的支持梯度来自声明的硬分量，gap/normal 的质量偏好保留在可选阶段接口中。完整工具参考由独立普通成本持续优化，其位置与完整姿态偏好和事件声明的允许集合分别消费；事件自由 yaw 仍按原容差验收。`measurements` 提供解释性原量；`projection_requirements`、`project_residual`、`goal_requirements_at` 和 `goal_residual` 供事件 IK、边界投影和 OMPL 状态检查。`goal_residual(normalized=True)` 为事件 IK 缩放各声明的超差正部，其允许集合与未缩放形式相同。计算以机器人候选状态为变量，目标与允许范围保持为固定输入。

`anchors` 合并源路径节点、目标进度和过程边界；`stop_progress` 收集明确要求停止的事件。`bound_motion_block` 以最早尚未满足的独立效果依赖界定本次可执行区间，返回复制后的问题与区间记录，并保留原目标身份和阶段坐标。参考节点、开闭指令和名义附着按各自字段参与编译。`requirement_consumption`、`verification_issues` 和 `remaining_obligations` 提供这些要求的消费及剩余记录。

## 6. 支撑任务的几何输入契约

`on` 分支接收 `body_points`、物体接触区域点及其外法线、目标支撑面的有限区域。物体点与法线在物体表示坐标中定义，支撑面在任务共同参考坐标中定义。`SupportRequirement.raw_inequalities` 将物体点和接触点经过候选工具位姿与固定物体变换映入共同参考坐标，再计算接触间隙、全部物体点的最低高度、在支撑面内的投影范围以及材料外法线方向。

具体地，记候选工具位姿与 `tool_from_carrier` 的乘积为 `(R,p)`，物体点为 `x_i`，接触点为 `c_k`，物体接触面的单位外法线为 `n_b`。变换后的点为 `x′_i=R x_i+p`、`c′_k=R c_k+p`。支撑面由原点 `o`、单位法线 `n` 和平面正交基矩阵 `U∈R³ˣ²` 定义；其有限凸区域的第 `j` 条二维半空间写为 `a_jᵀy+b_j≤0`。记间隙容差为 `ε_g`，法向夹角容差为 `ε_θ`，数值位置容差为 `ε_num`，支撑边界内缩量为 `m`，四项原始条件为：

```text
gap       = max_k |nᵀ(c′_k−o)| − ε_g
body      = −min_i nᵀ(x′_i−o) − ε_num
footprint = max_i,j [a_jᵀ Uᵀ(x′_i−o)+b_j] + m − ε_num
normal    = ‖R n_b+n‖₂ − 2 sin(ε_θ/2)
```

四项均以小于等于零为满足。前三项以米计；第四项是两个单位法线之间的弦长条件，表达物体外法线与支撑面法线反向对齐。`normalized_inequalities` 使用 `max(ε_g,ε_num)` 缩放前三项，使用声明角容差和数值角容差对应弦长的较大者缩放法向项。实例的 `hard_components` 是四个名称的无重复子集，可显式为空；默认 body/footprint 是硬条件，gap/normal 是 `quality` 阶段偏好。该声明随任务和续规划义务保存，所有硬消费者据此选取相同分量；四项原始几何及阈值继续完整保留。

参考 API 接收显式物体坐标约定。若 TSDF 使用相机坐标 `C`、任务点使用物体坐标 `B`，则 `T_C_B` 将任务点转换至 TSDF 坐标，法线使用该变换的旋转部分。随后各计算通过同一物体表示坐标和共同参考坐标连接。技能表示中的任务点与功能区域保留其任务含义；当前物体的碰撞占据由场景模型表达。二者共享坐标，并不要求以碰撞模型替换技能表示。当前输入契约与持物改造计划的对应关系见[架构与计划差异审查](../../../sim-version-plans/2026-10-19-put-planner-audit/2026-10-20-model-restoration-plan.md)。

## 7. 输入辅助模块的调用范围

`pipeline.py` 为数值任务入口提供 `motion_kind`、`run_stage`、`run_sequence` 和 `run_prepared`。`motion_kind` 从已选任务关系决定自由或机构运动类型；`run_stage` 将已绑定观察、目标关系和物体点交给运动适配与目标计算，返回该阶段的数值结果；`run_sequence` 沿明确的阶段顺序传递对象位姿；`run_prepared` 读取已准备输入并保存结果。`serializable/write_json` 将这些结果转为可保存的普通数据。`program` 和 `planning_problem` 复用其中的 `_surface/_interior`，把明确记录构造为有限支撑面或内部区域。

`observations.py` 定义相机记录 `Camera(K,width,height,depth_scale)`。`load_calibration` 读取针孔内参和深度单位，`load_rgbd` 返回对应图像、米制深度与标定；`backproject` 以 `K⁻¹[u,v,1]` 和深度恢复相机点。`rigid_registration` 从明确的三维对应估计刚体变换和误差；`poses_from_tracks/poses_from_models` 将已有点轨迹或模型姿态组织为具名时间上的物体位姿。`track_rgbd` 通过独立跟踪 worker 取得固定点身份的观测序列，供示教输入准备。`planning_observation` 与机器人自过滤直接复用 `_intrinsics`；观测、区域及支撑输入准备复用 `backproject`。

`grounding.py` 中 `infer_task` 从指令与源/当前图像产生结构化语义任务，`select_roles` 从检测候选选择对象角色，`propose_query_regions` 提出可在图像中核对的区域查询，`match_points` 在指定 masks 内计算对应像素。`selection.select_grounding_candidates` 将已有源语义、当前指令和明确视图候选交给选择器，返回所选候选或未决状态。它们为所选技能与几何绑定提供输入；数值规划从已经确定的任务与场景开始。各接口完整输入和直接返回记录见[实现索引](07-implementation-index.md)。

已有实体模型通过 `observed_owners` 直接进入场景；新实体可以由 mask 和深度建立。事件前表示坐标 C 与事件工具位姿 E_g 确定固定关系 A=E_g⁻¹C_pre，之后工具 FK 给出 C(t)=E(t)A。源任务坐标至 C 的变换 M_D 同时用于任务点重表达，参考生成消费 J=M_D⁻¹。
