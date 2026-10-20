# 附着、接触与手指

代码基线：`ATTACHED_TOOL_CODE_COMMIT`。

## 1. 实体观测与固定关系

`CollisionOwner` 包含实体身份、局部原始观测场、世界坐标 `PosePath`、来源、运动声明和有序 `Attachment`。位姿路径描述关系建立前的实体位置；场的局部样本始终保留。

设 C 为观测表示坐标，`C_pre=T_W_C` 为关系建立前的表示位姿，`E_g=T_W_E` 为声明生效事件处的规划工具/TCP位姿。以连杆 FK 表达时，`E_g=T_W_base·T_base_link(q_g)·T_link_tool`，其中 `q_g` 是该事件处的具名关节反馈，左右两项分别为基座和工具标定。固定关系为 `A=E_g⁻¹C_pre`。候选状态产生 `E(t)` 后，实体位姿为 `C(t)=E(t)A`。该运算沿用已有模型，当前实体可见性不参与。

`Attachment` 保存区间、`position_tool_owner`、`quaternion_tool_owner`、条件身份与可选释放身份。设区间为 `[a,b]`、当前实际技能进度为 `s`：具有释放声明时，`active(s)` 在 `a≤s<b` 为真；没有释放声明时，在 `s≥a` 为真，包含窗口终点并继续延续。`CollisionOwner.pose` 与碰撞配对共同使用这个判定。释放后使用该候选释放锚点的预测位姿；`with_release_history` 为搜索分支保留已选前缀的释放位姿，未来释放仍为变量。

源任务坐标到 C 的关系记为 `M_D`。任务点通过 `M_D` 重表达，法线使用其旋转部分；`generate_reference` 使用 `J=M_D⁻¹`。源轨迹、功能区域、任务点和原事件仍承担技能表示职责。

## 2. 有限接触定义

`ContactRule` 保存原始任务特征点、法线、边或多边形类型、目标区域坐标、活动区间、间隙和法向要求。机器人特征明确指向 `robot_body`，点在零手指位移时的 TCP 坐标中表达；`feature_joint_translation` 将具名棱柱关节坐标映入该坐标。普通实体特征在其自身坐标中表达。

`OnlinePlanningScene._permission` 将当前特征和目标区域转成目标坐标中的有限半空间，并计算法向残差。调用方将许可转到世界坐标，以 `PairPermission` 交给同一原始场查询。许可要求两侧点分别落在各自明确区域中；区域外的实体部分继续进行非穿透查询。特征点描述任务接触位置，不定义实体占据。仅有零距离或公共边界不足以建立体积交叠证据，也不足以证明任务所要求的有限区域接触成立。

required 接触调用 `query_region_intersection`，寻找有限区域内受支持的零表面。原始零样本或同一完整支持单元中的可靠异号线段证明零表面存在；严格负值内部不足以证明表面接触。法向、活动区间和手指名义状态另行计算。缺少零表面与数值未决分别记录。

`bind_grasp_contact_rules` 将规则身份绑定到既有抓取条件。`validate_grasp_contact_arrival` 校验名义目标与 close 事件的共同身份和进度。执行按事件应用许可；末端相对预测目标的误差由事后评价消费。

`build_material_contact_rules` 将材料记录中的原任务特征和 `robot_body` 与声明的目标区域组合。`grasp_region.check_grasp_region_compatibility` 在明确手指关节区间中求共同的假设位置，使每个材料特征与至少一个目标许可区域相交。其线性规划使用世界坐标半空间及对手指坐标的仿射导数；原点集各自的法向高度范围完整保留。该输入描述任务许可，实体是否实际具有受支持零表面仍由场景接触查询计算。

## 3. 工具与附着实体组成的组合体

`OnlinePlanningScene` 必填的 `tool_geometry_names` 给出完整工具组件的已注册几何实体 ID。记录入口调用 `robot_validation_fixture.tool_geometry_names(fixture)`：适配器读取已知 robot-only 模型与 fixture 声明的手指关节，求这些关节所属机构的公共父节点，再选取该节点子树中明确登记的碰撞实体。当前机器人得到掌体、两指体和两指腹五个成员。通用场景直接消费这些 ID，机器人其余部件保留各自身份。

当前实体的 `Attachment.active(progress)` 为真时，该实体与工具组件属于同一组合体。场景排除工具成员与该实体的内部配对；两个实体同时附着于本场景同一 TCP 时，也排除它们之间的内部配对。排除发生在原几何查询之前，按状态样本累计到 `internal_assembly_pair_count`。组合体各成员仍分别保留原生形状或原始观测场、局部坐标和位姿，继续对背景、未附着实体及其余机器人部件进行原几何查询。机器人自身配对继续使用其独立模型策略。

`validate_problem` 将每个 `Attachment` 对应到原 `declared_attachment` 或 `candidate_attachment` 条件，核对实体身份、生效事件与固定变换；显式释放对应同一 `predicted_release` 条件及区间终点。组合体成员由实际 `progress` 确定，有限任务接触所用的 `contact_permission_progress` 不提前建立附着。`ContactRule` 保留原特征、区域、间隙和法向计算，分别表达任务要求的有限接触。

场景快照保存工具成员与各实体的 `Attachment`。执行入口从原 `declared_attachment` 条件及对应实体关系记录 `declared_attachment_bindings`，包含条件、实体、活动区间、释放身份、工具成员和来源。该记录表达候选采用的模型关系，物理接触与任务效果由各自的观察或事后评价记录。

## 4. 初始支撑

`PreparedInitialSupport` 消费两个原始场、入口关系和已声明平面。它用同一非穿透查询逐个检查原支撑单元；只有入口已确认体积交叠、返回 `intersection` 的原单元进入 `selected_cells`，未决单元另存于 `unresolved_cells`。单纯触面不构成选格依据，也不需要为非穿透规则新增豁免。`PairPermission.excluded_second_cells` 仅排除该实体对中已选的原始单元；邻接单元继续接受普通非穿透查询，许可没有额外距离范围。

`query_minimum_height` 为整个已表示实体相对支撑平面求高度区间。后续高度下界高于入口基线上界减数值容差时确认满足；后续上界低于入口下界减容差时报告下降违反量和局部梯度，其余为未决。基线两端截断到不大于零。活动区间采用左闭右开，记录类型为 `initial_observed_support_v2`。

## 5. 指令、名义几何与反馈

`PrismaticGripper` 负责具名关节、控制能力、耦合、事件和时钟。位置跟踪消费求解关节；定时或二值控制消费独立手指曲线。`bind_clock` 绑定与正式轨迹一致的物理时间；`state` 返回坐标、不确定状态和关节残差。完整机器人 FK 再将这些坐标转换为原生机器人实体位姿。

`FingerTrajectory` 产生实际指令，`NominalFingerGeometry` 表达规划中声明的未来手指坐标。未来几何缺失的手指相关查询记录为 conditional；实际执行使用具名关节反馈检查模型。指令位置与名义接触状态分别保存。
