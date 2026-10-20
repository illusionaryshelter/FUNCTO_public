# 附着、接触与手指

代码基线：`__IMPLEMENTATION_COMMIT__`。

## 1. 实体观测与固定关系

`CollisionOwner` 包含实体身份、局部原始观测场、世界坐标 `PosePath`、来源、运动声明和有序 `Attachment`。位姿路径描述关系建立前的实体位置；场的局部样本始终保留。

设 C 为观测表示坐标，`C_pre=T_W_C` 为关系建立前的表示位姿，`E_g=T_W_E` 为声明生效事件处的规划工具/TCP位姿。以连杆 FK 表达时，`E_g=T_W_base·T_base_link(q_g)·T_link_tool`，其中 `q_g` 是该事件处的具名关节反馈，左右两项分别为基座和工具标定。固定关系为 `A=E_g⁻¹C_pre`。候选状态产生 `E(t)` 后，实体位姿为 `C(t)=E(t)A`。该运算沿用已有模型，当前实体可见性不参与。

`Attachment` 保存区间、`position_tool_owner`、`quaternion_tool_owner`、条件身份与可选释放身份。`CollisionOwner.pose` 对关系生效区间应用同一固定变换。释放后使用该候选第一个释放锚点的预测位姿；没有释放声明时关系延续。`with_release_history` 为搜索分支保留已选前缀的释放位姿，未来释放仍为变量。

源任务坐标到 C 的关系记为 `M_D`。任务点通过 `M_D` 重表达，法线使用其旋转部分；`generate_reference` 使用 `J=M_D⁻¹`。源轨迹、功能区域、任务点和原事件仍承担技能表示职责。

## 2. 有限接触定义

`ContactRule` 保存原始任务特征点、法线、边或多边形类型、目标区域坐标、活动区间、间隙和法向要求。机器人特征明确指向 `robot_body`，点在零手指位移时的 TCP 坐标中表达；`feature_joint_translation` 将具名棱柱关节坐标映入该坐标。普通实体特征在其自身坐标中表达。

`OnlinePlanningScene._permission` 将当前特征和目标区域转成目标坐标中的有限半空间，并计算法向残差。调用方将许可转到世界坐标，以 `PairPermission` 交给同一原始场查询。许可要求两侧点分别落在各自明确区域中；区域外的实体部分继续检查通常距离。特征点描述任务接触位置，不定义实体占据。

required 接触调用 `query_region_intersection`，寻找有限区域内受支持的零表面。原始零样本或同一完整支持单元中的可靠异号线段证明零表面存在；严格负值内部不足以证明表面接触。法向、活动区间和手指名义状态另行计算。缺少零表面与数值未决分别记录。

`bind_grasp_contact_rules` 将规则身份绑定到既有抓取条件。`validate_grasp_contact_arrival` 校验名义目标与 close 事件的共同身份和进度。执行按事件应用许可；末端相对预测目标的误差由事后评价消费。

`build_material_contact_rules` 将材料记录中的原任务特征和 `robot_body` 与声明的目标区域组合。`grasp_region.check_grasp_region_compatibility` 在明确手指关节区间中求共同的假设位置，使每个材料特征与至少一个目标许可区域相交。其线性规划使用世界坐标半空间及对手指坐标的仿射导数；原点集各自的法向高度范围完整保留。该输入描述任务许可，实体是否实际具有受支持零表面仍由场景接触查询计算。

## 3. 声明附着内部接触

`HoldingContact` 将一个明确 `robot_body` 与实体及附着区间关联。`bind_holding_contacts` 消费材料身份表与原声明关系；`validate_holding_conditions` 比较实体身份、固定变换和区间。许可仅覆盖登记的实体对。观测缺失的指垫不会扩大为整个手指的豁免。

## 4. 初始支撑

`PreparedInitialSupport` 消费两个原始场、入口关系和已声明平面。入口明确返回 intersection 的原支撑单元进入 `selected_cells`，未决单元单列。`PairPermission.excluded_second_cells` 仅排除该实体对中这些原始单元；邻接单元继续检查通常距离。

`query_minimum_height` 为整个已表示实体相对支撑平面求高度区间。后续高度下界高于入口基线上界减数值容差时确认满足；后续上界低于入口下界减容差时报告下降违反量和局部梯度，其余为未决。基线两端截断到不大于零。活动区间采用左闭右开，记录类型为 `initial_observed_support_v2`。

## 5. 指令、名义几何与反馈

`PrismaticGripper` 负责具名关节、控制能力、耦合、事件和时钟。位置跟踪消费求解关节；定时或二值控制消费独立手指曲线。`bind_clock` 绑定与正式轨迹一致的物理时间；`state` 返回坐标、不确定状态和关节残差。完整机器人 FK 再将这些坐标转换为连杆场位姿。

`FingerTrajectory` 产生实际指令，`NominalFingerGeometry` 表达规划中声明的未来手指坐标。未来几何缺失的手指相关查询记录为 conditional；实际执行使用具名关节反馈检查模型。指令位置与名义接触状态分别保存。
