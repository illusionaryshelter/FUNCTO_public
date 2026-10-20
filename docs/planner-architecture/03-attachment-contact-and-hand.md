# 附着、接触与夹爪

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。本组件定义被操作物体与工具的名义刚体关系、任务允许接触的几何区域，以及手指指令和规划几何在时间上的对应。

## 1. 物体与附着的数据结构

`CollisionOwner` 表示一个具有明确身份、局部碰撞查询和位姿路径的场景对象。`owner_id` 连接任务、观测分区与接触规则；`query` 保存局部几何；`pose_path` 给出对象坐标在共同参考坐标中的位姿。`geometry_complete` 是调用者对完整物体模型的声明。`moving`、`local_spheres`、`sphere_materials` 和 `sphere_cover_error_m` 共同定义作为运动查询作用方时使用的球覆盖。

`Attachment` 保存生效进度区间、工具中的 owner 平移与四元数、声明条件身份，以及可选释放条件身份。记工具世界位姿为 `E(s)`，把物体表示坐标中的点变换到工具坐标的固定变换为 `A=T_tool_owner`，则活动附着内的对象位姿为

```text
T_world_owner(s) = E(s) A .
```

`CollisionOwner.pose(position, quaternion, progress, released_poses)` 对每个查询进度选择位姿路径、活动附着或释放后冻结位姿，返回位置与四元数。`predicted_pose` 查询对象原位姿路径。释放边界通过场景 `_release_pose` 读取同一候选曲线在该进度的位姿；`with_release_history` 保留已经计算的释放结果，供后续局部状态查询使用。

`scene.holding.T_carrier_eef` 将工具点变到 carrier 坐标，它是上述 `A` 的逆变换。任务编译保存此关系，场景构造时转换为 `Attachment`。`HoldingContact` 引用相同声明条件和区间，使碰撞许可与名义运动使用同一对象关系。

当前 `CollisionOwner` 对运动作用方要求有限球覆盖；`OnlinePlanningScene` 对 `geometry_complete=False` 的运动物体产生 `unsupported/payload` 分项。`native_world_collision_pairs` 登记机器人和具备完整基本体注册的运动对象。这个类型范围决定当前对象参与距离复核的方式。

## 2. 有限接触区域的几何含义

`ContactRule` 描述一个作用方的指定材料区域与目标对象的有限区域之间的允许关系。这里“材料”指模型中具有固定位置和法线的表面或边缘；作用方可以是机器人，也可以是另一个对象。规则包含 `actor`、`target_owner`、`material_id`、`sphere_indices`、作用区间、目标局部区域、作用方局部材料几何、法线、间隙/方向容差、建模余量、是否要求接触及来源。

`build_material_contact_rules(material_binding, declarations)` 将具名材料记录与规则声明合并，返回 `ContactRule` 元组。机器人材料通常由 `bind_fixture_contact_materials` 与 `calibrate_box_material_features` 提供，包含工具坐标中的材料几何及其随手指关节移动的仿射关系。完整对象材料由 `bind_box_owner_materials` 提供，连接到同一长方体实体及其覆盖球。

`bind_grasp_contact_rules(problem, contacts, activations)` 把已声明规则身份连接到 `grasp_contact_activation` 的 close 事件与区间。`validate_grasp_contact_arrival` 检查名义抓取要求的身份、目标对象和容差关系。实际激活使用原 close 身份和已绑定模型；执行中的几何目标误差进入事后评价。

`require_native_material` 接受当前实现支持的完整平面挤出材料或原生盒边缘材料。`bind_native_owner_geometry` 验证完整长方体集合、原生实体身份及球覆盖所属关系，返回原生几何字典、球到实体对应和实体球组。`check_native_owner_material` 检查规则材料与该实体表面的对应。非机器人 `ContactRule` 在场景构造时消费这些记录。

## 3. 接触准备与运行计算

`PreparedContactFields.prepare` 输入目标候选场、TSDF 源几何或基本体接触几何、规则集合、球最大半径及网格参数。它枚举实际可能出现的规则子集，为相应允许区域准备固定字段，同时保留需要接触时可用的源表面见证。返回对象包含 `regions`、查询通道、球组和来源，之后由目标 query 持有。查询通道保存对应允许区域组合的候选距离数组，接触见证保留同一源几何中可供要求接触条件使用的表面点；两者分别服务碰撞许可与接触要求。

对于 TSDF 目标，`FrozenObservedGeometry.prepare_rules` 将完整源组件与各许可区域对应；`component_indices(..., active_rule_ids)` 使用同一组件身份选择保留集合。`PreparedContactFields.query(spheres, policies, ordinary, intersects_geometry)` 消费当前局部球、规则是否激活、几何变换及普通候选值，返回接触模式下的碰撞值和表示缺口。`required_contact` 计算明确要求接触时的联合条件。

`OnlinePlanningScene._permission` 根据作用方和目标位姿、手指坐标、进度及规则材料生成当前几何策略。`native_material_conditions` 在目标坐标中计算完整材料的有限支持关系：平面材料检查间隙、侧边和背面，盒边缘材料使用其两个相邻面定义的法向范围。可选接触的法向条件在可能接触时生效；`required=True` 的规则在整个声明活动区间内承担接触条件。

`ContactCellGeometry`、`ObservedContactGeometry` 和 `LayeredContactCellGeometry` 还提供源几何接触计算 API。前者消费完整有限盒集合，中者消费 TSDF 表面/体积组件，后者消费已经选择的粗细层。`make_contact_geometry` 根据实际数据契约选择实现。它们的结果分别服务相应查询或审计流程；生产固定接触字段的热路径由 `PreparedContactFields.query` 执行。

## 4. 声明持物内部接触

`HoldingContact` 连接一个机器人材料实体、附着对象、对应 `condition_id`、区间和完整球索引。`bind_holding_contacts` 消费 `PlanningProblem`、机器人材料绑定和声明记录，返回与已有附着一致的接触对象。`validate_holding_materials` 检查机器人实体和球身份，`validate_holding_conditions` 检查对象、附着矩阵和生效区间。

场景查询时，已绑定材料与被持对象之间的内部接触在声明区间内生效；同一材料对其他对象及背景继续执行普通查询。该处理描述名义刚体模型的内部关系。实际持物效果由独立观察器或事后评价记录，和这份模型声明分别保存。

## 5. 初始支撑关系

`InitialSupportRelation` 表示请求入口已声明存在的物体与独立观测支撑层关系。输入包括作用物体、目标支撑 owner、从入口开始的区间、目标坐标中的平面原点和法线，以及观测身份和支撑假设来源。它与功能事件上的最终 `SupportRequirement` 分别构造。

`PreparedInitialSupport` 当前接收完整运动基本体及静态 TSDF 支撑目标。构造时，将物体所有原生顶点放入目标坐标，计算其相对平面的最小高度 `h0`，并令基准 `b=min(0,h0)`。`_select_components` 用入口实体和 Coal 查询选出该支撑层中实际与入口模型相交的有限源组件身份。字段准备与实体复核都使用这组固定身份。

`components(actor_pose, target_pose, progress)` 重新计算整物体最低高度 `h(s)`，返回活动掩码与 `relu(b−h(s)−numerical_tolerance)`。`collision_values` 在活动时使用扣除上述入口组件后的候选场，在区间外使用普通场。`record` 保存关系、入口高度、基准、选中组件和准备后的字段，恢复时对应原关系。

## 6. 手指指令、名义几何与实际反馈

`FixedFingerTargets` 保存 `joint_names`、`source` 和 `events`。每条目标事件以 `skill_id`、`event_id`、`kind`、`positions` 指定具名手指位置。`bind` 按技能与事件身份匹配轨迹中的命令，核对其发生次数身份和控制范围，再取得本次曲线中的物理时间。`FingerTrajectory.build` 输入具名手指的入口位置/速度、命令列表、总时长、可执行终点、控制步长以及位置/速度/加速度范围，生成分段三次 Hermite 指令曲线。

`FingerTrajectory.sample/sample_torch` 在任意共享物理时间返回指令位置与速度。`describe/from_mapping` 保存和恢复原曲线；`validate_execution` 将保存结果与实际入口、指令身份、控制步长和设备范围相互核对。`_segment` 为一次指令寻找符合明确运动范围、且终点落在设备控制网格上的坡段；`_extrema` 检查其位置、速度与加速度极值。

`NominalFingerGeometry` 保存规划碰撞使用的手指位置预测及其有效前缀。`bind_nominal_finger_geometry` 在自由运动分支使用命令多项式，几何有效性延续到首次 close；在 `declared_attachment` 分支将入口测量位置冻结为名义持物几何，并令几何速度为零，有效性延续到首次 open 或具名释放。`valid_mask` 给出指定物理时刻的有效性，`sample` 按上述分支返回名义位置。实际下发的 `FingerTrajectory` 仍从测得的入口位置与速度连续构造；命令曲线和几何预测分别序列化。

`PrismaticGripper` 保存独立手指控制类型、具名坐标、机械范围、耦合关系、局部球、每球对各手指关节的平移矩阵、碰撞对与命令。`bind_clock` 将其指令和名义几何绑定到本次机械臂时钟。`state` 根据查询类型选择当前测量、名义指令或机械范围，输出 `GripperState`：中心坐标、半范围、控制残差与不确定标记。`spheres` 把局部手指球放置到工具/世界坐标，并按尚未确定的关节范围形成对应包络。

`PrismaticSelfCollision` 使用明确的手指实体及运动关系计算当前手部实体碰撞。`build_post_grasp_robot_config` 将已测量手指坐标写入独立手指关节配置，并返回机械臂活动关节入口；其前置数据为具名机器人反馈。运动学树、工具标定和手指所属实体由机器人 fixture 提供。
