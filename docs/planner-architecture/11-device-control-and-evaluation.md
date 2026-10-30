# 设备执行、命令衔接与仿真评价

代码基线：`3266476`。本文档提交登记实际实现编号；生产代码内容与该实现提交相同。规划器输入、求解与候选验收仍见对应组件；本组件消费已接受的完整关节多项式和显式设备状态。

## 数据、坐标与职责

机械臂参考 q_r、dq_r、ddq_r 分别为具名关节的位置、速度和加速度，单位为 rad、rad/s、rad/s²；棱柱手指对应 m、m/s、m/s²。q、dq 为同一设备时间的关节反馈。机器人固定基座、TCP 标定和官方固定配置模型定义 FK；世界 W 中工具位姿用于同刻指令跟踪及事后显示。当前物体位姿、接触和效果由独立仿真评价器记录。

`FingerTrajectory` 保存已发出指令的入口、当前测量入口、原事件时钟和完整曲线。`incoming_command` 明确具名指令位置、解析速度、目标、剩余斜坡时长、来源、反馈身份、源曲线时间及已激活原事件；`measured_initial_position/velocity` 独立保存实际关节状态。`stationary_finger_command` 只用于调用者明确已处于固定目标保持的设备；`capture_active_command` 从前段实际派发对应的多项式取当前指令及尚未结束的斜坡，不继承前段未来事件。`from_mapping` 读取当前唯一模式，`validate_execution` 分别核对实测入口与新读取的原生命令。

`NominalFingerGeometry` 保存规划中声明的手指形状。自由运动模型使用原有受限 Hermite 插值：从实测 q/dq 通向原事件目标，在首次接触事件边界结束有效性。指令曲线则继续既有命令，两者入口可以不同。声明附着时，几何固定为实测入口 q，原释放边界结束有效性。`PrismaticGripper.bind_clock` 核对指令记录的测量入口、名义几何的实际入口及共同物理时钟。未知未来几何仍使用原条件状态。

## 机械臂控制与时间

`TrajectoryExecutor.execute` 在每个设备步的起始时刻 t_n 对同一正式多项式求 q_r、dq_r、ddq_r，内部加速度结点使用右侧、最后结点使用左侧。手指仍按原事件和步末采样协议施令。物理步结束后在实际时间记录参考与反馈，跟踪检查比较这一同刻对。`barrier_reference_reached` 表示记录已覆盖原参考边界。

`RobosuiteExecutionBackend.apply_control` 复用 robosuite 的质量矩阵 M、偏置 b 及原 Kp、Kd，计算 u=M[ddq_r+Kp(q_r-q)+Kd(dq_r-dq)+z]+b。z 为后端持有的积分加速度状态，单位 rad/s²。正标量设备增益满足 Kd²≥3Kp；r=(Kd−sqrt(Kd²−3Kp))/3，Ki=r²(Kd−2r)，给出名义恒定参考误差模型的重复慢实根。该设备增益构造由固定控制配置确定。

原生 unit-gain arm motor 的 ctrlrange、已启用 actuator forcerange 和 joint actuator-force range 都作用于同一 N·m 坐标。后端取其交集 [L,U]，只对总 u 裁剪并写一次机械臂命令。`step` 根据原生实际推进 h 确认区间，计算 Δτ=M h Ki(q_r−q)，将增量限制到 [min(L−u,0),max(U−u,0)]，再令 z←z+M⁻¹Δτ。这样保留质量耦合，同时抑制已饱和方向的积分增长。采样误差、M、u 在实际派发时保存；只有确认物理推进的区间提交一次积分，后续派生量刷新错误不抹去已完成区间。

`request_control_rebase` 由调用者明确声明控制接管或参考重新定基。下一次输出以旧已发力矩减新基础力矩求 z，使该次总输出连续；后续普通片段保留 z。`abort_hold` 取消尚未实施的重定基，继承 z 并立即使用停止参考和速度阻尼。连续输出匹配与制动分别由这两个接口表达。

## 固定机器人手指重力

`robot_gravity.RobotGravity(model,joint_names,source)` 接受明确的官方 robot-only 模型和完整具名标量关节布局，独占 scratch MjData。官方机器人及其固定末端配置不会随任务更换，形状、惯性及固定组件关系持续复用；基座标定确定重力坐标。`compute(q)` 在该模型中置零速度/加速度，以 MuJoCo kinematics、comPos、comVel、rne 求 g(q)，返回对应关节广义力；转动关节为 N·m，平移关节为 N。

后端初始化和每次 `apply_control` 以实际具名 q 计算手指 g，写入原 affine position actuator 的常数偏置。总手指力为原位置/速度伺服力加 g，由原生总力范围统一限幅；ctrl 继续表示原位置目标。普通执行和 `abort_hold` 调用同一路径。`physical-trajectory.npz` 保存手指重力、位置伺服、总请求、限幅后模型力及饱和标记；实际接触和原生步后力由独立记录提供。

## 状态、文件和实际调用

| API | 前置数据、输入与输出 | 调用关系 |
|---|---|---|
| `snapshot_control_state/restore_control_state` | 完成边界的具名 z、原生时间和完整控制策略；恢复核对同一设备与时刻 | 状态拥有者调用，普通片段共用同一后端 |
| `simulation_snapshot.save_snapshot` | 暂停后端 → mjSTATE_INTEGRATION 与独立控制 JSON 的 NPZ | 连续会话在片段边界保存 |
| `restore_native_snapshot` | 相同原生模型及 NPZ → 原生状态、对应控制记录 | 原生恢复并构造后端后调用 restore_control_state |
| `restore_recorded_simulation` | 原场景/录制入口及 robot_mjcf_path → 后端和来源 | 记录式运行器一次初始化，随后反馈只含机器人状态 |
| `run_recorded_skill.finger_trajectory_factory` | 任务、实际反馈、显式 incoming_command、设备界 → 原事件时钟上的指令构造器 | 规划场景 bind_clock 调用 |
| `position_controlled_hand` | 实际入口、设备界、命令构造器和完整任务条件 → PrismaticGripper | 规划和结果绑定共用同一几何接口 |
| `run_recorded_simulation.run` | 正式结果或轨迹文件、当前仿真与反馈、可选 reference_feedback → 执行及独立评价档案 | reference_feedback 保留规划时样本身份；执行器再次核对实际数值和实时 ctrl |
| `ContinuousSimulationSession.execute` | 正式结果、独立回调、reference_feedback、诊断证据身份 → 阶段报告 | 共用同一后端，透传既有运行器参数 |
| `ContinuousSimulationSession.capture` | 既有相机标定和当前后端 → RGB-D、当前曝光时间标定和机器人反馈 | 感知/规划期间物理时间暂停 |

采集或录制反馈适配器必须提供独立 active-command 记录；实测 q/dq 只描述当前形状与运动。完整源技能轨迹、区域、相对关系、功能坐标及交互/释放事件仍作为输入保留。

## 独立评价与连续任务结果

`NativeSkillPhysicsEvaluator.observe` 经事后回调采集仿真实际运行环境中产生的真值；`finalize` 保存完整采样覆盖、原功能目标、效果、过程和终停测量。`summarize_skill_evidence` 只消费这些已完成记录。`diagnostic_goal_ids` 选择原功能目标身份；`diagnostic_evidence_ids` 使用 `[集合,身份]` 选择 effect_evidence 的 condition_id、process_evidence 的 identity 或 terminal_evidence 的 predicate。未选择项保留 required；原始 F/U、数值、对象和时刻完整保留。

完整执行且全部 required 项为 T 时单段评价 passed；无 required 项时为 unverified，表示本段只记录诊断。连续任务的抓取有效性由完成后的后续实际操作证据评价，双侧接触或稳定窗口的诊断不能否定已经完成后续操作的抓取。连续会话推进消费正式规划和执行状态；评价输出不进入动作、规划或重规划决策。程序中明确声明的独立在线任务观察仍使用原观察接口。

必要回归覆盖原生积分、饱和耦合、显式接管与制动、完整状态恢复、命令/测量不一致的衔接、同刻采样、原事件及目标误差永久执行边界、机器人解析重力和统一事后角色。对应入口见 tests/test_integral_backend.py、test_integral_execution.py、test_snapshot_state.py、test_command_reference.py、test_robot_gravity.py、test_gripper_trajectory.py、test_recorded_hand_bounds.py、test_recorded_skill_evaluation.py。
