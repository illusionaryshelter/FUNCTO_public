# 机器人规划与实际反馈检查

代码基线：`__NATIVE_MODEL_COMMIT__`。

`RobotStateValidator` 绑定一个已知 `RobotGeometryChecker`、完整关节顺序及机械臂/手指分组、原配对、速度限制、参考基座及 TCP 标定。构造时给出的 FK 位置/角度数值容差用于比较同一机器人模型的运动学约定。

| API | 输入和计算 | 输出及消费者 |
|---|---|---|
| `validate_planning_binding(problem, gripper)` | 任务关节、工具/基座标定和手指声明 | 一致性检查；规划绑定 |
| `_planning_positions(..., time_s, gripper)` | 当前规划关节及同一物理时钟的名义手指 | 完整具名状态和不确定实体；规划自身检查 |
| `check_planning_self(joint_positions, joint_names, *, time_s, gripper)` | 完整名义状态下的原生机器人查询 | 完整具名状态、原限位、必检实体对的分离或数值未决记录 |
| `planning_binding_record()` | 已绑定模型、校准、范围及策略 | 完整规划输入记录 |
| `describe()` | 已绑定检查器 | 模型身份、数值尺度及 `pair_order`；执行器确定距离向量顺序 |
| `check_state(feedback, *, stage, trajectory_time)` | 实际完整 q/dq、参考坐标、TCP反馈及原模型 | passed、violations、`distances_m`、最小必检间隙及查询证据；每步实际控制 |
| `preflight(trajectory, initial_feedback, execution_limits, *, start_time=0., until=None, current_time=None)` | 保存多项式、实际入口与设备范围 | 入口机器人检查及连续标量 q/dq 极值；执行入口 |

`distances_m` 按 `describe().pair_order` 保存每一必检实体对的有符号距离；完整 excluded/mechanical 策略保存在 pair_policy 与 planning_binding_record；`min_required_clearance_m` 汇总必检对。`numerical_boundary` 与确定的原模型穿透分别记录。`_pose_difference` 计算模型/反馈 FK 差，`_positive_scalar` 解析数值参数。

实际检查消费机器人反馈和已知本体模型；任务观测场由规划器消费。末端相对冻结预测目标的误差属于事后评价，不参与此验证器的控制判定。预检保留入口检查与多项式极值职责，不重复进行整条任务场景几何认证。
