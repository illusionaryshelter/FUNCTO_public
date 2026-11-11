# 完整源表示与连续参考

代码基线：`dbb6c1d`。实现为 [representation.py](../../../manipulation/representation.py)，任务及边界消费者见 [输入与任务模型](../01-inputs-and-task-model.md)和 [timed_reference](timed_reference.md)。

## 坐标与职责

T_X_Y将Y中列向量映到X；位置为m、角度为rad、源时间为s。B是固定物体坐标，F是物体功能坐标，E是标定工具。组件将同步源轨迹编成完整技能关系，再将源关系、当前功能坐标、预测物体对应与声明附着组成名义物体及工具路径。入口位姿和事件目标分别作为边界数据，由物理时钟消费者建立连接。

## API与数据依赖

| API | 输入、计算与输出 |
|---|---|
| `sample_poses(poses,timestamps,query_times)` | 用原秒数与PosePath的平移/Slerp插值采样实际记录；查询范围为记录支持区间 |
| `trajectory_summary(...)` | 从同一连续采样运算读取交互位姿与接近方向，保留缺失事件和零位移状态 |
| `build_representation(...)` | 接收同步物体、参考体、工具及固定功能坐标，保留原帧、时间和事件；输出H_g、H_m、C_D及摘要 |
| `generate_reference(...)` | 接收完整表示、当前F、J、A和入口位姿，或明确机构模型/坐标；输出连续物体/工具路径、独立边界与完整绑定诊断 |

源`C_D=inv(T_W_B)T_W_E`，抓取`H_g=inv(L_G)C_D`，操作`H_m=inv(L_R)inv(T_W_reference)T_W_carrier`。H_m的终点为任意固定物体坐标，因此先对`Q(t)=H_m(t)inv(H_m(t0))`采样，再组合原初态；任意局部原点改变不会改变物理运动。原帧和完整时间序列继续保存。

抓取参考为当前F_G与原H_g的连续组合。自由操作参考以`L_B_M=inv(J)inv(H_m(t_interaction))`定义固定材料坐标M：连续材料路径先表达于当前功能坐标，物体和工具通过固定局部关系逐次派生。J把当前物体坐标映到源物体坐标，A把当前物体坐标映到工具；当前模型物体入口为E A。机构参考使用明确当前坐标和原源标量位移，由同一MotionModel的FK计算。

`is_initial_relative_rotation_rpy`按功能坐标中的XYZ外禀roll/pitch/yaw声明初始选择，R=Rz(yaw)Ry(pitch)Rx(roll)。`_initial_rotation_offset(nominal,functional,current,flags)`接收三个初始4×4位姿及布尔数组，返回功能坐标中的恒定左旋转和逐分量来源。`generate_reference`将原路径初态传给它；几何目标构造可在生成完整路径之前使用同一运算。完整相对SO(3)运动保持原组合。

## 连续路径与边界

`_component_path`在原秒数中分别采样关系再组成父路径与固定附着；`_right_path`先采样父路径再作固定右乘，保留旋转杆臂。`RotationOffsetPath`改变材料朝向，材料位置沿用原连续函数。生产区间在释放或明确held_until_time终止，完整源记录及绑定诊断继续保留。

返回`object_path/eef_path`、对应时间/采样矩阵及`summaries`；`object_boundaries/eef_boundaries`分别保存入口和名义事件目标。`reference_phase`将eef_path和eef_boundaries放入普通MotionPhase，`BoundToolReference`在请求实际时钟上绑定测得入口速度。任务容差、碰撞注册和设备执行由各自明确接口消费。

## 内部运算

`_timeline/_progress/_aligned_poses`处理同步时间与数组语义；`_relative`计算同步父子关系；`_sampled_summary`使用传入采样函数读取静态摘要；`_require_same_pose`核对声明坐标的代数一致性。`generate_reference`内部的function_path建立当前F路径，diagnostic保存完整源绑定采样，publish发布路径和边界，mechanism_path组合原标量位移，material_path组合功能坐标和材料运动。

依赖为既有NumPy、SciPy Rotation、本库MotionModel、PosePath和RotationOffsetPath。新的观测几何目标与物体预测输入由 [reference_transfer](reference_transfer.md)构造，使用上述同一接口。
