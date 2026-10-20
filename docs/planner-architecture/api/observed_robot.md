# 完整机器人可微运动学

代码基线：`__NATIVE_MODEL_COMMIT__`。

`pure_kinematics(path)` 读取当前 cuRobo 接口所需的纯关节树 URDF。该输入格式要求与碰撞形状来源分开；机器人碰撞实体由共享的 `RobotGeometryChecker` 注册。

`DifferentiableRobotKinematics(configuration, *, frames, T_reference_base, device)` 使用已安装 cuRobo 的 URDF 构造接口，保留所有独立关节，包括手指。该构造只负责具名连杆位姿，原生形状由场景持有。

`poses(position, joint_names)` 消费 `[B,H,J]` 完整具名坐标，按原树关节顺序排列，返回各请求 frame 的世界位置与 wxyz 四元数，并保留关节导数。基座变换在此应用。`OnlinePlanningScene` 随后对解析形状再乘 `T_frame_geometry`；已烘焙到连杆坐标的凸顶点使用单位局部变换。
