# 记录入口的原生机器人适配

代码基线：`b32dfbe5108be41ddb2d866c5813e6ff391e2918`。

`build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp)` 从 fixture 明确给出的 robot-only 模型和关节 URDF 构造检查器。碰撞清单分别声明机器人实体与固定安装体；每项保留 geom/frame 身份。原模型的关节范围、速度限制、工具标定和机械接触声明进入 `RobotStateValidator`。

`_known_shapes(fixture, model)` 读取已编译原模型的顶点和固定变换。mesh 的碰撞语义为该原组件的凸体，盒体和圆柱保留解析形状；同时返回原实体局部坐标，供图像过滤围绕正确原点施加 padding。它不读取任务环境模型。

`_native_pair_policies(model, geometry_names)` 根据原模型的同体/固定连接、原生父子过滤和逐 geom 碰撞掩码给出完整配对。机械接触声明只作用于其确切实体对。模型策略不随技能调整。

`robot_self_filter_shapes(fixture, frame_poses)` 把同一 robot-only 模型的组件放到图像采集时 FK 下，输出 `RobotCollisionShape`。`run_prepared` 分别使用图像采集反馈和规划入口反馈，两个时间的数据职责明确分开。
