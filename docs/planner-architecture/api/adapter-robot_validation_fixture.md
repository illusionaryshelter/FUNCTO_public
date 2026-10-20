# 记录入口的原生机器人适配

代码基线：`ATTACHED_TOOL_CODE_COMMIT`。

`build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp)` 从 fixture 明确给出的 robot-only 模型和关节 URDF 构造检查器。碰撞清单分别声明机器人实体与固定安装体；每项保留 geom/frame 身份。原模型的关节范围、速度限制、工具标定和机械接触声明进入 `RobotStateValidator`。

`_known_shapes(fixture, model)` 读取已编译原模型的顶点和固定变换。mesh 的碰撞语义为该原组件的凸体，盒体和圆柱保留解析形状；同时返回原实体局部坐标，供图像过滤围绕正确原点施加 padding。它不读取任务环境模型。

`_native_pair_policies(model, geometry_names)` 根据原模型的同体/固定连接、原生父子过滤和逐 geom 碰撞掩码给出完整配对。机械接触声明只作用于其确切实体对。模型策略不随技能调整。

`robot_self_filter_shapes(fixture, frame_poses)` 把同一 robot-only 模型的组件放到图像采集时 FK 下，输出 `RobotCollisionShape`。`run_prepared` 分别使用图像采集反馈和规划入口反馈，两个时间的数据职责明确分开。

`tool_geometry_names(fixture)` 读取 `fixture["robot_mjcf_path"]` 指向的已知机器人模型、明确 `finger_joint_names` 与已注册 `collision_geometries`。它从各手指关节所属 body 的父节点向根遍历，取这些祖先链最近的公共节点，返回其子树中已登记的 geom 名称元组。当前机器人返回掌体、两指体和两指腹五个实体。该函数只解释机器人机构与实体清单，不消费任务物体、图像或接触状态。

`run_prepared` 将此元组保存到运行报告，并以必填 `tool_geometry_names` 传给 `build_scene_from_rgbd`。通用场景据此和当前有效 `Attachment` 确定组合体内部配对；原生机器人自身配对仍由 `_native_pair_policies` 提供。
