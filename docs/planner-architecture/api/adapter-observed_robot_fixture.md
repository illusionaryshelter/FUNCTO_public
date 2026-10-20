# 已有任务实体与运动关系恢复

代码基线：`b32dfbe5108be41ddb2d866c5813e6ff391e2918`。

`load_observed_owner(path)` 从 JSON 所指的 `FrozenObservedField` 文件恢复原样本、权重、支持和实体坐标，结合 `initial_pose`、attachments、owner_id 与来源生成 `CollisionOwner`。已有实体的恢复与当前可见深度无关。

`problem_owner_binding(problem, owner_id)` 消费已经编译的同一实体各阶段，取得固定初始位姿及去重后的附着区间。输出保留原条件、释放身份和进度，工具到实体的固定变换来自已声明关系。

已知机器人由 `robot_validation_fixture` 构造。此模块只恢复任务实体与关系，供 `run_recorded_skill` 读取已有实体及对应运动输入。
