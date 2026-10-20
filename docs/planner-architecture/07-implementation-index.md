# 实现符号索引

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。本索引逐项对应所列 Python 文件中的类、函数、方法及工厂内局部定义。API 的阶段职责、输入语义和消费者见[接口契约](06-api-contracts.md)，数学过程见前五章。

索引直接读取代码语法树。函数条目保留完整签名、返回类型声明、所有显式返回或生成表达式、直接读取的对象属性、写入目标和直接调用符号；类条目保留继承及具名类型字段。返回分支按源码顺序列出。对象属性列表记录本函数出现的读取操作，其中部分属性在本函数内建立；调用前置条件由组件正文逐项说明。调用记录保留源码表达式，可结合对应组件章确定具体运行对象。函数正常执行至末尾时隐式返回 `None`；可见写入目标和直接调用分别列出。

覆盖 **70 个文件、1303 个定义**。各文件的条目数量包括内部辅助函数与运行时工厂类。

| 模块 | 定义数 | 组件说明 |
|---|---:|---|
| [manipulation/representation.py](api/representation.md) | 20 | [01](01-inputs-and-task-model.md) |
| [manipulation/program.py](api/program.md) | 19 | [01](01-inputs-and-task-model.md) |
| [manipulation/motion.py](api/motion.md) | 21 | [01](01-inputs-and-task-model.md) |
| [manipulation/function_motion.py](api/function_motion.md) | 5 | [01](01-inputs-and-task-model.md) |
| [manipulation/goals.py](api/goals.md) | 14 | [01](01-inputs-and-task-model.md) |
| [manipulation/grasp.py](api/grasp.md) | 8 | [01](01-inputs-and-task-model.md) |
| [manipulation/skill.py](api/skill.md) | 13 | [01](01-inputs-and-task-model.md) |
| [manipulation/planning_problem.py](api/planning_problem.md) | 104 | [01](01-inputs-and-task-model.md) |
| [manipulation/regions.py](api/regions.md) | 21 | [01](01-inputs-and-task-model.md) |
| [manipulation/support_grounding.py](api/support_grounding.md) | 4 | [01](01-inputs-and-task-model.md) |
| [manipulation/pipeline.py](api/pipeline.md) | 10 | [01](01-inputs-and-task-model.md) |
| [manipulation/grounding.py](api/grounding.md) | 13 | [01](01-inputs-and-task-model.md) |
| [manipulation/selection.py](api/selection.md) | 7 | [01](01-inputs-and-task-model.md) |
| [manipulation/planning_observation.py](api/planning_observation.md) | 31 | [02](02-observation-and-collision.md) |
| [manipulation/robot_self_filter.py](api/robot_self_filter.md) | 6 | [02](02-observation-and-collision.md) |
| [manipulation/tsdf_geometry.py](api/tsdf_geometry.md) | 28 | [02](02-observation-and-collision.md) |
| [manipulation/online_observed_geometry.py](api/online_observed_geometry.md) | 14 | [02](02-observation-and-collision.md) |
| [manipulation/online_geometry.py](api/online_geometry.md) | 46 | [02](02-observation-and-collision.md) |
| [manipulation/planning_scene.py](api/planning_scene.md) | 96 | [02](02-observation-and-collision.md) |
| [manipulation/tsdf_spatial_query.py](api/tsdf_spatial_query.md) | 8 | [02](02-observation-and-collision.md) |
| [manipulation/scene_surface_batch.py](api/scene_surface_batch.md) | 10 | [02](02-observation-and-collision.md) |
| [manipulation/native_voxel_sources.py](api/native_voxel_sources.md) | 4 | [02](02-observation-and-collision.md) |
| [manipulation/voxel_layers.py](api/voxel_layers.md) | 22 | [02](02-observation-and-collision.md) |
| [manipulation/mapper_surface.py](api/mapper_surface.md) | 1 | [02](02-observation-and-collision.md) |
| [manipulation/collision_records.py](api/collision_records.md) | 4 | [02](02-observation-and-collision.md) |
| [manipulation/observations.py](api/observations.md) | 16 | [02](02-observation-and-collision.md) |
| [manipulation/contact_binding.py](api/contact_binding.md) | 4 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/online_contact.py](api/online_contact.md) | 18 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/contact_geometry.py](api/contact_geometry.md) | 43 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/native_owner_geometry.py](api/native_owner_geometry.md) | 2 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/holding_contact.py](api/holding_contact.md) | 5 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/initial_support.py](api/initial_support.md) | 10 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/gripper_state.py](api/gripper_state.md) | 3 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/gripper_trajectory.py](api/gripper_trajectory.md) | 29 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/prismatic_self_collision.py](api/prismatic_self_collision.md) | 7 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/tool_geometry.py](api/tool_geometry.md) | 11 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/planning.py](api/planning.md) | 183 | [04](04-solver-and-search.md) |
| [manipulation/planning_ik.py](api/planning_ik.md) | 7 | [04](04-solver-and-search.md) |
| [manipulation/cubic_trajectory.py](api/cubic_trajectory.md) | 19 | [04](04-solver-and-search.md) |
| [manipulation/physical_joint_domain.py](api/physical_joint_domain.md) | 21 | [04](04-solver-and-search.md) |
| [manipulation/physical_joint_qp.py](api/physical_joint_qp.md) | 14 | [04](04-solver-and-search.md) |
| [manipulation/curobo_solver_components.py](api/curobo_solver_components.md) | 38 | [04](04-solver-and-search.md) |
| [manipulation/functional_augmented_lagrangian.py](api/functional_augmented_lagrangian.md) | 9 | [04](04-solver-and-search.md) |
| [manipulation/collision_path.py](api/collision_path.md) | 6 | [04](04-solver-and-search.md) |
| [manipulation/optimizer_device_checks.py](api/optimizer_device_checks.md) | 12 | [04](04-solver-and-search.md) |
| [manipulation/resource_admission.py](api/resource_admission.md) | 5 | [04](04-solver-and-search.md) |
| [manipulation/tensor_cpu_snapshot.py](api/tensor_cpu_snapshot.md) | 1 | [04](04-solver-and-search.md) |
| [manipulation/runtime_profiling.py](api/runtime_profiling.md) | 2 | [04](04-solver-and-search.md) |
| [manipulation/execution.py](api/execution.md) | 44 | [05](05-acceptance-and-execution.md) |
| [manipulation/execution_binding.py](api/execution_binding.md) | 9 | [05](05-acceptance-and-execution.md) |
| [manipulation/robot_validation.py](api/robot_validation.md) | 12 | [05](05-acceptance-and-execution.md) |
| [manipulation/robot_geometry_check.py](api/robot_geometry_check.md) | 35 | [05](05-acceptance-and-execution.md) |
| [manipulation/robot_coverage.py](api/robot_coverage.md) | 17 | [05](05-acceptance-and-execution.md) |
| [manipulation/geometry_motion_bound.py](api/geometry_motion_bound.md) | 14 | [05](05-acceptance-and-execution.md) |
| [manipulation/robosuite_execution.py](api/robosuite_execution.md) | 26 | [05](05-acceptance-and-execution.md) |
| [manipulation/planning_diagnostics.py](api/planning_diagnostics.md) | 15 | [05](05-acceptance-and-execution.md) |
| [manipulation/online_native_confirmation.py](api/online_native_confirmation.md) | 14 | [05](05-acceptance-and-execution.md) |
| [tests/planning/prepare_recorded_task.py](api/adapter-prepare_recorded_task.md) | 14 | [05](05-acceptance-and-execution.md) |
| [tests/planning/run_recorded_skill.py](api/adapter-run_recorded_skill.md) | 13 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_task_planning.py](api/adapter-recorded_task_planning.md) | 5 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_owner_geometry.py](api/adapter-recorded_owner_geometry.md) | 7 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_observation.py](api/adapter-recorded_observation.md) | 5 | [05](05-acceptance-and-execution.md) |
| [tests/planning/robot_fixture.py](api/adapter-robot_fixture.md) | 31 | [05](05-acceptance-and-execution.md) |
| [tests/planning/robot_validation_fixture.py](api/adapter-robot_validation_fixture.md) | 2 | [05](05-acceptance-and-execution.md) |
| [tests/planning/robot_coverage_fixture.py](api/adapter-robot_coverage_fixture.md) | 1 | [05](05-acceptance-and-execution.md) |
| [tests/planning/contact_fixture.py](api/adapter-contact_fixture.md) | 1 | [05](05-acceptance-and-execution.md) |
| [tests/planning/run_recorded_simulation.py](api/adapter-run_recorded_simulation.md) | 11 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_simulation.py](api/adapter-recorded_simulation.md) | 3 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_skill_evaluation.py](api/adapter-recorded_skill_evaluation.md) | 23 | [05](05-acceptance-and-execution.md) |
| [tests/planning/continuous_simulation.py](api/adapter-continuous_simulation.md) | 7 | [05](05-acceptance-and-execution.md) |
