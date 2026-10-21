# 实现符号索引

代码基线：`9e8b365`。本索引覆盖下表所列当前组件及实验适配器；各阶段职责、必要输入和调用关系见[API 契约](06-api-contracts.md)。

`build_api_index.py` 从源码语法树生成每个类、函数、方法和局部定义。签名、显式返回、属性读取、写入目标和直接调用均保留源码表达式。属性可能在函数内部建立；调用前置条件由组件正文解释。正常执行至函数末尾时隐式返回 None。

覆盖 **53 个文件、932 个定义**。

| 模块 | 定义数 | 组件说明 |
|---|---:|---|
| [manipulation/representation.py](api/representation.md) | 18 | [01](01-inputs-and-task-model.md) |
| [manipulation/program.py](api/program.md) | 20 | [01](01-inputs-and-task-model.md) |
| [manipulation/motion.py](api/motion.md) | 21 | [01](01-inputs-and-task-model.md) |
| [manipulation/function_motion.py](api/function_motion.md) | 5 | [01](01-inputs-and-task-model.md) |
| [manipulation/goals.py](api/goals.md) | 14 | [01](01-inputs-and-task-model.md) |
| [manipulation/grasp.py](api/grasp.md) | 8 | [01](01-inputs-and-task-model.md) |
| [manipulation/skill.py](api/skill.md) | 13 | [01](01-inputs-and-task-model.md) |
| [manipulation/planning_problem.py](api/planning_problem.md) | 110 | [01](01-inputs-and-task-model.md) |
| [manipulation/regions.py](api/regions.md) | 21 | [01](01-inputs-and-task-model.md) |
| [manipulation/support_grounding.py](api/support_grounding.md) | 4 | [01](01-inputs-and-task-model.md) |
| [manipulation/pipeline.py](api/pipeline.md) | 10 | [01](01-inputs-and-task-model.md) |
| [manipulation/grounding.py](api/grounding.md) | 13 | [01](01-inputs-and-task-model.md) |
| [manipulation/selection.py](api/selection.md) | 7 | [01](01-inputs-and-task-model.md) |
| [manipulation/planning_observation.py](api/planning_observation.md) | 14 | [02](02-observation-and-collision.md) |
| [manipulation/robot_self_filter.py](api/robot_self_filter.md) | 6 | [02](02-observation-and-collision.md) |
| [manipulation/online_geometry.py](api/online_geometry.md) | 18 | [02](02-observation-and-collision.md) |
| [manipulation/planning_scene.py](api/planning_scene.md) | 26 | [02](02-observation-and-collision.md) |
| [manipulation/collision_records.py](api/collision_records.md) | 4 | [02](02-observation-and-collision.md) |
| [manipulation/observations.py](api/observations.md) | 16 | [02](02-observation-and-collision.md) |
| [manipulation/contact_binding.py](api/contact_binding.md) | 2 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/initial_support.py](api/initial_support.md) | 9 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/gripper_state.py](api/gripper_state.md) | 3 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/gripper_trajectory.py](api/gripper_trajectory.md) | 29 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/planning.py](api/planning.md) | 165 | [04](04-solver-and-search.md) |
| [manipulation/planning_ik.py](api/planning_ik.md) | 7 | [04](04-solver-and-search.md) |
| [manipulation/cubic_trajectory.py](api/cubic_trajectory.md) | 19 | [04](04-solver-and-search.md) |
| [manipulation/physical_joint_domain.py](api/physical_joint_domain.md) | 21 | [04](04-solver-and-search.md) |
| [manipulation/physical_joint_qp.py](api/physical_joint_qp.md) | 14 | [04](04-solver-and-search.md) |
| [manipulation/curobo_solver_components.py](api/curobo_solver_components.md) | 38 | [04](04-solver-and-search.md) |
| [manipulation/functional_augmented_lagrangian.py](api/functional_augmented_lagrangian.md) | 9 | [04](04-solver-and-search.md) |
| [manipulation/optimizer_device_checks.py](api/optimizer_device_checks.md) | 12 | [04](04-solver-and-search.md) |
| [manipulation/resource_admission.py](api/resource_admission.md) | 5 | [04](04-solver-and-search.md) |
| [manipulation/tensor_cpu_snapshot.py](api/tensor_cpu_snapshot.md) | 1 | [04](04-solver-and-search.md) |
| [manipulation/runtime_profiling.py](api/runtime_profiling.md) | 2 | [04](04-solver-and-search.md) |
| [manipulation/execution.py](api/execution.md) | 43 | [05](05-acceptance-and-execution.md) |
| [manipulation/execution_binding.py](api/execution_binding.md) | 9 | [05](05-acceptance-and-execution.md) |
| [manipulation/robot_validation.py](api/robot_validation.md) | 11 | [05](05-acceptance-and-execution.md) |
| [manipulation/robot_geometry_check.py](api/robot_geometry_check.md) | 15 | [05](05-acceptance-and-execution.md) |
| [manipulation/robosuite_execution.py](api/robosuite_execution.md) | 25 | [05](05-acceptance-and-execution.md) |
| [manipulation/planning_diagnostics.py](api/planning_diagnostics.md) | 15 | [05](05-acceptance-and-execution.md) |
| [tests/planning/run_recorded_skill.py](api/adapter-run_recorded_skill.md) | 13 | [05](05-acceptance-and-execution.md) |
| [tests/planning/recorded_task_planning.py](api/adapter-recorded_task_planning.md) | 5 | [05](05-acceptance-and-execution.md) |
| [manipulation/implicit_distance.py](api/implicit_distance.md) | 28 | [02](02-observation-and-collision.md) |
| [manipulation/observed_mapping.py](api/observed_mapping.md) | 10 | [02](02-observation-and-collision.md) |
| [manipulation/observed_robot.py](api/observed_robot.md) | 4 | [02](02-observation-and-collision.md) |
| [manipulation/observed_collision.py](api/observed_collision.md) | 5 | [02](02-observation-and-collision.md) |
| [manipulation/observed_coverage.py](api/observed_coverage.md) | 4 | [02](02-observation-and-collision.md) |
| [tests/planning/observed_robot_fixture.py](api/adapter-observed_robot_fixture.md) | 2 | [05](05-acceptance-and-execution.md) |
| [manipulation/grasp_region.py](api/grasp_region.md) | 4 | [03](03-attachment-contact-and-hand.md) |
| [manipulation/native_field_distance.py](api/native_field_distance.md) | 7 | [02](02-observation-and-collision.md) |
| [manipulation/tool_geometry.py](api/tool_geometry.md) | 11 | [02](02-observation-and-collision.md) |
| [tests/planning/robot_fixture.py](api/adapter-robot_fixture.md) | 31 | [02](02-observation-and-collision.md) |
| [tests/planning/robot_validation_fixture.py](api/adapter-robot_validation_fixture.md) | 6 | [02](02-observation-and-collision.md) |
