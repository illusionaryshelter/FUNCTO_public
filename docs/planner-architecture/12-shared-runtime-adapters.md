# 通用规划运行适配、固定机器人资源与结果显示

本组件代码基线为 `1e21ae6`；本次为同代码内容的文档登记。原控制、命令续接和独立评价代码基线为 `3266476`，其计算语义见[设备控制与评价](11-device-control-and-evaluation.md)。

## 输入、计算与坐标

适配层位于 `tests/planning`。它将显式请求中的完整技能表示、标定 RGB-D、机器人关节反馈和固定机器人参数交给生产规划器，并保存该次返回结果。物理模型及其评价目录只在仿真运行侧使用。所有技能共用同一构造与调用接口；约定的 grasp/manipulation 分类仍由源程序和原评价接口解释。

世界坐标 W 由标定定义；`T_A_B` 把 B 中坐标变换到 A，平移单位为 m、转角单位为 rad。具名机械臂关节采用 rad，棱柱手指采用 m；速度与加速度分别使用对应单位除以 s、s²。正式轨迹时间从本次曲线入口计起，物理时间由仿真后端提供，源演示时间由保存的进度映射提供。

## 规划调用

`run_preference_request.run(request_path,factory_path,output)` 从请求读取源数据路径、观测路径、角色声明与当前设备命令。显式问题工厂提供 `prepare(request)` 和 `build_problem(request)`：前者生成观测与机器人绑定输入，后者返回 `PlanningProblem` 和来源记录。适配器把 `incoming_command` 作为独立设备状态写入准备后的反馈记录，以 `bind_pose_objectives` 绑定完整角色声明，再调用 `run_prepared`。普通续接的终止合同直接取正式绑定中的 `validation.export.reference_termination`。输出为同一次 `PlanningResult`，并保存规划命令和结果索引。

`run_recorded_skill.run_prepared(args,problem,provenance,backend_factory=MotionBackend)` 负责 RGB-D 自体过滤、TSDF 与观测支持域、设备与机器人绑定、预算及结果归档。`backend_factory(robot,robot_validator=...,budget=...)` 创建这一次请求使用的驻留后端；preference 入口提供 `PreferenceMotionBackend`，默认入口沿用 `MotionBackend`。该后端的 `plan` 产生结果，其 `close` 在拥有它的调用结束时释放资源。函数返回同一结果，不再次构造任务或重采样参考。

`recorded_task_planning.plan_and_record` 调用传入后端的 `plan` 一次，再按完成状态保存正式轨迹、拒绝诊断曲线、完整执行绑定和累计请求账本。preference 诊断中的 `joint_problem` 与 `objective_trials` 分别写入 `planner-internals/joint-problem.json` 和 `objective-trials.json`。失败曲线保留原拒绝来源，规划未产生曲线时保留缺失状态。此步骤只消费完成证据。

普通参考 IK 的运行设置由 [native_setup](../preference-planner/native-setup.md) 管理。原生模型只构造一次，数值预算、事件目标与任务几何继续由原接口传递。

## 固定机器人与独立场景输入

`config/robots/fixed_franka/robot-fixture.json` 保存固定机器人的具名关节、力学范围、设备控制范围、工具与固定基座绑定、官方碰撞实体和机械接触声明；同目录 `kinematics.yml`、`kinematics.urdf` 和 `robot_only.xml` 保存原运动学及官方机器人本体。模型包括官方固定末端执行器。机器人及配置跨任务固定，因此部件形状和固定关系可复用，当前位姿由标定、关节反馈和 FK 计算。

`scene/<scene>/requests` 保存场景选用的请求模板、源录制与评价身份。当前运行目录中的 `execution-fixture.json` 合并固定机器人参数与该次仿真 `episode_path/catalog_path/source_scene_xml`，仅供仿真执行及独立评价。规划请求直接读取固定机器人文件。源技能的功能区域、物体—工具关系、轨迹与事件仍由源表示提供；任务实体碰撞表示来自现场 RGB-D 的 TSDF 及观测支持域。

`simulation_model.build_simulation_model(scene_path,robot_root=...)` 以 MuJoCo `MjSpec` 编译显式物理场景。它遍历声明的固定机器人子树，对原 `contype!=0 or conaffinity!=0` 的物理 geom 设置原生 `fluid_ellipsoid=1`。流体形状因此由已有官方物理形状确定；视觉副本不重复计入，其他对象与无物理 geom 的本体保留原生定义。函数返回 `MjModel` 和所选具名 geom 的来源记录，供环境恢复前使用。

`current_catalog.save_current_catalog(simulation,source_catalog_path,scene_path,output_path)` 根据源实体的具名 geom/site 在当前已编译模型中解析 ID，再调用既有 `collect.gt.model_catalog` 保存实际模型目录。源模型与当前模型的数字 ID 不要求相等；目录记录当前仿真实体与关节身份，供独立评价解释实际状态。它不推进物理时间。

`recorded_entrance.reference_entrance(planned_feedback)` 将保存的具名 `start_q/start_dq/eef_pose/sim_time_s` 转为执行器入口记录，保留坐标、来源、反馈身份与时间。执行器把该参考入口与当前机器人反馈分别读取和核对。

## 结果显示

`visualize_preference_result.py` 接收保存的 run/request/resources 和输出目录，调用保存曲线的 Hermite 多项式与官方机器人 FK，生成可直接打开的 `index.html`。`robot_poses` 计算参考、候选与物理轨迹的全程工具位姿；物理位姿来自执行记录。`archive` 读取数值档案，`nearest_indices` 建立显示时刻与原样本的对应；投影使用原图像标定。

`result_template.html` 将源区域、迁移区域、采样点与轨迹叠加在原场景中，保留缩放、共享时间选择、当前位置和姿态坐标轴、旋转数值及时间来源。每条轨迹只显示当前时刻的一个姿态。拒绝与未知采用原记录，评价指标与实际操作图像各自显示。`check_preference_display.js` 对生成页读取全部轨迹时刻，核对显示坐标和单一当前姿态的接口。

## 核对范围

`test_reference_runtime` 直接使用现有 cuRobo 配置类，检查不同迭代预算、原模型与成本对象、收敛阈值和分组数量。`test_recorded_task_planning` 与 `test_recorded_planning_ledger` 检查后端恰好调用一次、拒绝诊断来源和重试累计账本。`test_simulation_model` 检查机器人流体 geom 选择、原生惯量定义保留、实际机器人状态布局和物理参数不变。连续场景运行及资源峰值由对应阶段实验记录给出。
