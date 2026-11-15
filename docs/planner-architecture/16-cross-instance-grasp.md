# 跨实例抓取、官方机器人几何与观测场

代码基线为 FUNCTO `561f768`。本文描述本轮集成工作树中的实际实现；新增代码的集成提交引用由总架构入口在提交时统一登记。对应源码为 [cross_instance_grasp.py](../../manipulation/cross_instance_grasp.py)、[grasp_pose_refinement.py](../../manipulation/grasp_pose_refinement.py)、[grounding_robot_geometry.py](../../manipulation/grounding_robot_geometry.py) 与 [grounding_geometry_evidence.py](../../manipulation/grounding_geometry_evidence.py)。模型进程接口见 [local_models/ARCHITECTURE.md](../../local_models/ARCHITECTURE.md)。

## 研究对象与坐标

组件的输入是已加载的技能表示、已完成的跨实例区域对应、当前现场的标定 RGB-D、图像采集时刻的机器人关节反馈和固定官方机器人模型。技能表示包含完整 `H_g`、`C_D`、源示教物体与工具轨迹、功能坐标、区域、交互事件及释放事件。区域对应提供源点身份、当前观测元素身份、两者的显式分配以及当前任务功能坐标 `F`。

每个公共 view 的 `observation/feature/crop_xyxy` 描述所选功能 reference 的观测域。`owner_observation` 是独立的 owner RGB-D bank，`owner_mask` 保留该 bank 的原图 mask；`owner_evidence` 的 `owner_id/scope/source` 声明注册身份、可见证据范围和来源，`reference_material_owner_id/reference_material_source` 声明本次 reference 材料的几何 owner 和来源。object scope 中两份观测可以共享同一 bank；功能部件及其 owner 的原 mask 和点身份分别保留。`background_mask` 明确声明静态背景端点。局部 owner 证据以外尚无背景身份声明时，该 mask 为空，相关像素保持未分配状态。

区域证据组件在同一个完整深度 surface 上按原观测身份建立两个材料来源：`independent_owner_*` 保留独立 owner 观测，`local_material_*` 保留显式声明属于同一 owner 的 reference 实测材料。两者的 observation/element ID 经唯一化形成 owner 模型点域，原 surface 成员及三个原图 mask 保留。局部材料的 canonical 像素与视角索引由 `local_material_observation_pixels/local_material_observation_view_indices` 给出，消费端直接使用该来源完成分区。

矩阵 `T_A_B` 将 B 坐标系中的齐次列坐标变换到 A；平移、深度和关节滑移量均以米表示，旋转量以弧度表示。源示教世界记为 `W_s`，当前标定世界记为 `W_t`。`E` 是机器人实际控制的工具坐标，`G` 是同一固定末端执行器的抓取坐标，`T_tool_grasp` 表示 `E←G`。`G` 的 +X 为闭合方向、+Z 为接近方向。AnyGrasp 原生坐标 A 的 +X 为接近方向、+Y 为闭合方向。

源功能坐标 `F_s` 与完整关系 `H_g(t)` 产生源图像捕获坐标中的名义工具关系 `E_s=F_s H_g(t_g)`，其中 `t_g` 是技能声明的交互时间。`source_capture_grasp` 通过共享 `generate_reference` 的恒等父坐标引用原事件时钟和插值；采集反馈提供该调用的实际入口工具位姿。源物体图像位姿与 `C_D(t_g)` 的组成另存为来源比较。

当前任务坐标 `F` 与工具运动锚点 `B` 分别保存。每个最终工具假设 `E_i` 产生 `B_i=E_i H_g(t_g)^{-1}`。共享参考生成器随后计算 `B_i H_g(t)`，并继续维护原事件、释放和入口边界。`F` 仍用于任务几何和区域坐标。每条完整参考分别进入原规划器，轨迹的最终选择由完整规划结果完成。

## 实际计算与数据调用关系

`run_cross_instance_grasp` 从区域结果的当前 owner 元素序列构造 AnyGrasp 输入，采用第一个声明视角的标定将点变成 float32 相机坐标。`target_owner_element_ids` 与 `target_owner_points_world_m` 一一对应并保留输入顺序；`target_element_ids` 与 `matching_target_indices` 描述独立的区域 reference 域。`record_owner_model_input` 先求每个源行的 `region_ids=target_element_ids[matching_target_indices]`，再按严格相等的 element ID 得到 owner 输入行及 steering mask。未进入所供 owner 观测域的 region ID 保留其 source-row ID、`-1` owner 行和 unsupported 状态；空间重合不建立元素身份。object scope 的两个域相同时，点输入和 steering mask 保留原字节。`AnyGraspBackend.predict` 运行独立模型进程并返回原生 17 列输出、网络实际消费的体素点、该点序列的区域 mask 和采样 seed 身份。

`generate_and_rank` 对完整原生行做精确去重，保留第一次出现的原始行身份和可恢复的反向映射。每个不同原生行绑定两个实际机器人手指标签。原生插入深度定位指尖前导支撑面；官方末端执行器的固定几何关系将其转换为实际 `E`。候选宽度通过官方开闭关节状态的仿射孔径关系变成两个明确的名义滑移坐标。

区域关系提供软位姿先验 `E_prior` 和尺度 `s=target_rms_m`。原评分为

`J = ||p-p_prior||²/s² + ||R-R_prior||²_F/8 + 1-quality`。

三项权重均为 1，稳定排序保留精确同分候选的输入顺序。完整工具旋转参与第二项，包括绕接近轴的旋转。原生 quality 属于原 AnyGrasp 标签类及其实际绑定候选。

选中的原始 `E` 接受两个独立静态手形检查：`entrance_open` 使用采集关节编码器，`native_width` 使用选中候选的名义孔径。这两个输入状态共享同一个冻结工具位姿。`query_tool_candidates` 分别积分独立 `owner_mask` 与模型准入的显式背景，保存每个官方工具实体的三态查询与见证。`background_model_masks` 以 canonical 局部材料的原像素排除背景端点，原 `background_mask` 单独保存。这样，owner 语义 mask 漏分而实际已声明属于同一 owner 的 reference 材料保持其模型身份。空的背景证据产生未知支持。

随后 `local_models.unopose.predict_batch` 将源 RGB-D 与每个真实当前视角组成独立 pair。当前 query mask 消费各 view 的独立 `owner_mask`，该行同时记录其 `owner_evidence`。一次模型前向保留全部行。原始旋转 `R_query_reference` 经 SO(3) 表示后形成

`R_world_tool = R_world_query_camera R_query_reference R_world_source_cameraᵀ R_source_tool`。

每个行假设保留原始选中 `E` 的位置。原生模型平移和分数作为该行预测记录保存。方向改动后的抓取质量标记为 `unmeasured`。

`prepare_grasp_scene` 另行构造最终姿态细化所用的背景。其 CPU 观测阶段 `prepare_grasp_observations` 使用完整官方机器人在采集关节 FK 下对深度图进行机器人自过滤，再将独立 owner mask 与 `background_model_masks` 给出的背景模型候选传给现有 `_self_filtered_views/_partition_masks`。原语义边界和射线运算保留；独立 owner、已声明局部材料与静态背景之外的已观测像素记录为未分配。局部材料准入另存为 `admitted_local_material_mask`，三个原图 mask 分别归档。`owner_id` 是调用方显式提供的当前注册实体身份；`carrier`、`reference` 等技能职责名不代替实体身份。所有方向假设共用一次背景积分。该最终背景与前述静态 mask 诊断具有不同的分区及支持语义，分别保存。

`refine_pose` 在同一 TCP 原点上作两坐标旋转细化。两个正交工具轴来自官方 `T_tool_grasp` 的前两列；其叉积定义不单独参数化的旋转方向。仅原始实体查询确认 `intersection` 时调用 SLSQP。目标为 `0.5 qᵀq`，约束为显式最大角域及有观测支持的 TSDF 标量值。`field_margin_m` 表示 TSDF 优化目标值，几何含义由原始观测场决定。未知样本保存未知状态，并在数值约束中使用常数 1、零导数。最终结果使用原注册实体查询重新分类。优化器状态、提议位姿、前后见证和完整迭代记录同时保存。每个视角的提议均进入输出假设集合，完整轨迹规划负责各自的正式求解。

## 公共 API

| API | 必需数据、计算与输出 | 实际调用方 |
| --- | --- | --- |
| `run_cross_instance_grasp(skill, region_result, views, *, feedback, robot_fixture, owner_id, mapping_settings, max_tilt_rad, field_margin_m, approach_thresh_rad, checkpoint_path=None)` | 输入已加载技能、完成的区域结果和同序视角；依次执行原生抓取、静态检查、批量方向预测、统一背景和逐假设旋转细化；返回 `status/hypotheses/arrays/report`。每个 hypothesis 含 `id/T_world_function/T_world_tool/tool_reference_anchor_world/source`。 | 统一 skill grounding 入口；完整参考由该入口继续交给共享生成器。 |
| `source_capture_grasp(skill, source_frame_world, *, T_world_eef)` | 以完整 `H_g`、源 `C_D`、源图像位姿和实际入口 E 形成 `H_g(t_g)`、源捕获工具位姿及来源比较。 | 上述完整抓取入口。 |
| `record_owner_model_input(arrays, T_world_camera, input_camera_view_id)` | 保留 owner 点和 `target_owner_element_ids` 的原顺序；用 `target_element_ids[matching_target_indices]` 严格匹配 owner ID，返回 float32 相机点、原域 steering mask 和来源说明，并保存逐源行对应及全部 unsupported ID。 | 原生抓取入口。 |
| `generate_and_rank(points_camera_m, source_G, evidence, backend, seed=0, *, T_world_camera, region_mask, approach_thresh)` | 调用原生模型，精确全行去重、双标签官方机器人绑定和原三项评分；返回全部候选、原始网络输入、采样身份、排名和模型来源。 | 完整抓取入口；纯 CPU 差分可提供明确的原生行后端。 |
| `exact_native_rows(rows)` | 对所有 17 列精确分组；返回第一次出现索引、原行到稳定唯一行的映射和重复次数。 | `generate_and_rank`。 |
| `NoGraspCandidates` | 已完成原生预测但无候选时，携带原始控制量、输入消费情况、模型来源及实际可得的采样证据；完整入口另附实际模型点、owner/region element 对应与 unsupported 源行。 | 原生抓取调用的失败归档。 |
| `transport_orientation(raw_R, R_world_source_camera, R_world_query_camera, R_source_tool)` | 将原始相机间旋转、源/当前标定和源工具旋转组成当前工具方向。 | 每个真实 UNOPose batch 行。 |
| `refine_pose(T_world_tool, parts, field, basis_tool, max_tilt_rad, field_margin_m)` | 输入冻结 E、显式官方实体及其工具局部变换、统一观测场和两轴角域；返回旋转提议、三态查询、导数证据和优化器全过程。 | 每条方向假设。 |
| `query_tool_candidates(views, context, geometry, tool_poses, candidate_ids, interaction_time, observation_model, mask_scopes, *, finger_joint_positions_m, inspection_state_ids)` | 对显式 mask scope 分别积分并查询完整工具形状。`context.observation_source/capture_time_s` 只记录观测来源和采集时钟；返回场数组及按候选、手形、实体组织的查询记录。 | 原始 AnyGrasp E 的静态结果检查。 |
| `background_model_masks(views, local_material_observation_pixels, local_material_observation_view_indices)` | 直接索引 canonical owner 局部材料的原测量像素，从原背景候选中排除这些材料；返回每视角背景模型候选、局部材料像素 mask 和含来源/范围的准入记录。 | 静态背景检查与最终背景 CPU 准备共用。 |
| `prepare_grasp_observations(views, *, fixture, capture_frames, reference_frame, mapping_settings, owner_id, local_material_observation_pixels, local_material_observation_view_indices)` | 从公共 view 的独立 owner、显式背景和 canonical 局部材料执行原机器人自过滤和语义边界分区；返回标准化观测、实际端点分区、每视角范围说明及原 mask/支持数组。 | 最终背景的 CPU 准备阶段及输入语义回归测试。 |
| `prepare_grasp_scene(views, *, fixture, capture_feedback, tool_geometry, entrance_finger_positions, native_finger_positions, T_tool_grasp, tool_poses_world, max_tilt_rad, mapping_settings, owner_id, local_material_observation_pixels, local_material_observation_view_indices)` | 从官方模型和当前 RGB-D 构造 TCP 组件、两种手指状态、采集自过滤、分区和统一背景；返回 `parts/basis_tool/field/states/metadata/arrays`。 | 完整抓取入口的最终方向细化。 |
| `storage_domain(parts, tool_poses_world, max_tilt_rad, settings)` | 用完整官方实体的极值与 `2ρ sin(max_tilt/2)` 包围旋转后的采样范围，再加入截断和插值存储边界；返回有限栅格范围。 | 背景积分前的数组配置。 |
| `group_state_checks(checks, states, guide)` | 将同一刚性组件查询在各明确手指状态下归组，附上该状态原始关节配置与采样支持。 | 逐假设旋转结果记录。 |
| `build_robot_validator(fixture, *, reference_frame, T_reference_base, tool_frame, T_frame_tcp)` | 从官方机器人 MJCF/URDF、命名关节与机械许可建立完整 Coal/FK 模型、原生 pair 策略及模型一致性检查。 | 生产规划请求与机器人模型距离评估。 |
| `robot_self_filter_shapes(fixture, frame_poses)` | 将同一官方机器人各原始实体置于采集关节 FK 给出的世界位姿。 | 最终背景观测预处理和生产规划观测装配。 |
| `tool_geometry_names(fixture)` | 按已声明手指机制的机器人拓扑识别工具组件；输出注册几何 ID。 | 生产场景的工具成员声明。 |
| `build_tool_parts(fixture, feedback, entrance_finger_positions, native_finger_positions, tool_geometry)` | 保留与 TCP 同一 parentJoint 的固定组件及声明手指后代的各明确状态；返回实体、状态、采集 frame FK、组件说明、按组件排列的数值数组及机器人说明。 | `prepare_grasp_scene`。 |

`_generation_arrays` 仅发布已完成的模型/评分数组。`_known_shapes` 保留官方几何实体；`_native_geom_pose` 使用 MuJoCo 原始四元数矩阵约定；`_native_ignored_pairs` 与 `_native_pair_policies` 保留原生 welded/parent/bitmask 语义。`_model_q`、`_frame_placements` 和 `_depends_on_joint` 负责命名坐标、FK 与原拓扑成员查询。`_coal_geometry` 复建原始机器人 convex、box 或 cylinder。姿态细化的 `_check_parts` 查询各注册实体，`_relation` 汇总已完成实体分类。

## 依赖、证据与结果含义

区域输入来自 `transfer.cross_instance`；模型边界为 `local_models.anygrasp` 和 `local_models.unopose`；源参考运算为 `manipulation.representation`。机器人部分使用 MuJoCo、Pinocchio、Coal 和现有 `tool_geometry`；现场场使用已有 `planning_observation`、`robot_self_filter`、`observed_mapping` 和 `native_field_distance`。这些生产模块的导入关系在 FUNCTO 内闭合。

机器人及安装的末端执行器是固定官方配置，其原始形状和组件固定关系可跨任务复用；校准、关节编码器和 FK 确定当前姿态。现场任务实体及背景的几何由输入 RGB-D 的 TSDF 与观测支持表达。源示教的区域、相对关系及事件参与模型与参考计算。当前现场真实物体位姿、真实分割、接触和效果的仿真特权状态属于独立事后评估数据。

返回数组保留全部原候选行身份、原评分、实际网络消费点与 mask、seed、逐源行 region→owner 身份对应及全部 unsupported ID、两类 owner 材料来源 ID、UNOPose 的六类模型输入、原始旋转/平移/分数、独立 reference/owner/background 原 mask、局部材料与背景模型准入 mask、静态与最终观测场、官方组件变换及前后查询见证。`report` 记录 owner 可见证据来源、背景准入的 canonical 材料来源和排除范围、原质量所对应的位姿、方向改动后的质量语义和每条细化路径。实际物理轨迹在此组件完成时处于 `missing_not_executed`，其后由原规划/执行接口提供。
