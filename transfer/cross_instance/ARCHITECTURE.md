# 跨实例区域对应与功能坐标

代码基线：FUNCTO `96ee2dc`。本文件描述 `transfer/cross_instance/` 当前工作区实现；集成提交形成后，由集成者将这里更新为该实际实现提交。实验数值依据是 `2026-11-02-functo-presolve-integration` 的 `grounding_inputs.py`、`native_world_response.py`、`candidate_context.py`、`context_relations.py` 和 `neural_refinement.py` 中实际运行的数组计算。正式运行依赖本目录、库内公共组件及现有第三方依赖。

## 1. 研究对象和计算职责

该组件计算源功能区域在当前同类别不同实例上的区域对应。源区域中的每个原始观测行保持独立身份；每个当前目标元素保持原几何关联得到的身份。计算顺序为原生 MARCO 响应、移动区域中心的刚体粗配准、残差 SIREN 非刚体细化、完整对应的功能坐标读出。输出中的 `F = region_frame['T_world_region']` 是当前任务功能坐标，其用途是连接任务几何及后续参考生成。

公共入口已经接收加载完成的源技能和选定的当前实体观测。实体选择、源—目标实例关系判断、机器人反馈采集由调用者提供。工具候选计算消费本组件的完整映射及区域/实体关系；工具参考绑定和完整源运动的时间处理由各自组件执行。这样，区域计算的输入、几何域、优化器和结果能够独立复放。

## 2. 坐标、单位和身份

源点与源实体观测使用源图像采集时刻的世界坐标，当前点使用当前 RGB-D 的标定世界坐标。`T_world_camera_cv` 将 CV 相机坐标变换到相应世界坐标；相机 Z 是米制光学深度。所有外部位置及长度使用米。

源功能坐标由完整技能表示给出的 `T_world_body_at_image @ L_body_region` 组成。`source_row_ids` 是源观测表中 `region_mask=True` 的全部原始行号，`source_pixels` 保留这些行的原图像像素。`source_owner_points_world_m` 是源观测中 `entity_member=True` 的点集。源区域和源实体具有各自的显式语义。

当前几何表由显式 `owner_observation` 在完整当前深度上建立，原 `observation_entity_member` 保持独立 owner 观测的成员含义。每个相机的 `observation.object_mask` 在完整原测量像素上给出 reference 成员；这些测量经同一个 `observation_to_element` 得到 `target_element_ids` 与同序 `target_points_world_m`。因此，独立 owner mask 外的 reference 测量仍保留其原始身份。

`owner_evidence.reference_material_owner_id` 与 `owner_id` 明确相等时，该 view 的实际 reference 测量还具有 local material 的声明职责，来源保存在 `reference_material_source`。独立 owner 观测和这份局部材料观测分别保留各自的 observation/element ID；两类实际 observation ID 的唯一并集通过原表关联得到 `target_owner_element_ids` 和同序 `target_owner_points_world_m`。这表达声明 owner 的观测材料域，包含重叠来源的同一原元素一次。隐藏区域保持其观测状态。

`observation_ids`、`observation_element_indices`、`observation_view_indices` 和 `observation_pixels` 共同追溯 reference 的原始测量到目标元素。`element_observation_counts` 表示每个 reference 元素含有几份原测量；区域目标函数按一个元素计量一次几何质量。reference 和 owner 可以共享一个对象级观测 bank，也可以通过各自显式 mask 表达组件与较大 owner 的不同范围。

`matching_target_indices` 是 reference 目标数组局部行号，`matching_target_element_ids` 是对应原元素 ID。多个源行可以对应同一目标元素。工具模型以 `target_owner_element_ids` 精确关联这些 ID，保持 owner 点输入次序；owner 域之外的 reference 元素保留 source-row、原元素 ID 与 unsupported 状态。全部源行在粗配准、网络输出、最终对应和区域 frame 读出中保持原顺序。

## 3. 实际调用关系

```text
已加载 skill + 两份有序具名 views + 声明 observation_model
  → pipeline.run_region_transfer
  → evidence.prepare_evidence
      → dual_view.curved_geometry.build_target_surface（owner bank，本次构造一次）
      → dual_view.features.source_descriptors / prepare_view
      → response.query_views
  → pipeline.run_arrays
      → coarse.register_candidate_context
      → context.read_region_frame（粗配准摘要）
      → refinement.refine_identity_mapping
      → response.query_views（粗配准与细化位置的观测支持诊断）
      → context.read_region_frame / orientation_evidence
  → 完整 arrays、区域 F、关系证据、阶段状态及计时
```

给定原 `surface` 时，`prepare_evidence` 直接复用其观测—元素关联。这是固定输入数值复放接口；相机输入顺序须与 `surface['view_ids']` 相同。`run_arrays` 接受已准备数组和原生视角响应，运行相同粗配准及细化计算。

## 4. 公共输入和结果

`run_region_transfer(skill, views, *, observation_model, context=None, surface=None, sampling_radius_m=.03, coarse_steps=40, refinement_steps=200, seed=0)` 接收以下输入：

| 输入 | 实际生产者与内容 | 本组件消费用途 |
|---|---|---|
| `skill['source']` | `manipulation.skill.load_grounding_skill` 读出的观测、feature、原 crop、固定功能关系 | 源区域、源描述符、源实体关系和源功能坐标 |
| `views` | 公共观测准备产生的两份有序具名当前观测 | 当前几何元素、原生外观响应及测量支持 |
| `observation_model` | 当前 RGB-D 测量模型声明 | 光学深度误差和横向深度变化边界 |
| `context` | 公共入口给出的源、当前实体、前置状态及起始采集身份 | 按原内容归档为 `input_context` |
| `surface` | 可选的原几何元素表 | 复放已固定的元素身份和测量关联 |
| 原算法参数 | 原调用的粗配准步数、细化块数、随机种子和观测建图半径 | 对应算法的直接调用参数 |

每份 view 使用 `id`、`depth_m`、`K`、`T_world_camera_cv`、reference 的 `observation/feature/crop_xyxy`、owner 的 `owner_observation/owner_mask/owner_evidence` 和显式 `background_mask`。`owner_evidence` 的 `owner_id/scope/source` 说明独立实体观测身份、范围和来源；`reference_material_owner_id/reference_material_source` 说明 reference 局部材料所属实体的显式声明。`crop_xyxy` 是 reference feature 生产时采用的原图像裁剪，响应计算直接使用这份输入。

目标表构造使用库原 `ObservationModel` 和 `build_target_surface`，其几何 view 的 `observation` 指向 `owner_observation`。reference 的 `object_mask` 按原测量像素独立选行，两个域复用同一几何表和元素关联。原生响应查询使用 reference mask、crop 和 feature，实体关系使用上述声明 owner 材料的唯一元素点集。`prepare_evidence` 在描述符及几何构造前检查恰好两份具名且 ID 不同的 view；公共入口按其调用合同记录缺失状态。

`independent_owner_observation_ids/element_ids` 和 `local_material_observation_ids/element_ids` 分别记录两类来源；`owner_observation_ids/element_indices/pixels/view_indices` 追溯唯一材料测量到 owner 输入元素。`local_material_observation_pixels/view_indices` 由该规范域直接提供给观测场消费者。背景 admission 以这份实际测量身份排除已声明的 owner 局部材料，原 reference、独立 owner 和 background masks 保持各自原输入。当前接口不以坐标接近程度重新分配点的实体身份。

结果字典包含 `status`、`stage`、`arrays`、原生 `views`、`input_context`、`input_evidence`、`source_rows`、`target_elements`、`source_frame` 和 `grounding_stage_seconds`。完整区域计算还包含 `coarse`、`coarse_frame`、`refinement`、`mapped_observation_support`、`region_frame`、`context` 和 `refinement_relation_max_matrix_difference`。

`status='completed'` 表示区域计算结束。原生目标元素缺少其所需外观时输出 `status='missing'`、`reason='missing_observed_target_appearance'`，并保留全部输入数组及未知响应。网络数值失败输出 `status='failed'`、`stage='failed_neural_refinement'`，保留优化已记录状态、失败试值和具体非有限项。这些状态的职责是描述本组件的输入与数值结果。

## 5. 原生外观证据

`evidence.prepare_evidence` 从源区域完整行号调用原 `source_descriptors`，读取原 FP32 MARCO 描述符。每份当前 feature 调用原 `prepare_view`，按原 FP32 点积生成 `[Q,Hf,Wf]` 响应网格；Q 为源区域行数。

`response.query_view` 将每个世界点投影到原图像，记录其光学深度、正权重双线性深度像素、reference mask 支持、横向斜率支持和 feature 插值足迹。该原生查询接口的 `owner_mask` 字段由准备组件明确绑定为 reference mask；实体几何的 owner bank 单独进入目标表与实体关系计算。只有具有当前测量支持的位置取得响应。其余位置保留 NaN、`unknown=True` 和原接口原因，如 `outside_owner`、`missing_depth_measurement`、`depth_discontinuity` 或 `occluded`。

`response.query_views` 对实际支持该点的视角做 raw 响应均值；无支持视角的点保持未知。每个视角的响应、有效位、原因、原测量 ID、深度区间和 feature 角点/权重随结果保留。`query_arrays` 将这些结果按稳定字段展平为数值数组，供统一归档。

`sample_native_pixels` 和 `sample_native_world` 使用原 float64 双线性插值与 PyTorch 导数。可微函数描述已知有效插值片内部的响应随位置变化；离散观测支持由 `query_view` 计算。粗配准和细化输出位置的再次查询保留其诊断证据，数值优化仍使用原观测元素域。

## 6. 移动区域中心的刚体粗配准

`coarse.register_candidate_context` 将源点按源区域 RMS 长度归一化，目标使用相同长度单位和 raw argmax 对应的均值原点。外观代价为每个源行的 `(raw_row_max - raw) / descriptor_channels`。

初始对应包含 raw argmax 一组和源功能坐标表达的 24 个正八面体旋转组，共 25 组。每组调用 `fit_rigid_with_context` 初始化，然后执行默认 40 次固定对应的六变量 SciPy `least_squares` 与全目标域 row argmin 更新。移动中心的目标实体关系在同一目标函数中随归一化平移变化。

`fit_candidate_context` 使用零初始化的旋转/平移增量，旋转为 `Exp(w) R_start`。求解器保持 `trf`、线性损失、`jac='2-point'`、三项 `1e-11` 停止参数和 `max_nfev=100`。粗配准以最终外观、归一化平方距离和实体关系平方误差之和选出最低值种子。

`coarse_*` 数组保存每个种子、每个迭代的对应、映射、旋转、平移、中心、拟合前/后及赋值后能量、求解器参数、梯度、状态、消息和评估次数。报告给出选中种子、源 RMS、目标归一化原点、实体观测矩及求解器合同。

## 7. 残差 SIREN 细化

`refinement.prepare_refinement` 在全部源行上调用维护中的 OAReg barycentric graph，使用 `n_neighbors=min(30,N-1)` 和原 `reg=1e-3`。邻域重构权重原样保留。正长度源邻接边进入长度应变项；每个源身份及其重构行持续存在。

源局部神经输入是按源功能轴表达的源 RMS 归一化坐标，位移输出轴是粗配准旋转乘源功能轴。`Siren` 使用 3 输入、128 隐藏宽度、3 隐藏层、3 输出、omega 30、CPU float64。最后线性层置零，所以初始映射等于选中的粗配准映射。

`objective_components` 返回六项单位权重能量和其总和：原外观代价、源单位中的平方距离、局部线性重构范数均值、对数长度应变均值、局部方向均值及实体关系平方误差。每个默认 200 个外循环块依次执行固定对应/方向下的 L-BFGS、当前映射对应的精确 proper-SVD 关系旋转、全目标域对应更新。L-BFGS 保持原 `max_iter=20`、`max_eval=25`、`strong_wolfe`、学习率 1、梯度阈值 `1e-7`、变化阈值 `1e-9` 和历史长度 100。

输出包括每个完整状态、每个拟合块、每次 closure 试值、源图邻接和权重、正长度边、源单位、神经输入/输出轴、模型参数以及最终映射和对应。非有限目标或参数梯度通过 `NeuralRefinementFailure` 携带完整部分结果；报告区分最后完成映射状态与失败 line-search 试值所处的模型参数状态。

OAReg 已作为工作区维护依赖存在于 `OAReg/model/model.py` 和 `OAReg/utils2/LLR.py`。本组件与库内 MARCO 适配相同，直接文件加载这两份数值实现；`Siren` 与 `barycenter_kneighbors_graph` 的计算保持原实现。NumPy、SciPy、PyTorch 和 scikit-learn 来自现有 recovery 环境。

## 8. 功能坐标与工具关系

`context.read_region_frame` 对完整源/映射身份对求最小二乘相似变换。源区域点的相似缩放和平移传播原功能原点，单位功能轴只旋转。返回 `T_world_region`、相似矩阵、尺度、残差、协方差奇异值及数值旋转唯一性。

`orientation_evidence` 使用完整区域对及各自实体观测，计算归一化局部关系、实体偏移关系、其合成矩阵和 proper Wahba 旋转。区域尺度、功能坐标 F、工具目标 E 和工具运动锚点具有各自计算职责。

`context_grasp_proposal` 以源工具坐标中的米制锚点和关系旋转产生刚性工具 proposal。`rank_candidates` 在实际已绑定的受控工具 E 位姿上，按原生 quality、位置差平方/区域 RMS 平方和完整旋转 Frobenius 差平方/8 三项原权重求和稳定排序。这两项 API 供工具候选组件调用；区域 pipeline 输出 F 及完整映射供它们消费。

## 9. API 明细

| API | 前置数据、计算和输出 |
|---|---|
| `pipeline.run_region_transfer` | 已加载 skill、两份具名观测及观测模型；准备一次元素/响应，返回完整区域结果 |
| `pipeline.run_arrays` | 已准备证据数组、原生响应 views 与来源记录；执行原粗配准、细化、frame 和支持查询 |
| `evidence.prepare_evidence` | 源观测/feature、当前观测/feature、可选原 surface；返回 `arrays, prepared_views, context, metadata` |
| `evidence.query_arrays` | 完整原生 query 与字段前缀；展平原数组和逐视角测量证据 |
| `response.sample_native_pixels` | 原网格、连续像素及原 crop；返回可微 `[Q,N]` 响应 |
| `response.sample_native_world` | 预先限定支持的世界点、标定和原响应；投影后可微采样 |
| `response.project_world` | 世界点与标定；返回原像素和米制光学深度 |
| `response.query_view` | 世界点、单视角原响应/深度/owner/测量 ID 和显式观测模型；返回响应、支持与完整测量足迹 |
| `response.query_views` | 同一世界点域、全部原生 views、源行号及观测模型；返回按有效视角均值的 raw 响应和逐视角证据 |
| `coarse.owner_moments` | 当前可见实体点；返回观测均值和经验方差 |
| `coarse.context_at` | 候选区域中心、实体矩；返回有界偏移向量和对米制中心的导数 |
| `coarse.joint_residual` | 旋转/平移增量、固定对应和归一化数据；返回位置与移动中心关系残差 |
| `coarse.fit_candidate_context` | 固定对应和块起始 R/t；返回拟合 R/t 及 SciPy 原结果 |
| `coarse.objective_terms` | 完整对应和当前 R/t；返回三项代价及总和 |
| `coarse.register_candidate_context` | 完整源/目标域、raw 响应、源功能坐标及实体观测；返回 25 种子全记录与选中身份 |
| `refinement.prepare_refinement` | 源/目标域、外观、实体点和粗映射；返回固定单位、图、原数组与 float64 张量 |
| `refinement.objective_components` | 当前映射、固定对应/方向及 prepared；返回六项可微代价及总和 |
| `refinement.assign_full_target` | 当前归一化映射及 prepared；返回每个源行的全目标域 argmin |
| `refinement.relation_rotation` | 当前归一化映射及 prepared；返回关系旋转和其完整证据 |
| `refinement.refine_identity_mapping` | 完整身份、粗配准和原参数；返回全部状态/试值/参数及优化报告 |
| `refinement.NeuralRefinementFailure` | 非有限试值诊断；携带当次完整部分数组和报告 |
| `context.context_vector` | 可见实体和区域中心；返回有界实体—区域关系与经验 RMS |
| `context.fit_rigid_with_context` | 固定身份对及固定关系向量；返回最优 proper R/t 和数值唯一性 |
| `context.read_region_frame` | 全部源/映射身份对和原功能坐标；返回相似摘要及真实任务 F |
| `context.orientation_evidence` | 全部区域身份对及实体观测；返回完整朝向证据与关系旋转 |
| `context.context_grasp_proposal` | 完整关系证据与源工具刚性位姿；返回米制锚点保持的工具 proposal |
| `context.rank_candidates` | 实际 E 位姿、原 quality、完整 E prior 和区域 RMS；返回三项成本、稳定顺序及选中行 |

内部 `_center` 在相对一个原点后取均值，以保持常量点集的精确零残差。`_proper_rotation` 从配对协方差计算 proper-SVD 及数值唯一性；`_source_grasp_anchor` 将源区域中心表达为源工具米制锚点。它们分别被相应公开数值函数直接调用。

## 10. 当前验证依据

低资源 CPU 对照在 recovery Python 和 `PYTHONNOUSERSITE=1` 下运行。纯数组迁移前后粗配准的 25 项数组、神经细化的 63 项数组及逐视角原生响应证据一致；检查还覆盖相似变换下功能原点/单位轴、米制工具锚点、unknown 原因及原生插值位置导数。

固定原观测的对象级重放采用原单线程 BLAS/PyTorch 算术，162 项区域数组及实际工具模型的点、元素 ID 和 steering mask 全部 dtype/shape/tobytes 一致；新增 `target_owner_element_ids` 与原 `target_element_ids` 同序。组件反例在一次 surface 构造中保留 18 份 reference 测量、12 个 reference 元素和 30 个 owner 元素，原生支持仍指向 reference mask，工具 steering 通过精确元素 ID 关联。

独立 owner mask 漏掉部分 reference 的反例仍保留全部 18 份 reference 测量；原表独立 owner 证据为 20 元素，按明确同 owner 声明接入的局部材料形成 32 个唯一材料元素。两类来源和原 masks 均保留，12 个 reference-only 元素可按原 ID 成为工具模型输入；声明不同 owner 时，这些元素的 unsupported 状态保持。背景原每视角 10 个候选像素保留，按实际材料测量排除 6 个后，模型 admission 为 4 像素。几何坐标重合且元素 ID 不同的反例保留 unsupported 身份；单 view 与重复 view ID 在描述符计算前报告输入合同错误。完整模型与规划验证由集成入口统一调度并保存阶段记录。
