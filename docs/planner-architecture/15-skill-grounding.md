# 完整技能 grounding 与实例关系分派

实现位置为 `manipulation/skill_grounding.py`、`grounding_observations.py`、`instance_relation.py`、`grounding_records.py`，以及既有 `skill.py`、`selection.py`、`representation.py` 的共用接口。当前实现位于本轮工作树；提交前代码基线为 `96ee2dc`，实际实现提交在集成验证后回填。跨实例区域的数值定义见 [区域组件](../../transfer/cross_instance/ARCHITECTURE.md)，抓取与观测几何见 [抓取组件](16-cross-instance-grasp.md)，规划及选择见 [规划候选组件](14-grounding-planning-portfolio.md)。

## 研究对象、坐标与输入职责

研究对象是一份完整源技能及其在当前传感观测中的实体绑定。源技能为版本2的 `metadata/source/representation`：`metadata` 提供抽取时确定的实体角色、功能描述和检测名词；`source` 提供源图像、区域、逐点身份、描述符及图像时刻的功能坐标；`representation` 提供完整运动、工具与物体关系、源时钟、交互和释放事件。`representation.kind` 在读取时已确定为 grasp 或 manipulation，分别描述其运动关系。

当前输入为有序具名 RGB-D、标定、机器人采集反馈和规划入口反馈。相机深度是 CV 光学 z 深度，单位 m；`T_world_camera_cv` 将相机列向量映到观测世界。每个 view 的 `id` 保持实际测量身份，`timestamp_s` 保持其采集时钟。区域中 source row、原像素、target element 以及原生观测行各自属于明确的索引域，数值计算和归档保持其对应关系。

F 表示当前任务功能坐标，代码字段为 `T_world_function` 或区域结果中的 `T_world_region`。E 表示当前标定工具坐标，候选字段为 `T_world_tool`。B 在本组件专指完整抓取运动的世界放置锚点，字段为 `tool_reference_anchor_world`；它与源技能中的物体表示坐标分别命名。所有 T_X_Y 将 Y 坐标映到 X，平移单位为 m、旋转计算单位为 rad、源时间与经过时间单位为 s。

源区域、完整 H_g/H_m/C_D、功能坐标和事件承担技能表示职责。当前 RGB-D、标定及具名机器人反馈承担观测与运动学职责；声明 J/A、机构和接触关系承担模型职责。当前仿真实际运行环境的真实物体位姿、真实分割、接触和效果属于独立评价输入。规划和执行各自使用的允许输入范围见 [数据职责](08-external-files-and-data-roles.md)。

## 公共请求与结果

`ground_skill(request, detector=None, client=None, feature_extractor=None)` 是传感输入的公共入口；`ground_prepared_skill(request, prepared_observations)` 消费同一观测阶段已经完成的明确结果。后者用于已保存的传感输入和确定性数值重放，调用方须保留真实选择、实例关系及其来源。两个入口调用同一 `_run/_ground` 数值和参考链。

| 请求字段 | 数据含义与实际消费者 |
|---|---|
| `skill` | 已加载完整技能；区域、抓取及参考生成各读取自身所需字段 |
| `instruction` | 本次指令；固定语义请求读取该字符串 |
| `bindings` | 程序给定 binding ID 到源/当前实体记录的映射，每项含 `skill/views/source_owner_id/current_entity_id/owner_role`，以及本次参考材料所属的明确 `owner_id/owner_source` |
| `reference_binding_id` | 当前技能参考所使用的绑定；其他已具备输入的语义绑定可参加同次 VLM 请求 |
| `feedback` | 当前工具入口位姿、世界坐标 twist 和具名机器人反馈；连续参考绑定消费 |
| `capture_feedback` | 图像采集时刻的具名机器人反馈；当前图像自过滤和抓取手指状态消费 |
| `context` | 明确的源技能、前置状态和起始状态来源；`observation_source/capture_time_s` 给观测几何提供来源和采集时钟 |
| `same_state_source` | 多观测描述同一几何状态的已有声明；原同实例输入接口消费 |
| `region_parameters` | 跨实例区域的原 observation model、采样、40步粗配准、200步细化和种子等显式参数；可传同一已构造 surface 作重放 |
| `grasp_parameters` | 官方机器人 fixture、明确当前几何 owner ID、原模型权重、方向控制和场构造参数 |
| `relations` | J/A、各自来源、当前物体模型/标量坐标、非附着接触与受控区间声明 |
| `support_geometry` | 既有 observed-reference 所需的场、有限支撑面、余量和精度，交给原支撑参考接口 |
| `output_directory` | 观测生产阶段保存实际检测与描述符的位置；结果归档由独立 writer 消费 |

结果保存 `prepared_observations`、实际区域与抓取结果、全部 `reference_hypotheses`、阶段记录、缺失原因和异常位置。每个 hypothesis 有稳定 `id`、F/E/B、完整 `reference`、本次 source-clock 的 `bound_reference`、自身 `reference_status` 和 `reference_diagnostics`。多个候选的入口与事件诊断分别存放；共同结果说明候选集合状态。规划只消费每个候选的完整名义 reference 和明确任务数据。

## 共同观测与一次语义请求

`grounding_observations.load_scene(scene_directory, view_ids)` 是文件边界，读取显式 view 文件夹中的 RGB、深度与标定，以及机器人反馈和采集元数据。`prepare` 也可直接消费内存中的同结构 view。两者的数值输入相同。

`skill.prepare_interaction_candidates` 使用源技能中已保存的 interaction reference 名词逐 view 调用 SAM。object 引用使用 entity scope，part 引用使用 interaction_reference scope；部件、接触区域和实体的语义范围保持原定义。实际 detector 候选在语义请求前保存，编号及 mask 原样保留。`skill.observe_interaction_reference` 是仍有独立调用者的局部语义选择接口，它与联合入口共用候选构造。

`selection.prepare_selection_evidence` 将源视觉语义、原图、编号候选及已有 selected_owner 上下文组织为模型证据；`parse_selection_response` 将模型实际返回的 view-local ID 解析为原 mask 和状态。这两个函数为局部选择和 `instance_relation.select_bindings` 共用。联合入口将所有已具备输入的 binding 图像按全局图片编号组合，进行一次 `chat_json`。

联合响应为 `bindings[binding_id]`，每项包括 `views`、`same_entity/same_entity_reason` 和 `source_target_relation/relation_reason`。前者比较当前多个 view 中的唯一选择；后者比较源示教实体与当前目标，取值为 same_instance、same_category_different_instance、different_category 或 unresolved。candidate ID 只在所属 binding 和 view 内有效。选中多个可替代目标保留 alternatives；源—目标类别已知不会消除当前实体关联的未决状态。

唯一选择且满足本次实例/当前视角关联输入的观测，通过原 `prepare_selected_observation` 反投影并保留像素身份。MARCO 对该原观测 crop 计算一次描述符，采用 `np.savez(..., feature=...)` 无压缩保存原 FP32 数值。共同阶段输出观测和特征；所选区域算法构造自己实际消费的几何表示。

每个 view 的 `observation/crop_xyxy/feature` 属于所选 reference 域。`owner_observation/owner_mask` 属于独立的实体观测；已有 `selected_owner` 的原 mask 通过 `segmentation.prepare_mask_observation` 使用同一回投影数学单独建立点银行，不制造检测分数或候选编号。object scope 可共享同一已有银行；仅有部件时保留 `local_material` 范围及明确 owner 声明。`owner_evidence` 保存几何 owner、范围、来源和 `reference_material_owner_id/reference_material_source`，两个原 mask 各自保持。

区域组件只构造一份完整深度 surface，保留原像素到测量行及 element 的对应。reference 观测域独立按原 reference mask 取真实测量；owner 原观测域与已声明同 owner 的 local 材料分别保存身份，再按真实 element ID 组成唯一的 owner 模型输入。owner mask 漏分处的 reference 像素仍属于原 reference 域。AnyGrasp 将原匹配的 element ID 映入该 owner 输入域，未支持的 ID 单独归档。显式 `background_mask` 保持原样，背景模型只排除同一 canonical 材料身份所指的实际像素；局部材料之外没有背景声明时保持未知。语义归属在区域组件解释一次，场构造消费已经形成的测量身份。

## 区域策略与完整参考

`instance_relation.region_strategy` 直接将已确定的源—目标关系映到对应区域职责。same_instance 调用原 `manipulation.region_transfer.run_region_transfer`，其 planar/surface 候选、数值计算、状态和功能坐标构造继续使用原输入。same_category_different_instance 调用新 `transfer.cross_instance.pipeline.run_region_transfer`，完整数值链包括原生描述符响应、候选中心粗配准、身份保持细化和区域 F。不同类别和未决关系保存自身语义结果。

跨实例区域当前采用两份真实、独立、已关联的标定观测。公共入口在数值计算前对单 view、重复 ID 和缺失 owner 深度保留实际 missing 及已提供的观测；底层直接调用者使用同一双 ID 输入合同。源行数量、view ID、几何实体 ID 和对象类别作为数据传递。数值算法的参数不从技能名或实验日期导出。

抓取分支消费完整 H_g。同实例按原功能关系放置；跨实例 `run_cross_instance_grasp` 保留原生 AnyGrasp 机器人绑定与排名、每份真实 UNOPose 旋转和固定 TCP 原点的最终方向修正。每个最终工具姿态 E_i 建立 `B_i = E_i inv(H_g(t_interaction))`，再调用 `generate_reference(..., T_world_function=F, tool_reference_anchor=B_i)`。任务功能几何读取 F，名义工具路径读取完整 `B_i H_g(t)`。未给独立 anchor 的原抓取关系采用 F 放置同一 H_g。

manipulation 分支将完整 H_m/C_D、F、J/A、机构坐标、支撑或非附着接触声明传给已有 `generate_reference/generate_observed_reference`。J 将当前物体表示坐标映到源物体表示坐标，A 将该物体表示坐标映到工具。有限支撑的几何目标由对应观测场和支撑接口构造。reference owner 的源图与 carrier 的源对应分别属于各自输入；已有前置技能表示和明确 J/A 按原职责消费。

共享生成器保持原帧、原时间、所有事件、非节点交互插值、释放截断以及连续父路径/右乘操作。`reference_phase` 传递同一 `eef_path` 和边界。`BoundToolReference` 用实际工具入口、世界坐标 twist 和受控源区间经过秒数生成 grounding 预览；规划在自己的请求时钟上对原名义操作数重新绑定。

## 状态、归档与查看

`_validate_relations` 在模型调用前核对 J/A 来源及已有工具坐标声明。区域缺失、优化失败、参考缺失和入口绑定异常分别保留其阶段。空原生抓取池通过 `NoGraspCandidates` 保存实际网络点、region mask、seed 与控制量；某个 B 无效时保留该候选异常，其他候选继续保存自己的完整参考和状态。`completed` 表示该入口的计算结束，实际区域/参考/规划/物理效果由各自状态表达。

`grounding_records.save(result, request, output_directory)` 是被动结果消费者。共同数值保存在 `grounding-arrays.npz`，包括原索引、响应、模型输入、几何场和优化证据；各原生 view 另存其测量支持。`region.json` 和 `grasp.json` 分别记录区域、抓取及场查询职责。`hypotheses/<id>/` 对一个或多个候选均采用同一布局，保存 nominal-reference、完整连续路径、source-clock bound reference、原边界和 HTML。`bound_reference.to_record()` 通过 `plain` 递归归档，名义 PosePath 同样保存完整父操作数。

本地 HTML 使用真实源/当前图像：源区域及采样点、当前迁移点和对应观测区域、功能坐标以及完整源/参考运动均直接叠在场景中。每条轨迹在进度条当前时刻显示一个位姿及其坐标轴，完整旋转数值、坐标和时钟来源可查阅。物理执行轨迹由执行记录提供；仅 grounding 的产物标明物理轨迹缺失。多个完整候选分别链接，writer 不承担选择职责。

`config/ground_skill.py` 只加载 skill、明确 view 文件、关系与模型参数，调用公共 `ground_skill`，再调用 writer。正式规划消费者调用 `planning_portfolio.plan_grounding_hypotheses`，每候选独立构造问题并求解，按共同固定标定评分交付原曲线。完整接口签名和内部定义由 [实现索引](07-implementation-index.md) 列出。

## 依赖与实际验证范围

组件复用 NumPy、Pillow、已有 SAM/MARCO/Gemini workers、既有区域及参考模块。跨实例数值使用已存在的 SciPy、PyTorch 与 OAReg；抓取使用已有 AnyGrasp/UNOPose 及固定官方机器人模型。学习模型在独立进程中串行执行；规划候选的进程生命周期和资源总量由 portfolio 组件负责。

当前合同测试覆盖一次联合请求、绑定/view 局部编号、两个身份关系独立性、替代目标与未决状态、薄文件入口的失败证据、F 与 B 不相同时的完整连续参考、非节点事件和释放、实际入口 twist，以及连续归档恢复。红杯重放属于实验验收记录，其感知输入、模型与规划结果逐项来源及字节比较在项目外 validation 目录保存。
