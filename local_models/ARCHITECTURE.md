# 本地感知与抓取模型接口

代码基线：FUNCTO `c568b75`。本文描述该实现中的模型调用、AnyGrasp/UNOPose和数组worker接口；当前文档提交仅登记实现引用。模型调用方使用CPU图像、数组或文件序列；每个GPU模型在独立子进程中加载并执行，完成CPU结果物化后退出。

## 责任与调用关系

`runtime.py` 管理模型生命周期、离线网络边界和已有资源预算。`sam31.py` 提供原尺寸候选 mask，`marco.py` 提供原生空间特征与对应解码，`tapip3d.py` 提供固定材料点身份的 RGB-D 时序跟踪。`gemini.py` 使用已有 s3e gateway 请求语义选择，网络协议与认证由已有 gateway 客户端维护。

`anygrasp.py` 从官方固定机器人末端执行器的模型声明建立抓取坐标绑定、前导指尖平面与孔径关系，再将当前现场观测 owner 点及区域成员 mask 交给原 SDK。CPU 调用方 `manipulation.cross_instance_grasp.generate_and_rank` 负责实际机器人候选绑定与完整原评分。`unopose.py` 对一个源观测与有序当前观测逐对准备输入，一次批量预测保存所有行；世界坐标组成在 `cross_instance_grasp.transport_orientation` 完成。

现有 `foundationpose.py` 保留显式 metric model 的位姿比较/跟踪 API，其 `models` 参数明确包含模型路径和尺度。当前跨实例抓取调用图的机器人形状由官方固定机器人配置提供，现场任务实体由 RGB-D 输入表达。

## 三种模型运输 API

| API | 输入 | 输出与实际生命周期 |
| --- | --- | --- |
| `run_worker(kind, images, options)` | `sam31/marco`、PIL 图像序列和对应参数。 | 将原 RGB 像素写入临时 PNG；通过公共串行 worker 返回 CPU 数组字典。 |
| `run_file_worker(kind, paths, options)` | `foundationpose/tapip3d`、明确已存在的视频图像路径及深度/标定/初始 mask 参数。 | 使用原文件名传递序列；worker 输出 NPZ 在子进程退出后以 `allow_pickle=False` 读取。 |
| `run_array_worker(kind, arrays, options)` | `anygrasp/unopose`、具名 NumPy 数组和标量/字符串参数。 | 通过未压缩输入 NPZ 保留 dtype、形状、身份行和 mask；与图像 API 使用同一个锁、环境和结果边界。 |

`_run_file_worker` 串行持有调用进程中的 worker 锁；`_run_file_worker_locked` 物化请求、继承现有离线环境、启动 `python -m local_models.runtime` 并等待完整退出。FoundationPose 的初始 mask 保持为独立 typed NPY。`main` 根据明确 `kind` 选择对应 `worker_main`，模型导入发生在各自分派之后。

worker 使用 `PYTHONNOUSERSITE=1`、本地已有权重、离线模型环境和受限 CPU 线程。`_block_network` 保持本地模型的网络连接边界。`_guard_memory` 共同计算 worker 与调用进程 RSS；现有上限为 12 GiB，总可用内存条件保留；Torch allocator 的 worker 上限为 8 GiB。`_reclaim_cpu_memory` 在已有内存检查处释放不可达对象及空闲 glibc arena。上述模型预算与现场 TSDF 的独立全局资源预算分别按其真实计算生效。

## AnyGrasp API 与数据语义

| API | 计算和输出 | 调用方 |
| --- | --- | --- |
| `AnyGraspBackend(checkpoint_path=..., gripper_geometry=..., tool_geometry=...)` | 从同一官方末端执行器计算 `T_tool_grasp`、开闭 pad gap、原前导指尖平面及 float32 可表示的最大合法孔径；保存原模型配置。 | 完整跨实例抓取入口。 |
| `AnyGraspBackend.predict(points_camera, controls, *, seed)` | 输入为 float32 相机点、同序区域 mask、相机接近向量和角域；返回原生完整 17 列行，并更新实际体素输入、区域 mask、采样 seed 与设备记录。 | 原 native candidate 生成。 |
| `AnyGraspBackend.joint_positions_for_widths(widths)` | 按官方开闭配置的仿射孔径关系，将原生预测孔径映射为明确顺序的两个滑移坐标。 | 原生候选的机器人绑定。 |
| `anygrasp.worker_main(arrays, output, *, checkpoint_path, max_gripper_width, gripper_height, seed, approach_thresh)` | 加载原 SDK，设置原随机种子，调用一次 `get_grasp`；forward hooks 记录网络实际消费的体素点/区域 mask 及 `stage2_seed_inds`；返回原始 native rows。 | runtime 的 `anygrasp` 分派。 |

AnyGrasp 的 17 列为原生 quality、孔径、手高、插入深度、9 个旋转矩阵元素、3 个中心坐标及 object identity。该输出完整保留，精确行去重与双标签机器人绑定发生在 CPU 调用方。区域 steering 和接近 steering 使用当前实际观测及源技能关系。官方机器人和安装手形固定，其 shape、固定关系和官方孔径配置可持续复用；任务实体的 shape 由实际输入点组成。

worker 输出包含 `native_rows/model_forward_observed/prepared_points_camera_m/model_region_mask/model_seed_indices/device`。无 forward 时使用明确的 `model_forward_observed=False` 和空数组运输，其调用方恢复为缺失证据；完成 forward 后的空候选仍保留实际采样输入。

## UNOPose API 与数据语义

| API | 计算和输出 | 调用方 |
| --- | --- | --- |
| `predict_batch(reference, queries, *, seed=1, checkpoint_path=None)` | 每个 observation 包含 `rgb/depth/intrinsics/mask`；保留各自原图尺寸；按 query 顺序运输实际 pair。 | 跨实例完整抓取入口。 |
| `unopose.worker_main(arrays, output, *, query_count, seed, checkpoint_path=None)` | 调用维护中的原 `prepare_pair`，逐 pair 使用相同明确 seed；按第 0 维拼接，一次模型前向返回每行原始 `pred_R/pred_t/pred_pose_score`。 | runtime 的 `unopose` 分派。 |

原始位姿表示 reference camera→query camera。prepare_pair 保留原 5,000 个源点和 2,048 个当前点、原 RGB 裁剪与 pixel choose 规则。输出 `model_input_rgb/pts/rgb_choose/tem1_rgb/tem1_pts/tem1_choose` 保留实际 CPU batch 全部数组；`batch_size/model_forward_calls` 与各阶段时间记录来自该次调用。query 数量由实际有序视角输入决定，模型不复制视角填充 batch。

当前抓取消费者只用预测旋转、两侧相机标定和完整源工具方向组成当前方向。原始平移和模型分数保存为模型证据。最终方向改变后的抓取质量和完整轨迹的求解结果由对应后续计算记录。

## 既有模型 API

| 模块/API | 输入、实际计算和输出 |
| --- | --- |
| `sam31.detect(image, labels)` | 对实际原 RGB 图的明确文本概念做本地候选检测，返回每个非空实例的 label、score、xyxy box 和原尺寸 Boolean mask。`_validate_labels` 去除相同文本的重复请求；`worker_main` 负责原模型前向。 |
| `marco.extract_pair(image1, image2)`、`extract_image(image)` | 对两个或一个真实输入图分别返回原模型 FP32 `[1,1024,240,240]` 特征；单图接口复用同一原图预处理及 backbone。 |
| `marco.match_features(pair, pixels, source_size, target_mask)` | 复用原 MARCO dot-product response、resize rounding 和 masked decode。`resize_scales/allowed_grid/masked_decode` 分别维护像素比例、允许网格与原 Gaussian 解码；`_geometry/_preprocessing/preprocess_image` 读取并复用已有第三方原函数。 |
| `tapip3d.track(...)` | 输入同步 RGB-D 路径、K、米制深度比例、初始实体 mask 及每帧 `camera_from_world` 或显式静态相机声明；输出固定点身份的世界轨迹与可见性。`camera_extrinsics` 维护坐标声明，`sample_queries` 在实际首帧深度中选择身份；`_read_frame/_outputs/worker_main` 负责原序列运输与输出。 |
| `foundationpose.track(...)` | 输入同步 RGB-D、初始 mask 和显式模型/尺度列表，返回原 model→CV camera 位姿及 rendered-depth 对照证据。`_inputs/_pose/_outputs` 维护模型和坐标语义；`_load_frame/_load_mesh/_geometry/_numpy/_centered_pose/_track_sequence/worker_main` 分别负责输入、原模型、深度比较与时序调用。 |
| `Gemini.chat(...)`、`chat_json(...)` | 调用已有 s3e Gemini backend 对文本和已标注图像请求一次生成；JSON 接口返回一个明确对象，原请求或解析错误保留。gateway 生命周期由 backend close 完成。 |

跨实例抓取模块的详细坐标、官方模型边界、原质量评分和观测场接口见 [跨实例抓取架构](../docs/planner-architecture/16-cross-instance-grasp.md)。
