# Point2Pose + Track-On-R / SAM 3.1：四套最终工程 Profile

适配目标：MuJoCo 3.5.0 或 RealSense RGB-D；在线 / 离线；RTX 5070 Ti 16 GiB；小/中型机器人操作对象。此包依据 Point2Pose upstream main（2026-10-10 检索）、MuJoCo 3.5 的 API/源码设计。**仅完成配置结构检查、Python 编译与数据结构单元测试；未在你的设备、MuJoCo 场景或完整 Point2Pose 依赖上执行端到端运行。** 所有阈值均为起始配置，非已标定的最优值。

## 文件

| 文件 | 场景 | 匹配/位姿求解 | 优化 |
|---|---|---|---|
| `mujoco_online.yaml` | MuJoCo 在线仿真 RGB-D | `svd_residual_outlier` | 无全局/局部图优化 |
| `mujoco_offline.yaml` | MuJoCo 离线 RGB-D | `svd_cluster_ransac` + SDF refine | 全局 + 局部图 |
| `realsense_online.yaml` | RealSense 在线 RGB-D | `svd_residual_outlier` | 无全局/局部图优化 |
| `realsense_offline.yaml` | RealSense 离线 RGB-D | `svd_cluster_ransac` + SDF refine | 全局 + 局部图 |
| `rgbd_bridge.py` | 两种输入源 + `Frame` 桥 | `RGBDPacket`, `make_frame`, `MuJoCoRGBD`, `RealSenseRGBD` | 不装载 SAM |
| `run_offline.py` | 离线预处理序列的执行入口 | 直接调用 `ModularPipeline.step()` | 需 Point2Pose 全套依赖 |
| `import_point2pose_skill.py` | 已记录的 Point2Pose 估计转版本 2 技能包 | 复用 `source_region_frame`、`build_representation`、`save_grounding_skill` | 不运行模型 |
| `validate_profiles.py` | profile 和 RGB-D 数据契约自检 | 不需 GPU | 静态校验 |

### 主要约定

1. 四份 YAML 都有 `pipeline.params.use_segmenter: false`。**必须每帧传入 `Frame.mask`，不能直接运行原始 SAM2 点击交互的 RealSense 示例。** 原样 demo 会向 dummy segmenter 发送 prompt，接口不匹配。在线模式由你已有的 SAM 3.1 provider 实时生成 mask；离线模式单独运行 SAM 3.1 并落盘后读取。
2. `segmenter.type: dummy` 用于保留 Point2Pose 配置结构，并**不负责**提供 mask。首帧应有至少一个非空实例 mask，后续被遮挡可传空 mask，但实例索引必须稳定。
3. `tracker.type: trackon` + 本地 `track_on_r.pt` + `vit_backbone: dinov3_s_plus`。DINOv3 已下载到 `~/recovery/models/dinov3_s_plus`，需按你先前的本地模型加载方式修改 Track-On-R 的 DINOv3 adapter（或者让 Hugging Face 缓存可用）。
4. `sampler.type: super_point_balanced` 在 Point2Pose 中**只使用 SuperPoint 特征检测**；不额外构建 LightGlue matcher。SuperPoint 权重须已经缓存。
5. MuJoCo 的 `mujoco.params` 仅供 **你自己的 `MuJoCoRGBD` 接入层**读取，不是 Point2Pose 原生数据源；RealSense 的 `realsense.params.rs_serial` 是原生 demo 的序列号元数据，也需替换。四份 YAML 是 Point2Pose **计算管线**参数，不自动启动摄像头或 SAM。
6. 当前源码实际的全局图优化开关是 `pipeline.params.use_key_frame_graph`，不是旧示例 `use_graph_optimization`；`frame_reg_mode` 有效值是 `f2f` 或 `f2m`，不支持 `hybrid`。

## 一、安装与路径

- 四份 YAML 可作为独立配置读取；本地已验证的 Point2Pose profile 也可直接传给 runner。
- `rgbd_bridge.py`、`run_offline.py`、`validate_profiles.py` 保留在 FUNCTO 的 `config/` 下。runner 复用 FUNCTO 的时间线及刚体矩阵校验，从已有 Point2Pose 仓库根目录启动。
- 修改所有四个 YAML 的 `tracker.params.checkpoint_path`，当前默认 `/home/shelter/recovery/models/track_on_r.pt` 只是此前交谈中可能的路径；**按实际文件路径修正**。
- 运行：

```bash
PYTHONNOUSERSITE=1 /home/shelter/miniforge3/envs/recovery/bin/python /home/shelter/recovery/FUNCTO_public/config/validate_profiles.py
```

需要原生 Point2Pose、Track-On 源码、其 CUDA/MMCV 依赖、LightGlue SuperPoint、PyCUDA/TSDF 依赖。MuJoCo 3.5.0 和 RealSense SDK 依需要安装。

## 二、输入：统一的 `RGBDPacket` 和 Point2Pose `Frame`

- `rgb`: `[H,W,3] uint8`, **RGB 通道顺序**，不应 RGB/BGR 交换两次。
- `depth_m`: `[H,W] float32`，**米**，光轴正深度（Z depth），缺失置 0。
- `K`: `[3,3] float64`，与 RGB 和 depth **同分辨率/像素坐标系**；像素采样必须遵守当前相机内参和畸变校正的一致性。
- `mask`: `[N,H,W]` 二值实例 mask，`N=pipeline.params.max_num_obj`；通过 `make_frame` 变成 GPU 上的 `[N,1,H,W]` Bool Tensor。
- `timestamp_s`: 原始采集秒数；缺失时为 `None`，不按默认帧率或处理墙钟补造。
- `depth_factor=1.0`：数据在进入 Point2Pose **之前已经转换为米**。
- 多物体：按独立刚体固定实例索引，遮挡时给原索引空 mask，不能重排或把整个铰链柜体当单一刚体。

MuJoCo 原生 `Renderer.enable_depth_rendering()` 返回已经线性化的相机光轴深度（米），**不要再从非线性 Z-buffer 换算**，也不要重复上下翻转。相机坐标为右、上、后（视线方向 -Z），转换为 OpenCV 右、下、前：

```python
R_world_cv = R_world_mujoco @ np.diag([1.0, -1.0, -1.0])
```

`MuJoCoRGBD` 必须显式接收实际采集使用的 `K`；不在桥接层按图像尺寸或视场角重新推导主点。当前 collect 渲染采用整数像素中心，主点为 `((W-1)/2, (H-1)/2)`，因此误用 `(W/2,H/2)` 会引入半像素偏差。`K` 必须与当前 RGB、深度和 mask 的像素网格一致，Point2Pose 接口要求零 skew 的透视内参；不支持正交相机。

RealSense：使用 `rs.align(rs.stream.color)` 同步 depth 至彩色相机，并利用 `first_depth_sensor().get_depth_scale()`：

```python
depth_m = depth_raw.astype(np.float32) * depth_scale_m
```

`depth_raw` 是深度相机读取的原生像素值，`depth_scale_m` 是每单位对应的米数。**不要写死 /1000**。彩色相机如果仍有镜头畸变，需要保证颜色、深度和 mask 在同一个校正后的 pinhole 坐标系；不能只单独 undistort RGB 而不同时转换 depth/K/masks。

### 数据源示范（在线 MuJoCo）

```python
# 放入你已有的仿真循环，而不是复制成一个新的 step 循环
from omegaconf import OmegaConf
from point2pose.pipeline.modular_pipeline import ModularPipeline
from config.rgbd_bridge import MuJoCoRGBD, make_frame

cfg = OmegaConf.load('configs/custom/mujoco_online.yaml')
camera = MuJoCoRGBD(model, data, camera_name='tracking_camera', width=640, height=480,
                    K=capture_K)  # 当前渲染输出实际使用的 3×3 相机内参
pipe = ModularPipeline(cfg)
for frame_id in range(1000):
    # mj_step(model, data) 通常由你自己的仿真/控制器完成
    packet = camera.capture()  # rgb + depth + K + T_world_cam_cv
    masks = your_sam31_provider(packet.rgb)  # 用户接口，输出 [N,H,W] 二值 mask
    frame = make_frame(packet, masks, frame_id,
                       max_num_obj=cfg.pipeline.params.max_num_obj)
    pose_array = pipe.step(frame)
    # pose_array[i] 对应物体相对于起始相机坐标系的相对 Pose
camera.close()
```

在线 RealSense 将上例 `camera` 改为 `RealSenseRGBD(serial='你的序列号')`；`make_frame` 和 `ModularPipeline` **完全共用**。

### 相机运动与世界 Pose

Point2Pose 的 `obj.pose` 表示从首帧地图（首相机）到当前相机的物体变换（初始化为单位矩阵），而 `obj.init_pose` 还定义可视化/物体局部坐标。MuJoCo 尤其是腕部相机运动时，不要把 `obj.pose` 直接当世界位姿；应保留每帧 `packet.T_world_cam_cv` 并结合 object frame：

```python
T_world_object = packet.T_world_cam_cv @ (pipe.objects[i].pose @ pipe.objects[i].init_pose)
```

上述计算的是 Point2Pose 定义的**物体局部帧**在 MuJoCo 世界中的估计变换，不一定直接等于 MuJoCo MJCF `body` 的原点。若需和模拟 `body_xpos` / `body_xmat` 比较，须先标定固定的 object-local 到 body-local 坐标转换。对 RealSense 移动相机，如果没有外部相机位姿（机器人 FK、SLAM、标定），只能得到相对相机位姿，不能直接声称为世界位姿。

## 三、离线：预计算分割、单独运行 Point2Pose

建议**两个独立 GPU 进程**：先 SAM 3.1 整段视频分割，把 mask 与实例 ID 写入 CPU/磁盘，释放其显存；再 Track-On-R + SuperPoint + Pose。离线推理不必要把全部视频帧、全部 SAM 状态同时留在 RTX 5070 Ti 的显存。若 SAM 视频分割支持流式输出并将状态/视频 offload 到 CPU，启用它。

`run_offline.py` 接受以下统一的数据布局：

```text
sequence_001/
├── camera.json             # 实例 ID、原始帧号、采集时间、K、可选固定相机外参
├── rgb/
│   ├── 000000.png
│   ├── 000001.png
│   └── ...
├── depth/
│   ├── 000000.npy          # float32, meters (Z-depth), already RGB-aligned
│   ├── 000001.npy
│   └── ...
└── masks/
    └── obj_0/
        ├── 000000.png      # 0 / 255 mask; same resolution; stable ID
        ├── 000001.png
        └── ...
```

多物体扩展 `masks/obj_1/`、`obj_2/`，其顺序必须始终对应 `object_ids`。runner 根据该列表设置原生 `max_num_obj`，并使用 `use_segmenter: false` 接收已经计算好的 mask；跟踪、注册、图优化、SDF 参数全部保留所传 profile 的原值。

`camera.json` 内的 K 字段按下式定义：

```json
{
  "K": [[600.0, 0.0, 319.5], [0.0, 600.0, 239.5], [0.0, 0.0, 1.0]],
  "object_ids": ["rigid_part"],
  "frame_indices": [10, 17, 24],
  "timestamps_s": [1.2, 1.21, 1.43],
  "T_world_camera_cv": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
}
```

**上面的内参和单位外参均为格式演示，必须替换为实际采集标定；未知外参应省略。** `frame_indices` 与 `timestamps_s` 按 RGB PNG 文件名字典序对齐，保留原始值；原始帧号可以不连续。Point2Pose 内部单独使用从零连续编号，输出同时保存两套编号的映射。`timestamps_s` 缺失时只保留几何估计，不能据此编译需要真实时序的完整技能包。

支持 `intrinsics/<frame>.npy`（每帧 3×3 K）和 `camera_poses/<frame>.npy`（每帧 4×4 `T_world_camera_cv`）；存在时覆盖显式提供的固定相机参数。两处都没有外参的帧保持世界位姿缺失，绝不使用单位矩阵代替。RGB、深度和外参必须来自同一采集状态。

```bash
# 从 Point2Pose 项目根目录启动；仍然需要实际的 PyTorch/MMCV/GTSAM 等依赖
PYTHONNOUSERSITE=1 /home/shelter/miniforge3/envs/recovery/bin/python \
  /home/shelter/recovery/FUNCTO_public/config/run_offline.py \
  --config configs/local/trackon_sam31.yaml \
  --sequence /absolute/path/sequence_001 \
  --output /absolute/path/new_recording_directory
```

RealSense 离线序列同样使用离线入口和相应已有 profile。`--output` 必须是新目录，记录文件不覆盖既有实验：

- `estimated_poses_by_frame.npz` 保存 `object_ids`、`frame_indices`、`tracker_frame_ids`、`frame_names`、`timestamp_s`、逐帧 `K` / `T_world_camera_cv`，以及 `T_camera_B` / `T_world_B`。缺失时间或外参以 NaN 加独立可用性布尔数组明确表达。
- `T_initial_camera_B` 是首帧的 `obj.init_pose`，在整段序列冻结并逐帧核对；`T_camera_initial_map` 是原生 `obj.pose`。每帧按 `T_camera_B = T_camera_initial_map @ T_initial_camera_B`、`T_world_B = T_world_camera_cv @ T_camera_B` 组合。物体局部坐标 `B` 不是仿真 MJCF body 真值坐标，不能跳过两者之间的固定标定。
- `tracking_evidence.json` 保存原生注册残差、内点、有效点索引、`lost` / `lost_streak`、跳变拒绝信息与所用原生残差阈值。标签区分初始化、缺失当前前端结果、沿用旧估计、原生证据有警告、满足原生残差条件。最后一种标签不证明跟踪正确，也不等于执行成功。
- 运行异常会保存已处理帧、失败原始帧号和 `error.txt`；不把异常截断的结果标成完整序列。原生墙钟 `.txt` 日志单独保存在 `native-wall-time-poses/`，不能替代采集时间。

Point2Pose 在没有有效注册或拒绝跳变时仍可能返回 4×4 矩阵，静止物体有效注册时也可能返回相同矩阵。因此 runner 通过透传 `frontend.step` 的原始返回对象记录注册证据，不根据矩阵是否变化判断新观测，不改动任何注册阈值。

`import_point2pose_skill.import_point2pose_skill()` 正式消费上述记录，调用同一个 `build_representation()` 和 `save_grounding_skill()` 保存版本 2 包。`import_collect_skill.py` 则保留 collect 仿真 body/EEF 真值入口，二者的运动来源明确区分。

```python
from config.import_point2pose_skill import import_point2pose_skill

result = import_point2pose_skill(
    tracking_directory, new_bundle_directory,
    metadata=metadata, source=visual_source, kind="grasp",
    owner_bindings={"carrier": "rigid_part"},
    binding_source="已给定的刚体实例与 mask 顺序记录",
    eef={"frame_indices": original_frame_ids, "timestamps": acquisition_seconds,
         "T_world_eef": measured_eef_poses,
         "source": "同步机器人记录中的工具位姿",
         "world_alignment_source": "该工具观测与相机世界坐标的已有标定"},
)
```

<!-- BEGIN explicit source-region geometry contract. -->
`visual_source` 必须显式提供 `region_geometry="planar" | "surface"`，并使用技能包现有的 image、observation 点库、depth_m、crop_xyxy、frame_index、timestamp_s、owner_id、region_image_mask 等视觉字段。`planar` 使用原平面规则；`surface` 使用实际 RGB-D 点关联与观测图邻域。入口不按任务名选择模式，也不在拒绝后自动切换。collect 导入 CLI 对应必填 `--region-geometry planar|surface`。

禁止传入旧 `T_world_body_at_image`、`L_body_region` 或 `frame_construction`。适配器从对应原始帧的真实 `T_world_B` 重建物体位姿，调用现有 `source_region_frame()` 在同一个 Point2Pose 固定 `B` 下重新计算 `L`；不能沿用仿真 body 坐标下的旧 `L`。可选 `function_frame={T_world_function, frame_index, timestamp_s, source}` 显式提供同一源图像时刻的世界区域坐标；省略时从给定区域的测量点和相机坐标构造。迁移区域的采样种子不会改写已提供的功能坐标系 F。
<!-- END explicit geometry declaration and unchanged frame semantics. -->

`owner_bindings` 显式绑定 carrier/reference 实例，操作表示要求 reference 轨迹，且源区域必须属于对应 reference。EEF 若提供，必须已经与全部原始帧号、采集秒数、世界坐标对齐；这里不做插值或标定。`approach_time`、`interaction_time`、`release_time` 和 `events` 均是显式原时间，提供任何事件时必须同时提供 `event_source`。构造 `H_g` 或 `C_D` 需要 EEF；`H_m` 允许缺 EEF，并保留 `C_D` 不可用。没有 interaction_time 的短区间可保存为观测表示包，但交互位姿和接近方向仍明确缺失，不能称为完整技能。

整个原始区间逐帧核对 NPZ 与跟踪证据。失败记录、缺时间/world，或任一已绑定实例含 stale/missing/未知状态，返回 `status="missing_source_observations"` 与具体缺项，不写包、不跳帧。warning 候选可保留，完整逐帧原始证据随包保存；正常写入返回 `status="saved_observation_segment"`，该状态只表示保存成功，不表示追踪准确或执行成功。包来源分别标明 Point2Pose 估计运动、已给视觉区域和同步 EEF 记录。

<!-- BEGIN the grounding command preserves nominal paths and explicit boundaries. -->
本节 grounding 实现基线：`IMPLEMENTATION_COMMIT`。
`ground_skill.py` 从当前场景的 `robot_feedback.json` 读取 `eef_pose`（世界坐标下工具控制坐标的位姿），传入 `generate_reference`。抓取参考通过当前抓取功能坐标和源 `H_g` 组合完整位姿，实际末端入口独立写入边界声明；持物操作结合当前 `A`（物体局部坐标到工具坐标的固定变换）得到实际物体入口。`A_context` 已声明的工具或世界坐标名称若与反馈冲突，在模型调用前拒绝；不要求增加重复观测。

`generate_reference` 的 `object_path/eef_path` 是经 grounding 组合得到的名义路径，受控区间止于释放或明确持物终点。名义数组和完整连续表达式分别保存在 `nominal-reference.npz` 与 `nominal-reference-paths.json`；原帧、源时间、运动、交互及释放事件完整保留。`is_initial_relative_rotation_rpy` 只由 manipulation 的材料参考计算消费。

工具名义路径生成后，`ground_skill.py` 通过 `reference_phase` 将原边界传给 `BoundToolReference`。输入 `robot_feedback.json` 还须提供 `eef_twist=[vx,vy,vz,wx,wy,wz]`：它表示当前工具原点的世界坐标线速度和角速度，单位分别为 m/s、rad/s，来自当前具名 q/dq、固定官方机器人运动学及标定。本次参考时钟为原受控源区间的经过秒数，起点使用同一反馈的 `eef_pose`，交互目标使用原迁移结果。缺少速度时报告 `bind_reference` 失败，工具参考保持 `unbound`。

`reference.npz` 保存绑定后的 `eef_poses`、源时刻 `eef_timestamps`、经过秒数 `elapsed_s`、入口位姿/速度和原源帧/时间。`reference-paths.json` 保存源事件及 `BoundToolReference.to_record` 的完整输入表达式，通过 `from_record` 恢复；`reference-samples.npy` 为该对象的均匀时间采样。`nominal_reference_status`、`reference_binding_status`、`reference_status` 分别表示名义生成、实际绑定计算与最终工具参考状态。

网页在源/目标场景中保留实体轮廓、区域和采样点，并按源时刻对应展示源实际记录与入口绑定工具参考的位置、姿态轴、旋转矩阵和 RPY。每条轨迹只显示滑块所选时刻的一个位姿。当前执行轨迹缺失状态单独保留。后续规划以原名义路径、其请求时钟及具名 q/dq 调用同一 `BoundToolReference`，参考 IK、普通轨迹成本及保存曲线评分读取同一规划绑定接口。
<!-- END source-time geometry and physical-time planning reference have explicit consumers. -->

## 在线 SDF 取舍

在线两份 Profile **默认关闭 TSDF 重建**（`reconstructor.params.build_sdf_after_global_opt: false`），避免在 16GB GPU 上与 SAM 3.1、Track-On-R 竞争显存并引入关键帧耗时峰值。Point2Pose 的 `f2m` 稀疏点地图仍可正常做位姿注册；关闭的是 dense TSDF 重建，不是 f2m 特征点地图。上游初始化时可能出现 “Failed to integrate keyframe” 提示，因明确禁用 TSDF 而触发，不意味着跟踪失败。需要在线增量重建时可以设回 `true`，但应增加显存预警并监测体素尺寸。离线两份仍开启 SDF 与 SDF refine。

## 四、关键参数与调节策略

初始参数（根据 Point2Pose real-time config、HO3D experiment，针对 16GB VRAM 和 640×480 小/中型刚体做工程折中，不是经验证的最优值）：

| 参数 | MuJoCo 在线 | MuJoCo 离线 | RealSense 在线 | RealSense 离线 |
|---|---:|---:|---:|---:|
| 单次采样点数 | 20 | 32 | 20 | 30 |
| SuperPoint 最大候选点 | 512 | 1024 | 512 | 1024 |
| 最少可见点触发补样 | 12 | 18 | 12 | 18 |
| 注册类型 | 残差迭代 SVD | 多假设 RANSAC + SDF | 残差迭代 SVD | 多假设 RANSAC + SDF |
| 注册恢复残差（米） | .005 | .005 | .009 | .009 |
| TSDF voxel（米） | .007 | .004 | .007 | .005 |
| TSDF padding（米） | .06 | .06 | .07 | .07 |
| TSDF 重建 | 关闭 | 开启 | 关闭 | 开启 |
| Global graph | 关闭 | 开启 | 关闭 | 开启 |
| Local graph | 关闭 | 开启 | 关闭 | 开启 |

**实际调节顺序：**

1. 实测 `registration_stats.txt` 或已记录的内点数与 3D 残差，按**跟踪像素误差 + 深度噪声**而不是仅按深度传感器精度设置 `register.residual_thres` 与 `map_growth_max_mean_residual`。默认 MuJoCo 阈值虽较 RealSense 紧，但 Track-On-R 并非仿真真值点追踪器，阈值过小会长期处于 lost。
2. 小物体/工具的首选分辨率维持 Track-On-R `384×512`。不要在 16GB 显存紧张时一开始就牺牲输入分辨率；先运行 SAM 的离线单独 Pass，监测查询点数以及 TSDF 内存。
3. **柜门/大抽屉面板**：默认的 `sdf_filter_max_radius=0.40m` 和毫米级 TSDF voxel 只面向小/中型目标。长宽约 0.8–1.2m 的面板，按尺寸将 radius 调到 `.7`–`1.0`，同时 voxel 放粗到 `.008`–`.012`m；需要防止 SDF dense grid 爆内存。即使不追求重建，也不能把柜体+门板合成一个刚体。
4. 如 Pose 因抓取遮挡暂时失效，检查 Track-On-R 的重检测与 `pose_jump_guard_relax_when_lost`，并与 mask ID 一致性配合；不建议只放宽单个滤波阈值来掩盖错误匹配。
5. 模型选择是速度/精度取舍，不是仅靠 `offline` YAML 就能让 Track-On-R 自动执行双向 smoothing 或全片回溯。真正的离线**双向优化**需要另外做后处理；当前两份离线 Profile 仅启用仓库已有的多假设注册和关键帧图优化，不承诺全局最优解。

### 重要的源码限制

- `bbox_sdf_refine_enable` 四份配置均默认关闭（它提升 3D 框估计，不直接提升物体位姿），需要 3D bbox 再单独开启。
- `sdf_max_voxels` 和 `sdf_max_extent_m` 出现在某些官方 YAML 中，但目前的 `SDFBuilder` 并不读取这两个键，因此本包**不设置假的显存上限**。安全的做法是统计目标 AABB 和 voxels 预估、控制 `sdf_voxel_size`、`sdf_bounds_padding`、对象大小，并在 TSDFVolume 分配前增加硬性检查。
- MuJoCo 的 `Renderer` RGB + depth 需要在**同一仿真状态**连续渲染；不能先取 RGB、`mj_step` 后再取 depth。RGB/深度图不会自行绑定 mask，对象 mask 的来源由独立 SAM 3.1 实现决定。
- 上游原生 RealSense demo 以 SAM2 的 point-prompt 点击流程启动，**不能配合 `use_segmenter: false` 直接运行**；你需要使用 `rgbd_bridge.py` 接收 SAM 3.1 的 mask 再调用 Point2Pose。
- Point2Pose Pose 定义为独立刚体；含滑动关节的抽屉、带铰链的柜门属于多个刚体，每个刚体单独跟踪。可进一步在 Pose 后端加入铰链角/滑动距离/SE(2) 的运动学约束，当前官方 YAML 不包含此类约束模块。
- 如果你希望对比 MuJoCo GT 与视觉估计，请用 `T_world_cam_cv` 转到世界系，注意 object origin 固定偏置，勿将坐标系不一致的误差当模型误差。

## 五、上线前验收

1. `validate_profiles.py` 校验 4 份 YAML：组件字段、互斥选项、权重路径格式、在线/离线开关和 TSDF 初始体素大小。
2. 对 `rgbd_bridge.py` 的 `MuJoCoRGBD` 用一个已知位置 3D 点做单像素投影与深度回投误差测试；误差明显大于像素精度时不要开始 Pose 调参。
3. 用 Point2Pose 当前环境运行 30 帧 smoke test：固定物体、无遮挡 → 单体移动 → 单手遮挡 → 完全移出视角 → 恢复。检查 mask 顺序、追踪点、注册内点与 lost 转移。
4. 分别运行 MuJoCo 在线和离线同一段视频，记录位置/角度误差；RealSense 使用不同距离与手部遮挡验证。最后才调整 pose-jump guard 和点数。
5. 测 5070 Ti `torch.cuda.mem_get_info()`，额外结合 `nvidia-smi` 查看 PyCUDA 和 MuJoCo EGL 的**非 PyTorch 显存**。离线尽量使用 SAM 进程先退出再运行 Point2Pose。

## Source references

- Point2Pose default: https://github.com/tzuyuan/point-to-pose/blob/main/configs/realsense/default.yaml
- Point2Pose HO3D: https://github.com/tzuyuan/point-to-pose/blob/main/configs/ho3d_exp/eccv_final.yaml
- Point2Pose pipeline: https://github.com/tzuyuan/point-to-pose/blob/main/point2pose/pipeline/modular_pipeline.py
- Point2Pose Frame: https://github.com/tzuyuan/point-to-pose/blob/main/point2pose/data_types/frame.py
- MuJoCo renderer/camera: https://mujoco.readthedocs.io/en/stable/programming/visualization.html
- MuJoCo Python rendering: https://mujoco.readthedocs.io/en/stable/python.html
- MuJoCo camera projection helper: https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_support.c
