# 仿真场景组件

加载与准备接口的代码基线为 `9041422`，对应 `scene/__init__.py`、
`scene/prepare.py`、`transfer/scenes.py::restore_named_state`。
本次更新改变 `drawer_original` 的场景数据和观测文件；上述接口代码保持该基线。
场景数据与本文一同提交，代码基线与文档、场景数据提交分别记录。

## 职责与数据边界

本组件保存可直接恢复的 MuJoCo 仿真环境。每个目录是一份显式选择的场景实例，
包含世界结构、初始积分状态、机器人反馈和已标定的相机观测。
场景选择通过目录名表达，加载器对所有实例执行相同操作。

`scene.xml` 与 `initial-state.npz` 用于仿真世界构造、状态恢复和传感渲染。
RGB-D、标定和机器人反馈是感知及后续任务计算的观测入口。
`seg_geom_id.npy` 是仿真渲染器产生的真实几何 ID，只供结果分析；对象观测掩码由感知步骤建立。
仿真实际运行环境中产生的物体位姿、分割、接触和效果不进入 grounding、参考生成或规划决策。
任务实体的规划几何由现场 RGB-D 建立的 TSDF 及观测支持域表达。
固定机器人及固定末端配置采用官方本体模型：形状和组件固定关系可跨任务复用，
模型位姿由标定、关节反馈和运动学计算。

## 文件与坐标语义

| 文件 | 内容与作用 |
|---|---|
| `scene.xml` | MuJoCo 世界、机器人、物体、关节及相机定义；相对资源路径解析到共享 `scene/assets` |
| `initial-state.npz` | `state_spec` 为 `mjSTATE_INTEGRATION`，`integration_state` 为本 XML 对应的完整积分状态向量 |
| `robot_feedback.json` | 命名机器人关节位置、可用的速度通道、工具坐标系和世界系下末端位姿；本次重建同时保存原生关节位置与速度 |
| `<view>/camera.json` | 固定渲染相机的位置、OpenGL 相机四元数及透视视锥参数 |
| `<view>/calibration.json` | 内参 `K`、CV 相机到世界的变换 `T_world_camera_cv`、观测时间和实际渲染设备 |
| `<view>/rgb.png` | 固定视角的真实仿真 RGB 图像 |
| `<view>/depth_m.npy` | 与 RGB 对齐的米制深度 |
| `<view>/seg_geom_id.npy` | 同次渲染的几何 ID，用于仿真结果分析 |
| `layout.json` | 派生场景的显式构造参数、位移、状态保留验证和各相机渲染记录 |
| `index.html` | 有对照记录的场景提供原图与新图的本地比较入口，图片链接打开原始分辨率 |

空间单位为米，世界系为场景的共同参考系。位姿矩阵采用齐次列向量约定。
`T_world_camera_cv` 把 CV 相机坐标变换到世界坐标；相机配置中的 OpenGL
四元数属于渲染接口，二者通过 `collect.camera.FixedCamera` 的标定实现衔接。
世界 Z 轴向上；既有 `top` 相机下，图像下方对应世界 −Y，机器人基座位于这一侧。

## API 与实际调用关系

`scene_directory(case)` 接收一个直接子目录名，返回该实例的绝对目录，前置数据是
目录中的 `scene.xml`。它不搜索实验记录，也不选择另一场景。

`load_scene(case)` 首先调用 `scene_directory`，再编译 XML 创建 `MjModel` 和
`MjData`，检查积分状态类型及向量维度，调用 `mj_setState` 和 `mj_forward`，
最终返回模型与已恢复的数据。此过程不推进物理时间。

`prepare_scene(source_case, case, *, keep_body, remove_body)` 是现有的单家具居中准备接口：
读取源实例，删除指定世界根节点，把保留的固定家具根节点 XY 放到桌面中心，
生成一个新的直接子目录。它通过 `restore_named_state` 重建模型维度变化后的状态，
再用 `FixedCamera` 保存 `view_a`、`view_b`、`top` 的全部观测。
此接口要求目标目录不存在，其操作语义是居中；当前 `drawer_original` 的相对位移
由该实例 `layout.json` 中的显式构造数据定义。

`transfer.scenes.restore_named_state(model, source_model, source_data)` 在模型结构改变后，
按关节名和类型复制仍存在的 q、dq，按执行器名复制控制输入，按 body 名复制 mocap
位姿，并保留仿真时间。调用者随后执行 `mj_forward`，为新模型重新保存完整积分状态，
避免把不同模型的状态数组按下标直接拼接。

`collect.camera.FixedCamera(model, data, width, height, camera_config=...)`
读取保存的相机配置；`capture(data)` 从同一状态产生 RGB、深度和几何 ID，
`calibration` 给出相机标定和实际渲染信息，`close()` 释放渲染资源。
这些调用依赖已有 recovery 环境中的 MuJoCo、NumPy、Pillow 和 OpenGL。

## 当前 drawer_original 的构造定义

源实例为 `original`。删除 `wooden_shelf_1_main`、`white_yellow_mug_1_main`，
以及世界根节点上的厨房橱柜、炉灶、锅具和冰箱背景几何；删除这些元素独占的 XML
资源声明。共享资源文件继续服务其他场景。

平移对象为 `white_cabinet_1_main` 及其完整子树。纵深定义为该子树在源初始状态下，
已有原生 box 组件沿世界 Y 轴的整体外尺寸，包含抽屉及把手。
对于旋转矩阵 R、半尺寸 h、单位测量轴 a，一个 box 的投影半长为
`sum(abs(a @ R) * h)`；全部组件投影区间的并集外界给出纵深。
本实例纵深为 `0.2210142943833166 m`，位移为
`[0, -0.1105071471916583, 0] m`，固定根位置由
`[0.0379608, 0.281288, 0.905]` 变为
`[0.0379608, 0.1707808528083417, 0.905] m`。
该尺寸计算只承担仿真布局构造中的移动距离定义。

机器人、block 的位姿以及保留关节的原始位置、速度由源状态恢复。
三个相机沿用源相机参数，从同一 `1.0000000000000007 s` 状态重新渲染。
新场景的状态维度、几何 ID、RGB-D 和标定共同属于新编译模型；
`layout.json` 保存具体构造及验证结果。
