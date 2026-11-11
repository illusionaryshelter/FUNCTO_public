# 物体对应、几何目标与完整参考迁移

代码基线：`dbb6c1d`。实现为 [reference_transfer.py](../../../manipulation/reference_transfer.py)，连续路径运算见 [representation](representation.md)，观测几何见 [observed_support](observed_support.md)。

## 研究对象与坐标

组件将源技能静态任务关系、完整物体运动、预测的物体对应、目标功能区域、现场RGB-D占据几何及机器人入口组成工具参考。D为源物体固定坐标，C为原TSDF固定坐标，E为标定TCP，F为目标功能区域，W为当前标定世界。`T_X_Y` 将Y中列向量映到X；位置以m计，旋转以rad计，时间沿用源记录秒数。A=T_E_C为事件锚定的声明附着，J=T_D_C为预测的物体坐标对应。K=T_C_G=[I,c_C]定义占据体积中心坐标G。

源H_m、C_D、功能坐标、材料点、交互与释放事件属于完整技能表示。现场场与支撑面来自RGB-D；工具入口来自机器人反馈。UNOPose相机间预测是允许观测上的模型输出。当前仿真物体真值由独立评价组件持有。

## API与实际计算

| API | 前置输入、计算及输出 |
|---|---|
| `predicted_body_correspondence(...)` | 接收Cr→Cq的预测、两相机标定、源技能图像物体位姿及原场坐标，返回M=T_C_D；预测旋转经现有SciPy矩阵转换表示于SO(3) |
| `support_region_frame(functional_frame,surface)` | 保留区域原点，将区域第一轴投影到观测平面，第二轴由法向叉乘第一轴得到，输出同一支撑法向下的功能坐标 |
| `observed_manipulation_targets(...)` | 从源交互静态关系、初态和释放时刻关系选取原RPY基准，估计原场占据体积中心；分别计算交互/释放目标旋转和有限承托几何，返回F、K、旋转选择证据与两个目标 |
| `generate_observed_reference(...)` | 先调用目标构造，再调用原`generate_reference`迁移完整H_m/C_D；在释放几何目标锚定物体路径，按A反解工具路径，返回原连续路径和边界接口 |

相机坐标分别为源Cr、当前Cq。源世界记为Ws，当前世界记为W，原场坐标为C。整物体对应为

`M=inv(T_W_C) T_W_Cq T_Cq_Cr inv(T_Ws_Cr) T_Ws_D`，`J=inv(M)`。

源物体图像位姿和源相机标定必须属于同一源世界、同一图像时刻。当前相机标定与C属于当前观测。局部抓取区域配准继续表达区域对应；整物体对应由本API的显式预测输入确定。

设H*=H_m(t_interaction)，H0=H_m(t0)。原材料坐标在当前场中的固定关系为`L_C_M=inv(J) inv(H*)`。功能坐标中的名义初态为`F H0 inv(H*)`，机器人模型给出的当前材料初态为`E A L_C_M`。两者进入原 `_initial_rotation_offset`：按源声明的布尔roll/pitch/yaw选择初始分量，得到功能坐标中的恒定旋转Delta。事件目标旋转为`R_W_C=R_W_F Delta R_H_m(event) R_J`。

`place_observed_on`以此旋转和区域原点为中心平面位置参考，使用原占据集合决定最低法向高度及有限边界内的平移。交互和释放目标各自保存完整源静态关系、占据中心、原方向区间、见证及数值精度。完整物体路径构造发生在这些目标计算完成之后。

原连续物体路径B(s)在释放目标B_goal处作`B_new(s)=B(s)inv(B(1))B_goal`；工具路径为`E_new(s)=B_new(s)inv(A)`。交互和释放的独立几何目标进入对应边界声明。`reference_phase`继续读取工具路径和工具边界；`BoundToolReference`以请求实际时钟、测得TCP入口位姿及速度建立连续连接。任务事件可用原`PoseEventRequirement`的固定变换A K表达中心要求，求解器继续计算原通用接口。

## 输出与依赖

返回记录保留完整源帧、时间、H_m、C_D、事件、原连续路径、释放右变换、当前几何目标、工具/物体路径及两类边界。源路径经`PosePath`逐组件采样，旋转选择沿用`RotationOffsetPath`。所有数值依赖均为既有NumPy、SciPy及本库观测查询；有限区域投影复用既有SLSQP平移求解器。

实际调用为`continue_held_motion.build_reference → generate_observed_reference → observed_manipulation_targets → estimate_occupied_volume_center/place_observed_on`，目标确定后调用`generate_reference`及连续右组合。`continue_held_motion.build_problem`把结果组成原任务接口；`update_held_reference_html.update`仅采样、绑定和显示同一参考。
