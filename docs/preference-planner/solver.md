# 联合计算的数值求解适配器

## 代码与研究对象

代码基线：`1323d81`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

令Z为终止等式消元后的具名关节系数，展平顺序为控制行优先、关节列其次。`JointComputation` 将Z展开为同一完整三次曲线C，产生物理时间上的任务值与显式位姿先验目标。适配器提交的标量不等式统一使用 `g(Z) <= 0`；SciPy SLSQP接口中的不等式采用相反符号。原连续关节条件的极值计算仍由 `CubicJointDomain` 负责。

## 调用与求值

`solve_trajectory(computation, initial, *, iterations, check_request, pending_geometry_bytes, report_out)` 由 `PreferencePlanner.optimize` 调用。前置数据为已经固定的任务行、见证几何行、物理时钟、关节域、终止映射和原资源上限。`initial` 为减少后的Z；返回 `ConstrainedSolution.controls` 仍为Z，由调用者恢复C并进入原完整资格检查。

首次 `computation.task(..., solver_rows_out=...)` 同时产生完整任务值与声明选择。原入口已固定的等式按原索引处理；原不可改变的任务行若违反其条件，函数直接返回带具体行值的失败记录。一次SciPy调用中的行维度与顺序固定。新增实体见证只能在该调用返回后由原完整实体查询产生。

内部 `evaluate(kind, x)` 以同一展平Z为缓存键，分别调用目标、任务和物理条件。目标值为运动质量、任务过程偏好、显式功能位姿先验及原观测几何方向项之和；Torch对其标量做一阶反传。任务值由固定选择后的原声明行与几何必要行组成。连续关节值与解析Jacobian直接来自减少坐标下的物理域接口。所有请求检查、计数、耗时与中断保留原调用关系。

目标采用原固定正数缩放 `max(1, abs(f0), ||grad(f0)||_inf)`，随后整个调用使用同一缩放。正缩放保留精确极小点；有限迭代和SLSQP停止条件仍受数值尺度影响。`scipy.optimize.minimize(..., method='SLSQP', jac=True)` 负责数值更新，迭代额度、停止精度和原约束单位均沿用原实现。

## 批量Jacobian

`task_jacobian(values, controls, *, check_request, admit_work)` 接收当前固定任务向量和Z张量。设条件数为m、变量数为n，Torch的 `autograd.grad(..., grad_outputs=I_m, is_grads_batched=True)` 一次计算m个向量—Jacobian乘积，返回m×n矩阵。常量行通过显式零依赖获得零导数。原Pinocchio桥已采用保留批次维度的张量收缩，原模型和FK解析导数直接复用。

该操作只求一阶导数。Coal最近特征与观测场三线性导数由几何组件提供；这些标量及其方向的含义见 [几何组件](geometry.md)。批量计算与逐行计算的等价性、几何导数与有限差分的一致性是两个独立验证对象。所用现成接口见 [Torch官方文档](https://docs.pytorch.org/docs/2.14/generated/torch.autograd.grad.html)。

`batch_derivative_storage_bytes(*, rows, variables, samples, frames, joints, geometry_workspace)` 根据实际数组维度返回反向工作区字典。输出单位矩阵、返回Jacobian、Pinocchio反向数组、位姿与曲线反向数组、实体及有限许可变换反向数组分别计入。`rows=0` 时，标量目标仍需要一次Pinocchio反传，输出基与任务Jacobian大小为零。

首次内存准入把原SLSQP工作区、原物理采样行、原场景及几何前向工作区与完整反向字典相加。原几何工作区中的单次额外Pinocchio反向项已包含在完整反向字典中，因而从前者扣除一次。每次反向开始前，当前前向图属于已测驻留内存，新增完整反向数组再次交给原 `require_host_memory`；整个调用使用同一请求的内存上限。

## 记录与结果解释

`report_out` 由调用者持有。`latest_trial`、`latest_complete_trial`、`last_callback_iterate` 分别记录最近试探、该控制量已完成全部必要求值的试探、SciPy最后一次迭代回调；每份记录保留自身控制量和已完成计算。原异常继续向调用者传播，已有记录用于导出诊断曲线。数值停止、试探记录与完整轨迹资格分别保存。

规划后端对同一保存曲线进行任务、连续关节、实体与连续时间检查，决定正式交付；该适配器不消费仿真效果或改变执行时钟。求解器返回状态不能代替该资格检查。阶段运行和数值审查记录位于项目外 sim-version-plans。

