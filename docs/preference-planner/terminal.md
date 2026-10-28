# 源时钟终止区间与三次曲线消元

代码基线：`1323d81`。本文描述 `manipulation/preference_planner/` 的当前实现；原规划组件基线为 `d3100d0`。

本组件由 `termination_clock.py` 和 `terminal.py` 构成。输入为已经绑定的源参考时间映射、归一化物理时间、原三次曲线仿射映射、测得入口关节位置/速度，以及可选的原正式终止合同。输出为保持原时间与关节语义的控制系数仿射映射。

`terminal_clock(problem, budget, progress_map, reference_time_map, duration, continuation, incoming_terminal_contract)` 返回归一化物理时间 u、源终止参数 z 及其来源。新请求使用已绑定的源时间映射；显式 rebind 通过原时钟接口恢复源映射并对齐已有设备时钟；普通续接将原声明的静止起点减去实际 elapsed。已经进入静止段时整个剩余区间保持常值；原后缀长度为零时保持零长度。该接口消费 z=1 所标识的终止区间，技能进度 s 和设备事件时钟由各自原接口保存。

`TerminalHold.from_clock(curve_map, clock_u, clock_z, source=...)` 识别终止后缀，按原三次系数建立常值条件。SciPy null_space 给出正交自由方向，NumPy lstsq 给出特解。设完整系数为 C，自由系数为 Z，输出满足 C=offset+matrix@Z。报告同时给出声明静止起点、三次空间隐含的常值起点、原/新维数和持续时间。

`reduce(C)` 对完整初值作正交最小二乘投影；`expand(Z)` 为 NumPy 或 Torch 恢复完整布局，Torch 保留导数。`pullback(J)` 将原条件 Jacobian 的完整系数列映射为自由系数列，保留控制行优先、关节列其次的顺序。原固定入口 q0/dq0 和原关节多项式均由同一 CubicAffineMap 表达。

正长度静止段与此前曲线保持原 C2 连接。无静止段时保留全部自由度；零自由变量时表示唯一曲线。正式导出仍用完整控制量构造原 Hermite 节点，验收通过 `_hold_polynomial_report` 对保存多项式独立计算后缀极值。终止投影属于初始化步骤，任务和几何资格继续由同一曲线的原接口评价。

`test_termination_clock.py` 覆盖源时间与进度区分、重新绑定、冻结合同续接及零长度后缀；`test_backend.py` 覆盖终止投影、原多项式导出、零自由变量和几何条件交换。保存七关节实例的投影与导数审查属于实验记录。
