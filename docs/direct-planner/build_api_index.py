"""Generate a source-aligned API index for the migrated library component.

Module contracts below describe inputs, outputs and responsibilities. Function
signatures and direct import relationships are read from the actual package;
the architecture README supplies the mathematical and lifecycle interpretation.
"""

import ast
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]/'manipulation/direct_planner'
CONTRACTS = {
    'direct_preference': ('公开后端与数值规划对象',
        'PlanningProblem、OnlinePlanningScene、官方机器人配置、验证器、预算、ReferenceObjective、初始化声明和 support_presolve 布尔配置。',
        'DirectPreferenceBackend 持有请求资源；DirectPreferencePlanner 绑定参考和计算，生成初值、调用 IPOPT、提供候选与诊断；backend.plan 返回 PlanningResult。'),
    'direct_setup': ('时钟、官方配置与原物理域',
        '当前具名 q/dq、任务事件、现场手指模型、机器人配置及原进度/续接声明。',
        '绑定参考时钟、原配置的关节位置/速度/加速度/jerk 界限、CubicJointDomain 与共享运动学；候选域检查输出连续极值报告。'),
    'direct_lifecycle': ('累计请求与候选交付',
        '任务、现场、调用方持有的后端和预算，以及可选请求账本、续接、取消、截止时刻和关节参考。',
        '完成输入绑定、初始化、求解、候选资格及原执行绑定，返回包含曲线或失败原因的 PlanningResult 和累计账本。'),
    'direct_computation': ('固定数值问题与同点共享',
        '原 TrajectoryComputation、末段仿射映射、几何提供器、完整参考目标、support_presolve 配置及自由曲线系数。',
        'DirectComputation/BoundComputation 建立固定行列结构；ComputationPoint 保存目标、条件、稀疏导数、空间见证和导出候选所需的同点状态。'),
    'reference_objective': ('完整参考与统一目标',
        '显式误差尺度和权重、BoundToolReference、物理查询时刻及共享工具位姿/速度。',
        '冻结原参考样本和归一化物理积分权重，返回完整参考、角运动、关节运动及原声明偏好的分项标量和可微图。'),
    'reference_initialization': ('官方运动链与末端提案',
        'Pinocchio 机器人模型、具名关节、工具和基座标定、原参考、入口 q/dq、关节区间及显式提案预算。',
        'SymbolicKinematics 提供符号 FK、位姿 Jacobian 和连续四元数；有界拟合及原物理时钟连接生成全部末端分支提案和证据。'),
    'reference_projection': ('完整参考初值拟合',
        '原曲线与末段映射、同点机器人声明、完整参考、required 事件以及末端连接系数。',
        '编译稀疏残差和 Jacobian，在原系数位置盒内执行 SciPy 最小二乘，返回拟合系数、目标分项、次数及停止状态；原完整问题负责后续排序。'),
    'terminal_binding': ('末段静止参数化',
        '原三次曲线基、参考进度时钟与终端来源。',
        '返回 TerminalHold 的仿射系数映射；expand/reduce/pullback 分别恢复完整系数、求自由坐标和传递导数。'),
    'coefficient_coordinates': ('固定求解变量坐标',
        '完整参考尺度、原时间积分、初值关节状态、关节范围与原样条映射。',
        '按关节块形成度量并 Cholesky 分解，返回固定线性变量映射及条件数；原目标和约束在恢复后的原系数上计算。'),
    'differentiable_kinematics': ('共享原生运动学与反向接口',
        '同一官方 Pinocchio 模型、请求参考系标定、具名关节 q/dq 和参考连杆集合。',
        '一次原生调用给出连杆位置、wxyz 四元数、线/角速度及关节导数；Torch 反向接口把相同一阶信息传给任务和目标。'),
    'support_linearization': ('任务事件单位和支撑解析因子',
        '原任务声明、完整具名机器人状态、连杆 Jacobian、曲线系数和技能材料点索引。',
        '返回等式/不等式向量、原行掩码及支撑体积/足迹的解析因子；required 位姿集合采用等价的归一化平方残差，其余任务使用原计算。'),
    'support_presolve': ('支撑半空间的静态预消元',
        '原变量任务行、required 位置球、同物理查询处的固定功能/材料坐标及支撑半空间。',
        '用区间算术计算刚性旋转不变的材料半径与半空间上界；返回被保留位置球蕴含的删行掩码与原行证明记录。'),
    'sparse_task': ('任务稀疏系数导数',
        '原条件掩码、支撑因子、事件时刻与曲线局部支撑结构。',
        '保留完整评分行，按 support_presolve 选择位置球蕴含的压缩行或全部原变量行；绑定解析索引并批量填槽，剩余行自动微分。'),
    'sparse_physical': ('连续关节条件与导数',
        '原 CubicJointDomain、末段映射和完整关节曲线系数。',
        '返回各多项式段的位置/速度极值、加速度端值、jerk 有符号条件和局部系数导数，保留原物理秒单位。'),
    'sparse_rows': ('固定稀疏行列布局',
        '各条件行实际依赖的系数列和总变量数。',
        '生成按行存储、CCS 交付次序和局部槽索引；任务、几何、关节域和求解器共用这一明确结构。'),
    'constraint_envelope': ('同实体条件的最大值表示',
        '完整几何行、原行身份与结构列。',
        '按同一注册实体对及条件取最大值，返回等价的整体不等式和实际胜出行；稀疏导数使用该行的原数值。'),
    'geometry_constraints': ('原模型的几何条件与空间见证',
        '原任务查询时刻、场景实体/关系、官方机器人模型、名义手指和完整曲线系数。',
        '固定时钟/事件侧布局，组合原生距离、TSDF 查询和区间驻点，输出每行条件、刚体/关节因子、未知支持与空间见证，并提供稀疏系数导数。'),
    'batch_distance': ('GPU 场查询与 CPU 原生计算的组合',
        '明确注册的几何对象、机器人及声明运动实体源、原行对应的位姿/许可。',
        '建立固定源点和设备场；submit 提交各组设备查询，PendingDistance.finish 统一回传并按原次序选取胜出点和一阶导数。'),
    'gpu_field': ('原标量场和官方机器人实体的设备查询',
        'float64 点/刚体位姿、现场 TSDF 数值/支持、官方机器人模型资源与有限接触许可。',
        'Warp 核返回原标量值、支持状态、最大违反及原点索引；小结果留在设备供 BatchDistance 统一消费。'),
    'native_interval_distance': ('机器人实体距离的区间驻点',
        '官方机器人实体对、原端点距离、曲线多项式、关节状态、连杆 Jacobian 和名义手指。',
        '原连续速度范围排除常量尾区间，其余区间以距离时间导数括根并调用 Brent，返回胜出时刻、距离、最近点及关节导数。'),
    'native_gradient': ('最近点到关节的编译导数',
        '12 项原生距离记录、两参考连杆原点、两组 6×N Jacobian 和共同祖先列掩码。',
        'Numba float64 CPU ufunc 输出 N 列距离导数，支持标量、批次及 strided 数组；共同刚体运动对应列为零。'),
    'nominal_hand_geometry': ('声明的位置控制手指模型',
        '原手指命令多项式、任务条件、事件时钟及声明进度范围。',
        '返回同一时钟上的名义手指位置、速度和条件未知区间；模型由命令、声明关系及关节反馈建立。'),
    'scene_relationships': ('显式关系的目标项',
        '原现场接触/初始支撑声明、固定材料点、工具位姿、物理时间和技能进度。',
        '调用原关系目标计算，返回同一图中的关系偏好标量及普通证据；实体非穿透由几何条件提供。'),
    'material_hull': ('技能材料点的线性约束索引',
        '技能表示中的完整材料点数组。',
        '返回半空间约束所需的原点索引，按实际仿射维数选择线性极值点；原技能点数组和软目标保持完整。'),
    'candidate_result': ('同点候选条件与正式验证元数据',
        '求解器保留的 ComputationPoint、其原曲线、任务/场景声明、连续关节域和统一目标。',
        '消费同点原任务与几何值，补齐原连续域、末段及声明关系信息，形成原绑定接口消费的接受/拒绝结果及具体证据。'),
    'shared_solver': ('CasADi/IPOPT 与稀疏回调',
        '已绑定的同一数值问题、自由初值、实际预算/中断接口和固定数值坐标。',
        '精确同点复用提供 f、g 和一阶导数；IPOPT 使用有限记忆曲率和 MUMPS，返回最低目标可行点或失败诊断，并保存实际停止状态和计数。'),
}


def definitions(tree):
    """List module functions and class methods in their real source order."""
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            yield node.name, node
        elif isinstance(node, ast.ClassDef):
            for member in node.body:
                if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    yield node.name+'.'+member.name, member


def main():
    """Write component contracts together with signatures and import consumers."""
    trees = {name:ast.parse((PACKAGE/(name+'.py')).read_text()) for name in CONTRACTS}
    local = {name:sorted({node.module for node in ast.walk(tree)
                         if isinstance(node,ast.ImportFrom) and node.level==1})
             for name,tree in trees.items()}
    text = ['# 直接规划器 API 与实现索引',
            '实际代码基线与完整数学定义见 [组件架构](README.md)。本页的函数签名和包内依赖由当前源码生成；模块契约说明数据输入、计算职责及返回内容。',
            '公开包入口为 `DirectPreferenceBackend`、`DirectPreferencePlanner`、`ReferenceObjective`、`plan_trajectory` 与原 `bind_pose_objectives`。内部模块通过包内相对导入组合，实验文件读取由调用方负责。']
    for name,(title,inputs,outputs) in CONTRACTS.items():
        consumers = sorted(other for other,deps in local.items() if name in deps)
        dependencies = ', '.join('`'+dep+'`' for dep in local[name]) or '原 FUNCTO / 数值库接口'
        consumer_text = ', '.join('`'+dep+'`' for dep in consumers) or '包公开入口或调用方'
        text.extend([f'## {name}：{title}',
                     f'[源码](../../manipulation/direct_planner/{name}.py)。输入：{inputs}',
                     '计算和输出：'+outputs,
                     f'直接包内依赖：{dependencies}。包内调用组件：{consumer_text}。'])
        rows = ['| API 签名 | 调用作用 |', '|---|---|']
        for symbol,node in definitions(trees[name]):
            signature = symbol+'('+ast.unparse(node.args)+')'
            description = (ast.get_docstring(node) or outputs).split('\n')[0]
            rows.append('| `'+signature.replace('|','\\|')+'` | '+description.replace('|','\\|')+' |')
        text.append('\n'.join(rows))
    (HERE/'api.md').write_text('\n\n'.join(text)+'\n')
    print('Wrote API contracts for',len(CONTRACTS),'modules')


if __name__ == '__main__':
    main()
