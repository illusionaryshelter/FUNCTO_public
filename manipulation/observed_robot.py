"""Complete differentiable robot kinematics for registered native entities."""

# BEGIN registered native entities use the complete kinematic tree directly.
import xml.etree.ElementTree as ET

import torch

from .pin_kinematics import PinKinematics


def pure_kinematics(path):
    """Read the declared kinematic tree and reject geometry-bearing inputs."""
    # BEGIN 区分本函数的输入限制与机器人模型来源。
    # 下方检查只表达当前接口接受纯运动学 URDF，不定义碰撞几何的来源规则。
    # 官方随附、固定配置、跨任务复用的机器人 mesh 允许使用；任务物体、
    # 环境及其他对象保持 RGB-D 原场表示，不引入 mesh/CAD。完整原因见
    # robot_geometry_check.py 模块开头。
    # 不得据此检查推导出机器人或指垫必须通过 RGB-D 重新建模。
    # END 运动学文件限制不构成禁止官方机器人碰撞模型的依据。
    root = ET.parse(path).getroot()
    if root.findall(".//visual") or root.findall(".//collision") or root.findall(".//mesh"):
        raise ValueError("The robot model must contain only kinematic link and joint declarations")
    return root


class DifferentiableRobotKinematics:
    """Place native entities with the already registered Pinocchio model.

    The planning solver may optimize only the arm. The scene appends the
    already selected measured/nominal finger coordinates before invoking this
    complete model. Every call uses the same CPU float64 Pinocchio algorithm.
    Tensor copies connect CUDA/float32 proposal inputs to that algorithm and
    restore its output to the input device/dtype while retaining autograd.

    The official robot and fixed end-effector configuration are reusable
    across tasks because their shape and component relationships remain
    fixed. This adapter receives the validator's existing kinematic model;
    its poses come from named joints and the same base calibration.
    """

    # BEGIN scene FK reuses the registered model and one CPU algorithm.
    def __init__(self, model, *, frames, T_reference_base):
        self.model = model
        self._bridge = PinKinematics(model, frames=frames, T_reference_base=T_reference_base)
        self.joint_names = self._bridge.joint_names
        self.frames = self._bridge.frames

    def poses(self, position, joint_names):
        """Return original-device link poses with the same named-joint VJP."""
        cpu_position = position.to(device="cpu", dtype=torch.float64)
        poses = self._bridge.poses(cpu_position, joint_names)
        return {frame: tuple(value.to(device=position.device, dtype=position.dtype) for value in pose)
                for frame, pose in poses.items()}
    # END device copies carry gradients without introducing another FK model.
# END the returned link poses carry the maintained library's joint gradients.
