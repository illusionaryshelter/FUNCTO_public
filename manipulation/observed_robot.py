"""Pure robot kinematics for the common observed-field geometry interface."""

# BEGIN registered observed link fields use native kinematic poses directly.
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .observed_mapping import FrozenObservedField
from .planning_problem import quaternion_multiply, quaternion_rotate


@dataclass(frozen=True)
class ObservedBody:
    """An observed component rigidly fixed in the named link coordinate frame."""

    name: str
    frame: str
    field: FrozenObservedField | None
    rigid_component: str
    source: str

    def record(self):
        return {"name": self.name, "frame": self.frame,
                "field": None if self.field is None else self.field.record(),
                "rigid_component": self.rigid_component, "source": self.source}

    @classmethod
    def from_record(cls, row):
        return cls(row["name"], row["frame"],
                   None if row["field"] is None else FrozenObservedField.from_record(row["field"]),
                   row["rigid_component"], row["source"])


def pure_kinematics(path):
    """Read the declared kinematic tree and reject geometry-bearing inputs."""
    root = ET.parse(path).getroot()
    if root.findall(".//visual") or root.findall(".//collision") or root.findall(".//mesh"):
        raise ValueError("The robot model must contain only kinematic link and joint declarations")
    return root


class DifferentiableRobotKinematics:
    """Use all named joints, including nominal fingers, for observed link poses.

    The planning solver may optimize only the arm. The scene appends the
    already selected measured/nominal finger coordinates before invoking this
    complete tree. No future finger pose is obtained from the solver's locked
    entrance configuration.
    """

    def __init__(self, configuration, *, frames, T_reference_base, device):
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg
        kinematics = configuration["robot_cfg"]["kinematics"]
        pure_kinematics(kinematics["urdf_path"])
        # The maintained URDF-only constructor exposes all independent joints
        # and explicitly constructs no collision model. Solver configuration
        # fields such as locked fingers therefore cannot alter this tree.
        native = KinematicsCfg.from_basic_urdf(
            kinematics["urdf_path"], kinematics["base_link"], list(frames),
            device_cfg=DeviceCfg(device=str(device)))
        self.model = Kinematics(native, compute_spheres=False, compute_com=False)
        self.joint_names = tuple(self.model.joint_names)
        base = np.asarray(T_reference_base, float)
        self.position = base[:3, 3].copy()
        self.quaternion = Rotation.from_matrix(base[:3, :3]).as_quat()[[3, 0, 1, 2]]

    def poses(self, position, joint_names):
        from curobo.types import JointState
        names = tuple(joint_names)
        if set(names) != set(self.joint_names):
            raise ValueError("Observed robot poses require the complete named joint coordinates")
        ordered = position[..., [names.index(name) for name in self.joint_names]].contiguous()
        state = self.model.compute_kinematics(
            JointState.from_position(ordered, joint_names=list(self.joint_names)))
        base_p = torch.as_tensor(self.position, dtype=position.dtype, device=position.device)
        base_q = torch.as_tensor(self.quaternion, dtype=position.dtype, device=position.device)
        tools = state.tool_poses
        result = {}
        for index, frame in enumerate(tools.tool_frames):
            q, p = tools.quaternion[..., index, :], tools.position[..., index, :]
            result[frame] = (base_p + quaternion_rotate(base_q, p),
                             quaternion_multiply(base_q, q))
        return result
# END the returned link poses carry the maintained library's joint gradients.
