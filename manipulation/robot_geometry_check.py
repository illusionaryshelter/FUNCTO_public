"""Independent named-joint FK and distances between observed robot fields.

Pinocchio consumes a geometry-free kinematic tree. Each registered body keeps
its measured trilinear field in its named link frame. A missing observation
remains a missing field; it does not become an invented occupied volume.
"""

# BEGIN 机器人模型来源边界与设计依据。
# 禁止的是通过三维重建为任务相关物体制作 mesh：这需要人工建模，
# 且每遇到一个新物体都要重复，不能作为任务运行的前提。
# 机械臂官方随附的本体 mesh 允许使用；它是同一机械臂跨任务持续复用
# 的已知模型，不需要为每个任务物体重新制作。
# 已知机器人与指垫的形状、相对连杆的固定变换由本体模型给出，
# 模型位姿由基座标定和关节反馈的正运动学（FK）计算，不以相机可见为前提。
# 当前检查器仍只接收 ObservedBody、查询观测场，尚未恢复官方本体几何；
# 这一实现现状不能被解释为禁止官方机器人 mesh 或要求重新观测机器人。
# END 保留模型来源及上述原因，避免再次把任务物体重建禁令扩大到机器人。

# BEGIN one observed-entity geometry contract for robot model checks.
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pinocchio as pin

from .implicit_distance import query_distance
from .observed_robot import ObservedBody, pure_kinematics


def _name(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit nonempty name")
    return value


def _rigid_pose(value, label="T_reference_base"):
    pose = np.asarray(value)
    if (pose.shape != (4, 4) or pose.dtype.kind not in "iuf"
            or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-12, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-10, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-10, rtol=0)):
        raise ValueError(f"{label} must be a finite proper rigid 4x4 pose")
    return pose.astype(float, copy=True)


@dataclass(frozen=True)
class PairQuery:
    """An explicit pair policy, independent of distance or kinematic adjacency.

    Required pairs consume the requested separation. Excluded and mechanical
    contact pairs keep their declaration without manufacturing a distance.
    """
    first: str
    second: str
    policy: str
    source: str


class RobotGeometryChecker:
    """Rigid link poses, named limits and the common implicit-set query."""

    def __init__(self, urdf_path, *, base_frame, geometries):
        self.urdf_path = str(Path(urdf_path).resolve())
        self.base_frame = _name(base_frame, "base frame")
        tree = pure_kinematics(self.urdf_path)
        self.urdf_xml = ET.tostring(tree, encoding="unicode")
        links = {node.attrib["name"] for node in tree.findall("link")}
        joints = tree.findall("joint")
        if links - {node.find("child").attrib["link"] for node in joints} != {base_frame}:
            raise ValueError("The calibrated base must be the unique kinematic root")
        if any(node.attrib["type"] not in {"fixed", "revolute", "prismatic"}
               or node.find("mimic") is not None for node in joints):
            raise ValueError("This model requires independent bounded scalar joints")
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self._data = self.model.createData()
        self.joint_names = tuple(str(name) for name in self.model.names[1:])
        self._q_columns = {}
        for name in self.joint_names:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError("Each movable joint needs one named position and velocity")
            self._q_columns[name] = int(joint.idx_q)
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        if (set(self._q_columns.values()) != set(range(self.model.nq))
                or not np.isfinite([lower, upper]).all() or np.any(lower > upper)):
            raise ValueError("Named joints must cover finite ordered model limits")
        self.joint_limits = {name: (float(lower[i]), float(upper[i]))
                             for name, i in self._q_columns.items()}
        self.joint_velocity_limits = {name: float(self.model.velocityLimit[i])
                                     for name, i in self._q_columns.items()}
        self.bodies = {}
        for body in geometries:
            if not isinstance(body, ObservedBody) or body.name in self.bodies:
                raise ValueError("Robot geometry requires unique observed-body records")
            self._frame_id(body.frame)
            self.bodies[body.name] = body
        if not self.bodies:
            raise ValueError("The robot needs explicit body identities")
        self.numerics = {"distance_precision_m": 1e-4,
                         "representation": "observed_trilinear_occupied_set"}
        self.native_query_storage_nbytes = 0

    def _frame_id(self, name):
        index = int(self.model.getFrameId(_name(name, "frame")))
        if index >= self.model.nframes:
            raise ValueError(f"Unknown kinematic frame: {name}")
        return index

    @property
    def geometry_names(self):
        return tuple(self.bodies)

    @property
    def geometry_sources(self):
        return {name: body.source for name, body in self.bodies.items()}

    def model_record(self):
        return {"urdf_xml": self.urdf_xml,
                "geometries": [body.record() for body in self.bodies.values()]}

    def geometry_depending_on(self, joint_names):
        unknown = set(joint_names)
        if not unknown <= set(self.joint_names):
            raise ValueError("Unknown joint identities in robot geometry query")
        result = []
        for name, body in self.bodies.items():
            joint = int(self.model.frames[self._frame_id(body.frame)].parentJoint)
            while joint:
                if str(self.model.names[joint]) in unknown:
                    result.append(name)
                    break
                joint = int(self.model.parents[joint])
        return tuple(result)

    def query(self, *, joint_names, joint_positions, T_reference_base, pairs,
              frames=(), separation_m=.002, check_request=None):
        """Report represented-set evidence; unknown observations stay separate.

        ``clear`` certifies the requested separation only on represented
        observed sets. A missing body does not report a clear distance and
        does not impose a robot-camera-field-of-view restriction.
        """
        names, values = tuple(joint_names), np.asarray(joint_positions)
        if (len(set(names)) != len(names) or set(names) != set(self.joint_names)
                or values.shape != (len(names),) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError("Robot FK requires every finite named joint exactly once")
        base = _rigid_pose(T_reference_base)
        q, violations = pin.neutral(self.model), {}
        for name, value in zip(names, values):
            q[self._q_columns[name]] = value
            lower, upper = self.joint_limits[name]
            if not lower <= value <= upper:
                violations[name] = {"position": float(value), "lower": lower, "upper": upper}
        selected = tuple(dict.fromkeys((self.base_frame, *frames,
                                       *(body.frame for body in self.bodies.values()))))
        pin.forwardKinematics(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        poses = {name: base @ self._data.oMf[self._frame_id(name)].homogeneous for name in selected}
        geometry_poses = {name: poses[body.frame] for name, body in self.bodies.items()}
        rows, identities = [], set()
        for pair in pairs:
            if check_request is not None:
                check_request()
            identity = frozenset((pair.first, pair.second))
            if (len(identity) != 2 or identity in identities or not identity <= self.bodies.keys()
                    or pair.policy not in {"required", "excluded", "mechanical_contact", "diagnostic"}):
                raise ValueError("Body pairs need unique registered endpoints and explicit policy")
            identities.add(identity)
            first, second = self.bodies[pair.first], self.bodies[pair.second]
            row = {"first": pair.first, "second": pair.second,
                   "policy": pair.policy, "policy_source": _name(pair.source, "pair source")}
            if pair.policy in {"excluded", "mechanical_contact"}:
                row.update(status="permitted", lower_m=None, upper_m=None)
            elif first.rigid_component == second.rigid_component:
                row.update(status="same_rigid_body", lower_m=None, upper_m=None)
            elif first.field is None or second.field is None:
                row.update(status="missing_observation", lower_m=None, upper_m=None)
            else:
                row.update(query_distance(first.field, geometry_poses[first.name],
                    second.field, geometry_poses[second.name], separation_m=separation_m,
                    precision_m=self.numerics["distance_precision_m"], check_request=check_request))
            rows.append(row)
        return {"joint_names": list(names), "joint_positions": values.astype(float).tolist(),
                "joint_limit_violations": violations,
                "frame_poses": {name: pose.tolist() for name, pose in poses.items()},
                "geometry_poses": {name: pose.tolist() for name, pose in geometry_poses.items()},
                "geometry_sources": self.geometry_sources, "numerics": dict(self.numerics),
                "pairs": rows, "separation_m": float(separation_m)}
# END geometry and numerical uncertainty are reported without proxy penetration.
