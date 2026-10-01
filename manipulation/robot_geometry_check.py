"""Independent fixed-base FK and native convex-solid diagnostics on the CPU.

Pinocchio owns FK; Coal owns narrow-phase solid distances. This module neither
constructs a planning scene nor changes collision policy. Input vertices must
describe EACH native convex collision solid in its named frame, in metres.
In particular, a triangle surface or a hull of several nonconvex components is
not an interchangeable representation. Results are floating-point diagnostics,
not exact arithmetic, a free-space certificate, or trajectory validation.
"""

# BEGIN independent named robot FK and native-solid numerical diagnostics.
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

import coal
import numpy as np
import pinocchio as pin


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
class NativeSolid:
    """One native convex solid, with geometry transforms already baked in.

    Compiled mesh scale/recentering must be applied exactly once by the source
    adapter. Baking the compiled geom transform into these vertices also keeps
    any source affine scaling out of Coal's rigid Transform3s. A native box is
    represented exactly by its eight corners. Curved primitives require their
    own exact representation and must not be silently tessellated here.
    """

    name: str
    frame: str
    vertices_frame: np.ndarray
    source: str


@dataclass(frozen=True)
class PairQuery:
    """Caller-supplied policy identity; even excluded pairs retain raw results.

    ``required`` participates in the caller's collision contract; ``excluded``
    has an explicit exclusion source; ``diagnostic`` makes no policy claim.
    No pair is inferred from adjacency, a robot name, or a task name.
    """

    first: str
    second: str
    policy: str
    source: str


@dataclass(frozen=True)
class NativePrimitive:
    """An analytic native box or cylinder, without mesh approximation.

    Box dimensions are full x/y/z lengths. Cylinder dimensions are radius and
    full axial length; the local cylinder axis is z. All lengths are metres and
    T_frame_geometry is a proper rigid pose. Explicitly retain a separate geom
    for every component instead of taking a hull over the assembly.
    """

    name: str
    frame: str
    shape_kind: str
    dimensions: tuple
    T_frame_geometry: np.ndarray
    source: str


def _convex_solid(vertices):
    points = np.asarray(vertices)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
            or points.dtype.kind not in "iuf" or not np.isfinite(points).all()
            or np.linalg.matrix_rank(points - points.mean(axis=0)) != 3):
        raise ValueError("A native convex solid requires finite full-dimensional Nx3 vertices")
    native = coal.StdVec_Vec3s()
    for point in points:
        native.append(np.asarray(point, dtype=float))
    # Convex is a filled volume. BVHModel would test triangle surfaces and miss
    # containment, even when its source triangle mesh happens to be watertight.
    return coal.ConvexBase.convexHull(native, True, "Qt")


class RobotGeometryChecker:
    """Independent diagnostic model with complete scalar joint coordinates.

    The supported contract is a fixed-base tree with bounded revolute/prismatic
    joints and explicit fixed frames. Missing, duplicate and extra q names fail;
    input order is immaterial. Limit violations are reported without clipping
    observed coordinates, so the original feedback remains inspectable.
    """

    def __init__(self, urdf_path, *, base_frame, geometries):
        required = ("buildModelFromUrdf", "forwardKinematics", "updateFramePlacements", "neutral")
        if any(not callable(getattr(pin, name, None)) for name in required):
            raise ImportError("Robot Pinocchio API is required from the installed 'pin' distribution; "
                              "the imported 'pinocchio' module is not that robotics package")
        self.urdf_path = str(Path(urdf_path).resolve())
        self.base_frame = _name(base_frame, "base_frame")
        tree = ET.parse(self.urdf_path).getroot()
        links = [node.attrib["name"] for node in tree.findall("link")]
        joints = tree.findall("joint")
        joint_names = [node.attrib["name"] for node in joints]
        if len(links) != len(set(links)) or len(joint_names) != len(set(joint_names)):
            raise ValueError("URDF link and joint names must be unique")
        roots = set(links) - {node.find("child").attrib["link"] for node in joints}
        if roots != {base_frame}:
            raise ValueError("base_frame must name the unique fixed URDF root")
        for node in joints:
            if node.attrib["type"] not in ("fixed", "revolute", "prismatic") or node.find("mimic") is not None:
                raise ValueError("This checker requires independent bounded scalar joints and fixed frames")
        self.model = pin.buildModelFromUrdf(self.urdf_path)
        self._data = self.model.createData()
        self.joint_names = tuple(str(name) for name in self.model.names[1:])
        self._q_columns = {}
        for name in self.joint_names:
            joint = self.model.joints[self.model.getJointId(name)]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError("Every movable joint must have exactly one q and dq coordinate")
            self._q_columns[name] = int(joint.idx_q)
        if set(self._q_columns.values()) != set(range(self.model.nq)):
            raise ValueError("Named joints must cover the complete fixed-base model")
        lower, upper = self.model.lowerPositionLimit, self.model.upperPositionLimit
        if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
            raise ValueError("Every joint must have finite ordered position limits")
        self.joint_limits = {name: (float(lower[column]), float(upper[column]))
                             for name, column in self._q_columns.items()}
        self._geometries = {}
        for record in geometries:
            name = _name(record.name, "geometry name")
            if name in self._geometries:
                raise ValueError(f"Duplicate geometry name: {name}")
            self._frame_id(record.frame)
            if isinstance(record, NativeSolid):
                shape, local_pose = _convex_solid(record.vertices_frame), np.eye(4)
            elif isinstance(record, NativePrimitive):
                dimensions = np.asarray(record.dimensions)
                count = {"box": 3, "cylinder": 2}.get(record.shape_kind)
                if (count is None or dimensions.shape != (count,) or dimensions.dtype.kind not in "iuf"
                        or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
                    raise ValueError("Primitive requires positive box full lengths or cylinder radius/full length")
                dimensions = tuple(float(value) for value in dimensions)
                shape = coal.Box(*dimensions) if record.shape_kind == "box" else coal.Cylinder(*dimensions)
                local_pose = _rigid_pose(record.T_frame_geometry, "T_frame_geometry")
            else:
                raise ValueError("Geometry must be an explicit native convex solid or analytic primitive")
            self._geometries[name] = (_name(record.frame, "geometry frame"), shape,
                                      _name(record.source, "geometry source"), local_pose)
        if not self._geometries:
            raise ValueError("At least one explicitly sourced native solid is required")
        self._request = coal.DistanceRequest()
        self._request.enable_signed_distance = True
        # Keep maintained library defaults. Their approximate metre-scale
        # accuracy is not a strict error bound and must not be used to waive a
        # collision. In particular, do not tune GJK/EPA below 1e-6 here.
        self.numerics = {"gjk_tolerance": float(self._request.gjk_tolerance),
                         "epa_tolerance": float(self._request.epa_tolerance),
                         "enable_signed_distance": True,
                         "tolerance_is_error_bound": False}

    def _frame_id(self, name):
        _name(name, "frame")
        index = int(self.model.getFrameId(name))
        if index >= self.model.nframes:
            raise ValueError(f"Unknown URDF frame: {name}")
        return index

    def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=()):
        """Return JSON-ready FK, raw signed distances, witnesses and limits.

        Only the explicitly requested pairs are queried. This function reports
        numerical geometry relations without an overall collision-valid flag:
        policy exclusions, unknown space and continuous motion belong to the
        original caller, and no inferred ignore can enter this diagnostic.
        """
        names = tuple(joint_names)
        if (any(not isinstance(name, str) for name in names)
                or len(names) != len(set(names)) or set(names) != set(self.joint_names)):
            raise ValueError("joint_names must contain every model joint exactly once")
        values = np.asarray(joint_positions)
        if values.shape != (len(names),) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("joint_positions must be one finite scalar per named joint")
        world_base = _rigid_pose(T_reference_base)
        q = pin.neutral(self.model)
        violations = {}
        for name, value in zip(names, values):
            q[self._q_columns[name]] = value
            lower, upper = self.joint_limits[name]
            if value < lower or value > upper:
                violations[name] = {"position": float(value), "lower": lower, "upper": upper}
        selected_frames = tuple(dict.fromkeys((self.base_frame, *frames,
                                              *(value[0] for value in self._geometries.values()))))
        frame_ids = {name: self._frame_id(name) for name in selected_frames}
        pair_list, identities = [], set()
        for pair in pairs:
            if pair.first not in self._geometries or pair.second not in self._geometries or pair.first == pair.second:
                raise ValueError("Pair endpoints must be two distinct explicit geometry names")
            identity = frozenset((pair.first, pair.second))
            if identity in identities:
                raise ValueError("Duplicate unordered geometry pair")
            if pair.policy not in ("required", "excluded", "diagnostic"):
                raise ValueError("Pair policy must be required, excluded, or diagnostic")
            _name(pair.source, "pair policy source")
            identities.add(identity)
            pair_list.append(pair)
        pin.forwardKinematics(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        poses = {name: world_base @ self._data.oMf[index].homogeneous
                 for name, index in frame_ids.items()}
        geometry_poses = {name: poses[frame] @ local for name, (frame, _, _, local) in self._geometries.items()}
        transforms = {name: coal.Transform3s(pose[:3, :3], pose[:3, 3])
                      for name, pose in geometry_poses.items()}
        rows = []
        boundary = max(self.numerics["gjk_tolerance"], self.numerics["epa_tolerance"])
        for pair in pair_list:
            # A fresh result is essential: Coal updates an accumulated minimum.
            result = coal.DistanceResult()
            coal.distance(self._geometries[pair.first][1], transforms[pair.first],
                          self._geometries[pair.second][1], transforms[pair.second],
                          self._request, result)
            distance = float(result.min_distance)
            first = np.asarray(result.getNearestPoint1(), dtype=float)
            second = np.asarray(result.getNearestPoint2(), dtype=float)
            normal = np.asarray(result.normal, dtype=float)
            if not np.isfinite(np.concatenate(([distance], first, second, normal))).all():
                raise RuntimeError(f"Coal returned nonfinite geometry evidence for {pair.first}, {pair.second}")
            relation = ("separated" if distance > boundary else "penetrating"
                        if distance < -boundary else "numerical_boundary")
            rows.append({"first": pair.first, "second": pair.second,
                         "policy": pair.policy, "policy_source": pair.source,
                         "signed_distance_m": distance, "relation": relation,
                         "witness_first": first.tolist(), "witness_second": second.tolist(),
                         "normal_first_to_second": normal.tolist(),
                         "witness_identity_error_m": float(np.linalg.norm(second-first-distance*normal))})
        return {"joint_names": list(names), "joint_positions": values.astype(float).tolist(),
                "joint_limit_violations": violations,
                "frame_poses": {name: pose.tolist() for name, pose in poses.items()},
                "geometry_poses": {name: pose.tolist() for name, pose in geometry_poses.items()},
                "geometry_sources": {name: value[2] for name, value in self._geometries.items()},
                "numerics": dict(self.numerics), "pairs": rows}
# END independent named robot FK and native-solid numerical diagnostics.
