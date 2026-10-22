"""Named-joint FK and distances between declared native robot solids.

Pinocchio owns kinematics; Coal queries each supplied convex collision entity
or analytic primitive. These are reusable robot model components, independently
of camera visibility. Their mechanical pair policy is supplied explicitly.
"""

# BEGIN 机器人模型来源边界与设计依据。
# 禁止的是通过三维重建为任务相关物体制作 mesh：这需要人工建模，
# 且每遇到一个新物体都要重复，不能作为任务运行的前提。
# 机械臂官方随附的本体 mesh 允许使用；它是同一机械臂跨任务持续复用
# 的已知模型，不需要为每个任务物体重新制作。
# 已知机器人与指垫的形状、相对连杆的固定变换由本体模型给出，
# 模型位姿由基座标定和关节反馈的正运动学（FK）计算，不以相机可见为前提。
# 本检查器使用上述已知本体碰撞实体；任务物体的原 TSDF 由场景查询使用。
# END 保留模型来源及上述原因，避免再次把任务物体重建禁令扩大到机器人。

# BEGIN restore the approved known-model checker without task-field conversion.
from copy import deepcopy
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

    ``required`` must remain separated; ``mechanical_contact`` explicitly
    permits contact between exactly these two robot solids within their
    independently checked mechanical joint domain. ``excluded`` records native
    filtering; ``diagnostic`` makes no acceptance claim. All retain distances.
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
    """Complete named robot FK and declared native-solid numerical evidence."""

    def __init__(self, urdf_path, *, base_frame, geometries):
        self.urdf_path = str(Path(urdf_path).resolve())
        self.base_frame = _name(base_frame, "base_frame")
        tree = ET.parse(self.urdf_path).getroot()
        # BEGIN retain the consumed source model in the planning binding.
        # A filesystem path alone cannot identify the kinematic tree or the
        # supplied collision solids after the source files have changed.
        self.urdf_xml = ET.tostring(tree, encoding="unicode")
        self._geometry_model = []
        # END serialized identity contains no independently reconstructed model.
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
        # BEGIN expose the same URDF velocity contract to independent preflight.
        self.joint_velocity_limits = {name: float(self.model.velocityLimit[column])
                                      for name, column in self._q_columns.items()}
        # END expose the same URDF velocity contract to independent preflight.
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
            # BEGIN preserve the exact inputs used to construct each Coal shape.
            identity = {"name": name, "frame": record.frame, "source": record.source}
            if isinstance(record, NativeSolid):
                identity.update(kind="convex", vertices_frame=np.asarray(record.vertices_frame, float).tolist())
            else:
                identity.update(kind=record.shape_kind, dimensions=list(dimensions),
                                T_frame_geometry=local_pose.tolist())
            self._geometry_model.append(identity)
            # END individual components remain distinct in the frozen binding.
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
        # BEGIN retain one pair schema in bounded native blocks, with cold solvers.
        # This workspace is used only by query(), whose explicit native pairs
        # all require raw evidence regardless of their acceptance policy.
        self._native_distance_batches = None
        # BEGIN retain one validated request layout for this fixed robot model.
        # Joint names, requested frames and frozen PairQuery records identify
        # only read-only model relationships. A changed layout is validated
        # before replacing this entry. Numerical states, calibration, FK,
        # distance results and native solver histories remain query-local.
        self._query_schema = None
        # END repeated states reuse metadata, never geometric evidence.
        # Reserve before any lazy query allocation. Each unordered native pair
        # can occur once. A block has at most 256 unique geometries; its dense
        # Pinocchio pair map therefore cannot grow quadratically with the full
        # robot inventory. At default EPA=64, 64 KiB/pair covers the working and
        # pristine distance solvers, GeometryData's other functors/results,
        # the working EPA's 68 vertices/132 faces, and allocation overhead.
        count = len(self._geometries)
        pair_count = count * (count-1) // 2
        block_geometries = min(count, 256)
        name_bytes = sum(len(name.encode("utf-8")) for name in self._geometries)
        self.native_query_storage_nbytes = (
            pair_count * (32768 + 512 * int(self._request.epa_max_iterations))
            + ((pair_count+127)//128) * (4096 + block_geometries*4096 + 8*block_geometries**2 + 2*name_bytes))

    def _frame_id(self, name):
        _name(name, "frame")
        index = int(self.model.getFrameId(name))
        if index >= self.model.nframes:
            raise ValueError(f"Unknown URDF frame: {name}")
        return index

    @property
    def geometry_names(self):
        """Stable native-solid order; an acceptance layer must cover all pairs."""
        return tuple(self._geometries)

    @property
    def geometry_sources(self):
        return {name: value[2] for name, value in self._geometries.items()}

    def model_record(self):
        """Serialize the consumed URDF and every explicit native solid input."""
        return {"urdf_xml": self.urdf_xml, "geometries": deepcopy(self._geometry_model)}

    def geometry_depending_on(self, joint_names):
        unknown = set(joint_names)
        if not unknown <= set(self.joint_names):
            raise ValueError("Unmodelled joints must be named coordinates of this robot")
        result = []
        for name, (frame, _, _, _) in self._geometries.items():
            joint = int(self.model.frames[self._frame_id(frame)].parentJoint)
            while joint:
                if str(self.model.names[joint]) in unknown:
                    result.append(name)
                    break
                joint = int(self.model.parents[joint])
        return tuple(result)

    def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(),
              check_request=None):
        """Return JSON-ready FK, raw signed distances, witnesses and limits.

        Only the explicitly requested pairs are queried. This function reports
        numerical geometry relations without an overall collision-valid flag:
        policy exclusions, unknown space and continuous motion belong to the
        original caller, and no inferred ignore can enter this diagnostic.
        """
        if check_request is not None:
            check_request()
        names = tuple(joint_names)
        # BEGIN prepare the complete named layout only when its schema changes.
        schema = self._query_schema
        same_names = schema is not None and names == schema["joint_names"]
        if same_names:
            columns, limits = schema["joint_columns"], schema["joint_limits"]
        else:
            if (any(not isinstance(name, str) for name in names)
                    or len(names) != len(set(names)) or set(names) != set(self.joint_names)):
                raise ValueError("joint_names must contain every model joint exactly once")
            columns = tuple(self._q_columns[name] for name in names)
            limits = tuple(self.joint_limits[name] for name in names)
        # END each call still validates its actual joint values and calibration.
        values = np.asarray(joint_positions)
        if values.shape != (len(names),) or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError("joint_positions must be one finite scalar per named joint")
        world_base = _rigid_pose(T_reference_base)
        q = pin.neutral(self.model)
        violations = {}
        for name, value, column, (lower, upper) in zip(names, values, columns, limits):
            q[column] = value
            if value < lower or value > upper:
                violations[name] = {"position": float(value), "lower": lower, "upper": upper}
        # BEGIN the last complete immutable schema owns frame and pair validation.
        # PairQuery is frozen and its validated fields are strings. Tuple
        # equality therefore retains endpoint direction, policy and source,
        # including newly supplied records with the same declared values.
        requested_frames, pair_list = tuple(frames), tuple(pairs)
        if (not same_names or requested_frames != schema["frames"]
                or pair_list != schema["pairs"]):
            selected_frames = tuple(dict.fromkeys((self.base_frame, *requested_frames,
                                                  *(value[0] for value in self._geometries.values()))))
            frame_ids = {name: self._frame_id(name) for name in selected_frames}
            identities = set()
            for pair in pair_list:
                if pair.first not in self._geometries or pair.second not in self._geometries or pair.first == pair.second:
                    raise ValueError("Pair endpoints must be two distinct explicit geometry names")
                identity = frozenset((pair.first, pair.second))
                if identity in identities:
                    raise ValueError("Duplicate unordered geometry pair")
                if pair.policy not in ("required", "excluded", "diagnostic", "mechanical_contact"):
                    raise ValueError("Pair policy must be required, excluded, diagnostic, or mechanical_contact")
                _name(pair.source, "pair policy source")
                identities.add(identity)
            schema = {"joint_names": names, "joint_columns": columns, "joint_limits": limits,
                      "frames": requested_frames, "frame_ids": frame_ids, "pairs": pair_list}
            self._query_schema = schema
        frame_ids = schema["frame_ids"]
        # END FK, pair order and cold native distance queries keep their original path.
        pin.forwardKinematics(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        poses = {name: world_base @ self._data.oMf[index].homogeneous
                 for name, index in frame_ids.items()}
        geometry_poses = {name: poses[frame] @ local for name, (frame, _, _, local) in self._geometries.items()}
        # BEGIN run independent native pairs in C++, retaining their exact order.
        # FK and the two matrix multiplications above deliberately stay in their
        # original NumPy/Pinocchio order; only prepared rigid matrices are copied
        # into the many-pair distance workspace below.
        distances = self._native_pair_distances(pair_list, geometry_poses)
        # END finite input validation precedes all native distance work.
        rows = []
        boundary = max(self.numerics["gjk_tolerance"], self.numerics["epa_tolerance"])
        # BEGIN inspect every original pair's fresh result, including exclusions.
        for pair, result in zip(pair_list, distances):
            if check_request is not None:
                check_request()
            # Pinocchio clears each DistanceResult before its native query.
            # _native_pair_distances additionally resets the entire solver;
            # default GJK direction alone does not reset its support hint.
            distance = float(result.min_distance)
            # END report construction and nonfinite rejection retain their order.
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

    def _native_pair_distances(self, pairs, poses):
        """Yield original-order results without retaining warm solver state.

        Pinocchio's geometry model has a dense geometry-to-pair map. Limiting
        each workspace to 128 pairs (at most 256 geometries) bounds that map to
        256 by 256 integer entries even for a large, sparse pair list. Every block
        of the most recent complete pair schema is retained, so larger fixed
        schemas do not rebuild blocks on every state. Changing pair order
        releases the old schema before allocating its replacement.
        All shape allocations finish before functors retain GeometryObject
        addresses. The geometry model outlives both the working and pristine
        functor vectors.

        A batch may perform later independent distance work before Python checks
        an earlier result. Every nonfinite result still rejects the whole call;
        native exceptions propagate with Pinocchio's pair context and no partial
        report is returned. This interface never owns scene contact permissions.
        """
        identities = tuple((pair.first, pair.second) for pair in pairs)
        schema = self._native_distance_batches
        if schema is None or schema[0] != identities:
            # Drop both old references before construction, including when the
            # new schema is empty or construction raises. No stale or partially
            # configured workspace can serve the following query.
            self._native_distance_batches = None
            schema = None
            blocks = []
            for start in range(0, len(identities), 128):
                block_pairs = identities[start:start+128]
                names = tuple(dict.fromkeys(name for pair in block_pairs for name in pair))
                columns = {name: index for index, name in enumerate(names)}
                model = pin.GeometryModel()
                for name in names:
                    model.addGeometryObject(pin.GeometryObject(
                        name, 0, pin.SE3.Identity(), self._geometries[name][1]))
                for first, second in block_pairs:
                    model.addCollisionPair(pin.CollisionPair(columns[first], columns[second]))
                data = pin.GeometryData(model)
                for index in range(len(block_pairs)):
                    data.distanceRequests[index] = self._request
                # Value-copy the never-evaluated solver vector. Restoring this
                # vector resets GJK/EPA/support hints, including the history
                # that Coal retains even with gjk_initial_guess=DefaultGuess.
                pristine = data.distance_functors.copy()
                blocks.append((names, model, data, pristine))
            schema = (identities, tuple(blocks))
            self._native_distance_batches = schema
        for names, model, data, pristine in schema[1]:
            data.distance_functors = pristine
            for index, name in enumerate(names):
                pose = poses[name]
                data.oMg[index] = pin.SE3(pose[:3, :3], pose[:3, 3])
            # Returned minimum index is intentionally ignored: all pairs and
            # their signed distances/witnesses/normals remain separately read.
            pin.computeDistances(model, data)
            for index in range(len(model.collisionPairs)):
                yield data.distanceResults[index]
# END known robot geometry remains separate from observed task fields.
