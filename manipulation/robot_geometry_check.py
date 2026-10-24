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

from ._coal_scene_batch import NativePairTable, MAX_BATCH_SIZE


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
        # BEGIN shared fixed pair table and independently budgeted transient queries.
        # Both public reports and planning batches use the same cold Coal kernel.
        # Cached data identifies model relationships only, never past evidence.
        self._native_pair_table = None
        self._query_schema = None
        count = len(self._geometries)
        pair_count = count * (count - 1) // 2
        # Table: native owners/indices plus Python identity tuples and names.
        self.native_query_storage_nbytes = 4096 + count * 128 + pair_count * 128
        # For P>0, S=floor(MAX_BATCH_SIZE/P) gives S*G<=2*MAX_BATCH_SIZE.
        # A single larger state uses at most the full model inventory. Two
        # chunks can coexist while a generator resumes: matrices, native
        # transform copies, float lists and output arrays are counted here.
        # The last term covers one live cold GJK/EPA solver, not one per pair.
        self.native_query_temporary_nbytes = (
            2 * (max(2 * MAX_BATCH_SIZE, count) * 512
                 + max(MAX_BATCH_SIZE, pair_count) * 512)
            + 32768 + 512 * int(self._request.epa_max_iterations))
        # END temporary buffers remain live-query work even after table creation.

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

    # BEGIN report and numeric batch interfaces share preparation and one distance kernel.
    def _prepare_query(self, *, joint_names, joint_positions, T_reference_base,
                       pairs, frames=(), check_request=None):
        """Validate one complete named state and compute its float64 FK poses."""
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
        # END metadata reuse leaves each state and calibration independently evaluated.
        pin.forwardKinematics(self.model, self._data, q)
        pin.updateFramePlacements(self.model, self._data)
        poses = {name: world_base @ self._data.oMf[index].homogeneous
                 for name, index in frame_ids.items()}
        geometry_poses = {name: poses[frame] @ local for name, (frame, _, _, local) in self._geometries.items()}
        return names, values, violations, poses, geometry_poses, pair_list

    def _pair_table(self, pairs):
        """Return the fixed table after _prepare_query validates the pair schema."""
        identities = tuple((pair.first, pair.second) for pair in pairs)
        schema = self._native_pair_table
        if schema is None or schema[0] != identities:
            # Release the replaced table before allocating its successor.
            self._native_pair_table = None
            schema = None
            names = tuple(dict.fromkeys(name for pair in identities for name in pair))
            columns = {name: index for index, name in enumerate(names)}
            indices = np.asarray([(columns[a], columns[b]) for a, b in identities],
                                 dtype=np.intp).reshape(-1, 2)
            table = NativePairTable([self._geometries[name][1] for name in names], indices)
            schema = (identities, names, table)
            self._native_pair_table = schema
        return schema[1], schema[2]

    @property
    def native_query_resident_nbytes(self):
        """Current table storage; model-owned shapes and transient arrays are separate."""
        schema = self._native_pair_table
        if schema is None:
            return 0
        return int(schema[2].storage_nbytes) + 256 + len(schema[0]) * 80 + len(schema[1]) * 16

    def pair_evidence(self, pair, row):
        """Attach declared identity and numerical relation to an owned numeric row.

        Columns are distance, first/second witness xyz, normal xyz, witness
        identity error, and an optional full-entity separation lower bound.
        Distances and witnesses are metres in the calibrated reference frame;
        Coal's unit direction is dimensionless. Policy remains caller-owned.
        """
        boundary = max(self.numerics["gjk_tolerance"], self.numerics["epa_tolerance"])
        distance = row[0]
        relation = ("separated" if distance > boundary else "penetrating"
                    if distance < -boundary else "numerical_boundary")
        return {"first": pair.first, "second": pair.second,
                "policy": pair.policy, "policy_source": pair.source,
                "signed_distance_m": distance, "relation": relation,
                "witness_first": row[1:4], "witness_second": row[4:7],
                "normal_first_to_second": row[7:10], "witness_identity_error_m": row[10]}

    def query(self, *, joint_names, joint_positions, T_reference_base, pairs, frames=(),
              check_request=None):
        """Return JSON-ready FK, limits and raw distances for explicit model pairs.

        This complete report serves robot validation and execution preflight.
        Pair policy and unknown motion are interpreted by those consumers.
        """
        names, values, violations, poses, geometry_poses, pairs = self._prepare_query(
            joint_names=joint_names, joint_positions=joint_positions,
            T_reference_base=T_reference_base, pairs=pairs, frames=frames,
            check_request=check_request)
        geometry_names, table = self._pair_table(pairs)
        matrices = np.asarray([[geometry_poses[name] for name in geometry_names]],
                              dtype=np.float64).reshape(1, len(geometry_names), 4, 4)
        if check_request is not None:
            check_request()
        numeric = table.query(matrices, np.zeros((1, len(pairs)), bool), self._request)[0].tolist()
        if check_request is not None:
            check_request()
        rows = []
        for pair, row in zip(pairs, numeric):
            if check_request is not None:
                check_request()
            rows.append(self.pair_evidence(pair, row))
        return {"joint_names": list(names), "joint_positions": values.astype(float).tolist(),
                "joint_limit_violations": violations,
                "frame_poses": {name: pose.tolist() for name, pose in poses.items()},
                "geometry_poses": {name: pose.tolist() for name, pose in geometry_poses.items()},
                "geometry_sources": {name: value[2] for name, value in self._geometries.items()},
                "numerics": dict(self.numerics), "pairs": rows}

    def query_pair_batches(self, *, joint_names, joint_positions, T_reference_base,
                           pairs, projection_mask, check_request=None):
        """Yield owned numeric rows [state, pair, 12] in input order.

        Every state uses _prepare_query's original float64 FK and matrix
        products. Projection bounds are computed only at true mask entries.
        The scene consumes Python floats/lists directly for its retained JSON
        evidence. Each chunk has at most MAX_BATCH_SIZE pair evaluations,
        except that a larger single state remains one complete query.
        Cancellation is checked during state preparation, before/after each
        native call, and by the scene while consuming each pair. Errors within
        a native chunk reject it as a whole, with state and pair indices.
        """
        joint_names, pairs = tuple(joint_names), tuple(pairs)
        positions = np.asarray(joint_positions)
        if (positions.ndim != 2 or positions.shape[1] != len(joint_names)
                or positions.dtype.kind not in "iuf" or not np.isfinite(positions).all()):
            raise ValueError("joint_positions must be finite named state rows")
        mask = np.asarray(projection_mask)
        if mask.dtype != np.bool_ or mask.shape != (len(positions), len(pairs)):
            raise ValueError("projection_mask must have one Boolean per state and declared pair")
        state_count = max(1, MAX_BATCH_SIZE // max(1, len(pairs)))
        geometry_names = tuple(dict.fromkeys(name for pair in pairs for name in (pair.first, pair.second)))
        for begin in range(0, len(positions), state_count):
            end = min(begin + state_count, len(positions))
            matrices = np.empty((end - begin, len(geometry_names), 4, 4), dtype=np.float64)
            for offset, position in enumerate(positions[begin:end]):
                _, _, _, _, geometry_poses, _ = self._prepare_query(
                    joint_names=joint_names, joint_positions=position,
                    T_reference_base=T_reference_base, pairs=pairs,
                    check_request=check_request)
                for column, name in enumerate(geometry_names):
                    matrices[offset, column] = geometry_poses[name]
            _, table = self._pair_table(pairs)
            if check_request is not None:
                check_request()
            try:
                numeric = table.query(matrices, mask[begin:end], self._request)
            except RuntimeError as error:
                raise RuntimeError(f"Robot pair batch states [{begin}, {end}): {error}") from error
            if check_request is not None:
                check_request()
            yield numeric.tolist()
    # END no native solver history or duplicate distance backend is retained.
# END known robot geometry remains separate from observed task fields.
