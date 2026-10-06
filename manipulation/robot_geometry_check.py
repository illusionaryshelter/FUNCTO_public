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

from .resource_admission import DEFAULT_MAX_HOST_RSS_BYTES, require_host_memory


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


# BEGIN distinguish legal mechanical contact from native collision filtering.
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
# END a physics-enabled contact is not inherently an illegal robot state.


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
        # BEGIN native distance batches use this installed robotics API.
        required = ("buildModelFromUrdf", "forwardKinematics", "updateFramePlacements", "neutral",
                    "GeometryModel", "GeometryData", "GeometryObject", "CollisionPair", "SE3", "computeDistances")
        # END missing native APIs fail before any state is evaluated.
        if any(not callable(getattr(pin, name, None)) for name in required):
            raise ImportError("Robot Pinocchio API is required from the installed 'pin' distribution; "
                              "the imported 'pinocchio' module is not that robotics package")
        self.urdf_path = str(Path(urdf_path).resolve())
        self.base_frame = _name(base_frame, "base_frame")
        tree = ET.parse(self.urdf_path).getroot()
        # BEGIN retain the consumed source model in the planning binding.
        # A filesystem path alone cannot identify the kinematic tree or the
        # supplied collision solids after the source files have changed.
        self._model_urdf = ET.tostring(tree, encoding="unicode")
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
        # BEGIN retain exact convex vertices for whole-overlap contact localization.
        self._solid_vertices = {}
        # END curved shapes retain analytic Coal geometry and explicit localization gaps.
        for record in geometries:
            name = _name(record.name, "geometry name")
            if name in self._geometries:
                raise ValueError(f"Duplicate geometry name: {name}")
            self._frame_id(record.frame)
            if isinstance(record, NativeSolid):
                shape, local_pose = _convex_solid(record.vertices_frame), np.eye(4)
                self._solid_vertices[name] = np.asarray(record.vertices_frame, float).copy()
            elif isinstance(record, NativePrimitive):
                dimensions = np.asarray(record.dimensions)
                count = {"box": 3, "cylinder": 2}.get(record.shape_kind)
                if (count is None or dimensions.shape != (count,) or dimensions.dtype.kind not in "iuf"
                        or not np.isfinite(dimensions).all() or np.any(dimensions <= 0)):
                    raise ValueError("Primitive requires positive box full lengths or cylinder radius/full length")
                dimensions = tuple(float(value) for value in dimensions)
                shape = coal.Box(*dimensions) if record.shape_kind == "box" else coal.Cylinder(*dimensions)
                local_pose = _rigid_pose(record.T_frame_geometry, "T_frame_geometry")
                self._solid_vertices[name] = _box_vertices(dimensions) if record.shape_kind == "box" else None
            else:
                raise ValueError("Geometry must be an explicit native convex solid or analytic primitive")
            self._geometries[name] = (_name(record.frame, "geometry frame"), shape,
                                      _name(record.source, "geometry source"), local_pose)
            # BEGIN preserve the exact inputs used to construct each Coal shape.
            identity = {"name": name, "frame": record.frame, "source": record.source}
            if isinstance(record, NativeSolid):
                identity.update(kind="convex", vertices_frame=self._solid_vertices[name].tolist())
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
        # Scene TSDF components keep their ordered permission/early-stop loop.
        # This workspace is used only by query(), whose explicit native pairs
        # all require raw evidence regardless of their acceptance policy.
        self._native_distance_batches = None
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
        # END no old distance, permission or solver history is reused as evidence.

    def _frame_id(self, name):
        _name(name, "frame")
        index = int(self.model.getFrameId(name))
        if index >= self.model.nframes:
            raise ValueError(f"Unknown URDF frame: {name}")
        return index

    # BEGIN expose immutable geometry identities for a complete acceptance policy.
    @property
    def geometry_names(self):
        """Stable native-solid order; an acceptance layer must cover all pairs."""
        return tuple(self._geometries)

    @property
    def geometry_sources(self):
        return {name: value[2] for name, value in self._geometries.items()}
    # END expose immutable geometry identities for a complete acceptance policy.

    # BEGIN provide source geometry without exposing mutable Coal objects.
    def model_record(self):
        """Serialize the consumed URDF and every explicit native solid input."""
        from copy import deepcopy
        return {"urdf_xml": self._model_urdf, "geometries": deepcopy(self._geometry_model)}
    # END callers receive owned JSON-compatible evidence, not shared buffers.

    # BEGIN identify exactly which solids depend on unmodelled actuator coordinates.
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
    # END an unknown finger never removes a rigid palm or unrelated arm link.

    # BEGIN independent scene/target/payload query using the formal model snapshot.
    def bind_scene(self, scene_binding, *, previous_checker=None,
                   max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
        """Build once from the owned finite scene snapshot, without GPU queries."""
        from copy import deepcopy
        from manipulation.planning_problem import PosePath
        from manipulation.motion import MotionModel
        # BEGIN unchanged source records share geometry, never query or policy state.
        from .tsdf_geometry import ObservedTsdfGeometry
        previous_shapes, previous_records = {}, {}
        if previous_checker is not None:
            for row in previous_checker._scene_shapes:
                if isinstance(row["shape"], ObservedTsdfGeometry):
                    previous_shapes[(row["owner"], row["layer"], row["name"])] = row
            for partition in [previous_checker._scene_binding["background"],
                              *previous_checker._scene_binding["owners"]]:
                owner = partition.get("owner_id", "background")
                geometry = partition["geometry"]
                layers = geometry["layers"] if "layers" in geometry else {None: geometry}
                for layer, values in layers.items():
                    for record in values["solids"]:
                        previous_records[(owner, layer, record["name"])] = record
        # The existing scene stays alive during replacement. Its pending shapes
        # are counted once by identity even if several rollout clones share it.
        require_host_memory(_owned_record_bytes(scene_binding),
            pending_existing_bytes=ObservedTsdfGeometry.pending_storage_total_nbytes(),
            max_rss_bytes=max_host_rss_bytes)
        # END compare complete records at binding, not names or dataset identities.
        self._scene_binding = deepcopy(scene_binding)
        self._scene_owners = {}
        self._scene_shapes = []
        self._has_observed_tsdf = False
        self._scene_gaps = []
        self._scene_native_aliases = {}
        self._scene_release_predictions = {}
        # BEGIN each immutable scene owns one last exact surface result per pair.
        # Rebinding discards shape identities and every cached witness. No
        # geometry, pose, contact permission or signed-piece verdict is shared
        # between different scene snapshots by this cache.
        self._scene_surface_distance_cache = {}
        # BEGIN the next state may requery a previous component at its new pose.
        # A new checker epoch has its own hints even when source maps are shared.
        self._scene_component_hints = {}
        # END old distances and permission verdicts are never stored in hints.
        # END entry count is bounded by the admitted finite shape-pair inventory.
        # BEGIN keep both observed resolutions and their shared body identities.
        self._scene_layers = {}
        self._scene_body_groups = tuple(deepcopy(scene_binding.get("robot_collision_bodies", ())))
        if any("layers" in part["geometry"] for part in [scene_binding["background"], *scene_binding["owners"]]):
            from .voxel_layers import validate_body_groups
            self._scene_body_groups = validate_body_groups(self._scene_body_groups,
                                                           len(scene_binding["robot_sphere_materials"]))
        # END ordinary single-field scenes require no shared layer decisions.
        for partition in [scene_binding["background"], *scene_binding["owners"]]:
            owner_id = partition.get("owner_id", "background")
            # BEGIN defer layered query gaps until the saved target layer is known.
            geometry = partition["geometry"]
            if "layers" in geometry:
                if set(geometry["layers"]) != {"coarse", "fine"}:
                    raise ValueError("Independent layered geometry requires exactly coarse and fine fields")
                from .voxel_layers import FineRegion
                region = geometry["fine_region"]
                margin = float(region["measurement_allowance_m"])
                if not np.isfinite(margin) or margin < 0:
                    raise ValueError("Layered measurement allowance must be finite and nonnegative")
                self._scene_layers[owner_id] = {"grid_pose": _pose7(region["pose"]),
                    "region": FineRegion(region), "measurement_allowance_m": margin,
                    "query_gaps": {key: deepcopy(value["query_gaps"]) for key, value in geometry["layers"].items()}}
                layers = geometry["layers"]
                if partition.get("moving", False):
                    raise ValueError("A layered observed map cannot supply a complete moving actor solid")
            else:
                layers = {None: geometry}
                for gap in geometry["query_gaps"]:
                    self._scene_gaps.append({"owner": owner_id, "reason": gap})
            # END no coarse layer is substituted for an absent selection.
            if owner_id != "background":
                path = deepcopy(partition["pose_path"])
                if path["motion_model"] is not None:
                    path["motion_model"] = MotionModel(**path["motion_model"])
                self._scene_owners[owner_id] = (partition, PosePath(**path))
                aliases = partition["native_geometry_names"]
                if aliases:
                    if (owner_id in self._scene_layers or len(aliases) != 1
                            or len(geometry["solids"]) != 1 or aliases[0] not in self._geometries):
                        raise ValueError("Fixed installation alias requires exactly one existing native solid")
                    if partition["moving"] or not partition["geometry_complete"] or partition["attachments"]:
                        raise ValueError("Native solid aliases cannot hide moving or partial owner geometry")
                    self._scene_native_aliases[owner_id] = aliases[0]
            # BEGIN retain each complete occupied inventory without taking a union.
            for layer, layer_geometry in layers.items():
                for record in layer_geometry["solids"]:
                    self._has_observed_tsdf |= record["kind"] == "tsdf"
                    # BEGIN a real fine-map replacement keeps identical other layers.
                    identity = (owner_id, layer, record["name"])
                    previous = previous_shapes.get(identity)
                    if previous is not None and previous_records[identity] == record:
                        shape, vertices, local_pose = (previous["shape"], previous["vertices"],
                                                       previous["local_pose"])
                    else:
                        shape, vertices, local_pose = _scene_shape(
                            record, max_host_rss_bytes=max_host_rss_bytes)
                    # END different source payloads cannot alias a former shape/tree.
                    if shape is None:
                        continue  # A validated positive-only field has no occupied solid.
                    self._scene_shapes.append({"owner": owner_id, "name": record["name"], "shape": shape,
                        "vertices": vertices, "local_pose": local_pose, "source": record["source"], "layer": layer})
            # END the selected inventory remains the target owner's own geometry.
        if len(self._scene_shapes) * (len(self._geometries)+len(self._scene_shapes)) > 100000:
            raise ValueError("Independent scene pair inventory exceeds the bounded CPU query budget")
        # BEGIN native authorization is bound to explicit names, never inferred materials.
        from .planning_scene import RobotOwnerContactPermission
        self._robot_owner_permissions = tuple(RobotOwnerContactPermission(**row)
            for row in scene_binding.get("robot_owner_permissions", ()))
        identities = [rule.rule_id for rule in self._robot_owner_permissions]
        if (len(set(identities)) != len(identities)
                or set(identities) & {row["rule_id"] for row in scene_binding["contacts"]}):
            raise ValueError("Robot-owner and material permission IDs must be unique")
        for rule in self._robot_owner_permissions:
            if (rule.target_owner not in self._scene_owners
                    or not set(rule.native_geometry_names) <= self._geometries.keys()):
                raise ValueError("Robot-owner permission requires bound owner and native robot geometry names")
        # END no MuJoCo/native physical contact policy is changed here.
        # BEGIN account once for the owned scene, decoded shapes and native cache.
        from .tsdf_geometry import ObservedTsdfGeometry
        self.scene_storage_nbytes = self.native_query_storage_nbytes + _owned_record_bytes(self._scene_binding) + sum(
            row["shape"].storage_nbytes if isinstance(row["shape"], ObservedTsdfGeometry)
            else (0 if row["vertices"] is None else np.asarray(row["vertices"]).nbytes * 4)
            for row in self._scene_shapes)
        # A state keeps at most two rows per physical pair (surface distance
        # and a streamed component summary), plus one failure witness. The
        # per-pair reserve covers 64 Coal contacts and each finite rule result;
        # search candidates and native LP/BVH scratch reuse scene-sized memory.
        # The independent fixed native-pair cache is already reserved above;
        # it is not duplicated as scene-sized scratch on every scene query.
        pair_bound = len(self._scene_shapes) * (len(self._geometries) + len(self._scene_shapes))
        # BEGIN reserve exact surface inputs/results before scene admission.
        # Each pair retains two pose byte strings, the complete native request,
        # one distance and two XYZ witnesses; 4 KiB also covers Python storage.
        self.scene_storage_nbytes += pair_bound * 4096
        # END no trajectory-length history or extra geometry is retained.
        # BEGIN avoid reserving a second and third copy of persistent TSDF rows.
        # iter_collision_components yields references to immutable cached Coal
        # shapes/vertices, including inside contact-region checks. A cache miss
        # creates one bounded component; the batch reserve below additionally
        # covers references and original world transforms. Spatial-tree search,
        # source candidates and native LP/BVH work fit the doubled non-cache
        # allowance; persistent source shapes are not duplicated per query.
        component_cache_bytes = sum(row["shape"].collision_cache_storage_nbytes
            for row in self._scene_shapes if isinstance(row["shape"], ObservedTsdfGeometry))
        self.scene_query_workspace_nbytes = (pair_bound * (131072 + 4096 * len(scene_binding["contacts"]))
            + 2 * (self.scene_storage_nbytes - self.native_query_storage_nbytes - component_cache_bytes)
            # BEGIN one live batch: shape reserve plus references/pose scratch.
            + (4096 * (4096 + 1024) if component_cache_bytes else 0))
            # END the reserve is per query, never multiplied by map components.
        # BEGIN surface jobs own separate bounded descriptors, solvers and stacks.
        from .scene_surface_batch import surface_batch_workspace_bytes
        surface_targets = sum(isinstance(row["shape"], ObservedTsdfGeometry)
                              and row["shape"].surface_shape is not None for row in self._scene_shapes)
        surface_pairs = surface_targets * (len(self._geometries) + len(self._scene_shapes))
        self.scene_query_workspace_nbytes += surface_batch_workspace_bytes(surface_pairs, self._request)
        # END this is additional to the resident surface-result cache above.
        # END full cache capacity remains charged in scene_storage_nbytes once.
        # END planning clones share this immutable scene allocation.

    def query_scene(self, robot_state, *, tool_pose, progress, finger_positions, state_kind,
                    owner_poses=None, nominal_owner_poses=None, condition_evidence=None,
                    active_robot_owner_permissions=frozenset(), robot_owner_permission_progress=None,
                    layer_selection=None, layer_selection_index=(0, 0), select_layers=False):
        """Query every robot/scene and moving-owner/other-owner solid pair.

        Nominal attached motion is an explicit conditional hypothesis. Actual
        moving-owner geometry needs current observation or affirmed attachment
        evidence. Neither an open/close command nor hidden simulator state is
        read. Static owner poses remain the explicitly bound scene assumption.
        Layered targets consume the original planner's archived body decision
        at the supplied (batch, horizon) index. This checker never reselects a
        layer using its independently computed double-precision FK. Only an
        explicit select_layers=True request creates a decision for a new state.
        """
        from copy import deepcopy
        import torch
        from scipy.spatial.transform import Rotation
        if state_kind not in {"actual", "nominal"} or not np.isfinite(progress) or progress < 0:
            raise ValueError("Independent scene query requires actual/nominal scope and bound progress")
        if not hasattr(self, "_scene_binding"):
            raise ValueError("Independent scene query has no bound scene model")
        # BEGIN actual permission is an executor-owned arrival latch, never a clock inference.
        active = active_robot_owner_permissions
        known = {rule.rule_id for rule in self._robot_owner_permissions}
        if active is None:
            if state_kind != "nominal":
                raise ValueError("Actual robot-owner permission requires an explicit active set")
            active = known
        elif (not isinstance(active, (set, frozenset))
                or any(not isinstance(identity, str) for identity in active) or not active <= known):
            raise ValueError("Active robot-owner permissions must name only rules in the bound scene")
        # Nominal checks share the planner's conditional law. Actual checks
        # require both the declared interval and this execution's fresh latch.
        permission_progress = progress if robot_owner_permission_progress is None else robot_owner_permission_progress
        if not np.isfinite(permission_progress) or permission_progress < 0:
            raise ValueError("Robot-owner permission requires finite nonnegative progress")
        if state_kind == "actual" and permission_progress != progress:
            raise ValueError("Actual permission must use the actual bound progress")
        permitted_rules = [rule for rule in self._robot_owner_permissions
            if rule.interval[0] <= permission_progress <= rule.interval[1]
            and (state_kind == "nominal" or rule.rule_id in active)]
        # END ordinary collision evidence below is still queried and archived.
        # BEGIN retain unpermitted distances before component evidence is reduced.
        # Whole-pair permissions also apply to separated pieces, whose normal
        # fast path has no reason to attach a contact-permission field. Native
        # installation pairs retain their independent required/excluded law.
        permitted_robot_pairs = {(name, rule.target_owner) for rule in permitted_rules
                                 for name in rule.native_geometry_names}
        clearance = {"minimum_recorded_distance_m": None,
                     "minimum_recorded_adjusted_distance_m": None,
                     "minimum_pair": None, "minimum_adjusted_pair": None,
                     "measured_count": 0, "omitted_ambiguous_surface_count": 0,
                     "omitted_finite_patch_subset_count": 0}
        # END this diagnostic never changes collision acceptance or enumeration.
        owner_poses, condition_evidence = owner_poses or {}, condition_evidence or {}
        # BEGIN consume this candidate's owner poses without claiming observations.
        nominal_owner_poses = nominal_owner_poses or {}
        if nominal_owner_poses and (state_kind != "nominal" or owner_poses):
            raise ValueError("Candidate owner poses require nominal scope without current observations")
        if not set(nominal_owner_poses) <= self._scene_owners.keys():
            raise ValueError("Candidate owner poses must name bound scene owners")
        # END attachment and release assumptions below remain explicit.
        binding = self._scene_binding
        gaps, unresolved, violations, rows = deepcopy(self._scene_gaps), [], [], []
        poses = {"background": np.eye(4)}
        uncertain_owners = set()
        for identity, (owner, path) in self._scene_owners.items():
            if not owner["geometry_complete"]:
                unresolved.append({"owner": identity, "kind": "partial_observed_geometry"})
            active = [relation for relation in owner["attachments"]
                      if progress >= relation["interval"][0] and
                      (progress < relation["interval"][1] or relation["release_condition_id"] is None)]
            if active and condition_evidence.get(active[-1]["condition_id"]) is False:
                violations.append({"kind": "current_scene_binding_invalid", "owner": identity,
                                   "condition_id": active[-1]["condition_id"], "reason": "attachment_hypothesis_refuted"})
            # BEGIN current observations do not erase a refuted declared release.
            released = [(index, relation) for index, relation in enumerate(owner["attachments"])
                        if progress >= relation["interval"][1] and relation["release_condition_id"] is not None]
            if not active and released and condition_evidence.get(released[-1][1]["release_condition_id"]) is False:
                violations.append({"kind": "current_scene_binding_invalid", "owner": identity,
                    "condition_id": released[-1][1]["release_condition_id"], "reason": "release_hypothesis_refuted"})
            # END the measured pose supplies geometry, not success of an effect.
            if identity in owner_poses:
                poses[identity] = _rigid_pose(owner_poses[identity], "current observed owner pose")
                continue
            if active:
                relation = active[-1]
                condition = condition_evidence.get(relation["condition_id"])
                # BEGIN missing Holding is a permitted hypothesis. Only an
                # explicit dependency can put a barrier in the executable curve.
                if condition is not True and state_kind == "actual":
                    uncertain_owners.add(identity)
                offset = np.eye(4)
                offset[:3, :3] = Rotation.from_quat(np.asarray(relation["quaternion_tool_owner"])[[1, 2, 3, 0]]).as_matrix()
                offset[:3, 3] = relation["position_tool_owner"]
                poses[identity] = tool_pose @ offset
                if condition is not True:
                    unresolved.append({"owner": identity, "condition_id": relation["condition_id"], "kind": "conditional_attachment"})
                # END nominal geometry remains checked under its declared hypothesis.
            else:
                if released:
                    index, relation = released[-1]
                    prediction = self._scene_release_predictions.get((identity, index))
                    # BEGIN the planner supplies release FK for this exact candidate.
                    if identity in nominal_owner_poses:
                        prediction = {"pose": _rigid_pose(nominal_owner_poses[identity]),
                                      "release_progress": relation["interval"][1]}
                    # END no shared release cache is mutated during batch validation.
                    if prediction is None:
                        gaps.append({"owner": identity, "reason": "same_candidate_release_fk_unbound"})
                        continue
                    poses[identity] = prediction["pose"].copy()
                    unresolved.append({"owner": identity, "kind": "predicted_release_pose",
                                       **{key: prediction[key] for key in ("time_s", "release_progress") if key in prediction},
                                       "condition_id": relation["release_condition_id"]})
                    if state_kind == "actual":
                        uncertain_owners.add(identity)
                else:
                    lo, hi = owner["pose_interval"]
                    p, q = path.sample(torch.tensor(np.clip((progress-lo)/(hi-lo), 0., 1.), dtype=torch.float64))
                    pose = np.eye(4)
                    pose[:3, 3] = p.numpy()
                    pose[:3, :3] = Rotation.from_quat(q.numpy()[[1, 2, 3, 0]]).as_matrix()
                    poses[identity] = pose
                    if owner["moving"] and state_kind == "actual":
                        uncertain_owners.add(identity)
                        unresolved.append({"owner": identity, "kind": "current_motion_prediction_unobserved"})
            # BEGIN match the same owner motion used by this candidate's costs.
            if identity in nominal_owner_poses:
                poses[identity] = _rigid_pose(nominal_owner_poses[identity], "candidate owner pose")
            # END supplied predictions never erase conditional effect evidence.
        actors = []
        materials = binding["native_geometry_materials"]
        unmodelled_geometries = set(robot_state.get("unmodelled_geometry_names", ()))
        # BEGIN keep explicitly owned static installation on the scene side.
        # bind_scene admits aliases only for complete static, unattached owners.
        # Their native solids remain in the complete robot pair query, and the
        # target loop below still verifies their pose and checks every moving
        # robot/payload actor against them. Adding the same installation as a
        # robot actor would invent static-installation/background collision
        # obligations absent from the shared planning scene contract. A fixed
        # robot base without this explicit owner identity remains an actor.
        static_installation = set(self._scene_native_aliases.values())
        for name, (_, shape, source, _) in self._geometries.items():
            if name in unmodelled_geometries or name in static_installation:
                continue
            pose = np.asarray(robot_state["geometry_poses"][name])
            actors.append({"owner": "robot", "name": name, "shape": shape, "pose": pose,
                           "vertices": self._solid_vertices[name], "material": materials.get(name), "source": source})
        # END ownership never removes a native self pair or installation target.
        targets = []
        for row in self._scene_shapes:
            if row["owner"] not in poses:
                continue
            current = {**row, "pose": poses[row["owner"]] @ row["local_pose"],
                       "material": materials.get(row["owner"]+"/"+row["name"])}
            alias = self._scene_native_aliases.get(row["owner"])
            if alias is not None:
                native_pose = np.asarray(robot_state["geometry_poses"][alias])
                if not np.allclose(current["pose"], native_pose, atol=1e-8, rtol=0):
                    gaps.append({"owner": row["owner"], "reason": "fixed_native_alias_pose_changed"})
                    continue
                # The supplied native model owns analytic shape and complete
                # pair policy. The planner's outward proxy is not substituted.
                current.update(shape=self._geometries[alias][1], vertices=self._solid_vertices[alias],
                               pose=native_pose, native_alias=alias)

            targets.append(current)
            if (row["owner"] in self._scene_owners and self._scene_owners[row["owner"]][0]["moving"]
                    and row["owner"] not in self._scene_layers):
                actors.append(current)
        # BEGIN validate the full independent body against its saved query envelope.
        if select_layers:
            if layer_selection is not None:
                raise ValueError("An archived layer decision and a new measured-state selection are mutually exclusive")
            layer_selection = self._select_state_layers(robot_state, actors, poses)
            layer_selection_index = (0, 0)
        decisions, selection_checks, selection_snapshot = self._consume_layer_selection(
            layer_selection, layer_selection_index, actors, poses, gaps, unmodelled_geometries)
        # END a query gap cannot silently switch either the layer or the body.
        # BEGIN only independent zero-surface facts execute before ordered consumption.
        from .scene_surface_batch import SurfaceDistanceBatch
        surface_queries = SurfaceDistanceBatch(actors, targets, decisions,
            self._request, self._scene_surface_distance_cache)
        # END state order, signed-component hints and contact decisions remain serial.
        checked, stopped_pairs, component_summaries = set(), set(), {}
        # BEGIN separate actual distance work from components covered by bounds.
        component_searches = {}
        # END one aggregate per physical body/target/layer bounds report storage.
        boundary = max(self.numerics["gjk_tolerance"], self.numerics["epa_tolerance"])
        # BEGIN exact local TSDF components replace expanded ESDF sample cubes.
        for actor_body in actors:
            # BEGIN reuse only this actor's unchanging native world transform.
            # A TSDF actor expands into new component rows with different local
            # frames; those rows must still construct their individual poses.
            actor_transform = None
            # END the lazy value never survives this one scene query.
            # BEGIN carry the unchanged physical-pair identity through its pieces.
            for first, second, pair_key in _scene_component_pairs(
                    actor_body, targets, decisions, boundary, rows, self._request, checked, stopped_pairs,
                    surface_queries, hints=self._scene_component_hints,
                    statistics=component_searches):
            # END component identities still remain on the individual rows.
                row_count = len(rows)
                # BEGIN a separated native prefix represents actual Coal calls.
                measured_count = second.get("_component_count", 1)
                # END near/contact components always retain their individual row.
                issue_counts = tuple(len(items) for items in (violations, gaps, unresolved))
                try:
            # END each component retains its original owner, body and layer identity.
                    if first["owner"] == second["owner"]:
                        continue
                    # BEGIN exactly one archived target layer applies to this complete body.
                    selected = decisions.get((first["owner"], first["name"], second["owner"]))
                    if second["owner"] in self._scene_layers:
                        if selected is None or second["layer"] != selected["layer"]:
                            continue
                    # END unselected coarse/fine solids never enter distance or permission queries.
                    if first["owner"] == "robot" and "native_alias" in second:
                        alias = second["native_alias"]
                        if first["name"] == alias:
                            continue  # Explicitly the identical native solid, not another owner.
                        native = [pair for pair in robot_state["pairs"] if
                                  {pair["first"], pair["second"]} == {first["name"], alias}]
                        if len(native) != 1:
                            gaps.append({"owner": second["owner"], "reason": "native_installation_pair_not_queried"})
                        else:
                            rows.append({"actor": "robot", "actor_geometry": first["name"],
                                         "target_owner": second["owner"], "target_geometry": alias,
                                         "native_pair_evidence": deepcopy(native[0]),
                                         "scope": "same explicitly bound native installation solid/pair"})
                            # BEGIN native installation distance obeys native pair policy.
                            if native[0]["policy"] == "required":
                                _record_unpermitted_clearance(clearance, native[0]["signed_distance_m"], 0.,
                                    first["name"], second["owner"], alias, "native_required_pair")
                            # END task contact permissions cannot waive a required native self pair.
                        continue
                    # BEGIN unchanged actors recur across many TSDF components.
                    if first is actor_body and actor_transform is not None:
                        first_transform = actor_transform
                    else:
                        first_transform = coal.Transform3s(first["pose"][:3, :3], first["pose"][:3, 3])
                        if first is actor_body:
                            actor_transform = first_transform
                    transforms = [first_transform,
                        coal.Transform3s(second["pose"][:3, :3], second["pose"][:3, 3])]
                    # END distance/contact requests and results remain fresh.
                    # BEGIN TSDF prefixes already used the same native leaf call.
                    distance = second.get("_prequeried_distance")
                    if distance is None:
                        distance = coal.DistanceResult()
                        coal.distance(first["shape"], transforms[0], second["shape"], transforms[1], self._request, distance)
                    # END ordinary analytic pairs still require their single query.
                    value = float(distance.min_distance)
                    if not np.isfinite(value):
                        raise RuntimeError("Independent scene distance returned nonfinite evidence")
                    # BEGIN omit detail allocation only for a discarded separated piece.
                    # Coal still checks every selected component. Beyond both
                    # the numerical boundary and measurement margin, this piece
                    # cannot create contacts, permissions or unresolved issues.
                    # The streaming summary retains the first minimum on ties;
                    # an equal or larger distance only increments its exact
                    # tested count. No nearest-point array/list is then needed.
                    margin = 0. if selected is None else selected["measurement_allowance_m"]
                    summary = component_summaries.get(pair_key)
                    if (summary is not None and value > max(boundary, margin)
                            and value >= rows[summary["index"]]["signed_distance_m"]):
                        rows[summary["index"]]["component_summary"]["tested"] += measured_count
                        # BEGIN a permitted nearest piece cannot hide an unpermitted next piece.
                        if (first["owner"] == "robot"
                                and (first["name"], second["owner"]) not in permitted_robot_pairs):
                            _record_unpermitted_clearance(clearance, value, margin,
                                first["name"], second["owner"], second["name"], "queried_source_component",
                                measured_count=measured_count)
                        # END preserve scalar evidence even when detail allocation is omitted.
                        continue
                    # END the finally summary sees no appended row in this case.
                    row = {"actor": first["owner"], "actor_geometry": first["name"],
                           "target_owner": second["owner"], "target_geometry": second["name"],
                           "actor_source": first["source"], "target_source": second["source"],
                           "signed_distance_m": value, "witness_first": np.asarray(distance.getNearestPoint1()).tolist(),
                           "witness_second": np.asarray(distance.getNearestPoint2()).tolist(),
                           "normal_first_to_second": np.asarray(distance.normal).tolist(),
                           "relation": "separated" if value > boundary else "penetrating" if value < -boundary else "numerical_boundary",
                           "permission": None, "contact_region_checks": []}
                    # BEGIN retain the exact local field piece used for this result.
                    for label, item in (("actor", first), ("target", second)):
                        if "tsdf_component_id" in item:
                            row[label+"_tsdf_component"] = list(item["tsdf_component_id"])
                    # END a component penetration is not a global union penetration depth.
                    # BEGIN retain geometric separation separately from sensor margin rejection.
                    if selected is not None:
                        row.update(layer=selected["layer"], body_id=selected["body_id"],
                                   measurement_allowance_m=margin,
                                   measurement_margin_violated=value <= margin,
                                   measurement_adjusted_distance_m=value-margin)
                    rows.append(row)
                    if value > max(boundary, margin):
                        continue
                    # END the measured margin never becomes material modeling allowance.
                    request, result = coal.CollisionRequest(), coal.CollisionResult()
                    request.enable_contact, request.num_max_contacts = True, 64
                    coal.collide(first["shape"], transforms[0], second["shape"], transforms[1], request, result)
                    row["contacts"] = [{"position": np.asarray(contact.pos).tolist(), "normal": np.asarray(contact.normal).tolist(),
                                        "penetration_depth": float(contact.penetration_depth)} for contact in result.getContacts()]
                    row["contact_report_truncated"] = result.numContacts() >= request.num_max_contacts
                    # BEGIN authorized whole-pair contact does not consume a local patch allowance.
                    permissions = [rule for rule in permitted_rules
                        if first["owner"] == "robot" and first["name"] in rule.native_geometry_names
                        and second["owner"] == rule.target_owner]
                    if permissions:
                        row["permission"] = permissions[0].rule_id
                        row["permission_scope"] = "explicit_robot_owner_pair"
                        row["permission_authorization"] = ("conditional_nominal_arrival" if state_kind == "nominal"
                                                            else "current_execution_arrival_latch")
                        continue
                    # END signed distance, witnesses and physical contacts remain raw evidence.
                    # BEGIN resolve a Coal numerical boundary using complete solid intersections.
                    # A sub-tolerance positive gap can be reported by GJK as zero
                    # distance/contact. Infeasibility of the complete convex
                    # intersection is a finished separation query, not missing
                    # contact evidence. Preserve Coal's raw values unchanged.
                    if abs(value) <= boundary:
                        overlap = _overlap_within_region(_world_vertices(first["vertices"], first["pose"]),
                            _world_vertices(second["vertices"], second["pose"]), (np.zeros((1, 3)), np.zeros(1)))
                        row["boundary_intersection_check"] = overlap
                        if overlap.get("completed") and overlap.get("intersection_empty"):
                            row["relation"] = "numerical_boundary_with_proven_empty_intersection"
                            if margin == 0.:
                                continue
                    # END no contact allowance is used to establish disjoint geometry.
                    # BEGIN unordered physical pairs may have either directed
                    # material permission. Enumeration order must not choose law.
                    possible_rules = [(actor, target, rule) for actor, target in ((first, second), (second, first))
                                      for rule in binding["contacts"] if rule["actor"] == actor["owner"]
                                      and rule["target_owner"] == target["owner"]
                                      and rule["interval"][0] <= progress <= rule["interval"][1]]
                    unbound_material = any(actor["material"] is None for actor, _, _ in possible_rules)
                    candidates = [(actor, target, rule) for actor, target, rule in possible_rules
                                  if rule["material_id"] == actor["material"]]
                    incomplete = False
                    for actor, target, rule in candidates:
                        actor_pose = tool_pose if actor["owner"] == "robot" else poses[actor["owner"]]
                        region, reason = _feature_region(rule, actor_pose, poses[target["owner"]], finger_positions)
                        evidence = {"rule_id": rule["rule_id"], "actor": actor["owner"], "target_owner": target["owner"], "reason": reason}
                        if reason == "finite_material_patch_region_query_incomplete":
                            incomplete = True
                        if region is not None and value >= -rule["modeling_allowance_m"]-rule["numerical_position_tolerance_m"]:
                            evidence.update(_overlap_within_region(_world_vertices(actor["vertices"], actor["pose"]),
                                                                   _world_vertices(target["vertices"], target["pose"]), region))
                            # BEGIN measurement uncertainty needs complete finite-region evidence.
                            # A separated original solid has an empty intersection,
                            # which cannot authorize its nonempty measurement band.
                            if margin > 0 and evidence.get("contained"):
                                raw = {key: value for key, value in evidence.items()
                                       if key not in {"rule_id", "actor", "target_owner", "reason"}}
                                evidence = {key: value for key, value in evidence.items()
                                            if key in {"rule_id", "actor", "target_owner", "reason"}}
                                evidence["raw_solid_overlap"] = raw
                                evidence.update(_overlap_within_region(
                                    _world_vertices(actor["vertices"], actor["pose"]),
                                    _world_vertices(target["vertices"], target["pose"]), region,
                                    first_margin_m=margin if actor is first else 0.,
                                    second_margin_m=margin if target is first else 0.))
                                evidence["measurement_allowance_m"] = margin
                                evidence["margin_geometry"] = "support_plane_outer_polytope_of_native_solid_plus_ball"
                            # END no unchanged modeling allowance authorizes unqueried expanded overlap.
                            if evidence.get("intersection_empty") and value < -boundary:
                                evidence.update(completed=False, contained=False,
                                    reason="penetrating_distance_disagrees_with_empty_solid_intersection")
                            incomplete |= not evidence["completed"]
                            if evidence.get("contained"):
                                row["permission"] = rule["rule_id"]
                        row["contact_region_checks"].append(evidence)
                        if row["permission"]:
                            break
                    # END both directions still require their exact material, patch and interval.
                    if row["permission"] is None:
                        issue = {"kind": ("independent_scene_measurement_margin" if value > boundary and margin > 0
                                          else "independent_scene_collision"), "pair_index": len(rows)-1,
                                 "actor": first["owner"], "target_owner": second["owner"], "evidence": deepcopy(row)}
                        if unbound_material:
                            gaps.append({**issue, "reason": "contact_material_native_geometry_binding_missing"})
                        elif incomplete:
                            gaps.append({**issue, "reason": "local_contact_region_query_incomplete"})
                        elif first["owner"] in uncertain_owners or second["owner"] in uncertain_owners:
                            unresolved.append({**issue, "kind": "conditional_scene_collision",
                                               "scope": "predicted owner geometry; not confirmed actual collision"})
                        else:
                            violations.append(issue)
                finally:
                    # BEGIN consume each fresh piece after its exact contact decision.
                    # This must precede _summarize_component_result: its minimum
                    # representative may be a permitted contact and would lose
                    # the nearest unpermitted component of that same body pair.
                    if len(rows) > row_count:
                        current = rows[-1]
                        # A locally permitted overlap does not measure the
                        # distance to the rest of this complete solid outside
                        # that finite patch. This remains true for an analytic
                        # cuboid with no TSDF components or query gaps.
                        if (current.get("actor") == "robot" and current.get("permission")
                                and current.get("permission_scope") != "explicit_robot_owner_pair"):
                            clearance["omitted_finite_patch_subset_count"] += 1
                        if (current.get("actor") == "robot" and "signed_distance_m" in current
                                and not current.get("permission")
                                and (current["actor_geometry"], current["target_owner"])
                                    not in permitted_robot_pairs):
                            _record_unpermitted_clearance(clearance, current["signed_distance_m"],
                                current.get("measurement_allowance_m", 0.),
                                current["actor_geometry"], current["target_owner"], current["target_geometry"],
                                "queried_source_component", measured_count=measured_count)
                    # END neither the saved witness nor its distance participates in acceptance.
                    # BEGIN stream complete component checks into bounded pair evidence.
                    if "tsdf_component_id" in first or "tsdf_component_id" in second:
                        _summarize_component_result(pair_key, rows, row_count,
                            (violations, gaps, unresolved), issue_counts,
                            component_summaries, stopped_pairs, measured_count=measured_count)
                    # END a failed pair can stop; an accepted pair checks every candidate.
        # Required contact additionally needs actual supplied target occupancy
        # in the finite region; an abstract patch can contain unobserved air.
        required_checks = []
        for rule in binding["contacts"]:
            if not rule["required"] or not rule["interval"][0] <= progress <= rule["interval"][1]:
                continue
            if rule["target_owner"] not in poses or (rule["actor"] != "robot" and rule["actor"] not in poses):
                gaps.append({"rule_id": rule["rule_id"], "reason": "required_contact_owner_pose_unbound"})
                continue
            actor_materials = [value for name, value in materials.items() if
                               (name in self._geometries if rule["actor"] == "robot" else name.startswith(rule["actor"]+"/"))]
            if rule["material_id"] not in actor_materials:
                gaps.append({"rule_id": rule["rule_id"], "reason": "required_material_native_geometry_binding_missing"})
                continue
            actor_pose = tool_pose if rule["actor"] == "robot" else poses[rule["actor"]]
            region, reason = _feature_region(rule, actor_pose, poses[rule["target_owner"]], finger_positions)
            if reason == "finite_material_patch_region_query_incomplete":
                gaps.append({"rule_id": rule["rule_id"], "reason": reason})
                continue
            if reason == "missing_named_finger_geometry" and state_kind == "nominal":
                unresolved.append({"rule_id": rule["rule_id"], "kind": "future_material_feature_geometry_unbound"})
                continue
            if region is not None:
                # BEGIN required occupancy consumes one complete material-rule layer.
                selected_layer = None
                if rule["target_owner"] in self._scene_layers:
                    if rule["actor"] == "robot":
                        names = {name for group in self._scene_body_groups
                            if set(group["sphere_indices"]) & set(rule["sphere_indices"])
                            for name in group["geometry_names"]}
                    else:
                        names = {actor["name"] for actor in actors if actor["owner"] == rule["actor"]}
                    keys = {(rule["actor"], name, rule["target_owner"]) for name in names}
                    missing = sorted(key[1] for key in keys if key not in decisions)
                    if not keys or missing:
                        gaps.append({"rule_id": rule["rule_id"], "reason": "required_contact_layer_selection_unbound",
                                     "missing_actor_geometries": missing})
                        continue
                    selected_layers = {decisions[key]["layer"] for key in keys}
                    if len(selected_layers) != 1:
                        gaps.append({"rule_id": rule["rule_id"], "reason": "required_contact_mixed_body_layers",
                                     "selected_layers": sorted(selected_layers)})
                        continue
                    selected_layer = next(iter(selected_layers))
                checks = [{**_solid_intersects_region(_world_vertices(target["vertices"], target["pose"]), region),
                           "layer": target["layer"], "target_geometry": target["name"]}
                          for target in targets if target["owner"] == rule["target_owner"]
                          and target["layer"] == selected_layer]
                # END no inactive field can supply required contact occupancy.
                found = any(check.get("completed") and check.get("intersects") for check in checks)
                required_checks.append({"rule_id": rule["rule_id"], "target_components": checks,
                                        "target_occupancy_in_region": found})
                if not found and any(not check.get("completed") for check in checks):
                    gaps.append({"rule_id": rule["rule_id"], "reason": "required_target_occupancy_query_incomplete",
                                 "target_components": checks})
                    continue
                if not found:
                    reason = "required_target_occupancy_absent_from_finite_region"
            if reason:
                issue = {"kind": "independent_required_contact", "rule_id": rule["rule_id"], "reason": reason}
                if rule["actor"] in uncertain_owners or rule["target_owner"] in uncertain_owners:
                    unresolved.append({**issue, "scope": "conditional owner model"})
                else:
                    violations.append(issue)
        if unmodelled_geometries:
            unresolved.append({"kind": "future_finger_geometry_unbound", "geometry_names": sorted(unmodelled_geometries)})
        # BEGIN surface measurements require separate permission and completeness labels.
        # The TSDF broad phase enumerates every potentially colliding signed
        # piece, not every distant component. A zero-surface distance is useful
        # observed-geometry evidence only alongside that completed containment
        # check. A finite contact patch cannot classify the surface's nearest
        # point, so do not call its distance unpermitted in that case.
        for row in rows:
            if (row.get("actor") != "robot" or not row.get("diagnostic_only")
                    or row.get("distance_scope") != "observed_zero_surface_only"
                    or (row["actor_geometry"], row["target_owner"]) in permitted_robot_pairs):
                continue
            # BEGIN an unrelated material cannot borrow another pad's permission.
            # Match the same directed actor/material identities used by the
            # actual piece query above. A missing material binding remains
            # ambiguous, because its contact test would itself be incomplete.
            local_permission = False
            for rule in binding["contacts"]:
                if not rule["interval"][0] <= progress <= rule["interval"][1]:
                    continue
                if rule["actor"] == "robot" and rule["target_owner"] == row["target_owner"]:
                    material = materials.get(row["actor_geometry"])
                elif rule["actor"] == row["target_owner"] and rule["target_owner"] == "robot":
                    material = materials.get(row["target_owner"]+"/"+row["target_geometry"])
                else:
                    continue
                if material is None or material == rule["material_id"]:
                    local_permission = True
                    break
            # END owner identity alone never suppresses an arm/environment distance.
            if local_permission:
                clearance["omitted_ambiguous_surface_count"] += 1
                continue
            selected = decisions.get(("robot", row["actor_geometry"], row["target_owner"]))
            margin = 0. if selected is None else selected["measurement_allowance_m"]
            _record_unpermitted_clearance(clearance, row["signed_distance_m"], margin,
                row["actor_geometry"], row["target_owner"], row["target_geometry"], row["distance_scope"])
        limitations = []
        if gaps:
            limitations.append("required_query_gaps")
        if stopped_pairs:
            limitations.append("component_enumeration_stopped_after_rejection")
        if unmodelled_geometries:
            limitations.append("unmodelled_robot_geometry")
        if unresolved:
            limitations.append("conditional_or_partial_scene_evidence")
        if self._has_observed_tsdf:
            limitations.append("observed_tsdf_local_components_not_global_solid_distance")
        if clearance["omitted_ambiguous_surface_count"]:
            limitations.append("surface_permission_not_resolved_by_finite_patch")
        if clearance["omitted_finite_patch_subset_count"]:
            limitations.append("finite_patch_unpermitted_subset_distance_unmeasured")
        clearance.update(complete_query=not gaps and not stopped_pairs,
            distance_complete=not gaps and not stopped_pairs and not unmodelled_geometries
                and not unresolved and not self._has_observed_tsdf
                and not clearance["omitted_finite_patch_subset_count"],
            limitations=limitations,
            scope="Recorded unpermitted robot/environment distances only; source static checks, coverage and full trajectory acceptance remain separate")
        # END missing or incomplete measurements never become infinite free clearance.
        return {"passed": not violations and not gaps, "violations": violations, "query_gaps": gaps,
                "unresolved_assumptions": unresolved, "pairs": rows, "state_kind": state_kind,
                "required_contact_checks": required_checks,
                "unpermitted_clearance": clearance,
                "component_searches": list(component_searches.values()),
                "layer_selection_checks": selection_checks,
                "layer_selection": selection_snapshot,
                # BEGIN expose authorization separately from geometry relation and task effects.
                "active_robot_owner_permissions": sorted(rule.rule_id for rule in permitted_rules),
                "robot_owner_permission_progress": float(permission_progress),
                # END a permission does not establish actual contact or Holding.
                "progress": float(progress), "observed_scene_checked": not gaps and not uncertain_owners and not unmodelled_geometries,
                "supplied_scene_models_checked": not gaps,
                "complete_scene_geometry": (not self._has_observed_tsdf
                    and all(owner["geometry_complete"] for owner, _ in self._scene_owners.values())),
                "continuous_collision_certified": False, "unknown_space_certified": False,
                "scope": "source solids and observed TSDF zero/negative components at this state; complete adapted contact intersections"}
    # END raw geometry evidence is retained independently of every task permission.

    # BEGIN explicit new-state selection and independent saved-envelope consumption.
    def _select_state_layers(self, robot_state, actors, poses):
        """Select once for a newly measured or independently sampled state.

        This opt-in path uses the same FineRegion.select routine as native
        sphere costs. Each body supplies the exact local sphere cover and its
        FK frame. Archived nominal replay must instead pass its original
        record, because float64 FK must not silently choose a different layer.
        """
        import torch
        if not self._scene_layers:
            return {}
        count = len(self._scene_binding["robot_sphere_materials"])
        world = np.empty((count, 4), dtype=np.float64)
        for group in self._scene_body_groups:
            local = np.asarray(group["local_spheres"], dtype=float)
            if (local.shape != (len(group["sphere_indices"]), 4) or not np.isfinite(local).all()
                    or np.any(local[:, 3] < 0)):
                raise ValueError("A new-state layer choice requires each body's complete local sphere cover")
            pose = _rigid_pose(robot_state["frame_poses"][group["frame"]], "body sphere FK frame")
            world[group["sphere_indices"], :3] = local[:, :3] @ pose[:3, :3].T+pose[:3, 3]
            world[group["sphere_indices"], 3] = local[:, 3]
        actor_spheres = {"robot": (world, self._scene_body_groups)}
        for identity in sorted({row["owner"] for row in actors}-{"robot"}):
            owner = self._scene_owners[identity][0]
            if not owner["geometry_complete"]:
                raise ValueError("Layer choice for a moving actor requires its complete physical geometry")
            local = np.asarray(owner["local_spheres"], dtype=float)
            if local.ndim != 2 or local.shape[1] != 4 or not len(local):
                raise ValueError("A moving actor needs its complete local sphere cover")
            pose = poses[identity]
            # Match PlanningScene's complete moving-owner cover exactly: its
            # declared cover error is a radius term before the sensor margin.
            values = np.column_stack((local[:, :3] @ pose[:3, :3].T+pose[:3, 3],
                                      local[:, 3]+float(owner["sphere_cover_error_m"])))
            actor_spheres[identity] = (values, ({"body_id": identity,
                "sphere_indices": list(range(len(local))), "geometry_names": [identity]},))
        result = {}
        for target in self._scene_layers:
            if target not in poses:
                continue
            region = self._scene_layers[target]["region"]
            inverse = np.linalg.inv(poses[target])
            for actor, (spheres, groups) in actor_spheres.items():
                if actor == target:
                    continue
                local = np.column_stack((spheres[:, :3] @ inverse[:3, :3].T+inverse[:3, 3], spheres[:, 3]))
                result[f"{actor}/{target}"] = region.select(torch.from_numpy(local)[None, None], groups).record()
        return result

    def _consume_layer_selection(self, selections, index, actors, poses, gaps, unmodelled):
        """Validate identity, dimensions and native-solid containment, not cost.

        A stored envelope includes the measurement margin already. Requiring
        the independently transformed solid plus the same margin to fit in it
        prevents float32/float64 FK disagreement from silently broadening the
        region. Strict excess is archived; no numerical padding is invented.
        """
        import torch
        from .collision_records import decode_array, slice_layer_selection
        if not self._scene_layers:
            return {}, [], {}
        if (not isinstance(index, (tuple, list)) or len(index) != 2
                or any(type(value) is not int or value < 0 for value in index)):
            raise ValueError("Layer selection index must be a nonnegative (batch, horizon) pair")
        decisions, checks, snapshot = {}, [], {}
        actor_owners = {row["owner"] for row in actors}
        for target, layer in self._scene_layers.items():
            if target not in poses:
                continue
            grid_world_inverse = np.linalg.inv(poses[target] @ layer["grid_pose"])
            for actor in sorted(actor_owners):
                if actor == target:
                    continue
                key = f"{actor}/{target}"
                record = selections.get(key) if isinstance(selections, dict) else None
                if record is None:
                    gaps.append({"actor": actor, "owner": target, "reason": "shared_layer_selection_missing", "selection_key": key})
                    continue
                groups = (self._scene_body_groups if actor == "robot" else
                    ({"body_id": actor, "geometry_names": [row["name"] for row in actors if row["owner"] == actor]},))
                try:
                    if not isinstance(record, dict) or set(record) != {
                            "body_ids", "body_fine", "influence_aabb_grid_m", "index_lower", "index_upper"}:
                        raise ValueError("A shared layer decision needs its complete typed record")
                    ids = record["body_ids"]
                    # BEGIN shared selection arrays have one strict lossless binary format.
                    fine = decode_array(record["body_fine"])
                    envelope = decode_array(record["influence_aabb_grid_m"])
                    lower, upper = decode_array(record["index_lower"]), decode_array(record["index_upper"])
                    # END old nested-list records cannot enter independent replay.
                    if (ids != [group["body_id"] for group in groups] or fine.ndim != 3 or fine.dtype != bool
                            or fine.shape[2] != len(groups) or envelope.shape != (*fine.shape, 2, 3)
                            or envelope.dtype != np.dtype("<f8")
                            or lower.shape != (*fine.shape, 3) or upper.shape != lower.shape
                            or lower.dtype != np.dtype("<i8") or upper.dtype != np.dtype("<i8")
                            or not np.isfinite(envelope).all() or np.any(envelope[..., 0, :] > envelope[..., 1, :])
                            or np.any(lower > upper) or index[0] >= fine.shape[0] or index[1] >= fine.shape[1]):
                        raise ValueError("unaligned layer record")
                except (KeyError, TypeError, ValueError):
                    gaps.append({"actor": actor, "owner": target, "reason": "shared_layer_selection_invalid", "selection_key": key})
                    continue
                # BEGIN authenticate the saved support evidence without selecting again.
                support = layer["region"].validate_selection_envelope(
                    torch.tensor(envelope[index[0], index[1]], dtype=torch.float64),
                    torch.tensor(lower[index[0], index[1]], dtype=torch.int64),
                    torch.tensor(upper[index[0], index[1]], dtype=torch.int64))
                # END the native query's choice remains frozen even when its evidence fails.
                # Archive only this state's complete decision. Repeating an
                # entire nominal trajectory for every Coal state is quadratic
                # storage and would exceed the bounded validation budget.
                snapshot[key] = slice_layer_selection({key: record}, index=tuple(index))[key]
                for column, group in enumerate(groups):
                    body_actors = [row for row in actors if row["owner"] == actor and row["name"] in group["geometry_names"]]
                    if not body_actors:
                        continue  # Explicit static installation aliases are scene targets only.
                    item = {"actor": actor, "target_owner": target, "body_id": group["body_id"],
                            "layer": "fine" if fine[index[0], index[1], column] else "coarse",
                            "measurement_allowance_m": layer["measurement_allowance_m"],
                            "saved_indices_match_envelope": bool(support["indices_match"][column]),
                            "saved_envelope_fine_supported": bool(support["fine_supported"][column]),
                            "influence_aabb_grid_m": envelope[index[0], index[1], column].tolist(),
                            "index_lower": lower[index[0], index[1], column].tolist(), "index_upper": upper[index[0], index[1], column].tolist()}
                    if (set(group["geometry_names"]) != {row["name"] for row in body_actors}
                            or set(group["geometry_names"]) & unmodelled):
                        gaps.append({**item, "reason": "complete_native_body_geometry_unbound"})
                        continue
                    bounds = [_solid_axis_bounds(row["vertices"], row["shape"], grid_world_inverse @ row["pose"])
                              for row in body_actors]
                    if any(bound is None for bound in bounds):
                        gaps.append({**item, "reason": "native_body_envelope_query_incomplete"})
                        continue
                    margin = layer["measurement_allowance_m"]
                    actual = np.stack((np.min([bound[0] for bound in bounds], axis=0)-margin,
                                       np.max([bound[1] for bound in bounds], axis=0)+margin))
                    saved = envelope[index[0], index[1], column]
                    excess = max(0., float(np.max(saved[0]-actual[0])), float(np.max(actual[1]-saved[1])))
                    item.update(native_influence_aabb_grid_m=actual.tolist(), maximum_envelope_excess_m=excess,
                                envelope_contained=excess == 0.)
                    checks.append(item)
                    if excess > 0.:
                        gaps.append({**item, "reason": "native_body_outside_shared_query_envelope"})
                        continue
                    # BEGIN a copied fine flag cannot manufacture source-supported space.
                    if not item["saved_indices_match_envelope"]:
                        gaps.append({**item, "reason": "shared_layer_indices_do_not_match_envelope"})
                        continue
                    if item["layer"] == "fine" and not item["saved_envelope_fine_supported"]:
                        gaps.append({**item, "reason": "shared_fine_layer_source_support_invalid"})
                        continue
                    # END invalid support is a query gap, never permission to use coarse.
                    for gap in layer["query_gaps"][item["layer"]]:
                        gaps.append({**item, "reason": gap})
                    for row in body_actors:
                        decisions[actor, row["name"], target] = item
                # A native actor absent from the sphere-body contract cannot
                # inherit another body's decision or a coarse default.
                represented = {name for group in groups for name in group["geometry_names"]}
                for row in actors:
                    if row["owner"] == actor and row["name"] not in represented:
                        gaps.append({"actor": actor, "actor_geometry": row["name"], "owner": target,
                                     "reason": "native_actor_body_identity_unbound"})
        return decisions, checks, snapshot
    # END records from native queries remain authoritative even at ROI boundaries.

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
            # BEGIN query permitted mechanical contacts without deleting evidence.
            if pair.policy not in ("required", "excluded", "diagnostic", "mechanical_contact"):
                raise ValueError("Pair policy must be required, excluded, diagnostic, or mechanical_contact")
            # END permission affects acceptance, never the underlying distance.
            _name(pair.source, "pair policy source")
            identities.add(identity)
            pair_list.append(pair)
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

    # BEGIN bounded native many-pair queries with explicitly pristine history.
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
    # END native scene-component distance/contact evaluation stays independent.
# END independent named robot FK and native-solid numerical diagnostics.

# BEGIN independent full-solid scene geometry and local material permissions.
def _pose7(value):
    """Native scene poses use translation plus scalar-first quaternion."""
    from scipy.spatial.transform import Rotation
    row = np.asarray(value, float)
    if row.shape != (7,) or not np.isfinite(row).all() or not np.isclose(np.linalg.norm(row[3:]), 1., atol=1e-6):
        raise ValueError("Scene geometry requires a finite normalized xyz/wxyz pose")
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_quat(row[[4, 5, 6, 3]]).as_matrix()
    pose[:3, 3] = row[:3]
    return pose


def _box_vertices(dimensions):
    from itertools import product
    return np.asarray(list(product((-1., 1.), repeat=3))) * np.asarray(dimensions) / 2


def _world_vertices(vertices, pose):
    # BEGIN preserve a component's affine dimension and observed-field identity.
    from .tsdf_geometry import ObservedTsdfGeometry
    if isinstance(vertices, ObservedTsdfGeometry):
        return {"tsdf": vertices, "pose": pose}
    if isinstance(vertices, dict):
        return {"tsdf_component": np.asarray(vertices["tsdf_component"]) @ pose[:3, :3].T + pose[:3, 3]}
    # END a triangle/line/point is not promoted to a filled bounding box.
    return None if vertices is None else np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]


# BEGIN bounded broad phase over the unchanged local TSDF components.
def _owned_record_bytes(value):
    """Conservatively count the supported in-memory record without encoding it."""
    from sys import getsizeof
    size = getsizeof(value)
    if isinstance(value, dict):
        size += sum(_owned_record_bytes(key)+_owned_record_bytes(item) for key, item in value.items())
    elif isinstance(value, (list, tuple)):
        size += sum(_owned_record_bytes(item) for item in value)
    elif isinstance(value, np.ndarray):
        size += value.nbytes
    return size


# BEGIN bounded scalar clearance records survive per-pair contact summarization.
def _record_unpermitted_clearance(summary, distance, margin, actor_geometry,
                                 target_owner, target_geometry, distance_scope, *, measured_count=1):
    """Accumulate one already queried, unpermitted robot/environment distance.

    The caller owns policy and complete-solid checks. No additional Coal query
    or numerical allowance is introduced here. Raw and measurement-adjusted
    minima can have different witnesses and retain separate identities. This
    record is bounded independently of the number of TSDF components. Distance
    finiteness and margin validity were already checked at the query boundary;
    losing a minimum update allocates no new witness record or NumPy array.
    """
    # BEGIN a pure separated prefix has one minimum and many measured leaves.
    summary["measured_count"] += measured_count
    # END its fixed pair/margin preserves both raw and adjusted minima.
    raw_minimum = (summary["minimum_recorded_distance_m"] is None
                   or distance < summary["minimum_recorded_distance_m"])
    adjusted = distance-margin
    adjusted_minimum = (summary["minimum_recorded_adjusted_distance_m"] is None
                        or adjusted < summary["minimum_recorded_adjusted_distance_m"])
    if not raw_minimum and not adjusted_minimum:
        return
    pair = {"actor_geometry": actor_geometry, "target_owner": target_owner,
            "target_geometry": target_geometry, "distance_scope": distance_scope}
    if raw_minimum:
        summary["minimum_recorded_distance_m"] = distance
        summary["minimum_pair"] = pair
    if adjusted_minimum:
        summary["minimum_recorded_adjusted_distance_m"] = adjusted
        summary["minimum_adjusted_pair"] = pair
# END stable first-minimum ties preserve source enumeration order.


# BEGIN the component enumerator supplies the stable whole-body pair key once.
def _summarize_component_result(key, rows, previous_count, issues, issue_counts, summaries, stopped,
                                *, measured_count=1):
    """Retain one decisive witness and complete counts, without storing all pieces.

    A rejected physical pair needs no second witness. An accepted pair still
    visits every possible intersection and proves each permission. Replacing
    its representative row does not change any earlier pair's stable index.
    """
    if len(rows) == previous_count:
        return
    current = rows[-1]
    # Conditional collisions remain allowed unresolved hypotheses. They must
    # not stop a later component from exposing a mandatory query gap.
    failed = any(len(items) > count for items, count in zip(issues[:2], issue_counts[:2]))
    if key not in summaries:
        index, tested, permitted = len(rows)-1, 0, 0
        summaries[key] = {"index": index, "conditional_index": None, "conditional_count": 0}
    else:
        index = summaries[key]["index"]
        previous = rows[index]
        counts = previous["component_summary"]
        tested, permitted = counts["tested"], counts["permitted"]
        rows.pop()
        if failed or current["signed_distance_m"] < previous["signed_distance_m"]:
            rows[index] = current
    rows[index]["component_summary"] = {
        "tested": tested+measured_count, "permitted": permitted+int(current.get("permission") is not None),
        "stopped_after_pair_failure": failed,
        "scope": "candidate components covered by exact queries or conservative bounds unless a decisive pair failure already rejects the state"}
    for items, count in zip(issues, issue_counts):
        for issue in items[count:]:
            if "pair_index" in issue:
                issue["pair_index"] = index
    unresolved, start = issues[2], issue_counts[2]
    summary = summaries[key]
    if len(unresolved) > start:
        summary["conditional_count"] += len(unresolved)-start
        if summary["conditional_index"] is None:
            summary["conditional_index"] = start
            del unresolved[start+1:]
        else:
            del unresolved[start:]
        unresolved[summary["conditional_index"]]["component_issue_count"] = summary["conditional_count"]
    if failed:
        stopped.add(key)
# END summary ordering, counts and first-minimum witness selection are unchanged.


def _transform_bounds(bounds, pose):
    """Transform all eight AABB corners; the result is an enclosing AABB."""
    lower, upper = bounds
    corners = _box_vertices(np.asarray(upper)-lower)+(np.asarray(upper)+lower)/2
    points = _world_vertices(corners, pose)
    return points.min(axis=0), points.max(axis=0)


def _component_rows(row, world_bounds, margin):
    """Yield every field component that can meet the other complete body.

    The expansion is only a broad-phase search radius. Coal still receives the
    original component vertices and contact permission uses the same original
    finite material region. Source body and selected target layer names survive
    component expansion so neither filtering nor permission becomes per-mesh.
    """
    from .tsdf_geometry import ObservedTsdfGeometry
    geometry = row["shape"]
    if not isinstance(geometry, ObservedTsdfGeometry):
        yield row
        return
    local_bounds = None if world_bounds is None else _transform_bounds(
        (world_bounds[0]-margin, world_bounds[1]+margin), np.linalg.inv(row["pose"]))
    # BEGIN reuse fixed component geometry while recomputing the current pose.
    # Cached points already use the point/segment primitive's local frame;
    # applying row.pose @ local once preserves complete-overlap witnesses.
    for kind, index, shape, local, points in geometry.iter_collision_components(local_bounds):
        yield {**row, "shape": shape, "pose": row["pose"] @ local,
               "vertices": {"tsdf_component": points}, "tsdf_component_id": (kind, index)}
    # END the selected owner/layer and all query-specific fields remain intact.


def _scene_component_pairs(actor, targets, decisions, boundary, evidence_rows, request,
                           visited, stopped, surface_queries, *, hints, statistics):
    """Query a surface distance and all potentially intersecting signed pieces.

    A positive triangle distance cannot reject the negative-volume containment
    check. The BVH row is therefore diagnostic; acceptance below uses every
    candidate zero/negative component, including lower-dimensional samples.
    """
    from .tsdf_geometry import ObservedTsdfGeometry
    # BEGIN collector and consumer share selection/deduplication for every pair.
    from .scene_surface_batch import selected_body_pairs
    for target, key, selected in selected_body_pairs(actor, targets, decisions, visited):
        # END non-surface pairs retain the same visited effect as surface pairs.
        margin = max(boundary, 0. if selected is None else selected["measurement_allowance_m"])
        a_tsdf = isinstance(actor["shape"], ObservedTsdfGeometry)
        b_tsdf = isinstance(target["shape"], ObservedTsdfGeometry)
        if not a_tsdf and b_tsdf and target["shape"].surface_shape is not None:
            # BEGIN consume prefetched evidence at the original error/cache position.
            value, witness_first, witness_second = surface_queries.consume(actor, target)
            # END fresh report lists still isolate public evidence from the cache.
            evidence_rows.append({"actor": actor["owner"], "actor_geometry": actor["name"],
                "target_owner": target["owner"], "target_geometry": target["name"],
                "layer": target["layer"], "signed_distance_m": value,
                "distance_scope": "observed_zero_surface_only",
                "witness_first": witness_first.tolist(),
                "witness_second": witness_second.tolist(),
                "diagnostic_only": True, "scope": "observed zero-set triangles; signed pieces checked separately"})
        target_bounds = _solid_axis_bounds(target["vertices"], target["shape"], target["pose"])
        for first in _component_rows(actor, target_bounds, margin):
            if key in stopped:
                break
            first_bounds = _solid_axis_bounds(first["vertices"], first["shape"], first["pose"])
            # BEGIN query TSDF leaves in bounded, order-preserving native runs.
            for second in _target_component_runs(first, target, first_bounds, margin, request,
                                                 hints=hints, statistics=statistics):
                if key in stopped:
                    break
                # BEGIN one identity per body pair, independent of piece count.
                yield first, second, key
                # The next generator step performs distances, so stop before
                # requesting it when Python has rejected this physical pair.
                if key in stopped:
                    break
                # END callers reuse the exact same key for streamed evidence.
            # END Python still decides whether any near/contact item may continue.
# END unknown regions never acquire a fabricated boundary or filled cell.


# BEGIN spatial search retains original leaf results and source-order consumption.
def _target_component_runs(first, target, world_bounds, margin, request, *, hints, statistics):
    """Yield measured separated-prefix minima and every required near component.

    Static spatial nodes may exclude leaves using conservative geometric bounds.
    Remaining raw results retain the original source order and contact policy;
    counts explicitly distinguish actual queries from node-covered components.
    """
    from .tsdf_geometry import ObservedTsdfGeometry
    from .tsdf_spatial_query import iter_component_distances
    geometry = target["shape"]
    if not isinstance(geometry, ObservedTsdfGeometry):
        yield target
        return
    local_bounds = None if world_bounds is None else _transform_bounds(
        (world_bounds[0]-margin, world_bounds[1]+margin), np.linalg.inv(target["pose"]))
    identity = (first["owner"], first["name"], target["owner"], target["name"], target["layer"])
    saved = hints.get(identity)
    previous = saved[1] if saved is not None and saved[0] is geometry else -1
    work = statistics.setdefault(identity, {
        "actor": first["owner"], "actor_geometry": first["name"],
        "target_owner": target["owner"], "target_geometry": target["name"], "layer": target["layer"],
        "candidate_count": 0, "bound_pruned_count": 0, "exact_leaf_query_count": 0,
        "seed_query_count": 0, "prior_hint_query_count": 0, "node_bound_count": 0,
        "retained_pass_count": 0,
        "query_count": 0, "completed_queries": 0, "completed": False,
        "workspace_nbytes_upper": 0,
        "scope": "physical component work; a seed can be queried before its source-order evidence is consumed"})
    best, best_id = float("inf"), None
    for flat, component, pose, distance, measured in iter_component_distances(
            geometry, first["shape"], first["pose"], target["pose"], local_bounds, margin, request,
            previous_component=previous, statistics=work):
        value = float(distance.min_distance)
        if value < best or (value == best and (best_id is None or flat < best_id)):
            best, best_id = value, flat
            hints[identity] = (geometry, flat)
        kind, component_id, shape, _, points = component
        yield {**target, "shape": shape, "pose": pose,
               "vertices": {"tsdf_component": points}, "tsdf_component_id": (kind, component_id),
               "_prequeried_distance": distance, "_component_count": measured}
# END a next-state hint is only an ID; no old distance or permission survives.


# BEGIN exact body support bounds in the selected field's own grid axes.
def _solid_axis_bounds(vertices, shape, pose):
    """Bound complete source solids or observed TSDF pieces in supplied axes."""
    from .tsdf_geometry import ObservedTsdfGeometry
    if isinstance(vertices, ObservedTsdfGeometry):
        return _transform_bounds(vertices.bounds, pose)
    if isinstance(vertices, dict):
        points = _world_vertices(vertices, pose)["tsdf_component"]
        return points.min(axis=0), points.max(axis=0)
    if vertices is not None:
        points = _world_vertices(vertices, pose)
        return points.min(axis=0), points.max(axis=0)
    if isinstance(shape, coal.Cylinder):
        rotation = pose[:3, :3]
        extent = shape.radius*np.linalg.norm(rotation[:, :2], axis=1)+shape.halfLength*np.abs(rotation[:, 2])
    elif isinstance(shape, coal.Sphere):
        extent = np.full(3, shape.radius)
    else:
        return None
    return pose[:3, 3]-extent, pose[:3, 3]+extent
# END no corner tessellation substitutes for a cylinder's circular support.


def _scene_shape(record, *, max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
    """Build one explicitly described filled volume, with no mesh-hull fallback."""
    kind = record["kind"]
    vertices = None
    # BEGIN explicit observed-field geometry keeps components and sign evidence.
    if kind == "tsdf":
        from .tsdf_geometry import ObservedTsdfGeometry
        shape = ObservedTsdfGeometry(record, max_host_rss_bytes=max_host_rss_bytes)
        if shape.bounds is None:
            return None, None, _pose7(record["pose"])
        return shape, shape, _pose7(record["pose"])
    # END no whole-environment convex hull or ESDF cube is constructed.
    if kind == "mesh":
        vertices = np.asarray(record["vertices"], float)
        shape = _convex_solid(vertices)
    else:
        sizes = np.asarray(record["dimensions"], float)
        count = {"cuboid": 3, "cylinder": 2, "sphere": 1}.get(kind)
        if count is None or sizes.shape != (count,) or not np.isfinite(sizes).all() or np.any(sizes <= 0):
            raise ValueError("Independent scene primitives need explicit positive native dimensions")
        shape = {"cuboid": coal.Box, "cylinder": coal.Cylinder, "sphere": coal.Sphere}[kind](*sizes)
        if kind == "cuboid":
            vertices = _box_vertices(sizes)
    return shape, vertices, _pose7(record["pose"])


def _feature_region(rule, actor_pose, target_pose, finger_positions):
    """Express the complete material polygon and allowed patch in world axes.

    These are the same ContactRule data consumed by generation. This CPU
    implementation never reads its sphere masks or differentiable residuals.
    """
    from scipy.spatial import ConvexHull
    points = np.asarray(rule["feature_points"], float)
    mapping = rule["feature_joint_translation"]
    if mapping is not None:
        if finger_positions is None:
            return None, "missing_named_finger_geometry"
        points = points + np.einsum("vcj,j->vc", np.asarray(mapping, float), finger_positions)
    world = _world_vertices(points, actor_pose)
    normal = actor_pose[:3, :3] @ np.asarray(rule["feature_normal"], float)
    axes = target_pose[:3, :3] @ np.asarray(rule["patch_axes"], float)
    origin = target_pose[:3, :3] @ np.asarray(rule["patch_origin"], float) + target_pose[:3, 3]
    tolerance = float(rule["numerical_position_tolerance_m"])
    angle = np.arctan2(np.linalg.norm(np.cross(normal, axes[:, 2])), -normal @ axes[:, 2])
    if angle > rule["normal_tolerance_rad"] + rule["numerical_angle_tolerance_rad"]:
        return None, "material_normal_outside_permission"
    # BEGIN a larger pad may contact only a finite part of the target patch.
    # Do not demand containment of the whole feature. The complete solid
    # intersection below must lie in every supplied material/patch halfspace.
    # SVD supplies a basis for this supplied planar polygon. ConvexHull in 2D
    # produces all its edge inequalities, including a nonrectangular feature.
    basis = np.linalg.svd(world-world[0], full_matrices=True)[2][:2].T
    local = (world-world[0]) @ basis
    edges = ConvexHull(local).equations
    rows, bounds = [], []
    for edge in edges:
        axis = basis @ edge[:2]
        rows.append(axis)
        bounds.append(float(axis @ world[0] - edge[2] + tolerance))
    allowance = float(rule["modeling_allowance_m"])
    for axis in (normal, -normal):
        rows.append(axis)
        bounds.append(float(axis @ world[0] + allowance + tolerance))
    for column, half in enumerate(rule["patch_half_size"]):
        for sign in (-1., 1.):
            axis = sign * axes[:, column]
            rows.append(axis)
            bounds.append(float(axis @ origin + half + tolerance))
    # A permitted model overlap must remain in the local target boundary band.
    for sign in (-1., 1.):
        axis = sign * axes[:, 2]
        rows.append(axis)
        bounds.append(float(axis @ origin + allowance + tolerance))
    # Evaluate gap on the corresponding point projected onto the ACTUAL
    # material plane, rather than on unrelated corners outside the patch.
    height = float(normal @ world[0])
    cosine = float(axes[:, 2] @ normal)
    gap_axis = axes[:, 2]-cosine*normal
    gap_offset = cosine*height-float(axes[:, 2] @ origin)
    for sign, limit in ((1., rule["gap_bounds"][1]), (-1., -rule["gap_bounds"][0])):
        rows.append(sign*gap_axis)
        bounds.append(float(limit-sign*gap_offset+tolerance))
    region = np.asarray(rows), np.asarray(bounds)
    from scipy.optimize import linprog
    feasible = linprog(np.zeros(3), A_ub=region[0], b_ub=region[1], bounds=[(None, None)]*3,
                      method="highs", options={"primal_feasibility_tolerance": 1e-9,
                                               "dual_feasibility_tolerance": 1e-9})
    if feasible.status == 2:
        return None, "finite_material_patch_gap_region_empty"
    if not feasible.success:
        return None, "finite_material_patch_region_query_incomplete"
    return region, None
    # END only the finite overlap, not the entire material face, needs permission.


# BEGIN required contact needs a complete occupied target in the allowed region.
def _solid_intersects_region(vertices, region):
    """Check existential intersection independently of planner sphere costs."""
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    options = {"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9}
    if vertices is None:
        return {"completed": False, "reason": "required_curved_target_needs_complete_region_adapter"}
    # BEGIN required contact needs the actual zero set, never unknown-facing caps.
    if isinstance(vertices, dict) and "tsdf" in vertices:
        from .contact_geometry import convex_component_intersects_region
        geometry, pose = vertices["tsdf"], vertices["pose"]
        lower, upper = np.empty(3), np.empty(3)
        for column, axis in enumerate(np.eye(3)):
            for sign in (-1., 1.):
                bound = linprog(sign*axis, A_ub=region[0], b_ub=region[1],
                    bounds=[(None, None)]*3, method="highs", options=options)
                if not bound.success:
                    return {"completed": bound.status == 2, "intersects": False,
                            "reason": "required_region_empty" if bound.status == 2 else "required_region_bounds_incomplete"}
                (lower if sign == 1. else upper)[column] = bound.x[column]
        local_bounds = _transform_bounds((lower, upper), np.linalg.inv(pose))
        count = 0
        for _, _, points in geometry.iter_components(local_bounds, surface=True, negative=False):
            count += 1
            result = convex_component_intersects_region(_world_vertices(points, pose), region)
            if not result["completed"] or result["intersects"]:
                return {**result, "component_queries": count, "geometry_scope": "observed_tsdf_zero_set"}
        return {"completed": True, "intersects": False, "component_queries": count,
                "geometry_scope": "observed_tsdf_zero_set"}
    if isinstance(vertices, dict) and "tsdf_component" in vertices:
        from .contact_geometry import convex_component_intersects_region
        return convex_component_intersects_region(vertices["tsdf_component"], region)
    # END negative pieces remain collision evidence without inventing contact surfaces.
    planes = ConvexHull(vertices).equations
    result = linprog(np.zeros(3), A_ub=np.concatenate((region[0], planes[:, :3])),
                     b_ub=np.r_[region[1], -planes[:, 3]], bounds=[(None, None)]*3,
                     method="highs", options=options)
    if result.status == 2:
        return {"completed": True, "intersects": False, "method": "complete_target_region_feasibility"}
    if not result.success:
        return {"completed": False, "reason": "required_target_region_query_incomplete", "solver_status": int(result.status)}
    return {"completed": True, "intersects": True, "witness": result.x.tolist(),
            "method": "complete_target_region_feasibility", "numerical_certificate": False}
# END permission-only rules never invent required occupancy or contact force.


def _overlap_within_region(first_vertices, second_vertices, region, *, first_margin_m=0., second_margin_m=0.):
    """Use SciPy/HiGHS support LPs over the entire convex solid intersection.

    Every inequality of the finite material/patch region is maximized over
    the entire intersection. One Coal contact witness is insufficient here.
    A proven infeasible intersection is empty and contained vacuously. Other
    failed queries remain gaps; no contact allowance or enlarged epsilon
    turns an uncompleted whole-region check into permission.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    if first_vertices is None or second_vertices is None:
        return {"completed": False, "reason": "curved_contact_requires_complete_local_region_adapter"}
    # BEGIN lower-dimensional TSDF pieces use complete barycentric support LPs.
    first_component = isinstance(first_vertices, dict)
    second_component = isinstance(second_vertices, dict)
    if first_component or second_component:
        from .contact_geometry import (convex_component_overlap_within_region,
                                       component_component_overlap_within_region)
        if first_component and second_component:
            return component_component_overlap_within_region(
                first_vertices["tsdf_component"], second_vertices["tsdf_component"], region,
                first_margin_m=first_margin_m, second_margin_m=second_margin_m)
        solid = second_vertices if first_component else first_vertices
        component = first_vertices["tsdf_component"] if first_component else second_vertices["tsdf_component"]
        solid_margin = second_margin_m if first_component else first_margin_m
        component_margin = first_margin_m if first_component else second_margin_m
        if component_margin:
            return component_component_overlap_within_region(
                solid, component, region, first_margin_m=solid_margin, second_margin_m=component_margin)
        planes = ConvexHull(solid).equations
        return convex_component_overlap_within_region(
            (planes[:, :3], -planes[:, 3]), component, region, actor_margin=solid_margin)
    # END all component vertices participate; contact witness limits are irrelevant.
    # BEGIN each shifted plane contains the native solid plus an isotropic ball.
    # Their intersection is a conservative outer polytope, including at sharp
    # corners. Proving its entire target overlap lies inside the unchanged
    # material region proves containment for the measured uncertainty band.
    first_planes, second_planes = ConvexHull(first_vertices).equations, ConvexHull(second_vertices).equations
    equations = np.concatenate((first_planes, second_planes))
    A = equations[:, :3]
    b = np.r_[-first_planes[:, 3]+first_margin_m*np.linalg.norm(first_planes[:, :3], axis=1),
              -second_planes[:, 3]+second_margin_m*np.linalg.norm(second_planes[:, :3], axis=1)]
    # END sensor margin does not alter material modeling or patch inequalities.
    maxima = []
    for axis, bound in zip(*region):
        result = linprog(-axis, A_ub=A, b_ub=b, bounds=[(None, None)]*3, method="highs",
                         options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            # BEGIN distinguish a completed empty-set proof from a failed solver query.
            if result.status == 2:
                return {"completed": True, "contained": True, "intersection_empty": True,
                        "solver_status": int(result.status), "method": "empty_complete_convex_intersection",
                        "numerical_certificate": False}
            # END infeasibility is a geometric result; other statuses remain incomplete.
            return {"completed": False, "reason": "overlap_support_query_incomplete", "solver_status": int(result.status)}
        value = float(axis @ result.x-bound)
        maxima.append(value)
        if value > 0.:
            return {"completed": True, "contained": False, "maximum_constraint_excess_m": value,
                    "outside_witness": result.x.tolist(), "method": "whole_convex_intersection_support"}
    return {"completed": True, "contained": True, "maximum_constraint_excess_m": max(maxima),
            "method": "whole_convex_intersection_support", "numerical_certificate": False}
# END filled-solid intersection checks do not infer whole-pair task permission.
