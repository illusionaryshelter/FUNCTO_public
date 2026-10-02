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

    # BEGIN expose immutable geometry identities for a complete acceptance policy.
    @property
    def geometry_names(self):
        """Stable native-solid order; an acceptance layer must cover all pairs."""
        return tuple(self._geometries)

    @property
    def geometry_sources(self):
        return {name: value[2] for name, value in self._geometries.items()}
    # END expose immutable geometry identities for a complete acceptance policy.

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
    def bind_scene(self, scene_binding):
        """Build once from the owned finite scene snapshot, without GPU queries."""
        from copy import deepcopy
        from manipulation.planning_problem import PosePath
        from manipulation.motion import MotionModel
        self._scene_binding = deepcopy(scene_binding)
        self._scene_owners = {}
        self._scene_shapes = []
        self._scene_gaps = []
        self._scene_native_aliases = {}
        self._scene_release_predictions = {}
        for partition in [scene_binding["background"], *scene_binding["owners"]]:
            owner_id = partition.get("owner_id", "background")
            for gap in partition["geometry"]["query_gaps"]:
                self._scene_gaps.append({"owner": owner_id, "reason": gap})
            if owner_id != "background":
                path = deepcopy(partition["pose_path"])
                if path["motion_model"] is not None:
                    path["motion_model"] = MotionModel(**path["motion_model"])
                self._scene_owners[owner_id] = (partition, PosePath(**path))
                aliases = partition["native_geometry_names"]
                if aliases:
                    if len(aliases) != 1 or len(partition["geometry"]["solids"]) != 1 or aliases[0] not in self._geometries:
                        raise ValueError("Fixed installation alias requires exactly one existing native solid")
                    if partition["moving"] or not partition["geometry_complete"] or partition["attachments"]:
                        raise ValueError("Native solid aliases cannot hide moving or partial owner geometry")
                    self._scene_native_aliases[owner_id] = aliases[0]
            for record in partition["geometry"]["solids"]:
                shape, vertices, local_pose = _scene_shape(record)
                self._scene_shapes.append({"owner": owner_id, "name": record["name"], "shape": shape,
                                           "vertices": vertices, "local_pose": local_pose, "source": record["source"]})
        if len(self._scene_shapes) * (len(self._geometries)+len(self._scene_shapes)) > 100000:
            raise ValueError("Independent scene pair inventory exceeds the bounded CPU query budget")

    def query_scene(self, robot_state, *, tool_pose, progress, finger_positions, state_kind,
                    owner_poses=None, condition_evidence=None):
        """Query every robot/scene and moving-owner/other-owner solid pair.

        Nominal attached motion is an explicit conditional hypothesis. Actual
        moving-owner geometry needs current observation or affirmed attachment
        evidence. Neither an open/close command nor hidden simulator state is
        read. Static owner poses remain the explicitly bound scene assumption.
        """
        from copy import deepcopy
        import torch
        from scipy.spatial.transform import Rotation
        if state_kind not in {"actual", "nominal"} or not np.isfinite(progress) or progress < 0:
            raise ValueError("Independent scene query requires actual/nominal scope and bound progress")
        if not hasattr(self, "_scene_binding"):
            raise ValueError("Independent scene query has no bound scene model")
        owner_poses, condition_evidence = owner_poses or {}, condition_evidence or {}
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
                    if prediction is None:
                        gaps.append({"owner": identity, "reason": "same_candidate_release_fk_unbound"})
                        continue
                    poses[identity] = prediction["pose"].copy()
                    unresolved.append({"owner": identity, "kind": "predicted_release_pose",
                                       "time_s": prediction["time_s"], "condition_id": relation["release_condition_id"]})
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
        actors = []
        materials = binding["native_geometry_materials"]
        unmodelled_geometries = set(robot_state.get("unmodelled_geometry_names", ()))
        for name, (_, shape, source, _) in self._geometries.items():
            if name in unmodelled_geometries:
                continue
            pose = np.asarray(robot_state["geometry_poses"][name])
            actors.append({"owner": "robot", "name": name, "shape": shape, "pose": pose,
                           "vertices": self._solid_vertices[name], "material": materials.get(name), "source": source})
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
            if row["owner"] in self._scene_owners and self._scene_owners[row["owner"]][0]["moving"]:
                actors.append(current)
        checked = set()
        boundary = max(self.numerics["gjk_tolerance"], self.numerics["epa_tolerance"])
        for first in actors:
            for second in targets:
                if first["owner"] == second["owner"]:
                    continue
                identity = tuple(sorted(((first["owner"], first["name"]), (second["owner"], second["name"]))))
                if identity in checked:
                    continue
                checked.add(identity)
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
                    continue
                transforms = [coal.Transform3s(row["pose"][:3, :3], row["pose"][:3, 3]) for row in (first, second)]
                distance = coal.DistanceResult()
                coal.distance(first["shape"], transforms[0], second["shape"], transforms[1], self._request, distance)
                value = float(distance.min_distance)
                if not np.isfinite(value):
                    raise RuntimeError("Independent scene distance returned nonfinite evidence")
                row = {"actor": first["owner"], "actor_geometry": first["name"],
                       "target_owner": second["owner"], "target_geometry": second["name"],
                       "actor_source": first["source"], "target_source": second["source"],
                       "signed_distance_m": value, "witness_first": np.asarray(distance.getNearestPoint1()).tolist(),
                       "witness_second": np.asarray(distance.getNearestPoint2()).tolist(),
                       "normal_first_to_second": np.asarray(distance.normal).tolist(),
                       "relation": "separated" if value > boundary else "penetrating" if value < -boundary else "numerical_boundary",
                       "permission": None, "contact_region_checks": []}
                rows.append(row)
                if value > boundary:
                    continue
                request, result = coal.CollisionRequest(), coal.CollisionResult()
                request.enable_contact, request.num_max_contacts = True, 64
                coal.collide(first["shape"], transforms[0], second["shape"], transforms[1], request, result)
                row["contacts"] = [{"position": np.asarray(contact.pos).tolist(), "normal": np.asarray(contact.normal).tolist(),
                                    "penetration_depth": float(contact.penetration_depth)} for contact in result.getContacts()]
                row["contact_report_truncated"] = result.numContacts() >= request.num_max_contacts
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
                    if region is not None and value >= -rule["modeling_allowance_m"]-rule["numerical_position_tolerance_m"]:
                        evidence.update(_overlap_within_region(_world_vertices(actor["vertices"], actor["pose"]),
                                                               _world_vertices(target["vertices"], target["pose"]), region))
                        incomplete |= not evidence["completed"]
                        if evidence.get("contained"):
                            row["permission"] = rule["rule_id"]
                    row["contact_region_checks"].append(evidence)
                    if row["permission"]:
                        break
                # END both directions still require their exact material, patch and interval.
                if row["permission"] is None:
                    issue = {"kind": "independent_scene_collision", "pair_index": len(rows)-1,
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
        # Required contact uses the complete material feature relation; mere
        # permission for an intersecting pair cannot prove that relation holds.
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
            _, reason = _feature_region(rule, actor_pose, poses[rule["target_owner"]], finger_positions)
            if reason == "missing_named_finger_geometry" and state_kind == "nominal":
                unresolved.append({"rule_id": rule["rule_id"], "kind": "future_material_feature_geometry_unbound"})
                continue
            if reason:
                issue = {"kind": "independent_required_contact", "rule_id": rule["rule_id"], "reason": reason}
                if rule["actor"] in uncertain_owners or rule["target_owner"] in uncertain_owners:
                    unresolved.append({**issue, "scope": "conditional owner model"})
                else:
                    violations.append(issue)
        if unmodelled_geometries:
            unresolved.append({"kind": "future_finger_geometry_unbound", "geometry_names": sorted(unmodelled_geometries)})
        return {"passed": not violations and not gaps, "violations": violations, "query_gaps": gaps,
                "unresolved_assumptions": unresolved, "pairs": rows, "state_kind": state_kind,
                "progress": float(progress), "observed_scene_checked": not gaps and not uncertain_owners and not unmodelled_geometries,
                "supplied_scene_models_checked": not gaps,
                "complete_scene_geometry": all(owner["geometry_complete"] for owner, _ in self._scene_owners.values()),
                "continuous_collision_certified": False, "unknown_space_certified": False,
                "scope": "independent filled native solids at this state and complete local overlap regions where adapted"}
    # END raw geometry evidence is retained independently of every task permission.

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
    if isinstance(vertices, dict):
        return {"voxel_boxes": vertices["voxel_boxes"], "pose": pose}
    return None if vertices is None else np.asarray(vertices) @ pose[:3, :3].T + pose[:3, 3]


def _scene_shape(record):
    """Build one explicitly described filled volume, with no mesh-hull fallback."""
    kind = record["kind"]
    vertices = None
    if kind == "voxel":
        shape_xyz = np.asarray(record["shape_xyz"], dtype=int)
        size = float(record["voxel_size"])
        indices = np.asarray(record["occupied_flat_indices"], dtype=np.int64)
        if (shape_xyz.shape != (3,) or np.any(shape_xyz <= 0) or np.prod(shape_xyz) > 8_000_000
                or not np.isfinite(size) or size <= 0 or indices.ndim != 1 or not 0 < len(indices) <= 250_000
                or len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= np.prod(shape_xyz))):
            raise ValueError("Independent voxel solids require their complete bounded occupied-cell inventory")
        dtype = np.dtype(record["field_dtype"])
        if dtype.kind != "f" or dtype.itemsize not in (2, 4, 8):
            raise ValueError("Independent field samples require a supported finite floating dtype")
        field = np.frombuffer(bytes.fromhex(record["field_bytes_hex"]), dtype=dtype)
        if (field.size != np.prod(shape_xyz) or not np.isfinite(field).all()
                or record["occupancy_rule"] != "ESDF_sample_strictly_positive"
                or not np.array_equal(indices, np.flatnonzero(field > 0))):
            raise ValueError("Occupied-cell inventory changed from the exact original field")
        ijk = np.column_stack(np.unravel_index(indices, tuple(shape_xyz)))
        # OctoMap's lattice is anchored at zero. Build in positive grid-cell
        # coordinates, then apply the native lower grid corner exactly once.
        shape = coal.makeOctree((ijk + .5)*size, size)
        boxes = np.asarray(shape.toBoxes(), float)[:, :4]
        vertices = {"voxel_boxes": boxes}
        local = _pose7(record["pose"])
        shift = np.eye(4)
        shift[:3, 3] = -shape_xyz*size/2
        return shape, vertices, local @ shift
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
    patch = (world-origin) @ axes
    tolerance = float(rule["numerical_position_tolerance_m"])
    angle = np.arctan2(np.linalg.norm(np.cross(normal, axes[:, 2])), -normal @ axes[:, 2])
    if angle > rule["normal_tolerance_rad"] + rule["numerical_angle_tolerance_rad"]:
        return None, "material_normal_outside_permission"
    if np.any(np.abs(patch[:, :2]) > np.asarray(rule["patch_half_size"]) + tolerance):
        return None, "material_feature_outside_target_patch"
    if np.any(patch[:, 2] < rule["gap_bounds"][0]-tolerance) or np.any(patch[:, 2] > rule["gap_bounds"][1]+tolerance):
        return None, "material_feature_outside_gap_bounds"
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
    return (np.asarray(rows), np.asarray(bounds)), None


def _overlap_within_region(first_vertices, second_vertices, region):
    """Use SciPy/HiGHS support LPs over the entire convex solid intersection.

    Every inequality of the finite material/patch region is maximized over
    the entire intersection. One Coal contact witness is insufficient here.
    Failed/infeasible numerical queries stay gaps; no arbitrary epsilon turns
    an uncompleted whole-region check into permission.
    """
    from scipy.optimize import linprog
    from scipy.spatial import ConvexHull
    if first_vertices is None or second_vertices is None:
        return {"completed": False, "reason": "curved_contact_requires_complete_local_region_adapter"}
    if isinstance(first_vertices, dict) or isinstance(second_vertices, dict):
        def bounds(vertices):
            if isinstance(vertices, dict):
                boxes, pose = vertices["voxel_boxes"], vertices["pose"]
                local_lo, local_hi = np.min(boxes[:, :3]-boxes[:, 3:4]/2, axis=0), np.max(boxes[:, :3]+boxes[:, 3:4]/2, axis=0)
                corners = _box_vertices(local_hi-local_lo)+(local_hi+local_lo)/2
                points = _world_vertices(corners, pose)
            else:
                points = vertices
            return points.min(axis=0), points.max(axis=0)
        def components(vertices, other_bounds):
            if not isinstance(vertices, dict):
                return [vertices]
            boxes, pose = vertices["voxel_boxes"], vertices["pose"]
            centres = boxes[:, :3] @ pose[:3, :3].T + pose[:3, 3]
            extent = boxes[:, 3:4]/2 * np.abs(pose[:3, :3]).sum(axis=1)
            selected = np.flatnonzero(np.all(centres+extent >= other_bounds[0], axis=1)
                                      & np.all(centres-extent <= other_bounds[1], axis=1))
            if len(selected) > 512:
                return None
            return [_world_vertices(_box_vertices([boxes[i, 3]]*3)+boxes[i, :3], pose) for i in selected]
        first = components(first_vertices, bounds(second_vertices))
        second = components(second_vertices, bounds(first_vertices))
        if first is None or second is None or len(first)*len(second) > 512:
            return {"completed": False, "reason": "complete_contact_leaf_query_budget"}
        evidence, intersecting = [], 0
        for a in first:
            for b in second:
                check = _overlap_within_region(a, b, region)
                # HiGHS infeasibility here proves these two complete convex
                # leaves have no volume intersection. Keep that raw status.
                if check.get("solver_status") == 2:
                    continue
                evidence.append(check)
                intersecting += 1
                if not check.get("completed") or not check.get("contained"):
                    return {**check, "leaf_queries": len(evidence), "scope": "all potentially overlapping occupied leaves"}
        if not intersecting:
            return {"completed": False, "reason": "coal_boundary_without_resolved_volume_intersection"}
        return {"completed": True, "contained": True, "leaf_queries": len(evidence),
                "method": "complete_occupied_leaf_intersections", "numerical_certificate": False}
    equations = np.concatenate([ConvexHull(first_vertices).equations, ConvexHull(second_vertices).equations])
    A, b = equations[:, :3], -equations[:, 3]
    maxima = []
    for axis, bound in zip(*region):
        result = linprog(-axis, A_ub=A, b_ub=b, bounds=[(None, None)]*3, method="highs",
                         options={"primal_feasibility_tolerance": 1e-9, "dual_feasibility_tolerance": 1e-9})
        if not result.success:
            return {"completed": False, "reason": "overlap_support_query_incomplete", "solver_status": int(result.status)}
        value = float(axis @ result.x-bound)
        maxima.append(value)
        if value > 0.:
            return {"completed": True, "contained": False, "maximum_constraint_excess_m": value,
                    "outside_witness": result.x.tolist(), "method": "whole_convex_intersection_support"}
    return {"completed": True, "contained": True, "maximum_constraint_excess_m": max(maxima),
            "method": "whole_convex_intersection_support", "numerical_certificate": False}
# END filled-solid intersection checks do not infer whole-pair task permission.
