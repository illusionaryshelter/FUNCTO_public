"""Planning on frozen observed implicit sets and declared rigid relationships.

Robot entities retain their native models; task entities retain their
observed trilinear fields. Optimization and acceptance use the same
nonpenetration rule. Native self checks retain their original model pair
policy and numerical boundary.
"""

# BEGIN direct entity pairs replace sphere proposals and solid confirmation.
from copy import copy, deepcopy
from dataclasses import asdict
from itertools import combinations

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from .collision_records import encode_array, decode_array
from .implicit_distance import PairPermission
from .observed_mapping import FrozenObservedField
from .observed_robot import DifferentiableRobotKinematics
from .robot_geometry_check import RobotGeometryChecker
from .observed_collision import pair_residual
# BEGIN scene-owned immutable and per-query native projection bounds.
from .native_field_distance import _native_bounds, _transform, native_pair_lower_bound
# END support computations retain the registered native entity.
from .planning_problem import PosePath, quaternion_rotate, quaternion_conjugate, quaternion_multiply
from .planning_scene import (Attachment, CollisionOwner, ContactRule, ObservedSpace,
                             PrismaticGripper, _tensor, _interval, _world_points)
from .optimizer_device_checks import constant_tensor


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (tuple, list)):
        return [_plain(row) for row in value]
    if isinstance(value, dict):
        return {key: _plain(row) for key, row in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _matrices(pose):
    """Snapshot the same differentiable rigid poses for interval queries."""
    position, quaternion = (value.detach().cpu().numpy() for value in pose)
    result = np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy()
    result[..., :3, 3] = position
    result[..., :3, :3] = Rotation.from_quat(
        quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape(*position.shape[:-1], 3, 3)
    return result


class OnlinePlanningScene:
    """Known robot collision entities and original observed task fields.

    Robot geometry is shared with the state validator and placed by complete
    joint FK. Unknown-space decisions for moving scene owners retain the
    supplied policy and the original measured free-ray coverage.
    """

    # BEGIN the scene shares the approved native robot model and explicit roles.
    def __init__(self, *, background, owners, coverage, reference_frame,
                 robot_model, fixed_geometry_names, tool_geometry_names, contacts=(), gripper=None,
                 has_gripper=True, unknown_policy,
                 initial_support_relations=(), numerical_position_tolerance_m=1e-6,
                 numerical_rotation_tolerance_rad=1e-6):
        if (not isinstance(background, FrozenObservedField)
                or any(not isinstance(owner.query, FrozenObservedField) for owner in owners)):
            raise TypeError("All entities must retain their raw observed field")
        self.background, self.owners, self.coverage = background, tuple(owners), coverage
        self.reference_frame = reference_frame
        if not isinstance(robot_model, RobotGeometryChecker):
            raise TypeError("The scene requires the same known robot checker as its validator")
        self.robot_model = robot_model
        self.fixed_geometry_names = tuple(fixed_geometry_names)
        # BEGIN a declared tool component and its attached owners form one assembly.
        self.tool_geometry_names = tuple(tool_geometry_names)
        robot_ids = set(robot_model.geometry_names)
        if (not set(self.fixed_geometry_names) <= robot_ids
                or len(set(self.fixed_geometry_names)) != len(self.fixed_geometry_names)
                or not set(self.tool_geometry_names) <= robot_ids
                or len(set(self.tool_geometry_names)) != len(self.tool_geometry_names)
                or len({owner.owner_id for owner in self.owners}) != len(self.owners)):
            raise ValueError("Tool members, fixed installations and scene owners need registered unique identities")
        self.required_robot_links = tuple(dict.fromkeys(
            frame for frame, _, _, _ in robot_model._geometries.values()))
        # BEGIN one local projection range per immutable registered entity.
        self._native_local_bounds = {}
        for name, (_, geometry, _, _) in robot_model._geometries.items():
            bounds = _native_bounds(geometry, _transform(np.eye(4)))
            bounds.flags.writeable = False
            self._native_local_bounds[name] = bounds
        # END rollout clones share these pose-independent model bounds.
        self.contacts = tuple(contacts)
        self.gripper, self.has_gripper = gripper, has_gripper
        self.unknown_policy = deepcopy(unknown_policy)
        if (set(unknown_policy) != {"conditional_regions", "outside_regions"}
                or unknown_policy["outside_regions"] not in {"forbid", "require_observation", "allow_attempt"}):
            raise ValueError("Owner unknown-space policy requires explicit regions and outside action")
        for region in unknown_policy["conditional_regions"]:
            lower, upper = np.asarray(region["lower_m"]), np.asarray(region["upper_m"])
            if lower.shape != (3,) or upper.shape != (3,) or np.any(lower >= upper):
                raise ValueError("Unknown-space regions need finite ordered three-dimensional bounds")
        owner_ids = {owner.owner_id for owner in self.owners}
        if len({rule.rule_id for rule in self.contacts}) != len(self.contacts):
            raise ValueError("Contact relation identities must be unique")
        for rule in self.contacts:
            if (rule.target_owner not in owner_ids or rule.actor not in {"robot", *owner_ids}
                    or rule.actor == rule.target_owner
                    or (rule.actor == "robot" and rule.robot_body not in robot_ids)):
                raise ValueError("Finite features must name registered distinct entities")
        # END internal membership comes from Attachment, independently of contact materials.
        self.numerical_position_tolerance_m = float(numerical_position_tolerance_m)
        self.numerical_rotation_tolerance_rad = float(numerical_rotation_tolerance_rad)
        self.required_progress_anchors = tuple(sorted(
            {x for rule in self.contacts for x in rule.interval}
            | {x for owner in self.owners for relation in owner.attachments for x in relation.interval}
            | {x for relation in initial_support_relations for x in relation.interval}))
        self.initial_support_relations = tuple(initial_support_relations)
        self._initial_support_models = {}
        if self.initial_support_relations:
            from .initial_support import PreparedInitialSupport
            by_owner = {owner.owner_id: owner for owner in self.owners}
            self._initial_support_models = {row.relation_id: PreparedInitialSupport(
                row, by_owner[row.actor], by_owner[row.target_owner],
                self.numerical_position_tolerance_m) for row in self.initial_support_relations}
        self._fixed_release_poses = {}
        self.observation_report, self.observation_partition_masks = {}, {}
        self.grasp_contact_activations = ()
        self.robot_pairs = ()
        self.robot_validator, self.kinematics = None, None
        self.check_request = None
        self.last_query_report = {}
        # BEGIN immutable material coordinates belong to the scene geometry.
        self._objective_material_samples = None
        # END trial poses never replace or append optimization samples.
    # END neither image visibility nor joint dependence defines model roles.

    # BEGIN bind the same known model to the complete differentiable joint tree.
    def bind_online_robot(self, *, validator, check_request=None):
        """Reuse the validator's complete model and calibrated Pinocchio FK."""
        if validator.checker is not self.robot_model:
            raise ValueError("Scene and validator must share the same known robot model")
        self.robot_validator, self.robot_pairs = validator, validator.pairs
        # BEGIN the scene shares the existing model rather than rebuilding a tree.
        self.kinematics = DifferentiableRobotKinematics(validator.checker.model,
            frames=self.required_robot_links, T_reference_base=validator.T_reference_base)
        # END every caller's named coordinates use this same CPU FK algorithm.
        self.check_request = check_request
    # END per-entity local transforms are applied when geometric poses are formed.

    def clone_for_rollout(self):
        result = copy(self)
        result.owners = tuple(copy(owner) for owner in self.owners)
        result._fixed_release_poses = dict(self._fixed_release_poses)
        result.gripper = None if self.gripper is None else copy(self.gripper)
        result.last_query_report = {}
        return result

    @staticmethod
    def _release_pose(relation, position, quaternion, progress):
        """Gather exact release FK; a missing anchor is an explicit violation.

        The first occurrence is the physical release. Later samples on the
        same progress plateau may move the hand while the owner stays fixed.
        When absent, the gathered numeric value is not accepted as
        a release prediction: ``found`` is false and subsequent samples fail.
        Keeping arithmetic finite avoids poisoning an earlier, still-attached
        query's derivatives when it does not need a release state yet.
        """
        if relation.release_condition_id is None:
            raise ValueError("Continued Holding has no physical release anchor")
        # Match while progress is still in its input dtype. Converting a rounded
        # FP32 event (e.g. 17+1/3) to double before matching invents a discrepancy.
        matches = (progress - relation.interval[1]).abs() <= 1e-7
        index = matches.to(torch.int64).argmax(-1)
        p = torch.gather(position, 1, index[:, None, None].expand(-1, 1, 3))
        q = torch.gather(quaternion, 1, index[:, None, None].expand(-1, 1, 4))
        return (p + quaternion_rotate(q, _tensor(relation.position_tool_owner, p)),
                quaternion_multiply(q, _tensor(relation.quaternion_tool_owner, q))), matches.any(-1)

    def with_release_history(self, position, quaternion, progress):
        """Copy release FK from an already selected prefix into a child branch.

        Future releases remain variables. The first reached occurrence of a
        release owns its pose, including on a progress plateau; extending a
        branch cannot replace it with the later hand pose. Full optimization
        still uses the original scene, where releases remain shared decisions.
        """
        result = self.clone_for_rollout()
        for owner in self.owners:
            for index, relation in enumerate(owner.attachments):
                key = (owner.owner_id, index)
                if relation.release_condition_id is None or key in result._fixed_release_poses:
                    continue
                if not bool((progress >= relation.interval[1]-1e-7).any()):
                    continue
                pose, found = self._release_pose(relation, position, quaternion, progress)
                if not bool(found.all()):
                    raise ValueError("A selected search prefix must include its exact release anchor")
                result._fixed_release_poses[key] = pose
        return result

    def _permission(self, rule, actor_pose, target_pose, progress, coordinates,
                    *, uncertain_hand=None):
        # BEGIN complete finite overlap, without requiring the whole pad in the patch.
        points = _tensor(rule.feature_points, progress)
        if rule.feature_joint_translation is not None:
            mapping = _tensor(rule.feature_joint_translation, progress)
            points = points + torch.einsum("vcj,bhj->bhvc", mapping, coordinates)
            # The declared future joint curve determines this feature pose.
            # Without that curve its contact region is conditional.
            if uncertain_hand is None:
                raise ValueError("Moving material permissions need the shared hand-state certainty")
            certain = ~uncertain_hand
        else:
            certain = torch.ones_like(progress, dtype=torch.bool)
        world = _world_points(points, *actor_pose)
        target_local = quaternion_rotate(quaternion_conjugate(target_pose[1]).unsqueeze(-2),
                                         world - target_pose[0].unsqueeze(-2))
        actor_normal = quaternion_rotate(actor_pose[1], _tensor(rule.feature_normal, progress))
        target_normal = quaternion_rotate(target_pose[1], _tensor(rule.patch_axes[:, 2], progress))
        angle = torch.atan2(torch.linalg.vector_norm(torch.linalg.cross(actor_normal, target_normal), dim=-1),
                            -(actor_normal * target_normal).sum(-1))
        normal = torch.relu(angle - rule.normal_tolerance_rad - rule.numerical_angle_tolerance_rad)
        active = _interval(progress, rule.interval)
        inverse_target = quaternion_conjugate(target_pose[1])
        material_normal = quaternion_rotate(inverse_target, actor_normal)
        edges_world = quaternion_rotate(actor_pose[1].unsqueeze(-2), _tensor(rule._feature_edge_normals, progress))
        edge_normals = quaternion_rotate(inverse_target.unsqueeze(-2), edges_world)
        origin = target_local[..., 0, :]
        tolerance = rule.numerical_position_tolerance_m
        rows = [edge_normals]
        bounds = [(edge_normals*origin.unsqueeze(-2)).sum(-1)-_tensor(rule._feature_edge_offsets, progress)+tolerance]
        material_heights = (material_normal.unsqueeze(-2)*target_local).sum(-1)
        for sign in (-1., 1.):
            rows.append(sign*material_normal.unsqueeze(-2))
            bounds.append((sign*material_heights).max(-1).values.unsqueeze(-1)+rule.modeling_allowance_m+tolerance)
        patch_axes, patch_origin = _tensor(rule.patch_axes, progress), _tensor(rule.patch_origin, progress)
        for column, width in enumerate((*rule.patch_half_size, rule.modeling_allowance_m)):
            for sign in (-1., 1.):
                axis = sign*patch_axes[:, column]
                rows.append(axis.expand(*progress.shape, 1, 3))
                bounds.append((axis@patch_origin+width+tolerance).expand(*progress.shape, 1))
        # A contact point belongs to the material's thin extrusion. Project
        # it back onto that actual face before evaluating the declared gap;
        # this allows a curved target to touch just part of a larger pad.
        patch_normal = patch_axes[:, 2]
        # BEGIN gap belongs to the actual feature's affine projection.
        if rule.feature_kind == "edge":
            tangent_world = quaternion_rotate(actor_pose[1], _tensor(rule._feature_tangent, progress))
            tangent = quaternion_rotate(inverse_target, tangent_world)
            gap_axis = (tangent*patch_normal).sum(-1).unsqueeze(-1)*tangent
            gap_offset = (origin*(patch_normal-gap_axis)).sum(-1)-patch_normal@patch_origin
        else:
            cosine = (material_normal*patch_normal).sum(-1)
            gap_axis = patch_normal-cosine.unsqueeze(-1)*material_normal
            gap_offset = cosine*material_heights[..., 0]-patch_normal@patch_origin
        # END an edge never borrows the rest of its supporting plane.
        for sign, limit in ((1., rule.gap_bounds[1]), (-1., -rule.gap_bounds[0])):
            rows.append((sign*gap_axis).unsqueeze(-2))
            bounds.append((limit-sign*gap_offset+tolerance).unsqueeze(-1))
        region_normals, region_bounds = torch.cat(rows, dim=-2), torch.cat(bounds, dim=-1)
        residual = normal
        policy = {"robot_body": rule.robot_body, "active": active & certain,
                  "normals": region_normals, "bounds": region_bounds,
                  "feature_points": target_local,
                  "allowance": rule.modeling_allowance_m,
                  "normal_valid": normal <= 0}
        return (policy, residual * active.to(progress.dtype),
                (active & ~certain).to(progress.dtype))

    def optimizer_preferences(self, progress, joint_position, joint_names):
        """Squared tracking error for optimizer use only; no acceptance change.

        A tracking drive already specifies one executable command at every
        progress. Its hard numerical band remains in components(). This small
        objective keeps optimization attracted to that command after entering
        the feasible band, instead of stopping at a float32-sensitive boundary.
        Binary drives supply no command width and therefore no such objective.
        Material contact and functional pose freedoms are never centered here.
        """
        hand = self.gripper
        if hand is None or hand.controller != "position_tracking":
            return torch.zeros_like(progress)
        if joint_position is None or joint_names is None:
            raise ValueError("Tracking preferences require the actual named solver joints")
        lower, upper = hand.joint_bounds(progress)
        if lower.shape != (*progress.shape, len(hand.joint_names)) or upper.shape != lower.shape:
            raise ValueError("Tracking command shape must match progress and native fingers")
        indices = [tuple(joint_names).index(name) for name in hand.joint_names]
        command = (lower+upper)/2  # Same single-command contract as hand.state().
        # BEGIN use the same ordered integer columns without host transfer in replay.
        indices = constant_tensor(indices, joint_position, dtype=torch.long)
        return ((joint_position[..., indices]-command)**2).sum(-1)

    def project_residual(self, progress, joint_position, joint_names):
        """Independent executable finger-drive equalities for OMPL projection.

        Output has J-rank(A) rows for tracking and zero rows otherwise. The
        caller must include the same native A q=b coupling in its problem;
        duplicate coupling rows are intentionally not introduced here. These
        are exact equalities, with projection convergence handled separately
        from the native controller command or task's permissible motion.
        """
        if self.gripper is None or self.gripper.controller != "position_tracking":
            return joint_position[..., :0]
        hand = self.gripper
        indices = [tuple(joint_names).index(name) for name in hand.joint_names]
        lower, upper = hand.joint_bounds(progress)
        command = (lower + upper) / 2
        # BEGIN graph-scoped integer columns retain the native joint order.
        indices = constant_tensor(indices, joint_position, dtype=torch.long)
        # END CPU projection continues to use the original as_tensor conversion.
        return ((joint_position[..., indices] - command)
                @ _tensor(hand._drive_basis.T, joint_position))


    def validate_problem(self, problem, *, goal_identities=None):
        if problem.reference_frame != self.reference_frame:
            raise ValueError("Scene and robot must use the same calibrated reference frame")
        conditions = {row["condition_id"]: row for row in problem.conditions if "condition_id" in row}
        # Bind finite contact declarations to the same nominal goal and close
        # occurrence. Execution activates by command occurrence, never by an
        # FK-minus-predicted-target arrival gate.
        from .planning_scene import (grasp_contact_rule_ids, validate_grasp_contact_arrival,
                                     event_occurrence_key)
        activations = [row for row in problem.conditions if row.get("kind") == "grasp_contact_activation"]
        pose_ids = (list(range(len(problem.pose_events))) if goal_identities is None
                    else list(goal_identities["pose_events"]))
        used = set()
        for row in activations:
            ids = grasp_contact_rule_ids(row, [asdict(rule) for rule in self.contacts])
            if used.intersection(ids):
                raise ValueError("Each finite contact belongs to one activation condition")
            original = row["arrival_goal_id"][1]
            commands = [event for event in problem.events if event.kind == "close"
                        and event_occurrence_key(event) == event_occurrence_key(row["activation_event"])]
            if original not in pose_ids or len(commands) != 1:
                raise ValueError("Contact activation needs its nominal goal and close occurrence")
            requirement = problem.pose_events[pose_ids.index(original)]
            validate_grasp_contact_arrival(row, requirement)
            if (requirement.actual_relation is not None or requirement.progress != commands[0].progress
                    or commands[0].progress != row["interval"][0]):
                raise ValueError("Contact goal and command must share the declared progress")
            used.update(ids)
        self.grasp_contact_activations = tuple(deepcopy(activations))
        for owner in self.owners:
            for relation in owner.attachments:
                declaration = conditions[relation.condition_id]
                if (declaration.get("carrier_id") != owner.owner_id
                        or declaration.get("kind") not in {"declared_attachment", "candidate_attachment"}
                        or declaration["progress"] != relation.interval[0]):
                    raise ValueError("Rigid relationship must preserve its declared owner and event")
                # BEGIN the declared motion and collision owner use the same transform.
                expected = np.linalg.inv(np.asarray(declaration["T_carrier_eef"], float))
                actual = np.eye(4)
                actual[:3, 3] = relation.position_tool_owner
                actual[:3, :3] = Rotation.from_quat(
                    relation.quaternion_tool_owner[[1, 2, 3, 0]]).as_matrix()
                if not np.allclose(actual, expected, atol=1e-7, rtol=0):
                    raise ValueError("Collision owner and declared attachment transform disagree")
                # END no contact-material association is needed for assembly membership.
                if relation.release_condition_id is not None:
                    release = conditions[relation.release_condition_id]
                    if release.get("kind") != "predicted_release" or release["progress"] != relation.interval[1]:
                        raise ValueError("Release must retain its event and progress")
        if self.gripper is not None:
            actual = {(row["skill_id"], row["event_id"], row.get("occurrence", 0),
                       row["command"], row["progress"]) for row in self.gripper.commands}
            expected = {(row.skill_id, row.event_id, row.occurrence, row.kind, row.progress)
                        for row in problem.events if row.kind in {"open", "close", "hold"}}
            if actual != expected:
                raise ValueError("Hand commands must retain the problem's event occurrences")

    def _owner_poses(self, position, quaternion, progress, owner_poses, costs):
        poses = {}
        for owner in self.owners:
            if owner_poses is not None and owner.owner_id in owner_poses:
                matrix = np.asarray(owner_poses[owner.owner_id], float)
                p = position.new_tensor(matrix[:3, 3]).expand(*progress.shape, 3)
                q = quaternion.new_tensor(Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]])
                poses[owner.owner_id] = p, q.expand(*progress.shape, 4)
                continue
            releases = []
            for index, relation in enumerate(owner.attachments):
                if relation.release_condition_id is None:
                    releases.append(None)
                elif (owner.owner_id, index) in self._fixed_release_poses:
                    releases.append(self._fixed_release_poses[(owner.owner_id, index)])
                else:
                    pose, found = self._release_pose(relation, position, quaternion, progress)
                    releases.append(pose)
                    costs[f"relationship/missing_release/{owner.owner_id}/{index}"] = (
                        (progress >= relation.interval[1]) & ~found[:, None]).to(progress.dtype)
            poses[owner.owner_id] = owner.pose(position, quaternion, progress, releases)
        return poses

    # BEGIN one official-robot self query serves both initialization and full scenes.
    def _robot_self_components(self, complete_q, complete_names, robot_poses,
                               robot_matrices, hand_state, zero, *,
                               record_pair, record_nonclear, include_bounds):
        """Query the fixed robot's required entity pairs at prepared states.

        Entity shapes and fixed link transforms are the official model of
        this unchanged robot/end-effector configuration, reusable across
        tasks. Named feedback/nominal hand coordinates and calibrated FK
        determine poses; task fields and camera coverage do not enter here.
        The caller supplies differentiable poses and their same-state matrix
        snapshots. Coal distances own the classification. Penetrating-pair
        witnesses supply the existing local derivative; numerical boundaries
        remain separately reported hard evidence.
        """
        from .implicit_distance import inverse_points
        from .observed_collision import _world

        shape = tuple(complete_q.shape[:-1])
        uncertain = (np.zeros(shape, bool) if hand_state is None
                     else hand_state.uncertain.detach().cpu().numpy())
        unmodelled = (set() if hand_state is None else
                     set(self.robot_model.geometry_depending_on(self.gripper.joint_names)))
        report = {"collision_rule": "nonpenetration", "pair_queries": 0,
                  "status_counts": {}, "nonclear_pairs": [],
                  "internal_assembly_pair_count": 0, "tool_geometry_names": list(self.tool_geometry_names),
                  "robot_geometry_source": "registered_native_model",
                  "robot_view_coverage_status": "not_queried",
                  "coverage_scope": "not_queried"}
        q_cpu = complete_q.detach().cpu().numpy()
        self_costs, self_unresolved = [], []
        for index in np.ndindex(shape):
            evidence = self.robot_model.query(joint_names=complete_names,
                joint_positions=q_cpu[index], T_reference_base=self.robot_validator.T_reference_base,
                pairs=self.robot_validator.required_pairs, check_request=self.check_request)
            value = zero[index]
            undecided = value
            for row in evidence["pairs"]:
                first_id, second_id = row["first"], row["second"]
                entity_key = {"first": f"robot/{first_id}", "second": f"robot/{second_id}"}
                if uncertain[index] and ({first_id, second_id} & unmodelled):
                    record_pair(index, "collision/robot/self", first_id, second_id,
                        {"status": "unmodelled_motion", "lower_m": None}, geometry_entity_key=entity_key)
                    continue
                status = row["relation"]
                lower = None
                if include_bounds:
                    lower = native_pair_lower_bound(
                        self.robot_model._geometries[first_id][1],
                        np.asarray(evidence["geometry_poses"][first_id], float),
                        self.robot_model._geometries[second_id][1],
                        np.asarray(evidence["geometry_poses"][second_id], float),
                        row["normal_first_to_second"])
                record_pair(index, "collision/robot/self", first_id, second_id,
                    {**row, "status": {"separated": "clear", "penetrating": "intersection",
                                       "numerical_boundary": "unresolved"}[status],
                     "lower_m": lower, "distance_scope": "full_entities"}, geometry_entity_key=entity_key)
                report["pair_queries"] += 1
                report["status_counts"][status] = report["status_counts"].get(status, 0) + 1
                if status == "numerical_boundary":
                    undecided = undecided + 1.
                elif status == "penetrating":
                    a = inverse_points(row["witness_first"], robot_matrices[first_id][index])
                    b = inverse_points(row["witness_second"], robot_matrices[second_id][index])
                    pa = tuple(item[index] for item in robot_poses[first_id])
                    pb = tuple(item[index] for item in robot_poses[second_id])
                    normal = _tensor(row["normal_first_to_second"], complete_q)
                    signed = ((_world(b, pb) - _world(a, pa)) * normal).sum()
                    value = value - row["signed_distance_m"] - (signed - signed.detach())
                # BEGIN display evidence retains the actual query clock.
                if status != "separated":
                    record_nonclear(index, "collision/robot/self", first_id, second_id,
                        row, -row["signed_distance_m"] if status == "penetrating" else 0.)
                # END the original pair residual determines within-component severity.
            self_costs.append(value)
            self_unresolved.append(undecided)
        return {"collision/robot/self": torch.stack(self_costs).reshape(shape),
                "unresolved/robot/self": torch.stack(self_unresolved).reshape(shape)}, report
    # END all consumers retain the same entity identities and distance semantics.

    def components(self, robot_spheres, position, quaternion, progress,
                   joint_position=None, joint_names=None, *, trajectory_time,
                   active_contact_rule_ids=None, contact_permission_progress=None,
                   owner_poses=None, geometry_evidence_out=None, geometry_scope="full"):
        """Return same-state residuals and retain distance/coverage diagnostics.

        ``robot_spheres`` is an empty native-kinematics output. Registered
        native entities carry all robot geometry. This API returns original
        entity, contact, relationship and coverage conditions. The independent
        finite optimization function is ``optimization_objective``.
        ``geometry_scope='robot'`` computes the same official self pairs and
        hand state without constructing task-owner poses or querying fields.
        """
        if geometry_scope not in {"robot", "full"}:
            raise ValueError("Geometry scope must be robot or full")
        if self.kinematics is None or joint_position is None:
            raise ValueError("Observed robot geometry must be bound to its named joint tree")
        # BEGIN caller role does not introduce a geometric clearance.
        costs = {}
        permission_progress = progress if contact_permission_progress is None else contact_permission_progress
        # BEGIN complete pair evidence is an explicit interval-verification output.
        # Unlike display diagnostics, this list includes every required pair
        # and every skipped relation at each physical query state.
        evidence_time, evidence_progress, evidence_permission = (
            value.detach().cpu().numpy() for value in
            (trajectory_time, progress, permission_progress))

        def record_pair(index, component, first, second, evidence, *, geometry_entity_key,
                        permission_kind="none"):
            if geometry_evidence_out is not None:
                geometry_evidence_out.append({"state_index": list(index),
                    "time_s": float(evidence_time[index]),
                    "progress": float(evidence_progress[index]),
                    "permission_progress": float(evidence_permission[index]),
                    "component": component, "first": first, "second": second,
                    "geometry_entity_key": geometry_entity_key,
                    "permission_kind": permission_kind, **evidence})
        # END hard residuals and optional proof evidence have separate consumers.
        # BEGIN bounded diagnostics retain every component's failure location.
        # Pair iteration is entity-major, so the earliest physical time is
        # selected explicitly. Severity is the original pair's hard residual:
        # penetration metres for robot self pairs and the original 0/1 field
        # intersection classification for observed pairs. Values are compared
        # only within one component. No geometric query is introduced here.
        selected_nonclear = {}

        def record_nonclear(index, component, first, second, evidence, residual):
            row = {"component": component, "first": first, "second": second,
                "state_index": list(index), "time_s": float(evidence_time[index]),
                "progress": float(evidence_progress[index]),
                "permission_progress": float(evidence_permission[index]),
                "pair_residual_value": float(residual), **evidence}
            if component not in selected_nonclear:
                selected_nonclear[component] = {"first": row, "maximum": row}
                return
            selected = selected_nonclear[component]
            order = lambda item: (item["time_s"], item["state_index"])
            if order(row) < order(selected["first"]):
                selected["first"] = row
            maximum = selected["maximum"]
            if (row["pair_residual_value"] > maximum["pair_residual_value"]
                    or (row["pair_residual_value"] == maximum["pair_residual_value"]
                        and order(row) < order(maximum))):
                selected["maximum"] = row

        def nonclear_diagnostics():
            rows = []
            for selected in selected_nonclear.values():
                first, maximum = selected["first"], selected["maximum"]
                if first is maximum:
                    rows.append({**first, "selection": ["first", "maximum_pair_residual"]})
                else:
                    rows.extend(({**first, "selection": ["first"]},
                                 {**maximum, "selection": ["maximum_pair_residual"]}))
            return rows
        # END each component contributes at most two same-state evidence rows.
        hand_state, coordinates, half = None, None, None
        complete_q, complete_names = joint_position, tuple(joint_names)
        if self.gripper is not None:
            hand_state = self.gripper.state(progress, joint_position, joint_names,
                trajectory_time=trajectory_time, measured=active_contact_rule_ids is not None)
            coordinates, half = hand_state.coordinates, hand_state.half
            costs["gripper/controller"] = hand_state.controller_residual
            costs["conditional/hand_geometry"] = hand_state.uncertain.to(progress.dtype)
            missing_names = [name for name in self.gripper.joint_names if name not in complete_names]
            if missing_names:
                complete_q = torch.cat((complete_q, coordinates[..., [
                    self.gripper.joint_names.index(name) for name in missing_names]]), dim=-1)
                complete_names += tuple(missing_names)
        # BEGIN place each original collision entity using complete joint FK.
        link_poses = self.kinematics.poses(complete_q, complete_names)
        robot_poses = {}
        for name, (frame, _, _, local) in self.robot_model._geometries.items():
            link_position, link_quaternion = link_poses[frame]
            local_position = _tensor(local[:3, 3], link_position)
            local_quaternion = _tensor(
                Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]], link_quaternion)
            robot_poses[name] = (
                link_position + quaternion_rotate(link_quaternion, local_position),
                quaternion_multiply(link_quaternion, local_quaternion))
        # END geometry IDs remain distinct from their supporting link frames.
        # BEGIN self geometry precedes every task-field computation.
        robot_matrices = {name: _matrices(pose) for name, pose in robot_poses.items()}
        zero = complete_q.sum(-1) * 0 + position.sum(-1) * 0
        for pose in robot_poses.values():
            zero = zero + pose[0].sum(-1) * 0
        robot_costs, report = self._robot_self_components(complete_q, complete_names,
            robot_poses, robot_matrices, hand_state, zero,
            record_pair=record_pair, record_nonclear=record_nonclear,
            include_bounds=geometry_evidence_out is not None)
        costs.update(robot_costs)
        report["geometry_scope"] = geometry_scope
        if geometry_scope == "robot":
            report["nonclear_pairs"] = nonclear_diagnostics()
            self.last_query_report = report
            return costs
        report["coverage_scope"] = "represented_observed_occupancy"
        # END full scenes continue from this same completed self query.
        poses = self._owner_poses(position, quaternion, progress, owner_poses, costs)
        identity = (torch.zeros_like(position), torch.zeros_like(quaternion))
        identity[1][..., 0] = 1
        poses["background"] = identity
        fields = {owner.owner_id: owner.query for owner in self.owners}
        fields["background"] = self.background
        matrices = {name: _matrices(pose) for name, pose in poses.items()}
        # BEGIN one batch zero retains every position and named-joint gradient.
        # Clear or skipped pairs still return a differentiable scalar. Form
        # that zero once over the batch, retaining each entity's connection
        # in the same-state classification output.
        for pose in poses.values():
            zero = zero + pose[0].sum(-1) * 0
        # END a shared zero changes neither geometry queries nor their verdicts.
        policies = {}
        for rule in self.contacts:
            actor_pose = (position, quaternion) if rule.actor == "robot" else poses[rule.actor]
            policy, normal, uncertain = self._permission(rule, actor_pose,
                poses[rule.target_owner], permission_progress, coordinates,
                uncertain_hand=None if hand_state is None else hand_state.uncertain)
            if active_contact_rule_ids is not None and rule.rule_id not in active_contact_rule_ids:
                policy["active"] = torch.zeros_like(policy["active"])
            # The finite declared feature, target patch, gap and normal all
            # participate in permission. Required feature obligations remain
            # separate residuals even when every collision pair is clear.
            active = policy["active"] & policy["normal_valid"]
            policies[rule.rule_id] = (policy["normals"].detach().cpu().numpy(),
                                     policy["bounds"].detach().cpu().numpy(),
                                     active.detach().cpu().numpy())
            if rule.required:
                from .observed_collision import query_region_intersection
                gaps, undecided = [], []
                normals_cpu, bounds_cpu, _ = policies[rule.rule_id]
                interval_active = policy["active"].detach().cpu().numpy()
                target_field = fields[rule.target_owner]
                target_matrices = matrices[rule.target_owner]
                for index in np.ndindex(tuple(progress.shape)):
                    evidence = {"status": "inactive"}
                    if interval_active[index]:
                        target_matrix = target_matrices[index]
                        world_normals = normals_cpu[index] @ target_matrix[:3, :3].T
                        world_bounds = bounds_cpu[index]+world_normals @ target_matrix[:3, 3]
                        evidence = query_region_intersection(target_field, target_matrix,
                            world_normals, world_bounds, check_request=self.check_request)
                    gap = zero[index] + float(evidence["status"] == "missing_contact")
                    gaps.append(gap)
                    undecided.append(float(evidence["status"] == "unresolved"))
                normal_cost = normal*policy["active"].to(normal.dtype)*(1-uncertain)
                costs[f"contact/{rule.rule_id}"] = normal_cost+torch.stack(gaps).reshape(progress.shape)
                costs[f"unresolved/contact/{rule.rule_id}"] = progress.new_tensor(undecided).reshape(progress.shape)
                costs[f"conditional/contact/{rule.rule_id}"] = uncertain
        # A pair owns only its declared feature regions; no rule can exempt
        # an entire finger because a smaller pad has missing observations.
        progress_cpu = permission_progress.detach().cpu().numpy()
        uncertain_cpu = (np.zeros(tuple(progress.shape), bool) if hand_state is None
                         else hand_state.uncertain.detach().cpu().numpy())
        unmodelled = (set() if hand_state is None else
                     set(self.robot_validator.checker.geometry_depending_on(self.gripper.joint_names)))
        # BEGIN uncertainty belongs to registered robot entities, not display names.
        unmodelled_entity_keys = {f"robot/{name}" for name in unmodelled}
        # END a scene owner may retain the same name as a robot component.
        pairs = []
        moving_owners = {owner.owner_id for owner in self.owners if owner.moving}
        for name, (_, geometry, _, _) in self.robot_model._geometries.items():
            for target, field in fields.items():
                # Explicit fixed installations constrain moving owners.
                # Their static support against background is not a motion pair.
                if name in self.fixed_geometry_names and target not in moving_owners:
                    continue
                pairs.append((f"collision/robot/{target}", name, target, geometry, field,
                    robot_poses[name], poses[target], robot_matrices[name], matrices[target],
                    "robot", target))
        for owner in self.owners:
            if owner.moving:
                pairs.append((f"collision/{owner.owner_id}/background", owner.owner_id, "background",
                    owner.query, self.background, poses[owner.owner_id], poses["background"],
                    matrices[owner.owner_id], matrices["background"], owner.owner_id, "background"))
        for first, second in combinations(self.owners, 2):
            if first.moving or second.moving:
                if not first.moving:
                    first, second = second, first
                pairs.append((f"collision/{first.owner_id}/{second.owner_id}", first.owner_id, second.owner_id,
                    first.query, second.query, poses[first.owner_id], poses[second.owner_id],
                    matrices[first.owner_id], matrices[second.owner_id], first.owner_id, second.owner_id))
        # BEGIN original members retain their geometry for all external pairs.
        attachment_progress = progress.detach().cpu().numpy()
        attached = {owner.owner_id: np.logical_or.reduce(
            [relation.active(attachment_progress) for relation in owner.attachments])
            if owner.attachments else np.zeros(attachment_progress.shape, bool)
            for owner in self.owners}
        # BEGIN world ranges belong to this call's exact entity/state poses.
        native_world_bounds = {}
        # END no range survives into a later trajectory query.
        for component, first_id, second_id, first, second, first_pose, second_pose, fm, sm, actor, target in pairs:
            residuals = []
            entity_key = {"first": f"robot/{first_id}" if actor == "robot" else f"owner/{first_id}",
                          "second": "background" if target == "background" else f"owner/{second_id}"}
            rules = [rule for rule in self.contacts
                     if ((actor == "robot" and rule.actor == "robot" and rule.target_owner == target
                          and rule.robot_body == first_id)
                         or (actor != "robot" and {rule.actor, rule.target_owner} == {first_id, second_id}))]
            support = [model for model in self._initial_support_models.values()
                       if model.relation.actor == actor and model.relation.target_owner == target]
            for index in np.ndindex(tuple(progress.shape)):
                if self.check_request is not None:
                    self.check_request()
                # BEGIN each pair reuses this state's batch zero, including skips.
                state_zero = zero[index]
                if uncertain_cpu[index] and unmodelled_entity_keys.intersection(entity_key.values()):
                    residuals.append(state_zero)
                    record_pair(index, component, first_id, second_id,
                        {"status": "unmodelled_motion", "lower_m": None}, geometry_entity_key=entity_key)
                    continue
                held_internal = (target in attached and attached[target][index]
                    and ((actor == "robot" and first_id in self.tool_geometry_names)
                         or (actor in attached and attached[actor][index])))
                if held_internal:
                    residuals.append(state_zero)
                    report["internal_assembly_pair_count"] += 1
                    record_pair(index, component, first_id, second_id,
                        {"status": "internal_assembly", "lower_m": None},
                        geometry_entity_key=entity_key,
                        permission_kind="declared_rigid_assembly")
                    continue
                # END skipped pairs retain zero derivatives to all entity positions.
                regions = []
                for rule in rules:
                    normals, bounds, active = policies[rule.rule_id]
                    if active[index]:
                        target_matrix = matrices[rule.target_owner][index]
                        world_normals = normals[index] @ target_matrix[:3, :3].T
                        world_bounds = bounds[index]+world_normals @ target_matrix[:3, 3]
                        region = (world_normals, world_bounds)
                        regions.append((region, region))
                permission = PairPermission(regions) if regions else None
                for model in support:
                    if model.relation.interval[0] <= progress_cpu[index] < model.relation.interval[1]:
                        permission = model.permission(sm[index], additional=permission)
                # BEGIN prepare a world range only for an actual nonempty query.
                native_bounds = None
                if actor == "robot" and second.root is not None:
                    key = first_id, index
                    if key not in native_world_bounds:
                        native_world_bounds[key] = _native_bounds(first, _transform(fm[index]))
                    native_bounds = (self._native_local_bounds[first_id], native_world_bounds[key])
                # END different partners share only the entity's geometric range.
                # BEGIN each original pair supplies its own same-state evidence.
                value, evidence = pair_residual(first,
                    (first_pose[0][index], first_pose[1][index]), fm[index], second,
                    (second_pose[0][index], second_pose[1][index]), sm[index],
                    permission=permission, check_request=self.check_request, zero=state_zero,
                    native_bounds=native_bounds)
                # END no optimization record changes this original query.
                residuals.append(value)
                record_pair(index, component, first_id, second_id, evidence,
                    geometry_entity_key=entity_key,
                    permission_kind="none" if permission is None else "finite_region")
                report["pair_queries"] += 1
                status = evidence["status"]
                report["status_counts"][status] = report["status_counts"].get(status, 0)+1
                if status != "clear":
                    record_nonclear(index, component, first_id, second_id, evidence,
                                    float(status == "intersection"))
            value = torch.stack(residuals).reshape(progress.shape)
            costs[component] = costs.get(component, torch.zeros_like(progress))+value
            # Field uncertainty remains in status_counts and nonclear_pairs.
            # Only confirmed penetration contributes a collision residual.
        # END a zero-distance unresolved domain is not a clearance violation.
        # END assembly membership only removes internal pairs while Attachment is active.
        for model in self._initial_support_models.values():
            support_components = model.components(poses[model.relation.actor], poses[model.relation.target_owner],
                                                  progress, check_request=self.check_request)
            costs.update(support_components)
        # BEGIN only moving scene owners consume the declared unknown-space policy.
        from .observed_coverage import query_coverage
        report["owner_coverage"] = {}
        for owner in self.owners:
            if not owner.moving:
                continue
            unknown, unresolved = [], []
            rows = []
            for index in np.ndindex(tuple(progress.shape)):
                evidence = query_coverage(owner.query, matrices[owner.owner_id][index],
                    self.coverage, self.unknown_policy["conditional_regions"],
                    check_request=self.check_request)
                unknown.append(float(evidence["status"] == "unknown_witness"))
                unresolved.append(float(evidence["status"] == "unresolved"))
                if evidence["status"] != "covered" and len(rows) < 8:
                    rows.append({"state_index": list(index), **evidence})
            conditional = self.unknown_policy["outside_regions"] == "allow_attempt"
            prefix = "conditional/" if conditional else ""
            costs[f"{prefix}unknown/forbidden/{owner.owner_id}"] = progress.new_tensor(
                unknown).reshape(progress.shape)
            costs[f"{prefix}unresolved/coverage/{owner.owner_id}"] = progress.new_tensor(
                unresolved).reshape(progress.shape)
            report["owner_coverage"][owner.owner_id] = rows
        # END robot camera coverage cannot enter a refusal or recovery component.
        report["nonclear_pairs"] = nonclear_diagnostics()
        self.last_query_report = report
        return costs

    # BEGIN one deterministic finite objective serves a constrained optimizer.
    def optimization_objective(self, position, quaternion, progress, joint_position,
                               joint_names, *, trajectory_time, robot_link_poses,
                               contact_permission_progress=None, report_out=None):
        """Return a scalar objective using original geometry and current poses.

        Inputs have batch/time leading dimensions and each row is stably
        ordered by physical time, retaining the event's declared side order
        at equal times. Quaternions are unit wxyz rotations produced by FK.
        ``robot_link_poses`` maps
        every required reference-frame robot link to differentiable position
        and wxyz quaternion tensors from the caller's calibrated FK. Entity
        fixed transforms, owner attachment/release relations, hand certainty,
        finite contact regions and initial-support cell identities retain
        their existing scene definitions. Material samples remain fixed for
        the whole solve. Original ``components`` and interval queries retain
        all collision, missing-contact and unknown-space decisions.

        The finite sample objective can miss an overlap between its material
        points. Original support/permission branch switches and closest-feature
        ties are nondifferentiable. A constant zero field contributes no TSDF
        direction. These are objective-scope facts reported to the caller;
        they do not change any scene acceptance criterion.
        """
        from .sampled_geometry_objective import (MaterialSamples, pair_objective,
            contact_objective, native_pair_objective)

        fields = {owner.owner_id: owner.query for owner in self.owners}
        fields["background"] = self.background
        if self._objective_material_samples is None:
            geometries = {**{("field", name): field for name, field in fields.items()},
                          **{("robot", name): row[1] for name, row in self.robot_model._geometries.items()}}
            prepared = {identity: MaterialSamples(geometry) for identity, geometry in
                        {id(geometry): geometry for geometry in geometries.values()}.items()}
            self._objective_material_samples = {key: prepared[id(geometry)]
                                                for key, geometry in geometries.items()}
        samples = self._objective_material_samples
        report = {"scope": "fixed_original_material_samples",
                  "derivative_scope": "unchanged_support_permission_and_nearest_feature_branches",
                  "acceptance_source": "original_entity_and_continuous_interval_queries",
                  "finite_samples_do_not_cover_all_intersections": True,
                  "zero_field_has_no_tsdf_direction": True,
                  "pair_state_count": 0, "native_pair_state_count": 0,
                  "zero_contribution_separated_pair_count": 0,
                  "sample_count_scope": "material_points_actually_queried",
                  "internal_assembly_pair_count": 0,
                  "unmodelled_pair_count": 0, "contact_without_zero_samples": [],
                  "fixed_sample_counts": {f"{kind}/{name}": len(item.points)
                                          for (kind, name), item in samples.items()}}
        permission_progress = progress if contact_permission_progress is None else contact_permission_progress
        hand_state, coordinates = None, None
        if self.gripper is not None:
            hand_state = self.gripper.state(progress, joint_position, joint_names,
                                           trajectory_time=trajectory_time, measured=False)
            coordinates = hand_state.coordinates
        uncertain = (np.zeros(tuple(progress.shape), bool) if hand_state is None
                     else hand_state.uncertain.detach().cpu().numpy())
        unmodelled = (set() if hand_state is None else
                     set(self.robot_model.geometry_depending_on(self.gripper.joint_names)))
        robot_poses = {}
        for name, (frame, _, _, local) in self.robot_model._geometries.items():
            link_position, link_quaternion = robot_link_poses[frame]
            local_quaternion = _tensor(Rotation.from_matrix(local[:3, :3]).as_quat()[[3, 0, 1, 2]],
                                       link_quaternion)
            robot_poses[name] = (
                link_position + quaternion_rotate(link_quaternion, _tensor(local[:3, 3], link_position)),
                quaternion_multiply(link_quaternion, local_quaternion))
        robot_matrices = {name: _matrices(pose) for name, pose in robot_poses.items()}
        relationship_components = {}
        poses = self._owner_poses(position, quaternion, progress, None, relationship_components)
        identity_quaternion = torch.zeros_like(quaternion)
        identity_quaternion[..., 0] = 1
        poses["background"] = torch.zeros_like(position), identity_quaternion
        matrices = {name: _matrices(pose) for name, pose in poses.items()}
        zero = position.sum() * 0 + quaternion.sum() * 0 + joint_position.sum() * 0
        values = []
        # Robot entities use the fixed official collision model: its shapes
        # and fixed configuration are reusable across tasks, with calibrated
        # FK placing them at each nominal trajectory state.
        for pair in self.robot_validator.required_pairs:
            for index in np.ndindex(tuple(progress.shape)):
                if uncertain[index] and {pair.first, pair.second} & unmodelled:
                    report["unmodelled_pair_count"] += 1
                    continue
                values.append(native_pair_objective(samples["robot", pair.first],
                    tuple(row[index] for row in robot_poses[pair.first]), robot_matrices[pair.first][index],
                    samples["robot", pair.second], tuple(row[index] for row in robot_poses[pair.second]),
                    robot_matrices[pair.second][index]))
                report["native_pair_state_count"] += 1
        policies = {}
        for rule in self.contacts:
            actor_pose = (position, quaternion) if rule.actor == "robot" else poses[rule.actor]
            policy, normal, conditional = self._permission(rule, actor_pose, poses[rule.target_owner],
                permission_progress, coordinates, uncertain_hand=None if hand_state is None else hand_state.uncertain)
            policies[rule.rule_id] = (policy["normals"].detach().cpu().numpy(),
                policy["bounds"].detach().cpu().numpy(),
                (policy["active"] & policy["normal_valid"]).detach().cpu().numpy())
            if rule.required:
                target_samples = samples["field", rule.target_owner]
                if not len(target_samples.zero_points):
                    report["contact_without_zero_samples"].append(rule.rule_id)
                for index in np.ndindex(tuple(progress.shape)):
                    if bool(policy["active"][index]) and not bool(conditional[index]):
                        values.append(normal[index].square() + contact_objective(target_samples,
                            policy["normals"][index], policy["bounds"][index]))
        moving = {owner.owner_id for owner in self.owners if owner.moving}
        pairs = []
        for name in self.robot_model._geometries:
            for target in fields:
                if name not in self.fixed_geometry_names or target in moving:
                    pairs.append(("robot", name, target))
        pairs.extend(("field", owner.owner_id, "background") for owner in self.owners if owner.moving)
        for first, second in combinations(self.owners, 2):
            if first.moving or second.moving:
                if not first.moving:
                    first, second = second, first
                pairs.append(("field", first.owner_id, second.owner_id))
        attachment_progress = progress.detach().cpu().numpy()
        permission_cpu = permission_progress.detach().cpu().numpy()
        attached = {owner.owner_id: np.logical_or.reduce(
            [relation.active(attachment_progress) for relation in owner.attachments])
            if owner.attachments else np.zeros(tuple(progress.shape), bool) for owner in self.owners}
        for kind, first, second in pairs:
            first_pose = robot_poses[first] if kind == "robot" else poses[first]
            actor = "robot" if kind == "robot" else first
            rules = [rule for rule in self.contacts if (
                kind == "robot" and rule.actor == "robot" and rule.robot_body == first
                and rule.target_owner == second) or (kind == "field"
                and {rule.actor, rule.target_owner} == {first, second})]
            support = [model for model in self._initial_support_models.values()
                       if model.relation.actor == actor and model.relation.target_owner == second]
            for index in np.ndindex(tuple(progress.shape)):
                if self.check_request is not None:
                    self.check_request()
                if kind == "robot" and uncertain[index] and first in unmodelled:
                    report["unmodelled_pair_count"] += 1
                    continue
                if second in attached and attached[second][index] and (
                    kind == "robot" and first in self.tool_geometry_names
                    or kind == "field" and first in attached and attached[first][index]):
                    report["internal_assembly_pair_count"] += 1
                    continue
                regions = []
                for rule in rules:
                    normals, bounds, active = policies[rule.rule_id]
                    if active[index]:
                        target_matrix = matrices[rule.target_owner][index]
                        world_normals = normals[index] @ target_matrix[:3, :3].T
                        region = world_normals, bounds[index] + world_normals @ target_matrix[:3, 3]
                        regions.append((region, region))
                permission = PairPermission(regions) if regions else None
                for model in support:
                    if model.relation.interval[0] <= permission_cpu[index] < model.relation.interval[1]:
                        permission = model.permission(matrices[second][index], additional=permission)
                values.append(pair_objective(samples[kind, first],
                    tuple(row[index] for row in first_pose), samples["field", second],
                    tuple(row[index] for row in poses[second]), permission=permission, report=report))
                report["pair_state_count"] += 1
        for model in self._initial_support_models.values():
            components = model.components(poses[model.relation.actor], poses[model.relation.target_owner],
                                          progress, check_request=self.check_request)
            values.extend((components[f"initial_support/{model.relation.relation_id}"]
                           / model.actor_field.voxel_size).square().reshape(-1).unbind())
        result = zero + (torch.stack(values).sum() / progress.numel() if values else zero)
        report["relationship_components"] = {
            key: value.detach().cpu().numpy().tolist() for key, value in relationship_components.items()}
        if report_out is not None:
            report_out.update(report)
        return result
    # END original classification and coverage remain in their existing APIs.

    def evaluate(self, *args, **kwargs):
        output = kwargs.pop("components_out", None)
        # BEGIN this aggregation contains original scene conditions only.
        components = self.components(*args, **kwargs)
        if output is not None:
            output.update(components)
        return torch.stack([value for name, value in components.items()
                            if not name.startswith("conditional/")]).sum(0)
        # END missing/unknown/undecided states do not masquerade as smooth costs.

    def estimated_workspace_allocation(self, batch, horizon):
        """Separate measured-RSS inventory from still-unallocated scene work."""
        fields = [self.background, *(owner.query for owner in self.owners)]
        unique_fields = tuple({id(value): value for value in fields}.values())
        # BEGIN resident observations and lazy scene allocations have distinct ownership.
        resident = sum(field.storage_nbytes for field in unique_fields)
        pending = 0
        # Native pair buffers are lazy and are replaced when the exact pair
        # layout changes. Only the already matching layout belongs to RSS;
        # otherwise retain the existing full-model allocation upper bound.
        schema = self.robot_model._native_distance_batches
        expected = (None if self.robot_validator is None else
                    tuple((pair.first, pair.second) for pair in self.robot_validator.required_pairs))
        if schema is not None and schema[0] == expected:
            resident += self.robot_model.native_query_storage_nbytes
        else:
            pending += self.robot_model.native_query_storage_nbytes
        # Each original cell has at most eight zero vertices and twelve edge
        # roots. Deduplication can reduce this storage, so the preconstruction
        # bound covers fixed material arrays; the temporary bound below also
        # covers construction's zero-point list and NumPy working arrays.
        material_counts = [sum(len(field.occupied_points(node)) for node in field.nodes
                               if node.cell >= 0) for field in unique_fields]
        zero_counts = [20 * len(field.cells) for field in unique_fields]
        # BEGIN material selection shares one immutable range index per field.
        # Each original cell owns two int64 offsets; three float64 values
        # bound actual material-to-tree-centre coordinate differences.
        index_bytes = [16 * len(field.cells) + 24 for field in unique_fields]
        if self._objective_material_samples is None:
            pending += sum(32 * count + 24 * zeros + 96 + index for count, zeros, index in
                           zip(material_counts, zero_counts, index_bytes))
            pending += len(self.robot_model.geometry_names) * (27 * 32 + 96)
        else:
            samples = {id(value): value for value in self._objective_material_samples.values()}.values()
            resident += sum(value.points.nbytes + value.cells.nbytes + value.zero_points.nbytes
                            + (0 if value.point_bounds is None else value.point_bounds.nbytes)
                            + (0 if value.negative_bounds is None else value.negative_bounds.nbytes)
                            + (0 if value.cell_ranges is None else value.cell_ranges.nbytes + value.cell_radius.nbytes)
                            for value in samples)
        rows = max((len(rule.feature_points) + 10 for rule in self.contacts), default=0)
        # The additional 64 bytes/material record cover expanded int64 point
        # indices and their repeat buffer, selected coordinate copies, and
        # active-index mapping. The 96 bytes/cell cover tree result conversion,
        # centre/axis masks, ordered ranges and cumsum/repeat offsets. Their
        # lifetime is one direction of one pair, independently of horizon.
        temporary = max([0, *(320 * count + 96 * len(field.cells)
                              for count, field in zip(material_counts, unique_fields)),
                         *((200 + 8 * rows) * count for count in zero_counts)])
        # END construction and resident phases account for the same unique index.
        # Point derivatives are reduced to entity-pose derivatives pairwise;
        # the trajectory retains no material-point-sized autograd arrays.
        entities = len(self.robot_model.geometry_names)+len(self.owners)+1
        states = int(batch)*int(horizon)
        live = states*(entities*256+entities*entities*512)
        # The scene geometry and Pinocchio workspaces reside on the host.
        # Native IK can pass device joints through the CPU FK adapter; count
        # its named input and link-pose output together with their reverse
        # gradient copies at float64 capacity. No field samples live on CUDA.
        transfer = 16 * states * (len(self.robot_model.joint_names)
                                  + 7 * len(self.required_robot_links))
        return dict(cpu_resident=resident, cpu_pending_shared=pending, cpu_live=live+temporary, gpu_shared=0,
                    gpu_live=transfer, gpu_retained=0, gpu_graph_live=0)
        # END each admission adds only pending arrays and one simultaneous query.

    def describe(self):
        return {"model": "native_robot_observed_scene_v1", "reference_frame": self.reference_frame,
            "robot_geometry_source": "registered_native_model",
            "owners": [owner.owner_id for owner in self.owners],
            "robot_entities": list(self.robot_model.geometry_names),
            "fixed_geometry_names": list(self.fixed_geometry_names),
            "tool_geometry_names": list(self.tool_geometry_names),
            "unknown_policy": deepcopy(self.unknown_policy), "robot_view_coverage_status": "not_queried",
            "collision_rule": "nonpenetration",
            "required_progress_anchors": self.required_progress_anchors,
            "observation_report": deepcopy(self.observation_report)}

    def to_execution_binding(self):
        # BEGIN scene and task references share the same typed path records.
        from .execution_binding import plain
        # END owner-motion validity is still defined by CollisionOwner.
        owners = []
        for owner in self.owners:
            path = owner.pose_path
            owners.append({"owner_id": owner.owner_id, "source": owner.source,
                "moving": owner.moving,
                "pose_interval": list(owner.pose_interval),
                "pose_path": plain(path),
                "attachments": _plain([asdict(row) for row in owner.attachments]),
                "query": owner.query.record()})
        hand = self.gripper
        hand_record = None
        if hand is not None:
            hand_record = {name: _plain(getattr(hand, name)) for name in (
                "controller", "joint_names", "current_positions", "joint_limits", "coupling_matrix",
                "coupling_offset", "commands", "source", "control_timestep_s", "numerical_joint_tolerance")}
            hand_record["nominal_model"] = (None if hand.nominal_finger_trajectory is None
                                             else hand.nominal_finger_trajectory.describe())
        # BEGIN retain original geometry, calibration, policy and named FK inputs.
        native = {**self.robot_model.model_record(),
            "base_frame": self.robot_model.base_frame,
            "T_reference_base": (None if self.robot_validator is None
                                 else self.robot_validator.T_reference_base.tolist()),
            "pairs": [asdict(pair) for pair in self.robot_pairs]}
        return {"schema": "native_robot_observed_scene_v1", **self.describe(), "owners": owners,
            "robot_model": native,
            "background": self.background.record(), "contacts": _plain([asdict(row) for row in self.contacts]),
            "tool_geometry_names": list(self.tool_geometry_names),
            "grasp_contact_activations": _plain(self.grasp_contact_activations),
            "initial_support_relations": [row.record() for row in self.initial_support_relations],
            "gripper": hand_record, "has_gripper": self.has_gripper,
            "numerical_position_tolerance_m": self.numerical_position_tolerance_m,
            "numerical_rotation_tolerance_rad": self.numerical_rotation_tolerance_rad,
            "coverage": {"known_free": encode_array(self.coverage.known_free),
                "origin": self.coverage.origin.tolist(), "voxel_size": self.coverage.voxel_size,
                "observation_ids": list(self.coverage.observation_ids)}}
        # END scene serialization does not recreate a robot observation field.

    binding_record = to_execution_binding
# END observations remain fixed while each candidate supplies its own rigid poses.
