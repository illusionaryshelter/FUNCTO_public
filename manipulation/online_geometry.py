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
from .implicit_distance import PairPermission, transform_points
from .observed_mapping import FrozenObservedField
from .observed_robot import DifferentiableRobotKinematics
from .robot_geometry_check import RobotGeometryChecker
from .observed_collision import pair_residual
from .planning_problem import PosePath, quaternion_rotate, quaternion_conjugate, quaternion_multiply
from .planning_scene import (Attachment, CollisionOwner, ContactRule, ObservedSpace,
                             PrismaticGripper, _tensor, _interval, _world_points)
from .optimizer_device_checks import constant_tensor
from .holding_contact import HoldingContact, validate_holding_conditions


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
                 robot_model, fixed_geometry_names, contacts=(), gripper=None,
                 has_gripper=True, unknown_policy, holding_contacts=(),
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
        robot_ids = set(robot_model.geometry_names)
        if (not set(self.fixed_geometry_names) <= robot_ids
                or len(set(self.fixed_geometry_names)) != len(self.fixed_geometry_names)
                or len({owner.owner_id for owner in self.owners}) != len(self.owners)):
            raise ValueError("Fixed installations and scene owners need explicit unique identities")
        self.required_robot_links = tuple(dict.fromkeys(
            frame for frame, _, _, _ in robot_model._geometries.values()))
        self.contacts, self.holding_contacts = tuple(contacts), tuple(holding_contacts)
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
        if len({rule.rule_id for rule in (*self.contacts, *self.holding_contacts)}) != len(self.contacts)+len(self.holding_contacts):
            raise ValueError("Contact relation identities must be unique")
        for rule in self.contacts:
            if (rule.target_owner not in owner_ids or rule.actor not in {"robot", *owner_ids}
                    or rule.actor == rule.target_owner
                    or (rule.actor == "robot" and rule.robot_body not in robot_ids)):
                raise ValueError("Finite features must name registered distinct entities")
        for relation in self.holding_contacts:
            if relation.robot_body not in robot_ids or relation.target_owner not in owner_ids:
                raise ValueError("An internal attachment contact needs its exact registered body")
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
        self.robot_configuration, self.robot_pairs = {}, ()
        self.robot_validator, self.kinematics = None, None
        self.check_request = None
        self.last_query_report = {}
    # END neither image visibility nor joint dependence defines model roles.

    # BEGIN bind the same known model to the complete differentiable joint tree.
    def bind_online_robot(self, *, configuration, validator, device, check_request=None):
        """Bind the complete joint tree; each query supplies its own finger state."""
        if validator.checker is not self.robot_model:
            raise ValueError("Scene and validator must share the same known robot model")
        self.robot_validator, self.robot_pairs = validator, validator.pairs
        self.robot_configuration = deepcopy(configuration)
        self.kinematics = DifferentiableRobotKinematics(configuration,
            frames=self.required_robot_links, T_reference_base=validator.T_reference_base, device=device)
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
        validate_holding_conditions(self.holding_contacts, self.owners, problem.conditions)
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

    def components(self, robot_spheres, position, quaternion, progress,
                   joint_position=None, joint_names=None, *, trajectory_time,
                   active_contact_rule_ids=None, contact_permission_progress=None,
                   owner_poses=None, query_role="acceptance"):
        """Return same-state residuals and retain distance/coverage diagnostics.

        ``robot_spheres`` is an empty native-kinematics output. Registered
        native entities carry all robot geometry. Query roles identify the
        caller in diagnostics; both roles use the same nonpenetration rule.
        """
        if query_role not in {"optimization", "acceptance"}:
            raise ValueError("Observed geometry requires an explicit query role")
        if self.kinematics is None or joint_position is None:
            raise ValueError("Observed robot geometry must be bound to its named joint tree")
        # BEGIN caller role does not introduce a geometric clearance.
        costs = {}
        permission_progress = progress if contact_permission_progress is None else contact_permission_progress
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
        poses = self._owner_poses(position, quaternion, progress, owner_poses, costs)
        identity = (torch.zeros_like(position), torch.zeros_like(quaternion))
        identity[1][..., 0] = 1
        poses["background"] = identity
        fields = {owner.owner_id: owner.query for owner in self.owners}
        fields["background"] = self.background
        matrices = {name: _matrices(pose) for name, pose in poses.items()}
        robot_matrices = {name: _matrices(pose) for name, pose in robot_poses.items()}
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
                    gap=progress[index]*0+float(evidence["status"]=="missing_contact")
                    if evidence["status"]=="missing_contact":
                        # A nearby zero supplies a local optimization direction;
                        # the interval query above owns contact classification.
                        actor_matrix=_matrices(actor_pose)[index]
                        local_center=np.mean(rule.feature_points,axis=0)
                        world_center=transform_points(local_center,actor_matrix)
                        target_center=(world_center-target_matrix[:3,3])@target_matrix[:3,:3]
                        boundary=target_field.local_zero(target_center)
                        if boundary is not None:
                            point=progress.new_tensor(boundary[1])
                            excess=torch.relu((policy["normals"][index]*point).sum(-1)-policy["bounds"][index])
                            direction=torch.linalg.vector_norm(excess)
                            gap=gap+direction-direction.detach()
                    gaps.append(gap)
                    undecided.append(float(evidence["status"] == "unresolved"))
                costs[f"contact/{rule.rule_id}"] = normal*policy["active"].to(normal.dtype)*(1-uncertain)+torch.stack(gaps).reshape(progress.shape)
                costs[f"unresolved/contact/{rule.rule_id}"] = progress.new_tensor(undecided).reshape(progress.shape)
                costs[f"conditional/contact/{rule.rule_id}"] = uncertain
        # A pair owns only its declared feature regions; no rule can exempt
        # an entire finger because a smaller pad has missing observations.
        progress_cpu = permission_progress.detach().cpu().numpy()
        uncertain_cpu = (np.zeros(tuple(progress.shape), bool) if hand_state is None
                         else hand_state.uncertain.detach().cpu().numpy())
        unmodelled = (set() if hand_state is None else
                     set(self.robot_validator.checker.geometry_depending_on(self.gripper.joint_names)))
        # BEGIN native self checks use the same validator pair contract.
        report = {"query_role": query_role, "collision_rule": "nonpenetration", "pair_queries": 0,
                  "status_counts": {}, "nonclear_pairs": [],
                  "robot_geometry_source": "registered_native_model",
                  "robot_view_coverage_status": "not_queried",
                  "coverage_scope": "represented_observed_occupancy"}
        from .implicit_distance import inverse_points
        from .observed_collision import _world
        q_cpu = complete_q.detach().cpu().numpy()
        self_costs, self_unresolved = [], []
        for index in np.ndindex(tuple(progress.shape)):
            evidence = self.robot_model.query(joint_names=complete_names,
                joint_positions=q_cpu[index], T_reference_base=self.robot_validator.T_reference_base,
                pairs=self.robot_pairs, check_request=self.check_request)
            value = complete_q[index].sum() * 0
            undecided = value
            for row in evidence["pairs"]:
                if row["policy"] != "required":
                    continue
                first_id, second_id = row["first"], row["second"]
                if uncertain_cpu[index] and ({first_id, second_id} & unmodelled):
                    continue
                status = row["relation"]
                report["pair_queries"] += 1
                report["status_counts"][status] = report["status_counts"].get(status, 0) + 1
                if status == "numerical_boundary":
                    undecided = undecided + 1.
                elif status == "penetrating":
                    # Native witnesses supply a local rigid-motion direction;
                    # the original signed query alone owns the verdict.
                    a = inverse_points(row["witness_first"], robot_matrices[first_id][index])
                    b = inverse_points(row["witness_second"], robot_matrices[second_id][index])
                    pa = tuple(item[index] for item in robot_poses[first_id])
                    pb = tuple(item[index] for item in robot_poses[second_id])
                    normal = _tensor(row["normal_first_to_second"], complete_q)
                    signed = ((_world(b, pb) - _world(a, pa)) * normal).sum()
                    value = value - row["signed_distance_m"] - (signed - signed.detach())
                if status != "separated" and len(report["nonclear_pairs"]) < 32:
                    report["nonclear_pairs"].append({"component": "collision/robot/self",
                        "state_index": list(index), **row})
            self_costs.append(value)
            self_unresolved.append(undecided)
        costs["collision/robot/self"] = torch.stack(self_costs).reshape(progress.shape)
        costs["unresolved/robot/self"] = torch.stack(self_unresolved).reshape(progress.shape)
        # END camera coverage and task margins cannot change self legality.
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
        for component, first_id, second_id, first, second, first_pose, second_pose, fm, sm, actor, target in pairs:
            residuals = []
            rules = [rule for rule in self.contacts
                     if ((actor == "robot" and rule.actor == "robot" and rule.target_owner == target
                          and rule.robot_body == first_id)
                         or (actor != "robot" and {rule.actor, rule.target_owner} == {first_id, second_id}))]
            support = [model for model in self._initial_support_models.values()
                       if model.relation.actor == actor and model.relation.target_owner == target]
            for index in np.ndindex(tuple(progress.shape)):
                if self.check_request is not None:
                    self.check_request()
                zero = first_pose[0][index].sum()*0+second_pose[0][index].sum()*0
                if uncertain_cpu[index] and (first_id in unmodelled or second_id in unmodelled):
                    residuals.append(zero)
                    continue
                held_internal = any(row.robot_body == first_id and row.target_owner == target
                    and row.interval[0] <= progress_cpu[index] < row.interval[1]
                    for row in self.holding_contacts) if actor == "robot" else False
                if held_internal:
                    residuals.append(zero)
                    continue
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
                value, evidence = pair_residual(first, (first_pose[0][index], first_pose[1][index]), fm[index],
                    second, (second_pose[0][index], second_pose[1][index]), sm[index],
                    permission=permission, check_request=self.check_request)
                residuals.append(value)
                report["pair_queries"] += 1
                status = evidence["status"]
                report["status_counts"][status] = report["status_counts"].get(status, 0)+1
                if status != "clear" and len(report["nonclear_pairs"]) < 32:
                    report["nonclear_pairs"].append({"first": first_id, "second": second_id,
                        "component": component, "state_index": list(index), **evidence})
            value = torch.stack(residuals).reshape(progress.shape)
            costs[component] = costs.get(component, torch.zeros_like(progress))+value
            # Field uncertainty remains in status_counts and nonclear_pairs.
            # Only confirmed penetration contributes a collision residual.
        # END a zero-distance unresolved domain is not a clearance violation.
        for model in self._initial_support_models.values():
            costs.update(model.components(poses[model.relation.actor], poses[model.relation.target_owner],
                                         progress, check_request=self.check_request))
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
        self.last_query_report = report
        return costs or {"observed_geometry": torch.zeros_like(progress)}

    def evaluate(self, *args, **kwargs):
        output = kwargs.pop("components_out", None)
        components = self.components(*args, **kwargs)
        if output is not None:
            output.update(components)
        return torch.stack([value for name, value in components.items()
                            if not name.startswith("conditional/")]).sum(0)

    def estimated_workspace_allocation(self, batch, horizon):
        fields = [self.background, *(owner.query for owner in self.owners)]
        shared = sum(field.storage_nbytes for field in {id(value): value for value in fields}.values())
        shared += self.robot_model.native_query_storage_nbytes
        # Queries run serially on CPU; retained autograd data consists of
        # rigid poses and pair witnesses rather than dense sample-grid copies.
        entities = len(self.robot_model.geometry_names)+len(self.owners)+1
        states = int(batch)*int(horizon)
        live = states*(entities*256+entities*entities*256)
        return dict(cpu_shared=shared, cpu_live=live, gpu_shared=0,
                    gpu_live=live, gpu_retained=0, gpu_graph_live=0)

    def describe(self):
        return {"model": "native_robot_observed_scene_v1", "reference_frame": self.reference_frame,
            "robot_geometry_source": "registered_native_model",
            "owners": [owner.owner_id for owner in self.owners],
            "robot_entities": list(self.robot_model.geometry_names),
            "fixed_geometry_names": list(self.fixed_geometry_names),
            "unknown_policy": deepcopy(self.unknown_policy), "robot_view_coverage_status": "not_queried",
            "collision_rule": "nonpenetration",
            "required_progress_anchors": self.required_progress_anchors,
            "observation_report": deepcopy(self.observation_report)}

    def to_execution_binding(self):
        owners = []
        for owner in self.owners:
            path = owner.pose_path
            owners.append({"owner_id": owner.owner_id, "source": owner.source,
                "moving": owner.moving,
                "pose_interval": list(owner.pose_interval),
                "pose_path": {"progress": _plain(path.progress), "poses": _plain(path.poses), "source": path.source,
                    "motion_model": None if path.motion_model is None else _plain(asdict(path.motion_model)),
                    "coordinates": _plain(path.coordinates), "attachment": _plain(path.attachment)},
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
            "robot_model": native, "robot_configuration": deepcopy(self.robot_configuration),
            "background": self.background.record(), "contacts": _plain([asdict(row) for row in self.contacts]),
            "holding_contacts": _plain([asdict(row) for row in self.holding_contacts]),
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
