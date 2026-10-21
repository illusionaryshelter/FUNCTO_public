"""Whole-entity separation on the original continuous joint reference.

The checker combines pose-query distance bounds with point-speed bounds over
time intervals. It reports proofs, represented intersections and unresolved
intervals separately. Observed free-space coverage and execution decisions
remain responsibilities of the scene and its caller.
"""

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry_motion_bound import (
    SphereBody, build_geometry_motion_bound, _add_up, _mul_up, _norm_up,
    _linear_norm_up,
)
from .implicit_distance import box_corners


def _cpu(value):
    return value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)


def _polynomial_ranges(curve, start, end, derivative=0):
    """Enclose a cubic or its derivative using Bernstein convex hulls.

    Every intersected original polynomial piece contributes. Power-to-
    Bernstein conversion uses long-double intermediates and a binary64
    outward allowance proportional to the arithmetic's absolute terms.
    This is an interval bound, rather than a sampled derivative maximum.
    """
    polynomial = curve.derivative(derivative)
    degree = polynomial.c.shape[0] - 1
    lower = np.full(polynomial.c.shape[2:], np.inf)
    upper = np.full_like(lower, -np.inf)
    from math import comb
    for i, (left, right) in enumerate(zip(polynomial.x[:-1], polynomial.x[1:])):
        a, b = max(float(start), float(left)), min(float(end), float(right))
        if b <= a:
            continue
        x, h = np.longdouble(a-left), np.longdouble(b-a)
        source = polynomial.c[:, i].astype(np.longdouble)[::-1]
        shifted = []
        arithmetic_scale = np.zeros_like(source[0])
        for k in range(degree+1):
            terms = [source[j]*comb(j, k)*x**(j-k)*h**k for j in range(k, degree+1)]
            shifted.append(sum(terms))
            arithmetic_scale += sum(np.abs(term) for term in terms)
        controls = np.asarray([sum(shifted[k]*np.longdouble(comb(i, k))/comb(degree, k)
                                   for k in range(i+1)) for i in range(degree+1)])
        error = np.longdouble(128*np.finfo(float).eps)*arithmetic_scale
        lo = np.nextafter(np.asarray(controls.min(0)-error, float), -np.inf)
        hi = np.nextafter(np.asarray(controls.max(0)+error, float), np.inf)
        # Exact constant coordinates retain structural zero velocity.
        if derivative:
            structural_zero = np.all(source == 0, axis=0)
            lo = np.where(structural_zero, 0., lo)
            hi = np.where(structural_zero, 0., hi)
        lower, upper = np.minimum(lower, lo), np.maximum(upper, hi)
    return lower, upper


def _progress_times(trajectory, values):
    """Map progress events to all physical crossings, including plateaus."""
    times = np.asarray(trajectory.progress_u, float)*trajectory.duration
    progress = np.asarray(trajectory.progress_s, float)
    output = []
    for value in values:
        output.extend(times[progress == value].tolist())
        for a, b, sa, sb in zip(times[:-1], times[1:], progress[:-1], progress[1:]):
            if sa < value < sb:
                output.append(float(a+(b-a)*(value-sa)/(sb-sa)))
    return output


@dataclass
class SceneMotionBounds:
    """Point-speed coefficients bound to one scene's complete entity models.

    Local enclosing spheres are used only to integrate rotational velocity
    into displacement bounds. Collision queries retain the original native
    entities and original observed fields.
    """
    scene: object
    robot: object
    attached_names: dict
    owner_radii: dict

    @classmethod
    def from_scene(cls, scene, *, tool_link, T_link_tool):
        bodies, attached_names, radii = [], {}, {}
        for name, (frame, _, _, local) in scene.robot_model._geometries.items():
            source_radius = max(_norm_up(point) for point in box_corners(*scene._native_local_bounds[name]))
            radius = _add_up(_norm_up(local[:3, 3]),
                _mul_up(_linear_norm_up(local[:3, :3]), source_radius))
            bodies.append(SphereBody(f"robot/{name}", frame, [[0., 0., 0., radius]], {}))
        for owner in scene.owners:
            if owner.query.root is None:
                radius = 0.
                corners = np.zeros((1, 3))
            else:
                node = owner.query.nodes[owner.query.root]
                corners = box_corners(node.lower, node.upper)
                radius = max(_norm_up(point) for point in corners)
            radii[owner.owner_id] = radius
            for index, relation in enumerate(owner.attachments):
                local = np.eye(4)
                local[:3, :3] = Rotation.from_quat(
                    np.asarray(relation.quaternion_tool_owner)[[1, 2, 3, 0]]).as_matrix()
                local[:3, 3] = relation.position_tool_owner
                local_radius = _add_up(_norm_up(local[:3, 3]),
                    _mul_up(_linear_norm_up(local[:3, :3]), radius))
                tool = np.asarray(T_link_tool, float)
                complete_radius = _add_up(_norm_up(tool[:3, 3]),
                    _mul_up(_linear_norm_up(tool[:3, :3]), local_radius))
                name = f"attachment/{owner.owner_id}/{index}"
                attached_names[(owner.owner_id, index)] = name
                bodies.append(SphereBody(name, tool_link,
                    [[0., 0., 0., complete_radius]], {}))
        params = scene.kinematics.model.config.kinematics_config
        arguments = {key: _cpu(getattr(params, key)) for key in
                     ("fixed_transforms", "link_map", "joint_map", "joint_map_type", "joint_offset_map")}
        robot = build_geometry_motion_bound(**arguments, joint_names=tuple(params.joint_names),
            joint_position_limits=_cpu(params.joint_limits.position),
            link_name_to_idx_map=params.link_name_to_idx_map, bodies=bodies, extra_joint_limits={})
        return cls(scene, robot, attached_names, radii)

    def declaration_times(self, trajectory):
        """Map event times to the actual left/right relation limits.

        A progress plateau has an incoming side only at its entrance and an
        outgoing side only at its exit. Its interior retains the exact mode.
        """
        result = {}
        for value in self.scene.required_progress_anchors:
            crossing = _progress_times(trajectory, [value])
            if not crossing:
                continue
            first, last = min(crossing), max(crossing)
            if value > trajectory.progress_s[0]:
                result.setdefault(first, set()).add(-1)
            if value < trajectory.progress_s[-1]:
                result.setdefault(last, set()).add(1)
        return result

    def partition_times(self, trajectory):
        result = [0., trajectory.duration,
                  *(np.asarray(trajectory.progress_u)*trajectory.duration),
                  *self.declaration_times(trajectory)]
        for owner in self.scene.owners:
            a, b = owner.pose_interval
            result.extend(_progress_times(trajectory, a+(b-a)*owner.pose_path.progress))
        gripper = self.scene.gripper
        if gripper is not None and gripper.nominal_finger_trajectory is not None:
            # This physical-time boundary changes only hand model validity.
            # It does not declare a different progress/contact relation side.
            result.append(float(gripper.nominal_finger_trajectory.valid_until_s))
        return np.unique([float(value) for value in result if 0. <= value <= trajectory.duration])

    def _owner_speed(self, owner, start_progress, end_progress, duration):
        middle = (start_progress+end_progress)/2
        radius = self.owner_radii[owner.owner_id]
        for index, relation in reversed(tuple(enumerate(owner.attachments))):
            if relation.active(middle):
                return self.attached_names[(owner.owner_id, index)]
            if relation.release_condition_id is not None and middle >= relation.interval[1]:
                return 0.
        a, b = owner.pose_interval
        u0, u1 = np.clip((np.asarray([start_progress, end_progress])-a)/(b-a), 0., 1.)
        if u0 == u1 or len(owner.pose_path.progress) == 1:
            return 0.
        path = owner.pose_path
        rate = float(np.nextafter((u1-u0)/duration, np.inf))
        result = 0.
        reach = _add_up(_norm_up(path.attachment[:3, 3]),
                        _mul_up(_linear_norm_up(path.attachment[:3, :3]), radius))
        for i, (left, right) in enumerate(zip(path.progress[:-1], path.progress[1:])):
            if right <= u0 or left >= u1:
                continue
            scale = _mul_up(rate, float(np.nextafter(1/(right-left), np.inf)))
            if path.motion_model is None:
                linear = _norm_up(path.poses[i+1, :3, 3]-path.poses[i, :3, 3])
                angle = float((Rotation.from_matrix(path.poses[i, :3, :3]).inv()*
                               Rotation.from_matrix(path.poses[i+1, :3, :3])).magnitude())
                speed = _add_up(linear, _mul_up(np.nextafter(angle, np.inf), reach))
            else:
                model = path.motion_model
                delta = path.coordinates[i+1]-path.coordinates[i]
                if model.kind == "revolute":
                    arm = _add_up(_norm_up(model.reference_pose[:3, 3]-model.axis_point), reach)
                    speed = _mul_up(abs(float(delta[0])), _mul_up(_norm_up(model.axis), arm))
                elif model.kind == "prismatic":
                    speed = _mul_up(abs(float(delta[0])), _norm_up(model.axis))
                else:
                    linear = _norm_up(model.basis[:, :2] @ delta[:2])
                    rotation = 0. if model.kind == "translation_plane" else _mul_up(
                        abs(float(delta[2])), _mul_up(_norm_up(model.axis), reach))
                    speed = _add_up(linear, rotation)
            result = max(result, _mul_up(scale, speed))
        return result

    def interval(self, trajectory, start, end):
        """Return entity speed bounds and explicit unavailable coordinates.

        The caller first verifies continuous joint positions against the same
        model limits. The planner performs this strict polynomial-domain
        check before invoking interval geometry. A nominal hand curve must
        likewise retain its declared model's coordinate domain.
        """
        lower, upper = _polynomial_ranges(trajectory._curve, start, end, 1)
        rates = dict(zip(trajectory.joint_names, np.maximum(abs(lower), abs(upper))))
        gripper = self.scene.gripper
        nominal = None if gripper is None else gripper.nominal_finger_trajectory
        if nominal is not None and end <= nominal.valid_until_s:
            lo, hi = _polynomial_ranges(nominal._curve, start, end, 1)
            for name, value in zip(nominal.joint_names, np.maximum(abs(lo), abs(hi))):
                if name not in rates:
                    rates[name] = value
        speeds, unavailable = {"background": 0.}, {}
        for name, coefficients in zip(self.robot.body_names, self.robot.coefficients):
            missing = [joint for joint, coefficient in zip(self.robot.joint_names, coefficients)
                       if coefficient != 0 and joint not in rates]
            if missing:
                unavailable[name] = {"reason": "unmodelled_joint_motion", "joint_names": missing}
            else:
                value = 0.
                for joint, coefficient in zip(self.robot.joint_names, coefficients):
                    value = _add_up(value, _mul_up(float(coefficient), float(rates.get(joint, 0.))))
                speeds[name] = value
        s0, s1 = np.interp(np.asarray([start, end])/trajectory.duration,
                           trajectory.progress_u, trajectory.progress_s)
        for owner in self.scene.owners:
            name = f"owner/{owner.owner_id}"
            value = self._owner_speed(owner, float(s0), float(s1), end-start)
            if isinstance(value, str):
                if value in unavailable:
                    unavailable[name] = unavailable[value]
                else:
                    speeds[name] = speeds[value]
            else:
                speeds[name] = value
        # Auxiliary attachment covers describe alternative relation states;
        # only each owner's currently active cover contributes to this span.
        for name in self.attached_names.values():
            speeds.pop(name, None)
            unavailable.pop(name, None)
        return {"speeds": speeds, "unavailable": unavailable}


def check_interval_geometry(trajectory, motion, query_geometry, *, max_query_points,
                            spatial_step_m, initial_states=()):
    """Refine unresolved time intervals using bounded whole-entity movement.

    ``query_geometry(times_s, point_budget=...)`` returns a mapping with
    ``states``, actual physical ``query_points`` (including required anchors),
    and ``capacity_exhausted``. Each state contains ``time_s``, ``side``
    (-1/0/+1), and complete ``pairs``. Each pair's ``geometry_entity_key``
    names first/second entities as ``robot/<geometry ID>``, ``owner/<owner ID>``
    or ``background``; bare first/second names are display identities.
    Multiple relation sides at one physical
    time are distinct evidence states. Initial states were already charged by
    the caller; ``max_query_points`` is its remaining shared point budget.
    """
    if int(max_query_points) != max_query_points or max_query_points < 0:
        raise ValueError("Interval checking requires a nonnegative integer remaining point budget")
    if not np.isfinite(spatial_step_m) or spatial_step_m <= 0:
        raise ValueError("Spatial subdivision requires a positive finite maximum movement")
    cache, intersections, seen_intersections = {}, [], set()
    queried_times, consumed = set(), 0

    def retain(states):
        for state in states:
            key = (float(state["time_s"]), int(state["side"]))
            cache[key] = state
            for pair in state["pairs"]:
                entities = pair["geometry_entity_key"]
                identity = (*key, pair["component"], entities["first"], entities["second"])
                if pair["status"] == "intersection" and identity not in seen_intersections:
                    seen_intersections.add(identity)
                    intersections.append({"time_s": key[0], "side": key[1], **pair})

    retain(initial_states)
    declarations = motion.declaration_times(trajectory)
    partition = np.asarray(motion.partition_times(trajectory), float)
    pending = deque(zip(partition[:-1], partition[1:]))
    proved, unresolved = [], []

    def get_states(start, end):
        nonlocal consumed
        keys = [(float(start), 1 if 1 in declarations.get(start, ()) and start > 0 else 0),
                (float(end), -1 if -1 in declarations.get(end, ()) and end < trajectory.duration else 0)]
        needed = sorted({time for time, side in keys if (time, side) not in cache})
        exhausted = False
        if needed:
            response = query_geometry(needed, point_budget=int(max_query_points-consumed))
            used = int(response["query_points"])
            if used < 0 or used > max_query_points-consumed:
                raise ValueError("Geometry query exceeded its explicit physical point budget")
            consumed += used
            retain(response["states"])
            queried_times.update(float(state["time_s"]) for state in response["states"])
            exhausted = bool(response["capacity_exhausted"])
        return [cache.get(key) for key in keys], exhausted

    while pending:
        start, end = map(float, pending.popleft())
        if not intersections:
            states, exhausted = get_states(start, end)
        if intersections:
            unresolved.extend({"start_s": float(a), "end_s": float(b),
                "maximum_entity_movement_m": None,
                "reason": "not_checked_after_intersection"}
                for a, b in [(start, end), *pending])
            pending.clear()
            break
        bounds = motion.interval(trajectory, start, end)
        speeds, unavailable = bounds["speeds"], bounds["unavailable"]
        movement = _mul_up(max(speeds.values(), default=0.), end-start)
        base = {"start_s": start, "end_s": end,
                "maximum_entity_movement_m": None if unavailable else movement}
        if any(state is None for state in states):
            unresolved.append({**base, "reason": "sampling_capacity" if exhausted else "missing_boundary_side"})
            continue
        maps = [{(row["component"], row["geometry_entity_key"]["first"],
                  row["geometry_entity_key"]["second"]): row
                 for row in state["pairs"]} for state in states]
        failures, proofs = [], []
        for key in sorted(set(maps[0]) | set(maps[1])):
            a, b = maps[0].get(key), maps[1].get(key)
            display = a if a is not None else b
            item = {"component": key[0], "first": display["first"], "second": display["second"],
                    "geometry_entity_key": display["geometry_entity_key"]}
            if a is None or b is None:
                failures.append({**item, "reason": "missing_pair_evidence"})
                continue
            if "intersection" in (a["status"], b["status"]):
                failures.append({**item, "reason": "confirmed_endpoint_intersection"})
                continue
            if a["status"] == b["status"] == "internal_assembly":
                proofs.append({**item, "basis": "declared_rigid_assembly"})
                continue
            if a.get("empty_set") and b.get("empty_set"):
                proofs.append({**item, "basis": "empty_represented_set"})
                continue
            if any(name in unavailable or name not in speeds for name in key[1:]):
                failures.append({**item, "reason": "unmodelled_entity_motion"})
                continue
            relative = _add_up(speeds[key[1]], speeds[key[2]])
            finite_permission = any(row.get("permission_kind", "none") != "none" for row in (a, b))
            if finite_permission:
                failures.append({**item, "reason": "dynamic_permission"})
                continue
            distance = [row.get("lower_m") for row in (a, b)]
            change = _mul_up(relative, end-start)
            if (all(value is not None for value in distance)
                    and (max(distance) > change or min(distance) > _mul_up(change, .5))):
                proofs.append({**item, "basis": "full_entity_distance_and_motion",
                               "endpoint_lower_m": distance, "relative_speed_m_s": relative})
            else:
                failures.append({**item, "reason": "unresolved_separation"})
        if not failures:
            proved.append({**base, "pairs": proofs})
            continue
        reason = ("unmodelled_entity_motion" if all(row["reason"] == "unmodelled_entity_motion"
                   for row in failures) else "spatial_resolution" if movement <= spatial_step_m else None)
        middle = (start+end)/2
        if reason is None and (middle == start or middle == end):
            reason = "time_resolution"
        if reason is not None:
            unresolved.append({**base, "reason": reason, "pairs": failures, "proved_pairs": proofs})
        elif consumed >= max_query_points:
            unresolved.append({**base, "reason": "sampling_capacity", "pairs": failures, "proved_pairs": proofs})
        else:
            pending.appendleft((middle, end))
            pending.appendleft((start, middle))
    return {"confirmed_intersections": intersections, "proved_intervals": proved,
            "unresolved_intervals": unresolved, "query_points": consumed,
            "queried_times_s": sorted(queried_times), "spatial_step_m": float(spatial_step_m),
            "scope": "continuous_declared_geometry_model"}
