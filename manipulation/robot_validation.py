"""Mandatory independent robot checks at the execution boundary.

This gate covers native robot/installation solids, their complete pair policy,
model FK and scalar joint limits. When a formal scene snapshot is bound it also
checks observed scene/target/payload solids and finite local material permissions.
It never certifies hidden space or collision between checked instants. Future independently actuated fingers are never frozen to a measured
snapshot or replaced by position commands to claim a trajectory certificate.
"""

# BEGIN independent preflight and actual-state execution gates.
from copy import deepcopy
from dataclasses import asdict
from itertools import combinations

import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.spatial.transform import Rotation

from manipulation.robot_geometry_check import RobotGeometryChecker, _name, _rigid_pose


def _positive_scalar(value, label):
    if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return float(value)


# BEGIN one execution-facing failure schema, with original scene evidence intact.
def _scene_gate_violations(scene):
    """Name physical violations and mandatory query gaps without losing cause."""
    def identity(issue):
        if "rule_id" in issue:
            return "contact_rule:" + str(issue["rule_id"])
        if "owner" in issue:
            return "owner:" + str(issue["owner"])
        evidence = issue.get("evidence", issue)
        if "actor" in evidence or "target_owner" in evidence:
            return "/".join(str(evidence.get(key, "")) for key in
                            ("actor", "actor_geometry", "target_owner", "target_geometry"))
        return str(issue["reason"])
    violations = [{**deepcopy(issue), "name": identity(issue)} for issue in scene["violations"]]
    violations.extend({"kind": "independent_scene_query_incomplete", "name": identity(issue),
                       "reason": issue["reason"], "query_evidence": deepcopy(issue)}
                      for issue in scene["query_gaps"])
    return violations
# END an uncompleted required query is never a generic exception or a false pass.


class RobotStateValidator:
    """Complete native pair policy and independent FK for measured states.

    Constructor transforms and identity tolerances describe a model binding,
    not physical tracking error. The executor separately supervises its shared
    ExecutionLimits. Every required pair must be positively separated beyond
    Coal's reported numerical boundary. A model-declared ``mechanical_contact``
    pair permits exactly those two solids to contact within the independently
    checked mechanical joint domain, including native soft-contact overlap.
    Native-filtered and mechanically permitted pairs still retain raw distances
    and their distinct policy provenance. No permission is inferred from
    proximity, a control command, native contact enablement or a task name.
    """

    name = "pinocchio_coal_robot_state"

    def __init__(self, checker, *, joint_names, arm_joint_names, finger_joint_names,
                 pairs, named_velocity_limits, reference_frame, T_reference_base,
                 tool_frame, T_frame_tcp, fk_translation_tolerance_m,
                 fk_rotation_tolerance_rad, source):
        if not isinstance(checker, RobotGeometryChecker):
            raise TypeError("An independent RobotGeometryChecker is required")
        self.checker = checker
        self.joint_names = tuple(joint_names)
        self.arm_joint_names, self.finger_joint_names = tuple(arm_joint_names), tuple(finger_joint_names)
        if (self.joint_names != self.arm_joint_names + self.finger_joint_names
                or len(self.joint_names) != len(set(self.joint_names))
                or set(self.joint_names) != set(checker.joint_names)
                or not self.arm_joint_names or not self.finger_joint_names):
            raise ValueError("The explicit arm/finger order must cover the complete native model")
        self.reference_frame = _name(reference_frame, "reference frame")
        self.tool_frame = _name(tool_frame, "tool frame")
        checker._frame_id(self.tool_frame)
        self.T_reference_base = _rigid_pose(T_reference_base)
        self.T_frame_tcp = _rigid_pose(T_frame_tcp, "T_frame_tcp")
        self.fk_translation_tolerance_m = _positive_scalar(fk_translation_tolerance_m, "Model FK translation tolerance")
        self.fk_rotation_tolerance_rad = _positive_scalar(fk_rotation_tolerance_rad, "Model FK rotation tolerance")
        if self.fk_rotation_tolerance_rad > np.pi:
            raise ValueError("Model FK angular tolerance cannot exceed pi radians")
        self.source = _name(source, "model/policy source")
        if not isinstance(named_velocity_limits, dict) or set(named_velocity_limits) != set(self.joint_names):
            raise ValueError("Native velocity limits must cover every model joint exactly")
        self.velocity_limits = {name: _positive_scalar(named_velocity_limits[name], f"Velocity limit for {name}")
                                for name in self.joint_names}
        if self.velocity_limits != checker.joint_velocity_limits:
            raise ValueError("Declared hardware velocities must equal the independent URDF's named velocity limits")
        self.pairs = tuple(pairs)
        expected = {frozenset(pair) for pair in combinations(checker.geometry_names, 2)}
        supplied = [frozenset((pair.first, pair.second)) for pair in self.pairs]
        # BEGIN model-level mechanical permissions are separate pair identities.
        if (len(supplied) != len(set(supplied)) or set(supplied) != expected
                or any(pair.policy not in ("required", "excluded", "mechanical_contact") for pair in self.pairs)):
            raise ValueError("Acceptance requires a complete unique required/excluded/mechanical_contact policy for every native geometry pair")
        # END every native solid pair is still queried exactly once.
        for pair in self.pairs:
            _name(pair.source, "native pair policy source")

    # BEGIN bind independent scene validation to the exact accepted input snapshot.
    def bind_scene(self, binding, progress_at_time, nominal_fingers=None):
        """Bind f(t)->progress and optional nominal f(t)->(named finger q,dq).

        The optional callback represents declared future geometry, never an
        actuator command. Actual checks always take every coordinate directly
        from feedback. The scene copy is owned by this validation epoch.
        """
        if binding["reference_frame"] != self.reference_frame or not callable(progress_at_time):
            raise ValueError("Independent scene and clock must share the bound reference frame")
        if nominal_fingers is not None and not callable(nominal_fingers):
            raise TypeError("Nominal finger geometry must be an explicit time sampler")
        hand = binding.get("gripper")
        if hand is not None and tuple(hand["joint_names"]) != self.finger_joint_names:
            raise ValueError("Scene material translations and independent actual fingers must share the same named order")
        self._scene_binding = deepcopy(binding)
        self._progress_at_time = progress_at_time
        self._nominal_fingers = nominal_fingers
        self.checker.bind_scene(binding)

    def _scene_state(self, robot_state, tool_pose, finger_positions, time_s, *, state_kind, feedback=None,
                     progress=None):
        feedback = feedback or {}
        return self.checker.query_scene(robot_state, tool_pose=tool_pose,
            progress=float(self._progress_at_time(float(time_s)) if progress is None else progress), finger_positions=finger_positions,
            state_kind=state_kind, owner_poses=feedback.get("scene_owner_poses"),
            condition_evidence=feedback.get("scene_condition_evidence"))

    # BEGIN release anchors come from the same executable cubic and bound clock.
    def _bind_release_predictions(self, trajectory):
        """Compute conditional release FK at its first bound progress occurrence.

        This is the same explicit first-occurrence convention as the planning
        scene's release gather. It does not invert arbitrary progress queries
        to guess a finger time or claim the object was observed at this pose.
        """
        from scipy.spatial.transform import Rotation
        predictions = {}
        s, u = np.asarray(trajectory.progress_s, float), np.asarray(trajectory.progress_u, float)
        if s.shape != u.shape or len(s) < 2 or np.any(np.diff(s) < 0) or np.any(np.diff(u) <= 0):
            raise ValueError("Release predictions require the exact finite monotone bound clock")
        for owner in self._scene_binding["owners"]:
            for index, relation in enumerate(owner["attachments"]):
                if relation["release_condition_id"] is None:
                    continue
                progress = relation["interval"][1]
                matches = np.flatnonzero(s >= progress)
                if not len(matches):
                    raise ValueError("Release relation exceeds the executable progress domain")
                right = int(matches[0])
                parameter = u[right] if s[right] == progress or right == 0 else np.interp(progress, s[right-1:right+1], u[right-1:right+1])
                time_s = float(parameter * trajectory.duration)
                arm = trajectory._curve(time_s)
                fingers = ([sum(self.checker.joint_limits[name])/2 for name in self.finger_joint_names]
                           if self._nominal_fingers is None else self._nominal_fingers(time_s)[0])
                # Unmodelled finger descendants cannot determine a release TCP.
                joint = int(self.checker.model.frames[self.checker._frame_id(self.tool_frame)].parentJoint)
                if self._nominal_fingers is None:
                    while joint:
                        if str(self.checker.model.names[joint]) in self.finger_joint_names:
                            raise ValueError("A release TCP depending on unknown fingers needs explicit nominal geometry")
                        joint = int(self.checker.model.parents[joint])
                state = self.checker.query(joint_names=self.joint_names, joint_positions=np.r_[arm, fingers],
                    T_reference_base=self.T_reference_base, pairs=(), frames=[self.tool_frame])
                offset = np.eye(4)
                offset[:3, :3] = Rotation.from_quat(np.asarray(relation["quaternion_tool_owner"])[[1, 2, 3, 0]]).as_matrix()
                offset[:3, 3] = relation["position_tool_owner"]
                pose = np.asarray(state["frame_poses"][self.tool_frame]) @ self.T_frame_tcp @ offset
                predictions[(owner["owner_id"], index)] = {"pose": pose, "time_s": time_s,
                    "source": "independent FK of the same bound cubic at first release progress"}
        self.checker._scene_release_predictions = predictions
    # END predicted release remains a model hypothesis, not actual release evidence.

    # BEGIN required scene/event boundaries are explicit independent query times.
    def _scene_query_times(self, trajectory, endpoints, start, stop):
        values = list(endpoints)
        for event in trajectory.events:
            for key in ("time", "activation_time_s"):
                if event.get(key) is not None:
                    values.append(float(event[key]))
        anchors = [value for rule in self._scene_binding["contacts"] for value in rule["interval"]]
        anchors += [value for owner in self._scene_binding["owners"]
                    for relation in owner["attachments"] for value in relation["interval"]]
        # Use the same binary64 clock arithmetic as sample() and the mode
        # probes below, even when the archived native progress keys are f32.
        s, u = np.asarray(trajectory.progress_s, float), np.asarray(trajectory.progress_u, float)
        for anchor in anchors:
            exact = np.flatnonzero(s == anchor)
            if len(exact):
                values.extend((float(u[exact[0]]*trajectory.duration), float(u[exact[-1]]*trajectory.duration)))
            elif s[0] <= anchor <= s[-1]:
                right = int(np.searchsorted(s, anchor))
                values.append(float(np.interp(anchor, s[right-1:right+1], u[right-1:right+1])*trajectory.duration))
        hand = self._scene_binding.get("gripper")
        if hand is not None and hand["nominal_model"] is not None:
            values.extend(hand["nominal_model"]["times"])
        result = np.asarray(values, float)
        if not np.isfinite(result).all():
            raise ValueError("Required independent scene query times must be finite")
        return np.unique(result[(result >= start) & (result <= stop)])
    # END querying a boundary never changes the native curve or its clock.

    # BEGIN each mode switch is checked on both sides at the identical state.
    def _scene_progress_modes(self, trajectory, time_s):
        """Keep q(t), fingers(t) and TCP(t) fixed while selecting adjacent laws.

        Contact/attachment masks are discontinuous even for continuous motion.
        At a progress plateau only its entrance has a preceding mode and only
        its exit has a following mode. Float64 neighbours are consumed directly
        by the independent checker, never converted to a planner float32 grid.
        """
        modes = [(float(self._progress_at_time(float(time_s))), "bound_progress")]
        anchors = {value for rule in self._scene_binding["contacts"] for value in rule["interval"]}
        anchors.update(value for owner in self._scene_binding["owners"]
                       for relation in owner["attachments"] for value in relation["interval"])
        s, u = np.asarray(trajectory.progress_s, float), np.asarray(trajectory.progress_u, float)
        for anchor in sorted(anchors):
            if not s[0] <= anchor <= s[-1]:
                continue
            exact = np.flatnonzero(s == anchor)
            if len(exact):
                first, last = u[exact[0]]*trajectory.duration, u[exact[-1]]*trajectory.duration
            else:
                right = int(np.searchsorted(s, anchor))
                first = last = float(np.interp(anchor, s[right-1:right+1], u[right-1:right+1])*trajectory.duration)
            if time_s == first and anchor > s[0]:
                modes.append((float(np.nextafter(anchor, -np.inf)), "before_switch"))
            if time_s == last and anchor < s[-1]:
                modes.append((float(np.nextafter(anchor, np.inf)), "after_switch"))
        return modes
    # END mode probes do not advance the command clock or freeze actual fingers.

    def _nominal_scene_preflight(self, trajectory, times):
        """Discrete future solid evidence with explicit uncertainty and coverage.

        No actual finger snapshot is frozen along the reference. Without a
        declared model, future hand geometry remains an allowed unresolved
        assumption; the report cannot claim that its solids were checked.
        Actual complete robot/scene checks remain mandatory while executing.
        """
        unmodelled = (() if self._nominal_fingers is not None else
                      self.checker.geometry_depending_on(self.finger_joint_names))
        known_pairs = tuple(pair for pair in self.pairs if pair.first not in unmodelled and pair.second not in unmodelled)
        queries = [(time_s, self._scene_progress_modes(trajectory, time_s)) for time_s in times]
        requested = sum(len(modes) for _, modes in queries)
        if requested > 2048:
            gap = {"reason": "independent_nominal_state_budget", "requested": requested}
            return {"passed": False, "violations": _scene_gate_violations({"violations": [], "query_gaps": [gap]}),
                    "states": [], "observed_scene_checked": False, "query_gaps": [gap]}
        states, violations, gaps = [], [], []
        for time_s, modes in queries:
            arm_q, arm_dq = trajectory._curve(time_s), trajectory._curve(time_s, 1)
            if self._nominal_fingers is None:
                # Pinocchio requires a complete vector to evaluate the tree.
                # These interior placeholders affect ONLY excluded descendant
                # solids; no finger geometry, limit or relation is certified.
                finger_q = [sum(self.checker.joint_limits[name])/2 for name in self.finger_joint_names]
                finger_dq = np.zeros(len(self.finger_joint_names))
                contact_fingers = None
            else:
                finger_q, finger_dq = self._nominal_fingers(float(time_s))
                contact_fingers = np.asarray(finger_q)
            q, dq = np.r_[arm_q, finger_q], np.r_[arm_dq, finger_dq]
            if q.shape != (len(self.joint_names),) or dq.shape != q.shape or not np.isfinite(np.r_[q, dq]).all():
                raise ValueError("Nominal full state must supply each named arm/finger coordinate")
            robot = self.checker.query(joint_names=self.joint_names, joint_positions=q,
                T_reference_base=self.T_reference_base, pairs=known_pairs, frames=[self.tool_frame])
            robot["unmodelled_geometry_names"] = unmodelled
            tool_pose = np.asarray(robot["frame_poses"][self.tool_frame]) @ self.T_frame_tcp
            for name, row in robot["joint_limit_violations"].items():
                violations.append({"kind": "nominal_joint_position", "name": name, "time_s": float(time_s), **row})
            for name, speed in zip(self.joint_names, dq):
                if abs(speed) > self.velocity_limits[name]:
                    violations.append({"kind": "nominal_joint_velocity", "name": name, "time_s": float(time_s), "measured": float(speed)})
            for pair in robot["pairs"]:
                if pair["policy"] == "required" and pair["relation"] != "separated":
                    violations.append({"kind": "independent_nominal_robot_collision", "name": pair["first"]+" / "+pair["second"],
                                       "time_s": float(time_s), "evidence": pair})
            # BEGIN raw evidence identifies the same-time one-sided rule scope.
            for progress, mode in modes:
                scene = self._scene_state(robot, tool_pose, contact_fingers, time_s,
                    state_kind="nominal", progress=progress)
                scope = {"time_s": float(time_s), "mode": mode, "progress": progress}
                violations.extend({**row, **scope} for row in _scene_gate_violations(scene))
                gaps.extend({**row, **scope} for row in scene["query_gaps"])
                states.append({**scope, "scene": scene, "robot_pairs": robot["pairs"]})
            # END an endpoint permission never suppresses an adjacent old mode.
        return {"passed": not violations and not gaps, "violations": violations, "query_gaps": gaps,
                "states": states, "observed_scene_checked": not gaps and not unmodelled and all(row["scene"]["observed_scene_checked"] for row in states),
                "unresolved_assumptions": ([] if not unmodelled else [{"kind": "future_finger_geometry_unbound", "geometry_names": list(unmodelled)}]),
                "continuous_collision_certified": False}
    # END only explicit geometry models enter nominal solid checks.

    def describe(self):
        """Static report defines the order of every per-state distance vector."""
        return {"name": self.name, "source": self.source,
                "joint_names": list(self.joint_names), "arm_joint_names": list(self.arm_joint_names),
                "finger_joint_names": list(self.finger_joint_names),
                "urdf_path": self.checker.urdf_path, "base_frame": self.checker.base_frame,
                "reference_frame": self.reference_frame, "T_reference_base": self.T_reference_base.tolist(),
                "tool_frame": self.tool_frame, "T_frame_tcp": self.T_frame_tcp.tolist(),
                "joint_position_limits": {name: list(self.checker.joint_limits[name]) for name in self.joint_names},
                "joint_velocity_limits": dict(self.velocity_limits),
                "geometry_sources": self.checker.geometry_sources,
                "pair_order": [asdict(pair) for pair in self.pairs],
                "numerics": dict(self.checker.numerics),
                "fk_model_translation_tolerance_m": self.fk_translation_tolerance_m,
                "fk_model_rotation_tolerance_rad": self.fk_rotation_tolerance_rad,
                # BEGIN FK tolerances compare known models, not Cartesian sensors.
                "fk_tolerance_scope": "Consistency of known robot models/calibration at the same measured joints; even a 1e-8 tolerance is not physical Cartesian sensing accuracy",
                "robot_pose_scope": "Fixed calibrated base and TCP FK estimates from measured joint positions; no independent Cartesian observation",
                # END physical tracking and task tolerances keep their own meaning.
                "scope": "robot self/installation solids, model FK and joint position/velocity limits only",
                "observed_scene_checked": False, "unknown_space_checked": False,
                "continuous_collision_certified": False,
                "scene_bound": hasattr(self, "_scene_binding"),
                "object_contact_permission_checked": bool(getattr(self, "_scene_binding", {}).get("contacts"))}

    def _pose_difference(self, actual, expected):
        return {"translation_m": float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                "rotation_rad": float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}

    def check_state(self, feedback, *, stage, trajectory_time):
        """Gate full actual q, after the executor validates identity and time.

        Clock freshness and physical tracking use the execution layer's own
        # BEGIN independent models do not manufacture independent measurements.
        guard. This independently checks model-frame identity, joint limits,
        robot solid separation and backend FK versus Pinocchio FK at the same
        measured q. Fixed-base/TCP consistency tolerances, including 1e-8 values,
        are numerical/model requirements, not sensor-accuracy certificates.
        # END original physical tracking limits remain owned by ExecutionLimits.
        No actuator command is read by this function.
        """
        _name(stage, "validation stage")
        if not np.isscalar(trajectory_time) or not np.isfinite(trajectory_time) or trajectory_time < 0:
            raise ValueError("Validation requires the finite nonnegative original trajectory time")
        for key, expected in (("joint_names", self.joint_names), ("arm_joint_names", self.arm_joint_names),
                              ("finger_joint_names", self.finger_joint_names)):
            if tuple(feedback[key]) != expected:
                raise ValueError(f"Actual feedback changed independent model identity: {key}")
        if feedback["reference_frame"] != self.reference_frame or feedback["tool_link"] != self.tool_frame:
            raise ValueError("Actual reference/TCP frame identity differs from the independently bound model")
        tool_transform = _rigid_pose(feedback["tool_transform"], "feedback tool_transform")
        if not np.array_equal(tool_transform, self.T_frame_tcp):
            raise ValueError("Actual feedback changed the independently bound link-from-TCP transform")
        base = _rigid_pose(feedback["base_pose"], "feedback base_pose")
        # BEGIN the supplied Cartesian pose is the backend's robot-model estimate.
        estimated_tool = _rigid_pose(feedback["tool_pose"], "feedback TCP FK estimate")
        # END neither validator reads hidden current object truth here.
        velocities = np.asarray(feedback["dq"])
        if velocities.shape != (len(self.joint_names),) or velocities.dtype.kind not in "iuf" or not np.isfinite(velocities).all():
            raise ValueError("Independent validation requires one finite actual velocity per named joint")
        result = self.checker.query(joint_names=self.joint_names, joint_positions=feedback["q"],
                                    T_reference_base=self.T_reference_base, pairs=self.pairs, frames=[self.tool_frame])
        expected_tool = np.asarray(result["frame_poses"][self.tool_frame]) @ self.T_frame_tcp
        base_error = self._pose_difference(base, self.T_reference_base)
        tool_error = self._pose_difference(estimated_tool, expected_tool)
        violations = []
        for name, error in (("base", base_error), ("tcp", tool_error)):
            for metric, maximum in (("translation_m", self.fk_translation_tolerance_m),
                                    ("rotation_rad", self.fk_rotation_tolerance_rad)):
                if error[metric] > maximum:
                    violations.append({"kind": "independent_model_fk", "name": f"{name}:{metric}",
                                       "measured": error[metric], "maximum": maximum})
        for name, row in result["joint_limit_violations"].items():
            violations.append({"kind": "independent_joint_position", "name": name,
                               "measured": row["position"], "lower": row["lower"], "upper": row["upper"]})
        for name, velocity in zip(self.joint_names, velocities):
            if abs(velocity) > self.velocity_limits[name]:
                violations.append({"kind": "independent_joint_velocity", "name": name,
                                   "measured": float(velocity), "maximum_absolute": self.velocity_limits[name]})
        for index, row in enumerate(result["pairs"]):
            if row["policy"] == "required" and row["relation"] != "separated":
                violations.append({"kind": "independent_robot_collision", "name": f"{row['first']} / {row['second']}",
                                   "measured": row["signed_distance_m"], "pair_index": index,
                                   "relation": row["relation"], "evidence": deepcopy(row)})
        required = [row["signed_distance_m"] for row in result["pairs"] if row["policy"] == "required"]
        # BEGIN retain legal contact evidence while keeping all scalar/FK guards.
        mechanical_contacts = [deepcopy(row) for row in result["pairs"]
                               if row["policy"] == "mechanical_contact" and row["relation"] != "separated"]
        # END allowed contact does not imply separation or stable task Holding.
        report = {"passed": not violations, "violations": violations, "stage": stage,
                "trajectory_time": float(trajectory_time),
                "distances_m": [row["signed_distance_m"] for row in result["pairs"]],
                "mechanical_contacts": mechanical_contacts,
                "min_required_clearance_m": min(required) if required else None,
                "base_model_error": base_error, "tcp_model_error": tool_error,
                # BEGIN disclose what the small residual actually establishes.
                "fk_error_scope": "Agreement of calibrated robot models at measured joints, not independent Cartesian sensing or physical accuracy",
                # END a zero model residual cannot fill a missing object observation.
                "actual_joint_positions": result["joint_positions"],
                "scope": "supplied robot solids at this actual state; no between-state or scene certificate"}
        # BEGIN current scene geometry consumes actual q and current explicit evidence.
        if hasattr(self, "_scene_binding"):
            fingers = np.asarray(feedback["q"], float)[len(self.arm_joint_names):]
            scene = self._scene_state(result, expected_tool, fingers, trajectory_time, state_kind="actual", feedback=feedback)
            report["scene"] = scene
            report["violations"].extend(_scene_gate_violations(scene))
            report["query_gaps"] = scene["query_gaps"]
            report["passed"] = report["passed"] and scene["passed"]
            report["scope"] = "supplied robot and observed scene solids at actual state; no hidden-space or continuous certificate"
        return report
        # END commands and nominal predictions never replace measured fingers.

    def preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0., until=None, current_time=None):
        """Check actual entrance and reference cubic q/dq over its whole prefix.

        Each scalar cubic's exact polynomial derivative supplies its critical
        times through SciPy's maintained root solver. Values are evaluated in
        binary64; this is a polynomial range check, not formal exact arithmetic.
        It certifies neither future finger positions nor continuous collision.
        The original evidence barrier defines the end of the checked prefix.
        """
        if (tuple(trajectory.joint_names) != self.arm_joint_names
                or tuple(execution_limits.joint_names) != self.joint_names
                or tuple(execution_limits.arm_joint_names) != self.arm_joint_names
                or tuple(execution_limits.finger_joint_names) != self.finger_joint_names):
            raise ValueError("Reference curve and physical execution contract must match the independent named model")
        if isinstance(start_time, bool) or not np.isscalar(start_time) or not np.isfinite(start_time) or start_time < 0:
            raise ValueError("Preflight start must be a finite nonnegative original trajectory time")
        # BEGIN feedback time and newly authorized nominal interval are separate.
        current_time = start_time if current_time is None else current_time
        if (isinstance(current_time, bool) or not np.isscalar(current_time) or not np.isfinite(current_time)
                or not 0 <= current_time <= trajectory.duration):
            raise ValueError("Current actual feedback must use its own original trajectory time")
        # END device tick alignment must not silently shift a contact-rule interval.
        if hasattr(self, "_scene_binding"):
            self._bind_release_predictions(trajectory)
        initial = self.check_state(initial_feedback, stage="preflight_actual_state", trajectory_time=float(current_time))
        times = np.asarray(trajectory.times, dtype=float)
        positions, velocities = np.asarray(trajectory.position, dtype=float), np.asarray(trajectory.velocity, dtype=float)
        expected_shape = (len(times), len(self.arm_joint_names))
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0
                or np.any(np.diff(times) <= 0) or positions.shape != expected_shape or velocities.shape != expected_shape
                or not np.isfinite(positions).all() or not np.isfinite(velocities).all()):
            raise ValueError("Preflight requires the complete finite ordered exported cubic and its derivatives")
        # BEGIN an explicit effect-authorized prefix can extend the same bound curve.
        stop = float(trajectory.executable_until if until is None else until)
        # END this does not rebuild the curve, retime events or assume new feedback.
        if not np.isfinite(stop) or not start_time <= stop <= times[-1]:
            raise ValueError("Original execution barrier must lie in the exported cubic")
        curve = CubicHermiteSpline(times, positions, velocities, axis=0)
        # The executor samples this stored polynomial. Refuse an object whose
        # public arrays and actual executable curve have diverged after export.
        if not np.array_equal(curve.x, trajectory._curve.x) or not np.array_equal(curve.c, trajectory._curve.c):
            raise ValueError("Exported arrays differ from the actual executable cubic")
        violations, extrema = deepcopy(initial["violations"]), {}
        endpoints = np.unique(np.r_[times[(times >= start_time) & (times <= stop)], start_time, stop])
        for index, name in enumerate(self.arm_joint_names):
            scalar = CubicHermiteSpline(times, positions[:, index], velocities[:, index])
            row = {}
            for order, label in ((0, "position"), (1, "velocity")):
                roots = scalar.derivative(order+1).roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots >= start_time) & (roots <= stop)]
                candidates = np.unique(np.r_[endpoints, roots])
                values = scalar(candidates, order)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite reference cubic extrema")
                lower_index, upper_index = int(np.argmin(values)), int(np.argmax(values))
                row[label] = {"minimum": float(values[lower_index]), "maximum": float(values[upper_index]),
                              "minimum_time_s": float(candidates[lower_index]),
                              "maximum_time_s": float(candidates[upper_index])}
            lower, upper = self.checker.joint_limits[name]
            for bound, expected in (("minimum", lower), ("maximum", upper)):
                value = row["position"][bound]
                if (value < expected if bound == "minimum" else value > expected):
                    violations.append({"kind": "reference_cubic_position", "name": name, "measured": value,
                                       "lower": lower, "upper": upper,
                                       "time_s": row["position"][bound+"_time_s"]})
            for bound in ("minimum", "maximum"):
                value = row["velocity"][bound]
                if abs(value) > self.velocity_limits[name]:
                    violations.append({"kind": "reference_cubic_velocity", "name": name, "measured": value,
                                       "maximum_absolute": self.velocity_limits[name],
                                       "time_s": row["velocity"][bound+"_time_s"]})
            extrema[name] = row
        # BEGIN nominal geometry is checked at the exported knots and authorized endpoint.
        nominal = None
        if hasattr(self, "_scene_binding"):
            nominal = self._nominal_scene_preflight(trajectory, self._scene_query_times(trajectory, endpoints, start_time, stop))
            violations.extend(nominal["violations"])
        return {"passed": not violations and initial["passed"] and (nominal is None or nominal["passed"]),
                "violations": violations, "nominal_scene": nominal,
                "query_gaps": ([] if nominal is None else nominal["query_gaps"]) + initial.get("query_gaps", []),
                "scope": "actual entrance solids/FK, continuous scalar q/dq and explicitly modeled discrete nominal solids",
                "checked_interval_s": [float(start_time), stop], "reference_cubic_extrema": extrema,
                "actual_state": initial, "future_finger_states_assumed": bool(nominal is not None and self._nominal_fingers is not None),
                "reference_collision_checked": bool(nominal is not None and nominal["states"]), "continuous_collision_certified": False,
                "observed_scene_checked": bool(nominal is not None and nominal["observed_scene_checked"]),
                "unknown_space_checked": False}
        # END missing future hand geometry remains explicit unresolved evidence.
# END independent preflight and actual-state execution gates.
