"""Measured robot model, joint limits and native mechanical pair checks.

Frozen observed-world queries belong to planning. Physical execution consumes
actual joint feedback and checks only the declared robot model here. Commands
never substitute for measured finger coordinates.
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


class RobotStateValidator:
    """Named limits, calibrated model identity and observed robot pair checks.

    Pair declarations are independent of task identity and distance evidence.
    Missing robot observations remain coverage records. Numerical distance
    uncertainty of two represented fields remains an explicit failed query.
    """

    name = "pinocchio_observed_robot_state"

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
                # BEGIN a generic robot may have no independently actuated hand.
                or not self.arm_joint_names):
                # END the complete named model is still mandatory.
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
        # BEGIN only required pairs can reject runtime robot geometry.
        self.required_pairs = tuple(pair for pair in self.pairs if pair.policy == "required")
        # END excluded/mechanical declarations remain in the full model policy.
        expected = {frozenset(pair) for pair in combinations(checker.geometry_names, 2)}
        supplied = [frozenset((pair.first, pair.second)) for pair in self.pairs]
        # BEGIN model-level mechanical permissions are separate pair identities.
        if (len(supplied) != len(set(supplied)) or set(supplied) != expected
                or any(pair.policy not in ("required", "excluded", "mechanical_contact") for pair in self.pairs)):
            raise ValueError("Acceptance requires a complete unique required/excluded/mechanical_contact policy for every native geometry pair")
        # END every registered body pair has one explicit policy.
        for pair in self.pairs:
            _name(pair.source, "native pair policy source")

    # BEGIN one observed-body acceptance policy at planning and execution states.
    def _planning_positions(self, joint_positions, joint_names, *, time_s, gripper):
        """Complete only declared coordinates; identify unknown descendants.

        Pinocchio requires the complete tree coordinates even when an actuator
        has no future geometry model. Interior placeholders affect only that
        actuator's descendants, which are explicitly excluded from acceptance.
        """
        names, values = tuple(joint_names), np.asarray(joint_positions)
        if (len(names) != len(set(names)) or not set(names) <= set(self.joint_names)
                or values.shape != (len(names),) or values.dtype.kind not in "iuf"
                or not np.isfinite(values).all()):
            raise ValueError("Planning self checks require unique named finite robot coordinates")
        if not np.isscalar(time_s) or not np.isfinite(time_s) or time_s < 0:
            raise ValueError("Planning self checks require finite nonnegative physical time")
        named = dict(zip(names, values.astype(float)))
        unmodelled = ()
        if gripper is None:
            if set(names) != set(self.joint_names):
                raise ValueError("A planning robot without a gripper must supply every native joint")
        else:
            if tuple(gripper.joint_names) != self.finger_joint_names:
                raise ValueError("Planning gripper and native model must share the exact finger partition")
            if gripper.controller == "position_tracking":
                if set(names) != set(self.joint_names):
                    raise ValueError("Position-tracking fingers must be active planning coordinates")
            else:
                if set(names) != set(self.arm_joint_names):
                    raise ValueError("Independently controlled fingers must not duplicate active arm coordinates")
                if time_s == 0:
                    fingers = gripper.current_positions
                # BEGIN CPU geometry admits the same typed prefix as Torch.
                elif (gripper.nominal_finger_trajectory is not None
                      and gripper.nominal_finger_trajectory.valid_mask(time_s)):
                    model = gripper.nominal_finger_trajectory
                    if tuple(model.joint_names) != self.finger_joint_names:
                        raise ValueError("Nominal finger geometry changed native joint identity")
                    # BEGIN the native endpoint has one explicit typed representation.
                    # Its float32 value can lie just beyond the binary64 model
                    # duration. Exact equality to that dtype's endpoint names
                    # the same terminal instant; no epsilon admits other times
                    # and no command or new clock replaces the nominal curve.
                    time_dtype = np.asarray(time_s).dtype
                    sample_time = float(time_s)
                    if time_dtype.kind == "f" and time_s == np.asarray(model.duration, dtype=time_dtype):
                        sample_time = float(model.duration)
                    if time_dtype.kind == "f" and time_s == np.asarray(model.valid_until_s, dtype=time_dtype):
                        sample_time = float(model.valid_until_s)
                    fingers, _ = model.sample(sample_time)
                    # END all nonendpoint queries retain their original physical time.
                else:
                    fingers = [sum(self.checker.joint_limits[name])/2 for name in self.finger_joint_names]
                    unmodelled = self.checker.geometry_depending_on(self.finger_joint_names)
                # END an expired nominal curve cannot restore unknown descendants.
                fingers = np.asarray(fingers)
                if (fingers.shape != (len(self.finger_joint_names),) or fingers.dtype.kind not in "iuf"
                        or not np.isfinite(fingers).all()):
                    raise ValueError("Planning finger geometry must supply every finite named coordinate")
                named.update(zip(self.finger_joint_names, fingers.astype(float)))
        return np.asarray([named[name] for name in self.joint_names]), unmodelled

    def validate_planning_binding(self, problem, gripper):
        """Bind model identity and measured entrance before allocating solvers.

        Collision feasibility remains the entrance gate's responsibility. This
        checks the source model, calibration and coordinate partition without
        reconstructing geometry or replacing the supplied measured entrance.
        """
        if (problem.reference_frame != self.reference_frame or problem.tool_link != self.tool_frame
                or not np.array_equal(problem.base_pose, self.T_reference_base)
                or not np.array_equal(problem.tool_transform, self.T_frame_tcp)):
            raise ValueError("Planning and native self geometry must share reference, base and TCP identity")
        self._planning_positions(problem.start_q, problem.joint_names, time_s=0., gripper=gripper)
        if gripper is not None:
            if not np.array_equal(gripper.joint_limits,
                                  [self.checker.joint_limits[name] for name in self.finger_joint_names]):
                raise ValueError("Planning fingers must use the source model's mechanical limits")
            if gripper.controller == "position_tracking":
                named = dict(zip(problem.joint_names, problem.start_q))
                if not np.array_equal([named[name] for name in self.finger_joint_names], gripper.current_positions):
                    raise ValueError("Planning fingers must preserve the same measured entrance")

    def check_planning_self(self, joint_positions, joint_names, *, time_s, gripper):
        """Check all known required native pairs at this one physical instant.

        Each required pair uses the registered observed fields at the supplied
        full joint state. ``passed`` covers only represented
        geometry; an unmodelled future hand is reported explicitly. Scalar
        joint limits still apply even when mechanical contact is permitted.
        """
        positions, unmodelled = self._planning_positions(
            joint_positions, joint_names, time_s=time_s, gripper=gripper)
        unknown = set(unmodelled)
        pairs = tuple(pair for pair in self.pairs if pair.policy == "required"
                      and pair.first not in unknown and pair.second not in unknown)
        result = self.checker.query(joint_names=self.joint_names, joint_positions=positions,
            T_reference_base=self.T_reference_base, pairs=pairs)
        # BEGIN direct observed-set acceptance consumes statuses, including uncertainty.
        violations = [row for row in result["pairs"]
                      if row["status"] in {"intersection", "margin_insufficient", "unresolved"}]
        distances = [row for row in result["pairs"] if row.get("lower_m") is not None]
        minimum = min(distances, key=lambda row: row["lower_m"], default=None)
        missing = [row for row in result["pairs"] if row["status"] == "missing_observation"]
        # END missing observations impose no camera-view rejection on the robot.
        # BEGIN admit only the active tensor's exact representation of a limit.
        # For example float32(.1) lies above the source binary64 upper limit
        # .1, while naming that same endpoint in the native planner. Keep the
        # original q for model FK and preserve the source-limit discrepancy as
        # evidence; neither an epsilon nor clipping admits a neighbouring q.
        # Independently sampled nominal fingers are absent from active_values
        # and therefore retain the source model's original hard limit check.
        limits = dict(result["joint_limit_violations"])
        rounded_limits = []
        active = np.asarray(joint_positions)
        if active.dtype.kind == "f":
            active_values = dict(zip(joint_names, active))
            for name, row in tuple(limits.items()):
                boundary = "lower" if row["position"] < row["lower"] else "upper"
                if (name in active_values
                        and active_values[name] == np.asarray(row[boundary], dtype=active.dtype)):
                    rounded_limits.append({"joint_name": name, "input_dtype": str(active.dtype),
                                           "represented_boundary": boundary, **row})
                    del limits[name]
        # END the execution gate and nonendpoint planning states remain strict.
        passed = not violations and not limits
        return {"passed": passed, "residual": float(not passed), "queried_pairs": len(pairs),
                "minimum_required_clearance_lower_m": None if minimum is None else minimum["lower_m"],
                "missing_observation_pairs": missing,
                "unmodelled_geometry_names": list(unmodelled), "violations": violations,
                "minimum_pair": minimum, "joint_limit_violations": limits,
                "rounded_limit_representations": rounded_limits,
                "joint_names": result["joint_names"], "joint_positions": result["joint_positions"]}

    def planning_binding_record(self):
        """Freeze the same source geometry, complete pair policy and numerics."""
        record = self.describe()
        # BEGIN planning identity retains the complete model pair policy.
        record["pair_order"] = record.pop("pair_policy")
        # END the smaller runtime distance layout cannot change model identity.
        record["model"] = self.checker.model_record()
        record["scope"] = "observed robot/installations and declared required/excluded/mechanical pair policy"
        return record
    # END planning and execution preserve the same required/excluded/mechanical policy.

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
                "pair_order": [asdict(pair) for pair in self.required_pairs],
                "pair_policy": [asdict(pair) for pair in self.pairs],
                "numerics": dict(self.checker.numerics),
                "fk_model_translation_tolerance_m": self.fk_translation_tolerance_m,
                "fk_model_rotation_tolerance_rad": self.fk_rotation_tolerance_rad,
                # BEGIN FK tolerances compare known models, not Cartesian sensors.
                "fk_tolerance_scope": "Consistency of known robot models/calibration at the same measured joints; even a 1e-8 tolerance is not physical Cartesian sensing accuracy",
                "robot_pose_scope": "Fixed calibrated base and TCP FK estimates from measured joint positions; no independent Cartesian observation",
                # END physical tracking and task tolerances keep their own meaning.
                "scope": "known robot/installation native pair policy, model FK and joint limits",
                "observed_scene_checked": False, "unknown_space_checked": False,
                "continuous_collision_certified": False, "world_collision_checked": False}

    def _pose_difference(self, actual, expected):
        return {"translation_m": float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                "rotation_rad": float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}

    def check_state(self, feedback, *, stage, trajectory_time):
        """Gate full actual q, after the executor validates identity and time.

        Clock freshness and physical tracking use the execution layer's own
        # BEGIN independent models do not manufacture independent measurements.
        guard. This independently checks model-frame identity, joint limits,
        observed robot separation and backend FK versus Pinocchio FK at the same
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
        # BEGIN execution checks known robot geometry without rebuilding a world scene.
        result = self.checker.query(joint_names=self.joint_names, joint_positions=feedback["q"],
            T_reference_base=self.T_reference_base, pairs=self.required_pairs, frames=[self.tool_frame])
        # END frozen observed fields and sphere confirmation are planning-only.
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
        # BEGIN model collision evidence uses the requested metric separation.
        for index, row in enumerate(result["pairs"]):
            if row["policy"] == "required" and row["status"] in {
                    "intersection", "margin_insufficient", "unresolved"}:
                violations.append({"kind": "independent_robot_geometry",
                    "name": f"{row['first']} / {row['second']}", "pair_index": index,
                    "status": row["status"], "evidence": deepcopy(row)})
        required = [row["lower_m"] for row in result["pairs"]
                    if row["policy"] == "required" and row.get("lower_m") is not None]
        # END a missing observation is a coverage record, never a collision.
        report = {"passed": not violations, "violations": violations, "stage": stage,
                "trajectory_time": float(trajectory_time),
                "pair_evidence": result["pairs"],
                "min_required_clearance_lower_m": min(required) if required else None,
                "base_model_error": base_error, "tcp_model_error": tool_error,
                # BEGIN disclose what the small residual actually establishes.
                "fk_error_scope": "Agreement of calibrated robot models at measured joints, not independent Cartesian sensing or physical accuracy",
                # END a zero model residual cannot fill a missing object observation.
                "actual_joint_positions": result["joint_positions"],
                "world_collision_checked": False,
                "scope": "supplied robot observed fields at this actual state; no between-state or scene certificate"}
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
        # BEGIN preflight validates actual robot feedback, never frozen world geometry.
        initial = self.check_state(initial_feedback, stage="preflight_actual_state", trajectory_time=float(current_time))
        # END task observation and independent task-effect checks remain executor responsibilities.
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
        # BEGIN native acceptance already owns this exact nominal curve.
        # Preflight keeps cheap scalar polynomial checks and the actual entrance;
        # it never starts another full-curve source-world certification chain.
        return {"passed": not violations and initial["passed"], "violations": violations,
                "scope": "Known robot geometry at measured entrance, model FK and scalar reference q/dq",
                "checked_interval_s": [float(start_time), stop], "reference_cubic_extrema": extrema,
                "actual_state": initial, "future_finger_states_assumed": False,
                "reference_collision_checked": False, "continuous_collision_certified": False,
                "observed_scene_checked": False, "world_collision_checked": False,
                "unknown_space_checked": False}
        # END actual fingers are always feedback coordinates, never commands.
# END independent preflight and actual-state execution gates.
