"""Mandatory independent robot checks at the execution boundary.

This gate covers the supplied native robot/installation solids, their complete
pair policy, model FK consistency and scalar joint limits. It does NOT cover
observed scene obstacles, unknown space, permitted object contacts, or collision
between checked instants. Those remain part of the original planning/scene
contract. Future independently actuated fingers are never frozen to a measured
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
                "scope": "robot self/installation solids, model FK and joint position/velocity limits only",
                "observed_scene_checked": False, "unknown_space_checked": False,
                "continuous_collision_certified": False, "object_contact_permission_added": False}

    def _pose_difference(self, actual, expected):
        return {"translation_m": float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                "rotation_rad": float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}

    def check_state(self, feedback, *, stage, trajectory_time):
        """Gate full actual q, after the executor validates identity and time.

        Clock freshness and physical tracking use the execution layer's own
        guard. This independently checks model-frame identity, raw native joint
        limits, robot solid separation and measured TCP versus Pinocchio FK.
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
        measured_tool = _rigid_pose(feedback["tool_pose"], "feedback tool_pose")
        velocities = np.asarray(feedback["dq"])
        if velocities.shape != (len(self.joint_names),) or velocities.dtype.kind not in "iuf" or not np.isfinite(velocities).all():
            raise ValueError("Independent validation requires one finite actual velocity per named joint")
        result = self.checker.query(joint_names=self.joint_names, joint_positions=feedback["q"],
                                    T_reference_base=self.T_reference_base, pairs=self.pairs, frames=[self.tool_frame])
        expected_tool = np.asarray(result["frame_poses"][self.tool_frame]) @ self.T_frame_tcp
        base_error = self._pose_difference(base, self.T_reference_base)
        tool_error = self._pose_difference(measured_tool, expected_tool)
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
        return {"passed": not violations, "violations": violations, "stage": stage,
                "trajectory_time": float(trajectory_time),
                "distances_m": [row["signed_distance_m"] for row in result["pairs"]],
                "mechanical_contacts": mechanical_contacts,
                "min_required_clearance_m": min(required) if required else None,
                "base_model_error": base_error, "tcp_model_error": tool_error,
                "actual_joint_positions": result["joint_positions"],
                "scope": "supplied robot solids at this actual state; no between-state or scene certificate"}

    def preflight(self, trajectory, initial_feedback, execution_limits):
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
        initial = self.check_state(initial_feedback, stage="preflight_actual_state", trajectory_time=0.)
        times = np.asarray(trajectory.times, dtype=float)
        positions, velocities = np.asarray(trajectory.position, dtype=float), np.asarray(trajectory.velocity, dtype=float)
        expected_shape = (len(times), len(self.arm_joint_names))
        if (times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all() or times[0] != 0
                or np.any(np.diff(times) <= 0) or positions.shape != expected_shape or velocities.shape != expected_shape
                or not np.isfinite(positions).all() or not np.isfinite(velocities).all()):
            raise ValueError("Preflight requires the complete finite ordered exported cubic and its derivatives")
        stop = float(trajectory.executable_until)
        if not np.isfinite(stop) or not 0 <= stop <= times[-1]:
            raise ValueError("Original execution barrier must lie in the exported cubic")
        curve = CubicHermiteSpline(times, positions, velocities, axis=0)
        # The executor samples this stored polynomial. Refuse an object whose
        # public arrays and actual executable curve have diverged after export.
        if not np.array_equal(curve.x, trajectory._curve.x) or not np.array_equal(curve.c, trajectory._curve.c):
            raise ValueError("Exported arrays differ from the actual executable cubic")
        violations, extrema = deepcopy(initial["violations"]), {}
        endpoints = np.unique(np.r_[times[times <= stop], stop])
        for index, name in enumerate(self.arm_joint_names):
            scalar = CubicHermiteSpline(times, positions[:, index], velocities[:, index])
            row = {}
            for order, label in ((0, "position"), (1, "velocity")):
                roots = scalar.derivative(order+1).roots(extrapolate=False)
                roots = roots[np.isfinite(roots) & (roots >= 0.) & (roots <= stop)]
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
        return {"passed": not violations, "violations": violations,
                "scope": "actual entrance geometry/FK and continuous reference scalar q/dq limits only",
                "checked_interval_s": [0., stop], "reference_cubic_extrema": extrema,
                "actual_state": initial, "future_finger_states_assumed": False,
                "reference_collision_checked": False, "continuous_collision_certified": False,
                "observed_scene_checked": False, "unknown_space_checked": False}
# END independent preflight and actual-state execution gates.
