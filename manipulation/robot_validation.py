"""Measured robot-state checks against the planner's bound online geometry.

The robot model supplies calibrated FK and scalar joint limits. A production
scene uses the accepted robot spheres and online distance fields, including
finite material permissions. Source-solid pair queries remain standalone
robot audits. Commands never substitute for measured finger coordinates.
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
        return str(issue.get("reason", issue.get("kind", "online_scene")))
    violations = [{"kind": "online_scene_collision", **deepcopy(issue), "name": identity(issue)} for issue in scene["violations"]]
    violations.extend({"kind": "online_scene_query_incomplete", "name": identity(issue),
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
        expected = {frozenset(pair) for pair in combinations(checker.geometry_names, 2)}
        supplied = [frozenset((pair.first, pair.second)) for pair in self.pairs]
        # BEGIN model-level mechanical permissions are separate pair identities.
        if (len(supplied) != len(set(supplied)) or set(supplied) != expected
                or any(pair.policy not in ("required", "excluded", "mechanical_contact") for pair in self.pairs)):
            raise ValueError("Acceptance requires a complete unique required/excluded/mechanical_contact policy for every native geometry pair")
        # END every native solid pair is still queried exactly once.
        for pair in self.pairs:
            _name(pair.source, "native pair policy source")

    # BEGIN one native-solid acceptance policy at planning and execution states.
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
                elif gripper.nominal_finger_trajectory is not None:
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
                    fingers, _ = model.sample(sample_time)
                    # END all nonendpoint queries retain their original physical time.
                else:
                    fingers = [sum(self.checker.joint_limits[name])/2 for name in self.finger_joint_names]
                    unmodelled = self.checker.geometry_depending_on(self.finger_joint_names)
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

        A positive sphere proxy cost is not consumed here, and a zero sphere
        cost cannot bypass these solid queries. ``passed`` covers only known
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
        violations = [row for row in result["pairs"] if row["relation"] != "separated"]
        minimum = min(result["pairs"], key=lambda row: row["signed_distance_m"], default=None)
        # BEGIN admit only the active tensor's exact representation of a limit.
        # For example float32(.1) lies above the source binary64 upper limit
        # .1, while naming that same endpoint in the native planner. Keep the
        # original q for Coal FK and preserve the source-limit discrepancy as
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
                "minimum_required_clearance_m": None if minimum is None else minimum["signed_distance_m"],
                "unmodelled_geometry_names": list(unmodelled), "violations": violations,
                "minimum_pair": minimum, "joint_limit_violations": limits,
                "rounded_limit_representations": rounded_limits,
                "joint_names": result["joint_names"], "joint_positions": result["joint_positions"]}

    # BEGIN planning world queries reuse the execution geometry and hand contract.
    def check_planning_world(self, joint_positions, joint_names, *, time_s, gripper,
                             scene_checker, progress, nominal_owner_poses,
                             layer_selection, robot_owner_permission_progress):
        """Check one nominal state against this request's owned source scene.

        The caller has already selected each map layer using the actual native
        sphere inputs. Its one-state record and candidate owner poses travel
        unchanged to the same scene query used by execution. The separate
        checker owns this planning binding; the execution checker's scene and
        release history are never rebound by an optimizer or search branch.
        """
        if not isinstance(scene_checker, RobotGeometryChecker):
            raise TypeError("Planning world acceptance needs a bound source scene checker")
        positions, unmodelled = self._planning_positions(
            joint_positions, joint_names, time_s=time_s, gripper=gripper)
        unknown = set(unmodelled)
        # Self acceptance already checks every required robot pair. Scene
        # aliases additionally retain their exact native pair evidence,
        # including excluded and mechanical pairs, without querying unrelated
        # robot pairs twice for every world state.
        aliases = set(scene_checker._scene_native_aliases.values())
        pairs = tuple(pair for pair in self.pairs
                      if (pair.first in aliases or pair.second in aliases)
                      and pair.first not in unknown and pair.second not in unknown)
        robot = scene_checker.query(joint_names=self.joint_names, joint_positions=positions,
            T_reference_base=self.T_reference_base, pairs=pairs, frames=[self.tool_frame])
        robot["unmodelled_geometry_names"] = unmodelled
        tool_pose = np.asarray(robot["frame_poses"][self.tool_frame]) @ self.T_frame_tcp
        fingers = (None if unmodelled else
                   positions[[self.joint_names.index(name) for name in self.finger_joint_names]])
        scene = scene_checker.query_scene(robot, tool_pose=tool_pose,
            progress=float(progress), finger_positions=fingers, state_kind="nominal",
            nominal_owner_poses=nominal_owner_poses,
            active_robot_owner_permissions=None,
            robot_owner_permission_progress=float(robot_owner_permission_progress),
            layer_selection=layer_selection, layer_selection_index=(0, 0))
        return {"residual": float(not scene["passed"]), "scene": scene,
                "unmodelled_geometry_names": list(unmodelled),
                "joint_names": list(self.joint_names), "joint_positions": positions.tolist()}
    # END a zero proxy cost never bypasses source world collision or query gaps.

    def planning_binding_record(self):
        """Freeze the same source geometry, complete pair policy and numerics."""
        record = self.describe()
        # Scene binding is a separate execution epoch, not robot model identity.
        record.pop("scene_bound")
        record.pop("object_contact_permission_checked")
        # BEGIN source audit identity stays independent of an online epoch.
        record["pair_order"] = [asdict(pair) for pair in self.pairs]
        # END changing execution distance layout cannot change the robot model.
        record["model"] = self.checker.model_record()
        record["scope"] = "complete source robot/installation solids and required/excluded/mechanical pair policy"
        return record
    # END planning never changes the execution gate's complete actual-state policy.

    # BEGIN execution reuses the planner's serialized online distance fields.
    def bind_scene(self, binding, progress_at_time, nominal_fingers=None):
        """Bind one scene epoch and its robot-sphere layout, without source meshes.

        Pinocchio supplies frame transforms at measured named joints. Distances,
        material contact rules and sphere self pairs come from the very same
        OnlinePlanningScene record used by native planning residuals.
        """
        from .online_geometry import OnlinePlanningScene
        if binding["reference_frame"] != self.reference_frame or not callable(progress_at_time):
            raise ValueError("Online scene and execution clock must share the bound reference frame")
        hand = binding.get("gripper")
        if hand is not None and tuple(hand["joint_names"]) != self.finger_joint_names:
            raise ValueError("Online scene and measured fingers must use the same named order")
        geometry = binding.get("robot_sphere_geometry")
        if not isinstance(geometry, list) or not geometry:
            raise ValueError("The online execution scene requires its native robot-sphere geometry")
        # BEGIN FK must interpret the saved native spheres in the same model.
        expected_model = {"urdf_path": str(self.checker.urdf_path), "base_frame": self.checker.base_frame,
                          "tool_frame": self.tool_frame, "T_reference_base": self.T_reference_base.tolist(),
                          "T_frame_tcp": self.T_frame_tcp.tolist()}
        if binding.get("robot_model") != expected_model:
            raise ValueError("Online robot sphere FK model and calibration differ from execution")
        # END matching link names alone cannot establish calibrated identity.
        indices = []
        for row in geometry:
            self.checker._frame_id(row["link_name"])
            centers, radii = np.asarray(row["centers_local"], float), np.asarray(row["radii"], float)
            ids = row["sphere_indices"]
            if (centers.shape != (len(ids), 3) or radii.shape != (len(ids),)
                    or not np.isfinite(centers).all() or not np.isfinite(radii).all()
                    or any(type(index) is not int or index < 0 for index in ids)):
                raise ValueError("Online robot spheres require finite local centers, radii and native indices")
            indices.extend(ids)
        if sorted(indices) != list(range(len(indices))):
            raise ValueError("Online robot sphere indices must cover the exact native layout once")
        scene = OnlinePlanningScene.from_binding(binding)
        self._scene_binding = deepcopy(binding)
        self._online_scene = scene
        self._online_sphere_count = len(indices)
        self._progress_at_time = progress_at_time

    def _scene_state(self, robot_state, tool_pose, time_s, *, feedback=None,
                     active_contact_rule_ids=frozenset()):
        """Transform the accepted native sphere layout with current measured FK."""
        feedback = feedback or {}
        spheres = np.empty((self._online_sphere_count, 4), float)
        for row in self._scene_binding["robot_sphere_geometry"]:
            pose = np.asarray(robot_state["frame_poses"][row["link_name"]], float)
            ids = np.asarray(row["sphere_indices"], int)
            spheres[ids, :3] = np.asarray(row["centers_local"], float) @ pose[:3, :3].T + pose[:3, 3]
            spheres[ids, 3] = row["radii"]
        scene = self._online_scene.query_state(spheres, tool_pose,
            float(self._progress_at_time(float(time_s))),
            joint_position=robot_state["joint_positions"], joint_names=robot_state["joint_names"],
            trajectory_time=float(time_s), active_contact_rule_ids=frozenset(active_contact_rule_ids),
            owner_poses=feedback.get("scene_owner_poses"),
            condition_evidence=feedback.get("scene_condition_evidence"))
        # Query gaps are explicit online failures when present; source-mesh
        # coverage and nominal release reconstruction are not execution gates.
        scene.setdefault("query_gaps", [])
        return scene
    # END fresh binding owns no inherited arrival latch or source-world checker.

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
                "pair_order": ([] if hasattr(self, "_online_scene") else [asdict(pair) for pair in self.pairs]),
                "numerics": dict(self.checker.numerics),
                "fk_model_translation_tolerance_m": self.fk_translation_tolerance_m,
                "fk_model_rotation_tolerance_rad": self.fk_rotation_tolerance_rad,
                # BEGIN FK tolerances compare known models, not Cartesian sensors.
                "fk_tolerance_scope": "Consistency of known robot models/calibration at the same measured joints; even a 1e-8 tolerance is not physical Cartesian sensing accuracy",
                "robot_pose_scope": "Fixed calibrated base and TCP FK estimates from measured joint positions; no independent Cartesian observation",
                # END physical tracking and task tolerances keep their own meaning.
                "scope": ("bound online geometry, measured robot spheres, model FK and joint limits"
                          if hasattr(self, "_online_scene") else
                          "standalone source robot self-collision audit, model FK and joint limits"),
                "observed_scene_checked": False, "unknown_space_checked": False,
                "continuous_collision_certified": False,
                "scene_bound": hasattr(self, "_scene_binding"),
                "object_contact_permission_checked": bool(getattr(self, "_scene_binding", {}).get("contacts"))}

    def _pose_difference(self, actual, expected):
        return {"translation_m": float(np.linalg.norm(actual[:3, 3]-expected[:3, 3])),
                "rotation_rad": float(Rotation.from_matrix(expected[:3, :3].T @ actual[:3, :3]).magnitude())}

    def check_state(self, feedback, *, stage, trajectory_time, active_contact_rule_ids=frozenset()):
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
        # BEGIN online acceptance needs FK only from the source robot model.
        frames = [self.tool_frame]
        if hasattr(self, "_online_scene"):
            frames.extend(row["link_name"] for row in self._scene_binding["robot_sphere_geometry"])
        result = self.checker.query(joint_names=self.joint_names, joint_positions=feedback["q"],
            T_reference_base=self.T_reference_base,
            pairs=() if hasattr(self, "_online_scene") else self.pairs,
            frames=list(dict.fromkeys(frames)))
        # END standalone source self-collision checks remain an explicit audit.
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
            # BEGIN the execution epoch alone authorizes a measured-arrival permission.
            scene = self._scene_state(result, expected_tool, trajectory_time, feedback=feedback,
                                      active_contact_rule_ids=active_contact_rule_ids)
            # END actual time by itself cannot remove a robot/owner collision.
            report["scene"] = scene
            report["violations"].extend(_scene_gate_violations(scene))
            report["query_gaps"] = scene["query_gaps"]
            report["passed"] = report["passed"] and scene["passed"]
            report["scope"] = "supplied robot and observed scene solids at actual state; no hidden-space or continuous certificate"
        return report
        # END commands and nominal predictions never replace measured fingers.

    def preflight(self, trajectory, initial_feedback, execution_limits, *, start_time=0., until=None, current_time=None,
                  active_contact_rule_ids=frozenset()):
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
        # BEGIN continuation preflight consumes this execution's same arrival latch.
        initial = self.check_state(initial_feedback, stage="preflight_actual_state", trajectory_time=float(current_time),
                                   active_contact_rule_ids=active_contact_rule_ids)
        # END a newly created validator does not infer prior arrival from progress.
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
                "query_gaps": initial.get("query_gaps", []),
                "scope": "Same online scene at measured entrance, model FK and scalar reference q/dq",
                "checked_interval_s": [float(start_time), stop], "reference_cubic_extrema": extrema,
                "actual_state": initial, "future_finger_states_assumed": False,
                "reference_collision_checked": False, "continuous_collision_certified": False,
                "observed_scene_checked": bool(hasattr(self, "_online_scene")),
                "unknown_space_checked": False}
        # END actual fingers are always feedback coordinates, never commands.
# END independent preflight and actual-state execution gates.
