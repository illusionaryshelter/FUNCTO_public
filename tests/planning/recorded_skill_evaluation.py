"""Independent native-physics evidence for an explicitly identified skill.

This module is evaluator-only: native object identity, live object poses and
contacts never enter a planning problem or an observation builder. Original
functional pose sets use their bound material frames, thresholds and clocks;
grasp stability uses the collector's existing segmentation predicates. A grade
requires the entire current curve and all relevant supported obligations.
Unmapped frames, unsupported predicates and unobserved intervals remain unknown.
"""

# BEGIN independent native skill evidence with original collector predicates.
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from collect.preprocess.segmentation import (
    DEFAULT_SETTINGS, _bridge_gaps, _first_stable_grasp, _resolve_actor, _runs,
)
from manipulation.execution import pose_goal_schedule
# BEGIN explicit evaluator roles reuse the existing execution-scope validator.
from manipulation.execution_binding import validate_result_only_goal_ids
# END result-only execution scope alone does not change a goal's grading role.
from manipulation.motion import _inverse, _pose_array
# BEGIN support grading reuses the original finite geometry and numerical contract.
from manipulation.goals import SupportSurface, check_on
from manipulation.regions import check_region_on
# END these pure metrics receive evaluator-local actual material poses only.
# BEGIN reconstruct typed original requirements without extending their public fields.
from manipulation.planning_problem import (
    EventTimeConstraint, NumericalTolerance, PoseEventRequirement,
    SupportRequirement, TimeBindingError,
    resolve_event_window,
)
# END the independent evaluator consumes the same serialized requirement contract.


# BEGIN simulation-only pose metrics never feed an execution decision.
def _actual_pose_goal(requirement, feedback, observation):
    """Evaluate an original pose allowed set in the independent simulation sink.

    NEVER call this from the production executor. A robot FK estimate compared
    with a frozen predicted target is not a measured physical tool-to-target
    relation. Target-relative error (including missing target truth) can only
    grade the recorded simulation; it must not veto close, abort, or replan.
    The evaluator supplies native poses with provenance and keeps the original
    tolerances. Its result is never returned as an online control observation.
    """
    measured = {"value": "U", "source": "Independent simulation posthoc pose evaluation",
        "actual_function_pose": None, "actual_target_pose": None,
        "position_error_m": None, "rotation_error_rad": None, "measurement_sources": []}
    relation = requirement.actual_relation
    poses = {} if observation is None else observation.get("scene_owner_poses", {})
    conditions = {} if observation is None else observation.get("scene_condition_evidence", {})
    missing = []

    def owner_pose(identity):
        if identity not in poses:
            missing.append(f"No current observed pose for owner {identity}")
            return None
        pose = _pose_array(poses[identity], f"Current observed owner pose {identity}")
        measured["measurement_sources"].append({"kind": "current_owner_pose", "owner_id": identity,
            "source": observation["source"], "timestamp_s": float(observation["timestamp_s"]), "pose": pose.tolist()})
        return pose

    def actual_tool():
        # BEGIN retain the estimator provenance alongside the actual joint clock.
        pose = _pose_array(feedback["tool_pose"], "Robot TCP estimate from measured joints")
        measured["measurement_sources"].append({"kind": "actual_robot_tcp", "source": feedback["source"],
            "pose_source": feedback["tool_pose_source"],
            "independent_observation": feedback["tool_pose_is_independent_observation"],
            "timestamp_s": float(feedback["timestamp_s"]), "feedback_id": feedback["feedback_id"], "pose": pose.tolist()})
        # END FK agreement alone never supplies a Cartesian sensor measurement.
        return pose

    if relation is not None and "function_owner_id" in relation:
        if relation["function_owner_id"] in poses:
            function = owner_pose(relation["function_owner_id"]) @ np.asarray(relation["T_owner_function"])
        elif (relation.get("attachment_condition_id") is not None
              and conditions.get(relation["attachment_condition_id"]) is True):
            function = actual_tool() @ requirement.tool_from_function
            measured["measurement_sources"].append({"kind": "current_verified_attachment",
                "condition_id": relation["attachment_condition_id"], "source": observation["source"],
                "timestamp_s": float(observation["timestamp_s"]), "value": True})
        else:
            function = None
            missing.append("No current owner pose or verified actual holding relation for the material function frame")
    else:
        function = actual_tool() @ requirement.tool_from_function
    if relation is not None and "target_owner_id" in relation:
        owner = owner_pose(relation["target_owner_id"])
        target = None if owner is None else owner @ np.asarray(relation["T_owner_target"])
    else:
        target = requirement.target
    measured["actual_function_pose"] = None if function is None else function.tolist()
    measured["actual_target_pose"] = None if target is None else target.tolist()
    if missing:
        measured["reason"] = "; ".join(missing)
        return measured
    position = float(np.linalg.norm(function[:3, 3]-target[:3, 3]))
    rotation = float(Rotation.from_matrix(target[:3, :3].T @ function[:3, :3]).magnitude())
    measured.update(position_error_m=position, rotation_error_rad=rotation,
        value="T" if position <= requirement.position_tolerance_m and rotation <= requirement.rotation_tolerance_rad else "F")
    return measured
# END these metrics remain simulation evidence, never online prerequisites.

# BEGIN explicitly diagnostic goals retain measurements without vetoing skill effects.
def validate_diagnostic_goal_ids(binding, values):
    """Validate explicit grading roles for the current geometric goal identities.

    Pose, support and mechanism-coordinate goals are evaluated after execution.
    Each remains required for success unless separately selected as diagnostic.
    Execution-policy lists and event references do not choose that grading role;
    process and effect predicates cannot be selected here.
    """
    # BEGIN grading validates current identities without an online geometry policy.
    return validate_result_only_goal_ids(binding, values)
    # END explicit diagnostic selection leaves all other obligations required.


def summarize_skill_evidence(report, *, binding, trajectory, diagnostic_goal_ids=()):
    """Regrade saved components without measuring, simulating or mutating them.

    The caller supplies the same formal binding/curve plus the newly declared
    diagnostic identities. Every goal retains its original T/F/U and metrics.
    All unselected goals, original effects, processes and terminal predicates
    remain required. Complete acquisition and full-curve execution are also
    required; this result never becomes an online observation or dependency fact.
    """
    # BEGIN diagnostic eligibility consumes goal identities alone.
    selected = validate_diagnostic_goal_ids(binding, diagnostic_goal_ids)
    # END trajectory measurements and obligation grading remain unchanged.
    if report["candidate_id"] != binding["candidate_id"]:
        raise ValueError("Evidence and grading roles must refer to the same formal candidate")
    goals = deepcopy(report["functional_goal_evidence"])
    expected = {(collection, index) for collection in ("pose_events", "coordinate_events", "supports")
                for index in binding.get("goal_identities", {}).get(collection, [])}
    identities = [tuple(row["identity"]) for row in goals]
    if len(set(identities)) != len(identities) or set(identities) != expected:
        raise ValueError("Grading must retain exactly every current original goal identity")
    for row in goals:
        row["evaluation_role"] = "diagnostic" if row["identity"] in selected else "required"
    other = {name: deepcopy(report[name]) for name in
             ("effect_evidence", "process_evidence", "terminal_evidence")}
    # Saved coverage describes acquisition, not whether a later caller retained
    # every required component. Match the original binding before aggregating.
    effect_ids = [row.get("condition_id") for row in other["effect_evidence"]]
    expected_effects = {row["condition_id"] for row in binding["problem"].get("conditions", [])
                        if row.get("effect_requirement")}
    if len(set(effect_ids)) != len(effect_ids) or set(effect_ids) != expected_effects:
        raise ValueError("Grading must retain exactly every original required effect")
    process_ids = [tuple(row.get("identity", ())) for row in other["process_evidence"]]
    expected_processes = {(collection, index)
        for collection in ("mechanisms", "pose_relations", "interiors", "couplings")
        for index in range(len(binding["problem"].get(collection, [])))}
    if len(set(process_ids)) != len(process_ids) or set(process_ids) != expected_processes:
        raise ValueError("Grading must retain exactly every original process obligation")
    kind = report["evaluation_kind"]
    if kind not in ("grasp", "manipulation"):
        raise ValueError("Grading needs the original supported skill kind")
    terminal_ids = [row.get("predicate") for row in other["terminal_evidence"]]
    expected_terminal = {"stable_grasp_at_final_stop"} if kind == "grasp" else set()
    if len(set(terminal_ids)) != len(terminal_ids) or set(terminal_ids) != expected_terminal:
        raise ValueError("Grading must retain the original final-stop obligations")
    for rows in other.values():
        for row in rows:
            row["evaluation_role"] = "required"
    components = goals + [row for rows in other.values() for row in rows]
    if any(row["value"] not in ("T", "F", "U") for row in components):
        raise ValueError("Every saved evidence component needs an explicit T/F/U value")
    required = [row for row in components if row["evaluation_role"] == "required"]
    values = [row["value"] for row in required]
    if not report["evidence_coverage"]["complete"]:
        passed, outcome = None, "incomplete_evidence"
    elif not report["completed_current_curve"]:
        passed, outcome = None, "not_evaluated"
    elif "F" in values:
        passed, outcome = False, "failed"
    elif not values or "U" in values:
        passed, outcome = None, "unverified"
    else:
        passed, outcome = True, "passed"
    return {
        "physics_success": passed, "physics_outcome": outcome,
        "functional_goal_evidence": goals, **other,
        "evaluation_policy": {
            "diagnostic_goal_ids": deepcopy(selected),
            "required_goal_ids": [row["identity"] for row in goals if row["evaluation_role"] == "required"],
            "criterion_source": "Explicit diagnostic_goal_ids declared by the evaluator caller. Unselected original goals, bound skill effects, scoped processes and declared terminal predicates retain their required roles.",
            "scope": "Explicit evaluator grading roles only; no change to planning, execution or runtime facts",
        },
        "success_definition": "The complete current curve is physically observed and every required goal, original effect, scoped process and final-stop predicate is T. Explicit diagnostic goals retain their original metrics and T/F/U but do not veto success; unsupported or unobserved required predicates remain U.",
    }
# END offline grading roles never alter evidence values or execution authorization.


def _body_pose(data, index):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[index].reshape(3, 3), data.xpos[index]
    return pose


def _site_pose(data, index):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.site_xmat[index].reshape(3, 3), data.site_xpos[index]
    return pose


# BEGIN truth stays inside the independent evaluator's original-goal measurements.
class _FunctionalGoalEvaluator:
    """Bind observed owner frames to native bodies without changing the candidate.

    A material frame needs a declared static body-from-owner transform and
    its source. A body identity alone supports contact/effect attribution but
    supplies no material geometry. Initial observed pose errors are never
    fitted away with native truth. No returned row is runtime input.
    """

    def __init__(self, model, catalog, planning_result, owner_bindings):
        self._model = model
        self._trajectory = planning_result.trajectory
        self._binding = planning_result.execution_binding
        if (planning_result.success is not True or self._trajectory is None
                or not isinstance(self._binding, dict)):
            raise ValueError("Independent goal grading requires the same formal execution result")
        problem = self._binding["problem"]
        if problem["reference_frame"] != "world":
            raise ValueError("Native truth grading requires an explicit world-frame candidate binding")
        self._requirements = {}
        self._unsupported_goals = []
        # BEGIN support goals preserve their declared geometry, not a surrogate pose target.
        self._support_bindings = {}
        self._support_numerical = (NumericalTolerance(**deepcopy(problem["numerical"]))
                                   if problem.get("supports") else None)
        # END numerical body clearance is distinct from the material gap/normal thresholds.
        for collection in ("pose_events", "coordinate_events", "supports"):
            rows = problem.get(collection, [])
            identities = self._binding.get("goal_identities", {}).get(collection, [])
            if (len(rows) != len(identities) or len(set(identities)) != len(identities)
                    or any(type(index) is not int or index < 0 for index in identities)):
                raise ValueError("Independent grading needs every original collection/index identity")
            for row, index in zip(rows, identities):
                identity = (collection, index)
                if collection == "pose_events":
                    self._requirements[identity] = PoseEventRequirement(**deepcopy(row))
                # BEGIN reconstruct the same public support object from the formal archive.
                elif collection == "supports":
                    supplied = deepcopy(row)
                    supplied["surface"] = SupportSurface(**supplied["surface"])
                    requirement = SupportRequirement(**supplied)
                    self._requirements[identity] = requirement
                    self._support_bindings[identity] = self._support_binding(requirement, problem)
                # END only unsupported coordinate predicates retain their previous unknown role.
                else:
                    self._unsupported_goals.append({"identity": list(identity), "value": "U",
                        "requirement": deepcopy(row),
                        "reason": "No native coordinate predicate with its original frame/coordinate binding is supplied"})
        owners = self._binding["scene"].get("owners", [])
        self._owners = {}
        if not isinstance(owner_bindings, list):
            raise ValueError("evaluation_owner_bindings must be an explicit list")
        # BEGIN static material coordinates are distinct from native body identity.
        for row in owner_bindings:
            if (not isinstance(row, dict) or set(row) not in (
                    {"owner_id", "native_body_name"},
                    {"owner_id", "native_body_name", "T_body_owner", "source"})
                    or any(not isinstance(row[key], str) or not row[key].strip()
                           for key in ("owner_id", "native_body_name"))):
                raise ValueError("An evaluator binding needs owner_id/native_body_name and, for material geometry, both T_body_owner and source")
            identity, name = row["owner_id"], row["native_body_name"]
            if identity in self._owners:
                raise ValueError("Evaluator owner bindings must be unique")
            selected = [owner for owner in owners if owner["owner_id"] == identity]
            bodies = [body for body in catalog["bodies"] if body["name"] == name]
            if (len(selected) != 1 or len(bodies) != 1
                    or model.body(name).id != bodies[0]["id"]):
                raise ValueError("Evaluator owner/native body identities must match one declared scene/catalog body")
            transform, source = None, None
            if "T_body_owner" in row:
                if not isinstance(row["source"], str) or not row["source"].strip():
                    raise ValueError("An evaluator material transform requires an explicit nonempty static source")
                transform = _pose_array(row["T_body_owner"], "Declared static evaluator body-from-owner frame")
                source = row["source"]
            self._owners[identity] = {"owner_id": identity, "native_body_name": name,
                "native_body_id": int(bodies[0]["id"]),
                "T_body_owner": transform, "source": source,
                "material_frame_status": "declared_static" if transform is not None else "unmapped"}
        # END missing transforms remain unknown rather than inheriting an estimated pose.
        self._owner_pose_samples = []

    # BEGIN use the existing compiled held metadata without guessing an owner from a name.
    @staticmethod
    def _support_binding(requirement, problem):
        """Resolve one material carrier from the held phases touching this goal.

        Both neighbours can include a shared boundary. Repeated references to
        the same declared attachment are one fact; two different facts, an
        incomplete binding or a conflicting transform remain unknown. This
        identifies a material frame for offline measurement, not actual Holding.
        """
        matched = {}
        for index, phase in enumerate(problem.get("phases", ())):
            metadata = phase.get("metadata", {})
            if (metadata.get("mode") != "held"
                    or not index <= requirement.progress <= index+1):
                continue
            try:
                carrier, condition_id = metadata["carrier_id"], metadata["attachment_condition_id"]
                if any(not isinstance(value, str) or not value for value in (carrier, condition_id)):
                    raise ValueError("Held phase lacks explicit carrier/condition identities")
                span = np.asarray(metadata["attachment_interval"], float)
                if (span.shape != (2,) or not np.isfinite(span).all()
                        or not span[0] <= index < index+1 <= span[1]
                        or not span[0] <= requirement.progress <= span[1]):
                    raise ValueError("Held attachment interval does not cover its phase and goal")
                attachment = _pose_array(metadata["attachment"], "Bound support owner-from-tool")
                conditions = [row for row in problem.get("conditions", ())
                              if row.get("condition_id") == condition_id]
                if (len(conditions) != 1 or conditions[0].get("carrier_id") != carrier
                        or conditions[0].get("kind") not in ("declared_attachment", "candidate_attachment")
                        or conditions[0].get("progress") != span[0]):
                    raise ValueError("Support carrier needs the same uniquely declared attachment condition")
                declared = _pose_array(conditions[0]["T_carrier_eef"], "Declared support attachment")
                if (not np.allclose(attachment, declared, atol=1e-10, rtol=0)
                        or not np.allclose(_inverse(attachment), requirement.tool_from_carrier,
                                           atol=1e-10, rtol=0)):
                    raise ValueError("Support material transform differs from its declared held carrier")
            except (KeyError, TypeError, ValueError) as error:
                return {"reason": f"Original support carrier binding is incomplete or inconsistent: {error}"}
            matched[(carrier, condition_id)] = {
                "carrier_id": carrier, "attachment_condition_id": condition_id}
        if len(matched) != 1:
            return {"reason": "Original support has no unique held carrier/attachment binding"}
        return next(iter(matched.values()))
    # END owner ambiguity cannot be repaired with native identity or TCP proximity.

    def observe(self, data):
        """Read and validate native transforms; do not publish observations."""
        poses = {}
        for identity, binding in self._owners.items():
            # BEGIN only an independently declared material frame follows native motion.
            if binding["T_body_owner"] is None:
                continue
            body = _pose_array(_body_pose(data, binding["native_body_id"]), "Native evaluator body pose")
            poses[identity] = _pose_array(body @ binding["T_body_owner"], "Actual evaluator owner frame")
            # END entry perception errors cannot become a fitted body-local displacement.
        self._owner_pose_samples.append(poses)

    def describe_bindings(self):
        return [{key: value.tolist() if isinstance(value, np.ndarray) else value
                 for key, value in row.items()} for row in self._owners.values()]

    def _measure(self, identity, index, times, tools):
        # BEGIN actual support follows the material body even after the hand retreats.
        if identity[0] == "supports":
            return self._measure_support(identity, index)
        # END no tool-derived attachment is extrapolated into the released phase.
        # BEGIN shared pure metrics retain the evaluator-only GT provenance.
        feedback = {"tool_pose": tools[index], "source": "Independent evaluator native TCP pose",
            "tool_pose_source": "Independent evaluator-only native GT site pose; not a runtime sensor",
            "tool_pose_is_independent_observation": False,
            "timestamp_s": float(times[index]),
            "feedback_id": f"{self._binding['candidate_id']}@native_evaluator_time={times[index]:.17g}"}
        # END no runtime encoder/FK observation is claimed for this local value.
        # This local dictionary is only an argument to the shared pure metric
        # calculation. It is never supplied to an executor or an observer hook.
        observation = {"scene_owner_poses": self._owner_pose_samples[index],
            "timestamp_s": float(times[index]), "source": "Independent evaluator native poses composed with declared static material frames"}
        measured = _actual_pose_goal(self._requirements[identity], feedback, observation)
        measured["source"] = "Independent evaluator applies the original functional allowed set to native truth"
        return measured

    # BEGIN independent support metrics preserve precisely the compiled allowed set.
    def _measure_support(self, identity, index):
        """Evaluate the observed material body against the frozen world patch.

        ``check_region_on`` supplies gap/normal diagnostics and ``check_on``
        supplies the whole-body footprint. Their convenience ``valid``/absolute
        minimum-height criteria are not a replacement for SupportRequirement:
        only its specified face supplies the contact gap, and its original
        numerical position budget applies to body height and footprint alone.
        """
        requirement = self._requirements[identity]
        binding = self._support_bindings[identity]
        measured = {"value": "U", "source": "Independent evaluator applies the original finite support allowed set to native carrier motion in its declared static material frame",
            "actual_carrier_pose": None, "carrier_binding": deepcopy(binding),
            "surface_scope": "Original frozen world SupportSurface; no native target replacement or online completion"}
        if "reason" in binding:
            measured["reason"] = binding["reason"]
            return measured
        carrier = binding["carrier_id"]
        poses = self._owner_pose_samples[index]
        if carrier not in poses:
            measured["reason"] = f"No explicit static evaluator material-frame mapping for support carrier {carrier}"
            return measured
        pose = poses[carrier]
        region = check_region_on(pose, requirement.contact_points, requirement.outward_normal,
            requirement.surface, gap_tolerance=requirement.gap_tolerance,
            normal_tolerance=requirement.normal_tolerance, margin=requirement.margin)
        body = check_on(pose, requirement.body_points, requirement.surface)
        points = requirement.body_points @ pose[:3, :3].T+pose[:3, 3]
        height = float(np.min((points-requirement.surface.origin) @ requirement.surface.normal))
        chord = float(np.linalg.norm(pose[:3, :3] @ requirement.outward_normal+requirement.surface.normal))
        angular_bound = float(2*np.sin(requirement.normal_tolerance/2))
        numerical = self._support_numerical.position_m
        violations = {
            "contact_gap_m": max(0., region["max_abs_gap_m"]-requirement.gap_tolerance),
            "body_penetration_m": max(0., -height-numerical),
            "body_boundary_m": max(0., requirement.margin-body["minimum_boundary_clearance_m"]-numerical),
            "normal_chord": max(0., chord-angular_bound)}
        measured.update(value="T" if not any(violations.values()) else "F",
            actual_carrier_pose=pose.tolist(), contact_face=region,
            body={**body, "minimum_signed_height_m": height}, violations=violations,
            normal_chord=chord, normal_chord_bound=angular_bound,
            gap_tolerance_m=requirement.gap_tolerance,
            normal_tolerance_rad=requirement.normal_tolerance,
            margin_m=requirement.margin, numerical_position_tolerance_m=numerical)
        return measured
    # END gap, angle, finite footprint and whole-body clearance retain their separate roles.

    def evaluate(self, times, elapsed, tools, execution_report, *, full_curve_completed):
        """Grade original goal times using trajectory samples, excluding holds."""
        evidence = deepcopy(self._unsupported_goals)
        origin = float(self._binding.get("request_time_origin_s", 0.))
        request_times = origin + times-times[0]
        history = [] if execution_report is None else execution_report.get("history", [])
        timing_events = self._binding.get("timing_events", [])
        # BEGIN every geometric measurement belongs to independent result grading.
        for identity, requirement in self._requirements.items():
            scheduled, due = pose_goal_schedule(self._trajectory, requirement.progress, self._model.opt.timestep)
            row = {"identity": list(identity), "value": "U", "query_count": 0,
                "execution_scope": "result",
                "progress": requirement.progress,
                "scheduled_curve_time_s": float(scheduled) if np.isfinite(scheduled) else None,
                "device_due_time_s": float(due) if np.isfinite(due) else None,
                "requirement_source": deepcopy(requirement.source),
                "scope": "Independent result evidence only; does not update runtime completion or release a dependency"}
            # END execution-policy metadata cannot label geometry as online evidence.
            # BEGIN preserve the original units and shape of each goal's allowed set.
            if identity[0] == "supports":
                row.update(gap_tolerance_m=requirement.gap_tolerance,
                    normal_tolerance_rad=requirement.normal_tolerance, margin_m=requirement.margin,
                    numerical_position_tolerance_m=self._support_numerical.position_m)
            else:
                row.update(position_tolerance_m=requirement.position_tolerance_m,
                           rotation_tolerance_rad=requirement.rotation_tolerance_rad)
            # END support goals do not acquire a full-position/yaw target.
            arrivals = [item for event in timing_events for item in event.get("timing", ())
                        if item["check_target"] == "arrival" and tuple(item["goal_id"]) == identity]
            if arrivals:
                try:
                    windows = [resolve_event_window(EventTimeConstraint(**item), history) for item in arrivals]
                except TimeBindingError as error:
                    row["reason"] = f"Original arrival window is unresolved: {error}"
                    evidence.append(row)
                    continue
                lower = max((value[0] for value in windows if value[0] is not None), default=None)
                upper = min((value[1] for value in windows if value[1] is not None), default=None)
                row["arrival_window_s"] = [lower, upper]
                mask = np.ones(len(times), dtype=bool)
                if lower is not None:
                    mask &= request_times >= lower-1e-9
                if upper is not None:
                    mask &= request_times <= upper+1e-9
                indexes = np.flatnonzero(mask)
                closed = full_curve_completed or (upper is not None and request_times[-1] >= upper-1e-9)
            else:
                # Shared strict activation chooses the first native sample,
                # including the first arrival on a progress plateau. Later
                # playback or hold motion cannot repair a missed point goal.
                indexes = np.flatnonzero(elapsed >= due)[:1]
                closed = bool(indexes.size)
            if not indexes.size:
                row["reason"] = "No trajectory sample covers the original goal time or arrival window"
                evidence.append(row)
                continue
            unknown = False
            for index in indexes:
                measured = self._measure(identity, int(index), times, tools)
                row["query_count"] += 1
                unknown |= measured["value"] == "U"
                if measured["value"] == "T":
                    break
            row.update(sample_index=int(index), sample_sim_time_s=float(times[index]),
                       sample_elapsed_s=float(elapsed[index]), measurement=measured)
            row["value"] = ("T" if measured["value"] == "T" else "U" if not closed or unknown else "F")
            if not closed and measured["value"] != "T":
                row["reason"] = "Execution ended before the original arrival opportunity closed"
            evidence.append(row)
        return evidence

    # BEGIN a final support measurement is explicitly separate from original goal grading.
    def final_support_diagnostics(self, times, elapsed, *, full_curve_completed):
        """Measure the last retained body sample without inventing a hold window."""
        return [{"identity": list(identity), "evaluation_role": "diagnostic",
            "completed_current_curve": bool(full_curve_completed),
            "sample_index": len(times)-1, "sample_sim_time_s": float(times[-1]),
            "sample_elapsed_s": float(elapsed[-1]),
            "measurement": self._measure_support(identity, len(times)-1),
            "scope": "Last normal-curve sample only; no declared terminal support or stability interval, and no effect on required original goals"}
            for identity in self._support_bindings]
    # END later support success never repairs an earlier original goal failure.

    # BEGIN online coverage comes exclusively from the unchanged execution report.
    def support_runtime_coverage(self, execution_report):
        """Expose missing typed support observations without adding a dependency."""
        execution = {} if execution_report is None else execution_report
        coverage = []
        for identity in self._support_bindings:
            rows = [row for row in execution.get("goal_evidence", ())
                    if tuple(row.get("identity", ())) == identity]
            completed = (list(identity) in execution.get("completed_goal_ids", ())
                and any(row.get("value") == "T" and row.get("completion_verified") is True for row in rows))
            value = "T" if completed else "F" if any(row.get("value") == "F" for row in rows) else "U"
            coverage.append({"identity": list(identity), "value": value,
                "reported_observation_count": len(rows),
                "source": "Unchanged online execution goal evidence only; independent native support measurements are excluded",
                "reason": ("Online report contains a verified typed support observation" if completed else
                    "Online report contains a failed support observation" if value == "F" else
                    "No verified online typed support observation was supplied"),
                "scope": "Observation coverage only; a support value does not create an undeclared open/release prerequisite"})
        return coverage
    # END independent physics T never turns an online U into a completed goal.

    def arrays(self):
        # BEGIN body-only identities never manufacture missing material poses in archives.
        identities = [key for key, binding in self._owners.items() if binding["T_body_owner"] is not None]
        poses = np.asarray([[row[key] for key in identities] for row in self._owner_pose_samples])
        return {"evaluation_owner_ids": np.asarray(identities, dtype=str),
                "evaluation_owner_poses_world": poses.reshape(len(self._owner_pose_samples), len(identities), 4, 4)}
        # END the declared static mapping and its source remain in the JSON report.
# END truth measurements have no route back into the runtime request ledger.


def _analyze_history(times, touches, tool_poses, target_poses, fingers,
                     finger_limits, settings):
    """Apply the source segmentation definition to this trial's measurements.

    Two-sided contact must refer to this same rigid target body. Internal
    contact gaps may be bridged, but neither endpoint is extended. Success at
    the stop requires the final completed stability window, not an earlier
    grasp that was subsequently lost.
    """
    relative = np.linalg.inv(target_poses) @ tool_poses
    positions = relative[:, :3, 3]
    rotations = Rotation.from_matrix(relative[:, :3, :3])
    simultaneous = np.all(touches, axis=1)
    bridged = _bridge_gaps(simultaneous, times, settings["contact_gap_s"])
    runs = _runs(bridged)
    first_stable, first_evidence = None, None
    final_stable, final_evidence = None, None
    longest = 0.
    for begin, end in runs:
        duration = float(times[end-1] - times[begin])
        longest = max(longest, duration)
        if duration < settings["stable_grasp_s"] - 1e-9:
            continue
        found, evidence = _first_stable_grasp(begin, end, times, positions, rotations,
                                               fingers, finger_limits, settings)
        if found is not None and first_stable is None:
            first_stable, first_evidence = found, evidence
        if end == len(times):
            final_begin = max(begin, int(np.searchsorted(
                times, times[-1]-settings["stable_grasp_s"]-1e-9)))
            found, evidence = _first_stable_grasp(final_begin, end, times, positions, rotations,
                                                   fingers, finger_limits, settings)
            if found == len(times)-1:
                final_stable, final_evidence = found, evidence
    lower = int(np.searchsorted(times, times[-1]-settings["stable_grasp_s"]-1e-9))
    final_metrics = {
        "window_start_sim_time": float(times[lower]), "window_end_sim_time": float(times[-1]),
        "window_duration_s": float(times[-1]-times[lower]),
        "relative_position_span_m": float(np.linalg.norm(np.ptp(positions[lower:], axis=0))),
        "relative_rotation_span_rad": float(np.max((rotations[lower].inv()*rotations[lower:]).magnitude())),
        "finger_coordinate_spans": np.ptp(fingers[lower:], axis=0).tolist(),
        "raw_bilateral_contact_fraction": float(np.mean(simultaneous[lower:])),
        "gap_bridged_bilateral_contact_fraction": float(np.mean(bridged[lower:])),
    }
    reasons = []
    if not simultaneous[-1]:
        reasons.append("no simultaneous contact from every finger group to the target at stop")
    if not bridged[lower:].all() or final_metrics["window_duration_s"] < settings["stable_grasp_s"]-1e-9:
        reasons.append("the final stop lacks a complete gap-bounded stable contact interval")
    if final_metrics["relative_position_span_m"] > settings["stable_relative_position_m"]:
        reasons.append("target-relative TCP translation exceeds the stable-grasp threshold")
    if final_metrics["relative_rotation_span_rad"] > settings["stable_relative_rotation_rad"]:
        reasons.append("target-relative TCP rotation exceeds the stable-grasp threshold")
    if np.any(np.asarray(final_metrics["finger_coordinate_spans"]) > finger_limits):
        reasons.append("finger motion exceeds the stable-grasp threshold")
    target_rotation = Rotation.from_matrix(target_poses[:, :3, :3])
    return {
        "stable_grasp_observed": first_stable is not None,
        "first_stable_sim_time": None if first_stable is None else float(times[first_stable]),
        "first_stable_evidence": first_evidence,
        "stable_grasp_at_stop": final_stable is not None,
        "stable_evidence_at_stop": final_evidence,
        "failure_reasons": reasons,
        "current_bilateral_target_contact": bool(simultaneous[-1]),
        "bilateral_contact_present_at_entry": bool(simultaneous[0]),
        "longest_gap_bridged_bilateral_contact_s": longest,
        "final_window_metrics": final_metrics,
        "maximum_target_translation_from_entry_m": float(np.linalg.norm(target_poses[:, :3, 3]-target_poses[0, :3, 3], axis=1).max()),
        "maximum_target_rotation_from_entry_rad": float((target_rotation[0].inv()*target_rotation).magnitude().max()),
        "final_target_pose_world": target_poses[-1].tolist(),
        "final_tool_pose_world": tool_poses[-1].tolist(),
    }, relative, simultaneous, bridged


class NativeSkillPhysicsEvaluator:
    """Collect native skill evidence and grade only supported predicates."""

    def __init__(self, model, *, catalog, actor, target_body_name,
                 finger_joint_names, tcp_site, binding_source, program,
                 reference_body_name, planning_result, evaluation_owner_bindings,
                 maximum_contact_rows=1_000_000, diagnostic_goal_ids=()):
        self._model = model
        self._skill_kind = program["skill_kind"]
        if self._skill_kind not in ("grasp", "manipulation"):
            raise ValueError("The explicit compiled skill kind must be grasp or manipulation")
        self._task_contract = {"skill_kind": self._skill_kind,
            "task": deepcopy(program.get("task")),
            "functional_events": deepcopy(program["demo"].get("functional_events", [])),
            "interpretation": "Declared source requirements, not current measured task outcomes"}
        resolved, _ = _resolve_actor(catalog, actor)
        if resolved["eef_site_name"] != tcp_site:
            raise ValueError("Evaluator actor and controlled TCP must be the same named site")
        if set(finger_joint_names) != {joint["name"] for joint in resolved["finger_joints"]}:
            raise ValueError("Evaluator morphology must identify exactly the controlled scalar fingers")
        if (type(maximum_contact_rows) is not int
                or not 1 <= maximum_contact_rows <= 1_000_000):
            raise ValueError("Bound retained native contacts to at most one million rows")
        self._maximum_contact_rows = maximum_contact_rows
        self._site = model.site(tcp_site).id
        if self._site != resolved["eef_site_id"]:
            raise ValueError("Catalog and native TCP identity disagree")
        target = [row for row in catalog["bodies"] if row["name"] == target_body_name]
        if len(target) != 1 or model.body(target_body_name).id != target[0]["id"]:
            raise ValueError("Target body must have one identical catalog/native identity")
        self._target = int(target[0]["id"])
        self._reference = None
        if reference_body_name is not None:
            reference = [row for row in catalog["bodies"] if row["name"] == reference_body_name]
            if len(reference) != 1 or model.body(reference_body_name).id != reference[0]["id"]:
                raise ValueError("Reference body must have one identical catalog/native identity")
            self._reference = int(reference[0]["id"])
        self._target_joints = [joint for joint in catalog["joints"]
            if joint["body_id"] == self._target and joint["qpos_count"] == joint["qvel_count"] == 1
            and joint["type"] in ("mjJNT_SLIDE", "mjJNT_HINGE")]
        for joint in self._target_joints:
            native = model.joint(joint["name"])
            if native.qposadr[0] != joint["qpos_start"] or native.dofadr[0] != joint["qvel_start"]:
                raise ValueError("Catalog and native carrier scalar-joint columns disagree")
        self._target_qpos = np.asarray([joint["qpos_start"] for joint in self._target_joints], dtype=int)
        self._target_qvel = np.asarray([joint["qvel_start"] for joint in self._target_joints], dtype=int)
        self._target_geoms = {row["id"] for row in catalog["geoms"] if row["body_id"] == self._target}
        if not self._target_geoms or self._target in resolved["body_ids"]:
            raise ValueError("The evaluator target must be a non-actor rigid collision body")
        relevant = set(resolved["geom_ids"]) | self._target_geoms
        for row in catalog["geoms"]:
            if row["id"] in relevant and (model.geom(row["name"]).id != row["id"]
                    or model.geom_bodyid[row["id"]] != row["body_id"]):
                raise ValueError("Evaluator geometry identities differ between catalog and native model")
        self._groups = [set(group) for group in resolved["contact_geom_group_ids"]]
        self._finger_indices = np.asarray([model.joint(name).qposadr[0] for name in finger_joint_names])
        self.settings = {key: DEFAULT_SETTINGS[key] for key in (
            "contact_distance_m", "contact_gap_s", "stable_grasp_s",
            "stable_relative_position_m", "stable_relative_rotation_rad",
            "stable_finger_slide_m", "stable_finger_hinge_rad", "release_absence_s")}
        self._finger_limits = np.asarray([
            self.settings["stable_finger_slide_m"] if model.jnt_type[model.joint(name).id] == mujoco.mjtJoint.mjJNT_SLIDE
            else self.settings["stable_finger_hinge_rad"] for name in finger_joint_names])
        self.binding = {"source": deepcopy(binding_source), "target_body_name": target_body_name,
            "target_body_id": self._target, "target_geom_ids": sorted(self._target_geoms),
            "target_geom_names": [model.geom(index).name for index in sorted(self._target_geoms)],
            "actor": deepcopy(actor), "contact_geom_group_ids": [sorted(group) for group in self._groups],
            "finger_joint_names": list(finger_joint_names), "tcp_site": tcp_site,
            "reference_body_name": reference_body_name, "reference_body_id": self._reference,
            "native_target_scalar_joints": [{key: joint[key] for key in
                ("name", "type", "range", "qpos_start", "qvel_start")} for joint in self._target_joints],
            "scope": "evaluator-only native identities; never supplied to planning"}
        # BEGIN original requirements and native frame identities stay evaluator-local.
        self._planning_result = planning_result
        self._goals = _FunctionalGoalEvaluator(model, catalog, planning_result, evaluation_owner_bindings)
        # BEGIN the caller must explicitly declare any diagnostic-only goals.
        self._diagnostic_goal_ids = validate_diagnostic_goal_ids(
            planning_result.execution_binding, diagnostic_goal_ids)
        # END default grading still requires every original functional goal.
        # END formal curve/scene data is read without adding a runtime observation.
        self._samples, self._contacts, self._offsets = [], [], [0]

    def observe(self, model, data, record):
        """Read one native state after forward; never write state or controls."""
        if model is not self._model:
            raise ValueError("A physics evaluator belongs to its original native model")
        now = float(data.time)
        if self._samples and not np.isclose(now-self._samples[-1][0], model.opt.timestep, atol=1e-8, rtol=0):
            raise ValueError("Evaluator samples must cover consecutive native physics steps")
        if self._offsets[-1] + data.ncon > self._maximum_contact_rows:
            raise ValueError("Native contact evidence exceeds the explicit local retention budget")
        touch = np.zeros(len(self._groups), dtype=bool)
        contacts = np.empty((data.ncon, 24), dtype=float)
        for index in range(data.ncon):
            contact = data.contact[index]
            a, b = int(contact.geom1), int(contact.geom2)
            valid = (contact.exclude == 0 and contact.efc_address >= 0
                     and contact.dist <= self.settings["contact_distance_m"])
            if valid:
                for group_index, group in enumerate(self._groups):
                    if (a in group and b in self._target_geoms) or (b in group and a in self._target_geoms):
                        touch[group_index] = True
            force = np.zeros(6)
            force_valid = contact.exclude == 0 and contact.efc_address >= 0
            if force_valid:
                mujoco.mj_contactForce(model, data, index, force)
            contacts[index] = np.r_[a, b, contact.exclude, contact.efc_address,
                contact.dist, contact.pos, contact.frame, force, force_valid]
        tool, target = _site_pose(data, self._site), _body_pose(data, self._target)
        fingers = data.qpos[self._finger_indices].copy()
        if not np.isfinite(tool).all() or not np.isfinite(target).all() or not np.isfinite(fingers).all():
            raise ValueError("Actual grasp evidence became non-finite")
        reference = None if self._reference is None else _body_pose(data, self._reference)
        # BEGIN finish all native reads before committing a synchronized sample.
        sample = (now, float(record["elapsed"]), tool, target, fingers, touch,
                  reference, data.qpos[self._target_qpos].copy(), data.qvel[self._target_qvel].copy())
        if (not np.isfinite(sample[0]) or not np.isfinite(sample[1])
                or not np.isfinite(sample[7]).all() or not np.isfinite(sample[8]).all()
                or (reference is not None and not np.isfinite(reference).all())):
            raise ValueError("Actual evaluator timestamps or carrier state became non-finite")
        self._goals.observe(data)
        self._samples.append(sample)
        # END an incomplete native read cannot append a partial evidence row.
        self._contacts.append(contacts)
        self._offsets.append(self._offsets[-1]+len(contacts))

    def finalize(self, output, *, execution_report, collection_report):
        """Save the evidence and grade only a completed physical prefix."""
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        # BEGIN a retained prefix cannot masquerade as complete GT evidence.
        capture_state = deepcopy(collection_report)
        expected = None if execution_report is None else execution_report["executed_steps"]+1
        times = np.asarray([row[0] for row in self._samples])
        elapsed = np.asarray([row[1] for row in self._samples])
        timestep = float(self._model.opt.timestep)
        reasons = []
        if not capture_state["completed"]:
            reasons.append("Evaluator collection did not complete without failure")
        if execution_report is None:
            reasons.append("No execution report establishes the sampled physical interval")
        else:
            if not execution_report["independent_evidence_sink"]["completed"]:
                reasons.append("Executor evidence callback did not cover every normal sample")
            if not execution_report["normal_step_accounting"]["step_accounting_exact"]:
                reasons.append("Normal physical step accounting is not exact")
        if len(self._samples) != expected:
            reasons.append("Retained sample count differs from initial state plus executed normal steps")
        if (not len(times) or not np.isfinite(times).all() or not np.isfinite(elapsed).all()
                or not np.isclose(elapsed[0], 0., atol=1e-8, rtol=0)
                or not np.allclose(np.diff(times), timestep, atol=1e-8, rtol=0)
                or not np.allclose(elapsed, np.arange(len(elapsed))*timestep, atol=1e-8, rtol=0)):
            reasons.append("Evidence does not cover the initial state and consecutive normal ticks")
        # Use retained timestamps, so a missing initial/interior state is not
        # mislabeled as a missing suffix. Never interpolate evidence rows.
        observed_ticks = elapsed[np.isfinite(elapsed)]/timestep
        rounded = np.rint(observed_ticks)
        observed_ticks = rounded[np.isclose(observed_ticks, rounded, atol=1e-5, rtol=0)].astype(np.int64)
        absent = (np.empty(0, dtype=np.int64) if expected is None else
                  np.setdiff1d(np.arange(expected), observed_ticks))
        runs = np.split(absent, np.flatnonzero(np.diff(absent) != 1)+1) if len(absent) else []
        missing_ranges = [[int(row[0]), int(row[-1])] for row in runs]
        coverage = {"complete": not reasons, "reasons": reasons,
            "expected_sample_count": expected, "retained_sample_count": len(self._samples),
            "normal_sample_index_convention": "0 is initial state; 1..executed_steps are normal steps; holds excluded",
            "missing_normal_sample_index_ranges": missing_ranges,
            "missing_normal_elapsed_intervals_s": [[a*timestep, b*timestep] for a, b in missing_ranges],
            "retained_elapsed_interval_s": None if not len(elapsed) else [float(elapsed[0]), float(elapsed[-1])]}
        if not self._samples:
            # BEGIN preserve explicit roles even when no evidence was captured.
            report = {"evaluation_kind": self._skill_kind, "physics_outcome": "incomplete_evidence",
                "physics_success": None, "sample_count": 0, "native_contact_rows": 0,
                "evidence_collection": capture_state, "evidence_coverage": coverage,
                "executor_evidence_sink": None if execution_report is None else
                    deepcopy(execution_report["independent_evidence_sink"]),
                "candidate_id": self._planning_result.execution_binding["candidate_id"],
                "completed_physical_prefix": False, "completed_current_curve": False,
                "evaluation_policy": {"diagnostic_goal_ids": deepcopy(self._diagnostic_goal_ids)}}
            # END missing evidence never yields a successful grade.
            (output/"evaluation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
            return report
        # END missing states are reported without fabricated samples or frames.
        tools = np.asarray([row[2] for row in self._samples])
        targets = np.asarray([row[3] for row in self._samples])
        fingers = np.asarray([row[4] for row in self._samples])
        touches = np.asarray([row[5] for row in self._samples])
        analysis, relative, simultaneous, bridged = _analyze_history(
            times, touches, tools, targets, fingers, self._finger_limits, self.settings)
        completed = (execution_report is not None
            and execution_report["status"] in ("stopped_for_evidence", "curve_finished")
            and execution_report["executed_steps"] > 0
            and execution_report["barrier_command_reached"])
        # Contact loss can be measured with the existing release debounce, but
        # it does not establish placement, containment or an articulation goal.
        # A trial that began without target contact cannot claim a release.
        any_touch = np.any(touches, axis=1)
        no_contact_runs = _runs(~any_touch)
        final_absence = next(((a, b) for a, b in no_contact_runs if b == len(times)), None)
        absence_duration = 0. if final_absence is None else float(times[-1]-times[final_absence[0]])
        release_confirmed = bool(final_absence is not None
            and np.any(any_touch[:final_absence[0]])
            and absence_duration >= self.settings["release_absence_s"]-1e-9)
        # BEGIN a component grasp predicate cannot replace the full original contract.
        trajectory = self._planning_result.trajectory
        full_curve_completed = bool(coverage["complete"] and completed and execution_report["status"] == "curve_finished"
            and elapsed[-1] >= pose_goal_schedule(trajectory, float(trajectory.progress_s[-1]),
                                                self._model.opt.timestep)[1]
            and elapsed[-1] >= trajectory.duration)
        goal_evidence = self._goals.evaluate(times, elapsed, tools, execution_report,
                                             full_curve_completed=full_curve_completed)
        # BEGIN original point goals and later support diagnostics remain separate.
        final_support = self._goals.final_support_diagnostics(times, elapsed,
            full_curve_completed=full_curve_completed)
        support_runtime = self._goals.support_runtime_coverage(execution_report)
        # END neither component contributes a new required terminal predicate.
        effect_evidence = []
        binding = self._planning_result.execution_binding
        for condition in binding["problem"].get("conditions", []):
            predicate = condition.get("effect_requirement")
            if not predicate:
                continue
            row = {"condition_id": condition["condition_id"], "skill_id": condition.get("skill_id"),
                "event_id": condition.get("event_id"), "predicate": predicate, "value": "U"}
            owner = self._goals._owners.get(condition.get("carrier_id"))
            if owner is None or owner["native_body_id"] != self._target:
                row["reason"] = "The original effect owner has no explicit mapping to this evaluated native body"
                effect_evidence.append(row)
                continue
            _, due = pose_goal_schedule(trajectory, float(condition["progress"]), self._model.opt.timestep)
            row["device_due_time_s"] = float(due) if np.isfinite(due) else None
            indexes = np.flatnonzero(elapsed >= due)[:1]
            if not indexes.size:
                row["reason"] = "Execution did not cover this original effect time"
                effect_evidence.append(row)
                continue
            index = int(indexes[0])
            row.update(sample_index=index, sample_sim_time_s=float(times[index]),
                       sample_elapsed_s=float(elapsed[index]))
            timed = [item for event in binding.get("timing_events", []) for item in event.get("timing", ())
                if item["check_target"] == "effect" and item["condition_id"] == condition["condition_id"]]
            if timed:
                row["reason"] = "Timed native effect acquisition is not established by a terminal contact snapshot"
            elif predicate == "stable_grasp_from_current_synchronized_evidence":
                if times[index]-times[0] < self.settings["stable_grasp_s"]-1e-9:
                    row["reason"] = "No complete original stable-grasp interval was observed before this effect time"
                else:
                    prefix, _, _, _ = _analyze_history(times[:index+1], touches[:index+1],
                        tools[:index+1], targets[:index+1], fingers[:index+1], self._finger_limits, self.settings)
                    row.update(value="T" if prefix["stable_grasp_at_stop"] else "F",
                        measurement=prefix, settings=deepcopy(self.settings),
                        source="Independent native contact and target-relative stability evaluation")
            elif predicate == "actual_release_from_current_feedback":
                prior_touch = np.any(touches[:index+1], axis=1)
                absent = next(((a, b) for a, b in _runs(~prior_touch) if b == index+1), None)
                if not np.any(prior_touch):
                    row["reason"] = "No preceding native target contact was observed for this release occurrence"
                elif times[index]-times[0] < self.settings["release_absence_s"]-1e-9:
                    row["reason"] = "The original release observation interval is not covered"
                else:
                    duration = 0. if absent is None else float(times[index]-times[absent[0]])
                    row.update(value="T" if duration >= self.settings["release_absence_s"]-1e-9 else "F",
                        contact_absence_s=duration, required_absence_s=self.settings["release_absence_s"],
                        source="Independent native contact-absence evaluation")
            else:
                row["reason"] = "No evaluator predicate is supplied for this original effect requirement"
            effect_evidence.append(row)
        process_evidence = []
        for collection in ("mechanisms", "pose_relations", "interiors", "couplings"):
            for index, requirement in enumerate(binding["problem"].get(collection, [])):
                process_evidence.append({"identity": [collection, index], "value": "U",
                    "source": deepcopy(requirement.get("source")),
                    "reason": "No complete native process predicate with the original frame/coordinate semantics is supplied"})
        # The existing grasp contract also requires stability at its final
        # stop. An earlier original effect witness is not a license to lose
        # the grasp before completion, and a late grasp cannot repair it.
        terminal_evidence = []
        if self._skill_kind == "grasp":
            terminal_evidence.append({"predicate": "stable_grasp_at_final_stop",
                "value": ("U" if not full_curve_completed
                    or times[-1]-times[0] < self.settings["stable_grasp_s"]-1e-9
                    else "T" if analysis["stable_grasp_at_stop"] else "F"),
                "sample_sim_time_s": float(times[-1]), "sample_elapsed_s": float(elapsed[-1]),
                "source": "Original independent grasp evaluator final stable-window requirement",
                "scope": "Separate from the original functional/effect checkpoint; both must hold"})
        # BEGIN one pure aggregation path supports both live capture and saved evidence.
        grade = summarize_skill_evidence({"candidate_id": binding["candidate_id"],
            "evaluation_kind": self._skill_kind,
            "functional_goal_evidence": goal_evidence, "effect_evidence": effect_evidence,
            "process_evidence": process_evidence, "terminal_evidence": terminal_evidence,
            "evidence_coverage": coverage, "completed_current_curve": full_curve_completed},
            binding=binding, trajectory=trajectory, diagnostic_goal_ids=self._diagnostic_goal_ids)
        # END no pose metric, original effect time or runtime observation is changed.
        target_q = np.asarray([row[7] for row in self._samples])
        target_dq = np.asarray([row[8] for row in self._samples])
        # BEGIN saved reports retain the shared explicit roles and unchanged metrics.
        report = {"evaluation_kind": self._skill_kind, **grade,
            "completed_physical_prefix": completed,
            "evidence_collection": capture_state, "evidence_coverage": coverage,
            "executor_evidence_sink": None if execution_report is None else
                deepcopy(execution_report["independent_evidence_sink"]),
            "completed_current_curve": full_curve_completed,
            "candidate_id": binding["candidate_id"],
            "evaluation_owner_bindings": self._goals.describe_bindings(),
            # BEGIN disclose online coverage without importing evaluator truth into it.
            "support_runtime_coverage": support_runtime,
            "final_support_diagnostics": final_support,
            # END no hidden success role is assigned to a last-sample diagnostic.
            "task_contract": self._task_contract,
            "binding": self.binding, "settings_source": "collect.preprocess.segmentation.DEFAULT_SETTINGS",
            "settings": self.settings, "finger_stability_limits": self._finger_limits.tolist(),
            "sample_count": len(times), "native_contact_rows": self._offsets[-1],
            "contact_sampling": "synchronized post-step mj_forward state, including an initial setup sample; native solved constraints and geometry are evaluator-only",
            "contact_force_scope": "diagnostic native force at the sampled state; force magnitude is not a grasp acceptance criterion",
            "component_evidence": {
                "configured_finger_contacts_at_stop": touches[-1].tolist(),
                "configured_finger_contact_absence_at_stop_s": absence_duration,
                "release_after_observed_target_contact_confirmed": release_confirmed,
                "release_confirmation_duration_s": self.settings["release_absence_s"],
                "release_scope": "absence of contact from the explicitly configured finger groups to this same native target; never implies task-goal satisfaction",
                "target_scalar_joint_positions_at_entry": target_q[0].tolist(),
                "target_scalar_joint_positions_at_stop": target_q[-1].tolist(),
                "target_scalar_joint_displacements": (target_q[-1]-target_q[0]).tolist()},
            "limits": ["Only this current curve and its actually sampled intervals are evaluated; no following skill is executed",
                       "Stable contact on a supported object does not certify a subsequent lift or load-bearing motion",
                       "This task predicate does not replace planner collision acceptance or certify every robot/environment contact",
                       "Simulator truth is used only by this independent evaluator; it never updates runtime owner poses, completed goals, effect history or a dependency barrier",
                       "Separate abort/dependency holds are not trajectory observations and cannot supply a missing goal or stable window"],
            }
        # END role assignment does not remove any measured component.
        if self._skill_kind == "grasp":
            report.update(analysis)
        else:
            report["grasp_stability_diagnostic"] = analysis
        contacts = np.concatenate(self._contacts, axis=0)
        extra = {"reference_pose_world": np.asarray([row[6] for row in self._samples])} if self._reference is not None else {}
        np.savez_compressed(output/"physics-evidence.npz", sim_time=times, elapsed=elapsed,
            tool_pose_world=tools, target_pose_world=targets, T_target_tool=relative,
            finger_q=fingers, touch_by_group=touches, simultaneous_touch=simultaneous,
            target_scalar_joint_names=np.asarray([joint["name"] for joint in self._target_joints], dtype=str),
            target_scalar_joint_q=target_q, target_scalar_joint_dq=target_dq,
            gap_bridged_simultaneous_touch=bridged,
            contact_offsets=np.asarray(self._offsets, dtype=np.int64),
            contact_geom1=contacts[:, 0].astype(np.int32), contact_geom2=contacts[:, 1].astype(np.int32),
            contact_exclude=contacts[:, 2].astype(np.int32), contact_efc_address=contacts[:, 3].astype(np.int32),
            contact_distance_m=contacts[:, 4], contact_position_world=contacts[:, 5:8],
            contact_frame=contacts[:, 8:17].reshape(-1, 3, 3), contact_force=contacts[:, 17:23],
            contact_force_valid=contacts[:, 23].astype(bool), **extra, **self._goals.arrays())
        (output/"evaluation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
# END independent native skill evidence with original collector predicates.
