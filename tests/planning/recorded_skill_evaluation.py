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
from manipulation.execution import _actual_pose_goal, pose_goal_schedule
# BEGIN explicit evaluator roles reuse the existing execution-scope validator.
from manipulation.execution_binding import validate_result_only_goal_ids
# END result-only execution scope alone does not change a goal's grading role.
from manipulation.motion import _inverse, _pose_array
from manipulation.planning_problem import (
    EventTimeConstraint, PoseEventRequirement, PosePath, TimeBindingError,
    _model_from_diagnostics, resolve_event_window,
)


# BEGIN explicitly diagnostic goals retain measurements without vetoing skill effects.
def validate_diagnostic_goal_ids(binding, trajectory, values):
    """Validate an explicit grading role without changing execution obligations.

    Result-only means that the executor does not wait for a goal; it does not
    mean that the goal is optional for success. A diagnostic goal must be named
    separately and already belong to that result-only policy. Reuse the same
    terminal/arrival/contact checks, and reject a goal used as a subsequent
    event's time origin. Process and effect predicates cannot be selected here.
    """
    # A previous completion must not bypass the shared arrival/terminal checks
    # when assigning a grading role. The binding itself remains untouched.
    selected = validate_result_only_goal_ids({**binding, "completed_goal_ids": []}, trajectory, values)
    declared = binding.get("device_policy", {}).get("result_only_goal_ids", [])
    for identity in selected:
        if identity not in declared:
            raise ValueError("A diagnostic goal must already be an explicit result-only goal")
        for event in (*trajectory.events, *binding.get("timing_events", ())):
            if any(tuple((item.get("origin_event") or {}).get("goal_id") or ()) == tuple(identity)
                   for item in event.get("timing", ())):
                raise ValueError("A goal used by a subsequent event cannot be diagnostic")
    return selected


def summarize_skill_evidence(report, *, binding, trajectory, diagnostic_goal_ids=()):
    """Regrade saved components without measuring, simulating or mutating them.

    The caller supplies the same formal binding/curve plus the newly declared
    diagnostic identities. Every goal retains its original T/F/U and metrics.
    All unselected goals, original effects, processes and terminal predicates
    remain required. Complete acquisition and full-curve execution are also
    required; this result never becomes an online observation or dependency fact.
    """
    selected = validate_diagnostic_goal_ids(binding, trajectory, diagnostic_goal_ids)
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

    The initial rigid offset is evaluator-only calibration. Subsequent native
    motion moves that same frame; a native body origin is never silently used
    as the original observed material frame. No returned row is runtime input.
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
                else:
                    self._unsupported_goals.append({"identity": list(identity), "value": "U",
                        "requirement": deepcopy(row),
                        "reason": "No native coordinate/support predicate with its original frame/coordinate binding is supplied"})
        owners = self._binding["scene"].get("owners", [])
        self._owners = {}
        if not isinstance(owner_bindings, list):
            raise ValueError("evaluation_owner_bindings must be an explicit list")
        entry_progress = float(problem.get("start_progress", 0.))
        for row in owner_bindings:
            if (not isinstance(row, dict) or set(row) != {"owner_id", "native_body_name"}
                    or any(not isinstance(value, str) or not value.strip() for value in row.values())):
                raise ValueError("Each evaluator owner binding needs exactly owner_id and native_body_name")
            identity, name = row["owner_id"], row["native_body_name"]
            if identity in self._owners:
                raise ValueError("Evaluator owner bindings must be unique")
            selected = [owner for owner in owners if owner["owner_id"] == identity]
            bodies = [body for body in catalog["bodies"] if body["name"] == name]
            if (len(selected) != 1 or len(bodies) != 1
                    or model.body(name).id != bodies[0]["id"]):
                raise ValueError("Evaluator owner/native body identities must match one declared scene/catalog body")
            owner = selected[0]
            lower, upper = owner["pose_interval"]
            if not lower <= entry_progress <= upper or upper <= lower:
                raise ValueError("An evaluator owner binding requires its current entry pose interval")
            path = deepcopy(owner["pose_path"])
            if path["motion_model"] is not None:
                path["motion_model"] = _model_from_diagnostics(path["motion_model"])
            entry_pose = PosePath(**path).sample_numpy((entry_progress-lower)/(upper-lower))
            self._owners[identity] = {"owner_id": identity, "native_body_name": name,
                "native_body_id": int(bodies[0]["id"]),
                "T_world_owner_entry": _pose_array(entry_pose, "Bound evaluator owner entry pose"),
                "T_body_owner": None}
        self._owner_pose_samples = []

    def observe(self, data):
        """Read and validate native transforms; do not publish observations."""
        poses = {}
        for identity, binding in self._owners.items():
            body = _pose_array(_body_pose(data, binding["native_body_id"]), "Native evaluator body pose")
            if binding["T_body_owner"] is None:
                binding["T_body_owner"] = _pose_array(
                    _inverse(body) @ binding["T_world_owner_entry"], "Evaluator material-frame calibration")
            poses[identity] = _pose_array(body @ binding["T_body_owner"], "Actual evaluator owner frame")
        self._owner_pose_samples.append(poses)

    def describe_bindings(self):
        return [{key: value.tolist() if isinstance(value, np.ndarray) else value
                 for key, value in row.items()} for row in self._owners.values()]

    def _measure(self, identity, index, times, tools):
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
            "timestamp_s": float(times[index]), "source": "Independent evaluator calibrated native owner poses"}
        measured = _actual_pose_goal(self._requirements[identity], feedback, observation)
        measured["source"] = "Independent evaluator applies the original functional allowed set to native truth"
        return measured

    def evaluate(self, times, elapsed, tools, execution_report, *, full_curve_completed):
        """Grade original goal times using trajectory samples, excluding holds."""
        evidence = deepcopy(self._unsupported_goals)
        origin = float(self._binding.get("request_time_origin_s", 0.))
        request_times = origin + times-times[0]
        history = [] if execution_report is None else execution_report.get("history", [])
        timing_events = self._binding.get("timing_events", [])
        result_only = self._binding.get("device_policy", {}).get("result_only_goal_ids", [])
        for identity, requirement in self._requirements.items():
            scheduled, due = pose_goal_schedule(self._trajectory, requirement.progress, self._model.opt.timestep)
            row = {"identity": list(identity), "value": "U", "query_count": 0,
                "execution_scope": "result" if list(identity) in result_only else "execution",
                "progress": requirement.progress,
                "scheduled_curve_time_s": float(scheduled) if np.isfinite(scheduled) else None,
                "device_due_time_s": float(due) if np.isfinite(due) else None,
                "position_tolerance_m": requirement.position_tolerance_m,
                "rotation_tolerance_rad": requirement.rotation_tolerance_rad,
                "requirement_source": deepcopy(requirement.source),
                "scope": "Independent result evidence only; does not update runtime completion or release a dependency"}
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

    def arrays(self):
        identities = list(self._owners)
        poses = np.asarray([[row[key] for key in identities] for row in self._owner_pose_samples])
        return {"evaluation_owner_ids": np.asarray(identities, dtype=str),
                "evaluation_owner_poses_world": poses.reshape(len(self._owner_pose_samples), len(identities), 4, 4)}
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
            planning_result.execution_binding, planning_result.trajectory, diagnostic_goal_ids)
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
