"""Independent native-physics evidence for an explicitly identified skill.

This module is evaluator-only: native object identity, live object poses and
contacts never enter a planning problem or an observation builder. Grasp
stability uses the collector's existing segmentation predicates and defaults.
Grasp grading certifies its measured interval only. Manipulation traces retain
native carrier/reference poses, scalar joints and contacts, while unsupported
task goals remain explicitly unverified.
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


def _body_pose(data, index):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.xmat[index].reshape(3, 3), data.xpos[index]
    return pose


def _site_pose(data, index):
    pose = np.eye(4)
    pose[:3, :3], pose[:3, 3] = data.site_xmat[index].reshape(3, 3), data.site_xpos[index]
    return pose


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
                 reference_body_name,
                 maximum_contact_rows=1_000_000):
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
        self._samples.append((now, float(record["elapsed"]), tool, target, fingers, touch,
                              reference, data.qpos[self._target_qpos].copy(), data.qvel[self._target_qvel].copy()))
        self._contacts.append(contacts)
        self._offsets.append(self._offsets[-1]+len(contacts))

    def finalize(self, output, *, execution_report):
        """Save the evidence and grade only a completed physical prefix."""
        if not self._samples:
            raise ValueError("A skill trial needs at least its observed initial native state")
        output = Path(output)
        output.mkdir(parents=True, exist_ok=True)
        times = np.asarray([row[0] for row in self._samples])
        elapsed = np.asarray([row[1] for row in self._samples])
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
        passed = bool(analysis["stable_grasp_at_stop"]) if completed and self._skill_kind == "grasp" else None
        outcome = ("not_evaluated" if not completed else "unverified" if self._skill_kind != "grasp"
                   else "passed" if passed else "failed")
        target_q = np.asarray([row[7] for row in self._samples])
        target_dq = np.asarray([row[8] for row in self._samples])
        report = {"evaluation_kind": self._skill_kind, "physics_outcome": outcome,
            "physics_success": passed, "completed_physical_prefix": completed,
            "task_contract": self._task_contract,
            "binding": self.binding, "settings_source": "collect.preprocess.segmentation.DEFAULT_SETTINGS",
            "settings": self.settings, "finger_stability_limits": self._finger_limits.tolist(),
            "sample_count": len(times), "native_contact_rows": self._offsets[-1],
            "contact_sampling": "synchronized post-step mj_forward state, including an initial setup sample; native solved constraints and geometry are evaluator-only",
            "contact_force_scope": "diagnostic native force at the sampled state; force magnitude is not a grasp acceptance criterion",
            "success_definition": ("At the stopping state, every configured finger contact group touches the same target body throughout a 0.2 s stable interval under the original gap/motion thresholds"
                if self._skill_kind == "grasp" else "No manipulation task-success predicate is inferred from grasp stability or source demonstration success"),
            "component_evidence": {
                "configured_finger_contacts_at_stop": touches[-1].tolist(),
                "configured_finger_contact_absence_at_stop_s": absence_duration,
                "release_after_observed_target_contact_confirmed": release_confirmed,
                "release_confirmation_duration_s": self.settings["release_absence_s"],
                "release_scope": "absence of contact from the explicitly configured finger groups to this same native target; never implies task-goal satisfaction",
                "target_scalar_joint_positions_at_entry": target_q[0].tolist(),
                "target_scalar_joint_positions_at_stop": target_q[-1].tolist(),
                "target_scalar_joint_displacements": (target_q[-1]-target_q[0]).tolist()},
            "limits": ["Only the measured grasp prefix is evaluated; no following skill is executed",
                       "Stable contact on a supported object does not certify a subsequent lift or load-bearing motion",
                       "This task predicate does not replace planner collision acceptance or certify every robot/environment contact",
                       "Simulator truth is used only by this independent evaluator; the executor evidence barrier is not automatically released"],
            }
        if self._skill_kind == "grasp":
            report.update(analysis)
        else:
            report["grasp_stability_diagnostic"] = analysis
            report["unverified_requirements"] = [
                {"requirement": self._task_contract["task"], "status": "unverified",
                 "reason": "Live carrier motion and contacts are recorded; no task-specific support, cavity, or planner-coordinate to native-joint predicate has been established"},
                {"requirement": "functional events", "status": "unverified",
                 "reason": "Source event identities and tolerances are retained; this diagnostic does not assume a current native goal frame or coordinate sign/zero binding"},
                {"requirement": "post-barrier continuation", "status": "not_executed",
                 "reason": "The original evidence barrier remains in force even when a component contact predicate is measured"}]
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
            contact_force_valid=contacts[:, 23].astype(bool), **extra)
        (output/"evaluation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        return report
# END independent native skill evidence with original collector predicates.
