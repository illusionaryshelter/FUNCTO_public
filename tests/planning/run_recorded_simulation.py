"""Run an exported skill prefix in its recorded native scene and retain physics.

All scene, morphology, candidate and camera identities are explicit inputs.
Object poses and native contacts are read only by the independent evaluator.
Runtime observations must come from an explicitly supplied usable observation
channel; simulator truth never supplies runtime owner poses or effect evidence.
``--no-video`` is a CPU-only execution option; selecting one or two ``--view``
arguments creates native renderers and must run separately from CUDA planning.
"""

# BEGIN manifest-driven physical trial with independent evidence and video.
from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT))

# BEGIN production execution owns control; this adapter owns recording setup.
from manipulation.execution import ExecutionLimits, FixedFingerTargets, TrajectoryExecutor, _feedback_json
from manipulation.execution_binding import rebind_execution_result
from recorded_simulation import restore_recorded_simulation, load_saved_trajectory
from recorded_task_planning import save_curve
from robot_validation_fixture import build_robot_validator
# END production execution and explicit recording adapter.
from recorded_skill_evaluation import NativeSkillPhysicsEvaluator


def _read(path):
    path = Path(path)
    if path.stat().st_size > 16 * 1024**2:
        raise ValueError(f"Trial metadata exceeds the bounded 16 MiB input size: {path}")
    return json.loads(path.read_text())


def _one(rows, key, value):
    found = [row for row in rows if row[key] == value]
    if len(found) != 1:
        raise ValueError(f"Expected one explicit {key}={value!r}; found {len(found)}")
    return found[0]


# BEGIN result-only scope always names one original formal goal identity.
def _goal_identity(value):
    """Parse a CLI identity without inferring a goal from its task or label."""
    collection, separator, raw_index = value.partition(":")
    if (not separator or collection not in ("pose_events", "coordinate_events", "supports")
            or not raw_index.isdecimal()):
        raise argparse.ArgumentTypeError("Use an original goal identity such as pose_events:0")
    return [collection, int(raw_index)]
# END semantic rebind checks membership, uniqueness and hard dependencies.


class _TrialVideos:
    """Stream native fixed-camera frames directly to the collector's codec."""

    def __init__(self, simulation, *, views, workspace_root, output, fps):
        import cv2
        from collect.camera import FixedCamera
        from recorded_observation import camera_config_from_calibration

        if not 1 <= len(views) <= 2 or len({view["id"] for view in views}) != len(views):
            raise ValueError("Choose one or two distinct explicit current view identities")
        self._cv2, self._simulation, self._entries = cv2, simulation, []
        self._counter, self._last_elapsed = 0, None
        self.records = []
        ratio = 1 / (fps * simulation.native_model.opt.timestep)
        if not np.isclose(ratio, round(ratio), atol=1e-10, rtol=0):
            raise ValueError("Video frequency must divide the native physics rate")
        self._stride = int(round(ratio))
        try:
            for view in views:
                calibration_path = (workspace_root/view["calibration_path"]).resolve()
                calibration = _read(calibration_path)
                camera = FixedCamera(simulation.native_model, simulation.native_data,
                    width=calibration["width"], height=calibration["height"], camera_name=None,
                    camera_config=camera_config_from_calibration(calibration))
                entry = {"camera": camera, "writer": None, "times": []}
                self._entries.append(entry)
                native = camera.calibration
                if (not np.allclose(native["K"], calibration["K"], atol=1e-9, rtol=0)
                        or not np.allclose(native["T_world_camera_cv"], calibration["T_world_camera_cv"], atol=1e-9, rtol=0)
                        or not np.allclose([native["near_m"], native["far_m"]],
                                           [calibration["near_m"], calibration["far_m"]], atol=1e-9, rtol=1e-9)):
                    raise ValueError("Native trial camera does not reproduce the original calibration")
                if any(token in native["rendering_device"]["renderer"].lower() for token in ("llvmpipe", "softpipe")):
                    raise RuntimeError("The selected video renderer is software; a native GPU rendering context is required")
                path = output/f"physical-{view['id']}.mp4"
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"),
                                         float(fps), (calibration["width"], calibration["height"]))
                entry["writer"] = writer
                if not writer.isOpened():
                    raise RuntimeError(f"The collector's existing mp4v codec could not open {path}")
                record = {"view_id": view["id"], "path": str(path.resolve()), "fps": float(fps),
                    "frame_count": 0, "calibration_path": str(calibration_path),
                    "renderer": native["rendering_device"], "capture_times_path": str((output/f"video-times-{view['id']}.npy").resolve()),
                    "timing": "constant-rate MP4; exact simulation capture times are saved separately; final off-grid barrier state is included once"}
                entry["record"] = record
                self.records.append(record)
        except Exception as error:
            # BEGIN preserve initialization failure while still closing every owner.
            try:
                self.close()
            except Exception as cleanup_error:
                raise error from cleanup_error
            # END cleanup failure is a cause, not a replacement verdict.
            raise

    def capture(self, elapsed, *, final=False):
        if self._last_elapsed is not None and abs(elapsed-self._last_elapsed) < 1e-10:
            return
        due = final or self._counter % self._stride == 0
        self._counter += 1
        if not due:
            return
        for entry in self._entries:
            frame = entry["camera"].capture(self._simulation.native_data)
            entry["writer"].write(self._cv2.cvtColor(frame["rgb"], self._cv2.COLOR_RGB2BGR))
            entry["times"].append([float(elapsed), float(self._simulation.native_data.time)])
            entry["record"]["frame_count"] += 1
        self._last_elapsed = float(elapsed)

    def close(self):
        # BEGIN each video owns cleanup even when another evidence sink fails.
        first_error = None
        for entry in self._entries:
            try:
                if entry["writer"] is not None:
                    entry["writer"].release()
            except Exception as error:
                first_error = first_error or error
            try:
                entry["camera"].close()
            except Exception as error:
                first_error = first_error or error
            try:
                if "record" in entry:
                    np.save(entry["record"]["capture_times_path"], np.asarray(entry["times"]))
            except Exception as error:
                first_error = first_error or error
        if first_error is not None:
            raise first_error
        # END every writer, native context and timestamp archive is attempted.


# BEGIN allow an upstream observer to supply current scene/effect evidence.
def run(args, *, observation_callback=None):
    """Execute one skill with optional upstream current-observation evidence.

    The native physics evaluator remains independent and never authorizes a
    runtime dependency or supplies owner poses. A caller may provide an
    existing proprioceptive or visual observation callback through the same
    contract; the CLI does not create a semantic sensor from simulator truth.
    """
# END absent observation evidence retains the executor's explicit U outcome.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    report = {"status": "preparing", "segment_id": args.segment,
        "trajectory_path": str(args.trajectory.resolve()), "trajectory_classification": None,
        # BEGIN independent position-target input and unverified robot handoff.
        "finger_targets_path": str(args.finger_targets.resolve()),
        "execution_limits_path": str(args.execution_limits.resolve()),
        "robot_feedback_path": None, "barrier_feedback_path": None,
        "evidence_passed": False,
        # END independent position-target input and unverified robot handoff.
        "execution_input": "formal_planning_result", "physics_outcome": "not_evaluated",
        # BEGIN distinguish online evidence from the independent offline grade.
        "online_observation_provided": observation_callback is not None,
        "effect_evidence_scope": ("Caller-supplied observation callback; executor validates current timestamps and provenance"
            if observation_callback is not None else
            "No online semantic observer; offline native physics grading does not resolve execution effects"),
        # END the recording fixture alone is not a closed-loop effect proof.
        "physics_success": None, "videos": [], "video_requested": not args.no_video,
        "execution_report_path": str(output/"execution"/"execution-report.json"),
        "physical_trajectory_path": str(output/"execution"/"physical-trajectory.npz"),
        "evaluation_report_path": str(output/"evaluation-report.json"),
        "physics_evidence_path": str(output/"physics-evidence.npz"),
        "scope": "One actual native-physics skill prefix; no next skill or full-sequence claim",
        "planner_inputs_changed": False, "execution_tested": False}
    simulation = evaluator = videos = execution = executor = None
    caught = None
    # BEGIN optional evidence sinks cannot become physical execution gates.
    report["evidence_collection"] = {name: {
        "configured": configured, "initialized": False, "completed": False,
        "disabled": False, "first_error": None,
        "attempted_callbacks": 0, "successful_callbacks": 0}
        for name, configured in (("evaluator", True), ("video", not args.no_video))}

    def evidence_failure(name, error, phase, record=None):
        state = report["evidence_collection"][name]
        state.update(disabled=True, completed=False)
        if state["first_error"] is None:
            # Evidence diagnostics must not raise while handling their own
            # failure, including an unreadable/non-finite simulator clock.
            try:
                sim_time = float(simulation.sim_time)
                sim_time = sim_time if np.isfinite(sim_time) else None
            except Exception:
                sim_time = None
            state["first_error"] = {"type": type(error).__name__, "message": str(error),
                "phase": phase, "curve_time": None if record is None else record["elapsed"],
                "sim_time": sim_time}
    # END no evidence error replaces the executor's primary physical error.
    try:
        manifest, fixture, feedback = _read(args.manifest), _read(args.robot_fixture), _read(args.feedback)
        root = Path(manifest["workspace_root"])
        segment = _one(manifest["segments"], "id", args.segment)
        program = _read(root/segment["program_path"])
        if program["skill_kind"] not in ("grasp", "manipulation"):
            raise ValueError("Only the explicit compiled grasp/manipulation skill contracts can be executed")
        report["skill_kind"] = program["skill_kind"]
        if feedback["segment_id"] != args.segment or feedback["entry_frame"] != segment["snapshot"]["entry_frame"]:
            raise ValueError("Named feedback and selected segment must identify the same entrance")
        if (feedback["joint_names"] != fixture["joint_names"]
                or feedback["arm_joint_names"] != fixture["arm_joint_names"]
                or feedback["finger_joint_names"] != fixture["finger_joint_names"]):
            raise ValueError("The trial fixture and named feedback must have identical robot bindings")
        # BEGIN preserve explicit coordinate identities for the actual handoff.
        tool_transform = np.asarray(feedback["tool_transform"], dtype=float)
        if (feedback["tool_link"] != fixture["tool_link"]
                or feedback["reference_frame"] != segment["snapshot"]["reference_frame"]
                or tool_transform.shape != (4, 4) or not np.isfinite(tool_transform).all()):
            raise ValueError("Robot handoff requires the initial feedback's explicit tool and reference-frame binding")
        initial_feedback = {"joint_names": feedback["joint_names"],
            "arm_joint_names": feedback["arm_joint_names"], "finger_joint_names": feedback["finger_joint_names"],
            "q": feedback["start_q"], "dq": feedback["start_dq"],
            "base_pose": feedback["base_pose"], "tool_pose": feedback["eef_pose"],
            "sim_time": feedback["sim_time_s"], "timestamp_s": feedback["sim_time_s"],
            "reference_frame": feedback["reference_frame"], "tool_link": feedback["tool_link"],
            "tool_transform": tool_transform.tolist(),
            "feedback_id": str(args.feedback.resolve()), "source": feedback["source"]}
        finger_targets = FixedFingerTargets.from_mapping(_read(args.finger_targets))
        # BEGIN require physical tracking and actual-hand limits explicitly.
        execution_limits = ExecutionLimits.from_mapping(_read(args.execution_limits))
        # END require physical tracking and actual-hand limits explicitly.
        # END preserve explicit coordinate identities for the actual handoff.
        recording = (root/manifest["source_recording"]).resolve()
        source_manifest_path = recording/"manifest.json"
        source = _read(source_manifest_path)
        environment = source["environment"]
        arm_config = environment["controller_config"]["body_parts"][args.controller_part]
        if (arm_config["type"] != "JOINT_POSITION" or arm_config["input_type"] != "absolute"
                or arm_config["gripper"]["type"] != "GRIP" or arm_config["interpolation"] is not None):
            raise ValueError("This trajectory executor requires the recorded absolute joint controller and uninterpolated GRIP")
        if Path(fixture["episode_path"]).resolve() != recording/"episode.h5":
            raise ValueError("Simulator fixture and input manifest refer to different source episodes")
        # BEGIN restore and execute the same formal result, without label gates.
        planning_result = load_saved_trajectory(args.trajectory)
        curve = planning_result.trajectory
        # END the loader checks exact candidate, clock and physical policy.
        # BEGIN use the same archive already checked by the formal loader.
        # Ordinary inputs keep their 16 MiB bound; the complete scene archive
        # owns its separate admission limit and is not read or decoded twice.
        report["trajectory_classification"] = planning_result.diagnostics["archive_classification"]
        # END reporting cannot observe a different second copy of the archive.
        if {event["skill_id"] for event in curve.events} != {args.segment}:
            raise ValueError("The exported candidate must contain only the selected skill instance")
        # BEGIN restore native context before constructing the production executor.
        simulation, restore_report = restore_recorded_simulation(scene_path=fixture["source_scene_xml"],
            catalog_path=fixture["catalog_path"], episode_path=fixture["episode_path"],
            frame_index=feedback["entry_frame"], base_link=fixture["base_link"], tcp_site=fixture["tcp_site"],
            arm_joint_names=fixture["arm_joint_names"], finger_joint_names=fixture["finger_joint_names"],
            arm_kp=arm_config["kp"], arm_kd=arm_config["kd"], velocity_limits=fixture["velocity_limits"])
        current = simulation.proprioception()
        differences = {"q": float(np.max(np.abs(current["q"]-np.asarray(feedback["start_q"])))),
            "dq": float(np.max(np.abs(current["dq"]-np.asarray(feedback["start_dq"])))),
            "base_pose": float(np.max(np.abs(current["base_pose"]-np.asarray(feedback["base_pose"])))),
            "tool_pose": float(np.max(np.abs(current["tool_pose"]-np.asarray(feedback["eef_pose"])))),
            "time": abs(current["sim_time"]-feedback["sim_time_s"])}
        if max(differences.values()) > 1e-9:
            raise ValueError(f"Restored native feedback differs from the selected observed entrance: {differences}")
        # END restoration is independent of trajectory execution.
        # BEGIN explicitly rebind the unchanged curve to the current device contract.
        source_candidate = planning_result.execution_binding["candidate_id"]
        planning_result = rebind_execution_result(planning_result,
            controller_policy=simulation.describe()["controller_policy"],
            result_only_goal_ids=args.result_only_goal_ids, finger_targets=finger_targets,
            execution_limits=execution_limits,
            source={"kind": "explicit_recorded_trial_execution_contract",
                "source_trajectory": str(args.trajectory.resolve()),
                "source_candidate_id": source_candidate})
        rebound_path = output/"rebound-planning"/"trajectory.npz"
        rebound_path.parent.mkdir()
        save_curve(rebound_path, planning_result.trajectory, classification="planner_accepted",
            execution_limits=execution_limits.describe(), planning_result=planning_result)
        report.update(source_candidate_id=source_candidate,
            execution_candidate_id=planning_result.execution_binding["candidate_id"],
            rebound_trajectory_path=str(rebound_path),
            result_only_goal_ids=deepcopy(args.result_only_goal_ids),
            controller_policy=simulation.describe()["controller_policy"],
            planning_execution_limits_match=True,
            planning_hand_contract="Explicit rebound device policy; physical limits and original finger polynomial are preserved")
        # END no solve, clock change, source archive mutation or truth feedback occurs.
        report.update(restored_feedback_errors=differences, restore=restore_report,
            original_controller_configuration={"source": str(source_manifest_path),
                "part": args.controller_part, "configuration": deepcopy(arm_config),
                "policy_frequency_hz": environment["control_freq_hz"]},
            provenance={"input_manifest": str(args.manifest.resolve()),
                "robot_fixture": str(args.robot_fixture.resolve()), "feedback": str(args.feedback.resolve()),
                "finger_targets": str(args.finger_targets.resolve()),
                "program": str((root/segment["program_path"]).resolve()),
                "evaluator_evidence": str(args.evaluator_evidence.resolve())})
        # BEGIN evaluator metadata and native identity failures are evidence-only.
        try:
            evidence = _read(args.evaluator_evidence)
            source_segment = _one(evidence["segments"], "segment_id", segment["source_library_segment"])
            if source_segment["interaction"] != program["skill_kind"]:
                raise ValueError("The evaluator's explicit source identity must match the compiled skill kind")
            actor = {key: deepcopy(evidence["actor"][key]) for key in
                     ("entity_name", "eef_site_name", "contact_geom_groups")}
            catalog = _read(fixture["catalog_path"])
            reference_role = program["demo"]["roles"].get("reference")
            reference_name = None
            if reference_role is not None and reference_role["body_id"] is not None:
                reference_name = _one(catalog["bodies"], "id", reference_role["body_id"])["name"]
            evaluator = NativeSkillPhysicsEvaluator(simulation.native_model, catalog=catalog,
                actor=actor, target_body_name=source_segment["carrier_body_name"],
                finger_joint_names=fixture["finger_joint_names"], tcp_site=fixture["tcp_site"],
                program=program, reference_body_name=reference_name,
                planning_result=planning_result, evaluation_owner_bindings=evidence["evaluation_owner_bindings"],
                binding_source={"path": str(args.evaluator_evidence.resolve()),
                    "source_segment_id": segment["source_library_segment"],
                    "fields_used": ["actor identity and contact-group names", "carrier_body_name",
                        "reference body identity and declared source function requirements",
                        "evaluation_owner_bindings: explicit observed owner IDs and native body names"],
                    "original_contact_outcome_or_object_trajectory_used": False})
            report["evidence_collection"]["evaluator"]["initialized"] = True
        except Exception as error:
            evidence_failure("evaluator", error, "initialize")
        # END failed GT evidence setup never prevents physical execution.
        if not args.no_video:
            # BEGIN camera initialization has the same independent boundary.
            try:
                views = [_one(segment["snapshot"]["views"], "id", name) for name in args.views]
                videos = _TrialVideos(simulation, views=views, workspace_root=root,
                                      output=output, fps=environment["control_freq_hz"])
                report["videos"] = videos.records
                report["evidence_collection"]["video"]["initialized"] = True
            except Exception as error:
                evidence_failure("video", error, "initialize")
            # END missing camera records are reported, never synthesized.

        def observe(model, data, record):
            # BEGIN executor supplies the initial state and each normal step once.
            # Stop only the failed sink; its peer can retain independent evidence.
            phase = "initial_state" if record["elapsed"] == 0. else "after_step"
            state = report["evidence_collection"]["evaluator"]
            if evaluator is not None and not state["disabled"]:
                state["attempted_callbacks"] += 1
                try:
                    evaluator.observe(model, data, record)
                    state["successful_callbacks"] += 1
                except Exception as error:
                    evidence_failure("evaluator", error, phase, record)
            state = report["evidence_collection"]["video"]
            if videos is not None and not state["disabled"]:
                state["attempted_callbacks"] += 1
                try:
                    videos.capture(record["elapsed"])
                    state["successful_callbacks"] += 1
                except Exception as error:
                    evidence_failure("video", error, phase, record)
            # END no evaluator return value or exception feeds runtime control.

        # BEGIN mandatory independent robot verification before native commands.
        validator = build_robot_validator(fixture,
            reference_frame=feedback["reference_frame"],
            T_reference_base=np.asarray(feedback["base_pose"], dtype=float),
            tool_frame=feedback["tool_link"], T_frame_tcp=tool_transform)
        executor = TrajectoryExecutor(simulation, geometry_checker=validator)
        report["status"] = "executing"
        execution = executor.execute(planning_result,
            finger_targets=finger_targets, initial_feedback=initial_feedback,
            execution_limits=execution_limits,
            output=output/"execution", step_callback=observe,
            # BEGIN pass real upstream observations without reading evaluator truth.
            observation_callback=observation_callback,
            # END the production executor owns validation and dependency gates.
            max_physics_steps=args.max_physics_steps,
            start_position_tolerance=args.start_position_tolerance,
            start_velocity_tolerance=args.start_velocity_tolerance)
        # END independent verification and the sole production execution loop.
        report["execution_tested"] = execution["executed_steps"] > 0
        # BEGIN export robot-only feedback for a separately verified handoff.
        report.update(robot_feedback_path=execution["robot_feedback_path"],
                      barrier_feedback_path=execution["barrier_feedback_path"],
                      handoff_status="requires current scene observation and verified holding evidence")
        # END export robot-only feedback for a separately verified handoff.
        report["status"] = "completed"
    except Exception as error:
        caught = error
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        # BEGIN a diagnostic write cannot replace the original physical failure.
        try:
            (output/"exception.txt").write_text(traceback.format_exc())
        except Exception as write_error:
            report["exception_write_error"] = f"{type(write_error).__name__}: {write_error}"
        # END remaining evidence finalization must still run.
    finally:
        # BEGIN recover the executor's own report without parsing its archive again.
        # Failed preflight can produce extensive geometry diagnostics. Its
        # in-memory report already owns those values and its real abort hold;
        # ordinary external inputs retain the separate 16 MiB read bound.
        if execution is None and executor is not None:
            execution = executor.last_execution_report
        if execution is not None:
            try:
                hold_steps = sum(0 if execution.get(key) is None else
                                 execution[key]["physical_steps"]
                                 for key in ("abort_hold", "dependency_hold"))
                # BEGIN preserve confirmed advancement separately from budget charges.
                # An uncertain native clock leaves confirmed steps as a lower
                # bound. Do not turn that bound, or a charged attempt, into an
                # exact actual count or an assertion that no physics occurred.
                accounting = execution["physics_step_accounting"]
                confirmed_steps = accounting["physical_steps"]
                exact_steps = accounting["step_accounting_exact"]
                normal_exact = execution["normal_step_accounting"]["step_accounting_exact"]
                report.update(execution_tested=(True if execution["executed_steps"] else
                                               False if normal_exact else None),
                    actual_physics_tested=(True if confirmed_steps else False if exact_steps else None),
                    trajectory_executed_steps=execution["executed_steps"],
                    hold_physics_steps=hold_steps,
                    actual_physics_steps=confirmed_steps if exact_steps else None,
                    confirmed_physics_steps=confirmed_steps,
                    charged_physics_steps=accounting["charged_steps"],
                    step_accounting_exact=exact_steps,
                    physics_step_accounting=deepcopy(accounting),
                    # Separate backend holds do not call the evaluator hook.
                    evaluator_excluded_hold_physics_steps=hold_steps,
                    robot_feedback_path=execution.get("robot_feedback_path"),
                    barrier_feedback_path=execution.get("barrier_feedback_path"))
                # END the executor owns the shared limit and every stop charge.
                # BEGIN retain bounded first-failure evidence without its full binding.
                report["execution_persistence_errors"] = execution.get("persistence_errors", [])
                # BEGIN expose the executor's independent callback completeness.
                report["independent_evidence_sink"] = deepcopy(execution["independent_evidence_sink"])
                # END this diagnostic is separate from physical guard status.
                preflight = execution.get("geometry_preflight", {})
                report["geometry_preflight_summary"] = {key: preflight[key] for key in (
                    "passed", "checked_interval_s", "reference_collision_checked",
                    "continuous_collision_certified", "observed_scene_checked",
                    "unknown_space_checked", "unknown_space_certified") if key in preflight}
                failure = execution.get("feedback_failure")
                if failure is not None:
                    violations = failure.get("violations", [])
                    first = violations[0] if violations else {}
                    report["feedback_failure_summary"] = {
                        key: failure[key] for key in ("point", "curve_time_s", "physical_elapsed_s",
                            "reference_arm_q", "reference_arm_dq", "actual_feedback") if key in failure}
                    report["feedback_failure_summary"].update(violation_count=len(violations),
                        first_violation={key: value for key, value in first.items()
                            if value is None or isinstance(value, (str, bool, int, float, np.generic))})
                # This is the same post-hold measurement already owned by the
                # executor, even when its dedicated feedback writer has failed.
                report["final_robot_feedback"] = execution.get("continuation", {}).get("actual_feedback")
                # END no per-state geometry list or planning archive is copied.
            except Exception as error:
                caught = caught or error
                report.update(status="error", execution_report_error=f"{type(error).__name__}: {error}")
        # END diagnostic extraction cannot skip evaluator or camera cleanup.
        # BEGIN grade only complete evaluator capture, independently of execution.
        collection = report["evidence_collection"]["evaluator"]
        collection["completed"] = bool(collection["initialized"] and not collection["disabled"]
            and execution is not None and execution["independent_evidence_sink"]["completed"])
        assessment = None
        if evaluator is not None:
            try:
                assessment = evaluator.finalize(output, execution_report=execution,
                    collection_report=collection)
                report.update(physics_outcome=assessment["physics_outcome"], physics_success=assessment["physics_success"])
                report["evaluation_coverage"] = assessment["evidence_coverage"]
            except Exception as error:
                evidence_failure("evaluator", error, "finalize")
                report["evaluation_error"] = f"{type(error).__name__}: {error}"
                assessment = None
        if assessment is None:
            report.update(physics_outcome="incomplete_evidence", physics_success=None)
            # A missing/failed evaluator still leaves an explicit unknown grade.
            # Do not fabricate native samples or replace a physical first error.
            expected = None if execution is None else execution["executed_steps"]+1
            retained = collection["successful_callbacks"]
            missing = ([retained, expected-1]
                if expected is not None and retained < expected else None)
            assessment = {"physics_outcome": "incomplete_evidence", "physics_success": None,
                "evidence_collection": deepcopy(collection),
                "evidence_coverage": {"complete": False,
                    "expected_sample_count": expected, "retained_sample_count": retained,
                    "missing_normal_sample_index_ranges": [] if missing is None else [missing],
                    "reason": "Evaluator unavailable or finalization failed; retained artifacts are incomplete"}}
            report["evaluation_coverage"] = assessment["evidence_coverage"]
            try:
                (output/"evaluation-report.json").write_text(json.dumps(
                    _feedback_json(assessment), indent=2, allow_nan=False)+"\n")
            except Exception as error:
                evidence_failure("evaluator", error, "write_incomplete_report")
                report["evaluation_report_write_error"] = f"{type(error).__name__}: {error}"
        # END GT collection/finalization failures cannot change physical status.
        if videos is not None:
            # BEGIN capture the actual final hold state even after a guard rejection.
            # This is a video frame only: no missing evaluator/online-observer
            # samples are synthesized for the separate hold interval.
            if execution is not None and not report["evidence_collection"]["video"]["disabled"]:
                try:
                    videos.capture(execution["elapsed_sim_s"], final=True)
                    report["video_final_elapsed_s"] = execution["elapsed_sim_s"]
                except Exception as error:
                    evidence_failure("video", error, "final_capture")
                    report["video_final_capture_error"] = f"{type(error).__name__}: {error}"
            # END failed final capture must still release every writer/context.
            try:
                videos.close()
            except Exception as error:
                evidence_failure("video", error, "close")
                report["video_error"] = f"{type(error).__name__}: {error}"
        # BEGIN video gaps remain separate from the GT evaluator's verdict.
        video_state = report["evidence_collection"]["video"]
        video_state["completed"] = bool(video_state["initialized"] and not video_state["disabled"]
            and execution is not None and execution["independent_evidence_sink"]["completed"])
        report["evidence_complete"] = bool(report["evaluation_coverage"]["complete"]
            and all(not state["configured"] or state["completed"]
                    for state in report["evidence_collection"].values()))
        # END incomplete recording never invents frames or changes a hold.
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        # BEGIN even the final summary write must preserve a prior execution error.
        try:
            (output/"simulation-report.json").write_text(json.dumps(_feedback_json(report), indent=2, allow_nan=False)+"\n")
        except Exception as error:
            caught = caught or error
        # END all evidence sinks above have already had their cleanup opportunity.
    if caught is not None:
        raise caught
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # BEGIN require target positions explicitly, without the binary formatter.
    for name in ("manifest", "robot-fixture", "feedback", "trajectory", "finger-targets", "execution-limits", "evaluator-evidence", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--segment", required=True)
    parser.add_argument("--controller-part", required=True)
    parser.add_argument("--result-only-goal", action="append", type=_goal_identity,
                        dest="result_only_goal_ids", default=[],
                        help="Original collection:index to evaluate as a result without an online dependency; repeat for each explicit goal")
    # END require target positions explicitly, without the binary formatter.
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--view", action="append", dest="views")
    camera.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-physics-steps", type=int, default=30_000)
    parser.add_argument("--start-position-tolerance", type=float, default=2e-6)
    parser.add_argument("--start-velocity-tolerance", type=float, default=2e-6)
    outcome = run(parser.parse_args())
    print(json.dumps({key: outcome[key] for key in ("status", "trajectory_path", "physics_outcome", "videos")}, indent=2))
# END manifest-driven physical trial with independent evidence and video.
