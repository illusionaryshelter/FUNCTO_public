"""Run an exported skill prefix in its recorded native scene and retain physics.

All scene, morphology, candidate and camera identities are explicit inputs.
Object truth is read only by setup and the independent physics evaluator.
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
from manipulation.execution import ExecutionLimits, FixedFingerTargets, TrajectoryExecutor
from recorded_simulation import restore_recorded_simulation, load_saved_trajectory
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
        except Exception:
            self.close()
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
        for entry in self._entries:
            if entry["writer"] is not None:
                entry["writer"].release()
            entry["camera"].close()
            if "record" in entry:
                np.save(entry["record"]["capture_times_path"], np.asarray(entry["times"]))


def run(args):
    """Restore and execute one skill; grade supported independent predicates."""
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
        "evaluation_mode": args.evaluation_mode, "physics_outcome": "not_evaluated",
        "physics_success": None, "videos": [], "video_requested": not args.no_video,
        "execution_report_path": str(output/"execution"/"execution-report.json"),
        "physical_trajectory_path": str(output/"execution"/"physical-trajectory.npz"),
        "evaluation_report_path": str(output/"evaluation-report.json"),
        "physics_evidence_path": str(output/"physics-evidence.npz"),
        "scope": "One actual native-physics skill prefix; no next skill or full-sequence claim",
        "planner_inputs_changed": False, "execution_tested": False}
    simulation = evaluator = videos = execution = None
    caught = None
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
        curve = load_saved_trajectory(args.trajectory, evaluation_mode=args.evaluation_mode,
                                      execution_limits=execution_limits)
        metadata = _read(args.trajectory.with_suffix(".json"))
        report["trajectory_classification"] = metadata["classification"]
        # BEGIN bind every physical trial to its identical recorded hand policy.
        # The saved-curve loader already checked classification and exact limits
        # before any simulator restoration or physical command can take place.
        report["planning_execution_limits_match"] = True
        report["planning_hand_contract"] = "identical saved planning/execution contract"
        # END bind every physical trial to its identical recorded hand policy.
        if {event["skill_id"] for event in curve.events} != {args.segment}:
            raise ValueError("The exported candidate must contain only the selected skill instance")
        evidence = _read(args.evaluator_evidence)
        source_segment = _one(evidence["segments"], "segment_id", segment["source_library_segment"])
        if source_segment["interaction"] != program["skill_kind"]:
            raise ValueError("The evaluator's explicit source identity must match the compiled skill kind")
        actor = {key: deepcopy(evidence["actor"][key]) for key in
                 ("entity_name", "eef_site_name", "contact_geom_groups")}
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
        report.update(restored_feedback_errors=differences, restore=restore_report,
            original_controller_configuration={"source": str(source_manifest_path),
                "part": args.controller_part, "configuration": deepcopy(arm_config),
                "policy_frequency_hz": environment["control_freq_hz"]},
            provenance={"input_manifest": str(args.manifest.resolve()),
                "robot_fixture": str(args.robot_fixture.resolve()), "feedback": str(args.feedback.resolve()),
                "finger_targets": str(args.finger_targets.resolve()),
                "program": str((root/segment["program_path"]).resolve()),
                "evaluator_evidence": str(args.evaluator_evidence.resolve())})
        catalog = _read(fixture["catalog_path"])
        reference_role = program["demo"]["roles"].get("reference")
        reference_name = None
        if reference_role is not None and reference_role["body_id"] is not None:
            reference_name = _one(catalog["bodies"], "id", reference_role["body_id"])["name"]
        evaluator = NativeSkillPhysicsEvaluator(simulation.native_model, catalog=catalog,
            actor=actor, target_body_name=source_segment["carrier_body_name"],
            finger_joint_names=fixture["finger_joint_names"], tcp_site=fixture["tcp_site"],
            program=program, reference_body_name=reference_name,
            binding_source={"path": str(args.evaluator_evidence.resolve()),
                "source_segment_id": segment["source_library_segment"],
                "fields_used": ["actor identity and contact-group names", "carrier_body_name", "reference body identity and declared source function requirements"],
                "original_contact_outcome_or_object_trajectory_used": False})
        evaluator.observe(simulation.native_model, simulation.native_data, {"elapsed": 0.})
        if not args.no_video:
            views = [_one(segment["snapshot"]["views"], "id", name) for name in args.views]
            videos = _TrialVideos(simulation, views=views, workspace_root=root,
                                  output=output, fps=environment["control_freq_hz"])
            report["videos"] = videos.records
            videos.capture(0.)

        def observe(model, data, record):
            evaluator.observe(model, data, record)
            if videos is not None:
                videos.capture(record["elapsed"])

        # BEGIN mandatory independent robot verification before native commands.
        validator = build_robot_validator(fixture,
            reference_frame=feedback["reference_frame"],
            T_reference_base=np.asarray(feedback["base_pose"], dtype=float),
            tool_frame=feedback["tool_link"], T_frame_tcp=tool_transform)
        executor = TrajectoryExecutor(simulation, geometry_checker=validator)
        report["status"] = "executing"
        execution = executor.execute(curve, evaluation_mode=args.evaluation_mode,
            finger_targets=finger_targets, initial_feedback=initial_feedback,
            execution_limits=execution_limits,
            output=output/"execution", step_callback=observe,
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
        if videos is not None:
            videos.capture(execution["elapsed_sim_s"], final=True)
        report["status"] = "completed"
    except Exception as error:
        caught = error
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
    finally:
        if execution is None and Path(report["execution_report_path"]).is_file():
            execution = _read(report["execution_report_path"])
            report["execution_tested"] = execution["executed_steps"] > 0
            # BEGIN expose failed actual feedback as well as completed barriers.
            report.update(robot_feedback_path=execution.get("robot_feedback_path"),
                          barrier_feedback_path=execution.get("barrier_feedback_path"))
            # END expose failed actual feedback as well as completed barriers.
        if evaluator is not None:
            try:
                assessment = evaluator.finalize(output, execution_report=execution)
                report.update(physics_outcome=assessment["physics_outcome"], physics_success=assessment["physics_success"])
            except Exception as error:
                caught = caught or error
                report.update(status="error", evaluation_error=f"{type(error).__name__}: {error}")
        if videos is not None:
            try:
                videos.close()
            except Exception as error:
                caught = caught or error
                report.update(status="error", video_error=f"{type(error).__name__}: {error}")
        report.update(elapsed_seconds=time.monotonic()-started,
                      peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
        (output/"simulation-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
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
    # END require target positions explicitly, without the binary formatter.
    parser.add_argument("--evaluation-mode", required=True,
                        choices=("accepted_plan", "diagnostic_rejected_candidate"))
    camera = parser.add_mutually_exclusive_group(required=True)
    camera.add_argument("--view", action="append", dest="views")
    camera.add_argument("--no-video", action="store_true")
    parser.add_argument("--max-physics-steps", type=int, default=30_000)
    parser.add_argument("--start-position-tolerance", type=float, default=2e-6)
    parser.add_argument("--start-velocity-tolerance", type=float, default=2e-6)
    outcome = run(parser.parse_args())
    print(json.dumps({key: outcome[key] for key in ("status", "trajectory_path", "physics_outcome", "videos")}, indent=2))
# END manifest-driven physical trial with independent evidence and video.
