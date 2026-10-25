"""Ground one portable skill in two explicit current RGB-D observations.

Models execute serially in the existing isolated workers. The same declared
object description is used once per view. This entry does not select prompts
or numerical parameters from a skill name, scene name or simulator truth.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback
from types import SimpleNamespace

import numpy as np
from PIL import Image

from manipulation.skill import load_grounding_skill, observe_interaction_reference
from manipulation.region_transfer import run_region_transfer
from manipulation.representation import generate_reference, trajectory_summary
from manipulation.reference_visualization import write_reference_html
from transfer.segmentation import prepare_selected_observation, store_detections
from transfer.features import write_json


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def ground_skill(skill_directory, scene_directory, output, *, instruction, relations=None):
    """Save every stage and return the native region/reference status.

    ``relations`` may provide explicitly sourced J/A records; it does not
    provide a target region. Missing A preserves an object-only reference.
    The current scene supplies actual robot feedback, calibrated RGB-D and an
    acquisition-state record. ``eef_twist`` is the measured world-frame TCP
    [linear, angular] velocity in m/s and rad/s. The tool reference is bound
    on source elapsed seconds; a planner rebinds its nominal operands when
    choosing another request clock. The scene's segmentation truth is never opened.
    """
    skill = load_grounding_skill(skill_directory)
    scene_directory, output = Path(scene_directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    feedback = json.loads((scene_directory / "robot_feedback.json").read_text())
    layout = json.loads((scene_directory / "layout.json").read_text())
    views = []
    for name in ("view_a", "view_b"):
        folder = scene_directory / name
        calibration = json.loads((folder / "calibration.json").read_text())
        with Image.open(folder / "rgb.png") as opened:
            rgb = opened.convert("RGB")
        views.append(dict(id=name, entity_id="current_instruction_entity", image=rgb,
                          depth_m=np.load(folder / "depth_m.npy", allow_pickle=False),
                          K=np.asarray(calibration["K"]),
                          T_world_camera_cv=np.asarray(calibration["T_world_camera_cv"]),
                          timestamp_s=calibration["timestamp_s"],
                          observation=None, feature=None, crop_xyxy=None,
                          missing_reasons=["selection_not_run"]))
    record = dict(skill_id=skill["metadata"]["skill_id"], skill=str(Path(skill_directory).resolve()),
                  scene=str(scene_directory.resolve()), instruction=instruction,
                  status="running", region_status="missing", reference_status="missing",
                  nominal_reference_status="missing", reference_binding_status="missing",
                  target_region_truth_used=False, executed=False, scene_layout=layout,
                  current_robot_feedback=feedback, stages=[])
    transfer, reference, tool_reference, stage = None, None, None, "validate_current_relations"
    try:
        # BEGIN validate already-declared coordinate identities before models.
        relations = relations or {}
        for field in ("J", "A"):
            if field in relations and not relations.get(field + "_source"):
                raise ValueError(f"Current {field} requires its explicit source")
        if "A" in relations:
            context = relations.get("A_context", {})
            for field in ("tool_id", "tool_frame", "reference_frame"):
                if field in context and context[field] != feedback.get(field):
                    raise ValueError(f"Current A_context.{field} conflicts with robot feedback")
        record["stages"].append(stage)
        stage = "select_current_entity"
        # END no new context fields or duplicate observations are required.
        # BEGIN persist each real detector result before the semantic request.
        # observe_interaction_reference calls the detector once per view, in
        # the supplied mapping order. Recording here preserves candidates even
        # if the following remote semantic request raises a transport error.
        from local_models.sam31 import detect
        from local_models.gemini import Gemini
        semantic_client = Gemini()
        record["semantic_model"] = {"model": semantic_client.model, "base_url": semantic_client.base_url}
        detection_views = iter(views)

        def recorded_detection(image, labels):
            view = next(detection_views)
            detected = detect(image, labels)
            store_detections(output / (view["id"] + "-detections"),
                             np.asarray(image), labels[0], detected)
            return detected

        observed = observe_interaction_reference(skill, instruction,
            {view["id"]: {"image": view["image"]} for view in views},
            detector=recorded_detection, client=semantic_client)
        # END evidence logging changes no detection or selection decision.
        selection_record = {"sam_queries": observed["sam_queries"],
                            "selection_scope": observed["selection_scope"], "views": {}}
        # BEGIN cross-view entity identity comes from the joint semantic record.
        association = observed["selection"]
        for key in ("same_entity", "same_entity_reason", "association_scope", "associated_views"):
            if key in association:
                selection_record[key] = association[key]
        unique_views = [name for name, value in association["views"].items()
                        if value["status"] == "selected" and len(value["regions"]) == 1]
        associated = len(unique_views) < 2 or association.get("same_entity") is True
        # END sharing a string ID alone never associates two observed instances.
        for view in views:
            name = view["id"]
            selected = observed["selection"]["views"][name]
            selection_record["views"][name] = {key: value for key, value in selected.items() if key != "regions"}
            if not associated:
                view["missing_reasons"] = ["cross_view_entity_identity_unconfirmed"]
                continue
            if selected["status"] != "selected" or len(selected["regions"]) != 1:
                view["missing_reasons"] = ["entity_selection:" + selected["status"]]
                continue
            prepared = prepare_selected_observation(np.asarray(view["image"]), view["depth_m"],
                view["K"], view["T_world_camera_cv"], selected["regions"][0])
            view.update(observation=prepared["observation"], crop_xyxy=prepared["crop_xyxy"],
                        missing_reasons=prepared["reasons"])
            if prepared["prepared"]:
                np.savez_compressed(output / (name + "-observation.npz"), **prepared["observation"])
                Image.fromarray(prepared["crop_rgb"]).save(output / (name + "-crop.png"))
        write_json(output / "selection.json", selection_record)
        record["selection"] = selection_record
        record["stages"].append(stage)
        stage = "extract_current_features"
        from local_models.marco import extract_image
        for view in views:
            if view["observation"] is not None:
                view["feature"] = extract_image(view["image"].crop(tuple(view["crop_xyxy"])))
                np.savez_compressed(output / (view["id"] + "-feature.npz"), feature=view["feature"])
        record["stages"].append(stage)
        stage = "dual_view_region"
        from transfer.dual_view.pipeline import prepare
        # BEGIN initialize only the explicitly declared region model.
        record["runtime"] = prepare(skill["source"]["region_geometry"])
        # END surface no longer compiles unused planar CUDA proposal kernels.
        same_state = ("Both RGB-D views and robot feedback were exported from one frozen scene "
                      f"at simulation time {layout['sim_time_s']}; scene preparation took zero physics steps.")
        # BEGIN declare the existing skill class at the shared region API.
        transfer = run_region_transfer(skill["source"], views,
            skill_kind=skill["representation"]["kind"], same_state_source=same_state)
        # END only manipulation changes the interaction-frame constructor.
        record.update(region_status=transfer["status"], region_reasons=transfer["reasons"])
        outputs = transfer.get("outputs")
        if outputs is not None:
            # BEGIN preserve model warnings in the formal entry and its HTML.
            record['region_geometry'] = skill['source']['region_geometry']
            record['region_warnings'] = list(outputs['result']['diagnostics'].get('quality_warnings', []))
            # END an accepted region may still have incomplete pose evidence.
            from transfer.experiment import store_prediction
            from transfer.multiview_experiment import save_surface
            save_surface(output, outputs["surface"])
            store_prediction(output / "region", outputs["result"], {})
            record["region_timing_seconds"] = outputs["stages_seconds"]
        if transfer.get("reference_frame") is not None:
            record["functional_frame"] = _plain(transfer["reference_frame"])
        record["stages"].append(stage)
        stage = "generate_reference"
        if skill["representation"]["kind"] == "grasp":
            if any(feedback[field] != skill["metadata"][field] for field in ("tool_id", "tool_frame")):
                raise ValueError("Grasp reference needs an explicit source/current tool-frame calibration")
        frame = transfer.get("reference_frame")
        # BEGIN retain nominal skill geometry and declare its observed entrance.
        # The saved tool pose and current A already determine the held body;
        # requiring another body observation would duplicate existing evidence.
        reference = generate_reference(skill["representation"],
            T_world_function=None if frame is None else frame["T_world_region"],
            J=relations.get("J"), A=relations.get("A"),
            T_world_eef=feedback.get("eef_pose"))
        arrays = {key: value for key, value in reference.items() if isinstance(value, np.ndarray)}
        # BEGIN nominal operands are named separately from the bound tool output.
        np.savez_compressed(output / "nominal-reference.npz", **arrays)
        # BEGIN persist the continuous reference consumed by reference_phase.
        # to_record retains parent compositions between saved pose samples;
        # boundary declarations keep the measured entrance separate.
        from manipulation.execution_binding import plain
        write_json(output / "nominal-reference-paths.json", plain({
            "frame_indices": reference["frame_indices"], "timestamps": reference["timestamps"],
            "events": reference["events"],
            "paths": {name: {"path": reference[name + "_path"],
                             "timestamps": reference[name + "_timestamps"],
                             "boundaries": reference[name + "_boundaries"]}
                      for name in ("object", "eef") if name + "_path" in reference}}))
        # END arrays and continuous expressions describe the same nominal motion.
        record.update(nominal_reference_status=reference["status"],
                      reference_status="unbound" if "eef_path" in reference else "missing",
                      reference_missing=dict(reference["missing"]),
                      nominal_reference_summaries=_plain(reference["summaries"]), relations=_plain(relations),
                      reference_boundary_checks=_plain(reference["boundary_checks"]),
                      reference_boundaries={name: _plain(reference[name + "_boundaries"])
                          for name in ("object", "eef") if name + "_boundaries" in reference},
                      nominal_reference_scope=reference["scope"],
                      nominal_reference_method=reference.get("method"))
        # BEGIN retain the exact declared rotation anchor alongside its generated poses.
        # This is reference-generation provenance in the functional frame, not
        # measured object orientation or an execution acceptance condition.
        if "rotation_transfer" in reference:
            record["reference_rotation_transfer"] = _plain(reference["rotation_transfer"])
        # END the report can identify the same rotation computation used by the path.
        # END the unchanged nominal expression remains available for planner rebinding.
        record["stages"].append(stage)
        if "eef_poses" in reference:
            # BEGIN complete grounding binds the current TCP pose and measured twist.
            # Source elapsed seconds are an explicit reference clock, not a
            # predicted execution duration. Use the planner's same binder.
            stage = "bind_reference"
            from manipulation.planning_problem import reference_phase
            from manipulation.timed_reference import BoundToolReference
            phase = reference_phase(reference, skill_id=skill["metadata"]["skill_id"],
                                    source="Current single-source dual-view grounding")
            source_clock = np.asarray(reference["eef_timestamps"])
            elapsed = source_clock-source_clock[0]
            tool_reference = BoundToolReference(SimpleNamespace(phases=(phase,)),
                [0., 1.], [0., 1.], float(elapsed[-1]), feedback["eef_pose"], feedback["eef_twist"])
            poses = tool_reference.sample_numpy(elapsed)
            np.savez_compressed(output / "reference.npz", eef_poses=poses,
                eef_timestamps=source_clock, elapsed_s=elapsed,
                initial_pose=tool_reference.initial_pose, initial_twist=tool_reference.initial_twist,
                source_frame_indices=reference["frame_indices"], source_timestamps=reference["timestamps"])
            write_json(output / "reference-paths.json", plain(dict(
                source_frame_indices=reference["frame_indices"], source_timestamps=reference["timestamps"],
                source_events=reference["events"], source_time_origin_s=float(source_clock[0]),
                tool_reference=tool_reference.to_record())))
            sample = tool_reference.sample_numpy(np.linspace(0., elapsed[-1], 21))
            np.save(output / "reference-samples.npy", sample, allow_pickle=False)
            summary = reference["summaries"]["eef"]
            record.update(reference_binding_status=tool_reference.diagnostics["status"],
                reference_status="reference" if tool_reference.diagnostics["entrance_twist_bound"] else "unbound",
                reference_binding=plain(tool_reference.diagnostics),
                reference_clock=dict(source_time_origin_s=float(source_clock[0]),
                    duration_s=float(elapsed[-1]), progress_u=[0., 1.], progress_s=[0., 1.],
                    meaning="source elapsed seconds; planner rebinds nominal operands to its request clock"),
                reference_summaries={"eef": plain(trajectory_summary(poses, source_clock,
                    approach_time=summary["approach_time"], interaction_time=summary["interaction_time"]))},
                reference_scope="current tool pose and twist bound to nominal interaction on source elapsed seconds",
                reference_method="BoundToolReference",
                reference_samples=dict(count=len(sample), file="reference-samples.npy"))
            record["stages"].append(stage)
            # END saved arrays, continuous records and the viewer use this same object.
        record["status"] = "completed"
    except Exception as error:
        record.update(status="failed", failed_stage=stage, error_type=type(error).__name__, error=str(error))
        (output / "error.txt").write_text(traceback.format_exc())
    finally:
        write_json(output / "report.json", record)
        write_reference_html(output, skill, views, transfer, reference, record,
                             tool_reference=tool_reference)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--relations", type=Path)
    args = parser.parse_args()
    relations = None if args.relations is None else json.loads(args.relations.read_text())
    record = ground_skill(args.skill, args.scene, args.output, instruction=args.instruction, relations=relations)
    print(json.dumps({key: record[key] for key in ("status", "region_status", "reference_status")}), flush=True)
    raise SystemExit(0 if record["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
