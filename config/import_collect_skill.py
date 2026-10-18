"""Read existing collect evidence into the formal FUNCTO skill representation.

This adapter does not segment demonstrations, run a semantic model, infer new
contact regions or edit collect. Its source region, image time, approach start
and interaction time are explicit inputs from the recorded evidence. General
coordinate arithmetic and portable file writing belong to manipulation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from manipulation.observations import backproject
from manipulation.regions import source_region_frame
from manipulation.representation import build_representation
from manipulation.skill import save_grounding_skill
from transfer.segmentation import _crop


def _json(path):
    return json.loads(Path(path).read_text())


def extract_recorded_skill(prepared, exported_segment, *, region_id, source_frame,
                           approach_frame, interaction_time, semantic,
                           region_geometry, frame_template=None):
    """Extract a selected existing region and its synchronized fixed-body track.

    ``exported_segment`` is produced by the unchanged collect.export_functo.
    Its command/physical events are reused without reinterpretation. Explicit
    semantic input names the observed object, not a skill-dependent SAM recipe.
    The offline owner mask is recorded simulator evidence, never presented as
    a SAM prediction. Current-scene inference uses actual selected detections.
    """
    # BEGIN geometry is a required region declaration, independent of grasp/place.
    if region_geometry not in ('planar', 'surface'):
        raise ValueError('region_geometry must be planar or surface')
    # END explicit caller choice.
    prepared, exported_segment = Path(prepared).resolve(), Path(exported_segment).resolve()
    manifest = _json(prepared / "manifest.json")
    episode = Path(manifest["source_episode"])
    evidence = _json(exported_segment / "evidence.json")
    # BEGIN exported geometry and motion must originate from this recording.
    if (Path(evidence["source"]["preprocessing"]).resolve() != prepared
            or Path(evidence["source"]["episode"]).resolve() != episode.resolve()):
        raise ValueError("Exported segment and prepared observations belong to different recordings")
    # END equal frame numbers alone do not establish recording identity.
    segment_id = evidence["segment_id"]
    annotation = next(row for row in _json(prepared / "annotations.json")["segments"]
                      if row["segment_id"] == segment_id)
    regions = next(row for row in _json(prepared / "regions.json")["segments"]
                   if row["segment_id"] == segment_id)
    region = next(row for row in regions["regions"] if row["region_id"] == region_id)
    if region["status"] != "resolved":
        raise ValueError("The supplied source region is not available in collect")
    kind = evidence["skill_kind"]
    owner_role = "carrier" if kind == "grasp" else "reference"
    owner_id = region["body_ref"]
    if region["body_id"] != evidence["roles"][owner_role]["body_id"]:
        raise ValueError("Region must belong to the declared grasp or operation-reference body")
    with np.load(exported_segment / "trajectory.npz", allow_pickle=False) as archive:
        tracks = {key: archive[key] for key in archive.files}
    frames, times = tracks["frame_indices"], tracks["timestamps"]
    image_rows, approach_rows = np.flatnonzero(frames == source_frame), np.flatnonzero(frames == approach_frame)
    if len(image_rows) != 1 or len(approach_rows) != 1:
        raise ValueError("Source image and approach frame must belong to this segment")
    local_row = int(image_rows[0])
    with h5py.File(prepared / "trajectories.h5", "r") as archive:
        global_rows = np.flatnonzero(archive["frame_index"][:] == source_frame)
        if len(global_rows) != 1:
            raise ValueError("Source frame does not identify one original observation")
        image_row = int(global_rows[0])
        body_at_image = archive["T_world_body"][image_row, region["body_id"]]
        if not np.isclose(archive["sim_time"][image_row], times[local_row], atol=1e-9, rtol=0):
            raise ValueError("Exported trajectory and original source image time disagree")
    with h5py.File(episode / "episode.h5", "r") as archive:
        rgb, depth = archive["frames/rgb"][image_row], archive["frames/depth_m"][image_row]
    with h5py.File(prepared / "masks.h5", "r") as archive:
        if not np.array_equal(archive[regions["h5_group"] + "/frame_indices"][:], frames):
            raise ValueError("Recorded mask and motion frame identities disagree")
        group = archive[region["h5_group"]]
        entity_mask = group["owner_mask"][local_row].astype(bool)
        region_mask = group["local_mask"][local_row].astype(bool)
    calibration = _json(episode / "camera_calibration.json")
    K, camera = np.asarray(calibration["K"]), np.asarray(calibration["T_world_camera_cv"])
    valid = np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero((entity_mask | region_mask) & valid)
    pixels = np.column_stack((x, y)).astype(np.int32)
    camera_points = backproject(depth, K, pixels_uv=pixels)
    xyz = camera_points @ camera[:3, :3].T + camera[:3, 3]
    entity_member, region_member = entity_mask[y, x], region_mask[y, x]
    if not region_member.any():
        raise ValueError("The given source region has no measured points at this image time")
    observation = dict(xyz=xyz, pixels=pixels, object_mask=entity_mask,
                       entity_member=entity_member, region_mask=region_member,
                       evidence_only=region_member & ~entity_member, K=K, T_world_camera_cv=camera)
    supplied_frame = None
    if frame_template is not None:
        template = _json(frame_template)
        if template["source_frame"] != source_frame:
            raise ValueError("Explicit compiled source frame uses another image time")
        # BEGIN identify the exact fixed body chart of a previously given frame.
        if Path(template["source_episode"]).resolve() != episode.resolve():
            raise ValueError("Compiled source frame belongs to another recording")
        with np.load(Path(frame_template).parent / "observations.npz", allow_pickle=False) as prior:
            if (not np.allclose(prior["T_camera_owner"], np.linalg.inv(camera) @ body_at_image,
                                atol=1e-7, rtol=0)
                    or not np.array_equal(prior["K"], K)
                    or not np.array_equal(prior["depth"], depth)):
                raise ValueError("Compiled source frame uses a different body or RGB-D calibration")
        with Image.open(Path(frame_template).parent / "rgb.png") as prior_image:
            if not np.array_equal(np.asarray(prior_image.convert("RGB")), rgb):
                raise ValueError("Compiled source frame belongs to another source image")
        # END source L cannot be borrowed from a different same-frame object.
        supplied_frame = body_at_image @ np.asarray(template["compiled_source_frame"])
    geometry = source_region_frame(T_world_body=body_at_image, T_world_camera=camera,
                                   points_world=xyz[region_member], T_world_function=supplied_frame)
    L = geometry["L_body_region"]
    provenance = dict(source_preprocessing=str(prepared), source_episode=str(episode),
                      segment_id=segment_id, source_kind=manifest["source_kind"],
                      gt_scope="Explicit offline simulation body motion and supplied visible region only",
                      region_id=region_id, region_source="regions.json and masks.h5, unchanged",
                      entity_mask_source="Recorded offline visible rigid-owner mask; not a SAM prediction",
                      approach_source=f"Explicit original frame {approach_frame}",
                      interaction_source="Explicit recorded event/annotation time supplied to this adapter",
                      source_frame_template=None if frame_template is None else str(Path(frame_template).resolve()))
    physical_events = evidence["physical_events"]
    releases = [event["timestamp"] for event in physical_events
                if event["kind"] == "release" and event["in_segment"]]
    if len(releases) > 1:
        raise ValueError("Select a segment with one declared continuous holding interval")
    world_reference = None
    if "T_reference_carrier" in tracks:
        world_reference = tracks["T_world_carrier"] @ np.linalg.inv(tracks["T_reference_carrier"])
    representation = build_representation(
        timestamps=times, frame_indices=frames, kind=kind,
        T_world_carrier=tracks["T_world_carrier"], T_world_reference=world_reference,
        T_world_eef=tracks["T_world_eef"], L_G=L if kind == "grasp" else None,
        L_R=L if kind == "manipulation" else None,
        approach_time=float(times[approach_rows[0]]), interaction_time=interaction_time,
        release_time=releases[0] if releases else None,
        events=physical_events + evidence["gripper_events"], provenance=provenance)
    roles = annotation["role_bindings"]
    metadata = dict(skill_id=annotation["semantic_label"], owner_role=owner_role,
                    semantic=semantic, roles=roles, tool_id=evidence["tool_id"],
                    tool_frame=evidence["tool_frame"], provenance=provenance,
                    task={"semantic_effect_candidates": evidence["semantic_effect_candidates"],
                          "missing": {"numeric_conditions": "Recorded descriptions do not define current task tolerances"}})
    # BEGIN persist only the geometry declaration; all recorded arrays stay exact.
    source = dict(region_geometry=region_geometry,
                  image=Image.fromarray(rgb), depth_m=depth, observation=observation,
                  crop_xyxy=_crop(entity_mask | region_mask, .15),
                  timestamp_s=float(times[local_row]), frame_index=int(source_frame),
                  owner_id=owner_id, T_world_body_at_image=body_at_image,
                  L_body_region=L, region_image_mask=region_mask, provenance=provenance,
                  frame_construction=geometry["details"])
    # END recorded source geometry metadata.
    return dict(metadata=metadata, source=source, representation=representation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--exported-segment", type=Path, required=True)
    parser.add_argument("--region", required=True)
    # BEGIN require callers to choose the geometric model for this source region.
    parser.add_argument('--region-geometry', choices=('planar', 'surface'), required=True)
    # END explicit CLI declaration.
    parser.add_argument("--source-frame", type=int, required=True)
    parser.add_argument("--approach-frame", type=int, required=True)
    parser.add_argument("--interaction-time", type=float, required=True)
    parser.add_argument("--semantic", type=Path, required=True)
    parser.add_argument("--frame-template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extract-feature", action="store_true")
    args = parser.parse_args()
    skill = extract_recorded_skill(args.prepared, args.exported_segment, region_id=args.region,
                                  source_frame=args.source_frame, approach_frame=args.approach_frame,
                                  interaction_time=args.interaction_time, semantic=_json(args.semantic),
                                  # BEGIN propagate the CLI region declaration.
                                  region_geometry=args.region_geometry, frame_template=args.frame_template)
                                  # END declared model propagation.
    if args.extract_feature:
        from local_models.marco import extract_image, FEATURE_SHAPE
        source = skill["source"]
        source["feature"] = extract_image(source["image"].crop(tuple(source["crop_xyxy"])))
        source["feature_model"] = dict(name="MARCO", shape=list(FEATURE_SHAPE),
                                       preprocessing="Existing native MARCO resize/pad; FP32 spatial map")
    save_grounding_skill(args.output, **skill)
    print(json.dumps({"output": str(args.output), "source_feature": args.extract_feature,
                      "skill_id": skill["metadata"]["skill_id"],
                      "frames": len(skill["representation"]["timestamps"])}), flush=True)


if __name__ == "__main__":
    main()
