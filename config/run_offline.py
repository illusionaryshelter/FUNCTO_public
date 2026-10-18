"""Record Point2Pose estimates with fixed body frames and original evidence.

Run from the existing Point2Pose repository root with precomputed masks. This
entry only adapts inputs and records the native frontend's returned evidence;
it does not change tracking, registration, graph optimization or thresholds.
"""

# BEGIN explicit offline observation identity, calibration and tracking evidence.
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np
from omegaconf import OmegaConf

# The installed Point2Pose source remains read-only. Keep this entry usable
# as a script from that repository, while importing FUNCTO's ordinary modules.
sys.path.insert(0, str(Path.cwd()))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.rgbd_bridge import RGBDPacket, make_frame
from manipulation.grasp import validate_timeline
from manipulation.motion import _pose_array, _poses_array
from manipulation.observations import _intrinsics


def _plain(value):
    """JSON records keep unavailable numeric evidence as null, never NaN."""
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_plain(value), ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def sequence_inputs(directory):
    """Read explicit instance IDs and original frame/time mappings before models."""
    directory = Path(directory)
    meta = json.loads((directory / "camera.json").read_text())
    objects = meta["object_ids"]
    if (not isinstance(objects, list) or not objects
            or any(not isinstance(value, str) or not value.strip() for value in objects)
            or len(set(objects)) != len(objects)):
        raise ValueError("object_ids must uniquely name the fixed rigid instances in mask order")
    paths = sorted((directory / "rgb").glob("*.png"))
    if not paths:
        raise ValueError("No RGB PNG frames in sequence/rgb/")
    raw_frames = meta["frame_indices"]
    frames = np.asarray(raw_frames)
    if (any(isinstance(value, (bool, np.bool_)) for value in raw_frames)
            or frames.shape != (len(paths),) or frames.dtype.kind not in "iu"
            or np.any(frames < 0) or np.any(frames[1:] <= frames[:-1])):
        raise ValueError("frame_indices must be ordered original integer IDs aligned with sorted RGB files")
    times = meta.get("timestamps_s")
    if times is not None:
        _, times = validate_timeline(frames, times)
    K = _intrinsics(meta["K"])
    camera = meta.get("T_world_camera_cv")
    if camera is not None:
        camera = _pose_array(camera, "T_world_camera_cv")
    rows = []
    for index, rgb in enumerate(paths):
        stem = rgb.stem
        depth = directory / "depth" / (stem + ".npy")
        masks = [directory / "masks" / f"obj_{obj}" / (stem + ".png") for obj in range(len(objects))]
        if not depth.is_file() or any(not mask.is_file() for mask in masks):
            raise FileNotFoundError(f"Incomplete RGB-D/mask files at original frame {frames[index]}")
        intrinsic_file = directory / "intrinsics" / (stem + ".npy")
        pose_file = directory / "camera_poses" / (stem + ".npy")
        intrinsic = _intrinsics(np.load(intrinsic_file, allow_pickle=False)) if intrinsic_file.exists() else K.copy()
        transform = (_pose_array(np.load(pose_file, allow_pickle=False), "per-frame T_world_camera_cv")
                     if pose_file.exists() else None if camera is None else camera.copy())
        rows.append(dict(rgb=rgb, depth=depth, masks=masks, frame_name=stem,
                         frame_index=int(frames[index]), tracker_frame_id=index,
                         timestamp_s=None if times is None else float(times[index]),
                         K=intrinsic, T_world_camera_cv=transform))
    return meta, rows


def tracking_evidence(result, objects, *, first, frame_reg_mode, residual_threshold_m):
    """Describe native frontend facts; no Boolean claims a verified fresh pose.

    In f2f, failure of the existing residual condition retains the old pose.
    In f2m, a high-residual candidate can still be returned with lost=False.
    The statuses below report those branches, never a new acceptance policy.
    All original diagnostics remain available independently of the label.
    """
    records = []
    for obj in objects:
        record = dict(object_index=int(obj.id), lost=bool(obj.lost), lost_streak=int(obj.lost_streak),
                      frame_reg_mode=frame_reg_mode, residual_threshold_m=float(residual_threshold_m))
        if first:
            record.update(status="initialized", reason="Fixed body frame anchored at this first observation")
        elif result is None or obj.id not in result.obj_poses:
            record.update(status="missing", reason="No current frontend object result; matrix is not new evidence")
        else:
            stats = result.reg_stats.get(obj.id, {})
            residuals = np.asarray(stats.get("residuals", []))
            mean = result.mean_residuals.get(obj.id)
            # Native cluster registration writes -1 distances when it returns
            # the old estimate after finding no candidate. Preserve that raw
            # sentinel, but never count it as an observed distance residual.
            observed = (residuals.size > 0 and np.isfinite(residuals).all()
                        and np.all(residuals >= 0))
            condition = mean is not None and np.isfinite(mean) and 0 < mean < residual_threshold_m
            rejected = bool(stats.get("pose_jump_guard_info", {}).get("rejected", False))
            record.update(registration_stats=_plain(stats), valid_indices=_plain(result.valid_indices.get(obj.id)),
                          mean_residual_m=_plain(mean), registration_residuals_observed=bool(observed),
                          frontend_residual_condition_met=bool(condition), jump_guard_rejected=rejected,
                          frontend_pose=_plain(result.obj_poses[obj.id]))
            if rejected or not observed or (frame_reg_mode == "f2f" and not condition):
                record.update(status="stale", reason="Native frontend retained/rejected the pose or had no current registration residuals")
            elif obj.lost or not condition:
                record.update(status="warning", reason="Current registration returned, but native lost/residual evidence remains unresolved")
            else:
                record.update(status="frontend_condition_met",
                              reason="Current registration residuals exist, jump guard did not reject, and the native residual condition holds")
        records.append(record)
    return records


def run_offline(config_path, sequence, output, *, device="cuda"):
    """Run the unmodified native pipeline and preserve every processed state.

    Point2Pose's internal frame IDs remain contiguous. The output separately
    maps them to original frame IDs and acquisition seconds. Missing camera
    pose/time stays missing, so the result cannot masquerade as a timed world
    trajectory suitable for compiling a complete skill bundle.
    """
    meta, rows = sequence_inputs(sequence)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    cfg = OmegaConf.load(config_path)
    # External masks and the actual instance count are native input choices.
    # Keep every tracker/register/graph/SDF parameter from the supplied profile.
    cfg.pipeline.params.use_segmenter = False
    cfg.pipeline.params.max_num_obj = len(meta["object_ids"])
    cfg.pipeline.params.pose_save_path = str(output.resolve() / "native-wall-time-poses")
    OmegaConf.save(cfg, output / "resolved_config.yaml")
    _write_json(output / "input_camera.json", meta)
    report = dict(status="running", config=str(Path(config_path).resolve()),
                  sequence=str(Path(sequence).resolve()), object_ids=meta["object_ids"], frames=[],
                  scope="Native RGB-D estimates and recorded diagnostics; not ground truth or execution evidence",
                  body_frame_definition="Each B is the frozen obj.init_pose chart in the initial camera map",
                  pose_convention="T_camera_B = obj.pose @ frozen T_initial_camera_B; T_world_B = T_world_camera_cv @ T_camera_B",
                  missing={} if meta.get("timestamps_s") is not None else {"timestamps_s": "Original acquisition times were not supplied"})
    observed_rows, camera_poses, world_poses, raw_poses = [], [], [], []
    fixed_body = None
    pipeline = None
    original_step = None
    current = {"result": None}
    active_row = None
    try:
        from point2pose.pipeline.modular_pipeline import ModularPipeline
        pipeline = ModularPipeline(cfg)
        original_step = pipeline.frontend.step

        def record_frontend(*args, **kwargs):
            result = original_step(*args, **kwargs)
            current["result"] = result
            return result

        # Observe the existing return exactly once, without modifying it or the
        # pipeline's consumers. Initialization legitimately has no registration.
        pipeline.frontend.step = record_frontend
        for row in rows:
            active_row = row
            bgr = cv2.imread(str(row["rgb"]), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Invalid RGB image: {row['rgb']}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            depth = np.load(row["depth"], allow_pickle=False)
            masks = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in row["masks"]]
            if any(mask is None for mask in masks):
                raise ValueError(f"Invalid instance mask at {row['frame_name']}")
            if row["tracker_frame_id"] == 0 and any(not mask.any() for mask in masks):
                raise ValueError("First frame must contain each declared rigid instance")
            packet = RGBDPacket(rgb, depth, row["K"], row["timestamp_s"], row["T_world_camera_cv"])
            frame = make_frame(packet, np.stack(masks), row["tracker_frame_id"], len(masks), device=device)
            current["result"] = None
            returned = _poses_array(pipeline.step(frame), "native Point2Pose return")
            if (returned.shape[0] != len(masks)
                    or [obj.id for obj in pipeline.objects] != list(range(len(masks)))):
                raise ValueError("Point2Pose object IDs changed the declared fixed instance order")
            result = current["result"]
            if result is not None and result.frame_id != row["tracker_frame_id"]:
                raise ValueError("Frontend diagnostics do not belong to this current tracker frame")
            initial = _poses_array([obj.init_pose for obj in pipeline.objects], "native initial body frames")
            if fixed_body is None:
                fixed_body = initial.copy()
            elif not np.array_equal(initial, fixed_body):
                raise ValueError("Point2Pose changed its fixed initial body frame within this sequence")
            motion = _poses_array([obj.pose for obj in pipeline.objects], "native object map motion")
            body = motion @ fixed_body
            world = (np.full_like(body, np.nan) if row["T_world_camera_cv"] is None
                     else row["T_world_camera_cv"] @ body)
            evidence = tracking_evidence(result, pipeline.objects, first=row["tracker_frame_id"] == 0,
                frame_reg_mode=pipeline.frontend.frame_reg_mode,
                residual_threshold_m=pipeline.frontend.reg_residual_thres)
            observed_rows.append(row)
            camera_poses.append(body)
            world_poses.append(world)
            raw_poses.append(motion)
            report["frames"].append(dict(frame_index=row["frame_index"], frame_name=row["frame_name"],
                tracker_frame_id=row["tracker_frame_id"], timestamp_s=row["timestamp_s"],
                camera_pose_available=row["T_world_camera_cv"] is not None, objects=evidence))
            print(f"Recorded original frame {row['frame_index']} as tracker frame {row['tracker_frame_id']}", flush=True)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      failed_frame=None if active_row is None else active_row["frame_index"])
        (output / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if pipeline is not None and original_step is not None:
            pipeline.frontend.step = original_step
        if fixed_body is not None:
            count, objects = len(observed_rows), len(meta["object_ids"])
            shape = (count, objects, 4, 4)
            payload = dict(object_ids=np.asarray(meta["object_ids"]),
                frame_indices=np.asarray([row["frame_index"] for row in observed_rows], dtype=np.int64),
                tracker_frame_ids=np.arange(count, dtype=np.int64),
                frame_names=np.asarray([row["frame_name"] for row in observed_rows]),
                timestamp_s=np.asarray([np.nan if row["timestamp_s"] is None else row["timestamp_s"] for row in observed_rows]),
                timestamp_available=np.asarray([row["timestamp_s"] is not None for row in observed_rows], dtype=bool),
                K=np.asarray([row["K"] for row in observed_rows]).reshape(count, 3, 3),
                T_world_camera_cv=np.asarray([np.full((4, 4), np.nan) if row["T_world_camera_cv"] is None
                                               else row["T_world_camera_cv"] for row in observed_rows]).reshape(count, 4, 4),
                world_camera_available=np.asarray([row["T_world_camera_cv"] is not None for row in observed_rows], dtype=bool),
                T_initial_camera_B=fixed_body, T_camera_initial_map=np.asarray(raw_poses).reshape(shape),
                T_camera_B=np.asarray(camera_poses).reshape(shape), T_world_B=np.asarray(world_poses).reshape(shape))
            np.savez_compressed(output / "estimated_poses_by_frame.npz", **payload)
            report["T_initial_camera_B"] = fixed_body
        if any(row["T_world_camera_cv"] is None for row in rows):
            report["missing"]["T_world_camera_cv"] = "One or more original camera-world poses were not supplied"
        report["processed_frames"] = len(observed_rows)
        _write_json(output / "tracking_evidence.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    run_offline(args.config, args.sequence, args.output, device=args.device)


if __name__ == "__main__":
    main()
# END this entry records the native tracker without changing its algorithm.
