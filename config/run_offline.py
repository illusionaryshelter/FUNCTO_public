"""Offline video runner for Point2Pose RGB-D frames + precomputed SAM masks.

Run from the Point2Pose REPOSITORY ROOT after installing its dependencies:
  python /path/to/run_offline.py --config /path/to/mujoco_offline.yaml \
      --sequence /path/to/my_sequence

Input contract documented in README.md. No models loaded by import; they load
when ModularPipeline is initialized.
"""
from __future__ import annotations
import argparse
import json
import time
import sys
from pathlib import Path

# Run this script with CWD set to the root of the installed Point2Pose repository.
sys.path.insert(0, str(Path.cwd()))

import cv2
import numpy as np
from omegaconf import OmegaConf

from rgbd_bridge import RGBDPacket, make_frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--sequence', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()

    cfg = OmegaConf.load(args.config)
    if cfg.pipeline.params.use_segmenter:
        raise ValueError('This runner needs use_segmenter: false and precomputed masks')
    n_obj = int(cfg.pipeline.params.max_num_obj)
    meta = json.loads((args.sequence / 'camera.json').read_text(encoding='utf-8'))
    K = np.asarray(meta['K'], dtype=np.float64)
    fps = float(meta.get('fps', 30))
    rgb_files = sorted((args.sequence / 'rgb').glob('*.png'))
    if not rgb_files:
        raise ValueError('No RGB PNG frames in sequence/rgb/')

    # Delay importing P2P until after config/data prechecks.
    from point2pose.pipeline.modular_pipeline import ModularPipeline
    pipeline = ModularPipeline(cfg)
    t0 = time.perf_counter()
    output_poses = []
    output_world_poses = []
    output_ts = []
    frame_ids = []
    timestamps = meta.get('timestamps_s', None)
    if timestamps is not None and len(timestamps) != len(rgb_files):
        raise ValueError('camera.json timestamps_s length must match RGB frame count')
    all_have_world_pose = True
    for count, rgb_path in enumerate(rgb_files):
        stem = rgb_path.stem
        depth_path = args.sequence / 'depth' / (stem + '.npy')
        masks_paths = [args.sequence / 'masks' / f'obj_{i}' / (stem + '.png')
                       for i in range(n_obj)]
        if not depth_path.exists() or not all(m.exists() for m in masks_paths):
            raise FileNotFoundError(f'Incomplete RGB-D/mask files at frame {stem}')
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f'Invalid RGB image: {rgb_path}')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = np.load(depth_path).astype(np.float32)
        masks = [cv2.imread(str(p), cv2.IMREAD_GRAYSCALE) for p in masks_paths]
        if any(m is None for m in masks):
            raise RuntimeError(f'Failed to read masks for frame: {stem}')
        # Empty/occluded masks are allowed after initialization, preserving ID.
        if count == 0 and any(not np.any(m) for m in masks):
            raise ValueError('First frame must contain nonempty masks for each object')
        # Optional per-frame intrinsics and camera-world transforms, especially
        # for moving MuJoCo wrist cameras or cameras with variable calibration.
        K_i_path = args.sequence / 'intrinsics' / (stem + '.npy')
        K_i = np.load(K_i_path) if K_i_path.exists() else K
        T_path = args.sequence / 'camera_poses' / (stem + '.npy')
        T_world_cam = np.load(T_path) if T_path.exists() else None
        if T_world_cam is not None and T_world_cam.shape != (4, 4):
            raise ValueError(f'Invalid camera pose shape: {T_path}')
        t_s = float(timestamps[count]) if timestamps is not None else count / fps
        pkt = RGBDPacket(rgb, depth_m, K_i, t_s,
                         T_world_cam_cv=T_world_cam)
        frame = make_frame(pkt, np.stack(masks), count, n_obj, device=args.device)
        poses = np.asarray(pipeline.step(frame), dtype=np.float64)
        if poses.shape != (n_obj, 4, 4):
            raise RuntimeError(f'Unexpected pose result shape: {poses.shape}')
        output_poses.append(poses.copy())
        output_ts.append(t_s)
        frame_ids.append(stem)
        if T_world_cam is not None:
            world_poses = np.stack([
                T_world_cam @ (pipeline.objects[obj_id].pose @
                               pipeline.objects[obj_id].init_pose)
                for obj_id in range(n_obj)
            ])
            output_world_poses.append(world_poses)
        else:
            all_have_world_pose = False
        if count % 50 == 0:
            print(f'Processed {count+1}/{len(rgb_files)} frames; {time.perf_counter()-t0:.1f}s')
    # This file uses the ORIGINAL VIDEO frame timestamps, avoiding the upstream
    # text logger's wall-time timestamps for offline processing.
    out_dir = Path(cfg.pipeline.params.pose_save_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(
        pose=np.stack(output_poses),
        frame_name=np.asarray(frame_ids),
        timestamp_s=np.asarray(output_ts, dtype=np.float64),
    )
    if all_have_world_pose:
        payload['T_world_obj_local'] = np.stack(output_world_poses)
    np.savez_compressed(out_dir / 'estimated_poses_by_frame.npz', **payload)
    print(f'Done: {len(rgb_files)} frames. Pose NPZ: {out_dir / "estimated_poses_by_frame.npz"}')


if __name__ == '__main__':
    main()
