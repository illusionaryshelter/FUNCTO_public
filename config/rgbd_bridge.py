"""Minimal adapters for Point2Pose with externally supplied instance masks.

This module does not load SAM 3.1. Call make_frame on EACH RGB-D frame with its
aligned mask(s), and pass the Frame to ModularPipeline.step(frame).

MuJoCo: tested against its documented 3.5 camera conventions by source review;
RealSense: uses pyrealsense2 aligned depth and *measured* device depth scale.
Physical camera execution and rendering have not been run in this environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class RGBDPacket:
    rgb: np.ndarray             # H,W,3 uint8, RGB channel order
    depth_m: np.ndarray         # H,W float32, optical z depth, meters; 0=invalid
    K: np.ndarray               # 3x3 float64, same coordinate frame as rgb+depth
    # BEGIN unknown acquisition time stays missing; processing time is unrelated.
    timestamp_s: Optional[float]
    # END no frame-rate default creates an observation timestamp.
    T_world_cam_cv: Optional[np.ndarray] = None  # world<-camera (OpenCV axes)


def make_frame(packet: RGBDPacket, binary_masks, frame_id: int,
               max_num_obj: int, device: str = 'cuda'):
    """Adapter to point2pose.data_types.frame.Frame.

    `binary_masks`: [N,H,W] binary array, or [H,W] for N=1; stable object order.
    SAM probability maps must be thresholded by the caller BEFORE this function.
    Do not reorder identities when an object becomes occluded; use zero masks.
    """
    import torch
    from point2pose.data_types.frame import Frame

    rgb = np.asarray(packet.rgb)
    depth = np.asarray(packet.depth_m, dtype=np.float32)
    K = np.asarray(packet.K, dtype=np.float64)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError('rgb must be HxWx3 uint8 RGB')
    h, w = rgb.shape[:2]
    if depth.shape != (h, w):
        raise ValueError('depth_m must be aligned pixel-wise with rgb')
    if K.shape != (3, 3) or not np.all(np.isfinite(K)):
        raise ValueError('K must be a finite 3x3 matrix')
    if min(float(K[0, 0]), float(K[1, 1])) <= 0:
        raise ValueError('Camera fx/fy must be positive')
    # BEGIN match the actual Point2Pose pinhole/depth input convention.
    if (not np.allclose(K[2], [0., 0., 1.], atol=1e-12, rtol=0)
            or K[0, 1] != 0 or K[1, 0] != 0):
        raise ValueError('Point2Pose requires a zero-skew pinhole K on the observed pixel grid')
    if (isinstance(frame_id, (bool, np.bool_)) or not isinstance(frame_id, (int, np.integer))
            or frame_id < 0):
        raise ValueError('frame_id must be a nonnegative tracker frame index')
    if packet.timestamp_s is not None and (isinstance(packet.timestamp_s, (bool, np.bool_))
            or not np.isscalar(packet.timestamp_s) or not np.isfinite(packet.timestamp_s)):
        raise ValueError('timestamp_s must be finite original acquisition seconds or None')
    # END input identity and calibration are not inferred from array dimensions.
    masks = np.asarray(binary_masks)
    if masks.ndim == 2:
        masks = masks[None]
    if masks.shape != (max_num_obj, h, w):
        raise ValueError(f'masks must be [{max_num_obj},{h},{w}], got {masks.shape}')
    # BEGIN a positive probability is not an already selected binary instance.
    if not np.isin(masks, [0, 1, 255]).all():
        raise ValueError('Threshold masks to binary values before make_frame')
    # END retain the supplied foreground pixels without a new threshold.
    if np.any(np.sum(masks > 0, axis=0) > 1):
        raise ValueError('Overlapping instance masks: resolve IDs first')
    mask_t = torch.from_numpy(np.ascontiguousarray(masks > 0)).unsqueeze(1).to(device)
    depth = np.where(np.isfinite(depth) & (depth > 0), depth, 0).astype(np.float32)
    return Frame(id=int(frame_id), rgb=np.ascontiguousarray(rgb),
                 depth=np.ascontiguousarray(depth), mask=mask_t,
                 intrinsics=K, depth_factor=1.0,
                 # BEGIN missing source time remains absent in the native Frame.
                 timestamp=None if packet.timestamp_s is None else float(packet.timestamp_s))
                 # END processing wall time never substitutes for acquisition time.


class MuJoCoRGBD:
    """Perspective camera renderer (no simulator stepping is done inside capture)."""
    def __init__(self, model, data, camera_name: str,
                 width: int = 640, height: int = 480, *, K):
        import mujoco
        if not mujoco.__version__.startswith('3.5.'):
            raise RuntimeError(f'Expected MuJoCo 3.5.x, got {mujoco.__version__}')
        self.mj = mujoco
        self.model, self.data = model, data
        self.camera_name = camera_name
        self.width, self.height = int(width), int(height)
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if cam_id < 0:
            raise ValueError(f'Unknown MuJoCo camera: {camera_name}')
        self.cam_id = cam_id
        if int(model.cam_projection[cam_id]) != int(mujoco.mjtProjection.mjPROJ_PERSPECTIVE):
            raise ValueError('Point2Pose uses a pinhole model: orthographic camera unsupported')
        # BEGIN use the capture calibration, including its pixel-center convention.
        # Inferring W/2,H/2 here disagrees with this project's verified integer
        # pixel centers. The caller supplies the K measured for these renders;
        # this bridge does not maintain another renderer calibration formula.
        self.K = np.asarray(K, dtype=np.float64).copy()
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or min(self.K[0, 0], self.K[1, 1]) <= 0
                or not np.allclose(self.K[2], [0., 0., 1.], atol=1e-12, rtol=0)
                or self.K[0, 1] != 0 or self.K[1, 0] != 0):
            raise ValueError('K must be the observed zero-skew pinhole calibration')
        # END the exact supplied principal point is preserved in every packet.
        self.renderer = mujoco.Renderer(model, height=self.height, width=self.width)

    def capture(self) -> RGBDPacket:
        """Do not call mj_step between RGB and depth render calls."""
        self.renderer.update_scene(self.data, camera=self.camera_name)
        self.renderer.disable_depth_rendering()
        rgb = np.array(self.renderer.render(), copy=True)
        self.renderer.enable_depth_rendering()
        depth = np.array(self.renderer.render(), copy=True, dtype=np.float32)
        self.renderer.disable_depth_rendering()
        # MuJoCo: camera local +X right, +Y up, -Z forward.
        # OpenCV: +X right, +Y down, +Z forward.
        R_world_mj = np.array(self.data.cam_xmat[self.cam_id]).reshape(3, 3)
        flip = np.diag([1.0, -1.0, -1.0])
        T = np.eye(4)
        T[:3, :3] = R_world_mj @ flip
        T[:3, 3] = np.asarray(self.data.cam_xpos[self.cam_id])
        return RGBDPacket(rgb=np.ascontiguousarray(rgb),
                          depth_m=depth, K=self.K.copy(),
                          timestamp_s=float(self.data.time), T_world_cam_cv=T)

    def close(self):
        self.renderer.close()


class RealSenseRGBD:
    """Capture aligned RGB-D; uses actual depth scale, not hardcoded 1/1000."""
    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        if serial:
            cfg.enable_device(str(serial))
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.depth_scale_m = float(profile.get_device().first_depth_sensor().get_depth_scale())
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        ip = color_profile.get_intrinsics()
        self.K = np.array([[ip.fx, 0, ip.ppx], [0, ip.fy, ip.ppy], [0, 0, 1]], dtype=np.float64)
        self.distortion = (ip.model, list(ip.coeffs))

    def capture(self) -> RGBDPacket:
        import cv2
        frames = self.align.process(self.pipeline.wait_for_frames())
        cf = frames.get_color_frame()
        df = frames.get_depth_frame()
        if not cf or not df:
            raise RuntimeError('Missing aligned RealSense RGB/depth frame')
        bgr = np.asanyarray(cf.get_data())
        depth_raw = np.asanyarray(df.get_data())
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth_m = depth_raw.astype(np.float32) * self.depth_scale_m
        # Relative video timestamp: RealSense hardware timestamp is milliseconds.
        t_s = float(cf.get_timestamp()) * .001
        return RGBDPacket(rgb=rgb, depth_m=depth_m,
                          K=self.K.copy(), timestamp_s=t_s)

    def close(self):
        self.pipeline.stop()
