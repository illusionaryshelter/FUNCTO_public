"""Offline RGB-D material-point tracking, isolated from FUNCTO's other models.

Sampling chooses measured surface points from each initial rigid-part mask.
TAPIP3D estimates their world-coordinate trajectories and visibility; neither
visibility nor a numerically finite trajectory certifies position accuracy.
Rigid fitting and task-specific semantic anchors belong to the caller.
"""

# BEGIN calibrated TAPIP3D worker: no torch/model import in the parent process.
from pathlib import Path
import sys

import numpy as np
from PIL import Image


QUERY_COUNT = 64
IMAGE_SIZE = (384, 512)
VISIBILITY_THRESHOLD = .9


def camera_extrinsics(extrinsics, count, *, stationary_camera=False):
    """Require known camera-from-world poses or an explicit stationary camera.

    A moving reference object is not a camera calibration. Its estimated
    motion is removed separately by the downstream observation adapter.
    """
    if not isinstance(stationary_camera, (bool, np.bool_)):
        raise ValueError("stationary_camera must be an explicit boolean")
    if extrinsics is None:
        if not stationary_camera:
            raise ValueError("Supply camera_from_world extrinsics or declare stationary_camera=True")
        return np.repeat(np.eye(4)[None], count, axis=0)
    if stationary_camera:
        raise ValueError("Use camera_from_world extrinsics or stationary_camera, not both")
    matrices = np.asarray(extrinsics, dtype=float)
    if (matrices.shape != (count, 4, 4) or not np.isfinite(matrices).all()
            or not np.allclose(matrices[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0)):
        raise ValueError("extrinsics must be finite camera_from_world matrices [T,4,4]")
    rotations = matrices[:, :3, :3]
    if (not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
        raise ValueError("camera_from_world extrinsics must be rigid SE(3)")
    return matrices.copy()


def _read_frame(rgb_path, depth_path, depth_scale, shape):
    """Decode one aligned observation; never fill absent measured depth."""
    with Image.open(rgb_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if Path(depth_path).suffix.lower() == ".npy":
        raw = np.load(depth_path, allow_pickle=False)
    else:
        with Image.open(depth_path) as image:
            raw = np.asarray(image)
    if rgb.shape != (*shape, 3) or raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError("RGB, aligned depth and masks must share the calibrated dimensions")
    depth = np.asarray(raw, dtype=np.float32) * depth_scale
    depth[~np.isfinite(depth) | (depth <= 0)] = 0
    if not np.any(depth > 0):
        raise ValueError("Every RGB-D frame requires valid measured depth")
    return rgb, depth


def sample_queries(depth, K, masks, camera_from_world0):
    """Select up to 64 unique, spread surface pixels per independent body.

    The two-pixel boundary margin matches the validated RGB-D experiment.
    Small parts may supply fewer unique pixels; fewer than six is an explicit
    failure. A farthest-point pixel selection avoids duplicate queries and
    concentrated texture corners, without claiming independent point noise.
    """
    import cv2

    pixels, owners = [], []
    for owner, mask in enumerate(masks):
        interior = cv2.erode(mask.astype(np.uint8), np.ones((5, 5), np.uint8),
                             borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
        y, x = np.nonzero(interior & np.isfinite(depth) & (depth > 0))
        candidates = np.column_stack((x, y)).astype(float)
        if len(candidates) < 6:
            raise ValueError(f"TAPIP3D body {owner} needs six valid surface queries inside its mask")
        count = min(QUERY_COUNT, len(candidates))
        picks = [int(np.argmax(np.linalg.norm(candidates - candidates.mean(0), axis=1)))]
        distances = np.full(len(candidates), np.inf)
        for _ in range(count - 1):
            distances = np.minimum(distances, np.sum((candidates - candidates[picks[-1]])**2, axis=1))
            picks.append(int(np.argmax(distances)))
        pixels.append(candidates[picks])
        owners.extend([owner] * count)
    uv = np.concatenate(pixels)
    z = depth[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    camera = np.column_stack((uv, np.ones(len(uv)))) @ np.linalg.inv(K).T * z[:, None]
    world_from_camera = np.linalg.inv(camera_from_world0)
    world = camera @ world_from_camera[:3, :3].T + world_from_camera[:3, 3]
    return np.column_stack((np.zeros(len(world)), world)).astype(np.float32), np.asarray(owners, dtype=np.int32), uv


def _outputs(output, count, queries, K, extrinsics, shape):
    """Check fixed query identities and geometry without correcting predictions."""
    n = len(queries)
    expected = {"coords": (count, n, 3), "visibs": (count, n),
                "query_points": (n, 4), "pixels": (count, n, 2)}
    result = {}
    for key, size in expected.items():
        if key not in output:
            raise ValueError(f"TAPIP3D worker omitted {key}")
        array = np.asarray(output[key])
        if array.shape != size or array.dtype.kind not in "bf":
            raise ValueError(f"TAPIP3D worker has invalid {key} shape or dtype")
        result[key] = array.copy()
    if result["visibs"].dtype != np.bool_:
        raise ValueError("TAPIP3D visibs must be boolean visibility, not an accuracy score")
    if (not np.isfinite(result["coords"]).all()
            or not np.array_equal(result["query_points"], queries)):
        raise ValueError("TAPIP3D must preserve finite trajectories and supplied query identities")
    coords = result["coords"]
    camera = np.einsum("tij,tnj->tni", extrinsics[:, :3, :3], coords) + extrinsics[:, None, :3, 3]
    uvz = np.einsum("ij,tnj->tni", K, camera)
    front = camera[..., 2] > 0
    projected = np.divide(uvz[..., :2], uvz[..., 2:], out=np.full_like(uvz[..., :2], np.nan),
                          where=front[..., None])
    if not np.allclose(projected, result["pixels"], rtol=1e-5, atol=1e-3, equal_nan=True):
        raise ValueError("TAPIP3D original-image projection disagrees with supplied calibration")
    height, width = shape
    result["projection_valid"] = (front & np.isfinite(projected).all(axis=-1)
        & (projected[..., 0] >= 0) & (projected[..., 0] <= width - 1)
        & (projected[..., 1] >= 0) & (projected[..., 1] <= height - 1))
    # Raw XYZ is retained. Missing image projections use an explicit mask so
    # downstream JSON evidence does not contain nonstandard NaN coordinates.
    result["pixels"] = np.where(np.isfinite(result["pixels"]), result["pixels"], 0)
    result["camera_from_world"] = extrinsics.copy()
    return result


def track(frame_paths, depth_paths, *, K, depth_scale, masks,
          extrinsics=None, stationary_camera=False):
    """Track all selected rigid parts in one bounded, offline model worker.

    ``masks`` are first-frame boolean [B,H,W], ordered by the caller. Returned
    ``query_owner`` fixes every track's body membership for the whole video.
    No instance meshes, semantic points or task names enter the model.
    """
    if isinstance(frame_paths, (str, Path)) or isinstance(depth_paths, (str, Path)):
        raise TypeError("RGB and depth paths must be ordered sequences")
    frames, depths = [Path(p).resolve() for p in frame_paths], [Path(p).resolve() for p in depth_paths]
    if not frames or len(frames) != len(depths):
        raise ValueError("TAPIP3D requires equal nonempty RGB and depth sequences")
    if any(not path.is_file() for path in frames + depths):
        raise FileNotFoundError("TAPIP3D requires existing local RGB-D files")
    matrix = np.asarray(K, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError("K must be a finite nonsingular pinhole intrinsic matrix")
    unit = np.asarray(depth_scale)
    if unit.shape != () or unit.dtype.kind not in "iuf" or not np.isfinite(unit) or unit <= 0:
        raise ValueError("depth_scale must be positive metres per raw depth unit")
    with Image.open(frames[0]) as image:
        shape = (image.height, image.width)
    selected = np.asarray(masks)
    if (selected.ndim != 3 or selected.shape[1:] != shape or not len(selected)
            or selected.dtype != np.bool_ or not selected.any(axis=(1, 2)).all()):
        raise ValueError("masks must be nonempty boolean [B,H,W] initial-instance masks")
    if np.any(selected.sum(axis=0) > 1):
        raise ValueError("Independent rigid-part masks must not overlap")
    E = camera_extrinsics(extrinsics, len(frames), stationary_camera=stationary_camera)
    _, first_depth = _read_frame(frames[0], depths[0], float(unit), shape)
    queries, owners, pixels = sample_queries(first_depth, matrix, selected, E[0])
    from .runtime import run_file_worker

    output = run_file_worker("tapip3d", frames, {
        "depth_paths": [str(path) for path in depths], "K": matrix.tolist(),
        "depth_scale": float(unit), "queries": queries.tolist(), "extrinsics": E.tolist(),
        "shape": list(shape)})
    result = _outputs(output, len(frames), queries, matrix, E, shape)
    result["query_owner"], result["query_pixels"] = owners, pixels
    return result


def worker_main(paths, output_path, *, depth_paths, K, depth_scale, queries, extrinsics, shape):
    """Run only the local calibrated RGB-D tracker under runtime's GPU/RAM guard."""
    recovery = Path(__file__).resolve().parents[2]
    source = recovery / "TAPIP3D"
    checkpoint = source / "checkpoints" / "tapip3d_final.pth"
    if not (source / "rgbd_tracker.py").is_file() or not checkpoint.is_file():
        raise FileNotFoundError("Local TAPIP3D RGB-D source/checkpoint is missing")
    count, (height, width) = len(paths), tuple(shape)
    # Cap decoded input storage before allocating whole-video arrays. The
    # runtime also monitors combined parent/worker RSS and caps CUDA at 8 GiB.
    if count * height * width * 7 > 2 * 2**30:
        raise ValueError("RGB-D decoded inputs exceed this worker's 2 GiB host input budget")
    video = np.empty((count, height, width, 3), dtype=np.uint8)
    depths = np.empty((count, height, width), dtype=np.float32)
    for index, (rgb, depth) in enumerate(zip(paths, depth_paths)):
        video[index], depths[index] = _read_frame(rgb, depth, depth_scale, (height, width))
    sys.path.insert(0, str(source))
    from rgbd_tracker import RGBDTracker

    tracker = RGBDTracker(checkpoint)
    result = tracker.track(video, depths, np.asarray(K), np.asarray(queries, dtype=np.float32),
                           extrinsics=np.asarray(extrinsics), image_size=IMAGE_SIZE,
                           num_iters=6, support_grid_size=8, vis_threshold=VISIBILITY_THRESHOLD)
    # Only small point arrays return to the parent; no full video, model
    # features, depth predictions or model-resident state cross the boundary.
    np.savez_compressed(output_path, **{key: result[key] for key in
                        ("coords", "visibs", "query_points", "pixels")})
# END calibrated TAPIP3D worker.
