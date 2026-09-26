"""Calibrated RGB-D observations and rigid-part motion, without semantic points.

All translations and depths are metres. A pose maps column-vector points from
the initial body frame into the initial camera reference frame. With a tracked
reference, its motion is removed before returning the body poses; without one,
the camera must be stationary. The body frame's axes initially match the camera
and its origin is the centroid of the initial valid body correspondences.

This module neither assigns object roles nor loads perception models. Optical
flow propagates correspondence hypotheses and geometric consensus checks their
rigidity; neither proves physical identity after occlusion or coherent drift.
It cannot recover an unobserved object or silently fill lost tracks.
"""

# BEGIN task-adaptive observations: independent of tool/function/grasp fields.
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Camera:
    """Pinhole calibration; ``depth_scale`` is metres per raw depth unit."""

    K: np.ndarray
    width: int
    height: int
    depth_scale: float = 0.001


def _intrinsics(K):
    matrix = np.asarray(K, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12)
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError("K must be a finite, nonsingular pinhole intrinsic matrix")
    return matrix


def load_calibration(path) -> Camera:
    """Read Open3D calibration, including its column-major flattened matrix.

    ``path`` may be intrinsic.json or its directory. ``depth_unit_m`` explicitly
    gives metres per raw value. The existing Open3D ``depth_scale`` metadata is
    a *divisor*, e.g. 1000 means millimetres. Either field may occur in intrinsic
    or its sibling config.json; conflicting declarations are rejected. When no
    scale is declared, the repository's millimetre convention (0.001) applies.
    This intentionally does not guess conventions from the magnitude of data.
    """
    path = Path(path)
    if path.is_dir():
        path = path / "intrinsic.json"
    data = json.loads(path.read_text())
    raw = np.asarray(data["intrinsic_matrix"], dtype=float)
    if raw.shape == (9,):
        raw = raw.reshape(3, 3, order="F")
    matrix = _intrinsics(raw).copy()
    dimensions = [data["width"], data["height"]]
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in dimensions):
        raise ValueError("Calibration width and height must be positive integers")
    scales = []
    metadata = [data]
    config_path = path.parent / "config.json"
    if config_path.is_file() and config_path != path:
        metadata.append(json.loads(config_path.read_text()))
    for record in metadata:
        for field in ("depth_unit_m", "depth_scale"):
            if field in record:
                value = float(record[field])
                if not np.isfinite(value) or value <= 0:
                    raise ValueError(f"{field} must be finite and positive")
                scales.append(value if field == "depth_unit_m" else 1.0 / value)
    if scales and not np.allclose(scales, scales[0], rtol=1e-9, atol=0):
        raise ValueError("Conflicting depth-unit declarations in calibration/config")
    matrix.setflags(write=False)
    return Camera(matrix, *dimensions, scales[0] if scales else 0.001)


def _frame_file(directory, index, suffixes):
    files = [Path(directory) / f"{index:05d}{suffix}" for suffix in suffixes]
    found = [path for path in files if path.is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected exactly one frame {index} in {directory}; found {len(found)}")
    return found[0]


def load_rgbd(directory, index=0):
    """Read one aligned RGB-D pair; invalid depths remain invalid for selection.

    Depth must already be registered to RGB. Intrinsic image dimensions are
    checked, rather than rescaling calibration or guessing a depth alignment.
    """
    root = Path(directory)
    camera = load_calibration(root)
    with Image.open(_frame_file(root / "rgb", index, (".jpg", ".png", ".jpeg"))) as image:
        rgb = image.convert("RGB")
    depth_path = _frame_file(root / "depth", index, (".png", ".tiff", ".tif", ".npy"))
    if depth_path.suffix == ".npy":
        raw_depth = np.load(depth_path, allow_pickle=False)
    else:
        with Image.open(depth_path) as image:
            raw_depth = np.asarray(image)
    if rgb.size != (camera.width, camera.height) or raw_depth.shape != (camera.height, camera.width):
        raise ValueError("RGB, aligned depth and calibration dimensions must agree")
    depth = np.asarray(raw_depth, dtype=float) * camera.depth_scale
    return rgb, depth, camera


def _mask(mask, shape, name="mask"):
    array = np.asarray(mask)
    if array.shape != shape or array.dtype != np.bool_:
        raise ValueError(f"{name} must be a boolean array with shape {shape}")
    if not array.any():
        raise ValueError(f"{name} has no selected pixels")
    return array


def backproject(depth_m, K, pixels_uv=None, mask=None):
    """Backproject metric depth, preserving explicit correspondence identities.

    Without ``pixels_uv``, select every finite positive depth (within mask, if
    given); no maximum range is imposed. With explicit N-by-2 (u, v) positions,
    sample nearest-pixel depth and retain subpixel rays. Any invalid selected
    depth, out-of-bounds pixel or mask exclusion raises, rather than dropping a
    row and accidentally changing correspondence IDs. No image mutation occurs.
    """
    depth = np.asarray(depth_m, dtype=float)
    matrix = _intrinsics(K)
    if depth.ndim != 2 or not depth.size:
        raise ValueError("Depth must be a nonempty two-dimensional array")
    selected_mask = None if mask is None else _mask(mask, depth.shape)
    if pixels_uv is None:
        valid = np.isfinite(depth) & (depth > 0)
        if selected_mask is not None:
            valid &= selected_mask
        rows, cols = np.nonzero(valid)
        pixels = np.column_stack((cols, rows)).astype(float)
        values = depth[rows, cols]
    else:
        pixels = np.asarray(pixels_uv, dtype=float)
        if pixels.ndim != 2 or pixels.shape[1] != 2 or not np.isfinite(pixels).all():
            raise ValueError("pixels_uv must be a finite N-by-2 array of (u, v)")
        if (np.any(pixels < 0) or np.any(pixels[:, 0] > depth.shape[1] - 1)
                or np.any(pixels[:, 1] > depth.shape[0] - 1)):
            raise ValueError("Selected pixels are out of bounds")
        rounded = np.rint(pixels).astype(np.int64)
        cols, rows = rounded.T
        values = depth[rows, cols]
        if not np.all(np.isfinite(values) & (values > 0)):
            raise ValueError("Selected pixels contain invalid depth")
        if selected_mask is not None and not selected_mask[rows, cols].all():
            raise ValueError("Selected pixels lie outside the mask")
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(matrix).T
    return rays * values[:, None]


def _noncollinear(points):
    """Three noncollinear correspondences suffice, including planar surfaces."""
    if len(points) < 3:
        return False
    spans = np.linalg.svd(points - points.mean(axis=0), compute_uv=False)
    return bool(spans[1] > max(1e-9, spans[0] * 1e-4))


class _RigidCorrespondences:
    """Use Open3D's rigid estimator inside correspondence-scored RANSAC.

    Scoring must use the supplied feature pairs, not nearest-neighbour cloud
    distance: a wrong feature permutation can still align two cloud surfaces.
    skimage supplies the maintained RANSAC sampling/termination implementation.
    """

    def estimate(self, source, target):
        import open3d as o3d

        if not _noncollinear(source) or not _noncollinear(target):
            return False
        first = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(source))
        second = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(target))
        indices = np.arange(len(source), dtype=np.int32)
        pairs = o3d.utility.Vector2iVector(np.column_stack((indices, indices)))
        estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
        self.params = estimator.compute_transformation(first, second, pairs)
        return bool(np.isfinite(self.params).all())

    def residuals(self, source, target):
        predicted = source @ self.params[:3, :3].T + self.params[:3, 3]
        return np.linalg.norm(predicted - target, axis=1)


def rigid_registration(source, target, max_error=0.01):
    """Fit a rigid source-to-target transform from known corresponding N-by-3 rows.

    Returns ``(T, diagnostics)``. The accepted consensus must contain at least
    half of the supplied correspondences and three noncollinear points; there
    is no rank-three requirement. ``max_error`` is a metric per-pair threshold,
    not an accuracy claim. Failure is explicit, including a nonrigid majority.
    """
    from skimage.measure import ransac

    source, target = np.asarray(source, dtype=float), np.asarray(target, dtype=float)
    if (source.ndim != 2 or source.shape[1] != 3 or target.shape != source.shape
            or not np.isfinite(source).all() or not np.isfinite(target).all()):
        raise ValueError("Registration requires finite, corresponding N-by-3 arrays")
    if not np.isfinite(max_error) or max_error <= 0:
        raise ValueError("max_error must be a positive metric distance")
    if not _noncollinear(source) or not _noncollinear(target):
        raise ValueError("Rigid registration needs at least three noncollinear correspondences")
    # Deterministic sampling only affects this call, not a shared global RNG.
    model, inliers = ransac(
        (source, target), _RigidCorrespondences, min_samples=3,
        residual_threshold=float(max_error), max_trials=1000,
        stop_probability=0.999, rng=np.random.default_rng(0),
        is_data_valid=lambda a, b: _noncollinear(a) and _noncollinear(b))
    if model is None or inliers is None:
        raise ValueError("No rigid correspondence consensus was found")
    # RANSAC refits its model on the consensus. Re-score that refined model so
    # diagnostics and acceptance do not describe an earlier minimal hypothesis.
    errors = model.residuals(source, target)
    inliers = errors <= max_error
    required = max(3, int(np.ceil(0.5 * len(source))))
    if (int(inliers.sum()) < required or not _noncollinear(source[inliers])
            or not _noncollinear(target[inliers])):
        raise ValueError("Insufficient rigid consensus; correspondences may be lost or nonrigid")
    transform = np.asarray(model.params).copy()
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7)):
        raise ValueError("Estimated transformation is not a proper rigid transform")
    return transform, {
        "num_correspondences": len(source), "num_inliers": int(inliers.sum()),
        "inlier_mask": inliers, "residuals_m": errors,
        "inlier_rmse_m": float(np.sqrt(np.mean(errors[inliers] ** 2))),
        "max_inlier_error_m": float(errors[inliers].max()),
        "source_spans_m": np.linalg.svd(source[inliers] - source[inliers].mean(axis=0),
                                          compute_uv=False),
    }


def _track_arrays(tracks, valid_mask, name):
    points = np.asarray(tracks, dtype=float)
    valid = np.asarray(valid_mask)
    if (points.ndim != 3 or points.shape[2] != 3 or not len(points)
            or valid.shape != points.shape[:2] or valid.dtype != np.bool_):
        raise ValueError(f"{name} needs tracks[N,M,3] and boolean valid_mask[N,M]")
    if not np.isfinite(points[valid]).all():
        raise ValueError(f"{name} marks nonfinite 3D observations as valid")
    return points, valid


def poses_from_tracks(tracks, *, valid_mask, reference_tracks=None,
                      reference_valid_mask=None, frame_indices=None, max_error=0.01):
    """Convert identity-preserving 3D tracks to body poses and local geometry.

    Column j must refer to the same physical feature at every frame. Missing
    observations require a false entry in ``valid_mask``; no association is
    inferred from unordered arrays. Registration uses only correspondences
    shared with frame zero. With a reference, ``inv(T_reference) @ T_body``
    removes camera/reference motion before applying the initial body offset.
    """
    points, valid = _track_arrays(tracks, valid_mask, "Body")
    count = len(points)
    if frame_indices is None:
        indices = np.arange(count, dtype=int)
    else:
        indices = np.asarray(frame_indices)
        if (indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices[1:] <= indices[:-1]).any()):
            raise ValueError("frame_indices must be nonnegative, strictly increasing integer frame IDs")
    if not _noncollinear(points[0, valid[0]]):
        raise ValueError("Initial body observations need noncollinear valid points")
    if (reference_tracks is None) != (reference_valid_mask is None):
        raise ValueError("Reference tracks and their valid mask must be supplied together")
    reference = ref_valid = None
    if reference_tracks is not None:
        reference, ref_valid = _track_arrays(reference_tracks, reference_valid_mask, "Reference")
        if len(reference) != count:
            raise ValueError("Body and reference tracks must use the same frame sequence")
        if not _noncollinear(reference[0, ref_valid[0]]):
            raise ValueError("Initial reference observations need noncollinear valid points")
    origin = points[0, valid[0]].mean(axis=0)
    initial = np.eye(4)
    initial[:3, 3] = origin
    poses = [initial]
    diagnostics = [{"frame_index": int(indices[0]), "initial_valid_points": int(valid[0].sum())}]
    for frame in range(1, count):
        shared = valid[0] & valid[frame]
        try:
            body, body_info = rigid_registration(points[0, shared], points[frame, shared], max_error)
            reference_motion = np.eye(4)
            ref_info = None
            if reference is not None:
                shared_ref = ref_valid[0] & ref_valid[frame]
                reference_motion, ref_info = rigid_registration(
                    reference[0, shared_ref], reference[frame, shared_ref], max_error)
        except ValueError as error:
            raise ValueError(f"Rigid tracking failed at frame {int(indices[frame])}: {error}") from error
        poses.append(np.linalg.solve(reference_motion, body @ initial))
        diagnostics.append({"frame_index": int(indices[frame]), "body": body_info,
                            "reference": ref_info})
    return {
        "poses": np.asarray(poses), "points": points[0, valid[0]] - origin,
        "frame_indices": indices.copy(), "diagnostics": diagnostics,
        "body_origin_camera0_m": origin,
        "body_frame": "initial correspondence centroid; axes aligned with initial camera",
        "reference_frame": "initial camera coordinates",
        "camera_assumption": ("reference motion compensated; reference defines relative motion"
                              if reference is not None else "stationary camera required"),
        "units": "metres", "pose_convention": "reference_from_body; column vectors",
    }


def _sample_tracks(depth, K, pixels, active):
    """Sample only valid depth rows while retaining the original feature IDs."""
    points = np.full((len(pixels), 3), np.nan)
    valid = active & np.isfinite(pixels).all(axis=1)
    valid &= ((pixels[:, 0] >= 0) & (pixels[:, 0] <= depth.shape[1] - 1)
              & (pixels[:, 1] >= 0) & (pixels[:, 1] <= depth.shape[0] - 1))
    candidates = np.flatnonzero(valid)
    rounded = np.rint(pixels[candidates]).astype(int)
    depths = depth[rounded[:, 1], rounded[:, 0]]
    valid[candidates] &= np.isfinite(depths) & (depths > 0)
    points[valid] = backproject(depth, K, pixels[valid])
    return points, valid


def _initial_features(gray, depth, mask, name):
    import cv2

    # Shrinking the ROI avoids mixing moving-part and background depth at its
    # boundary. This is a tracking observation mask, not a changed SAM output.
    selected = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    selected &= (np.isfinite(depth) & (depth > 0)).astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=250, qualityLevel=0.01,
                                    minDistance=5, mask=selected, blockSize=5)
    if corners is None or len(corners) < 6:
        raise ValueError(f"Insufficient textured, valid-depth features for {name}")
    return corners.reshape(-1, 2)


def _advance_flow(previous_gray, gray, pixels, active):
    import cv2

    indices = np.flatnonzero(active)
    updated = pixels.copy()
    surviving = np.zeros(len(pixels), dtype=bool)
    if not len(indices):
        return updated, surviving
    original = pixels[indices].astype(np.float32).reshape(-1, 1, 2)
    options = dict(winSize=(21, 21), maxLevel=3,
                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    forward, status, _ = cv2.calcOpticalFlowPyrLK(previous_gray, gray, original, None, **options)
    if forward is None:
        return updated, surviving
    backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous_gray, forward, None, **options)
    if backward is None:
        return updated, surviving
    valid = status.ravel().astype(bool) & back_status.ravel().astype(bool)
    valid &= np.isfinite(forward.reshape(-1, 2)).all(axis=1)
    valid &= np.linalg.norm(backward.reshape(-1, 2) - original.reshape(-1, 2), axis=1) <= 1.0
    surviving[indices[valid]] = True
    updated[indices[valid]] = forward.reshape(-1, 2)[valid]
    return updated, surviving


def track_rgbd(directory, mask, *, reference_mask=None, max_frames=None, max_error=0.01):
    """Track one rigid part through calibrated RGB-D, with bounded CPU memory.

    Masks select the body/reference in the first RGB frame. At most 250 optical
    flow features per entity and 3000 initial geometry points are retained.
    Forward/backward flow agreement and depth validity precede rigid RANSAC;
    losing sufficient correspondences raises instead of extrapolating poses.
    Files are read one frame at a time. No new feature IDs are silently inserted.
    """
    import cv2

    root = Path(directory)
    indices = sorted({int(path.stem) for path in (root / "rgb").iterdir()
                      if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit()})
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 2:
            raise ValueError("max_frames must be an integer of at least two")
        indices = indices[:max_frames]
    if len(indices) < 2:
        raise ValueError("RGB-D motion tracking requires at least two frames")
    image, depth, camera = load_rgbd(root, indices[0])
    body_mask = _mask(mask, depth.shape, "Body mask")
    ref_mask = None if reference_mask is None else _mask(reference_mask, depth.shape, "Reference mask")
    if ref_mask is not None and np.any(body_mask & ref_mask):
        raise ValueError("Body and reference masks must select distinct, nonoverlapping rigid parts")
    geometry = backproject(depth, camera.K, mask=body_mask)
    if len(geometry) > 3000:
        geometry = geometry[np.linspace(0, len(geometry) - 1, 3000, dtype=int)]
    old_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        masks = [body_mask] + ([] if ref_mask is None else [ref_mask])
        names = ["body", "reference"]
        pixels = [_initial_features(gray, depth, selected, names[i]) for i, selected in enumerate(masks)]
        active = [np.ones(len(group), dtype=bool) for group in pixels]
        histories, validity = [], []
        for group, visible in zip(pixels, active):
            observed, valid = _sample_tracks(depth, camera.K, group, visible)
            histories.append([observed])
            validity.append([valid])
        for index in indices[1:]:
            image, depth, current_camera = load_rgbd(root, index)
            if (not np.array_equal(camera.K, current_camera.K)
                    or camera.depth_scale != current_camera.depth_scale):
                raise ValueError("RGB-D calibration changed within the track")
            next_gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            for entity in range(len(pixels)):
                pixels[entity], active[entity] = _advance_flow(gray, next_gray, pixels[entity], active[entity])
                observed, valid = _sample_tracks(depth, camera.K, pixels[entity], active[entity])
                if valid.sum() < 6:
                    raise ValueError(f"RGB-D tracking lost {names[entity]} correspondences at frame {index}")
                histories[entity].append(observed)
                validity[entity].append(valid)
            gray = next_gray
        result = poses_from_tracks(
            np.asarray(histories[0]), valid_mask=np.asarray(validity[0]),
            reference_tracks=None if ref_mask is None else np.asarray(histories[1]),
            reference_valid_mask=None if ref_mask is None else np.asarray(validity[1]),
            frame_indices=indices, max_error=max_error)
    finally:
        cv2.setNumThreads(old_threads)
    result["points"] = geometry - result["body_origin_camera0_m"]
    result["tracking_method"] = "CPU pyramidal LK + forward/backward check + rigid correspondence RANSAC"
    return result
# END task-adaptive observations.
