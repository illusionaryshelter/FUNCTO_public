"""Calibrated RGB-D observations and rigid-part motion, without semantic points.

All translations and depths are metres. A pose maps column-vector points from
the initial body frame into the initial camera reference frame. With a tracked
reference, its motion is removed before returning the body poses; without one,
the camera must be stationary. The body frame's axes initially match the camera
and its origin is the centroid of the initial observed body geometry.

This module neither assigns object roles nor keeps GPU models resident.
FoundationPose estimates each rigid object's camera pose in an isolated worker.
Its model frame is bound once to the initial observed body frame; tracking does
not repeatedly sample depth at a projected point and assume material identity.
The worker must validate model-to-observation geometry, not merely return SE(3).
An object name alone does not establish that a mesh matches that instance.
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


# BEGIN FoundationPose coordinates: bind fixed material points once, not tracks.
def _model_poses(value, name):
    """Check the transform convention without repairing invalid predictions.

    These checks establish only numerical SE(3) validity. They do not measure
    pose accuracy, object identity, symmetry ambiguity, or tracking success.
    """
    poses = np.asarray(value, dtype=float)
    if (poses.ndim != 3 or poses.shape[1:] != (4, 4) or not len(poses)
            or not np.isfinite(poses).all()):
        raise ValueError(f"{name} must be finite poses with shape [T,4,4]")
    if not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-9, rtol=0):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotations = poses[:, :3, :3]
    if (not np.allclose(np.swapaxes(rotations, 1, 2) @ rotations, np.eye(3),
                        atol=1e-6, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1., atol=1e-6, rtol=0)):
        raise ValueError(f"{name} rotations must be proper orthonormal matrices")
    return poses


def poses_from_models(camera_from_model, initial_points, *,
                      reference_camera_from_model=None, frame_indices=None):
    """Convert validated model poses to a fixed observed body frame.

    ``camera_from_model[t]`` must be FoundationPose's returned transform of the
    *original* mesh, not its internally centered ``pose_last``. ``initial_points``
    are finite metric observations in the first camera frame. With their
    centroid c, P0=[I,c] maps body to camera zero. The single binding
    X=inv(B0)@P0 maps this body into the carrier mesh. For a tracked reference R,
    returned poses are R0@inv(Rt)@Bt@X; without a reference they are Bt@X and
    require a stationary camera. This removes reference motion, including any
    actual reference-object motion, rather than asserting a fixed world frame.

    Local points inv(P0)@p0 remain material-point hypotheses throughout the
    sequence. No further depth lookup changes their identity. This function is
    pure coordinate conversion, accepts a single frame, and makes no tracking
    quality claim; the FoundationPose worker performs observation validation.
    """
    models = _model_poses(camera_from_model, "Carrier camera_from_model")
    points = np.asarray(initial_points, dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not len(points)
            or not np.isfinite(points).all()):
        raise ValueError("initial_points must be a nonempty finite N-by-3 array")
    count = len(models)
    if frame_indices is None:
        indices = np.arange(count, dtype=int)
    else:
        indices = np.asarray(frame_indices)
        if (indices.shape != (count,) or not np.issubdtype(indices.dtype, np.integer)
                or (indices < 0).any() or (indices[1:] <= indices[:-1]).any()):
            raise ValueError("frame_indices must be nonnegative, strictly increasing integer frame IDs")
    reference = None
    if reference_camera_from_model is not None:
        reference = _model_poses(reference_camera_from_model, "Reference camera_from_model")
        if len(reference) != count:
            raise ValueError("Body and reference poses must use the same frame sequence")
    origin = points.mean(axis=0)
    initial = np.eye(4)
    initial[:3, 3] = origin
    model_from_body = np.linalg.solve(models[0], initial)
    poses = models @ model_from_body
    if reference is not None:
        poses = reference[0] @ np.linalg.solve(reference, poses)
    # This frame defines the coordinate gauge. Remove only floating-point
    # round-off from the exact identity B0@inv(B0)@P0, not a pose estimation error.
    poses[0] = initial
    return {
        "poses": poses, "points": points - origin,
        "frame_indices": indices.copy(), "model_from_body": model_from_body,
        "body_origin_camera0_m": origin,
        "body_frame": "initial observed geometry centroid; axes aligned with initial camera",
        "reference_frame": "initial camera coordinates",
        "camera_assumption": ("reference motion compensated; reference defines relative motion"
                              if reference is not None else "stationary camera required"),
        "units": "metres", "pose_convention": "reference_from_body; column vectors",
    }
# END FoundationPose coordinates.


# BEGIN explicit instance geometry for FoundationPose; no category-mesh guess.
def _pose_models(root, roles):
    """Resolve metric instance meshes from the existing RGB-D config file.

    Mesh paths are relative to config.json unless absolute. Scale must be
    explicit, positive, and either scalar or three per-axis factors. A category
    name is insufficient input; validation of the selected geometry against
    RGB-D remains the worker's responsibility.
    """
    config_path = root / "config.json"
    if not config_path.is_file():
        raise ValueError("FoundationPose requires config.json pose_models for the selected entities")
    config = json.loads(config_path.read_text())
    models = config.get("pose_models") if isinstance(config, dict) else None
    if not isinstance(models, dict):
        raise ValueError("FoundationPose requires config.json pose_models for the selected entities")
    selected = []
    for role in roles:
        model = models.get(role)
        if (not isinstance(model, dict) or not isinstance(model.get("mesh"), str)
                or not model["mesh"].strip()):
            raise ValueError(f"FoundationPose pose_models.{role} must specify its instance mesh")
        mesh = Path(model["mesh"]).expanduser()
        mesh = (mesh if mesh.is_absolute() else config_path.parent / mesh).resolve()
        if not mesh.is_file():
            raise FileNotFoundError(f"FoundationPose {role} mesh does not exist: {mesh}")
        scale = model.get("scale")
        components = scale if isinstance(scale, list) else [scale]
        numeric = all(isinstance(value, (int, float)) and not isinstance(value, bool)
                      for value in components)
        values = np.asarray(components, dtype=float) if numeric else np.array([])
        if (not numeric or (isinstance(scale, list) and len(scale) != 3)
                or not np.isfinite(values).all() or (values <= 0).any()):
            raise ValueError(f"FoundationPose {role} scale must be an explicit positive scalar or 3-vector")
        selected.append({"mesh": str(mesh), "scale": values.tolist() if isinstance(scale, list) else float(values[0])})
    return selected
# END explicit instance geometry.


def track_rgbd(directory, mask, *, reference_mask=None, max_frames=None):
    """Recover object/part motion with validated local FoundationPose poses.

    First-frame masks and config.json ``pose_models`` bind each selected rigid
    entity to its own instance mesh. The isolated worker owns tracking state
    and validates rendered geometry against RGB-D at every frame. This layer
    validates the transport contract and converts original-mesh poses once to
    the existing observed-body frame. No point tracker or fallback is used.
    """
    # BEGIN FoundationPose observation path.
    from local_models.foundationpose import track

    root = Path(directory)
    indices = sorted({int(path.stem) for path in (root / "rgb").iterdir()
                      if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem.isdigit()})
    if max_frames is not None:
        if isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 2:
            raise ValueError("max_frames must be an integer of at least two")
        indices = indices[:max_frames]
    if len(indices) < 2:
        raise ValueError("RGB-D motion tracking requires at least two frames")
    _, depth, camera = load_rgbd(root, indices[0])
    body_mask = _mask(mask, depth.shape, "Body mask")
    ref_mask = None if reference_mask is None else _mask(reference_mask, depth.shape, "Reference mask")
    if ref_mask is not None and np.any(body_mask & ref_mask):
        raise ValueError("Body and reference masks must select distinct, nonoverlapping rigid parts")
    masks = [body_mask] + ([] if ref_mask is None else [ref_mask])
    roles = ["carrier"] + ([] if ref_mask is None else ["reference"])
    models = _pose_models(root, roles)
    for role, selected in zip(roles, masks):
        if np.count_nonzero(selected & np.isfinite(depth) & (depth > 0)) < 6:
            raise ValueError(f"FoundationPose {role} needs at least six valid initial depth observations")
    geometry = backproject(depth, camera.K, mask=body_mask)
    if len(geometry) > 3000:
        geometry = geometry[np.linspace(0, len(geometry) - 1, 3000, dtype=int)]
    frame_paths, depth_paths = [], []
    # Validate all files before launching a GPU worker. Decoding is sequential,
    # so full videos are not accumulated in host memory by this preflight.
    for index in indices:
        _, depth, current_camera = load_rgbd(root, index)
        if (not np.array_equal(camera.K, current_camera.K)
                or camera.depth_scale != current_camera.depth_scale):
            raise ValueError("RGB-D calibration changed within the track")
        if np.count_nonzero(np.isfinite(depth) & (depth > 0)) < 4:
            raise ValueError(f"FoundationPose frame {index} has insufficient valid depth")
        frame_paths.append(_frame_file(root / "rgb", index, (".jpg", ".png", ".jpeg")))
        depth_paths.append(_frame_file(root / "depth", index, (".png", ".tiff", ".tif", ".npy")))
    tracked = track(frame_paths, depth_paths, K=camera.K, depth_scale=camera.depth_scale,
                    masks=np.stack(masks), models=models)
    if not isinstance(tracked, dict) or "camera_from_model" not in tracked:
        raise ValueError("FoundationPose worker must return camera_from_model")
    poses = np.asarray(tracked["camera_from_model"])
    if poses.shape != (len(indices), len(roles), 4, 4):
        raise ValueError("FoundationPose camera_from_model must cover every frame and selected entity")
    validated = np.asarray(tracked.get("validated"))
    if (validated.shape != (len(indices), len(roles)) or validated.dtype != np.bool_
            or not validated.all()):
        raise ValueError("FoundationPose requires successful geometry validation for every frame and entity")
    result = poses_from_models(
        poses[:, 0], geometry, reference_camera_from_model=None if ref_mask is None else poses[:, 1],
        frame_indices=indices)
    result["tracking_method"] = "FoundationPose instance-mesh poses + fixed material-point binding"
    result["tracking_models"] = dict(zip(roles, models))
    result["tracking_camera_from_model"] = poses.copy()
    result["tracking_diagnostics"] = {key: value for key, value in tracked.items() if key != "camera_from_model"}
    return result
    # END FoundationPose observation path.
# END task-adaptive observations.
