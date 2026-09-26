"""Local model-based rigid pose tracking with FoundationPose.

Models are explicitly supplied metric triangle meshes, never inferred from an
object label or reconstructed from a point track. Returned poses map the scaled
original mesh coordinates into the OpenCV camera frame (right/down/forward).
Geometry checks measure rendered-depth agreement, not calibrated confidence or
ground-truth accuracy; symmetries and coherent pose errors can remain ambiguous.
"""

# BEGIN model-based FoundationPose worker integration; no model loads on import.
from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image


# Depth agreement has a metric floor and an object-size term, rather than
# requiring every surface pixel to match at millimetre precision. Occlusion
# and invalid sensor pixels may remove support, but cannot count as agreement.
DEPTH_FLOOR_M = 0.01
DEPTH_DIAMETER_FRACTION = 0.05
MIN_SUPPORT_FRACTION = 0.1
MIN_SUPPORT_PIXELS = 16
REGISTER_ITERATIONS = 5
TRACK_ITERATIONS = 2


def _inputs(frame_paths, depth_paths, K, depth_scale, masks, models):
    """Validate explicit files, aligned initial masks and metric model scales."""
    if isinstance(frame_paths, (str, Path)) or isinstance(depth_paths, (str, Path)):
        raise TypeError("RGB and depth paths must be ordered sequences")
    frames, depths = [Path(p).resolve() for p in frame_paths], [Path(p).resolve() for p in depth_paths]
    if not frames or len(frames) != len(depths):
        raise ValueError("FoundationPose requires equal nonempty RGB and depth sequences")
    missing = [str(p) for p in frames + depths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing local RGB-D files: " + ", ".join(missing))
    matrix = np.asarray(K, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or abs(matrix[0, 1]) > 1e-12 or abs(matrix[1, 0]) > 1e-12
            or not np.allclose(matrix[2], [0, 0, 1], atol=1e-12, rtol=0)
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError("K must be a finite nonsingular pinhole matrix with zero skew")
    # Native depth2xyzmap uses fx/fy/cx/cy directly, so a general projective K
    # would disagree with its backprojection even if that matrix is invertible.
    depth_unit = np.asarray(depth_scale)
    if (depth_unit.shape != () or depth_unit.dtype.kind not in "iuf"
            or not np.isfinite(depth_unit) or depth_unit <= 0):
        raise ValueError("depth_scale must be positive metres per raw depth unit")
    with Image.open(frames[0]) as image:
        width, height = image.size
    selected = np.asarray(masks)
    if (selected.ndim != 3 or selected.shape[1:] != (height, width)
            or selected.dtype != np.bool_ or len(selected) == 0
            or not selected.reshape(len(selected), -1).any(axis=1).all()):
        raise ValueError("masks must be nonempty boolean [N,H,W] initial-instance masks")
    if not isinstance(models, list) or len(models) != len(selected):
        raise ValueError("models must be an ordered list with one explicit mesh per mask")
    normalized = []
    for model in models:
        if not isinstance(model, dict) or "mesh" not in model or set(model) - {"mesh", "scale"}:
            raise ValueError("Each model requires mesh and optional scale, with no inferred geometry")
        path = Path(model["mesh"])
        if not path.is_absolute() or not path.is_file():
            raise FileNotFoundError(f"FoundationPose requires an existing absolute mesh path: {path}")
        scale = np.asarray(model.get("scale", 1.))
        if (scale.dtype.kind not in "iuf" or scale.shape not in {(), (3,)}
                or not np.isfinite(scale).all() or (scale <= 0).any()):
            raise ValueError("Mesh scale must be a positive scalar or three positive axis scales")
        scale = np.broadcast_to(scale, (3,)).astype(float)
        normalized.append({"mesh": str(path.resolve()), "scale": scale.tolist()})
    return frames, depths, matrix, float(depth_scale), selected, normalized


def _pose(value):
    """Reject invalid SE(3) without repairing a failed estimator's rotation."""
    pose = np.asarray(value, dtype=float)
    if (pose.shape != (4, 4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-9, rtol=0)
            or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6, rtol=0)
            or not np.isclose(np.linalg.det(pose[:3, :3]), 1., atol=1e-6, rtol=0)):
        raise ValueError("FoundationPose returned an invalid rigid camera_from_model transform")
    return pose


def _outputs(output, frame_count, model_count):
    """Validate CPU transport and require explicit evidence for every pose."""
    shapes = {
        "camera_from_model": (frame_count, model_count, 4, 4),
        "validated": (frame_count, model_count),
        "rendered_pixels": (frame_count, model_count),
        "support_pixels": (frame_count, model_count),
        "support_fraction": (frame_count, model_count),
        "nonoccluded_pixels": (frame_count, model_count),
        "occluded_pixels": (frame_count, model_count),
        "free_space_conflict_pixels": (frame_count, model_count),
        "nonoccluded_support_fraction": (frame_count, model_count),
        "depth_median_error_m": (frame_count, model_count),
        "depth_p90_error_m": (frame_count, model_count),
        "depth_tolerance_m": (model_count,),
        "initial_mask_iou": (model_count,),
        "initial_mask_depth_coverage": (model_count,),
        "registration_candidate_count": (model_count,),
    }
    result = {}
    for key, shape in shapes.items():
        if key not in output:
            raise ValueError(f"FoundationPose worker omitted {key}")
        array = np.asarray(output[key])
        if array.shape != shape or array.dtype.kind not in "biuf" or not np.isfinite(array).all():
            raise ValueError(f"FoundationPose worker has invalid {key} shape, dtype or values")
        result[key] = array
    if result["validated"].dtype != np.bool_ or not result["validated"].all():
        raise ValueError("FoundationPose did not validate every requested pose")
    if result["camera_from_model"].dtype.kind != "f":
        raise ValueError("FoundationPose poses must use floating-point metric coordinates")
    for pose in result["camera_from_model"].reshape(-1, 4, 4):
        _pose(pose)
    for key in ("rendered_pixels", "support_pixels", "nonoccluded_pixels", "occluded_pixels",
                "free_space_conflict_pixels", "registration_candidate_count"):
        if result[key].dtype.kind not in "iu":
            raise ValueError(f"FoundationPose {key} must contain integer counts")
    if (np.any(result["rendered_pixels"] < MIN_SUPPORT_PIXELS)
            or np.any(result["support_pixels"] < MIN_SUPPORT_PIXELS)
            or np.any(result["support_pixels"] > result["rendered_pixels"])
            or np.any(result["nonoccluded_pixels"] < result["support_pixels"])
            or np.any(result["occluded_pixels"] < 0)
            or np.any(result["free_space_conflict_pixels"] < 0)
            or np.any(result["support_pixels"] + result["free_space_conflict_pixels"]
                      > result["nonoccluded_pixels"])
            or np.any(result["nonoccluded_pixels"] + result["occluded_pixels"]
                      > result["rendered_pixels"])
            or np.any(result["support_fraction"] < MIN_SUPPORT_FRACTION)
            or np.any(result["support_fraction"] > 1)
            or not np.allclose(result["support_fraction"],
                               result["support_pixels"] / result["rendered_pixels"], atol=1e-6)
            or np.any(result["nonoccluded_support_fraction"] <= .5)
            or np.any(result["nonoccluded_support_fraction"] > 1)
            or not np.allclose(result["nonoccluded_support_fraction"],
                               result["support_pixels"] / result["nonoccluded_pixels"], atol=1e-6)
            or np.any(result["depth_tolerance_m"] <= 0)
            or np.any(result["depth_median_error_m"] < 0)
            or np.any(result["depth_p90_error_m"] < result["depth_median_error_m"])
            or np.any(result["initial_mask_iou"] < MIN_SUPPORT_FRACTION)
            or np.any(result["initial_mask_iou"] > 1)
            or np.any(result["initial_mask_depth_coverage"] <= .5)
            or np.any(result["initial_mask_depth_coverage"] > 1)
            or np.any(result["registration_candidate_count"] < 2)):
        raise ValueError("FoundationPose worker returned unsupported or inconsistent geometry diagnostics")
    return result


def track(frame_paths, depth_paths, *, K, depth_scale, masks, models):
    """Return camera_from_model[T,N,4,4] and explicit per-entity depth checks.

    ``masks`` are boolean initial-frame masks in the same order as ``models``.
    Scale converts each existing mesh's vertices to metres without changing its
    local origin or axes. Initial registration uses the native 252-candidate
    grid and five refinement iterations; following frames use two iterations.
    Any missing model, incomplete registration or unsupported pose aborts the
    sequence. This model-based entry performs no point tracking or reconstruction.
    """
    frames, depths, matrix, unit, selected, normalized = _inputs(
        frame_paths, depth_paths, K, depth_scale, masks, models)
    from .runtime import run_file_worker

    output = run_file_worker(kind="foundationpose", paths=frames, options={
        "depth_paths": [str(p) for p in depths], "K": matrix.tolist(), "depth_scale": unit,
        "masks": selected, "models": normalized})
    return _outputs(output, len(frames), len(selected))


def _load_frame(rgb_path, depth_path, depth_scale, shape):
    with Image.open(rgb_path) as image:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
    if depth_path.suffix.lower() == ".npy":
        raw = np.load(depth_path, allow_pickle=False)
    else:
        with Image.open(depth_path) as image:
            raw = np.asarray(image)
    if rgb.shape != (*shape, 3) or raw.shape != shape or raw.dtype.kind not in "iuf":
        raise ValueError("RGB, aligned depth and masks must retain the same calibrated dimensions")
    depth = np.asarray(raw, dtype=np.float32) * depth_scale
    # FoundationPose's depth kernels use zero for absent measurements. Invalid
    # sensor values are removed, never replaced by fabricated metric depth.
    depth[~np.isfinite(depth) | (depth < .001)] = 0
    return rgb, depth


def _load_mesh(model):
    """Load one supplied rigid visual model, preserving its scale and origin."""
    import trimesh

    mesh = trimesh.load(model["mesh"], process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("FoundationPose requires one triangle mesh; assemble scene parts and materials explicitly")
    mesh.apply_scale(np.asarray(model["scale"], dtype=float))
    if (mesh.vertices.ndim != 2 or mesh.vertices.shape[1] != 3 or len(mesh.vertices) < 3
            or mesh.faces.ndim != 2 or mesh.faces.shape[1] != 3 or not len(mesh.faces)
            or not np.isfinite(mesh.vertices).all() or not np.isfinite(mesh.vertex_normals).all()
            or not np.isfinite(mesh.area) or mesh.area <= 0):
        raise ValueError("FoundationPose model must contain finite, nondegenerate metric triangles")
    # The native renderer accepts one texture image + UVs or vertex colours.
    # Silently discarding a MultiMaterial/PBR assembly would alter the model.
    if isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals):
        texture = getattr(mesh.visual.material, "image", None)
        uv = np.asarray(mesh.visual.uv)
        if (not isinstance(texture, Image.Image) or uv.shape != (len(mesh.vertices), 2)
                or not np.isfinite(uv).all()):
            raise ValueError("FoundationPose textured meshes need one local image and finite per-vertex UVs")
    return mesh


def _geometry(depth, rendered_depth, tolerance, initial_mask=None):
    """Measure projection/depth support while allowing partial occlusion.

    At least 10% of projected model pixels and 16 measured pixels must agree
    within max(10 mm, 5% model diameter). These are acceptance limits, not an
    accuracy guarantee. A nearer observation can occlude the model; a farther
    one contradicts its occupancy of observed free space. More than half of
    the measured, nonoccluded model projection must agree. Initial registration
    additionally requires depth support covering more than half the measured
    instance mask, and agreement outside that mask cannot support this object.
    Reported median/p90 use all observed rendered pixels, including occluders
    and mismatches; they are not clipped to the inlier set.
    """
    rendered = np.asarray(rendered_depth, dtype=float)
    if rendered.shape != depth.shape or not np.isfinite(rendered).all():
        raise ValueError("FoundationPose renderer returned invalid depth")
    footprint = rendered >= .001
    measured = np.isfinite(depth) & (depth >= .001)
    overlap = footprint & measured
    supported = overlap & (np.abs(depth - rendered) <= tolerance)
    occluded = overlap & (depth < rendered - tolerance)
    free_space_conflict = overlap & (depth > rendered + tolerance)
    # The denominator retains same-depth pixels outside the initial mask:
    # otherwise an oversized coplanar model could borrow background support.
    nonoccluded_count = int((supported | free_space_conflict).sum())
    iou, mask_coverage = None, None
    if initial_mask is not None:
        union = footprint | initial_mask
        iou = float((footprint & initial_mask).sum() / max(1, union.sum()))
        supported &= initial_mask
        mask_coverage = float(supported.sum() / max(1, (initial_mask & measured).sum()))
        if iou < MIN_SUPPORT_FRACTION:
            raise ValueError(f"Model projection does not support the initial mask (IoU={iou:.3f})")
        if mask_coverage <= .5:
            raise ValueError(f"Model depth does not cover a majority of the initial mask ({mask_coverage:.3f})")
    count, support = int(footprint.sum()), int(supported.sum())
    fraction = support / max(1, count)
    if count < MIN_SUPPORT_PIXELS or support < MIN_SUPPORT_PIXELS or fraction < MIN_SUPPORT_FRACTION:
        raise ValueError(f"No sufficient model/depth support: {support}/{count} pixels within {tolerance:.4f} m")
    nonoccluded_fraction = support / max(1, nonoccluded_count)
    if nonoccluded_fraction <= .5:
        raise ValueError("Model does not explain a majority of its nonoccluded projection: "
                         f"{support}/{nonoccluded_count} supported pixels; "
                         f"{int(free_space_conflict.sum())} free-space conflicts")
    errors = np.abs(depth[overlap] - rendered[overlap])
    return {"rendered_pixels": count, "support_pixels": support, "support_fraction": fraction,
            "nonoccluded_pixels": nonoccluded_count, "occluded_pixels": int(occluded.sum()),
            "free_space_conflict_pixels": int(free_space_conflict.sum()),
            "nonoccluded_support_fraction": nonoccluded_fraction,
            "depth_median_error_m": float(np.median(errors)),
            "depth_p90_error_m": float(np.quantile(errors, .9)), "initial_mask_iou": iou,
            "initial_mask_depth_coverage": mask_coverage}


def _numpy(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _centered_pose(camera_from_model, model_center):
    """Native mesh_tensors are centred; returned public poses are not."""
    result = np.asarray(camera_from_model, dtype=float).copy()
    result[:3, 3] += result[:3, :3] @ np.asarray(model_center, dtype=float)
    return result


def _track_sequence(frames, depths, K, depth_scale, masks, estimators, render_depth):
    """Run independent native object states, with injected CPU test boundaries."""
    count, entities = len(frames), len(estimators)
    shape = masks.shape[1:]
    result = {"camera_from_model": np.empty((count, entities, 4, 4)),
              "validated": np.zeros((count, entities), dtype=bool),
              "initial_mask_iou": np.empty(entities),
              "initial_mask_depth_coverage": np.empty(entities),
              "registration_candidate_count": np.empty(entities, dtype=np.int64),
              "depth_tolerance_m": np.array([max(DEPTH_FLOOR_M, DEPTH_DIAMETER_FRACTION * est.diameter)
                                             for est in estimators])}
    for key in ("rendered_pixels", "support_pixels", "support_fraction", "nonoccluded_pixels",
                "occluded_pixels", "free_space_conflict_pixels", "nonoccluded_support_fraction",
                "depth_median_error_m", "depth_p90_error_m"):
        result[key] = np.empty((count, entities), dtype=np.int64 if key.endswith("pixels") else float)
    for frame, (rgb_path, depth_path) in enumerate(zip(frames, depths)):
        rgb, depth = _load_frame(rgb_path, depth_path, depth_scale, shape)
        for entity, estimator in enumerate(estimators):
            try:
                if frame == 0:
                    if (masks[entity] & (depth >= .001)).sum() < MIN_SUPPORT_PIXELS:
                        raise ValueError("Initial mask has insufficient measured depth")
                    estimator.pose_last = None
                    pose = estimator.register(K=K, rgb=rgb, depth=depth, ob_mask=masks[entity],
                                              iteration=REGISTER_ITERATIONS)
                    # register can return a finite coarse matrix when filtered
                    # depth is insufficient. That branch never sets pose_last
                    # or scores; accepting only SE(3) would silently start lost.
                    if estimator.pose_last is None or not hasattr(estimator, "scores"):
                        raise ValueError("FoundationPose registration did not complete")
                    scores = _numpy(estimator.scores).reshape(-1)
                    if len(scores) < 2 or not np.isfinite(scores).all():
                        raise ValueError("FoundationPose registration did not score valid pose candidates")
                    result["registration_candidate_count"][entity] = len(scores)
                else:
                    pose = estimator.track_one(rgb=rgb, depth=depth, K=K, iteration=TRACK_ITERATIONS)
                pose = _pose(pose)
                rendered = render_depth(estimator, pose, K, *shape)
                check = _geometry(depth, rendered, result["depth_tolerance_m"][entity],
                                  initial_mask=masks[entity] if frame == 0 else None)
                for key, value in check.items():
                    if key in {"initial_mask_iou", "initial_mask_depth_coverage"}:
                        if frame == 0:
                            result[key][entity] = value
                    else:
                        result[key][frame, entity] = value
                result["camera_from_model"][frame, entity] = pose
                result["validated"][frame, entity] = True
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"FoundationPose failed at frame {frame} ({rgb_path.name}), model {entity}: {error}") from error
    return _outputs(result, count, entities)


def worker_main(paths, output, *, depth_paths, K, depth_scale, masks, models):
    """Load local networks once and process RGB-D frames serially in the worker."""
    frames, depths, matrix, unit, selected, normalized = _inputs(
        paths, depth_paths, K, depth_scale, masks, models)
    source = Path(__file__).resolve().parents[2] / "FoundationPose"
    required = [source / "estimater.py"]
    for run in ("2023-10-28-18-33-37", "2024-01-11-20-02-45"):
        required.extend(source / "weights" / "no_diffusion" / run / name
                        for name in ("model_best.pth", "config.yml"))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing local FoundationPose resources: " + ", ".join(missing))
    meshes = [_load_mesh(model) for model in normalized]
    sys.path.insert(0, str(source))
    import torch
    import nvdiffrast.torch as dr
    from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
    from Utils import nvdiffrast_render

    if not torch.cuda.is_available():
        raise RuntimeError("Local FoundationPose requires the recovery CUDA environment")
    torch.set_num_threads(2)
    scorer, refiner = ScorePredictor(), PoseRefinePredictor()
    context = dr.RasterizeCudaContext()
    estimators = []
    with tempfile.TemporaryDirectory(prefix="functo-foundationpose-") as debug_dir:
        try:
            for mesh in meshes:
                estimators.append(FoundationPose(
                    model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                    scorer=scorer, refiner=refiner, glctx=context, debug=0, debug_dir=debug_dir))

            def render_depth(estimator, pose, intrinsic, height, width):
                # Public poses refer to original mesh coordinates; the renderer
                # here uses the estimator's internally centred mesh tensors.
                centred = _centered_pose(pose, estimator.model_center)
                _, depth, _ = nvdiffrast_render(
                    K=intrinsic, H=height, W=width,
                    ob_in_cams=torch.as_tensor(centred, device="cuda", dtype=torch.float32)[None],
                    glctx=context, mesh_tensors=estimator.mesh_tensors, extra={})
                return depth[0].detach().cpu().numpy()

            with torch.inference_mode():
                result = _track_sequence(frames, depths, matrix, unit, selected, estimators, render_depth)
            np.savez_compressed(Path(output), **result)
        finally:
            # Each native estimator exports its own temporary OBJ outside the
            # debug directory; delete only paths owned by these instances.
            for estimator in estimators:
                if estimator.mesh_path is not None:
                    Path(estimator.mesh_path).unlink(missing_ok=True)
# END model-based FoundationPose worker integration.
