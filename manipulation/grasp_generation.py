"""Generate demonstration-guided grasp candidates, never execution permission.

All positions are metres; rotations and direction bounds use radians. ``T_A_B``
maps B coordinates into A. The learned gripper base G has +X along finger
closure and +Z along approach; the controlled tool frame E is independently
bound by ``T_tool_grasp`` from the native robot asset. Region permission,
actual finger contact, unknown space and complete robot motion belong to the
existing planning validators, not this model's scores.
"""

# BEGIN bounded demonstration, diffusion and OBB candidate generation.
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .motion import _inverse, _pose_array, _poses_array


# These are global search coverage budgets, not task acceptance tolerances.
# The three scales follow the supplied prototype; none are fitted to a skill.
PRIOR_SCALES = ((32, .008, 8.), (64, .025, 30.), (96, .065, 75.))
DIFFUSION_COUNT = 64
SCORE_CHUNK = 64
OBB_YAWS = 8
OBB_OFFSETS_CM = (-2., 0.)
OBB_MAX_CANDIDATES = 5 * 7 * OBB_YAWS * len(OBB_OFFSETS_CM)


@dataclass(frozen=True)
class GraspGenerationRequest:
    """Only explicit observations, robot frame binding and optional hard angle.

    The reference frame is gravity aligned (+Z up), as required by the library's
    XY OBB construction. ``model_points_reference`` is the visible target cloud, not a claim that
    its unobserved surface is known. ``approach_reference`` specifies desired
    inward gripper +Z. Omitting the maximum angle leaves every direction
    admissible; the demonstration never invents an angular hard constraint.
    """

    model_points_reference: np.ndarray
    T_reference_tool_prior: np.ndarray
    T_tool_grasp: np.ndarray
    approach_reference: np.ndarray | None = None
    hard_max_approach_angle_rad: float | None = None


def _cloud(value):
    points = np.asarray(value)
    if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 21
            or points.dtype.kind not in "fiu" or not np.isfinite(points).all()):
        raise ValueError("model_points_reference requires at least 21 finite XYZ observations")
    return np.array(points, dtype=np.float32, copy=True)


def _seed(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(value)


def _local_priors(prior, seed):
    """Keep the exact E pose and sample right-local offsets around that E pose.

    Rotating about the actual control point, rather than the model's base,
    avoids introducing a finger-length translation when orientation changes.
    The rigid E→G binding is applied only after sampling.
    """
    rng = np.random.default_rng(seed)
    result = [prior.copy()]
    for count, position_sigma, angle_sigma_deg in PRIOR_SCALES:
        shifts = rng.normal(0., position_sigma, (count, 3))
        rotations = Rotation.from_rotvec(
            rng.normal(0., np.deg2rad(angle_sigma_deg), (count, 3))).as_matrix()
        batch = np.repeat(prior[None], count, axis=0)
        batch[:, :3, :3] = prior[:3, :3] @ rotations
        batch[:, :3, 3] += shifts @ prior[:3, :3].T
        result.extend(batch)
    return np.asarray(result)


def generate_grasp_candidates(request, backend, *, seed=0):
    """Return the finite scored pool and explicit angular rejection evidence.

    All admissible candidates, including the exact demonstration, compete by
    ``0.65 * quality + 0.35 * prior_similarity`` with stable ID ties;
    similarity is a Gaussian of tool-origin displacement / 0.035 m and full
    SO(3) angle / 35 degrees. This is only a documented, global search order,
    never a probability of success or a quality threshold. No physical
    equivalence is inferred from finger symmetry, since arm paths can differ.
    """
    seed = _seed(seed)
    points = _cloud(request.model_points_reference)
    prior = _pose_array(request.T_reference_tool_prior, "T_reference_tool_prior")
    binding = _pose_array(request.T_tool_grasp, "T_tool_grasp")
    if not np.allclose(binding, backend.T_tool_grasp, atol=1e-12, rtol=0):
        raise ValueError("Request grasp frame differs from the model's native robot binding")
    approach = request.approach_reference
    limit = request.hard_max_approach_angle_rad
    if approach is not None:
        approach = np.asarray(approach, dtype=float)
        if (approach.shape != (3,) or not np.isfinite(approach).all()
                or np.linalg.norm(approach) <= 1e-12):
            raise ValueError("approach_reference must be a finite nonzero 3-vector")
        approach = approach / np.linalg.norm(approach)
    if limit is not None:
        if (approach is None or isinstance(limit, (bool, np.bool_))
                or not np.isfinite(limit) or not 0 <= limit <= np.pi):
            raise ValueError("A hard approach angle requires an explicit direction and angle in [0, pi]")
        limit = float(limit)

    local_tool = _local_priors(prior, seed)
    local_grasp = local_tool @ binding
    prepared, preprocessing = backend.prepare_cloud(points, seed=seed)
    generated, sources, generation = backend.generate(prepared, seed=seed)
    generated = np.asarray(generated, dtype=float)
    if generated.size == 0:
        generated = np.empty((0, 4, 4))
    else:
        generated = _poses_array(generated, "generated_grasp_poses")
    if len(sources) != len(generated) or any(source not in {"diffusion", "obb"} for source in sources):
        raise ValueError("Backend must identify every generated diffusion or OBB candidate")
    if len(generated) > DIFFUSION_COUNT + OBB_MAX_CANDIDATES:
        raise RuntimeError("Backend exceeded the declared generation budget")
    grasps = np.concatenate((local_grasp, generated))
    # The exact prior is retained without float32 model round-trips. Generated
    # poses use the inverse robot binding; no model TCP is assumed to be E.
    tools = np.concatenate((local_tool, generated @ _inverse(binding)))
    labels = ["demo_exact"] + ["demo_local"] * (len(local_tool) - 1) + list(sources)
    quality = np.asarray(backend.score(grasps, prepared, seed=seed), dtype=float)
    if (quality.shape != (len(grasps),) or not np.isfinite(quality).all()
            or np.any(quality < 0.) or np.any(quality > 1.)):
        raise ValueError("Discriminator must return one finite [0, 1] score per candidate")
    delta_p = np.linalg.norm(tools[:, :3, 3] - prior[:3, 3], axis=1)
    delta_r = Rotation.from_matrix(prior[:3, :3].T @ tools[:, :3, :3]).magnitude()
    similarity = np.exp(-.5 * ((delta_p / .035) ** 2 + (delta_r / np.deg2rad(35.)) ** 2))
    utility = .65 * quality + .35 * similarity
    # atan2 avoids acos amplifying dot-product roundoff near zero. Normalize
    # candidate axes independently because validated model poses carry FP32
    # roundoff. This changes angle computation, never the requested hard limit.
    axes = grasps[:, :3, 2]
    axes = axes / np.linalg.norm(axes, axis=1)[:, None]
    angles = (None if approach is None else np.arctan2(
        np.linalg.norm(np.cross(axes, approach), axis=1), axes @ approach))
    candidates, rejected = [], []
    for index, (tool_pose, grasp_pose, source) in enumerate(zip(tools, grasps, labels)):
        record = {
            "candidate_id": f"grasp_{index:04d}", "status": "candidate", "source": source,
            "T_reference_tool": tool_pose.copy(), "T_reference_grasp": grasp_pose.copy(),
            "quality": float(quality[index]), "ranking_score": float(utility[index]),
            "prior_translation_distance_m": float(delta_p[index]),
            "prior_rotation_distance_rad": float(delta_r[index]),
            "approach_angle_rad": None if angles is None else float(angles[index]),
        }
        if limit is not None and angles[index] > limit:
            record["rejection_reason"] = "explicit_hard_approach_angle"
            record["hard_max_approach_angle_rad"] = limit
            rejected.append(record)
        else:
            candidates.append(record)
    # BEGIN all sources compete under the same demonstration-guided objective.
    # Keep the exact prior as an ordinary candidate. Its similarity already
    # encodes the intended preference; a separate source priority would prevent
    # any higher-scoring alternative from improving a plannable but poor prior.
    # IDs resolve exact score ties only, without adding a quality threshold.
    candidates.sort(key=lambda item: (-item["ranking_score"], item["candidate_id"]))
    # END ranking changes search order, never physical acceptance requirements.
    for rank, candidate in enumerate(candidates):
        candidate["rank"] = rank
    return {
        "status": "candidate_pool", "candidates": candidates, "rejected_candidates": rejected,
        "diagnostics": {
            "seed": seed, "preprocessing": preprocessing, "generation": generation,
            "generated_count": len(grasps), "retained_count": len(candidates),
            "hard_angle_rejected_count": len(rejected),
            "source_counts": {name: labels.count(name) for name in sorted(set(labels))},
            "ranking": "all_candidates_0.65_quality_plus_0.35_prior_similarity_then_candidate_id",
            "prior_position_scale_m": .035, "prior_rotation_scale_rad": float(np.deg2rad(35.)),
            "execution_validated": False,
        },
    }


class GraspGenXBackend:
    """Installed PyTorch model plus bounded official GraspGenX primitives.

    Imports are lazy: CPU geometry/policy tests need no model or CUDA. Model
    errors propagate unchanged. An explicit checkpoint directory is required;
    this adapter never downloads resources, compiles TensorRT, or supplies a
    collision mesh. The native robot remains the collision authority.
    """

    def __init__(self, *, checkpoints_dir, gripper_geometry):
        import torch
        from graspgenx.grasp_server import GraspGenXSampler, SWEEP_VOLUME_ONLY_BACKBONES
        from graspgenx.utils.checkpoint_io import load_model_cfg
        from graspgenx.x_grippers import make_sweep_volume_gripper_info

        root = Path(checkpoints_dir).expanduser()
        self.T_tool_grasp = _pose_array(gripper_geometry["T_tool_grasp"], "T_tool_grasp")
        cfg = load_model_cfg(str(root / "gen"), str(root / "dis"))
        self._torch = torch
        self.model_point_limit = int(cfg.data.num_points)
        if not 21 <= self.model_point_limit <= 3500:
            raise ValueError("This resource-bounded adapter requires a model point budget of 21..3500")
        # BEGIN direct public sweep construction without the ZMQ serving import.
        # The library convenience factory imports serving.types through a
        # package initializer that requires its optional messaging stack. Use
        # the exact same public geometry constructor and sampler constructor
        # directly. Preserve the factory's backbone restriction: placeholders
        # in unused mesh/point fields must never condition an asset-based model
        # or become robot collision geometry.
        for name in ("diffusion", "discriminator"):
            backbone = str(getattr(cfg, name).gripper_backbone)
            if backbone not in SWEEP_VOLUME_ONLY_BACKBONES:
                raise ValueError(f"cfg.{name}.gripper_backbone={backbone!r} requires real gripper assets")
        params = gripper_geometry["sweep_volume_params"]
        vectors = ("extents_open", "offset_open", "extents_mid", "offset_mid")
        if not isinstance(params, dict) or set(params) != set(vectors) | {"gripper_type", "fingertip_depth"}:
            raise ValueError("Native sweep geometry requires its four vectors, gripper_type and explicit fingertip_depth")
        validated = {}
        for name in vectors:
            raw = np.asarray(params[name])
            if raw.shape != (3,) or raw.dtype.kind not in "fiu":
                raise ValueError(f"{name} requires three metric numeric values")
            value = raw.astype(np.float32)
            if not np.isfinite(value).all() or (name.startswith("extents_") and np.any(value <= 0)):
                raise ValueError(f"{name} requires finite values and strictly positive extents")
            validated[name] = value
        kind = params["gripper_type"]
        if isinstance(kind, (bool, np.bool_)) or not isinstance(kind, (int, np.integer)) or kind not in (0, 1, 2):
            raise ValueError("gripper_type must be a declared integer in {0, 1, 2}")
        depth = np.asarray(params["fingertip_depth"])
        if depth.shape != () or depth.dtype.kind not in "fiu" or not np.isfinite(depth) or depth <= 0:
            raise ValueError("fingertip_depth must be an explicit finite positive metric depth")
        validated.update(gripper_type=int(kind), fingertip_depth=float(depth))
        gripper_info = make_sweep_volume_gripper_info(**validated)
        self.sampler = GraspGenXSampler(cfg, gripper_info=gripper_info, use_tensorrt=False)
        # END the installed model path does not require optional serving tools.

    def prepare_cloud(self, points, *, seed):
        """Bound N before the official N×N kNN allocation; filter exactly once."""
        from graspgenx.utils.point_cloud import point_cloud_outlier_removal

        points = _cloud(points)
        count = len(points)
        if count > self.model_point_limit:
            indices = np.random.default_rng(seed).choice(count, self.model_point_limit, replace=False)
            points = points[indices]
        filtered, removed = point_cloud_outlier_removal(self._torch.from_numpy(points))
        result = filtered.cpu().numpy().astype(np.float32)
        if len(result) < 21:
            raise ValueError("Model cloud has fewer than 21 points after outlier removal; no raw-cloud substitution")
        return result, {"observed_points": count, "bounded_points": len(points),
                        "model_points": len(result), "removed_outliers": len(removed),
                        "filter": "graspgenx_L1_mean_20_neighbors_lt_0.014m"}

    def generate(self, points, *, seed):
        """One diffusion pass and at most 560 OBB poses; no quality pruning.

        The OBB spacing is max(3 cm, filtered AABB diagonal / 8). Since every
        oriented axis extent is bounded by that diagonal, each face has at
        most seven interior offsets. Five faces × seven offsets × eight yaws
        × two depth offsets gives the documented cap. Library OBB construction
        and face transforms are reused; scoring happens once for the union.
        """
        from graspgenx.grasp_server import GraspGenXSampler
        from graspgenx.samplers.graspmoe import (
            _compute_obb, _build_face_candidates, _interior_positions)

        torch = self._torch
        device = next(self.sampler.model.parameters()).device
        numpy_state = np.random.get_state()
        try:
            np.random.seed(seed % (2 ** 32))
            with torch.random.fork_rng(devices=[device.index]):
                torch.manual_seed(seed)
                diffusion, _ = GraspGenXSampler.run_inference(
                    points, self.sampler, grasp_threshold=-1., num_grasps=DIFFUSION_COUNT,
                    topk_num_grasps=DIFFUSION_COUNT, min_grasps=DIFFUSION_COUNT,
                    max_tries=1, remove_outliers=False)
                diff = diffusion.detach().cpu().numpy().reshape((-1, 4, 4))
                # PCA is an explicit library OBB method: unlike "advanced", it
                # does not silently perturb/replace geometry after an error.
                center, half, rotation = _compute_obb(points, mode="pca")
        finally:
            np.random.set_state(numpy_state)
        spacing = max(.03, float(np.linalg.norm(np.ptp(points, axis=0))) / 8.)
        yaws = np.linspace(0., 2 * np.pi, OBB_YAWS, endpoint=False)
        offsets = np.asarray(OBB_OFFSETS_CM) / 100.
        blocks = []
        long_axis = int(half[1] > half[0])
        # Each tuple identifies a face normal, face centre and in-plane axis.
        faces = [(np.array([0., 0., 1.]), center + [0., 0., half[2]], long_axis)]
        for axis in (0, 1):
            for sign in (-1., 1.):
                normal = sign * rotation[:, axis]
                faces.append((normal, center + normal * half[axis], 1 - axis))
        for normal, origin, tangent_axis in faces:
            positions = _interior_positions(float(half[tangent_axis]), spacing)
            if len(positions) > 7:
                raise RuntimeError("OBB position budget exceeded before candidate allocation")
            blocks.append(_build_face_candidates(
                face_origin_world=origin, approach_dir_world=normal,
                in_plane_axis_world=rotation[:, tangent_axis], positions_local=positions,
                yaws=yaws, z_offsets_m=offsets, gripper_depth_m=float(self.sampler.gripper.depth)))
        obb = np.concatenate(blocks)
        if len(diff) > DIFFUSION_COUNT or len(obb) > OBB_MAX_CANDIDATES:
            raise RuntimeError("Model generation exceeded its finite candidate budget")
        return np.concatenate((diff, obb)), ["diffusion"] * len(diff) + ["obb"] * len(obb), {
            "diffusion_count": len(diff), "obb_count": len(obb), "obb_max_candidates": OBB_MAX_CANDIDATES,
            "obb_spacing_m": spacing, "obb_method": "official_PCA_XY_top_and_four_sides",
            "obb_offsets_m": offsets.tolist(), "model": "GraspGenX_PyTorch",
        }

    def score(self, poses, points, *, seed):
        """One point-cloud encoding shared by every 64-pose scoring chunk.

        The first chunk uses the same data construction as the library's
        external-grasp scoring helper, but retains its returned object feature.
        Subsequent chunks call that helper with the feature explicitly supplied.
        This matters because PTv3 shuffles serialization even in eval mode.
        """
        from graspgenx.dataset.dataset import collate
        from graspgenx.samplers.graspmoe import _score_grasps_with_discriminator

        if not len(poses):
            return np.empty(0, dtype=np.float32)
        torch = self._torch
        device = next(self.sampler.model.parameters()).device
        center = points.mean(axis=0).astype(np.float32)
        centered = torch.from_numpy(points - center).to(device)
        first = np.asarray(poses[:SCORE_CHUNK], dtype=np.float32).copy()
        first[:, :3, 3] -= center
        numpy_state = np.random.get_state()
        try:
            np.random.seed(seed % (2 ** 32))
            with torch.random.fork_rng(devices=[device.index]), torch.inference_mode():
                torch.manual_seed(seed)
                # "pick" is the installed discriminator's required dataset
                # schema token (collate creates task_is_pick), shared by its
                # official external-grasp scorer; it is not a skill selector.
                data = {"task": "pick", "points": centered,
                        "inputs": torch.cat((centered, torch.zeros_like(centered)), dim=-1)}
                data = self.sampler.load_gripper_input(data)
                data["grasps"] = torch.from_numpy(first).to(device)
                data["grasp_key"] = "grasps"
                batch = collate([data])
                batch["grasp_key"] = "grasps"
                result, _, _ = self.sampler.model.grasp_discriminator.infer(batch)
                embedding = result["object_embedding"]
                scores = [result["grasp_confidence"][0, :, 0].detach().cpu().numpy()]
                for start in range(SCORE_CHUNK, len(poses), SCORE_CHUNK):
                    scores.append(_score_grasps_with_discriminator(
                        np.asarray(poses[start:start + SCORE_CHUNK], dtype=np.float32),
                        centered, center, self.sampler, object_embedding=embedding))
        finally:
            np.random.set_state(numpy_state)
        return np.concatenate(scores).astype(np.float32)
# END bounded demonstration, diffusion and OBB candidate generation.
