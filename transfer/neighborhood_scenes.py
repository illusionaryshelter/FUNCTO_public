"""Build support-location experiments without changing the frozen first suite.

This fixture/export module reuses original scene assets, records three interior
shelf levels from one visible camera, and imports an actual recorded mug-on-
shelf evidence mask. Scene labels, body transforms, and support identities are
written only to evaluation records. Automatic inputs remain observed RGB-D and
the explicitly supplied whole-entity segmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import scenes


ROOT = scenes.ROOT
OLD_DATA = scenes.DEFAULT_OUTPUT
DEFAULT_OUTPUT = OLD_DATA.parent / "data-neighborhood"
DEMO_SOURCE = ROOT / "sim-version-plans/2026-10-13-put-mug-region-marco/source/shelf_support"
RECORDING = ROOT / "collect/scripted_recordings/20260927T135735_590b9b7c"
PREPROCESSED = ROOT / "collect/preprocessed/20260927T135735_590b9b7c"
FRAME = 197
SUPPORT_NAMES = ("lower_interior", "middle_interior", "upper_interior", "exterior_top")
SUPPORT_BANDS = ((.0098, .011), (.0798, .081), (.1495, .152), (.2205, .222))
LOW_CAMERA_POSITION = np.array([-.38, -.60, 1.26])
LOW_CAMERA_TARGET = np.array([-.236, .278, 1.035])
RAW_VIEW_FILES = ("rgb.png", "depth_m.npy", "seg_geom_id.npy", "calibration.json",
                  "camera.json", "observation.npz")


def archive(path: Path) -> dict:
    """Load finite typed records without enabling NumPy object deserialization."""
    with np.load(path, allow_pickle=False) as values:
        return {key: values[key] for key in values.files}


def load_native_case(folder: Path):
    """Restore a case's own saved state into its own copied scene model."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    state = archive(folder / "state.npz")
    spec = int(state["state_spec"])
    if state["integration_state"].shape != (mujoco.mj_stateSize(model, spec),):
        raise ValueError("Saved state is incompatible with its scene model")
    mujoco.mj_setState(model, data, state["integration_state"], spec)
    mujoco.mj_forward(model, data)
    return model, data


def local_mesh_vertices(model, data) -> np.ndarray:
    """Recover the actual shelf visual vertices in its body's coordinates."""
    geom = model.geom("wooden_shelf_1_g0")
    body = model.body("wooden_shelf_1_main").id
    mesh = int(geom.dataid[0])
    vertices = model.mesh_vert[model.mesh_vertadr[mesh]:model.mesh_vertadr[mesh]+model.mesh_vertnum[mesh]]
    world = vertices @ data.geom_xmat[geom.id].reshape(3, 3).T + data.geom_xpos[geom.id]
    return (world-data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def annotated_shelf_supports(model, data) -> np.ndarray:
    """Read four actual surfaces; the transfer algorithm never sees their IDs."""
    return np.stack([scenes.annotated_mesh_support(model, data,
                    dict(scenes.FIXTURES["shelf"], z_band=band)) for band in SUPPORT_BANDS])


def world_supports(canonical_corners: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """Apply the evaluator's independently known body/shape map to corners."""
    return canonical_corners @ transform[:3, :3].T + transform[:3, 3]


def support_labels(points: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Label actual visible horizontal support surfaces, not vertical extent bins.

    The physical plane and its finite quadrilateral are both checked using the
    existing 0.4 mm evaluation tolerance. Vertical walls between tiers therefore
    retain -1 instead of inheriting a guessed shelf number from their height.
    """
    labels = np.full(len(points), -1, dtype=np.int32)
    for index, plane in enumerate(corners):
        mask = scenes.evaluate_visible_points(points, plane)["support_mask"]
        if np.any(labels[mask] >= 0):
            raise ValueError("Independent support annotations overlap")
        labels[mask] = index
    return labels


def augment_support_truth(observation: dict, truth: dict, corners: np.ndarray,
                          preferred: int) -> dict:
    """Give every evaluation record the same explicit support-label contract."""
    output = dict(truth)
    labels = support_labels(observation["xyz"], corners)
    output.update(support_id=labels, preferred_support_id=np.array(preferred, np.int32),
                  support_corners_all=corners, support_mask=labels == preferred)
    return output


def save_overlay(folder: Path, observation: dict, truth: dict) -> None:
    """Render evaluator labels separately from the unmodified model input RGB."""
    pixels = observation["pixels"]
    image = np.array(Image.open(folder / "rgb.png").convert("RGB"))
    for mask, color, blend in ((truth["support_mask"], [40, 180, 255], .45),
                               (truth["region_mask"], [255, 60, 60], .75)):
        x, y = pixels[np.asarray(mask, bool)].T
        image[y, x] = ((1-blend)*image[y, x] + blend*np.array(color)).astype(np.uint8)
    Image.fromarray(image).save(folder / "evaluation-overlay.png")


def copy_view(source: Path, target: Path) -> None:
    """Copy small observation files; originals and future labels cannot alias."""
    target.mkdir(parents=True, exist_ok=False)
    for name in RAW_VIEW_FILES:
        shutil.copy2(source / name, target / name)


def low_camera_config() -> dict:
    """Build one world-fixed look-at camera, shared by all three shelf states.

    Its intrinsics match the existing 640x480, 45-degree camera profile. Only
    position and orientation differ. The chosen low view keeps all three
    internal levels visible simultaneously; no asset, tier, or texture changes.
    """
    forward = LOW_CAMERA_TARGET-LOW_CAMERA_POSITION
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    quat = Rotation.from_matrix(rotation).as_quat()[[3, 0, 1, 2]]
    config = json.loads((scenes.ORIGINAL / "view_a/camera.json").read_text())
    config.update(position_world_m=LOW_CAMERA_POSITION.tolist(),
                  quaternion_world_camera_gl_wxyz=quat.tolist())
    return config


def observed_points(frame: dict, calibration: dict, mask: np.ndarray) -> dict:
    """Backproject only finite optical depth, retaining the original pixel IDs."""
    depth = np.asarray(frame["depth_m"])
    mask = np.asarray(mask, bool) & np.isfinite(depth) & (depth > 0)
    y, x = np.nonzero(mask)
    K, T = np.asarray(calibration["K"]), np.asarray(calibration["T_world_camera_cv"])
    rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(K).T
    points = (rays*depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
    return {"xyz": points, "pixels": np.column_stack([x, y]).astype(np.int32),
            "object_mask": mask, "K": K, "T_world_camera_cv": T}


def stable_surface_ids(model, seg: np.ndarray, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Use sorted geom names in both recording and targets, never raw geom IDs."""
    ids = scenes.descendant_geoms(model, "wooden_shelf_1_main")
    names = sorted(model.geom(int(index)).name for index in ids)
    lookup = np.full(model.ngeom, -1, np.int32)
    for index, name in enumerate(names):
        lookup[model.geom(name).id] = index
    output = lookup[seg[pixels[:, 1], pixels[:, 0]]]
    if np.any(output < 0):
        raise ValueError("Whole-shelf mask contains a non-shelf geometry owner")
    return output, np.asarray(names)


def canonical_truth(observation: dict, model, frame: dict, transform: np.ndarray) -> dict:
    """Export evaluator-only coordinates and stable owner identities."""
    inverse = np.linalg.inv(transform)
    ids, names = stable_surface_ids(model, frame["seg_geom_id"], observation["pixels"])
    return {"canonical_xyz": observation["xyz"] @ inverse[:3, :3].T + inverse[:3, 3],
            "canonical_to_world": transform, "world_to_canonical": inverse,
            "surface_id": ids, "surface_names": names}


def old_camera_visibility(canonical_corners: np.ndarray) -> list[dict]:
    """Retain why original A/B cannot test the chosen interior patches."""
    rows = []
    for variant in scenes.VARIANTS:
        for view in scenes.VIEWS:
            folder = OLD_DATA / "shelf" / variant / view
            observation = archive(folder / "observation.npz")
            transform = archive(folder / "evaluation.npz")["canonical_to_world"]
            corners = world_supports(canonical_corners, transform)
            depth = np.load(folder / "depth_m.npy", allow_pickle=False)
            for tier in range(3):
                truth = scenes.evaluate_visible_points(observation["xyz"], corners[tier])
                visibility = scenes.patch_visibility(truth, observation["pixels"], depth,
                                                     observation["K"], observation["T_world_camera_cv"])
                rows.append({"variant": variant, "view": view, "support": SUPPORT_NAMES[tier],
                             "observed_evidence_points": int(truth["region_mask"].sum()), **visibility})
    return rows


def copy_legacy_suite(output: Path, canonical_corners: np.ndarray) -> list[dict]:
    """Retain the original six observation pairs under the new label contract."""
    destination = output / "legacy"
    shutil.copytree(OLD_DATA, destination)
    for folder in sorted(destination.glob("*/*/view_*")):
        observation = archive(folder / "observation.npz")
        truth = archive(folder / "evaluation.npz")
        if folder.parts[-3] == "shelf":
            corners = world_supports(canonical_corners, truth["canonical_to_world"])
            preferred = 3
        else:
            corners = truth["support_corners"][None]
            preferred = 0
        truth = augment_support_truth(observation, truth, corners, preferred)
        np.savez_compressed(folder / "evaluation.npz", **truth)
    old_manifest = json.loads((OLD_DATA / "manifest.json").read_text())
    pairs = []
    for pair in old_manifest["pairs"]:
        pairs.append({**pair, "source": "legacy/"+pair["source"],
                      "target": "legacy/"+pair["target"],
                      "source_evidence_mean_reference": True,
                      "reference_definition": "source observed evidence arithmetic mean mapped by independent canonical transforms",
                      "evidence_mask_role": "given source evidence; target legacy rectangle is diagnostic only",
                      "feature_source": "original frozen feature pair with unchanged images and observations"})
    return pairs


def render_repeated_shelf(output: Path, canonical_corners: np.ndarray) -> list[dict]:
    """Render each complete shelf once, then give each query its own labels."""
    from collect.camera import FixedCamera

    pairs = []
    for variant in scenes.VARIANTS:
        old_case = OLD_DATA / "shelf" / variant
        model, data = load_native_case(old_case)
        capture = output / "captures/repeated" / variant
        capture.mkdir(parents=True, exist_ok=False)
        shutil.copy2(old_case / "scene.xml", capture / "scene.xml")
        shutil.copy2(old_case / "state.npz", capture / "state.npz")
        config = low_camera_config()
        camera = FixedCamera(model, data, 640, 480, camera_name=None, camera_config=config)
        try:
            frame, calibration, renderer = camera.capture(data), camera.calibration, camera.renderer_info
        finally:
            camera.close()
        ids = scenes.descendant_geoms(model, "wooden_shelf_1_main")
        observation = observed_points(frame, calibration, np.isin(frame["seg_geom_id"], ids))
        Image.fromarray(frame["rgb"]).save(capture / "rgb.png")
        np.save(capture / "depth_m.npy", frame["depth_m"], allow_pickle=False)
        np.save(capture / "seg_geom_id.npy", frame["seg_geom_id"], allow_pickle=False)
        np.savez_compressed(capture / "observation.npz", **observation)
        scenes.write_json(capture / "calibration.json", calibration)
        scenes.write_json(capture / "camera.json", config)
        original_calibration = json.loads((scenes.ORIGINAL / "view_a/calibration.json").read_text())
        if not np.allclose(calibration["K"], original_calibration["K"], atol=1e-9, rtol=0):
            raise ValueError("Low camera changed the agreed intrinsic calibration")
        transform = archive(old_case / "view_a/evaluation.npz")["canonical_to_world"]
        base = canonical_truth(observation, model, frame, transform)
        corners = world_supports(canonical_corners, transform)
        visibility_rows = []
        for tier in range(3):
            name = SUPPORT_NAMES[tier]
            folder = output / "repeated" / name / variant
            copy_view(capture, folder)
            truth = scenes.evaluate_visible_points(observation["xyz"], corners[tier])
            truth.update(base)
            truth = augment_support_truth(observation, truth, corners, tier)
            visibility = scenes.patch_visibility(truth, observation["pixels"], frame["depth_m"],
                                                 observation["K"], observation["T_world_camera_cv"])
            truth.update({key: np.array(value) for key, value in visibility.items()})
            if truth["region_mask"].sum() < 12 or not visibility["region_center_visible"]:
                raise ValueError(f"Interior observation prerequisite failed: {name}/{variant}")
            np.savez_compressed(folder / "evaluation.npz", **truth)
            save_overlay(folder, observation, truth)
            annotation = {"support": name, "preferred_support_id": tier,
                          "support_names": SUPPORT_NAMES,
                          "region_center_definition": "independent physical rectangle centre; primary reference uses source evidence mean",
                          "region_mask_role": "source localization evidence; target rectangle is diagnostic only",
                          "evidence_points": int(truth["region_mask"].sum()), **visibility}
            scenes.write_json(folder / "annotation.json", annotation)
            visibility_rows.append(annotation)
        scenes.write_json(capture / "audit.json", {"source_case": str(old_case),
            "scene_and_state_copied_without_change": True, "complete_repeated_shelf": True,
            "furniture_types_present": ["shelf"], "camera_position_world_m": LOW_CAMERA_POSITION.tolist(),
            "camera_lookat_world_m": LOW_CAMERA_TARGET.tolist(), "physics_steps": 0,
            "renderer": renderer, "observed_entity_points": len(observation["xyz"]),
            "support_visibility": visibility_rows})
        print(json.dumps({"capture": "repeated/"+variant,
                          "evidence_points": [r["evidence_points"] for r in visibility_rows]}), flush=True)
    for tier in range(3):
        name = SUPPORT_NAMES[tier]
        for variant in ("pose", "shape"):
            pairs.append({"id": f"repeated_{name}_{variant}", "group": "shelf_repeated",
                          "variation": f"{name}/{variant}", "source": f"repeated/{name}/source",
                          "target": f"repeated/{name}/{variant}", "source_evidence_mean_reference": True,
                          "reference_definition": "source observed evidence arithmetic mean mapped by independent canonical transforms",
                          "evidence_mask_role": "given source localization evidence; no target hard boundary"})
    return pairs


def point_visibility(world: np.ndarray, observation: dict, depth: np.ndarray,
                     support: np.ndarray) -> np.ndarray:
    """Check physical support membership and optical depth at projected samples."""
    K, T = observation["K"], observation["T_world_camera_cv"]
    camera = (world-T[:3, 3]) @ T[:3, :3]
    homogeneous = camera @ K.T
    pixels = np.rint(homogeneous[:, :2]/homogeneous[:, 2, None]).astype(int)
    inside = (camera[:, 2] > 0) & np.all((pixels >= 0) & (pixels < depth.shape[::-1]), axis=1)
    image = np.zeros(depth.shape, bool)
    x, y = observation["pixels"][support].T
    image[y, x] = True
    result = np.zeros(len(world), bool)
    x, y = pixels[inside].T
    tolerance = 2*camera[inside, 2]/min(K[0, 0], K[1, 1])
    result[inside] = image[y, x] & (np.abs(depth[y, x]-camera[inside, 2]) <= tolerance)
    return result


def recorded_mask_projection(canonical: np.ndarray, support_mask: np.ndarray,
                             source_transform: np.ndarray, source_observation: dict,
                             source_mask: np.ndarray) -> np.ndarray:
    """Project only the recorded evidence mask for optional shape diagnosis.

    This deliberately preserves its holes. It is not the required output
    region, an admissible placement set, or a target mask supplied to inference.
    """
    world = canonical @ source_transform[:3, :3].T + source_transform[:3, 3]
    T, K = source_observation["T_world_camera_cv"], source_observation["K"]
    camera = (world-T[:3, 3]) @ T[:3, :3]
    homogeneous = camera @ K.T
    pixels = np.rint(homogeneous[:, :2]/homogeneous[:, 2, None]).astype(int)
    inside = (camera[:, 2] > 0) & np.all((pixels >= 0) & (pixels < source_mask.shape[::-1]), axis=1)
    result = np.zeros(len(canonical), bool)
    result[inside] = source_mask[pixels[inside, 1], pixels[inside, 0]]
    return result & support_mask


def import_recorded_source(output: Path, canonical_corners: np.ndarray, original_model,
                           original_data) -> tuple[dict, dict, np.ndarray, np.ndarray, np.ndarray]:
    """Import the full, genuine frame and its unchanged 362-pixel evidence."""
    import mujoco
    from scipy.ndimage import binary_fill_holes

    folder = output / "demo/source"
    folder.mkdir(parents=True, exist_ok=False)
    meta = json.loads((DEMO_SOURCE / "region.json").read_text())
    if int(meta["frame"]) != FRAME:
        raise ValueError("Recorded source frame changed")
    rgb = np.array(Image.open(DEMO_SOURCE / "full_rgb.png").convert("RGB"))
    depth = np.load(DEMO_SOURCE / "depth_m.npy", allow_pickle=False)
    owner = np.array(Image.open(DEMO_SOURCE / "owner_mask.png")) > 0
    evidence = np.array(Image.open(DEMO_SOURCE / "interaction_mask.png")) > 0
    calibration = json.loads((DEMO_SOURCE / "calibration.json").read_text())
    geometry = archive(DEMO_SOURCE / "geometry.npz")
    with h5py.File(RECORDING / "episode.h5", "r") as episode:
        if not np.array_equal(rgb, episode["frames/rgb"][FRAME]) or not np.array_equal(depth, episode["frames/depth_m"][FRAME]):
            raise ValueError("Source images no longer match the recorded frame")
        seg = np.asarray(episode["frames/seg_geom_id"][FRAME])
        # BEGIN retain the frame's full-precision body pose. MuJoCo's exported
        # XML rounds fixed body positions, so XML defaults are not exact data.
        catalog = json.loads((RECORDING / "model_catalog.json").read_text())
        body_id = next(b["id"] for b in catalog["bodies"] if b["name"] == "wooden_shelf_1_main")
        recorded_position = np.asarray(episode["states/body_xpos"][FRAME, body_id])
        recorded_quat = np.asarray(episode["states/body_xquat"][FRAME, body_id])
        recorded_rotation = Rotation.from_quat(recorded_quat[[1, 2, 3, 0]]).as_matrix()
        # END full-precision source frame pose.
    with h5py.File(PREPROCESSED / "masks.h5", "r") as masks:
        segment = masks["segments/segment_002"]
        rows = np.flatnonzero(np.asarray(segment["frame_indices"]) == FRAME)
        if len(rows) != 1 or not np.array_equal(owner, segment["regions/shelf_top_contact_patch/owner_mask"][rows[0]]):
            raise ValueError("Source owner mask no longer matches the preprocessing record")
        legacy = np.asarray(segment["regions/shelf_top_contact_patch/local_mask"][rows[0]], bool)
        visible_support = np.array(Image.open(DEMO_SOURCE / "region_mask.png")) > 0
        if not np.array_equal(evidence, legacy & visible_support & np.isfinite(depth) & (depth > 0)):
            raise ValueError("Source interaction evidence changed after extraction")
    model = mujoco.MjModel.from_xml_path(str(RECORDING / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    original_vertices, recorded_vertices = local_mesh_vertices(original_model, original_data), local_mesh_vertices(model, data)
    if original_vertices.shape != recorded_vertices.shape:
        raise ValueError("Recorded shelf is a different mesh; same-asset truth is unavailable")
    mesh_error = float(cKDTree(original_vertices).query(recorded_vertices)[0].max())
    if mesh_error > 1e-9:
        raise ValueError("Recorded shelf is a different asset; cannot invent a canonical correspondence")
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = geometry["body_rotation"], geometry["body_translation"]
    # BEGIN tie the exported frame to the actually recorded body pose, rather
    # than trusting a separately saved matrix or rounded XML default by name.
    shelf_body = model.body("wooden_shelf_1_main").id
    if not (np.allclose(transform[:3, :3], recorded_rotation, atol=1e-12, rtol=0)
            and np.allclose(transform[:3, 3], recorded_position, atol=1e-12, rtol=0)):
        raise ValueError("Evidence body transform differs from the recorded frame's actual shelf pose")
    xml_rounding_error = float(np.linalg.norm(recorded_position-data.xpos[shelf_body]))
    # END physical source-frame verification.
    frame = {"rgb": rgb, "depth_m": depth, "seg_geom_id": seg}
    observation = observed_points(frame, calibration, owner)
    if len(observation["xyz"]) != 7711 or int(evidence.sum()) != 362:
        raise ValueError("The reviewed real-source point counts changed")
    base = canonical_truth(observation, model, frame, transform)
    corners = world_supports(canonical_corners, transform)
    truth = scenes.evaluate_visible_points(observation["xyz"], corners[3])
    # BEGIN recorded evidence has no artificial rectangular boundary.
    for key in ("region_corners", "normalized_region_bounds"):
        del truth[key]
    # END recorded evidence boundary semantics.
    truth.update(base)
    truth = augment_support_truth(observation, truth, corners, 3)
    x, y = observation["pixels"].T
    truth["region_mask"] = evidence[y, x]
    if not np.all(~truth["region_mask"] | truth["support_mask"]):
        raise ValueError("Recorded interaction evidence leaves the actual exterior support")
    mean = observation["xyz"][truth["region_mask"]].mean(0)
    truth["region_center"] = mean
    canonical_mean = mean @ base["world_to_canonical"][:3, :3].T + base["world_to_canonical"][:3, 3]
    # First-order pixel-area quadrature is a sampling-density diagnostic. The
    # arithmetic mean remains the declared reference used by all methods.
    pixels = observation["pixels"][truth["region_mask"]]
    K, T = observation["K"], observation["T_world_camera_cv"]
    normal_cv = truth["support_normal"] @ T[:3, :3]
    rays = np.column_stack([pixels, np.ones(len(pixels))]) @ np.linalg.inv(K).T
    incidence = np.abs(rays @ normal_cv)
    if np.any(incidence < 1e-6):
        raise ValueError("Recorded support is too grazing for pixel-area quadrature")
    weights = depth[pixels[:, 1], pixels[:, 0]]**2/(K[0, 0]*K[1, 1]*incidence)
    weights /= weights.sum()
    area_mean = np.average(observation["xyz"][truth["region_mask"]], axis=0, weights=weights)
    center_visible = point_visibility(mean[None], observation, depth, truth["support_mask"])[0]
    truth.update(region_visible_area_fraction=np.array(1.), region_center_visible=np.array(center_visible),
                 source_evidence_area_quadrature_center=area_mean,
                 source_evidence_density_centroid_shift_m=np.array(np.linalg.norm(area_mean-mean)))
    shutil.copy2(DEMO_SOURCE / "full_rgb.png", folder / "rgb.png")
    shutil.copy2(DEMO_SOURCE / "depth_m.npy", folder / "depth_m.npy")
    shutil.copy2(DEMO_SOURCE / "calibration.json", folder / "calibration.json")
    np.save(folder / "seg_geom_id.npy", seg, allow_pickle=False)
    np.savez_compressed(folder / "observation.npz", **observation)
    np.savez_compressed(folder / "evaluation.npz", **truth)
    save_overlay(folder, observation, truth)
    scenes.write_json(folder / "annotation.json", {"recording": str(RECORDING), "frame": FRAME,
        "extraction_scripts": [str(DEMO_SOURCE.parents[1] / "extract.py"), str(DEMO_SOURCE.parents[1] / "bind.py")],
        "source_RGB_and_depth_equal_recording": True, "source_owner_equal_preprocessed": True,
        "source_interaction_mask_unchanged": True, "owner_pixels": len(observation["xyz"]),
        "interaction_pixels": int(evidence.sum()), "internal_hole_pixels_preserved": int(binary_fill_holes(evidence).sum()-evidence.sum()),
        "same_asset_vertex_count": len(original_vertices), "same_asset_max_vertex_error_m": mesh_error,
        "evidence_body_transform_equals_recorded_frame": True,
        "recording_XML_fixed_body_translation_rounding_m": xml_rounding_error,
        "reference_definition": "arithmetic mean of all 362 actually observed evidence points",
        "sampling_density_centroid_shift_m": float(np.linalg.norm(area_mean-mean)),
        "whole_entity_segmentation": "recorded simulator owner mask; supplied input",
        "historical_scene_container_exclusivity": False,
        "historical_context": "Unmodified genuine recording contains both shelf and drawer; all newly tested target scenes contain shelf only",
        "region_mask_role": "localization evidence, not a target hard boundary or admissible placement set"})
    return observation, truth, evidence, canonical_mean, weights


def export_demo_targets(output: Path, canonical_corners: np.ndarray, source: dict,
                        source_truth: dict, evidence: np.ndarray,
                        canonical_mean: np.ndarray, weights: np.ndarray) -> list[dict]:
    """Reuse frozen original-view targets, with independently mapped evidence."""
    pairs = []
    source_points = source_truth["canonical_xyz"][source_truth["region_mask"]]
    for variant in scenes.VARIANTS:
        label = "original" if variant == "source" else variant
        original = OLD_DATA / "shelf" / variant / "view_a"
        folder = output / "demo" / ("target_"+label)
        copy_view(original, folder)
        observation = archive(folder / "observation.npz")
        prior = archive(original / "evaluation.npz")
        transform = prior["canonical_to_world"]
        corners = world_supports(canonical_corners, transform)
        truth = scenes.evaluate_visible_points(observation["xyz"], corners[3])
        # BEGIN the projected recorded mask is not the synthetic rectangle.
        for key in ("region_corners", "normalized_region_bounds"):
            del truth[key]
        # END recorded target diagnostic boundary semantics.
        truth.update({key: prior[key] for key in ("canonical_xyz", "world_to_canonical", "canonical_to_world", "surface_id", "surface_names")})
        if not np.array_equal(truth["surface_names"], source_truth["surface_names"]):
            raise ValueError("Source and target oracle owner identity maps differ")
        truth = augment_support_truth(observation, truth, corners, 3)
        truth["region_mask"] = recorded_mask_projection(truth["canonical_xyz"], truth["support_mask"],
                                                       source_truth["canonical_to_world"], source, evidence)
        truth["region_center"] = canonical_mean @ transform[:3, :3].T + transform[:3, 3]
        sampled = source_points @ transform[:3, :3].T + transform[:3, 3]
        depth = np.load(folder / "depth_m.npy", allow_pickle=False)
        visible = point_visibility(np.vstack([sampled, truth["region_center"]]), observation, depth, truth["support_mask"])
        # BEGIN keep a probability in [0, 1] despite floating-point sum error.
        truth.update(region_visible_area_fraction=np.array(np.clip(np.sum(weights*visible[:-1]), 0., 1.)),
                     region_center_visible=np.array(visible[-1]))
        # END bounded visibility fraction.
        if not visible[-1]:
            raise ValueError("Recorded evidence's reference centre is hidden in the target")
        np.savez_compressed(folder / "evaluation.npz", **truth)
        save_overlay(folder, observation, truth)
        scenes.write_json(folder / "annotation.json", {"source_observation": str(original),
            "input_files_copied_without_change": True, "furniture_types_present": ["shelf"],
            "reference_definition": "recorded source evidence arithmetic mean mapped by independently verified same-asset transform",
            "region_mask_role": "projected recorded mask for diagnostic display only; not required neighborhood output",
            "preferred_support_id": 3, "support_names": SUPPORT_NAMES,
            "visible_source_evidence_area_fraction": float(truth["region_visible_area_fraction"]),
            "reference_center_visible": bool(visible[-1])})
        pairs.append({"id": "demo_"+label, "group": "put_mug_on_shelf_recorded",
                      "variation": label, "source": "demo/source", "target": "demo/target_"+label,
                      "source_evidence_mean_reference": True,
                      "reference_definition": "all 362 observed source evidence points arithmetic mean, mapped by independent same-asset canonical transform",
                      "evidence_mask_role": "original ring is localization evidence, not required target shape",
                      "same_asset_verified": True})
    return pairs


def overview(output: Path, pairs: list[dict]) -> None:
    """Show newly added experiments; original comparison images remain intact."""
    selected = [pair for pair in pairs if pair["id"].startswith(("repeated_", "demo_"))]
    sheet = Image.new("RGB", (1280, len(selected)*515), "white")
    draw = ImageDraw.Draw(sheet)
    for row, pair in enumerate(selected):
        draw.text((10, row*515+10), pair["id"]+" / blue=support; red=source evidence or diagnostic projection", fill="black")
        for column, role in enumerate(("source", "target")):
            image = Image.open(output / pair[role] / "evaluation-overlay.png")
            sheet.paste(image, (column*640, row*515+35))
    sheet.save(output / "new-pairs-overview.png")


# BEGIN independent export audit: these assertions operate on saved artifacts,
# not on live render arrays, and none of the inspected labels reaches inference.
def validate(output: Path) -> dict:
    """Audit every saved view, pair, reused input, and independent label map.

    Camera JSON is the explicit rendering authority in this project. Named XML
    cameras remain unchanged but were not used for original view A/B either;
    equating them to a separately configured fixed camera would be false.
    """
    import xml.etree.ElementTree as ET

    def require(condition: bool, description: str) -> None:
        if not condition:
            raise ValueError("Dataset audit failed: " + description)

    def equal_archive(first: Path, second: Path) -> bool:
        a, b = archive(first), archive(second)
        return a.keys() == b.keys() and all(a[k].dtype == b[k].dtype and
                                          np.array_equal(a[k], b[k]) for k in a)

    manifest = json.loads((output / "manifest.json").read_text())
    pairs = manifest["pairs"]
    require(len(pairs) == 15 and len({p["id"] for p in pairs}) == 15, "15 unique pair IDs")
    fields = {"xyz", "pixels", "object_mask", "K", "T_world_camera_cv"}
    views = []
    for path in sorted(output.rglob("evaluation.npz")):
        folder = path.parent
        name = str(folder.relative_to(output))
        obs, truth = archive(folder / "observation.npz"), archive(path)
        require(set(obs) == fields, name + " observation field allowlist")
        xyz, pixels, K, T = (obs[k] for k in ("xyz", "pixels", "K", "T_world_camera_cv"))
        n = len(xyz)
        require(xyz.shape == (n, 3) and np.isfinite(xyz).all(), name + " finite Nx3 points")
        require(pixels.shape == (n, 2) and pixels.dtype.kind in "iu", name + " integer Nx2 pixels")
        depth = np.load(folder / "depth_m.npy", allow_pickle=False)
        require(obs["object_mask"].shape == depth.shape and obs["object_mask"].dtype == np.bool_, name + " entity image mask")
        y, x = np.nonzero(obs["object_mask"])
        require(np.array_equal(pixels, np.column_stack([x, y])), name + " exact point/mask order")
        require(np.isfinite(depth[y, x]).all() and np.all(depth[y, x] > 0), name + " finite positive optical depth")
        require(K.shape == (3, 3) and T.shape == (4, 4), name + " calibration shape")
        calibration = json.loads((folder / "calibration.json").read_text())
        require(all(np.array_equal(obs[k], np.asarray(calibration[k])) for k in ("K", "T_world_camera_cv")), name + " saved calibration equals observation")
        rays = np.column_stack([pixels, np.ones(n)]) @ np.linalg.inv(K).T
        expected = rays*depth[y, x, None] @ T[:3, :3].T + T[:3, 3]
        back_error = float(np.abs(expected-xyz).max())
        camera = (xyz-T[:3, 3]) @ T[:3, :3]
        projected = camera @ K.T
        projection_error = float(np.abs(projected[:, :2]/projected[:, 2, None]-pixels).max())
        require(back_error < 1e-12 and projection_error < 1e-8, name + " RGB-D projection roundtrip")
        support, preferred = truth["support_id"], truth["preferred_support_id"]
        require(support.shape == (n,) and support.dtype.kind in "iu", name + " support label alignment")
        require(preferred.shape == () and preferred.dtype.kind in "iu", name + " scalar preferred support")
        corners = truth["support_corners_all"]
        require(corners.ndim == 3 and corners.shape[1:] == (4, 3), name + " four physical support corners")
        require(0 <= int(preferred) < len(corners) and np.all((support >= -1) & (support < len(corners))), name + " label range")
        require(np.array_equal(truth["support_mask"], support == int(preferred)), name + " preferred support mask")
        require(truth["region_mask"].shape == (n,) and truth["region_mask"].dtype == np.bool_, name + " evidence alignment")
        require(np.all(~truth["region_mask"] | truth["support_mask"]), name + " evidence subset of preferred support")
        for index, plane in enumerate(corners):
            normal = np.cross(plane[1]-plane[0], plane[3]-plane[0])
            normal /= np.linalg.norm(normal)
            points = xyz[support == index]
            if len(points):
                require(np.abs((points-plane[0]) @ normal).max() <= .0004000001, name + " support plane residual")
        forward, inverse = truth["canonical_to_world"], truth["world_to_canonical"]
        require(forward.shape == inverse.shape == (4, 4), name + " canonical map shapes")
        require(np.allclose(forward[3], [0, 0, 0, 1], atol=1e-12, rtol=0) and
                np.allclose(inverse[3], [0, 0, 0, 1], atol=1e-12, rtol=0), name + " affine final row")
        inverse_error = float(max(np.abs(forward @ inverse-np.eye(4)).max(), np.abs(inverse @ forward-np.eye(4)).max()))
        canonical = xyz @ inverse[:3, :3].T + inverse[:3, 3]
        restored = truth["canonical_xyz"] @ forward[:3, :3].T + forward[:3, 3]
        point_error = float(max(np.abs(canonical-truth["canonical_xyz"]).max(), np.abs(restored-xyz).max()))
        require(inverse_error < 1e-12 and point_error < 1e-12, name + " inverse and point maps")
        require(truth["surface_id"].shape == (n,) and np.all((truth["surface_id"] >= 0) & (truth["surface_id"] < len(truth["surface_names"]))), name + " stable geometry owner IDs")
        if (folder / "camera.json").exists():
            config = json.loads((folder / "camera.json").read_text())
            quat = np.asarray(config["quaternion_world_camera_gl_wxyz"])
            rotation = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix() @ np.diag([1, -1, -1])
            require(np.allclose(T[:3, 3], config["position_world_m"], atol=1e-12, rtol=0) and
                    np.allclose(T[:3, :3], rotation, atol=1e-12, rtol=0), name + " camera JSON pose")
        else:
            require(name == "demo/source", name + " recording-only calibration exception")
            recorded = json.loads((RECORDING / "camera_calibration.json").read_text())
            require(all(np.array_equal(obs[k], np.asarray(recorded[k])) for k in ("K", "T_world_camera_cv")), name + " original recording calibration")
        row = {"view": name, "points": n, "evidence_points": int(truth["region_mask"].sum()),
               "support_points_by_id": [int((support == i).sum()) for i in range(len(corners))],
               "projection_error_pixels": projection_error, "backprojection_error_m": back_error,
               "canonical_inverse_error": inverse_error, "canonical_point_error_m": point_error}
        if name.startswith("repeated/"):
            visibility = scenes.patch_visibility(truth, pixels, depth, K, T)
            require(bool(visibility["region_center_visible"]) and row["evidence_points"] >= 12, name + " internal query observable")
            require(abs(visibility["region_visible_area_fraction"]-float(truth["region_visible_area_fraction"])) < 1e-12, name + " stored visibility")
            row.update(visibility)
        if name.startswith("legacy/"):
            old = OLD_DATA / folder.relative_to(output / "legacy")
            require(equal_archive(folder / "observation.npz", old / "observation.npz"), name + " frozen observation identity")
            require(np.array_equal(np.array(Image.open(folder / "rgb.png")), np.array(Image.open(old / "rgb.png"))), name + " frozen RGB identity")
            require(all(np.array_equal(np.load(folder / f, allow_pickle=False), np.load(old / f, allow_pickle=False)) for f in ("depth_m.npy", "seg_geom_id.npy")), name + " frozen depth and owner identity")
            row["all_reused_inputs_equal_frozen_data"] = True
        views.append(row)
    pair_rows = []
    for pair in pairs:
        source_folder, target_folder = (output / pair[role] for role in ("source", "target"))
        require(source_folder.is_relative_to(output) and target_folder.is_relative_to(output), pair["id"] + " internal references")
        source, target = archive(source_folder / "observation.npz"), archive(target_folder / "observation.npz")
        a, b = archive(source_folder / "evaluation.npz"), archive(target_folder / "evaluation.npz")
        require(a["region_mask"].sum() >= 12, pair["id"] + " nonempty source evidence")
        require(np.array_equal(a["surface_names"], b["surface_names"]), pair["id"] + " stable source/target owner names")
        mean = source["xyz"][a["region_mask"]].mean(0)
        mapping = b["canonical_to_world"] @ a["world_to_canonical"]
        reference = mean @ mapping[:3, :3].T + mapping[:3, 3]
        depth = np.load(target_folder / "depth_m.npy", allow_pickle=False)
        visible = bool(point_visibility(reference[None], target, depth, b["support_mask"])[0])
        require(visible, pair["id"] + " mapped source-mean reference visible on preferred support")
        if pair["id"].startswith("repeated_"):
            require(all(np.array_equal(source[k], target[k]) for k in ("K", "T_world_camera_cv")), pair["id"] + " identical actual fixed camera")
        pair_rows.append({"id": pair["id"], "source": pair["source"], "target": pair["target"],
                          "source_evidence_points": int(a["region_mask"].sum()),
                          "target_reference_evidence_center": reference.tolist(), "reference_center_visible": visible})
    original_tree = ET.parse(scenes.ORIGINAL / "scene.xml").getroot()
    original_cameras = {e.get("name"): dict(e.attrib) for e in original_tree.iter("camera")}
    xml_rows = []
    for scene_path in sorted((output / "legacy").glob("*/*/scene.xml")) + sorted((output / "captures/repeated").glob("*/scene.xml")):
        tree = ET.parse(scene_path).getroot()
        cameras = {e.get("name"): dict(e.attrib) for e in tree.iter("camera")}
        present = [kind for kind, f in scenes.FIXTURES.items() if tree.find(f"worldbody/body[@name='{f['root']}']") is not None]
        require(cameras == original_cameras and len(present) == 1, str(scene_path) + " preserved XML cameras and exclusive furniture")
        if scene_path.parent.parent.name == "repeated":
            old = OLD_DATA / "shelf" / scene_path.parent.name
            require(scene_path.read_bytes() == (old / "scene.xml").read_bytes() and equal_archive(scene_path.parent / "state.npz", old / "state.npz"), str(scene_path) + " frozen scene and state")
        xml_rows.append({"scene": str(scene_path.relative_to(output)), "furniture_types_present": present,
                         "named_XML_cameras_equal_original": True})
    recorded = json.loads((output / "demo/source/annotation.json").read_text())
    require(recorded["source_RGB_and_depth_equal_recording"] and recorded["source_owner_equal_preprocessed"] and recorded["source_interaction_mask_unchanged"], "real-source provenance")
    require(recorded["interaction_pixels"] == 362 and recorded["internal_hole_pixels_preserved"] == 79 and recorded["same_asset_max_vertex_error_m"] <= 1e-9, "unchanged recorded evidence and known same asset")
    result = {"status": "passed", "dataset": str(output), "pairs": len(pairs), "observations": len(views),
              "new_pair_ids": [p["id"] for p in pairs if p["id"].startswith(("repeated_", "demo_"))],
              "checks": ["all pair references and unique IDs", "strict observation field allowlist", "pixel/point/mask alignment and optical-depth backprojection", "evidence subset of preferred physical support", "canonical 4x4 inverses and point maps", "all low-camera tier queries and primary mapped means visible", "all 18 legacy RGB/depth/owner/observations value-identical", "stable source/target geometry owner identities", "independent fixed-camera calibration and unchanged XML cameras", "exclusive generated furniture and unchanged whole-shelf scenes", "recorded source 362 pixels and 79 hole pixels retained"],
              "camera_authority": "External fixed-camera JSON defines each rendered view; unchanged named XML cameras are a separate preserved scene property. Recorded source uses its original camera_calibration.json.",
              "views": views, "pair_references": pair_rows, "scenes": xml_rows, "recorded_source": recorded}
    scenes.write_json(output.parent / "data-neighborhood-validation.json", result)
    return result
# END independent saved-artifact audit.


def generate(output: Path) -> None:
    """Produce 15 pairs in a fresh directory without rewriting previous runs."""
    from scene import load_scene

    if output.exists():
        raise FileExistsError(f"Evidence directory already exists: {output}")
    output.mkdir(parents=True)
    model, data = load_scene("original")
    corners = annotated_shelf_supports(model, data)
    old_visibility = old_camera_visibility(corners)
    scenes.write_json(output / "original-camera-interior-visibility.json", old_visibility)
    pairs = copy_legacy_suite(output, corners)
    pairs.extend(render_repeated_shelf(output, corners))
    source, source_truth, evidence, mean, weights = import_recorded_source(output, corners, model, data)
    pairs.extend(export_demo_targets(output, corners, source, source_truth, evidence, mean, weights))
    scenes.write_json(output / "manifest.json", {"purpose": "interaction-location preference and visible metric neighborhood candidates",
        "inference_observation_fields": ["xyz", "pixels", "object_mask", "K", "T_world_camera_cv"],
        "evaluation_contract": {"support_id": "point-aligned int; -1 outside annotated physical support surfaces",
            "preferred_support_id": "scalar int, evaluator's intended physical surface",
            "region_mask": "source localization evidence or retained target evidence diagnostic; never a required target hard boundary",
            "region_center": "declared physical/evidence reference; primary mapped-source-mean reference is reported separately"},
        "support_names_shelf": SUPPORT_NAMES, "fixed_camera_for_internal_queries": low_camera_config(),
        "canonical_shelf_support_corners": corners.tolist(), "pairs": pairs,
        "old_data_preserved": str(OLD_DATA), "feature_reuse": "original six pair IDs retain identical model input observations; new pairs require extraction",
        "historical_demo_exclusivity": "historical RGB retains both containers; every target and newly rendered shelf scene contains only shelf",
        "no_robot_actions": True})
    overview(output, pairs)
    # BEGIN every generated dataset must pass its saved-artifact audit.
    validate(output)
    # END mandatory dataset export audit.
    print(json.dumps({"pairs": len(pairs), "new_pairs": 9, "new_rendered_views": 3,
                      "source_evidence_points": int(source_truth["region_mask"].sum())}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    generate(parser.parse_args().out.resolve())
