"""Export full RGB-D inputs for the independent category-segmentation study.

This is a fixture and evaluation exporter, not a perception component. It owns
the deliberately chosen assets, category prompts and controlled interventions.
SAM and registration may read only rgb.png, depth_m.npy, calibration.json and
the independently supplied source evidence. Scene IDs, simulator owners,
support labels and known transforms are confined to evaluation artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, label
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from . import scenes
from . import neighborhood_scenes as neighborhood


# BEGIN independent category-segmentation fixtures; original suites are read-only.
DEFAULT_OUTPUT = scenes.DEFAULT_OUTPUT.parent / "data-segmentation"
VARIANTS = {
    "source": {"scale": [1., 1., 1.], "translation": [0., 0., 0.], "yaw_deg": 0.},
    "shape": {"scale": [1.18, .85, 1.], "translation": [0., 0., 0.], "yaw_deg": 0.},
    "translation": {"scale": [1., 1., 1.], "translation": [.04, -.025, 0.], "yaw_deg": 0.},
    "yaw": {"scale": [1., 1., 1.], "translation": [0., 0., 0.], "yaw_deg": 12.},
    "color": {"scale": [1., 1., 1.], "translation": [0., 0., 0.], "yaw_deg": 0.},
}
COLOR_MULTIPLIER = np.array([.55, .8, 1.])
RECORDS = {
    "demo_mug": {"kind": "shelf", "frame": 197, "segment": "segment_002",
                 "region": "shelf_top_contact_patch", "preferred": 3,
                 "semantic_skill": "put_mug_on_shelf", "pixels": 362, "holes": 79},
    "demo_block": {"kind": "drawer", "frame": 1003, "segment": "segment_006",
                   "region": "drawer_interior_contact_patch", "preferred": 0,
                   "semantic_skill": "put_block_in_drawer", "pixels": 238, "holes": 0},
}


def archive(path: Path) -> dict:
    """Never enable object deserialization when reading numeric artifacts."""
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def drawer_opening() -> float:
    """Use the actual recorded interaction state, rather than a rounded value."""
    catalog = json.loads((neighborhood.RECORDING / "model_catalog.json").read_text())
    joint = next(j for j in catalog["joints"] if j["name"] == "white_cabinet_1_top_level")
    with h5py.File(neighborhood.RECORDING / "episode.h5", "r") as episode:
        return float(episode["states/qpos"][1003, joint["qpos_start"]])


def affine_transform(model, data, kind: str, scale: np.ndarray) -> np.ndarray:
    """Map the unscaled root frame to world; aligned joints obey the same map."""
    root = model.body(scenes.FIXTURES[kind]["root"]).id
    transform = np.eye(4)
    transform[:3, :3] = data.xmat[root].reshape(3, 3) @ np.diag(scale)
    transform[:3, 3] = data.xpos[root]
    return transform


def canonical_supports(model, data, kind: str) -> tuple[np.ndarray, tuple[str, ...]]:
    """Read actual support geometry at the unscaled source joint configuration."""
    fixture = scenes.FIXTURES[kind]
    if kind == "shelf":
        local = neighborhood.annotated_shelf_supports(model, data)
        names = neighborhood.SUPPORT_NAMES
    else:
        local = scenes.support_corners(model, data, kind)[None]
        names = ("drawer_floor" if kind == "drawer" else "cabinet_floor",)
    support_body = model.body(fixture["support_body"]).id
    world = local @ data.xmat[support_body].reshape(3, 3).T + data.xpos[support_body]
    inverse = np.linalg.inv(affine_transform(model, data, kind, np.ones(3)))
    return world @ inverse[:3, :3].T + inverse[:3, 3], names


def tint_actor(tree: ET.Element, actor: ET.Element) -> list[dict]:
    """Change rendered color without sharing edits across material users.

    MuJoCo's default geom rgba delegates color to its material. Resolve that
    effective color first, then write a geom rgba override. Material texture,
    alpha, reflectance, specular and shininess otherwise remain untouched.
    This happens before rendering; no pixel mask participates in the change.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(ET.tostring(tree, encoding="unicode"))
    changes = []
    default = np.array([.5, .5, .5, 1.], dtype=model.geom_rgba.dtype)
    for element in actor.iter("geom"):
        geom = model.geom(element.get("name"))
        rgba = model.geom_rgba[geom.id].copy()
        material = int(model.geom_matid[geom.id])
        if material >= 0 and np.array_equal(rgba, default):
            rgba = model.mat_rgba[material].copy()
        tinted = rgba.copy()
        tinted[:3] *= COLOR_MULTIPLIER
        element.set("rgba", scenes.numbers(tinted))
        changes.append({"geom": element.get("name"), "effective_rgba_before": rgba.tolist(),
                        "rgba_after": tinted.tolist(), "material_id": material})
    return changes


def prepare_case(output: Path, kind: str, variant: str, original_model, original_data,
                 opening: float, params: dict):
    """Build one exclusive container and restore original robot state by name."""
    import mujoco

    tree, audit = scenes.build_xml(kind, "source")
    actor = tree.find(f"worldbody/body[@name='{scenes.FIXTURES[kind]['root']}']")
    # BEGIN explicit fixture parameters allow independently declared interventions.
    scale = np.asarray(params["scale"])
    # END explicit fixture parameters.
    scenes.scale_actor(tree, actor, scale)
    actor.set("pos", scenes.numbers(np.fromstring(actor.get("pos"), sep=" ")+params["translation"]))
    quat = Rotation.from_euler("z", scenes.FIXTURES[kind]["base_yaw_deg"]+params["yaw_deg"], degrees=True).as_quat()
    actor.set("quat", scenes.numbers(quat[[3, 0, 1, 2]]))
    colors = tint_actor(tree, actor) if variant == "color" else []
    folder = output / "evaluation-scenes" / f"{kind}_{variant}"
    folder.mkdir(parents=True)
    ET.indent(tree)
    ET.ElementTree(tree).write(folder / "scene.xml", encoding="unicode")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = scenes.restore_named_state(model, original_model, original_data)
    fixture = scenes.FIXTURES[kind]
    if fixture["joint"]:
        joint = model.joint(fixture["joint"])
        axis = model.jnt_axis[joint.id]
        if np.count_nonzero(np.abs(axis) > 1e-12) != 1:
            raise ValueError("Fixture requires an axis-aligned slide")
        base_opening = opening if kind == "drawer" else fixture["opening"]
        data.qpos[int(joint.qposadr[0])] = base_opening * np.linalg.norm(scale*axis)
        data.qvel[int(joint.dofadr[0])] = 0.
    if kind == "drawer":
        for name in ("white_cabinet_1_middle_level", "white_cabinet_1_bottom_level"):
            joint = model.joint(name)
            data.qpos[int(joint.qposadr[0])] = 0.
            data.qvel[int(joint.dofadr[0])] = 0.
    mujoco.mj_forward(model, data)
    for name in ("robot0_base", "robot0_link7", "gripper0_right_rightfinger", "gripper0_right_leftfinger"):
        old, new = original_model.body(name).id, model.body(name).id
        if not (np.allclose(data.xpos[new], original_data.xpos[old], atol=1e-12, rtol=0)
                and np.allclose(data.xmat[new], original_data.xmat[old], atol=1e-12, rtol=0)):
            raise ValueError("Unrelated robot kinematics changed")
    spec = int(mujoco.mjtState.mjSTATE_INTEGRATION)
    state = np.empty(mujoco.mj_stateSize(model, spec))
    mujoco.mj_getState(model, data, state, spec)
    np.savez_compressed(folder / "state.npz", state_spec=spec, integration_state=state)
    audit.update(variant=variant, parameters=params, color_multiplier=COLOR_MULTIPLIER.tolist(),
                 color_changes=colors, drawer_recorded_opening_m=opening if kind == "drawer" else None,
                 segmentation_input="full RGB and one category prompt; no simulator owner input")
    # The old fixture's -0.14 opening must not survive in new audit metadata.
    audit.pop("support_annotation", None)
    scenes.write_json(folder / "audit.json", audit)
    return model, data


def truth_images(model, frame: dict, calibration: dict, kind: str,
                 transform: np.ndarray, corners: np.ndarray, names: tuple[str, ...]) -> dict:
    """Label only real furniture pixels, including finite support membership.

    A same-height table pixel from an imperfect SAM mask remains -1. Negative
    segmentation IDs are guarded before indexing, preventing background from
    accidentally acquiring the last geometry's semantic label.
    """
    seg, depth = np.asarray(frame["seg_geom_id"]), np.asarray(frame["depth_m"])
    geom_ids = scenes.descendant_geoms(model, scenes.FIXTURES[kind]["root"])
    entity = np.isin(seg, geom_ids) & np.isfinite(depth) & (depth > 0)
    observed = neighborhood.observed_points(frame, calibration, entity)
    support = np.full(depth.shape, -1, np.int32)
    x, y = observed["pixels"].T
    support[y, x] = neighborhood.support_labels(observed["xyz"], corners)
    surface_names = sorted(model.geom(int(i)).name for i in geom_ids)
    lookup = np.full(model.ngeom, -1, np.int32)
    for index, name in enumerate(surface_names):
        lookup[model.geom(name).id] = index
    surfaces = np.full(depth.shape, -1, np.int32)
    surfaces[entity] = lookup[seg[entity]]
    return {"entity_mask": entity, "support_id_image": support,
            "surface_id_image": surfaces, "surface_names": np.asarray(surface_names),
            "canonical_to_world": transform, "world_to_canonical": np.linalg.inv(transform),
            "support_corners_all": corners, "support_names": np.asarray(names),
            "raw_geom_id_image": seg}


def save_view(output: Path, view_id: str, kind: str, frame: dict, calibration: dict,
              truth: dict, audit: dict) -> dict:
    """Keep full camera observations separate from evaluation-only pixel truth."""
    folder = output / "views" / view_id
    folder.mkdir(parents=True)
    Image.fromarray(frame["rgb"]).save(folder / "rgb.png")
    np.save(folder / "depth_m.npy", frame["depth_m"], allow_pickle=False)
    scenes.write_json(folder / "calibration.json", calibration)
    np.savez_compressed(folder / "truth.npz", **truth)
    scenes.write_json(folder / "evaluation-audit.json", audit)
    return {"path": str(folder.relative_to(output)), "prompt": kind}


def capture_view(output: Path, model, data, kind: str, variant: str, view: str,
                 canonical: np.ndarray, names: tuple[str, ...], scale: np.ndarray) -> tuple[str, dict]:
    """Render the entire original-derived scene from a fixed external camera."""
    from collect.camera import FixedCamera

    # BEGIN explicit camera selection: view_b must use its own saved pose;
    # unknown view names must never silently render view_a under a new label.
    if view == "low":
        config = neighborhood.low_camera_config()
        calibration_view = "view_a"  # The low fixture shares these intrinsics.
    elif view in scenes.VIEWS:
        config = json.loads((scenes.ORIGINAL / view / "camera.json").read_text())
        calibration_view = view
    else:
        raise ValueError(f"Unknown fixture camera view: {view}")
    # END explicit saved-camera selection.
    camera = FixedCamera(model, data, 640, 480, camera_name=None, camera_config=config)
    try:
        frame, calibration, renderer = camera.capture(data), camera.calibration, camera.renderer_info
    finally:
        camera.close()
    # BEGIN check the selected camera's calibration, including its own pose.
    original = json.loads((scenes.ORIGINAL / calibration_view / "calibration.json").read_text())
    if not np.allclose(calibration["K"], original["K"], atol=1e-9, rtol=0):
        raise ValueError("Camera intrinsics changed")
    if view in scenes.VIEWS and not np.allclose(calibration["T_world_camera_cv"], original["T_world_camera_cv"], atol=1e-9, rtol=0):
        raise ValueError("Original camera pose changed")
    # END selected-camera calibration check.
    # BEGIN use the declared scale, independent of fixture naming.
    transform = affine_transform(model, data, kind, np.asarray(scale))
    # END explicit scale for evaluator geometry.
    corners = neighborhood.world_supports(canonical, transform)
    truth = truth_images(model, frame, calibration, kind, transform, corners, names)
    view_id = f"{kind}_{variant}_{view}"
    record = save_view(output, view_id, kind, frame, calibration, truth,
                       {"kind": kind, "variant": variant, "view": view, "camera": config,
                        "renderer": renderer, "furniture_types_present": [kind],
                        "physics_steps": 0, "entity_pixels": int(truth["entity_mask"].sum()),
                        "support_pixels": {name: int((truth["support_id_image"] == index).sum())
                                           for index, name in enumerate(names)}})
    print(json.dumps({"view": view_id, "entity_pixels": int(truth["entity_mask"].sum())}), flush=True)
    return view_id, record


def geom_body_vertices(model, data, geom_id: int) -> np.ndarray:
    """Recover visual mesh vertices in their own body, independent of root pose."""
    mesh_id = int(model.geom_dataid[geom_id])
    vertices = model.mesh_vert[model.mesh_vertadr[mesh_id]:model.mesh_vertadr[mesh_id]+model.mesh_vertnum[mesh_id]]
    world = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    body = int(model.geom_bodyid[geom_id])
    return (world-data.xpos[body]) @ data.xmat[body].reshape(3, 3)


def verify_recorded_asset(recorded_model, recorded_data, fixture_model, fixture_data,
                          kind: str) -> list[dict]:
    """Require the same visual meshes before declaring a canonical correspondence."""
    import mujoco

    rows = []
    for geom_id in scenes.descendant_geoms(recorded_model, scenes.FIXTURES[kind]["root"]):
        if recorded_model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH or recorded_model.geom_group[geom_id] == 0:
            continue
        name = recorded_model.geom(int(geom_id)).name
        target_id = fixture_model.geom(name).id
        source = geom_body_vertices(recorded_model, recorded_data, int(geom_id))
        target = geom_body_vertices(fixture_model, fixture_data, target_id)
        if source.shape != target.shape:
            raise ValueError(f"Recorded and fixture meshes differ: {name}")
        error = float(max(cKDTree(target).query(source)[0].max(), cKDTree(source).query(target)[0].max()))
        if error > 1e-9:
            raise ValueError(f"Recorded and fixture meshes differ: {name}, {error}")
        rows.append({"geom": name, "vertices": len(source), "max_error_m": error})
    if not rows:
        raise ValueError("No visual mesh was checked")
    return rows


def import_record(output: Path, query: str, canonical: np.ndarray, names: tuple[str, ...],
                  fixture_model, fixture_data) -> tuple[str, dict, np.ndarray]:
    """Import original recording pixels and the existing local interaction mask."""
    import mujoco

    record = RECORDS[query]
    kind, frame_index = record["kind"], record["frame"]
    catalog = json.loads((neighborhood.RECORDING / "model_catalog.json").read_text())
    root_id = next(b["id"] for b in catalog["bodies"] if b["name"] == scenes.FIXTURES[kind]["root"])
    with h5py.File(neighborhood.RECORDING / "episode.h5", "r") as episode:
        frame = {key: np.asarray(episode[f"frames/{key}"][frame_index])
                 for key in ("rgb", "depth_m", "seg_geom_id")}
        position = np.asarray(episode["states/body_xpos"][frame_index, root_id])
        quat = np.asarray(episode["states/body_xquat"][frame_index, root_id])
        recorded_qpos = np.asarray(episode["states/qpos"][frame_index])
    calibration = json.loads((neighborhood.RECORDING / "camera_calibration.json").read_text())
    with h5py.File(neighborhood.PREPROCESSED / "masks.h5", "r") as masks:
        segment = masks[f"segments/{record['segment']}"]
        rows = np.flatnonzero(np.asarray(segment["frame_indices"]) == frame_index)
        if len(rows) != 1:
            raise ValueError("Recording evidence has no unique selected frame")
        evidence = np.asarray(segment[f"regions/{record['region']}/local_mask"][rows[0]], bool)
    # The prior mug experiment already bound raw contact evidence to its actual
    # visible shelf surface. Preserve that published mask, including its holes.
    if query == "demo_mug":
        bound = np.asarray(Image.open(neighborhood.DEMO_SOURCE / "interaction_mask.png")) > 0
        visible = np.asarray(Image.open(neighborhood.DEMO_SOURCE / "region_mask.png")) > 0
        expected = evidence & visible & np.isfinite(frame["depth_m"]) & (frame["depth_m"] > 0)
        if not np.array_equal(bound, expected):
            raise ValueError("Published mug evidence no longer matches its original extraction")
        evidence = bound
    if int(evidence.sum()) != record["pixels"] or int(binary_fill_holes(evidence).sum()-evidence.sum()) != record["holes"]:
        raise ValueError("Existing evidence pixels or holes changed")
    model = mujoco.MjModel.from_xml_path(str(neighborhood.RECORDING / "scene.xml"))
    data = mujoco.MjData(model)
    data.qpos[:] = recorded_qpos
    mujoco.mj_forward(model, data)
    meshes = verify_recorded_asset(model, data, fixture_model, fixture_data, kind)
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(quat[[1, 2, 3, 0]]).as_matrix()
    transform[:3, 3] = position
    corners = neighborhood.world_supports(canonical, transform)
    truth = truth_images(model, frame, calibration, kind, transform, corners, names)
    if not np.all(truth["support_id_image"][evidence] == record["preferred"]):
        raise ValueError("Recorded evidence leaves its actual furniture support")
    _, components = label(evidence)
    if query == "demo_block" and components != 4:
        raise ValueError("Recorded block evidence must retain its four components")
    view_id = f"{query}_source"
    audit = {"kind": kind, "recording": str(neighborhood.RECORDING), "frame": frame_index,
             "semantic_skill": record["semantic_skill"], "source_kind": "scripted_sim",
             "region": record["region"], "evidence_pixels": int(evidence.sum()),
             "holes_preserved": record["holes"], "components_preserved": int(components),
             "entity_pixels": int(truth["entity_mask"].sum()), "same_asset_visual_meshes": meshes,
             "body_transform_source": "episode.h5 actual body_xpos/body_xquat; XML rounding not used",
             "historical_scene_container_exclusivity": False,
             "source_RGB_depth_unchanged": True, "whole_entity_mask_is_evaluation_only": True}
    return view_id, save_view(output, view_id, kind, frame, calibration, truth, audit), evidence


def load_view(output: Path, view: dict) -> tuple[dict, dict, dict]:
    """Read one saved view for exporter validation, never for inference."""
    folder = output / view["path"]
    depth = np.load(folder / "depth_m.npy", allow_pickle=False)
    calibration = json.loads((folder / "calibration.json").read_text())
    truth = archive(folder / "truth.npz")
    observed = neighborhood.observed_points({"depth_m": depth}, calibration, truth["entity_mask"])
    return observed, truth, {"depth_m": depth, "calibration": calibration}


def controlled_evidence(output: Path, view: dict, preferred: int) -> np.ndarray:
    """Define a source query only; target labels never enter its pixel mask."""
    observed, truth, frame = load_view(output, view)
    region = scenes.evaluate_visible_points(observed["xyz"], truth["support_corners_all"][preferred])["region_mask"]
    result = np.zeros(frame["depth_m"].shape, bool)
    x, y = observed["pixels"][region].T
    result[y, x] = True
    if int(result.sum()) < 12:
        raise ValueError("Source query is not observable")
    return result


def support_point_visibility(world: np.ndarray, observed: dict, depth: np.ndarray,
                             support: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Evaluate visibility with the correct optical depth at the sampled pixel.

    Rounding a projected subpixel location changes its ray. At a grazing shelf
    angle, half a pixel can change plane-intersection depth by several mm even
    though the same physical plane remains visible. Compare rendered depth to
    that rounded ray's intersection with the independently annotated plane,
    rather than to the unrounded world sample's camera Z. Owner and finite
    support membership remain mandatory; this is strictly an evaluator metric.
    """
    K, T = observed["K"], observed["T_world_camera_cv"]
    camera = (world-T[:3, 3]) @ T[:3, :3]
    homogeneous = camera @ K.T
    pixels = np.rint(homogeneous[:, :2]/homogeneous[:, 2, None]).astype(int)
    inside = (camera[:, 2] > 0) & np.all((pixels >= 0) & (pixels < depth.shape[::-1]), axis=1)
    support_image = np.zeros(depth.shape, bool)
    x, y = observed["pixels"][support].T
    support_image[y, x] = True
    normal = np.cross(corners[1]-corners[0], corners[3]-corners[0])
    normal /= np.linalg.norm(normal)
    rays = np.column_stack([pixels[inside], np.ones(int(inside.sum()))]) @ np.linalg.inv(K).T
    denominator = (rays @ T[:3, :3].T) @ normal
    nonparallel = np.abs(denominator) > 1e-8
    expected_depth = np.full(len(rays), np.nan)
    expected_depth[nonparallel] = ((corners[0]-T[:3, 3]) @ normal)/denominator[nonparallel]
    # Use the same 0.4 mm physical plane tolerance as the support annotation,
    # transformed to optical-Z units by this ray's incidence on that plane.
    tolerance = .0004/np.maximum(np.abs(denominator), 1e-8)
    result = np.zeros(len(world), bool)
    x, y = pixels[inside].T
    result[inside] = (support_image[y, x] & nonparallel & np.isfinite(expected_depth) & (expected_depth > 0)
                      & (np.abs(depth[y, x]-expected_depth) <= tolerance))
    return result


def add_query(output: Path, views: dict, pairs: list, query: str, source: str,
              kind: str, targets: list[str], view: str, preferred: int,
              evidence: np.ndarray, group: str) -> None:
    """Map all given source evidence through independent evaluation transforms."""
    evidence_path = output / "evidence" / f"{query}.npy"
    np.save(evidence_path, evidence, allow_pickle=False)
    observed, source_truth, source_frame = load_view(output, views[source])
    # Do not change this mean if SAM later excludes some evidence pixels.
    points = neighborhood.observed_points(source_frame, source_frame["calibration"], evidence)["xyz"]
    if len(points) != int(evidence.sum()):
        raise ValueError("Given evidence contains invalid depth")
    mean = points.mean(axis=0)
    inverse = source_truth["world_to_canonical"]
    canonical_mean = mean @ inverse[:3, :3].T + inverse[:3, 3]
    for variant in targets:
        target = f"{kind}_{variant}_{view}"
        target_observed, target_truth, target_frame = load_view(output, views[target])
        if not np.array_equal(source_truth["surface_names"], target_truth["surface_names"]):
            raise ValueError("Corresponding furniture surface-name maps differ")
        transform = target_truth["canonical_to_world"]
        mapped = canonical_mean @ transform[:3, :3].T + transform[:3, 3]
        x, y = target_observed["pixels"].T
        support = target_truth["support_id_image"][y, x] == preferred
        corners = target_truth["support_corners_all"][preferred]
        visible = bool(support_point_visibility(mapped[None], target_observed,
                      target_frame["depth_m"], support, corners)[0])
        # BEGIN support localization permits a shifted point on the correct
        # layer. The exact transported mean may be occluded and is diagnostic,
        # not a prerequisite or an automatic transfer-failure criterion.
        if not support.any():
            raise ValueError(f"Target physical support has no visible points: {query}/{variant}")
        source_canonical = points @ inverse[:3, :3].T + inverse[:3, 3]
        mapped_points = source_canonical @ transform[:3, :3].T + transform[:3, 3]
        projection_visible = support_point_visibility(mapped_points, target_observed,
                              target_frame["depth_m"], support, corners)
        # END separate diagnostic reference visibility from support visibility.
        variation = "original" if variant == "source" else variant
        pair_id = f"{query}_{variation}"
        truth_path = output / "pair-truth" / f"{pair_id}.json"
        scenes.write_json(truth_path, {"preferred_support_id": preferred,
            "source_reference_evidence_center_world": mean.tolist(),
            "target_reference_evidence_center_world": mapped.tolist(),
            "source_evidence_points": len(points), "target_reference_center_visible": visible,
            "target_support_visible_points": int(support.sum()),
            "target_source_evidence_projection_visible_fraction": float(projection_visible.mean()),
            "reference_definition": "arithmetic mean of all given source evidence points mapped through independent known canonical transforms"})
        pairs.append({"id": pair_id, "source": source, "target": target,
                      "evidence": str(evidence_path.relative_to(output)),
                      "pair_truth": str(truth_path.relative_to(output)),
                      "group": group, "variation": variation})


def validate(output: Path, expected_counts: tuple[int, int]) -> dict:
    """Validate saved full images, truth isolation, colors and source provenance."""
    manifest = json.loads((output / "manifest.json").read_text())
    views = manifest["views"]
    rows = []
    for view_id, view in views.items():
        folder = output / view["path"]
        observed, truth, frame = load_view(output, view)
        shape = frame["depth_m"].shape
        rgb = np.asarray(Image.open(folder / "rgb.png"))
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError("Full RGB dimensions or type changed")
        owner, surfaces, supports = truth["entity_mask"], truth["surface_id_image"], truth["support_id_image"]
        if owner.dtype != bool or any(a.shape != shape for a in (owner, surfaces, supports)):
            raise ValueError("Pixel truth does not align with the full image")
        if np.any(surfaces[~owner] != -1) or np.any(supports[~owner] != -1):
            raise ValueError("Non-furniture pixels acquired furniture truth")
        if not np.array_equal(surfaces >= 0, owner):
            raise ValueError("Surface IDs do not exactly cover valid furniture depth")
        corners = truth["support_corners_all"]
        x, y = observed["pixels"].T
        expected = neighborhood.support_labels(observed["xyz"], corners)
        if not np.array_equal(supports[y, x], expected):
            raise ValueError("Saved support labels differ from actual finite surfaces")
        if not np.allclose(truth["world_to_canonical"] @ truth["canonical_to_world"], np.eye(4), atol=1e-12, rtol=0):
            raise ValueError("Canonical matrices do not invert each other")
        T, K = observed["T_world_camera_cv"], observed["K"]
        camera = (observed["xyz"]-T[:3, 3]) @ T[:3, :3]
        uv = camera @ K.T
        error = float(np.max(np.abs(uv[:, :2]/uv[:, 2, None]-observed["pixels"])))
        if error > 1e-8:
            raise ValueError("Backprojected points do not return to source pixels")
        rows.append({"view": view_id, "entity_pixels": int(owner.sum()), "reprojection_error_px": error})
    color_rows = []
    for view_id, view in views.items():
        if "_color_" not in view_id:
            continue
        source = views[view_id.replace("_color_", "_source_")]
        a, b = output/view["path"], output/source["path"]
        if not np.array_equal(np.load(a/"depth_m.npy", allow_pickle=False), np.load(b/"depth_m.npy", allow_pickle=False)):
            raise ValueError("Color intervention changed optical depth")
        ca, cb = json.loads((a/"calibration.json").read_text()), json.loads((b/"calibration.json").read_text())
        if ca != cb:
            raise ValueError("Color intervention changed camera calibration")
        ta, tb = archive(a/"truth.npz"), archive(b/"truth.npz")
        if any(not np.array_equal(ta[key], tb[key]) for key in ta):
            raise ValueError("Color intervention changed geometry or segmentation truth")
        ra, rb = np.asarray(Image.open(a/"rgb.png")), np.asarray(Image.open(b/"rgb.png"))
        changed = np.any(ra != rb, axis=-1)
        if not np.any(changed & ta["entity_mask"]):
            raise ValueError("Color intervention did not change visible furniture")
        color_rows.append({"view": view_id, "geometry_depth_seg_calibration_equal": True,
                           "changed_RGB_pixels": int(changed.sum()),
                           "changed_entity_RGB_pixels": int((changed & ta["entity_mask"]).sum())})
    provenance_rows = []
    for query, record in RECORDS.items():
        view_id = f"{query}_source"
        if view_id not in views:
            continue
        folder = output/views[view_id]["path"]
        with h5py.File(neighborhood.RECORDING/"episode.h5", "r") as episode:
            rgb_equal = np.array_equal(np.asarray(Image.open(folder/"rgb.png")), episode["frames/rgb"][record["frame"]])
            depth_equal = np.array_equal(np.load(folder/"depth_m.npy", allow_pickle=False), episode["frames/depth_m"][record["frame"]])
        if not rgb_equal or not depth_equal:
            raise ValueError("Saved historical images differ from the selected original frame")
        evidence = np.load(output/"evidence"/f"{query}.npy", allow_pickle=False)
        if query == "demo_mug":
            expected = np.asarray(Image.open(neighborhood.DEMO_SOURCE/"interaction_mask.png")) > 0
        else:
            with h5py.File(neighborhood.PREPROCESSED/"masks.h5", "r") as masks:
                segment = masks[f"segments/{record['segment']}"]
                row = np.flatnonzero(np.asarray(segment["frame_indices"]) == record["frame"])
                if len(row) != 1:
                    raise ValueError("Historical evidence frame is no longer unique")
                expected = np.asarray(segment[f"regions/{record['region']}/local_mask"][row[0]], bool)
        if not np.array_equal(evidence, expected):
            raise ValueError("Saved source evidence differs from its original artifact")
        provenance_rows.append({"query": query, "frame": record["frame"], "RGB_depth_equal_recording": True,
                                "evidence_equal_original_artifact": True, "evidence_pixels": int(evidence.sum()),
                                "internal_hole_pixels": int(binary_fill_holes(evidence).sum()-evidence.sum())})
    pair_rows = []
    for pair in manifest["pairs"]:
        evidence = np.load(output/pair["evidence"], allow_pickle=False)
        source_observed, source_truth, source_frame = load_view(output, views[pair["source"]])
        target_observed, target_truth, target_frame = load_view(output, views[pair["target"]])
        reference = json.loads((output/pair["pair_truth"]).read_text())
        if evidence.dtype != bool or evidence.shape != source_frame["depth_m"].shape:
            raise ValueError("Evidence is not a full-image boolean mask")
        if not np.all(source_truth["support_id_image"][evidence] == reference["preferred_support_id"]):
            raise ValueError("Source query leaves its declared physical support")
        points = neighborhood.observed_points(source_frame, source_frame["calibration"], evidence)["xyz"]
        mean = points.mean(0)
        mapping = target_truth["canonical_to_world"] @ source_truth["world_to_canonical"]
        target = mean @ mapping[:3, :3].T + mapping[:3, 3]
        if (len(points) != reference["source_evidence_points"]
                or not np.allclose(mean, reference["source_reference_evidence_center_world"], atol=1e-12, rtol=0)
                or not np.allclose(target, reference["target_reference_evidence_center_world"], atol=1e-12, rtol=0)):
            raise ValueError("Saved evidence reference does not match original evidence mean")
        x, y = target_observed["pixels"].T
        support = target_truth["support_id_image"][y, x] == reference["preferred_support_id"]
        corners = target_truth["support_corners_all"][reference["preferred_support_id"]]
        visible = bool(support_point_visibility(target[None], target_observed, target_frame["depth_m"], support, corners)[0])
        mapped_points = points @ mapping[:3, :3].T + mapping[:3, 3]
        fraction = float(support_point_visibility(mapped_points, target_observed, target_frame["depth_m"], support, corners).mean())
        if (not support.any() or visible != reference["target_reference_center_visible"]
                or int(support.sum()) != reference["target_support_visible_points"]
                or not np.isclose(fraction, reference["target_source_evidence_projection_visible_fraction"], atol=1e-12, rtol=0)):
            raise ValueError("Saved target visibility diagnostics disagree with RGB-D")
        pair_rows.append({"pair": pair["id"], "target_support_visible_points": int(support.sum()),
                          "reference_center_visible": visible, "source_evidence_projection_visible_fraction": fraction})
    # BEGIN each experiment declares its frozen matrix before inference.
    if (len(views), len(manifest["pairs"])) != expected_counts:
        raise ValueError("Frozen view/pair matrix is incomplete")
    # END experiment-specific cardinality.
    result = {"status": "passed", "views": len(views), "pairs": len(manifest["pairs"]),
              "view_checks": rows, "color_checks": color_rows,
              "recorded_provenance_checks": provenance_rows, "pair_checks": pair_rows,
              "GT_only_files": manifest["evaluation_only_files"]}
    scenes.write_json(output.parent / f"{output.name}-validation.json", result)
    return result


def overview(output: Path, views: dict) -> None:
    """Provide a compact full-image contact sheet for human scene inspection."""
    width, height, columns = 320, 258, 5
    canvas = Image.new("RGB", (columns*width, ((len(views)+columns-1)//columns)*height), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (view_id, view) in enumerate(views.items()):
        x, y = (index % columns)*width, (index // columns)*height
        picture = Image.open(output/view["path"]/"rgb.png").resize((320, 240))
        canvas.paste(picture, (x, y+18))
        draw.text((x+4, y+3), view_id, fill="black")
    canvas.save(output/"overview.png")


def generate(output: Path, smoke: bool = False) -> dict:
    """Run serially and refuse to overwrite any previous generated observations."""
    from scene import load_scene

    output.mkdir(parents=True, exist_ok=False)
    (output/"evidence").mkdir()
    (output/"pair-truth").mkdir()
    original_model, original_data = load_scene("original")
    opening = drawer_opening()
    views, canonical, support_names = {}, {}, {}
    # The smoke uses the real block region and one newly isolated drawer view.
    # Full generation adds all categories and independent interventions only
    # after that minimal source-to-target contract has been checked externally.
    kinds = ("drawer",) if smoke else ("shelf", "drawer", "cabinet")
    variants = ("source",) if smoke else tuple(VARIANTS)
    recorded_evidence = {}
    for kind in kinds:
        for variant in variants:
            model, data = prepare_case(output, kind, variant, original_model, original_data, opening, VARIANTS[variant])
            if variant == "source":
                canonical[kind], support_names[kind] = canonical_supports(model, data, kind)
            camera_views = ("view_a", "low") if kind == "shelf" else ("view_a",)
            for camera_view in camera_views:
                view_id, record = capture_view(output, model, data, kind, variant, camera_view,
                                               canonical[kind], support_names[kind], np.asarray(VARIANTS[variant]["scale"]))
                views[view_id] = record
            if variant == "source" and kind in ("shelf", "drawer"):
                query = "demo_mug" if kind == "shelf" else "demo_block"
                view_id, record, evidence = import_record(output, query, canonical[kind], support_names[kind], model, data)
                views[view_id], recorded_evidence[query] = record, evidence
            del model, data
    return complete_dataset(output, views, recorded_evidence, opening, smoke)


def complete_dataset(output: Path, views: dict, recorded_evidence: dict,
                     opening: float, smoke: bool = False) -> dict:
    """Export query records and validate existing immutable camera captures."""
    kinds = ("drawer",) if smoke else ("shelf", "drawer", "cabinet")
    variants = ("source",) if smoke else tuple(VARIANTS)
    pairs = []
    if not smoke:
        for kind in kinds:
            source = f"{kind}_source_view_a"
            preferred = 3 if kind == "shelf" else 0
            evidence = controlled_evidence(output, views[source], preferred)
            add_query(output, views, pairs, kind, source, kind, list(VARIANTS)[1:], "view_a", preferred, evidence, kind)
        for preferred, name in enumerate(neighborhood.SUPPORT_NAMES[:3]):
            source = "shelf_source_low"
            evidence = controlled_evidence(output, views[source], preferred)
            add_query(output, views, pairs, f"repeated_{name}", source, "shelf", list(VARIANTS)[1:], "low", preferred, evidence, "repeated_interior")
    for query, evidence in recorded_evidence.items():
        record = RECORDS[query]
        add_query(output, views, pairs, query, f"{query}_source", record["kind"], list(variants),
                  "view_a", record["preferred"], evidence, query)
    scenes.write_json(output/"manifest.json", {"views": views, "pairs": pairs, "smoke": smoke,
        "schema": "full_rgbd_category_segmentation_v1", "original_scene": str(scenes.ORIGINAL/"scene.xml"),
        "variants": VARIANTS, "color_multiplier": COLOR_MULTIPLIER.tolist(),
        "drawer_recorded_opening_m": opening,
        "inference_files": ["views/*/rgb.png", "views/*/depth_m.npy", "views/*/calibration.json", "evidence/*.npy"],
        "evaluation_only_files": ["views/*/truth.npz", "views/*/evaluation-audit.json", "pair-truth/*.json", "evaluation-scenes/*"],
        "segmentation_contract": "one full RGB and one literal category prompt; no truth-based instance choice or mask repair",
        "historical_sources": "unaltered scripted MuJoCo recordings; only newly rendered targets enforce exclusive containers"})
    overview(output, views)
    return validate(output, (2, 1) if smoke else (22, 34))
# END independent category-segmentation fixtures.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    arguments = parser.parse_args()
    # BEGIN validation has the same explicit matrix contract as generation.
    result = validate(arguments.out.resolve(), (2, 1) if arguments.smoke else (22, 34)) if arguments.validate_only else generate(arguments.out.resolve(), arguments.smoke)
    # END explicit validation matrix.
    print(json.dumps({key: result[key] for key in ("status", "views", "pairs")}), flush=True)
