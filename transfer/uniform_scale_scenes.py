"""Frozen evaluation fixtures for furniture enlarged uniformly by 1.25 and 1.5.

Only this exporter reads simulator geometry. Inference keeps the existing full
RGB-D, literal category prompt and original source-evidence contract. Target
furniture scales about its bottom plane; cameras and unrelated bodies stay fixed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from scipy.spatial import cKDTree

from . import scenes
from . import segmentation_scenes as seg

# BEGIN independent uniform-scale experiment; no category logic enters inference.
BASE = scenes.DEFAULT_OUTPUT.parent
SOURCE = BASE / "data-segmentation-prompt-001"
SCALES = {"scale_125": 1.25, "scale_150": 1.5}
PROMPTS = {"shelf": "wooden cabinet", "drawer": "drawer", "cabinet": "box with an open door"}


def mesh_world(model, data, geom):
    """Return compiled mesh vertices in world coordinates for independent checks."""
    mesh = int(model.geom_dataid[geom])
    start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    return model.mesh_vert[start:start+count] @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]


def bottom_height(model, data, kind):
    """Compute the actual lowest geometry point, including rotated primitives."""
    import mujoco
    heights = []
    for geom in scenes.descendant_geoms(model, scenes.FIXTURES[kind]["root"]):
        shape = model.geom_type[geom]
        size, rotation = model.geom_size[geom], data.geom_xmat[geom].reshape(3, 3)
        if shape == mujoco.mjtGeom.mjGEOM_MESH:
            heights.append(mesh_world(model, data, geom)[:, 2].min())
            continue
        if shape == mujoco.mjtGeom.mjGEOM_BOX:
            extent = np.abs(rotation[2]) @ size
        elif shape == mujoco.mjtGeom.mjGEOM_CYLINDER:
            extent = size[0]*np.linalg.norm(rotation[2, :2])+size[1]*abs(rotation[2, 2])
        elif shape == mujoco.mjtGeom.mjGEOM_SPHERE:
            extent = size[0]
        else:
            raise ValueError(f"Unsupported geometry type in bottom audit: {shape}")
        heights.append(data.geom_xpos[geom, 2]-extent)
    return float(min(heights))


def verify_scale(source_model, source_data, model, data, kind, factor, anchor):
    """Check each mesh, primitive, body and joint against the known similarity.

    This audit derives world geometry directly from MuJoCo, independently of
    the canonical matrices used to create evaluator support labels. It never
    supplies point correspondences, scale, or support labels to inference.
    """
    import mujoco
    rows = []
    owned = set(scenes.descendant_geoms(source_model, scenes.FIXTURES[kind]["root"]))
    for geom in range(source_model.ngeom):
        name = source_model.geom(geom).name
        target = model.geom(name).id
        active_factor = factor if geom in owned else 1.
        expected = anchor + factor*(source_data.geom_xpos[geom]-anchor) if geom in owned else source_data.geom_xpos[geom]
        center_error = float(np.linalg.norm(expected-data.geom_xpos[target]))
        if center_error > 2e-7 or source_model.geom_type[geom] != model.geom_type[target]:
            raise ValueError(f"Geometry center/type mismatch: {name}, {center_error}")
        if not np.allclose(source_data.geom_xmat[geom], data.geom_xmat[target], atol=1e-9, rtol=0):
            raise ValueError(f"Geometry orientation changed: {name}")
        if source_model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH:
            a, b = mesh_world(source_model, source_data, geom), mesh_world(model, data, target)
            expected_points = anchor + factor*(a-anchor) if geom in owned else a
            if a.shape != b.shape:
                raise ValueError(f"Mesh topology changed: {name}")
            error = float(max(cKDTree(b).query(expected_points)[0].max(), cKDTree(expected_points).query(b)[0].max()))
        else:
            error = float(np.max(np.abs(source_model.geom_size[geom]*active_factor-model.geom_size[target])))
        if error > 2e-7:
            raise ValueError(f"Geometry scale mismatch: {name}, {error}")
        if geom in owned:
            rows.append({"geom": name, "center_error_m": center_error, "geometry_error_m": error})
    bottom_a, bottom_b = bottom_height(source_model, source_data, kind), bottom_height(model, data, kind)
    if abs(bottom_a-bottom_b) > 2e-7:
        raise ValueError("Furniture bottom plane moved")
    joint = scenes.FIXTURES[kind]["joint"]
    joint_check = None
    if joint:
        old, new = source_model.joint(joint), model.joint(joint)
        a, b = float(source_data.qpos[old.qposadr[0]]), float(data.qpos[new.qposadr[0]])
        if not np.isclose(b, factor*a, atol=1e-12, rtol=0):
            raise ValueError("Furniture opening did not scale")
        joint_check = {"joint": joint, "source_opening_m": a, "target_opening_m": b}
    return {"kind": kind, "factor": factor, "anchor_world": anchor.tolist(),
            "source_bottom_world_z_m": bottom_a, "target_bottom_world_z_m": bottom_b,
            "all_unrelated_geometry_unchanged": True, "joint": joint_check, "geometry": rows}


def generate(output):
    """Reuse all six frozen source views; render only eight enlarged targets."""
    from scene import load_scene
    output.mkdir(parents=True, exist_ok=False)
    (output/"evidence").mkdir()
    (output/"pair-truth").mkdir()
    source_manifest = json.loads((SOURCE/"manifest.json").read_text())
    views = {}
    for name in sorted({p["source"] for p in source_manifest["pairs"]}):
        record = source_manifest["views"][name]
        shutil.copytree(SOURCE/record["path"], output/record["path"])
        views[name] = record.copy()
    original_model, original_data = load_scene("original")
    opening = seg.drawer_opening()
    geometry_checks = []
    for kind in ("shelf", "drawer", "cabinet"):
        source_model, source_data = seg.prepare_case(output, kind, "source", original_model, original_data, opening, seg.VARIANTS["source"])
        canonical, names = seg.canonical_supports(source_model, source_data, kind)
        root = source_model.body(scenes.FIXTURES[kind]["root"]).id
        anchor = source_data.xpos[root].copy()
        anchor[2] = bottom_height(source_model, source_data, kind)
        for variant, factor in SCALES.items():
            # Translation is exactly the offset needed for scaling about the
            # bottom anchor. It is not a separate scene-translation intervention.
            shift = (1-factor)*(anchor-source_data.xpos[root])
            params = {"scale": [factor]*3, "translation": shift.tolist(), "yaw_deg": 0.}
            model, data = seg.prepare_case(output, kind, variant, original_model, original_data, opening, params)
            geometry_checks.append(verify_scale(source_model, source_data, model, data, kind, factor, anchor))
            for view in (("view_a", "low") if kind == "shelf" else ("view_a",)):
                view_id, record = seg.capture_view(output, model, data, kind, variant, view, canonical, names, np.full(3, factor))
                record["prompt"] = PROMPTS[kind]
                views[view_id] = record
            del model, data
        del source_model, source_data
    pairs, seen = [], set()
    for prior in source_manifest["pairs"]:
        query = Path(prior["evidence"]).stem
        if query in seen:
            continue
        seen.add(query)
        source = prior["source"]
        kind = json.loads((SOURCE/views[source]["path"]/"evaluation-audit.json").read_text())["kind"]
        truth = json.loads((SOURCE/prior["pair_truth"]).read_text())
        evidence = np.load(SOURCE/prior["evidence"], allow_pickle=False)
        seg.add_query(output, views, pairs, query, source, kind, list(SCALES),
                      "low" if source.endswith("_low") else "view_a", truth["preferred_support_id"], evidence, prior["group"])
        if (output/prior["evidence"]).read_bytes() != (SOURCE/prior["evidence"]).read_bytes():
            raise ValueError("Source evidence changed")
    manifest = {"schema": "full_rgbd_category_segmentation_v1", "views": views, "pairs": pairs,
        "smoke": False, "scale_factors": SCALES, "source_dataset": str(SOURCE),
        "scaling_anchor": "original furniture root XY and lowest geometry world Z",
        "inference_files": source_manifest["inference_files"],
        "evaluation_only_files": source_manifest["evaluation_only_files"],
        "segmentation_contract": source_manifest["segmentation_contract"],
        "historical_sources": source_manifest["historical_sources"]}
    scenes.write_json(output/"manifest.json", manifest)
    scenes.write_json(output/"uniform-geometry-checks.json", {"passed": True, "scenes": geometry_checks})
    seg.overview(output, views)
    return seg.validate(output, (14, 16))
# END independent uniform-scale experiment.

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=BASE/"data-uniform-scale-001")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    result = seg.validate(args.out.resolve(), (14, 16)) if args.validate_only else generate(args.out.resolve())
    print(json.dumps({k: result[k] for k in ("status", "views", "pairs")}), flush=True)
