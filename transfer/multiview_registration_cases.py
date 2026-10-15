"""Prepare the declared three-case multiview registration fixture.

Only this fixture module names cases or opens simulator annotations. Inference
consumes the separate manifest written here; its evaluation paths are opened
only by the explicit evaluation stage. Existing observations are immutable.
"""

import argparse
import json
from pathlib import Path
import shutil
from time import perf_counter

import numpy as np

from .features import write_json
from .segmented_experiment import archive


# BEGIN independent multiview fixture; no changes to the single-view runners.
CASE_IDS = ("shelf_scale_125", "drawer_shape-color", "cabinet_yaw")


def locations(root, output):
    """Keep dated fixture locations out of the generic inference interface."""
    root, output = Path(root).resolve(), Path(output).resolve()
    return {"root": root, "output": output,
            "previous": root / "sim-version-plans/2026-10-14-three-category-registration",
            "cross": root / "sim-version-plans/2026-10-14-cross-view-registration",
            "supplement": output / "cabinet_b"}


def generate(root, output):
    """Capture only the missing B camera from the frozen cabinet target state."""
    from . import segmentation_scenes as seg, neighborhood_scenes as neighborhood, scenes

    paths = locations(root, output)
    previous, supplement = paths["previous"] / "data", paths["supplement"]
    old = json.loads((previous / "manifest.json").read_text())
    prior = next(pair for pair in old["pairs"] if pair["id"] == "cabinet_yaw")
    destination = supplement / "data"
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "evidence").mkdir()
    (destination / "pair-truth").mkdir()
    source_id = prior["source"]
    source_view = old["views"][source_id]
    shutil.copytree(previous / source_view["path"], destination / source_view["path"])
    evidence = np.load(previous / prior["evidence"], allow_pickle=False)
    target_a = archive(previous / old["views"][prior["target"]]["path"] / "truth.npz")
    scene_folder = previous / "evaluation-scenes/cabinet_yaw"
    model, data = neighborhood.load_native_case(scene_folder)
    position, rotation, qpos = data.xpos.copy(), data.xmat.copy(), data.qpos.copy()
    canonical = neighborhood.world_supports(target_a["support_corners_all"], target_a["world_to_canonical"])
    target_id, target_view = seg.capture_view(destination, model, data, "cabinet", "yaw",
        "view_b", canonical, tuple(target_a["support_names"]), np.ones(3))
    target_view["prompt"] = old["views"][prior["target"]]["prompt"]
    np.testing.assert_array_equal(data.xpos, position)
    np.testing.assert_array_equal(data.xmat, rotation)
    np.testing.assert_array_equal(data.qpos, qpos)
    target_b = archive(destination / target_view["path"] / "truth.npz")
    for key in ("canonical_to_world", "world_to_canonical", "support_corners_all"):
        np.testing.assert_allclose(target_b[key], target_a[key], rtol=0, atol=1e-12)
    calibration_a = json.loads((previous / old["views"][prior["target"]]["path"] / "calibration.json").read_text())
    calibration_b = json.loads((destination / target_view["path"] / "calibration.json").read_text())
    np.testing.assert_allclose(calibration_a["K"], calibration_b["K"], rtol=0, atol=1e-9)
    if np.allclose(calibration_a["T_world_camera_cv"], calibration_b["T_world_camera_cv"], rtol=0, atol=1e-9):
        raise ValueError("The missing B observation must use the saved B camera pose")
    write_json(destination / "capture-checks.json", {"passed": True,
        "frozen_scene": scene_folder, "source_observation_reused_from": previous / source_view["path"],
        "camera_configuration": scenes.ORIGINAL / "view_b/camera.json",
        "body_poses_and_qpos_unchanged": True, "target_geometry_mapping_unchanged": True,
        "source_evidence_unchanged": True})
    complete_capture(root, output)


def complete_capture(root, output):
    """Finish saved observations without requiring the preferred surface visible.

    A second target camera may see none of the demonstrated support surface.
    That is an observation to preserve, not a reason to move the camera or alter
    the support annotation. The old single-view add_query/validate functions
    require visible support, so this multiview fixture writes its own measured
    visibility record. This stage also completes an already captured pair
    without rendering again or overwriting its RGB-D/evidence artifacts.
    """
    from . import segmentation_scenes as seg, neighborhood_scenes as neighborhood, scenes

    paths = locations(root, output)
    previous, destination = paths["previous"] / "data", paths["supplement"] / "data"
    old = json.loads((previous / "manifest.json").read_text())
    prior = next(pair for pair in old["pairs"] if pair["id"] == "cabinet_yaw")
    source_id, target_id = prior["source"], "cabinet_yaw_view_b"
    source_view = old["views"][source_id].copy()
    target_view = {"path": f"views/{target_id}", "prompt": old["views"][prior["target"]]["prompt"]}
    views = {source_id: source_view, target_id: target_view}
    truth_path = destination / "pair-truth/cabinet_yaw_a_to_b.json"
    for path in (destination / "manifest.json", destination / "state-checks.json", truth_path):
        if path.exists():
            raise FileExistsError(path)
    # The existing A source must remain identical, including its calibration.
    for filename in ("rgb.png", "depth_m.npy", "calibration.json"):
        if (previous / source_view["path"] / filename).read_bytes() != (destination / source_view["path"] / filename).read_bytes():
            raise ValueError(f"Captured supplement changed frozen source {filename}")
    source_observed, source_truth, source_frame = seg.load_view(destination, source_view)
    target_observed, target_truth, target_frame = seg.load_view(destination, target_view)
    target_a = archive(previous / old["views"][prior["target"]]["path"] / "truth.npz")
    for key in ("canonical_to_world", "world_to_canonical", "support_corners_all"):
        np.testing.assert_allclose(target_truth[key], target_a[key], rtol=0, atol=1e-12)
    for key in ("surface_names", "support_names"):
        np.testing.assert_array_equal(target_truth[key], target_a[key])
    original_b = json.loads((scenes.ORIGINAL / "view_b/calibration.json").read_text())
    for key in ("K", "T_world_camera_cv"):
        np.testing.assert_allclose(target_frame["calibration"][key], original_b[key], rtol=0, atol=1e-9)
    # Validate the saved label/geometry relation even when its preferred subset
    # is empty. No observed point or support label is created by this check.
    tx, ty = target_observed["pixels"].T
    np.testing.assert_array_equal(target_truth["support_id_image"][ty, tx],
        neighborhood.support_labels(target_observed["xyz"], target_truth["support_corners_all"]))
    evidence = np.load(previous / prior["evidence"], allow_pickle=False)
    evidence_path = destination / "evidence/cabinet.npy"
    if evidence_path.exists():
        np.testing.assert_array_equal(np.load(evidence_path, allow_pickle=False), evidence)
    else:
        np.save(evidence_path, evidence, allow_pickle=False)
    reference = json.loads((previous / prior["pair_truth"]).read_text())
    preferred = int(reference["preferred_support_id"])
    if not np.all(source_truth["support_id_image"][evidence] == preferred):
        raise ValueError("Source evidence changed its physical support identity")
    points = neighborhood.observed_points(source_frame, source_frame["calibration"], evidence)["xyz"]
    if len(points) != int(evidence.sum()):
        raise ValueError("Frozen source evidence includes invalid depth")
    mapping = target_truth["canonical_to_world"] @ source_truth["world_to_canonical"]
    mapped = points @ mapping[:3, :3].T + mapping[:3, 3]
    np.testing.assert_allclose(mapped.mean(0), reference["target_reference_evidence_center_world"], rtol=0, atol=1e-12)
    support = target_truth["support_id_image"][ty, tx] == preferred
    corners = target_truth["support_corners_all"][preferred]
    visible = seg.support_point_visibility(mapped, target_observed, target_frame["depth_m"], support, corners)
    center_visible = bool(seg.support_point_visibility(mapped.mean(0)[None], target_observed,
        target_frame["depth_m"], support, corners)[0])
    reference.update(target_reference_center_visible=center_visible,
        target_support_visible_points=int(support.sum()),
        target_source_evidence_projection_visible_fraction=float(visible.mean()))
    write_json(truth_path, reference)
    pair = {"id": "cabinet_yaw_a_to_b", "source": source_id, "target": target_id,
        "evidence": str(evidence_path.relative_to(destination)), "pair_truth": str(truth_path.relative_to(destination)),
        "group": "cabinet", "variation": "yaw"}
    description = {key: old[key] for key in ("schema", "inference_files", "segmentation_contract")}
    description.update(views=views, pairs=[pair], evaluation_only_files=["views/*/truth.npz",
        "views/*/evaluation-audit.json", "pair-truth/*.json", "state-checks.json", "capture-checks.json"])
    write_json(destination / "state-checks.json", {"passed": True,
        "completion_render_calls": 0, "saved_target_mapping_and_supports_equal_to_A": True,
        "original_B_calibration_preserved": True, "source_RGBD_calibration_evidence_unchanged": True,
        "target_support_visible_points": int(support.sum()), "target_reference_center_visible": center_visible,
        "capture_assertions_record": str(destination / "capture-checks.json") if (destination / "capture-checks.json").exists() else None,
        "zero_visible_preferred_support_is_valid": True})
    write_json(destination / "manifest.json", description)
    print(json.dumps({"status": "passed", "views": 2, "pairs": 1,
        "target_support_visible_points": int(support.sum()), "completion_render_calls": 0}), flush=True)


def prepare_sam(root, output):
    """Reuse the exact source SAM recording and request just one new target."""
    from . import segmented_experiment as experiment
    from .segmentation import detect_image, load_detections

    paths = locations(root, output)
    supplement, previous = paths["supplement"], paths["previous"]
    data, detections, run = supplement / "data", supplement / "detections", supplement / "inputs"
    _, views, pairs = experiment.manifest(data)
    pair = pairs[0]
    provenance, timings = [], []
    for name in (pair["source"], pair["target"]):
        view = views[name]
        rgb, destination = data / view["path"] / "rgb.png", detections / name
        if name == pair["source"]:
            old = previous / "detections" / name
            if (old / "rgb.png").read_bytes() != rgb.read_bytes():
                raise ValueError("Cached source SAM input differs from the frozen source RGB")
            if json.loads((old / "candidates.json").read_text())["prompt"] != view["prompt"]:
                raise ValueError("Cached source SAM prompt changed")
            load_detections(old)
            shutil.copytree(old, destination)
            provenance.append({"view": name, "reused_from": old})
        else:
            begin = perf_counter()
            detect_image(rgb, view["prompt"], destination)
            timings.append({"model": "sam31", "view": name, "fresh_call_ms": 1000 * (perf_counter() - begin)})
            provenance.append({"view": name, "fresh_detection": True})
    write_json(detections / "provenance.json", provenance)
    experiment.prepare_suite(data, detections, run, arms=("sam31",))
    old_index = json.loads((previous / "cases.json").read_text())
    old = Path(next(case["input_dir"] for case in old_index["cases"] if case["id"] == "cabinet_yaw"))
    new = run / "sam31" / pair["id"]
    old_observation, new_observation = archive(old / "source_observation.npz"), archive(new / "source_observation.npz")
    if old_observation.keys() != new_observation.keys():
        raise ValueError("Frozen source observation schema changed")
    for key in old_observation:
        np.testing.assert_array_equal(old_observation[key], new_observation[key])
    if (new / "source.png").read_bytes() != (old / "source.png").read_bytes():
        raise ValueError("Frozen source MARCO crop changed")
    write_json(supplement / "sam-timing.json", {"source_observation_and_crop_exactly_equal": True, "calls": timings})


def extract_features(root, output):
    """Extract an observed pair, or record an explicit missing-target skip."""
    from . import segmented_experiment as experiment

    paths = locations(root, output)
    supplement = paths["supplement"]
    new = supplement / "inputs/sam31/cabinet_yaw_a_to_b"
    timing_path = supplement / "marco-timing.json"
    if timing_path.exists():
        raise FileExistsError(timing_path)
    preparation = json.loads((new / "preparation.json").read_text())
    if not preparation["source"]["prepared"]:
        raise ValueError("The frozen source must be prepared even when a target is missing")
    if not preparation["target"]["prepared"]:
        reasons = preparation["target"]["reasons"]
        if preparation["prepared"] or not reasons:
            raise ValueError("Missing target preparation must include its original failure reasons")
        if (new / "features.npz").exists():
            raise ValueError("An explicitly missing target cannot have a MARCO pair artifact")
        write_json(timing_path, {"model": "marco", "status": "skipped_no_target_observation",
            "fresh_pair_call_ms": 0., "model_calls": 0, "target_preparation_reasons": reasons,
            "source_features_reused_from_observed_A_pair": True})
        print(json.dumps({"stage": "features", "status": "skipped_no_target_observation", "reasons": reasons}), flush=True)
        return
    if not preparation["prepared"]:
        raise ValueError("Pair preparation is inconsistent with its prepared source and target")
    begin = perf_counter()
    experiment.extract_suite(supplement / "data", supplement / "inputs", arms=("sam31",))
    elapsed = 1000 * (perf_counter() - begin)
    old_index = json.loads((paths["previous"] / "cases.json").read_text())
    old = Path(next(case["input_dir"] for case in old_index["cases"] if case["id"] == "cabinet_yaw"))
    with np.load(old / "features.npz", allow_pickle=False) as first, np.load(new / "features.npz", allow_pickle=False) as second:
        np.testing.assert_array_equal(first["feature1"], second["feature1"])
    write_json(timing_path, {"model": "marco", "status": "completed", "model_calls": 1, "fresh_pair_call_ms": elapsed,
        "source_features_exactly_equal": True, "timing_includes": ["worker_startup", "model_inference", "feature_file_io"]})


def write_manifest(root, output):
    """Declare real input paths and isolated evaluation paths for all cases."""
    paths = locations(root, output)
    previous, cross, supplement = paths["previous"], paths["cross"], paths["supplement"]
    old = json.loads((previous / "cases.json").read_text())
    cross_cases = json.loads((cross / "cases.json").read_text())
    a_inputs = {case["id"]: Path(case["input_dir"]) for case in old["cases"]}
    b_inputs = {case["id"].removesuffix("_a_to_b"): Path(case["input_dir"]) for case in cross_cases["cases"]}
    b_inputs["cabinet_yaw"] = supplement / "inputs/sam31/cabinet_yaw_a_to_b"
    data = Path(old["data"])
    original = json.loads((data / "manifest.json").read_text())
    original_pairs = {pair["id"]: pair for pair in original["pairs"]}
    cases = []
    for name in CASE_IDS:
        pair = original_pairs[name]
        source_folder = data / original["views"][pair["source"]]["path"]
        targets = []
        for view_id, input_dir in (("A", a_inputs[name]), ("B", b_inputs[name])):
            metadata = json.loads((input_dir / "inputs.json").read_text())
            prepared = json.loads((input_dir / "preparation.json").read_text())
            if not prepared["source"]["prepared"]:
                raise ValueError(f"Declared multiview source input is unprepared: {input_dir}")
            missing = not prepared["target"]["prepared"]
            reasons = prepared["target"]["reasons"]
            if bool(prepared["prepared"]) != (not missing) or (missing and not reasons):
                raise ValueError("Target preparation status or recorded reasons are inconsistent")
            required_files = ["source_observation.npz", "source.png"]
            if not missing:
                required_files += ["target_observation.npz", "features.npz", "target_selected_mask.npy"]
            elif (input_dir / "target_observation.npz").exists() or (input_dir / "features.npz").exists():
                raise ValueError("Missing target declaration conflicts with existing target/model artifacts")
            for filename in required_files:
                if not (input_dir / filename).is_file():
                    raise FileNotFoundError(input_dir / filename)
            targets.append({"id": view_id, "input_dir": str(input_dir),
                "observation_dir": str(Path(metadata["target"]).parent), "missing_target": missing,
                "missing_target_reasons": reasons if missing else [],
                "missing_target_stage": "segmentation_preparation" if missing else None})
        fresh_model_timing = []
        if name == "cabinet_yaw":
            for timing_file in (supplement / "sam-timing.json", supplement / "marco-timing.json"):
                fresh_model_timing.append({"path": str(timing_file), "record": json.loads(timing_file.read_text())})
        cases.append({"id": name, "source_input_dir": str(a_inputs[name]),
            "source_observation_dir": str(source_folder), "targets": targets,
            "evaluation": {"source_truth": str(source_folder / "truth.npz"),
                "target_truth": {target["id"]: str(Path(target["observation_dir"]) / "truth.npz") for target in targets},
                "pair_truth": str(data / pair["pair_truth"])},
            "model_timing": {"cached_inputs": True, "cached_model_ms": None,
                "cached_model_ms_reason": "Previous recordings have stage totals, not isolated per-case model timings",
                "fresh_model_timing": fresh_model_timing}})
    destination = paths["output"] / "manifest.json"
    if destination.exists():
        raise FileExistsError(destination)
    write_json(destination, {"schema": "multiview_grounding_v1", "cases": cases,
        "protocol": {"source_views": 1, "target_views": 2, "evaluation_paths_are_not_inference_inputs": True}})
    print(json.dumps({"manifest": str(destination), "cases": len(cases)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "complete", "sam", "features", "manifest"))
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    {"generate": generate, "complete": complete_capture, "sam": prepare_sam,
     "features": extract_features, "manifest": write_manifest}[args.stage](args.root, args.output)


if __name__ == "__main__":
    main()
# END independent multiview fixture.
