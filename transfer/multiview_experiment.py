"""Freeze multiview matches, run fixed registration backends, then evaluate.

The manifest declares observations separately from held-out evaluation files.
Neither input loading nor fusion/registration opens those evaluation files.
Every case's fused correspondences are written once and shared by all solvers.
"""

import argparse
import csv
import gc
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .features import write_json
from .segmented_experiment import archive, SAMPLING


# BEGIN independent multiview experiment; single-view experiment APIs are intact.
ALGORITHMS = ("ransac", "prosac_lo", "teaserpp")


def _name(value, label):
    """Names become directory components; paths belong in separate fields."""
    if not isinstance(value, str) or not value or value in (".", "..") or Path(value).name != value:
        raise ValueError(f"{label} must be a nonempty single path component")
    return value


def read_manifest(path):
    """Validate input declarations without opening any evaluation-only file."""
    path = Path(path).resolve()
    description = json.loads(path.read_text())
    if description.get("schema") != "multiview_grounding_v1":
        raise ValueError("Require a multiview_grounding_v1 manifest")
    cases = description.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("The manifest must declare at least one case")
    names = []
    for case in cases:
        names.append(_name(case["id"], "Case ID"))
        for key in ("source_input_dir", "source_observation_dir"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError(f"Missing {key}")
        targets = case.get("targets")
        if not isinstance(targets, list) or len(targets) != 2:
            raise ValueError("This experiment requires exactly two target observations")
        ids = [_name(target["id"], "View ID") for target in targets]
        if len(set(ids)) != len(ids):
            raise ValueError("Target view IDs must be unique")
        for target in targets:
            for key in ("input_dir", "observation_dir"):
                if not isinstance(target.get(key), str) or not target[key]:
                    raise ValueError(f"Target declaration requires {key}")
            if type(target.get("missing_target")) is not bool:
                raise ValueError("Every target must explicitly declare Boolean missing_target")
            reasons = target.get("missing_target_reasons")
            if not isinstance(reasons, list) or any(not isinstance(reason, str) or not reason for reason in reasons):
                raise ValueError("Missing-target reasons must be an explicit list of strings")
            if target["missing_target"]:
                if not reasons or target.get("missing_target_stage") != "segmentation_preparation":
                    raise ValueError("Missing targets require recorded segmentation-preparation reasons")
            elif reasons or target.get("missing_target_stage") is not None:
                raise ValueError("Prepared targets cannot carry missing-target failure declarations")
        # Evaluation references are checked structurally only. Existence and
        # contents are intentionally unavailable to all prediction stages.
        evaluation = case.get("evaluation")
        if (not isinstance(evaluation, dict) or not isinstance(evaluation.get("target_truth"), dict)
                or set(evaluation["target_truth"]) != set(ids)
                or not isinstance(evaluation.get("source_truth"), str)
                or not isinstance(evaluation.get("pair_truth"), str)):
            raise ValueError("Evaluation references must separately cover source, pair, and target views")
    if len(set(names)) != len(names):
        raise ValueError("Case IDs must be unique")
    return path, description


def _path(manifest_path, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (manifest_path.parent / path).resolve()


def selected_cases(description, selected=None):
    """Choose an explicit subset without changing the declared evaluation set."""
    if selected is None:
        return description["cases"]
    unknown = set(selected) - {case["id"] for case in description["cases"]}
    if unknown:
        raise ValueError(f"Unknown case IDs: {sorted(unknown)}")
    return [case for case in description["cases"] if case["id"] in selected]


def _check_observation_calibration(observation, folder):
    """Bind point-bank calibration to the full RGB-D observation declaration."""
    calibration = json.loads((folder / "calibration.json").read_text())
    for key in ("K", "T_world_camera_cv"):
        np.testing.assert_array_equal(observation[key], np.asarray(calibration[key]))


def _check_observed_depth(observation, depth):
    """Verify saved world points are backprojections of this exact depth image.

    This checks artifact identity at floating-point roundoff, not a physical
    sensor noise bound. No point is moved or accepted by a nearest match.
    """
    pixels = np.asarray(observation["pixels"])
    if (pixels.ndim != 2 or pixels.shape[1] != 2 or pixels.dtype.kind not in "iu"
            or np.any(pixels < 0) or np.any(pixels[:, 0] >= depth.shape[1])
            or np.any(pixels[:, 1] >= depth.shape[0])):
        raise ValueError("Saved observation pixels are outside their full depth image")
    values = depth[pixels[:, 1], pixels[:, 0]]
    if not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError("Saved observation includes invalid depth pixels")
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(observation["K"]).T
    transform = observation["T_world_camera_cv"]
    expected = (rays * values[:, None]) @ transform[:3, :3].T + transform[:3, 3]
    np.testing.assert_allclose(observation["xyz"], expected, rtol=1e-12, atol=1e-12)


def load_inputs(manifest_path, case, *, features=False):
    """Load prepared point identities and original depth, never simulator truth.

    Feature maps are opened sequentially. Only one source map and the two
    target maps survive loading; duplicate source maps are checked then freed.
    """
    source_dir = _path(manifest_path, case["source_input_dir"])
    preparation = json.loads((source_dir / "preparation.json").read_text())
    if not preparation["source"]["prepared"]:
        raise ValueError("Declared source input has not been prepared")
    source = archive(source_dir / "source_observation.npz")
    source_metadata = json.loads((source_dir / "inputs.json").read_text())
    source_crop = source_metadata["source_crop_xyxy"]
    source_full = _path(manifest_path, case["source_observation_dir"])
    _check_observation_calibration(source, source_full)
    source_depth = np.load(source_full / "depth_m.npy", allow_pickle=False)
    if source_depth.shape != source["object_mask"].shape:
        raise ValueError("Source full depth does not match its observation image")
    _check_observed_depth(source, source_depth)
    del source_depth
    source_feature = None
    if features:
        with np.load(source_dir / "features.npz", allow_pickle=False) as values:
            source_feature = values["feature1"]
    targets = []
    for target in sorted(case["targets"], key=lambda item: item["id"]):
        directory = _path(manifest_path, target["input_dir"])
        full = _path(manifest_path, target["observation_dir"])
        prepared = json.loads((directory / "preparation.json").read_text())
        if not prepared["source"]["prepared"]:
            raise ValueError("Every target pairing must preserve the prepared source observation")
        missing = target["missing_target"]
        if (missing != (not prepared["target"]["prepared"])
                or bool(prepared["prepared"]) != (not missing)
                or (missing and target["missing_target_reasons"] != prepared["target"]["reasons"])):
            raise ValueError("Explicit target declaration disagrees with its frozen preparation record")
        metadata = json.loads((directory / "inputs.json").read_text())
        candidate_source = archive(directory / "source_observation.npz")
        if candidate_source.keys() != source.keys() or metadata["source_crop_xyxy"] != source_crop:
            raise ValueError("All views must share exactly the same source observation and crop")
        for key in source:
            np.testing.assert_array_equal(candidate_source[key], source[key])
        del candidate_source
        depth = np.load(full / "depth_m.npy", allow_pickle=False)
        if depth.ndim != 2 or depth.dtype.kind not in "fiu" or min(depth.shape) < 1:
            raise ValueError("Target full depth must remain a nonempty numeric image")
        if missing:
            # Only an explicit preparation failure creates empty evidence.
            # Unknown file/shape/model errors still raise; no exception is
            # converted into a missing camera or replaced by another view.
            if (directory / "target_observation.npz").exists() or (directory / "features.npz").exists():
                raise ValueError("Missing target conflicts with a saved target/model artifact")
            from .segmentation import _calibration
            calibration = json.loads((full / "calibration.json").read_text())
            K, transform = _calibration(calibration["K"], calibration["T_world_camera_cv"])
            mask = np.zeros(depth.shape, dtype=bool)
            observation = {"xyz": np.empty((0, 3), dtype=float), "pixels": np.empty((0, 2), dtype=np.int64),
                "object_mask": mask, "entity_member": np.empty(0, dtype=bool),
                "region_mask": np.empty(0, dtype=bool), "evidence_only": np.empty(0, dtype=bool),
                "K": K, "T_world_camera_cv": transform}
            targets.append({"id": target["id"], "observation": observation, "depth_m": depth,
                "selected_mask": mask, "crop_xyxy": None, "feature": None,
                "missing_target": True, "missing_target_reasons": list(target["missing_target_reasons"]),
                "missing_target_stage": target["missing_target_stage"]})
            continue
        observation = archive(directory / "target_observation.npz")
        _check_observation_calibration(observation, full)
        mask = np.load(directory / "target_selected_mask.npy", allow_pickle=False)
        if depth.shape != observation["object_mask"].shape or mask.dtype != np.bool_:
            raise ValueError("Full depth and selected mask must align with the point-bank image")
        np.testing.assert_array_equal(mask, observation["object_mask"])
        _check_observed_depth(observation, depth)
        record = {"id": target["id"], "observation": observation,
                  "depth_m": depth, "selected_mask": mask, "crop_xyxy": metadata["target_crop_xyxy"],
                  "missing_target": False, "missing_target_reasons": [], "missing_target_stage": None}
        if features:
            with np.load(directory / "features.npz", allow_pickle=False) as values:
                np.testing.assert_array_equal(values["feature1"], source_feature)
                record["feature"] = values["feature2"]
        targets.append(record)
    return source, source_feature, source_crop, targets


def _save_fusion(directory, result, timing):
    """Separate typed arrays from metadata, including the candidate provenance."""
    arrays, metadata = {}, {}
    for key, value in result.items():
        if key == "bank":
            metadata["bank"] = {}
            for name, item in value.items():
                if isinstance(item, np.ndarray):
                    arrays[f"bank__{name}"] = item
                else:
                    metadata["bank"][name] = item
        elif isinstance(value, np.ndarray):
            arrays[key] = value
        else:
            metadata[key] = value
    np.savez_compressed(directory / "fusion.npz", **arrays)
    metadata.update(timing=timing, frozen=True, target_truth_used=False)
    write_json(directory / "fusion.json", metadata)


def load_fusion(directory):
    """Reload frozen matching output without calling any image model or matcher."""
    directory = Path(directory)
    metadata = json.loads((directory / "fusion.json").read_text())
    if not metadata.get("frozen") or metadata.get("target_truth_used") is not False:
        raise ValueError("Fusion must be a frozen prediction made without target truth")
    result = dict(metadata)
    result["bank"] = dict(metadata.get("bank", {}))
    for key, value in archive(directory / "fusion.npz").items():
        if key.startswith("bank__"):
            result["bank"][key.removeprefix("bank__")] = value
        else:
            result[key] = value
    return result


def fuse_suite(manifest, output, selected=None):
    """Compute each selected case's complete fusion field exactly once."""
    from .multiview_features import fuse_correspondences

    path, description = read_manifest(manifest)
    for case in selected_cases(description, selected):
        destination = Path(output) / "fusion" / case["id"]
        destination.mkdir(parents=True, exist_ok=False)
        begin = perf_counter()
        source, source_feature, source_crop, targets = load_inputs(path, case, features=True)
        loaded = perf_counter()
        found = fuse_correspondences(source, source_feature, source_crop, targets,
                                    sampling_radius_m=SAMPLING["sampling_radius_m"], count=160)
        finished = perf_counter()
        timing = {"input_and_feature_load_ms": 1000 * (loaded - begin),
                  "fusion_ms": 1000 * (finished - loaded),
                  "model_provenance": case.get("model_timing", {}), "fresh_model_calls": 0}
        _save_fusion(destination, found, timing)
        print(json.dumps({"case": case["id"], "stage": "fused",
            "correspondences": len(found["src_anchor_idx"]), "fusion_ms": timing["fusion_ms"]}), flush=True)
        del source, source_feature, targets, found
        gc.collect()


def predict_suite(manifest, output, selected=None, algorithms=ALGORITHMS):
    """One solver call per requested matrix cell, with no warmups or retries."""
    from . import core
    from .experiment import store_prediction
    from .multiview_grounding import infer_multiview_region

    if not algorithms or len(set(algorithms)) != len(algorithms) or not set(algorithms) <= set(ALGORITHMS):
        raise ValueError("Algorithms must be a distinct subset of the three declared backends")
    if "teaserpp" in algorithms:
        import teaserpp_python  # Native import is deliberately outside solver timings.
    path, description = read_manifest(manifest)
    for case in selected_cases(description, selected):
        source, _, _, targets = load_inputs(path, case)
        fused = load_fusion(Path(output) / "fusion" / case["id"])
        bank = fused["bank"]
        for algorithm in algorithms:
            destination = Path(output) / case["id"] / algorithm
            destination.mkdir(parents=True, exist_ok=False)
            begin = perf_counter()
            prepared = core.prepare_anchors(source["xyz"], bank["xyz"],
                fused["src_anchor_idx"], fused["tgt_anchor_idx"], mode="direct",
                registration_algorithm=algorithm, correspondence_scores=fused["scores"])
            registered = perf_counter()
            result = infer_multiview_region(source, targets, prepared, bank, **SAMPLING)
            finished = perf_counter()
            store_prediction(destination / "prediction", result, {})
            measurement = {"case": case["id"], "algorithm": algorithm, "status": "completed",
                "seed": 0, "requested_prediction_calls": 1, "actual_prediction_calls": 1,
                "warmup_calls": 0, "target_truth_used_by_prediction": False,
                "correspondence_count": len(fused["src_anchor_idx"]),
                "unique_source_count": len(np.unique(fused["src_anchor_idx"])),
                "unique_target_count": len(np.unique(fused["tgt_anchor_idx"])),
                "registration_and_anchor_preparation_ms": 1000 * (registered - begin),
                "region_generation_ms": 1000 * (finished - registered),
                "sequential_geometry_total_ms": 1000 * (finished - begin),
                "fusion_timing": fused["timing"], "fusion_diagnostics": fused["diagnostics"],
                "accepted": bool(result["accepted"]),
                "reasons": result["reasons"], "scale": prepared.scale,
                "rotation": prepared.rotation, "translation": prepared.translation,
                "diagnostics": prepared.diagnostics, "grounding_diagnostics": result["diagnostics"]}
            write_json(destination / "measurements.json", measurement)
            print(json.dumps({"case": case["id"], "algorithm": algorithm,
                "accepted": result["accepted"], "reasons": result["reasons"],
                "registration_ms": measurement["registration_and_anchor_preparation_ms"]}), flush=True)
        del source, targets, fused, bank
        gc.collect()


def restore_prediction(path):
    """Reconstruct observed modes from frozen prediction artifacts only."""
    path = Path(path)
    description = json.loads(path.with_suffix(".json").read_text())
    values = archive(path.with_suffix(".npz"))
    result = {"accepted": description["accepted"], "reasons": description["reasons"],
              "diagnostics": description["diagnostics"], "modes": []}
    for index, metadata in enumerate(description["modes"]):
        mode = dict(metadata)
        for key in ("seed_index", "center", "normal", "mask", "sample_indices"):
            mode[key] = values[f"mode_{index}_{key}"]
        mode["seed_index"] = int(mode["seed_index"])
        result["modes"].append(mode)
    return result


def transform_metrics(scale, rotation, translation, source, mapping, is_similarity):
    """Use point-mapping error for arbitrary affine truth, never a false scale."""
    from .core import apply_sim3
    if scale is None or not len(source):
        return None
    # BEGIN restore numeric arrays at the frozen JSON evaluation boundary.
    rotation, translation = np.asarray(rotation, dtype=float), np.asarray(translation, dtype=float)
    # END serialization changes types, never the saved transform values.
    predicted = apply_sim3(source, scale, rotation, translation)
    expected = source @ mapping[:3, :3].T + mapping[:3, 3]
    result = {"source_points_transform_rmse_m": float(np.sqrt(np.mean(np.sum((predicted - expected) ** 2, axis=1)))),
              "scale_relative_error": None, "rotation_error_deg": None, "translation_parameter_error_m": None}
    if is_similarity:
        true_scale = float(np.cbrt(np.linalg.det(mapping[:3, :3])))
        true_rotation = mapping[:3, :3] / true_scale
        cosine = np.clip((np.trace(np.asarray(rotation) @ true_rotation.T) - 1) / 2, -1, 1)
        result.update(scale_relative_error=abs(scale / true_scale - 1),
            rotation_error_deg=float(np.degrees(np.arccos(cosine))),
            translation_parameter_error_m=float(np.linalg.norm(np.asarray(translation) - mapping[:3, 3])))
    return result


def _error_summary(errors, tolerance):
    errors = np.asarray(errors)
    return {"count": len(errors), "median_error_m": float(np.median(errors)) if len(errors) else None,
            "p90_error_m": float(np.quantile(errors, .9)) if len(errors) else None,
            "near_truth_count": int(np.count_nonzero(errors <= tolerance)), "threshold_m": tolerance}


def evaluate_suite(manifest, output):
    """Open target truth only after every declared case/backend artifact exists."""
    from .evaluation import evaluate_prediction, evidence_reference_center

    path, description = read_manifest(manifest)
    output = Path(output)
    # This barrier also checks the native call record, rather than treating an
    # empty output directory as evidence that the requested prediction ran.
    for case in description["cases"]:
        for algorithm in ALGORITHMS:
            destination = output / case["id"] / algorithm
            for filename in ("prediction.json", "prediction.npz", "measurements.json"):
                if not (destination / filename).is_file():
                    raise FileNotFoundError("Finish every declared prediction before evaluating target truth: " + str(destination / filename))
            record = json.loads((destination / "measurements.json").read_text())
            if record["actual_prediction_calls"] != 1 or record["target_truth_used_by_prediction"]:
                raise ValueError("Evaluation requires exactly one truth-free prediction per matrix cell")
    rows, ground_truth = [], {}
    for case in description["cases"]:
        source, _, _, targets = load_inputs(path, case)
        fused = load_fusion(output / "fusion" / case["id"])
        bank, evaluation = fused["bank"], case["evaluation"]
        source_truth = archive(_path(path, evaluation["source_truth"]))
        pair_truth = json.loads(_path(path, evaluation["pair_truth"]).read_text())
        truths = {view_id: archive(_path(path, filename)) for view_id, filename in evaluation["target_truth"].items()}
        view_ids = list(bank["view_ids"])
        if view_ids != [target["id"] for target in targets]:
            raise ValueError("Frozen candidate observation order disagrees with the manifest")
        first_truth = truths[view_ids[0]]
        mapping = first_truth["canonical_to_world"] @ source_truth["world_to_canonical"]
        for truth in truths.values():
            np.testing.assert_allclose(truth["canonical_to_world"], first_truth["canonical_to_world"], atol=1e-12, rtol=0)
        singular = np.linalg.svd(mapping[:3, :3], compute_uv=False)
        similarity = bool(np.linalg.det(mapping[:3, :3]) > 0 and np.allclose(singular, singular[0], rtol=1e-10, atol=1e-12))
        reference = np.asarray(pair_truth["target_reference_evidence_center_world"])
        np.testing.assert_allclose(evidence_reference_center(source["xyz"], source["region_mask"], source_truth, first_truth),
                                   reference, rtol=0, atol=1e-9)
        support, owner_target = [], []
        for target in targets:
            px = target["observation"]["pixels"]
            truth = truths[target["id"]]
            support.append(truth["support_id_image"][px[:, 1], px[:, 0]])
            owner_target.append(truth["entity_mask"][px[:, 1], px[:, 0]])
        support, owner_target = np.concatenate(support), np.concatenate(owner_target)
        if len(support) != len(bank["observation_xyz"]):
            raise ValueError("Evaluation labels do not align with frozen observed target points")
        sx = source["pixels"]
        owner_source = source_truth["entity_mask"][sx[:, 1], sx[:, 0]]
        source_ids, target_ids = fused["src_anchor_idx"], fused["tgt_anchor_idx"]
        expected = source["xyz"][source_ids] @ mapping[:3, :3].T + mapping[:3, 3]
        errors = np.linalg.norm(bank["xyz"][target_ids] - expected, axis=1)
        ground_truth[case["id"]] = {"mapping_type": "Sim(3)" if similarity else "nonuniform_affine",
            "matrix": mapping, "singular_values": singular, "true_scale": float(singular[0]) if similarity else None,
            "target_reference_center_world": reference}
        for algorithm in ALGORITHMS:
            destination = output / case["id"] / algorithm
            row = json.loads((destination / "measurements.json").read_text())
            result = restore_prediction(destination / "prediction")
            tolerance = row["diagnostics"].get("ransac_tolerance_m")
            # Empty matching inputs can reject before a fitting tolerance is
            # needed. Report no threshold rather than fabricate a noise bound.
            correspondence = _error_summary(errors, tolerance) if tolerance is not None else {"count": len(errors), "threshold_m": None}
            observation_ids = bank["candidate_first_observation_index"][target_ids]
            correspondence.update(source_entity_count=int(owner_source[source_ids].sum()),
                                  target_entity_count=int(owner_target[observation_ids].sum()))
            row["quality"] = {"correspondence": correspondence,
                "transform": transform_metrics(row["scale"], row["rotation"], row["translation"], source["xyz"], mapping, similarity),
                "source_entity_fraction": float(owner_source.mean()),
                "true_source_entity_transform": transform_metrics(row["scale"], row["rotation"], row["translation"], source["xyz"][owner_source], mapping, similarity),
                "region": evaluate_prediction(result, bank["observation_xyz"],
                    {"support_id": support, "preferred_support_id": np.asarray(pair_truth["preferred_support_id"], dtype=np.int32)}, reference)}
            if all(key in fused for key in ("raw_source_idx", "raw_target_idx", "raw_valid")) and tolerance is not None:
                valid = fused["raw_valid"]
                raw_source = source["xyz"][fused["raw_source_idx"][valid]]
                raw_expected = raw_source @ mapping[:3, :3].T + mapping[:3, 3]
                row["quality"]["raw_correspondence"] = _error_summary(np.linalg.norm(
                    bank["xyz"][fused["raw_target_idx"][valid]] - raw_expected, axis=1), tolerance)
            # These are diagnostics from the already frozen matching fields,
            # not extra registrations, model calls, or alternative seed runs.
            if "per_view_best_candidate_idx" in fused and tolerance is not None:
                single_view = {}
                for view_index, view_id in enumerate(view_ids):
                    candidate_ids = fused["per_view_best_candidate_idx"][:, view_index]
                    valid = candidate_ids >= 0
                    raw_source = source["xyz"][fused["raw_source_idx"][valid]]
                    expected_view = raw_source @ mapping[:3, :3].T + mapping[:3, 3]
                    single_view[view_id] = _error_summary(np.linalg.norm(
                        bank["xyz"][candidate_ids[valid]] - expected_view, axis=1), tolerance)
                row["quality"]["single_view_rank_correspondence"] = single_view
            if "raw_view_supported" in fused and "raw_retained" in fused:
                support_counts = fused["raw_view_supported"][fused["raw_retained"]].sum(axis=1)
                row["retained_match_view_support"] = {"single_view": int(np.count_nonzero(support_counts == 1)),
                    "both_views": int(np.count_nonzero(support_counts == 2)),
                    "both_view_fraction": float(np.mean(support_counts == 2)) if len(support_counts) else None}
            write_json(destination / "measurements.json", row)
            rows.append(row)
        del source, targets, fused, bank, truths
        gc.collect()
    summary = {"rows": rows, "ground_truth": ground_truth, "protocol": {
        "predictions_per_case_algorithm": 1, "warmup_solver_calls": 0, "algorithm_order": ALGORITHMS,
        "case_order": [case["id"] for case in description["cases"]], "truth_loaded_after_all_predictions": True,
        "repeated_timing_or_seed_sweep": False, "model_inference_during_prediction": False,
        "geometry_timing_excludes": ["models", "fusion", "file_io", "evaluation", "plotting"]}}
    write_json(output / "summary.json", summary)
    flat = []
    for row in rows:
        transform = row["quality"]["transform"] or {}
        flat.append({"case": row["case"], "algorithm": row["algorithm"], "accepted": row["accepted"],
            "reasons": ";".join(row["reasons"]), "correspondence_count": row["correspondence_count"],
            "correspondence_near_truth_count": row["quality"]["correspondence"].get("near_truth_count"),
            "transform_rmse_m": transform.get("source_points_transform_rmse_m"),
            "registration_ms": row["registration_and_anchor_preparation_ms"],
            "region_generation_ms": row["region_generation_ms"],
            **{key: value for key, value in row["quality"]["region"].items() if key != "mode_metrics"}})
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(json.dumps({"evaluated": len(rows), "accepted": sum(row["accepted"] for row in rows)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("fuse", "predict", "evaluate"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", help="Optional comma-separated case IDs; evaluation always requires the full matrix")
    parser.add_argument("--algorithms", default=",".join(ALGORITHMS), help="Prediction-only backend subset")
    args = parser.parse_args()
    selected = args.cases.split(",") if args.cases else None
    if args.stage == "fuse":
        fuse_suite(args.manifest, args.output, selected)
    elif args.stage == "predict":
        predict_suite(args.manifest, args.output, selected, tuple(args.algorithms.split(",")))
    else:
        if selected is not None:
            raise ValueError("Evaluation must include every manifest case and all three backends")
        evaluate_suite(args.manifest, args.output)


if __name__ == "__main__":
    main()
# END independent multiview experiment.
