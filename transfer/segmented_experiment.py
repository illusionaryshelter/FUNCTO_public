"""Run category-prompt segmentation and direct interaction transfer separately.

Model inputs, predictions, and held-out measurements have distinct stages.
The SAM arm never reads scene truth while selecting instances, constructing
point banks, cropping images, matching, or generating neighborhoods. The
explicit gt_mask control supplies an annotated entity mask at preparation only;
it is a separate experiment, never a replacement for a failed SAM prediction.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .features import write_json


# BEGIN segmented transfer experiment; one numerical pipeline for every prompt.
ARMS = ("gt_mask", "sam31")
SAMPLING = {"sampling_radius_m": .03, "sampling_spacing_m": .005, "max_modes": 3}


def archive(path):
    """Read typed arrays without pickle; callers explicitly choose the stage."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def manifest(data_root):
    """Resolve declared inputs inside the dataset without opening any truth."""
    root = Path(data_root).resolve()
    description = json.loads((root / "manifest.json").read_text())
    views, pairs = description["views"], description["pairs"]
    if not views or not pairs or len({pair["id"] for pair in pairs}) != len(pairs):
        raise ValueError("Require nonempty views and unique experiment pair IDs")
    for name, view in views.items():
        if Path(name).name != name or not view["prompt"].strip():
            raise ValueError("View IDs and explicit category prompts are required")
        if not (root / view["path"]).resolve().is_relative_to(root):
            raise ValueError("Observation references must remain in the dataset")
    for pair in pairs:
        if Path(pair["id"]).name != pair["id"]:
            raise ValueError("Pair IDs must be single directory names")
        if pair["source"] not in views or pair["target"] not in views:
            raise ValueError("A pair references an undeclared observation")
        for key in ("evidence", "pair_truth"):
            if not (root / pair[key]).resolve().is_relative_to(root):
                raise ValueError("Pair file references must remain in the dataset")
    return root, views, pairs


def read_observation_inputs(root, view):
    """Load the complete RGB and depth; no simulator mask is available here."""
    folder = root / view["path"]
    calibration = json.loads((folder / "calibration.json").read_text())
    return (np.array(Image.open(folder / "rgb.png").convert("RGB")),
            np.load(folder / "depth_m.npy", allow_pickle=False),
            np.asarray(calibration["K"]), np.asarray(calibration["T_world_camera_cv"]))


def selected_pairs(pairs, selected):
    if selected is None:
        return pairs
    unknown = set(selected) - {pair["id"] for pair in pairs}
    if unknown:
        raise ValueError(f"Unknown requested pair IDs: {sorted(unknown)}")
    return [pair for pair in pairs if pair["id"] in selected]


def segment_suite(data_root, detection_root, selected=None):
    """Run each requested full-frame category query exactly once, serially."""
    from .segmentation import detect_image
    root, views, pairs = manifest(data_root)
    needed = {pair[role] for pair in selected_pairs(pairs, selected) for role in ("source", "target")}
    for name, view in views.items():
        if name not in needed:
            continue
        print(f"SAM category query: {name} / {view['prompt']}", flush=True)
        detect_image(root / view["path"] / "rgb.png", view["prompt"], Path(detection_root) / name)


def control_detection(root, view):
    """Supply an annotated entity only in the explicitly named GT control arm."""
    truth = archive(root / view["path"] / "truth.npz")
    mask = np.asarray(truth["entity_mask"], dtype=bool)
    y, x = np.nonzero(mask)
    if not len(x):
        return []
    return [{"label": view["prompt"], "score": 1., "mask": mask,
             "bbox": [float(x.min()), float(y.min()), float(x.max()+1), float(y.max()+1)]}]


def prepare_suite(data_root, detection_root, run_root, arms=ARMS, selected=None):
    """Freeze point banks and crops before any descriptor or GT measurement.

    Source evidence is an explicit input, independent of the predicted entity
    mask. The segmentation module preserves both provenances in its point bank.
    Only the gt_mask control reads an entity truth file during preparation.
    """
    from .segmentation import load_detections, prepare_observation
    root, views, pairs = manifest(data_root)
    for arm in arms:
        for pair in selected_pairs(pairs, selected):
            directory = Path(run_root) / arm / pair["id"]
            directory.mkdir(parents=True, exist_ok=False)
            evidence = np.load(root / pair["evidence"], allow_pickle=False)
            np.save(directory / "source_evidence_mask.npy", evidence, allow_pickle=False)
            results, inputs = {}, {"mask_source": arm, "pair": pair["id"]}
            for role in ("source", "target"):
                view_id = pair[role]
                view = views[view_id]
                detections = (control_detection(root, view) if arm == "gt_mask"
                              else load_detections(Path(detection_root) / view_id))
                rgb, depth, K, T = read_observation_inputs(root, view)
                # BEGIN bind stored masks to their exact full-frame RGB input.
                if arm == "sam31":
                    recorded_rgb = np.array(Image.open(Path(detection_root) / view_id / "rgb.png").convert("RGB"))
                    if not np.array_equal(recorded_rgb, rgb):
                        raise ValueError("SAM detections belong to a different RGB observation")
                # END input identity check; no content hash or scene labels.
                result = prepare_observation(rgb, depth, K, T, detections,
                    prompt=view["prompt"], source_region_mask=evidence if role == "source" else None)
                results[role] = {key: value for key, value in result.items()
                                 if key not in ("observation", "crop_rgb", "source_evidence_mask")}
                # A selected image mask remains evidence even if its depth is
                # unusable. Preserve it independently of point-bank admission.
                selected_index = result["selection"]["selected_index"]
                if selected_index is not None:
                    np.save(directory / f"{role}_selected_mask.npy",
                            detections[selected_index]["mask"], allow_pickle=False)
                if result["prepared"]:
                    np.savez_compressed(directory / f"{role}_observation.npz", **result["observation"])
                    Image.fromarray(result["crop_rgb"]).save(directory / f"{role}.png")
                    inputs[f"{role}_crop_xyxy"] = result["crop_xyxy"]
                inputs[role] = str(root / view["path"] / "rgb.png")
                inputs[f"{role}_view"] = view_id
                inputs[f"{role}_prompt"] = view["prompt"]
            prepared = all(results[role]["prepared"] for role in results)
            reasons = [f"{role}:{reason}" for role, result in results.items() for reason in result["reasons"]]
            write_json(directory / "preparation.json", {
                "prepared": prepared, "reasons": reasons, "mask_source": arm,
                "source": results["source"], "target": results["target"],
                "target_truth_used": arm == "gt_mask"})
            write_json(directory / "inputs.json", inputs)
            print(json.dumps({"arm": arm, "pair": pair["id"], "prepared": prepared, "reasons": reasons}), flush=True)


def extract_suite(data_root, run_root, arms=ARMS, selected=None):
    """Use the unchanged MARCO worker on frozen prepared crops, one at a time."""
    from .features import extract_pair
    _, _, pairs = manifest(data_root)
    for arm in arms:
        for pair in selected_pairs(pairs, selected):
            directory = Path(run_root) / arm / pair["id"]
            preparation = json.loads((directory / "preparation.json").read_text())
            if not preparation["prepared"]:
                continue
            print(f"MARCO pair: {arm}/{pair['id']}", flush=True)
            extract_pair(directory / "source.png", directory / "target.png", directory / "features.npz")


def predict_suite(data_root, run_root, arms=ARMS, selected=None):
    """Freeze every direct prediction without reading target truth or metrics."""
    from .core import prepare_anchors, infer_region, apply_sim3
    from .features import native_region_pairs
    from .experiment import feature_observation, store_pairs, store_prediction
    _, _, pairs = manifest(data_root)
    for arm in arms:
        for pair in selected_pairs(pairs, selected):
            directory = Path(run_root) / arm / pair["id"]
            preparation = json.loads((directory / "preparation.json").read_text())
            if (directory / "prediction.json").exists():
                raise FileExistsError(directory / "prediction.json")
            if not preparation["prepared"]:
                write_json(directory / "prediction.json", {
                    "executed": False, "accepted": False, "reasons": preparation["reasons"],
                    "stage": "segmentation_preparation", "target_support_or_transform_truth_used": False,
                    "given_target_entity_mask": arm == "gt_mask"})
                continue
            source = archive(directory / "source_observation.npz")
            target = archive(directory / "target_observation.npz")
            inputs = json.loads((directory / "inputs.json").read_text())
            features = archive(directory / "features.npz")
            source_chart = feature_observation(source, inputs["source_crop_xyxy"])
            target_chart = feature_observation(target, inputs["target_crop_xyxy"])
            pairs_found = native_region_pairs(source_chart, target_chart, source["region_mask"],
                                             (features["feature1"], features["feature2"]))
            pairs_found["raw_target_pixels"] += np.asarray(inputs["target_crop_xyxy"][:2])
            pairs_found["pixel_space"] = "original full-frame RGB pixels"
            del features
            gc.collect()
            local = prepare_anchors(source["xyz"], target["xyz"], pairs_found["src_anchor_idx"],
                                    pairs_found["tgt_anchor_idx"], mode="direct")
            # Save all consensus memberships, not only the capped final anchors.
            # This is recomputed from the returned fit and uses no scene labels.
            if local.scale is not None:
                residual = np.linalg.norm(apply_sim3(source["xyz"][pairs_found["src_anchor_idx"]],
                    local.scale, local.rotation, local.translation) - target["xyz"][pairs_found["tgt_anchor_idx"]], axis=1)
                pairs_found["fitting_residual_m"] = residual
                pairs_found["fitting_inlier_mask"] = residual <= local.diagnostics["ransac_tolerance_m"]
            store_pairs(directory / "native_region_pairs.npz", pairs_found)
            result = infer_region(source["region_mask"], local, target_pixels=target["pixels"],
                source_camera_origin=source["T_world_camera_cv"][:3, 3],
                target_camera_origin=target["T_world_camera_cv"][:3, 3], method="direct_local", **SAMPLING)
            store_prediction(directory / "direct_local", result, {})
            write_json(directory / "prediction.json", {
                "executed": True, "accepted": result["accepted"], "reasons": result["reasons"],
                "stage": "direct_local", "target_support_or_transform_truth_used": False,
                "given_target_entity_mask": arm == "gt_mask", "sampling_budget": SAMPLING})
            print(json.dumps({"arm": arm, "pair": pair["id"], "accepted": result["accepted"],
                              "reasons": result["reasons"]}), flush=True)


def restore_prediction(directory):
    """Reconstruct read-only mode records from a frozen prediction archive."""
    metadata = json.loads((directory / "direct_local.json").read_text())
    values = archive(directory / "direct_local.npz")
    modes = []
    for index, description in enumerate(metadata["modes"]):
        mode = dict(description)
        for key in ("seed_index", "center", "normal", "mask", "sample_indices"):
            mode[key] = values[f"mode_{index}_{key}"]
        mode["seed_index"] = int(mode["seed_index"])
        modes.append(mode)
    return {"accepted": metadata["accepted"], "reasons": metadata["reasons"],
            "diagnostics": metadata["diagnostics"], "modes": modes}


def mask_measurements(selected_mask, truth, evidence, preferred):
    """Evaluate an unchanged selected mask; never repair it with owner labels."""
    actual = np.asarray(truth["entity_mask"], dtype=bool)
    selected = np.zeros_like(actual) if selected_mask is None else selected_mask
    intersection, union = int((selected & actual).sum()), int((selected | actual).sum())
    support = truth["support_id_image"] == preferred
    return {"entity_iou": intersection/union if union else None,
            "entity_precision": intersection/int(selected.sum()) if selected.any() else None,
            "entity_recall": intersection/int(actual.sum()) if actual.any() else None,
            "selected_pixels": int(selected.sum()), "true_entity_pixels": int(actual.sum()),
            "preferred_support_recall": int((selected & support).sum())/int(support.sum()) if support.any() else None,
            "true_preferred_support_pixels": int(support.sum()),
            "source_evidence_mask_coverage": int((selected & evidence).sum())/int(evidence.sum()) if evidence is not None and evidence.any() else None}


def render_comparison(source_image, target_image, observations, selected_masks, evidence, truth_target,
                      preferred, result, metrics, arm, output):
    """Display full-frame mask selection and proposals, including failed cases."""
    from PIL import ImageDraw, ImageFont
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
    source_rgb = np.array(Image.open(source_image).convert("RGB"))
    target_rgb = np.array(Image.open(target_image).convert("RGB"))
    source, target = observations["source"], observations["target"]

    def overlay(rgb, mask, color):
        rgb[mask] = (.45*rgb[mask]+.55*np.asarray(color)).astype(np.uint8)

    first, second, third, fourth = source_rgb.copy(), target_rgb.copy(), target_rgb.copy(), target_rgb.copy()
    if selected_masks["source"] is not None:
        overlay(first, selected_masks["source"], [45, 170, 255])
    overlay(first, evidence, [255, 45, 55])
    if selected_masks["target"] is not None:
        overlay(second, selected_masks["target"], [255, 205, 35])
    overlay(third, truth_target["support_id_image"] == preferred, [45, 170, 255])
    colors = [[45, 220, 90], [70, 140, 255], [215, 95, 235]]
    if target is not None:
        for index, mode in reversed(list(enumerate(result["modes"]))):
            pixel = target["pixels"][mode["mask"]]
            image_mask = np.zeros(target_rgb.shape[:2], dtype=bool)
            image_mask[pixel[:, 1], pixel[:, 0]] = True
            overlay(fourth, image_mask, colors[index])
    pictures = [Image.fromarray(rgb) for rgb in (first, second, third, fourth)]
    if target is not None:
        draw = ImageDraw.Draw(pictures[3])
        for index, mode in reversed(list(enumerate(result["modes"]))):
            for x, y in target["pixels"][mode["sample_indices"]]:
                draw.ellipse((x-1, y-1, x+1, y+1), fill=tuple(colors[index]), outline="black")
            x, y = target["pixels"][mode["seed_index"]]
            draw.line((x-5, y, x+5, y), fill="white", width=2)
            draw.line((x, y-5, x, y+5), fill="white", width=2)
    width, height = target_rgb.shape[1], target_rgb.shape[0]
    sheet = Image.new("RGB", (2*width, 2*(height+64)), "white")
    accepted = "accepted" if result["accepted"] else "rejected"
    error = metrics["center_error_m"]
    error_text = "no primary" if error is None else f"reference distance {1000*error:.1f} mm"
    titles = [(f"{arm}: source selection", "Blue: entity mask; red: unchanged given evidence"),
              ("Target selection", "Yellow: unchanged selected mask; no GT filtering"),
              ("Held-out intended support", "Blue: real owner AND corresponding physical surface"),
              (f"Direct transfer: {accepted}; {error_text}", "Green/blue/purple: ordered proposals; not verified placements")]
    draw = ImageDraw.Draw(sheet)
    for index, (picture, title) in enumerate(zip(pictures, titles)):
        x, y = index % 2*width, index // 2*(height+64)
        sheet.paste(picture.resize((width, height)), (x, y+64))
        draw.text((x+8, y+7), title[0], fill="black", font=font)
        draw.text((x+8, y+31), title[1], fill="black", font=font)
    sheet.save(output)


def evaluate_suite(data_root, run_root, arms=ARMS, selected=None):
    """Open GT only after all requested predictions have been frozen on disk."""
    from .evaluation import evaluate_prediction
    root, views, all_pairs = manifest(data_root)
    pairs = selected_pairs(all_pairs, selected)
    run_root = Path(run_root)
    for arm in arms:
        for pair in pairs:
            if not (run_root / arm / pair["id"] / "prediction.json").exists():
                raise FileNotFoundError("Finish every prediction before evaluating target truth")
    rows = []
    for arm in arms:
        for pair in pairs:
            directory = run_root / arm / pair["id"]
            prediction = json.loads((directory / "prediction.json").read_text())
            preparation = json.loads((directory / "preparation.json").read_text())
            truth_source = archive(root / views[pair["source"]]["path"] / "truth.npz")
            truth_target = archive(root / views[pair["target"]]["path"] / "truth.npz")
            pair_truth = json.loads((root / pair["pair_truth"]).read_text())
            preferred = int(pair_truth["preferred_support_id"])
            reference = np.asarray(pair_truth["target_reference_evidence_center_world"])
            observations = {role: archive(directory / f"{role}_observation.npz")
                            if (directory / f"{role}_observation.npz").exists() else None
                            for role in ("source", "target")}
            source, target = observations["source"], observations["target"]
            selected_masks = {}
            for role in ("source", "target"):
                path = directory / f"{role}_selected_mask.npy"
                if path.exists() != (preparation[role]["selection"]["selected_index"] is not None):
                    raise ValueError("Saved selected mask disagrees with the selection record")
                selected_masks[role] = np.load(path, allow_pickle=False) if path.exists() else None
            evidence = np.load(root / pair["evidence"], allow_pickle=False)
            mask_metrics = {"source": mask_measurements(selected_masks["source"], truth_source, evidence, preferred),
                            "target": mask_measurements(selected_masks["target"], truth_target, None, preferred)}
            if prediction["executed"]:
                result = restore_prediction(directory)
            else:
                result = {"accepted": False, "reasons": prediction["reasons"], "diagnostics": {}, "modes": []}
            if target is None:
                target_points, labels = np.empty((0, 3)), np.empty(0, dtype=np.int32)
            else:
                x, y = target["pixels"].T
                target_points, labels = target["xyz"], truth_target["support_id_image"][y, x]
            aligned_truth = {"support_id": labels, "preferred_support_id": np.asarray(preferred, np.int32)}
            metrics = evaluate_prediction(result, target_points, aligned_truth, reference)
            write_json(directory / "metrics.json", metrics)
            report = {"pair": pair, "mask_source": arm, "prepared": preparation["prepared"],
                      "prediction": prediction, "mask_metrics": mask_metrics, "metrics": metrics,
                      "reference_evidence_center": reference,
                      "reference_definition": pair_truth.get("reference_definition", "unchanged full source evidence mean under independent scene geometry"),
                      "target_support_or_transform_truth_used_by_inference": False,
                      "given_target_entity_mask": arm == "gt_mask"}
            write_json(directory / "report.json", report)
            row = {"mask_source": arm, "pair": pair["id"], "group": pair["group"], "variation": pair["variation"],
                   "prepared": preparation["prepared"], "prediction_executed": prediction["executed"],
                   "reasons": ";".join(prediction["reasons"]),
                   **{key: value for key, value in metrics.items() if key != "mode_metrics"},
                   **{f"{role}_{key}": value for role, record in mask_metrics.items() for key, value in record.items()}}
            rows.append(row)
            render_comparison(root / views[pair["source"]]["path"] / "rgb.png",
                root / views[pair["target"]]["path"] / "rgb.png", observations, selected_masks, evidence,
                truth_target, preferred, result, metrics, arm, directory / "comparison.png")
    summary = {"protocol": "all declared pairs included; gt_mask is an explicit control, not a fallback",
               "pairs_per_arm": len(pairs), "rows": rows}
    write_json(run_root / "summary.json", summary)
    with (run_root / "metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"evaluated": len(rows), "pairs_per_arm": len(pairs),
                      "accepted": {arm: sum(r["accepted"] for r in rows if r["mask_source"] == arm) for arm in arms}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    segment = commands.add_parser("segment")
    segment.add_argument("--data", type=Path, required=True)
    segment.add_argument("--detections", type=Path, required=True)
    segment.add_argument("--pair", action="append")
    for command in ("prepare", "extract", "predict", "evaluate"):
        sub = commands.add_parser(command)
        sub.add_argument("--data", type=Path, required=True)
        sub.add_argument("--run", type=Path, required=True)
        sub.add_argument("--arm", action="append", choices=ARMS)
        sub.add_argument("--pair", action="append")
        if command == "prepare":
            sub.add_argument("--detections", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "segment":
        segment_suite(args.data, args.detections, args.pair)
    elif args.command == "prepare":
        prepare_suite(args.data, args.detections, args.run, args.arm or ARMS, args.pair)
    else:
        {"extract": extract_suite, "predict": predict_suite, "evaluate": evaluate_suite}[args.command](
            args.data, args.run, args.arm or ARMS, args.pair)


if __name__ == "__main__":
    main()
# END segmented transfer experiment.
