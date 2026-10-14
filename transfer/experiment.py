"""Run independent, manifest-driven RGB-D region-transfer comparisons.

The manifest names explicit source/target observation directories. Category
names are report labels only. Target evaluation files are opened only after
all automatic methods have finished; their geometry is then available solely
to metrics and to the explicitly named oracle diagnostic.
"""

import argparse
import csv
import gc
import json
from pathlib import Path

import numpy as np

from .features import (extract_pair, native_region_pairs, reciprocal_anchors,
                       write_json)
from .core import fps_indices


def read_archive(path):
    """Load a typed local record, rejecting object arrays and hidden pickle."""
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def read_pairs(data_root):
    """Require explicit, unique experiment pairs inside the selected data root."""
    data_root = Path(data_root).resolve()
    manifest = json.loads((data_root / "manifest.json").read_text())
    pairs = manifest["pairs"]
    ids = [pair["id"] for pair in pairs]
    if len(ids) != len(set(ids)) or not pairs:
        raise ValueError("Manifest requires distinct, nonempty experiment pairs")
    for pair in pairs:
        if Path(pair["id"]).name != pair["id"]:
            raise ValueError("Pair IDs must be single directory names")
        for role in ("source", "target"):
            directory = (data_root / pair[role]).resolve()
            if not directory.is_relative_to(data_root):
                raise ValueError("Pair observations must remain within the data root")
            for filename in ("rgb.png", "observation.npz"):
                if not (directory / filename).is_file():
                    raise FileNotFoundError(directory / filename)
    return pairs


def extract_suite(data_root, output_root, selected=None):
    """Extract each requested pair serially; existing evidence is not overwritten."""
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    for pair in read_pairs(data_root):
        if selected is not None and pair["id"] not in selected:
            continue
        directory = output_root / pair["id"]
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / "features.npz"
        if output.exists():
            raise FileExistsError(f"Feature evidence already exists: {output}; use a fresh output")
        source_image = data_root / pair["source"] / "rgb.png"
        target_image = data_root / pair["target"] / "rgb.png"
        # The given whole-entity mask defines one image crop per role. This
        # uses no local target annotation and avoids spending the model's
        # finite feature resolution on the original table and robot.
        crop_records = {}
        for role, image_path in (("source", source_image), ("target", target_image)):
            from PIL import Image
            observation = read_archive(data_root / pair[role] / "observation.npz")
            ys, xs = np.nonzero(observation["object_mask"])
            if not len(xs):
                raise ValueError("Cannot crop an unobserved entity")
            margin = int(np.ceil(0.15 * max(np.ptp(xs) + 1, np.ptp(ys) + 1)))
            width, height = observation["object_mask"].shape[::-1]
            bounds = [max(0, int(xs.min()) - margin), max(0, int(ys.min()) - margin),
                      min(width, int(xs.max()) + margin + 1),
                      min(height, int(ys.max()) + margin + 1)]
            Image.open(image_path).convert("RGB").crop(bounds).save(directory / f"{role}.png")
            crop_records[role + "_crop_xyxy"] = bounds
        print(f"MARCO native forward_backbone: {pair['id']}", flush=True)
        extract_pair(directory / "source.png", directory / "target.png", output)
        write_json(directory / "inputs.json", {
            "source": str(source_image), "target": str(target_image),
            "profile": "MARCO FP32 native 840 resize/padding, 1024x240x240 feature maps",
            "source_target_masks": "supplied whole-entity rendered segmentation; no target patch",
            "crop_margin_fraction": 0.15, **crop_records,
        })


def feature_observation(observation, bounds):
    """Change only the pixel chart to a recorded crop; keep point identities."""
    left, top, right, bottom = map(int, bounds)
    height, width = observation["object_mask"].shape
    if not (0 <= left < right <= width and 0 <= top < bottom <= height):
        raise ValueError("Feature crop bounds exceed their source observation")
    pixels = observation["pixels"] - np.array([left, top])
    if np.any(pixels < 0) or np.any(pixels >= [right - left, bottom - top]):
        raise ValueError("Feature crop excluded an observed entity point")
    return {"xyz": observation["xyz"], "pixels": pixels,
            "object_mask": observation["object_mask"][top:bottom, left:right]}


def store_pairs(path, pairs):
    """Save only typed arrays; descriptive metadata gets a separate JSON file."""
    np.savez_compressed(path, **{key: value for key, value in pairs.items()
                                if isinstance(value, np.ndarray)})
    write_json(Path(path).with_suffix(".json"), {
        key: value for key, value in pairs.items() if not isinstance(value, np.ndarray)})


def store_prediction(directory, result, metrics):
    """Keep diagnostic candidate arrays separate from the acceptance decision."""
    arrays = {key: np.asarray(result[key]) for key in (
                  "candidate_mask", "candidate_center", "candidate_surface", "src_anchor_idx",
                  "tgt_anchor_idx", "sim3_scale", "sim3_rotation", "sim3_translation")
              if result.get(key) is not None}
    arrays["accepted"] = np.asarray(bool(result["accepted"]))
    np.savez_compressed(directory.with_suffix(".npz"), **arrays)
    write_json(directory.with_suffix(".json"), {
        "accepted": bool(result["accepted"]), "reasons": result["reasons"],
        "diagnostics": result["diagnostics"], "metrics": metrics})


def evaluate_suite(data_root, features_root, output_root, selected=None):
    """Run the five frozen methods on each actual rendered observation pair."""
    from .core import TransferConfig, prepare_anchors, infer_region
    from .evaluation import (comparison_image, evaluate_prediction,
                             native_correspondence_metrics, oracle_correspondences)

    data_root, features_root = Path(data_root).resolve(), Path(features_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    config = TransferConfig()
    rows, reports = [], []
    for pair in read_pairs(data_root):
        if selected is not None and pair["id"] not in selected:
            continue
        directory = output_root / pair["id"]
        directory.mkdir(parents=True, exist_ok=False)
        source_directory = data_root / pair["source"]
        target_directory = data_root / pair["target"]
        source = read_archive(source_directory / "observation.npz")
        target = read_archive(target_directory / "observation.npz")
        # The source demonstration region is an input. No target evaluation
        # record exists in memory until the four automatic methods finish.
        source_truth = read_archive(source_directory / "evaluation.npz")
        source_region = source_truth["region_mask"]
        archive = read_archive(features_root / pair["id"] / "features.npz")
        feature_pair = (archive["feature1"], archive["feature2"])
        chart = json.loads((features_root / pair["id"] / "inputs.json").read_text())
        source_chart = feature_observation(source, chart["source_crop_xyxy"])
        target_chart = feature_observation(target, chart["target_crop_xyxy"])
        print(f"automatic correspondence + transfer: {pair['id']}", flush=True)
        anchors = reciprocal_anchors(source_chart, target_chart, feature_pair)
        direct = native_region_pairs(source_chart, target_chart, source_region, feature_pair)
        direct["raw_target_pixels"] += np.asarray(chart["target_crop_xyxy"][:2])
        direct["pixel_space"] = "original rendered RGB pixels; recorded crop offset restored"
        store_pairs(directory / "automatic_anchors.npz", anchors)
        store_pairs(directory / "native_region_pairs.npz", direct)
        del archive, feature_pair
        gc.collect()

        results = {}
        # The numerical core returns geometric rejection as ordinary data.
        # Contract/programming errors must stop this run, not be mislabeled as
        # a failed transfer and silently included in its performance figures.
        prepared = prepare_anchors(source["xyz"], target["xyz"],
                                   anchors["src_anchor_idx"], anchors["tgt_anchor_idx"],
                                   mode="automatic", config=config)
        for name in ("global_sim3", "relation_only", "relational"):
            results[name] = infer_region(source_region, prepared, method=name)
        local = prepare_anchors(source["xyz"], target["xyz"],
                                direct["src_anchor_idx"], direct["tgt_anchor_idx"],
                                mode="direct", config=config)
        results["direct_local"] = infer_region(source_region, local, method="direct_local")

        # The target labels become accessible here, after automatic inference.
        target_truth = read_archive(target_directory / "evaluation.npz")
        oracle_source, oracle_target, oracle_stats = oracle_correspondences(
            source, target, source_truth, target_truth, fps_indices(source["xyz"], 450))
        np.savez_compressed(directory / "oracle_anchors.npz", src_anchor_idx=oracle_source,
                            tgt_anchor_idx=oracle_target)
        oracle = prepare_anchors(source["xyz"], target["xyz"], oracle_source, oracle_target,
                                 mode="oracle", config=config)
        results["oracle_relational"] = infer_region(source_region, oracle, method="oracle_relational")
        metrics = {name: evaluate_prediction(result, target["xyz"], target_truth)
                   for name, result in results.items()}
        for name, result in results.items():
            store_prediction(directory / name, result, metrics[name])
            rows.append({"pair": pair["id"], "group": pair.get("group", ""),
                         "variation": pair.get("variation", ""), "method": name,
                         **metrics[name], "reasons": "; ".join(result["reasons"])})
        comparison_image(source_directory / "rgb.png", target_directory / "rgb.png",
                         source, target, source_region, target_truth["region_mask"],
                         results, metrics, directory / "comparison.png")
        report = {"pair": pair, "automatic_anchor_pairs": int(len(anchors["src_anchor_idx"])),
                  "direct_raw_matches": native_correspondence_metrics(direct, source_truth, target_truth),
                  "oracle": oracle_stats, "metrics": metrics,
                  "source_visible_patch_points": int(source_region.sum()),
                  "source_observed_centroid_to_declared_center_m": float(np.linalg.norm(
                      source["xyz"][source_region].mean(0) - source_truth["region_center"])),
                  "target_patch_used_by_automatic_inference": False,
                  "provided_entity_segmentation": True,
                  "source_region_visible_area_fraction": float(source_truth["region_visible_area_fraction"]),
                  "target_region_visible_area_fraction": float(target_truth["region_visible_area_fraction"]),
                  "source_region_center_visible": bool(source_truth["region_center_visible"]),
                  "target_region_center_visible": bool(target_truth["region_center_visible"])}
        write_json(directory / "report.json", report)
        reports.append(report)
        print(json.dumps({"pair": pair["id"], "methods": metrics}, allow_nan=False), flush=True)

    write_json(output_root / "summary.json", {"pairs": reports, "rows": rows})
    if rows:
        columns = list(dict.fromkeys(key for row in rows for key in row))
        with (output_root / "metrics.csv").open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    return rows


def audit_suite(data_root, results_root):
    """Explain saved correspondence errors without recomputing any prediction.

    This deliberately post-hoc stage can use target labels: it reads frozen
    predictions and reports their evidence, never feeds its errors back to the
    matcher, registration, thresholds, or scene construction.
    """
    from .evaluation import anchor_correspondence_metrics, summary_plot
    from PIL import Image, ImageDraw, ImageFont

    data_root, results_root = Path(data_root), Path(results_root)
    records = []
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 17)
    for pair in read_pairs(data_root):
        directory = results_root / pair["id"]
        if not (directory / "report.json").is_file():
            continue
        source_path, target_path = data_root / pair["source"], data_root / pair["target"]
        source, target = read_archive(source_path / "observation.npz"), read_archive(target_path / "observation.npz")
        source_truth, target_truth = read_archive(source_path / "evaluation.npz"), read_archive(target_path / "evaluation.npz")
        all_pairs = read_archive(directory / "automatic_anchors.npz")
        fitted = read_archive(directory / "global_sim3.npz")
        record = {"pair": pair["id"], "automatic_raw": anchor_correspondence_metrics(
            all_pairs["src_anchor_idx"], all_pairs["tgt_anchor_idx"], source_truth, target_truth)}
        ids, idt = fitted.get("src_anchor_idx", np.empty(0, int)), fitted.get("tgt_anchor_idx", np.empty(0, int))
        record["automatic_selected"] = anchor_correspondence_metrics(ids, idt, source_truth, target_truth)
        oracle = read_archive(directory / "oracle_anchors.npz")
        record["oracle_visible"] = anchor_correspondence_metrics(
            oracle["src_anchor_idx"], oracle["tgt_anchor_idx"], source_truth, target_truth)
        first = Image.open(source_path / "rgb.png").convert("RGB")
        second = Image.open(target_path / "rgb.png").convert("RGB")
        sheet = Image.new("RGB", (first.width + second.width, first.height + 62), "white")
        sheet.paste(first, (0, 62))
        sheet.paste(second, (first.width, 62))
        draw = ImageDraw.Draw(sheet)
        draw.text((10, 8), f"{pair['id']}: selected automatic anchors ({len(ids)})", font=font, fill="black")
        draw.text((10, 32), "Line color uses held-out point error for inspection only: green <-> red (0 to 3 cm)", font=font, fill="black")
        if len(ids):
            delta = target_truth["canonical_xyz"][idt] - source_truth["canonical_xyz"][ids]
            error = np.linalg.norm(delta @ target_truth["canonical_to_world"][:3, :3].T, axis=1)
            for source_index, target_index, distance in zip(ids, idt, error):
                source_pixel, target_pixel = source["pixels"][source_index], target["pixels"][target_index]
                start = tuple((source_pixel + [0, 62]).tolist())
                end = tuple((target_pixel + [first.width, 62]).tolist())
                fraction = min(float(distance / .03), 1.)
                color = (int(40 + 210 * fraction), int(220 - 170 * fraction), 50)
                draw.line([start, end], fill=color, width=1)
                for x, y in (start, end):
                    draw.ellipse((x-3, y-3, x+3, y+3), fill=color)
        sheet.save(directory / "anchor-audit.png")
        write_json(directory / "anchor-audit.json", record)
        records.append(record)
    write_json(results_root / "anchor-audit.json", records)
    summary_plot(json.loads((results_root / "summary.json").read_text())["rows"],
                 results_root / "comparison-metrics.png")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="serial offline MARCO model extraction")
    extract.add_argument("--data", required=True, type=Path)
    extract.add_argument("--out", required=True, type=Path)
    extract.add_argument("--pair", action="append")
    evaluate = sub.add_parser("evaluate", help="CPU inference, held-out metrics and overlays")
    evaluate.add_argument("--data", required=True, type=Path)
    evaluate.add_argument("--features", required=True, type=Path)
    evaluate.add_argument("--out", required=True, type=Path)
    evaluate.add_argument("--pair", action="append")
    audit = sub.add_parser("audit", help="post-hoc correspondence errors on frozen predictions")
    audit.add_argument("--data", required=True, type=Path)
    audit.add_argument("--results", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "extract":
        extract_suite(args.data, args.out, args.pair)
    elif args.command == "evaluate":
        evaluate_suite(args.data, args.features, args.out, args.pair)
    else:
        audit_suite(args.data, args.results)


if __name__ == "__main__":
    main()
