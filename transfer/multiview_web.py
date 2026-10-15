"""Export evaluated two-view results to one offline, interactive HTML file.

This presentation step reads frozen results and reuses the existing original-
pixel renderer. It never loads feature tensors, fits a transform, or opens
evaluation truth. Optional comparison rows must explicitly use the current
case IDs; dataset-specific naming belongs in that supplied data, not here.
"""

import argparse
import base64
from io import BytesIO
import json
import os
from pathlib import Path

import numpy as np

from .multiview_experiment import (ALGORITHMS, _path, load_fusion, load_inputs,
                                    read_manifest, restore_prediction)
from .multiview_visualization import (_read_rgb, _overlay, _draw_modes,
    _observation_domains, _mode_pixels, _summary_rows, _RED, _SAM, _GREEN, _SECONDARY)


# BEGIN offline presentation; all inference and evaluation artifacts stay frozen.
def image_uri(image):
    """Embed actual observed imagery so file:// browsing requires no server."""
    stream = BytesIO()
    image.save(stream, format="PNG")
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def metric_row(record):
    """Keep absent metrics null; rejected outputs retain their diagnostic values."""
    quality, diagnostics = record["quality"], record["diagnostics"]
    region = quality["region"]
    transform = quality["true_source_entity_transform"]
    return {"algorithm": record["algorithm"], "accepted": record["accepted"],
        # BEGIN required warnings stay distinct from stored hard rejection reasons.
        "reasons": record["reasons"], "quality_warnings": diagnostics["quality_warnings"],
        # END warning presentation does not recalculate historical acceptance.
        "object_rmse_mm": None if transform is None else
            1000 * transform["source_points_transform_rmse_m"],
        "center_error_mm": None if region["center_error_m"] is None else 1000 * region["center_error_m"],
        "registration_ms": record["registration_and_anchor_preparation_ms"],
        "region_ms": record["region_generation_ms"], "scale": record["scale"],
        "rotation_error_deg": None if transform is None else transform["rotation_error_deg"],
        "inliers": diagnostics.get("inliers", 0), "correspondences": record["correspondence_count"],
        "minimum_inlier_ratio": diagnostics["configuration"]["min_inlier_ratio"],
        "tau_mm": 1000 * diagnostics["ransac_tolerance_m"],
        "mode_count": region["mode_count"], "samples": region["top1_sample_count"],
        "sample_purity": region["top1_sample_preferred_support_fraction"],
        "grounding": record.get("grounding_diagnostics"),
        "correspondence_quality": quality.get("correspondence"),
        "raw_quality": quality.get("raw_correspondence"),
        "single_view_quality": quality.get("single_view_rank_correspondence")}


def export_web(manifest, results, destination, comparisons=None):
    """Build a self-contained page from a completed matrix and explicit inputs."""
    manifest_path, description = read_manifest(manifest)
    results, destination = Path(results).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(destination)
    summary = json.loads((results / "summary.json").read_text())
    rows = _summary_rows(summary, description["cases"])
    comparison_groups = [] if comparisons is None else json.loads(Path(comparisons).read_text())
    case_ids = {case["id"] for case in description["cases"]}
    for group in comparison_groups:
        if not isinstance(group["label"], str):
            raise ValueError("Comparison labels must be explicit strings")
        seen = set()
        for record in group["rows"]:
            key = (record["case"], record["algorithm"])
            if key in seen or key[0] not in case_ids or key[1] not in ALGORITHMS:
                raise ValueError("Comparison rows need unique declared case/algorithm identities")
            seen.add(key)
        group["rows"] = [{"case": record["case"], **metric_row(record)} for record in group["rows"]]

    cases = []
    for case in description["cases"]:
        source, _, _, targets = load_inputs(manifest_path, case)
        fused = load_fusion(results / "fusion" / case["id"])
        bank, diagnostics = fused["bank"], fused["diagnostics"]
        domains = _observation_domains(bank, targets)
        source_rgb = _read_rgb(_path(manifest_path, case["source_observation_dir"]) / "rgb.png")
        evidence = np.load(_path(manifest_path, case["source_input_dir"]) / "source_evidence_mask.npy",
                           allow_pickle=False)
        source_pixels = source["pixels"]
        np.testing.assert_array_equal(evidence[source_pixels[:, 1], source_pixels[:, 0]], source["region_mask"])
        target_declarations = {record["id"]: record for record in case["targets"]}
        original_images, view_records = {}, []
        for target in targets:
            view_id = target["id"]
            original_images[view_id] = _read_rgb(_path(manifest_path,
                target_declarations[view_id]["observation_dir"]) / "rgb.png")
            view_records.append({"id": view_id, "points": len(target["observation"]["xyz"]),
                "missing": target["missing_target"], "reasons": target["missing_target_reasons"],
                "input": image_uri(_overlay(original_images[view_id], target["selected_mask"], _SAM, 70))})
        predictions = []
        for algorithm in ALGORITHMS:
            record = rows[(case["id"], algorithm)]
            prediction = restore_prediction(results / case["id"] / algorithm / "prediction")
            # BEGIN reject stale summaries instead of silently losing warnings.
            if (prediction["accepted"] != record["accepted"]
                    or prediction["reasons"] != record["reasons"]
                    or prediction["diagnostics"]["quality_warnings"] != record["diagnostics"]["quality_warnings"]):
                raise ValueError("Frozen prediction and evaluated summary disagree")
            # END frozen warning evidence must agree before presentation.
            entries = {target["id"]: [] for target in targets}
            for number, mode in enumerate(prediction["modes"]):
                mask, pixels = _mode_pixels(mode, domains, len(bank["observation_xyz"]))
                color = _GREEN if mode["type"] == "primary" else _SECONDARY[(number - 1) % 2]
                entries[mode["view_id"]].append((number, mode, mask, pixels, color))
            observed_views = {}
            for view_id, items in entries.items():
                observed_views[view_id] = {"image": image_uri(_draw_modes(original_images[view_id], items)),
                    "modes": [{"number": number + 1, "primary": mode["type"] == "primary",
                        "points": int(mask.sum()), "samples": len(pixels)}
                        for number, mode, mask, pixels, _ in items]}
            predictions.append({**metric_row(record), "views": observed_views})
        cases.append({"id": case["id"], "source": image_uri(_overlay(source_rgb, evidence, _RED, 130)),
            "source_points": int(source["region_mask"].sum()), "views": view_records,
            "diagnostics": diagnostics, "predictions": predictions,
            "truth_mapping_type": summary["ground_truth"][case["id"]]["mapping_type"],
            "true_scale": summary["ground_truth"][case["id"]]["true_scale"],
            "fusion_ms": rows[(case["id"], ALGORITHMS[0])]["fusion_timing"]["fusion_ms"]})

    # Links remain relative for in-workspace provenance; all interactive data
    # and images are embedded and still work if this single HTML is copied.
    links = [{"label": label, "href": os.path.relpath(path, destination.parent)}
        for label, path in (("完整报告", results.parent / "results.md"),
                            ("独立审核", results.parent / "independent-audit.md"),
                            ("九组 JSON", results / "summary.json"), ("九组 CSV", results / "summary.csv"))]
    data = {"cases": cases, "comparisons": comparison_groups, "links": links,
        "protocol": summary["protocol"]}
    template = Path(__file__).with_suffix(".html").read_text()
    # JSON is data, not executable interpolation. Escaping '<' also protects
    # against a future observation label containing an HTML script end tag.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    if template.count("__EVIDENCE_JSON__") != 1:
        raise ValueError("The local template must contain exactly one evidence placeholder")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(template.replace("__EVIDENCE_JSON__", payload))
    print(json.dumps({"html": str(destination), "cases": len(cases), "predictions": len(rows),
        "bytes": destination.stat().st_size, "self_contained": True, "inference_calls": 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New standalone HTML file")
    parser.add_argument("--comparisons", type=Path, help="Explicitly case-aligned evaluated comparison rows")
    args = parser.parse_args()
    export_web(args.manifest, args.results, args.output, args.comparisons)


if __name__ == "__main__":
    main()
# END offline presentation.
