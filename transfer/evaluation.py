"""Ground-truth metrics and image reports, separate from transfer inference.

All masks index the same observed target points. A rejected candidate can be
measured for diagnosis, but is never counted as an accepted successful result.
The module imports neither the matcher nor the transfer solver.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


def evaluate_prediction(result, target_points, truth):
    """Measure the returned center and mask against independent scene labels."""
    points = np.asarray(target_points, dtype=float)
    gt = np.asarray(truth["region_mask"])
    support = np.asarray(truth["support_mask"])
    for name, mask in (("region_mask", gt), ("support_mask", support)):
        if mask.dtype != np.bool_ or mask.shape != (len(points),):
            raise ValueError(f"Evaluation {name} must be Boolean and point-aligned")
    true_center = np.asarray(truth["region_center"], dtype=float)
    if true_center.shape != (3,) or not np.isfinite(true_center).all():
        raise ValueError("Evaluation center must be a finite world point")
    radius = float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))
    if radius <= 0:
        raise ValueError("Evaluation entity has no spatial extent")
    metrics = {"accepted": bool(result["accepted"]), "gt_visible_points": int(gt.sum()),
               "target_rms_radius_m": radius, "region_iou": None,
               "center_error_m": None, "center_error_normalized": None,
               "predicted_mask_centroid_error_m": None,
               "support_hit": False, "nearest_observation_distance_m": None}
    mask_value = result.get("candidate_mask")
    center_value = result.get("candidate_center")
    if mask_value is not None:
        mask = np.asarray(mask_value)
        if mask.dtype != np.bool_ or mask.shape != (len(points),):
            raise ValueError("Prediction mask is not Boolean/point-aligned")
        union = int((mask | gt).sum())
        metrics["region_iou"] = float((mask & gt).sum() / union) if union else None
        metrics["predicted_points"] = int(mask.sum())
        if mask.any():
            metrics["predicted_mask_centroid_error_m"] = float(
                np.linalg.norm(points[mask].mean(0) - true_center))
            metrics["support_precision"] = float(support[mask].mean())
    if center_value is not None:
        center = np.asarray(center_value, dtype=float)
        if center.shape != (3,) or not np.isfinite(center).all():
            raise ValueError("Prediction center must be finite or explicitly absent")
        error = float(np.linalg.norm(center - true_center))
        distances = np.linalg.norm(points - center, axis=1)
        nearest = int(np.argmin(distances))
        # This reports which observed surface owns the closest measurement;
        # the distance is separate, rather than silently declaring free space
        # above the surface to be a measured contact point.
        metrics.update(center_error_m=error, center_error_normalized=error / radius,
                       support_hit=bool(support[nearest]),
                       nearest_observation_distance_m=float(distances[nearest]))
    metrics["accepted_support_hit"] = bool(metrics["accepted"] and metrics["support_hit"])
    return metrics


def oracle_correspondences(source, target, source_truth, target_truth, sample_indices):
    """Match known canonical surface coordinates for an explicit diagnostic.

    These coordinates are produced from the scene fixture, never from an
    estimated Sim(3). The nearest observed target sample must represent the
    same rendered geometry owner and be within two observed sample spacings.
    A mesh owner can contain several faces, so this remains an approximate
    correspondence diagnostic. Missing visibility remains missing; no hidden
    point is appended to the input.
    """
    source_canonical = np.asarray(source_truth["canonical_xyz"], dtype=float)
    target_canonical = np.asarray(target_truth["canonical_xyz"], dtype=float)
    if (source_canonical.shape != np.asarray(source["xyz"]).shape
            or target_canonical.shape != np.asarray(target["xyz"]).shape
            or not np.isfinite(source_canonical).all()
            or not np.isfinite(target_canonical).all()):
        raise ValueError("Oracle coordinates must align with finite observed points")
    tree = cKDTree(target_canonical)
    nearest_spacing = tree.query(target_canonical, k=2)[0][:, 1]
    positive = nearest_spacing[nearest_spacing > 1e-9]
    if not len(positive):
        raise ValueError("Oracle target has no distinct observed coordinates")
    tolerance = float(2 * np.median(positive))
    samples = np.asarray(sample_indices, dtype=np.int64)
    distance, target_indices = tree.query(source_canonical[samples])
    valid = distance <= tolerance
    # Scene surface identities are used only by this oracle diagnostic to
    # avoid a different geometry owner being called a matching observation.
    # One mesh can contain several faces; this is not a triangle identity test.
    if "surface_id" in source_truth and "surface_id" in target_truth:
        valid &= (np.asarray(source_truth["surface_id"])[samples]
                  == np.asarray(target_truth["surface_id"])[target_indices])
    retained = []
    for target_index in np.unique(target_indices[valid]):
        candidates = np.flatnonzero(valid & (target_indices == target_index))
        retained.append(int(candidates[np.argmin(distance[candidates])]))
    retained = np.asarray(retained, dtype=np.int64)
    return samples[retained], target_indices[retained].astype(np.int64), {
        "mode": "oracle canonical visible-surface correspondence; diagnostic only",
        "source_sample_count": int(len(samples)), "visible_pair_count": int(len(retained)),
        "canonical_tolerance_m": tolerance,
        "median_canonical_error_m": float(np.median(distance[retained])) if len(retained) else None,
    }


def native_correspondence_metrics(pairs, source_truth, target_truth):
    """Report raw region matches before their local geometric fitting step."""
    source_indices = np.asarray(pairs["raw_source_idx"])
    target_indices = np.asarray(pairs["raw_target_idx"])
    valid = np.asarray(pairs["raw_valid"])
    metrics = {"query_count": int(len(valid)), "valid_count": int(valid.sum()),
               "distinct_target_count": int(len(np.unique(target_indices[valid]))),
               "median_canonical_error_m": None}
    if valid.any() and "canonical_xyz" in source_truth and "canonical_xyz" in target_truth:
        source_points = np.asarray(source_truth["canonical_xyz"])[source_indices[valid]]
        target_points = np.asarray(target_truth["canonical_xyz"])[target_indices[valid]]
        metrics["median_canonical_error_m"] = float(
            np.median(np.linalg.norm(source_points - target_points, axis=1)))
    return metrics


def anchor_correspondence_metrics(source_indices, target_indices, source_truth, target_truth):
    """Audit anchor accuracy against known geometry after inference has ended.

    Predicted target world positions come from the evaluator's independent
    canonical transformation, not from the transfer method's estimated Sim3.
    Geometry-owner agreement is reported separately from metric point error.
    """
    source_indices = np.asarray(source_indices, dtype=np.int64)
    target_indices = np.asarray(target_indices, dtype=np.int64)
    result = {"count": int(len(source_indices)), "median_world_error_m": None,
              "p90_world_error_m": None, "geometry_owner_agreement": None}
    if not len(source_indices):
        return result
    source_canonical = np.asarray(source_truth["canonical_xyz"])[source_indices]
    target_canonical = np.asarray(target_truth["canonical_xyz"])[target_indices]
    linear = np.asarray(target_truth["canonical_to_world"])[:3, :3]
    error = np.linalg.norm((target_canonical - source_canonical) @ linear.T, axis=1)
    result.update(median_world_error_m=float(np.median(error)),
                  p90_world_error_m=float(np.quantile(error, .9)))
    if "surface_id" in source_truth and "surface_id" in target_truth:
        result["geometry_owner_agreement"] = float(np.mean(
            np.asarray(source_truth["surface_id"])[source_indices]
            == np.asarray(target_truth["surface_id"])[target_indices]))
    return result


def comparison_image(source_image, target_image, source, target, source_region,
                     target_region, results, metrics, output):
    """Render aligned visible-point overlays without inventing surface pixels."""
    from PIL import Image, ImageDraw, ImageFont

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16)
    small = ImageFont.truetype(font_path, 13)
    tile_width, tile_height = 480, 424
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 2), "white")

    def draw_tile(image_path, observation, mask, color, title, subtitle, slot):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        if mask is not None:
            pix = np.asarray(observation["pixels"])[np.asarray(mask, dtype=bool)].astype(int)
            rgb[pix[:, 1], pix[:, 0]] = (
                0.35 * rgb[pix[:, 1], pix[:, 0]] + 0.65 * np.asarray(color)).astype(np.uint8)
        im = Image.fromarray(rgb).resize((480, 360))
        x, y = (slot % 4) * tile_width, (slot // 4) * tile_height
        sheet.paste(im, (x, y + 64))
        draw = ImageDraw.Draw(sheet)
        draw.text((x + 8, y + 6), title, font=font, fill="black")
        draw.text((x + 8, y + 30), subtitle, font=small, fill="black")

    draw_tile(source_image, source, source_region, (255, 75, 75),
              "Source: supplied local region", "red = source annotation", 0)
    draw_tile(target_image, target, target_region, (60, 170, 255),
              "Target: independent evaluation region", "blue = held-out target annotation", 1)
    names = ["direct_local", "global_sim3", "relation_only", "relational", "oracle_relational"]
    for slot, name in enumerate(names, 2):
        result, record = results[name], metrics[name]
        status = "accepted" if result["accepted"] else "rejected candidate"
        error, iou = record["center_error_m"], record["region_iou"]
        text = f"{status}; center={'n/a' if error is None else f'{error*100:.1f} cm'}"
        text += f"; IoU={'n/a' if iou is None else f'{iou:.2f}'}"
        draw_tile(target_image, target, result.get("candidate_mask"),
                  (60, 225, 100) if result["accepted"] else (255, 180, 35), name, text, slot)
    draw = ImageDraw.Draw(sheet)
    x, y = 3 * tile_width + 12, tile_height + 12
    for line in ["Given full-entity segmentation", "Real rendered RGB-D + pretrained MARCO",
                 "No target region used by automatic inference", "Oracle = separate geometry diagnostic",
                 "Green: accepted region candidate", "Orange: rejected candidate (diagnostic)",
                 "Placement safety / robot execution not tested"]:
        draw.text((x, y), line, font=small, fill="black")
        y += 26
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def summary_plot(rows, output):
    """Export measured candidate quality while visibly marking every rejection."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = ["direct_local", "global_sim3", "relation_only", "relational", "oracle_relational"]
    colors = ["#89949d", "#3675ad", "#ed9e42", "#8157b0", "#32876c"]
    cases = list(dict.fromkeys(row["pair"] for row in rows))
    by_key = {(row["pair"], row["method"]): row for row in rows}
    figure, axes = plt.subplots(2, 1, figsize=(13.8, 8.4), sharex=True)
    width = .15
    for method_index, (method, color) in enumerate(zip(methods, colors)):
        for case_index, case in enumerate(cases):
            row = by_key[(case, method)]
            position = case_index + (method_index - 2) * width
            for axis, key, factor in ((axes[0], "center_error_m", 1000.), (axes[1], "region_iou", 1.)):
                value = row.get(key)
                if value is None:
                    axis.annotate("n/a", (position, 0), rotation=90, fontsize=7)
                    continue
                axis.bar(position, value * factor, width=width * .94,
                         color=color, edgecolor="#222222", linewidth=.5,
                         hatch="///" if not row["accepted"] else None,
                         alpha=.55 if not row["accepted"] else 1.)
    axes[0].set_ylabel("Center error (mm)")
    axes[1].set_ylabel("Visible-point region IoU")
    axes[1].set_ylim(0, 1.04)
    axes[1].set_xticks(np.arange(len(cases)), [case.replace("_", "\n") for case in cases])
    for axis in axes:
        axis.grid(axis="y", color="#dddddd", linewidth=.6)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
    labels = [Patch(facecolor=color, edgecolor="#222222", label=method)
              for method, color in zip(methods, colors)]
    labels.append(Patch(facecolor="white", edgecolor="#222222", hatch="///",
                        label="rejected candidate (diagnostic only)"))
    figure.legend(handles=labels, loc="upper center", ncol=3, frameon=False,
                  bbox_to_anchor=(.5, .965), fontsize=9)
    figure.suptitle("Local region transfer on isolated fixtures from original", fontsize=15, y=.995)
    figure.text(.5, .015,
                "Given entity masks and source region; real rendered RGB-D / MARCO. "
                "Shape = width x1.18, depth x0.85. No robot execution.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .89))
    output = Path(output)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
