"""Held-out surface and localization measurements for neighborhood proposals.

Source interaction pixels are demonstration evidence, not a required target
footprint. Evaluation therefore measures observed support, surface identity,
and position preference separately. No metric here certifies object fit,
collision freedom, stability, or robot reachability.
"""

from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


# BEGIN observed-neighborhood evaluation; no copied-patch IoU contract.
def evidence_reference_center(source_points, source_mask, source_truth, target_truth):
    """Map the source evidence mean with independent, known scene transforms.

    This is a localization reference for the same-asset experiments only. Its
    definition exactly matches the inference input's arithmetic evidence mean;
    it is not the center of a hypothetical placed object or a compulsory goal.
    Both transforms are held-out fixture annotations, never fitted estimates.
    """
    points = np.asarray(source_points, dtype=float)
    mask = np.asarray(source_mask)
    if mask.dtype != np.bool_ or mask.shape != (len(points),) or not mask.any():
        raise ValueError("Reference requires a nonempty point-aligned source mask")
    transforms = [np.asarray(source_truth["world_to_canonical"], dtype=float),
                  np.asarray(target_truth["canonical_to_world"], dtype=float)]
    for matrix in transforms:
        if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
                or not np.allclose(matrix[3], [0, 0, 0, 1])):
            raise ValueError("Reference geometry requires finite affine 4x4 transforms")
    source_center = np.r_[points[mask].mean(0), 1.]
    return (transforms[1] @ transforms[0] @ source_center)[:3]


def evaluate_prediction(result, target_points, truth, reference_center):
    """Measure all modes; keep algorithm acceptance separate from held-out hits.

    ``support_id`` labels independently annotated observed physical surfaces;
    -1 means no annotated support. A preferred ID records the demonstrated
    surface correspondence solely for evaluation. Reporting any-support and
    preferred-support hits separately avoids silently deciding task semantics.
    """
    points = np.asarray(target_points, dtype=float)
    support = np.asarray(truth["support_id"])
    preferred = np.asarray(truth["preferred_support_id"])
    if support.dtype.kind not in "iu" or support.shape != (len(points),):
        raise ValueError("support_id must be an integer per observed point")
    if preferred.shape != () or preferred.dtype.kind not in "iu" or int(preferred) < 0:
        raise ValueError("preferred_support_id must be a nonnegative integer scalar")
    preferred = int(preferred)
    reference = np.asarray(reference_center, dtype=float)
    if reference.shape != (3,) or not np.isfinite(reference).all():
        raise ValueError("Independent evidence reference must be a finite 3-vector")
    records = []
    all_samples = []
    for rank, mode in enumerate(result["modes"], 1):
        mask = np.asarray(mode["mask"])
        samples = np.asarray(mode["sample_indices"])
        center = np.asarray(mode["center"], dtype=float)
        normal = np.asarray(mode["normal"], dtype=float)
        seed = mode["seed_index"]
        if mask.dtype != np.bool_ or mask.shape != (len(points),):
            raise ValueError("Mode mask must index observed target points")
        if (samples.ndim != 1 or samples.dtype.kind not in "iu"
                or np.any(samples < 0) or np.any(samples >= len(points))
                or len(np.unique(samples)) != len(samples) or not mask[samples].all()):
            raise ValueError("Mode samples must be distinct observed neighborhood indices")
        if (isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer))
                or not 0 <= seed < len(points) or not mask[seed]):
            raise ValueError("Mode seed must be an observed member of its neighborhood")
        if (center.shape != (3,) or normal.shape != (3,)
                or not np.isfinite(np.r_[center, normal]).all()
                or not np.isclose(np.linalg.norm(normal), 1.)
                or not np.allclose(center, points[seed], atol=1e-10, rtol=0)):
            raise ValueError("Mode center/normal do not describe the observed seed")
        delta = points[mask] - center
        height = delta @ normal
        tangent_radius = np.linalg.norm(delta - height[:, None] * normal, axis=1)
        spacing = (float(cKDTree(points[samples]).query(points[samples], k=2)[0][:, 1].min())
                   if len(samples) > 1 else None)
        ids, counts = np.unique(support[samples], return_counts=True)
        record = {"rank": rank, "seed_index": int(seed),
                  "seed_support_id": int(support[seed]),
                  "preferred_support_hit": bool(support[seed] == preferred),
                  "any_support_hit": bool(support[seed] >= 0),
                  "center_error_m": float(np.linalg.norm(center - reference)),
                  "neighborhood_points": int(mask.sum()), "sample_count": int(len(samples)),
                  "neighborhood_preferred_support_fraction": float(np.mean(support[mask] == preferred)),
                  "neighborhood_any_support_fraction": float(np.mean(support[mask] >= 0)),
                  "sample_preferred_support_fraction": float(np.mean(support[samples] == preferred)) if len(samples) else None,
                  "sample_any_support_fraction": float(np.mean(support[samples] >= 0)) if len(samples) else None,
                  "sample_support_counts": {str(int(key)): int(value) for key, value in zip(ids, counts)},
                  "maximum_tangent_radius_m": float(tangent_radius.max()),
                  "maximum_plane_height_m": float(np.abs(height).max()),
                  "minimum_sample_spacing_m": spacing,
                  "best_sample_reference_distance_m": float(np.linalg.norm(points[samples] - reference, axis=1).min()) if len(samples) else None}
        records.append(record)
        all_samples.extend(samples.tolist())
    unique_samples = np.unique(all_samples).astype(np.int64)
    first = records[0] if records else None
    metrics = {
        "accepted": bool(result["accepted"]), "mode_count": len(records),
        "preferred_support_id": preferred,
        "preferred_visible_points": int(np.count_nonzero(support == preferred)),
        "any_support_visible_points": int(np.count_nonzero(support >= 0)),
        "top1_preferred_support_hit": bool(first and first["preferred_support_hit"]),
        "top1_any_support_hit": bool(first and first["any_support_hit"]),
        "topk_preferred_support_hit": any(record["preferred_support_hit"] for record in records),
        "topk_any_support_hit": any(record["any_support_hit"] for record in records),
        "center_error_m": first["center_error_m"] if first else None,
        "top1_neighborhood_points": first["neighborhood_points"] if first else 0,
        "top1_sample_count": first["sample_count"] if first else 0,
        "top1_neighborhood_preferred_support_fraction": first["neighborhood_preferred_support_fraction"] if first else None,
        "top1_neighborhood_any_support_fraction": first["neighborhood_any_support_fraction"] if first else None,
        "top1_sample_preferred_support_fraction": first["sample_preferred_support_fraction"] if first else None,
        "top1_sample_any_support_fraction": first["sample_any_support_fraction"] if first else None,
        "top1_best_sample_reference_distance_m": first["best_sample_reference_distance_m"] if first else None,
        "all_modes_unique_samples": int(len(unique_samples)),
        "topk_best_sample_reference_distance_m": float(np.linalg.norm(points[unique_samples] - reference, axis=1).min()) if len(unique_samples) else None,
        "maximum_tangent_radius_m": max((r["maximum_tangent_radius_m"] for r in records), default=None),
        "maximum_plane_height_m": max((r["maximum_plane_height_m"] for r in records), default=None),
        "minimum_sample_spacing_m": min((r["minimum_sample_spacing_m"] for r in records if r["minimum_sample_spacing_m"] is not None), default=None),
        "mode_metrics": records,
    }
    for name in ("top1_preferred_support_hit", "top1_any_support_hit", "topk_preferred_support_hit"):
        metrics["accepted_" + name] = bool(metrics["accepted"] and metrics[name])
    return metrics
# END observed-neighborhood evaluation.


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


# BEGIN neighborhood visualization: observed samples, never copied source hulls.
def comparison_image(source_image, target_image, source, target, source_region,
                     target_truth, reference_center, results, metrics, output):
    """Show three observed modes plus an explicitly held-out surface reference."""
    from PIL import Image, ImageDraw, ImageFont
    import textwrap

    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 16)
    small = ImageFont.truetype(font_path, 13)
    tile_width, tile_height = 480, 432
    sheet = Image.new("RGB", (tile_width * 4, tile_height * 2), "white")
    colors = [(50, 225, 110), (90, 150, 255), (220, 100, 245)]

    def draw_tile(image_path, observation, overlays, title, subtitle, slot, modes=()):
        rgb = np.array(Image.open(image_path).convert("RGB"))
        pixels = np.asarray(observation["pixels"], dtype=int)
        for mask, color in overlays:
            pix = pixels[np.asarray(mask, dtype=bool)]
            rgb[pix[:, 1], pix[:, 0]] = (
                .35 * rgb[pix[:, 1], pix[:, 0]] + .65 * np.asarray(color)).astype(np.uint8)
        im = Image.fromarray(rgb).resize((480, 360))
        draw = ImageDraw.Draw(im)
        factor = np.array([480 / rgb.shape[1], 360 / rgb.shape[0]])
        for rank, mode in enumerate(modes):
            for xy in pixels[mode["sample_indices"]] * factor:
                x, y = xy
                draw.ellipse((x-1.5, y-1.5, x+1.5, y+1.5), fill=colors[rank % 3], outline="black")
            x, y = pixels[mode["seed_index"]] * factor
            draw.line((x-5, y, x+5, y), fill="white", width=2)
            draw.line((x, y-5, x, y+5), fill="white", width=2)
            draw.text((x+5, y+2), str(rank+1), font=small, fill="white", stroke_width=1, stroke_fill="black")
        x, y = (slot % 4) * tile_width, (slot // 4) * tile_height
        sheet.paste(im, (x, y + 72))
        draw = ImageDraw.Draw(sheet)
        draw.text((x + 8, y + 5), title, font=font, fill="black")
        for line_index, line in enumerate(textwrap.wrap(subtitle, width=65)[:2]):
            draw.text((x + 8, y + 28 + line_index * 18), line, font=small, fill="black")

    draw_tile(source_image, source, [(source_region, (255, 75, 75))],
              "Source: demonstrated interaction evidence", "Red pixels remain evidence, including holes.", 0)
    support_ids = np.asarray(target_truth["support_id"])
    preferred = int(target_truth["preferred_support_id"])
    draw_tile(target_image, target, [(support_ids >= 0, (150, 160, 175)),
                                    (support_ids == preferred, (60, 170, 255))],
              "Held-out physical surface labels", "Blue = corresponding surface; gray = other supports.", 1)
    # The reference is projected using calibration only for the labeled GT tile.
    camera = (np.asarray(reference_center) - target["T_world_camera_cv"][:3, 3]) @ target["T_world_camera_cv"][:3, :3]
    if camera[2] > 0:
        uv = (target["K"] @ camera)[:2] / camera[2]
        width, height = Image.open(target_image).size
        if 0 <= uv[0] < width and 0 <= uv[1] < height:
            x, y = uv * [480/width, 360/height] + [tile_width, 72]
            draw = ImageDraw.Draw(sheet)
            draw.line((x-6, y, x+6, y), fill="yellow", width=2)
            draw.line((x, y-6, x, y+6), fill="yellow", width=2)

    names = ["direct_local", "global_sim3", "relation_only", "relational", "oracle_relational"]
    for slot, name in enumerate(names, 2):
        result, record = results[name], metrics[name]
        status = "accepted evidence" if result["accepted"] else "rejected / diagnostic"
        error = record["center_error_m"]
        error_text = "n/a" if error is None else f"{error*1000:.1f} mm"
        subtitle = f"{status}; center {error_text}; modes {len(result['modes'])}"
        overlays = [(mode["mask"], colors[index % 3]) for index, mode in reversed(list(enumerate(result["modes"])))]
        draw_tile(target_image, target, overlays, name, subtitle, slot, result["modes"])
    draw = ImageDraw.Draw(sheet)
    x, y = 3 * tile_width + 12, tile_height + 12
    for line in ["Given whole-entity masks and source evidence", "Modes 1/2/3: green / blue / purple",
                 "Dots: actual observed samples; cross: seed", "Yellow GT cross: corresponding evidence center",
                 "Same metric sampling budget in every case", "No target label enters automatic inference",
                 "Oracle = separate geometry diagnostic", "Object fit, stability and execution are not tested"]:
        draw.text((x, y), line, font=small, fill="black")
        y += 28
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
# END neighborhood visualization.


def summary_plot(rows, output):
    """Export localization and corresponding-surface evidence, not patch IoU."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    methods = ["direct_local", "global_sim3", "relation_only", "relational", "oracle_relational"]
    colors = ["#89949d", "#3675ad", "#ed9e42", "#8157b0", "#32876c"]
    cases = list(dict.fromkeys(row["pair"] for row in rows))
    by_key = {(row["pair"], row["method"]): row for row in rows}
    # BEGIN fixed-metric neighborhood comparison; keep rejected diagnostics visible.
    figure, axes = plt.subplots(2, 1, figsize=(max(13.8, 1.5 * len(cases)), 8.4), sharex=True)
    width = .15
    for method_index, (method, color) in enumerate(zip(methods, colors)):
        for case_index, case in enumerate(cases):
            row = by_key[(case, method)]
            position = case_index + (method_index - 2) * width
            for axis, key, factor in ((axes[0], "center_error_m", 1000.),
                                      (axes[1], "top1_sample_preferred_support_fraction", 1.)):
                value = row.get(key)
                if value is None:
                    axis.annotate("n/a", (position, 0), rotation=90, fontsize=7)
                    continue
                axis.bar(position, value * factor, width=width * .94,
                         color=color, edgecolor="#222222", linewidth=.5,
                         hatch="///" if not row["accepted"] else None,
                         alpha=.55 if not row["accepted"] else 1.)
    axes[0].set_ylabel("Evidence-center reference error (mm)")
    axes[1].set_ylabel("Samples on corresponding surface / all samples")
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
    figure.suptitle("Transferred location preferences and observed surface neighborhoods", fontsize=15, y=.995)
    figure.text(.5, .015,
                "Given entity masks and source interaction evidence. Uniform metric sampling budget. "
                "Shape = width x1.18, depth x0.85. No object-fit or robot-execution claim.",
                ha="center", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .89))
    output = Path(output)
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)
    # END fixed-metric neighborhood comparison.
