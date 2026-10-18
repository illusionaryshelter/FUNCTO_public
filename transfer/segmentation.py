"""SAM instances and observed RGB-D point banks for transfer experiments.

The only semantic input is an explicit, single text prompt. Instance selection
uses the given source evidence where available and native SAM scores otherwise.
No entity boxes, simulator owners, evaluation masks, or hidden geometry enter
this module. SAM masks are retained verbatim; given source evidence is stored
separately even where SAM misses it.

Model execution must be enclosed by the experiment's process-tree and whole-
device supervisor at the user's currently authorized limits. The shared local-
model runtime's internal limits alone do not establish that complete budget.
"""

# BEGIN explicit SAM evidence and RGB-D preparation; no ground-truth inputs.
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .features import write_json


def _prompt(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("prompt must be one explicit, nonempty text string")
    return value


def _rgb(value):
    image = np.asarray(value)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("rgb must be an unmodified uint8 [H,W,3] image")
    if min(image.shape[:2]) < 1:
        raise ValueError("rgb must have nonzero dimensions")
    return image


def _detections(value, shape, prompt):
    """Validate returned instances without thresholding or editing their masks."""
    if not isinstance(value, list):
        raise TypeError("detections must be the ordered list returned by SAM")
    result = []
    for index, detection in enumerate(value):
        if not isinstance(detection, dict):
            raise TypeError("Each detection must be a dictionary")
        if detection["label"] != prompt:
            raise ValueError("All detections must use the one recorded literal prompt")
        mask = np.asarray(detection["mask"])
        if mask.dtype != np.bool_ or mask.shape != shape:
            raise ValueError("SAM masks must be Boolean arrays at original RGB resolution")
        score = detection["score"]
        if (isinstance(score, (bool, np.bool_))
                or not isinstance(score, (int, float, np.integer, np.floating))
                or not np.isfinite(score)):
            raise ValueError("SAM scores must be finite numeric values")
        box = np.asarray(detection["bbox"])
        if box.shape != (4,) or box.dtype.kind not in "fiu" or not np.isfinite(box).all():
            raise ValueError("SAM boxes must be finite numeric xyxy 4-vectors")
        # Boxes are retained as model evidence only. Crops below are derived
        # from the selected mask, so a model box can never act as a prior.
        result.append({"index": index, "label": prompt, "score": float(score),
                       "bbox": box.astype(float).tolist(), "mask": mask.copy()})
    return result


def store_detections(output_dir, rgb, prompt, detections):
    """Save every returned candidate and the exact RGB before selecting one.

    Existing evidence is never overwritten. Empty model output is a recorded
    outcome, whereas malformed output raises an exception. Candidate indices
    refer to the ordered public SAM result, including its existing empty-mask
    boundary handling; they are not inferred simulator instance identifiers.
    """
    image, prompt = _rgb(rgb), _prompt(prompt)
    instances = _detections(detections, image.shape[:2], prompt)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    Image.fromarray(image).save(output_dir / "rgb.png")
    arrays, records = {}, []
    for instance in instances:
        key = f"candidate_{instance['index']}"
        arrays[key] = instance["mask"]
        records.append({name: instance[name] for name in ("index", "label", "score", "bbox")})
        records[-1].update(mask_key=key, mask_pixels=int(instance["mask"].sum()))
    np.savez_compressed(output_dir / "masks.npz", **arrays)
    metadata = {"prompt": prompt, "image_size_wh": list(image.shape[1::-1]),
                "candidate_count": len(instances), "candidates": records,
                "status": "detected" if instances else "no_detections",
                "masks_modified": False}
    write_json(output_dir / "candidates.json", metadata)
    return metadata


def load_detections(output_dir):
    """Restore stored candidates in their original order without pickle."""
    output_dir = Path(output_dir)
    metadata = json.loads((output_dir / "candidates.json").read_text())
    prompt = _prompt(metadata["prompt"])
    with Image.open(output_dir / "rgb.png") as image:
        shape = (image.height, image.width)
    if metadata["image_size_wh"] != [shape[1], shape[0]]:
        raise ValueError("Saved SAM image dimensions disagree with metadata")
    records = metadata["candidates"]
    if metadata["candidate_count"] != len(records):
        raise ValueError("Saved SAM candidate count disagrees with metadata")
    with np.load(output_dir / "masks.npz", allow_pickle=False) as arrays:
        if set(arrays.files) != {f"candidate_{i}" for i in range(len(records))}:
            raise ValueError("Saved SAM masks do not match the candidate indices")
        detections = []
        for index, record in enumerate(records):
            if record["index"] != index or record["mask_key"] != f"candidate_{index}":
                raise ValueError("Saved SAM candidates lost their original ordering")
            mask = arrays[record["mask_key"]]
            if int(mask.sum()) != record["mask_pixels"]:
                raise ValueError("Saved SAM mask occupancy disagrees with metadata")
            detections.append({"label": record["label"], "score": record["score"],
                               "bbox": record["bbox"], "mask": mask})
    checked = _detections(detections, shape, prompt)
    return [{key: value for key, value in item.items() if key != "index"} for item in checked]


def detect_image(image_path, prompt, output_dir):
    """Run one full RGB image and one literal prompt in the existing worker.

    The caller serializes complete stages and applies the explicit external
    resource supervisor. No boxes, prior masks, synonym attempts, or alternate
    detections are supplied to SAM. Worker errors propagate to the supervisor.
    """
    prompt = _prompt(prompt)
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"SAM evidence already exists: {output_dir}")
    with Image.open(image_path) as opened:
        image = opened.convert("RGB")
    from local_models.sam31 import detect
    detections = detect(image, [prompt])
    return store_detections(output_dir, np.asarray(image), prompt, detections)


def _calibration(K, T_world_camera_cv):
    K, T = np.asarray(K), np.asarray(T_world_camera_cv)
    if (K.shape != (3, 3) or T.shape != (4, 4)
            or K.dtype.kind not in "fiu" or T.dtype.kind not in "fiu"
            or not np.isfinite(K).all() or not np.isfinite(T).all()):
        raise ValueError("Calibration requires finite K[3,3] and T_world_camera_cv[4,4]")
    K, T = K.astype(float), T.astype(float)
    if (K[0, 0] <= 0 or K[1, 1] <= 0 or not np.allclose(K[2], [0, 0, 1])
            or abs(np.linalg.det(K)) <= np.finfo(float).tiny):
        raise ValueError("K must describe a nonsingular optical-depth camera")
    if (not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-7)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1.0, atol=1e-7)):
        raise ValueError("T_world_camera_cv must be a proper rigid camera transform")
    return K, T


def _rms(points):
    if not len(points):
        return None
    return float(np.sqrt(np.mean(np.sum((points - points.mean(0)) ** 2, axis=1))))


def _crop(mask, margin_fraction):
    y, x = np.nonzero(mask)
    margin = int(np.ceil(margin_fraction * max(np.ptp(x) + 1, np.ptp(y) + 1)))
    height, width = mask.shape
    return np.asarray([max(0, int(x.min()) - margin), max(0, int(y.min()) - margin),
                       min(width, int(x.max()) + margin + 1),
                       min(height, int(y.max()) + margin + 1)], dtype=np.int64)


# BEGIN selected-entity geometry shared by production and experiment callers.
def _selected_geometry(image, depth, K, T, selected, evidence, *, source,
                       crop_margin_fraction, result):
    """Backproject an already chosen mask, never reconsider its identity.

    Source evidence remains a separate observation even where SAM missed it.
    The selected mask itself is preserved, including pixels without depth;
    only the measured point bank needs positive finite optical depth.
    """
    valid_depth = np.isfinite(depth) & (depth > 0)
    valid_evidence = evidence & valid_depth
    raw_mask = selected["mask"]
    entity = raw_mask & valid_depth
    if not entity.any():
        result["reasons"].append("selected_instance_without_valid_depth")
        return result
    bank_mask = entity | valid_evidence
    y, x = np.nonzero(bank_mask)
    pixels = np.column_stack((x, y)).astype(np.int32)
    rays = np.column_stack((pixels, np.ones(len(pixels)))) @ np.linalg.inv(K).T
    points = (rays * depth[y, x, None]) @ T[:3, :3].T + T[:3, 3]
    entity_member, region = entity[y, x], valid_evidence[y, x]
    observation = {"xyz": points, "pixels": pixels, "object_mask": raw_mask.copy(),
                   "entity_member": entity_member, "region_mask": region,
                   "evidence_only": region & ~entity_member,
                   "K": K.copy(), "T_world_camera_cv": T.copy()}
    # Crop membership follows exactly the measured entity and given source
    # evidence. No target region annotation or estimated transform is used.
    crop_mask = raw_mask | evidence if source else entity
    bounds = _crop(crop_mask, float(crop_margin_fraction))
    left, top, right, bottom = bounds
    sam_rms, bank_rms = _rms(points[entity_member]), _rms(points)
    covered = int((region & entity_member).sum())
    evidence_count = int(valid_evidence.sum())
    result["diagnostics"].update(selected_mask_pixels=int(raw_mask.sum()),
        selected_valid_depth_pixels=int(entity.sum()), pointbank_points=len(points),
        entity_member_points=int(entity_member.sum()),
        evidence_only_points=int((region & ~entity_member).sum()),
        source_evidence_covered_pixels=covered if source else None,
        source_evidence_coverage_fraction=covered / evidence_count if evidence_count else None,
        source_evidence_center=points[region].mean(0).tolist() if source else None,
        sam_entity_rms_radius_m=sam_rms, pointbank_rms_radius_m=bank_rms,
        pointbank_to_sam_rms_ratio=bank_rms / sam_rms if sam_rms and sam_rms > 0 else None)
    result.update(prepared=True, observation=observation, crop_xyxy=bounds,
                  crop_rgb=image[top:bottom, left:right].copy())
    return result


def prepare_selected_observation(rgb, depth_m, K, T_world_camera_cv, selected,
                                 *, source_region_mask=None, crop_margin_fraction=0.15):
    """Prepare the entity already selected by the current instruction.

    ``selected`` is one actual SAM detection, with its unchanged label, score,
    box and mask. This entry accepts no list to rank. A depth-empty selected
    entity reports a missing observation; it never tries another instance.
    A given source region may extend outside the selected SAM mask because
    its identity was supplied separately from this incomplete visual mask.
    """
    image = _rgb(rgb)
    depth = np.asarray(depth_m)
    if depth.shape != image.shape[:2] or depth.dtype.kind not in "fiu":
        raise ValueError("depth_m must be a numeric optical-depth image aligned with rgb")
    if not isinstance(selected, dict):
        raise TypeError("selected must be one previously chosen detection")
    prompt = _prompt(selected.get("label"))
    checked = _detections([selected], image.shape[:2], prompt)[0]
    K, T = _calibration(K, T_world_camera_cv)
    if (isinstance(crop_margin_fraction, (bool, np.bool_))
            or not isinstance(crop_margin_fraction, (int, float, np.integer, np.floating))
            or not np.isfinite(crop_margin_fraction) or crop_margin_fraction < 0):
        raise ValueError("crop_margin_fraction must be finite and nonnegative")
    source = source_region_mask is not None
    evidence = (np.zeros(image.shape[:2], dtype=bool) if not source
                else np.asarray(source_region_mask))
    if evidence.dtype != np.bool_ or evidence.shape != image.shape[:2]:
        raise ValueError("source_region_mask must be a Boolean mask at original RGB resolution")
    evidence = evidence.copy()
    valid_evidence = evidence & np.isfinite(depth) & (depth > 0)
    result = {"prepared": False, "reasons": [], "observation": None,
              "crop_rgb": None, "crop_xyxy": None,
              "source_evidence_mask": evidence if source else None,
              "selection": {"policy": "provided_selected_instance",
                            "candidate_id": selected.get("candidate_id"),
                            "score": checked["score"]},
              "diagnostics": {"role": "source" if source else "target", "prompt": prompt,
                  "candidate_count": 1, "crop_margin_fraction": float(crop_margin_fraction),
                  "source_evidence_original_pixels": int(evidence.sum()) if source else None,
                  "source_evidence_valid_depth_pixels": int(valid_evidence.sum()) if source else None,
                  "source_evidence_invalid_depth_pixels": int((evidence & ~valid_evidence).sum()) if source else None}}
    if source and not valid_evidence.any():
        result["reasons"].append("source_evidence_without_valid_depth")
        return result
    return _selected_geometry(image, depth, K, T, checked, evidence, source=source,
                              crop_margin_fraction=crop_margin_fraction, result=result)
# END selected-entity preparation has no score-based identity decision.


def prepare_observation(rgb, depth_m, K, T_world_camera_cv, detections, *,
                        prompt, source_region_mask=None, crop_margin_fraction=0.15):
    """Bind one predicted instance and backproject the permitted RGB-D evidence.

    Source selection maximizes valid-depth evidence coverage, then native
    score, then the earliest candidate index. At least one given evidence pixel
    must be covered. Target selection uses score and then index only, before
    examining the selected instance's depth. A selected depth-empty instance
    fails explicitly; a lower-ranked candidate never silently replaces it.

    Source points are SAM-valid-depth UNION given-evidence-valid-depth. Its
    original SAM image mask remains unchanged, and ``entity_member``,
    ``region_mask`` and ``evidence_only`` distinguish the point roles. Global
    matching must sample entity_member; local queries must sample region_mask.
    Target points contain only SAM-valid-depth pixels. No existing entity-only
    point cloud is accepted because it could have already removed SAM errors
    using ground truth. The source crop includes the original SAM and supplied
    evidence masks; the target crop uses the selected SAM-valid-depth mask.
    """
    image, prompt = _rgb(rgb), _prompt(prompt)
    shape = image.shape[:2]
    depth = np.asarray(depth_m)
    if depth.shape != shape or depth.dtype.kind not in "fiu":
        raise ValueError("depth_m must be a numeric optical-depth image aligned with rgb")
    K, T = _calibration(K, T_world_camera_cv)
    if (isinstance(crop_margin_fraction, (bool, np.bool_))
            or not isinstance(crop_margin_fraction, (int, float, np.integer, np.floating))
            or not np.isfinite(crop_margin_fraction) or crop_margin_fraction < 0):
        raise ValueError("crop_margin_fraction must be finite and nonnegative")
    instances = _detections(detections, shape, prompt)
    valid_depth = np.isfinite(depth) & (depth > 0)
    source = source_region_mask is not None
    evidence = np.zeros(shape, dtype=bool) if not source else np.asarray(source_region_mask)
    if evidence.dtype != np.bool_ or evidence.shape != shape:
        raise ValueError("source_region_mask must be a Boolean mask at original RGB resolution")
    evidence = evidence.copy()
    valid_evidence = evidence & valid_depth
    evidence_count = int(valid_evidence.sum())
    candidate_records = []
    for item in instances:
        covered = int(np.count_nonzero(item["mask"] & valid_evidence))
        candidate_records.append({"index": item["index"], "score": item["score"],
            "mask_pixels": int(item["mask"].sum()),
            "valid_depth_pixels": int(np.count_nonzero(item["mask"] & valid_depth)),
            "source_evidence_covered_pixels": covered if source else None,
            "source_evidence_coverage_fraction": covered / evidence_count if evidence_count else None})
    selection = {"policy": "source_coverage_then_score_then_index" if source else "target_score_then_index",
                 "selected_index": None, "candidates": candidate_records}
    diagnostics = {"role": "source" if source else "target", "prompt": prompt,
        "candidate_count": len(instances), "crop_margin_fraction": float(crop_margin_fraction),
        "source_evidence_original_pixels": int(evidence.sum()) if source else None,
        "source_evidence_valid_depth_pixels": evidence_count if source else None,
        "source_evidence_invalid_depth_pixels": int((evidence & ~valid_depth).sum()) if source else None}
    result = {"prepared": False, "reasons": [], "selection": selection,
              "diagnostics": diagnostics, "observation": None, "crop_rgb": None,
              "crop_xyxy": None, "source_evidence_mask": evidence if source else None}
    if source and not evidence_count:
        result["reasons"].append("source_evidence_without_valid_depth")
        return result
    if not instances:
        result["reasons"].append("no_sam_detections")
        return result
    if source:
        selected_index = min(range(len(instances)), key=lambda i: (
            -candidate_records[i]["source_evidence_covered_pixels"], -instances[i]["score"], i))
        if candidate_records[selected_index]["source_evidence_covered_pixels"] == 0:
            result["reasons"].append("source_instance_without_evidence_overlap")
            return result
    else:
        selected_index = min(range(len(instances)), key=lambda i: (-instances[i]["score"], i))
    selection["selected_index"] = selected_index
    selected = instances[selected_index]
    # BEGIN reuse geometry after this experiment's explicit selection policy.
    return _selected_geometry(image, depth, K, T, selected, evidence, source=source,
                              crop_margin_fraction=crop_margin_fraction, result=result)
    # END production callers instead enter prepare_selected_observation directly.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metadata = detect_image(args.image, args.prompt, args.output)
    print(json.dumps({"status": metadata["status"], "candidate_count": metadata["candidate_count"],
                      "output": str(args.output)}, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
# END explicit SAM evidence and RGB-D preparation.
