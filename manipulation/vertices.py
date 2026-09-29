"""Observe and associate visible vertex hypotheses without fixed point roles.

The observer uses native-resolution RGB corners and metric RGB-aligned depth.
Its nearby depth sample is a measured surface point, not a recovered exact 3D
vertex. SAM owner masks are supplied by the caller; no model, mesh, object pose,
plane fitting, noise model, or skill-specific prompt is created here.

Association proposes explicit source/current ID pairs. Arbitrary list order
does not supply material identity, and a semantic proposal is not a geometric
consensus or an executable grasp. The rigid estimator consumes these pairs
separately and decides whether there are enough noncollinear measurements.
"""

# BEGIN reusable visible-vertex observations and ID association.
from copy import deepcopy
import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .grounding import _image
from .observations import _intrinsics, _mask, backproject


_PARAMETERS = {
    "max_corners": 32, "quality_level": 0.02, "minimum_distance_px": 4,
    "mask_expansion_px": 2, "subpixel_window_radius_px": 3,
    "maximum_subpixel_motion_px": 3, "maximum_depth_sample_distance_px": 2,
    "duplicate_corner_distance_px": 2,
}

_ASSOCIATION_INSTRUCTION = (
    "Inspect candidate image corners on ONE selected rigid polyhedral object. "
    "The two observations are of the same object shape, but viewpoints or object poses "
    "may differ. Images 1 and 2 give full source/current RGB context; images 3 and 4 "
    "magnify the corresponding object and number corner candidates. A numbered RGB "
    "corner is only a hypothesis: some are raster staircase artifacts or occlusion "
    "junctions with another object, not actual physical vertices. Select only visibly "
    "supported object vertices and establish one-to-one source/current correspondences. "
    "Use visible face/edge structure and appearance, not candidate number, image "
    "left/right ordering, or an assumption that the pose is unchanged. A corner where "
    "an occluder cuts a face is not a vertex. Do not infer hidden vertices or invent IDs. "
    "There is no required number of pairs: retain all clearly supported pairs. "
    "An unlisted candidate can remain undecided; do not force every candidate into a pair. "
    "If physical association is ambiguous due to symmetry or visibility, report "
    "ambiguous or insufficient instead of pretending it is known. No coordinates, "
    "normals, dimensions, poses or numeric geometry are requested. "
    "Return exactly {\"status\":\"matched|ambiguous|insufficient\", "
    "\"pairs\":[{\"source_id\":integer,\"target_id\":integer,\"reason\":string}], "
    "\"source_rejected_ids\":[integers],\"target_rejected_ids\":[integers],\"reason\":string}. "
    "Every reason must be nonempty. Status matched requires at least one supported pair; "
    "ambiguous and insufficient require pairs=[]. Rejected IDs must be actual supplied "
    "candidates and cannot also be matched. Return only these fields."
)


def _deduplicate(rows):
    """Keep the stronger image response and expose every removed observation.

    Subpixel refinement can merge originally separated detections. Distinct
    refined corners can also select the same raw-depth pixel. Either case
    must not create two independent votes in a later geometric consensus.
    """
    accepted = []
    candidates = [row for row in rows if row["status"] == "candidate"]
    for row in sorted(candidates, key=lambda item: (-item["corner_response"], item["id"])):
        duplicate = None
        reason = None
        for kept in accepted:
            if row["depth_pixel"] == kept["depth_pixel"]:
                duplicate, reason = kept, "duplicate raw-depth sample"
                break
            if np.linalg.norm(np.asarray(row["corner_pixel"])-kept["corner_pixel"]) < 2:
                duplicate, reason = kept, "subpixel convergence duplicate"
                break
        if duplicate is None:
            accepted.append(row)
        else:
            row.update(status="rejected", reason=reason,
                       rejected_duplicate_of=duplicate["id"])


def observe_vertex_candidates(image, depth, K, owner_mask):
    """Detect a variable number of RGB hypotheses and attach actual depth samples.

    ``image`` is a PIL image; ``depth`` is its aligned camera-z depth in metres;
    ``K`` uses this same native image grid; ``owner_mask`` is a nonempty boolean
    mask. No input is mutated or resized. The 32-corner limit is a detector cap,
    not a required semantic point count. Missing corners/depth yield an explicit
    empty observation rather than invented points or another geometry method.
    """
    image = _image(image)
    depth = np.asarray(depth, dtype=float)
    if depth.shape != (image.height, image.width):
        raise ValueError("RGB and aligned depth dimensions must agree")
    if min(image.size) < 11:
        raise ValueError("RGB is too small for the existing 3-pixel SubPix window")
    matrix = _intrinsics(K)
    owner = _mask(owner_mask, depth.shape, "owner_mask")
    report = {
        "method": "native RGB Shi-Tomasi + cornerSubPix + nearby raw owner depth",
        "parameters": dict(_PARAMETERS), "image_size": list(image.size),
        "status": "insufficient", "candidates": [], "candidate_count": 0,
        "plane_or_box_fit_used": False,
        "measurement_scope": "raw nearby owner samples; exact 3D vertices are not recovered",
    }
    valid = owner & np.isfinite(depth) & (depth > 0)
    yy, xx = np.nonzero(valid)
    if not len(xx):
        report["reason"] = "No finite positive owner depth"
        return report
    valid_pixels = np.column_stack((xx, yy))
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    detection_mask = cv2.dilate(owner.astype(np.uint8), np.ones((5, 5), np.uint8)) * 255
    initial = cv2.goodFeaturesToTrack(
        gray, maxCorners=32, qualityLevel=0.02, minDistance=4,
        mask=detection_mask, blockSize=3, useHarrisDetector=False)
    if initial is None:
        report["reason"] = "No RGB corner candidates"
        return report
    refined = cv2.cornerSubPix(
        gray, initial.copy(), (3, 3), (-1, -1),
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
    scores = cv2.cornerMinEigenVal(gray, blockSize=3)
    rows = []
    for number, (before, corner) in enumerate(zip(initial[:, 0], refined[:, 0]), 1):
        row = {
            "id": number, "initial_pixel": before.astype(float).tolist(),
            "corner_pixel": corner.astype(float).tolist(), "status": "candidate",
            "kind": "RGB corner; physical vertex identity unverified",
        }
        if (not np.isfinite(corner).all() or np.any(corner < 0)
                or corner[0] > image.width-1 or corner[1] > image.height-1):
            row.update(status="rejected", reason="Refined image corner is nonfinite or out of bounds")
            rows.append(row)
            continue
        distances = np.linalg.norm(valid_pixels-corner, axis=1)
        nearest = int(np.argmin(distances))
        displacement = float(np.linalg.norm(corner-before))
        row.update(subpixel_refinement_displacement_px=displacement,
                   depth_sample_distance_px=float(distances[nearest]))
        if displacement > 3.0 or distances[nearest] > 2.0:
            row.update(status="rejected", reason="Corner moved too far or no owner depth within 2 pixels")
        else:
            sample = valid_pixels[nearest]
            row.update(
                depth_pixel=sample.tolist(), depth_m=float(depth[sample[1], sample[0]]),
                point_camera=backproject(depth, matrix, sample[None], mask=owner)[0].tolist(),
                corner_response=float(scores[int(before[1]), int(before[0])]),
                measurement_scope="actual depth pixel point, not an exact 3D vertex",
            )
        rows.append(row)
    _deduplicate(rows)
    count = sum(row["status"] == "candidate" for row in rows)
    report.update(candidates=rows, candidate_count=count,
                  status="observed" if count else "insufficient")
    if not count:
        report["reason"] = "No distinct RGB hypotheses with usable nearby owner depth"
    return report


def _validated_rows(report, image, owner):
    """Validate persisted observation IDs and samples before exposing them to VLM."""
    if (not isinstance(report, dict) or report.get("image_size") != list(image.size)
            or not isinstance(report.get("candidates"), list)):
        raise ValueError("Vertex observation must match its native image")
    rows, seen, depth_pixels = [], set(), set()
    for row in report["candidates"]:
        if (not isinstance(row, dict) or type(row.get("id")) is not int or row["id"] <= 0
                or row["id"] in seen or not isinstance(row.get("status"), str)
                or row["status"] not in {"candidate", "rejected"}):
            raise ValueError("Vertex observation IDs/statuses must be unique and valid")
        seen.add(row["id"])
        if row["status"] == "rejected":
            continue
        for field in ("initial_pixel", "corner_pixel"):
            pixel = np.asarray(row.get(field), dtype=float)
            if (pixel.shape != (2,) or not np.isfinite(pixel).all() or np.any(pixel < 0)
                    or pixel[0] > image.width-1 or pixel[1] > image.height-1):
                raise ValueError(f"Candidate {field} must be a finite in-image pixel")
        sample = row.get("depth_pixel")
        if (not isinstance(sample, list) or len(sample) != 2
                or any(type(value) is not int for value in sample)
                or not (0 <= sample[0] < image.width and 0 <= sample[1] < image.height)
                or not owner[sample[1], sample[0]]):
            raise ValueError("Candidate depth pixel must identify an actual owner sample")
        if (tuple(sample) in depth_pixels
                or any(np.linalg.norm(np.asarray(row["corner_pixel"])-other["corner_pixel"]) < 2
                       for other in rows)):
            raise ValueError("Duplicate corner/depth samples cannot provide independent correspondence votes")
        point = np.asarray(row.get("point_camera"), dtype=float)
        z = row.get("depth_m")
        if (point.shape != (3,) or not np.isfinite(point).all()
                or isinstance(z, bool) or not isinstance(z, (int, float))
                or not np.isfinite(z) or z <= 0 or not np.isclose(point[2], z, atol=1e-9, rtol=0)):
            raise ValueError("Candidate needs a finite camera point consistent with its positive metric depth")
        depth_pixels.add(tuple(sample))
        rows.append(deepcopy(row))
    if type(report.get("candidate_count")) is not int or report["candidate_count"] != len(rows):
        raise ValueError("Vertex candidate_count disagrees with candidate statuses")
    return rows


def _candidate_panel(image, owner, rows):
    """Magnify unchanged RGB pixels; candidate labels never supply a point order."""
    yy, xx = np.nonzero(owner)
    box = (max(0, int(xx.min())-8), max(0, int(yy.min())-8),
           min(image.width, int(xx.max())+9), min(image.height, int(yy.max())+9))
    crop = image.crop(box)
    scale = min(8., 1024/max(crop.size))
    panel = crop.resize((round(crop.width*scale), round(crop.height*scale)), Image.Resampling.NEAREST)
    draw = ImageDraw.Draw(panel)
    for row in rows:
        x, y = (np.asarray(row["corner_pixel"])-box[:2]+.5)*scale-.5
        draw.ellipse((x-4, y-4, x+4, y+4), outline="#00ffdc", width=2)
        draw.text((x+6, y-10), str(row["id"]), fill="#ffff50", stroke_width=2, stroke_fill="black")
    return panel


def _validate_selection(response, source, current):
    """Reject invented/duplicate IDs and contradictory or geometry-bearing replies."""
    required = {"status", "pairs", "source_rejected_ids", "target_rejected_ids", "reason"}
    if not isinstance(response, dict) or set(response) != required:
        raise ValueError("Vertex association response has invalid fields")
    if (not isinstance(response["status"], str)
            or response["status"] not in {"matched", "ambiguous", "insufficient"}
            or not isinstance(response["reason"], str) or not response["reason"].strip()
            or not isinstance(response["pairs"], list)):
        raise ValueError("Vertex association status, reason or pairs are invalid")
    if ((response["status"] == "matched") != bool(response["pairs"])):
        raise ValueError("Matched status needs supported pairs; unresolved status cannot supply actionable pairs")
    lookups = ({row["id"]: row for row in rows} for rows in (source, current))
    lookup_s, lookup_t = lookups
    pairs, used_s, used_t = [], set(), set()
    for pair in response["pairs"]:
        if (not isinstance(pair, dict) or set(pair) != {"source_id", "target_id", "reason"}
                or not isinstance(pair["reason"], str) or not pair["reason"].strip()):
            raise ValueError("Malformed vertex correspondence")
        a, b = pair["source_id"], pair["target_id"]
        if (type(a) is not int or type(b) is not int or a not in lookup_s or b not in lookup_t
                or a in used_s or b in used_t):
            raise ValueError("Vertex association must use valid one-to-one candidate IDs")
        used_s.add(a)
        used_t.add(b)
        pairs.append({**deepcopy(pair), "source": deepcopy(lookup_s[a]), "target": deepcopy(lookup_t[b])})
    for field, lookup, used in (("source_rejected_ids", lookup_s, used_s),
                                ("target_rejected_ids", lookup_t, used_t)):
        rejected = response[field]
        if (not isinstance(rejected, list) or any(type(value) is not int or value not in lookup for value in rejected)
                or len(set(rejected)) != len(rejected) or used.intersection(rejected)):
            raise ValueError("Rejected IDs must be valid, unique and disjoint from matched IDs")
    return pairs


def associate_vertex_candidates(source_rgb, current_rgb, source_mask, current_mask,
                                source_report, current_report, *, client):
    """Propose visible same-shape correspondences using one geometry-only prompt.

    The injected client exposes ``chat_json(prompt, images=...)`` and owns all
    network/model calls and recording. This function performs no retry. A
    matched response may contain any positive number of pairs; three-point
    observability is the downstream estimator's responsibility. Ambiguous or
    insufficient evidence returns that status with no actionable pairs.
    """
    source_image, current_image = _image(source_rgb), _image(current_rgb)
    source_owner = _mask(source_mask, (source_image.height, source_image.width), "source_mask")
    current_owner = _mask(current_mask, (current_image.height, current_image.width), "current_mask")
    source = _validated_rows(source_report, source_image, source_owner)
    current = _validated_rows(current_report, current_image, current_owner)
    trace = {
        "source_candidate_ids": [row["id"] for row in source],
        "target_candidate_ids": [row["id"] for row in current],
        "native_image_sizes": [list(source_image.size), list(current_image.size)],
        "image_order": ["source RGB", "current RGB", "source numbered corners", "current numbered corners"],
        "scope": "visual correspondence proposals; no metric fit or physical identity certificate",
    }
    if not source or not current:
        return {"status": "insufficient", "pairs": [], "selection": None,
                "trace": {**trace, "vlm_calls": 0, "reason": "One observation has no usable corner candidates"}}
    if not callable(getattr(client, "chat_json", None)):
        raise ValueError("An explicit JSON client with chat_json is required")
    prompt = _ASSOCIATION_INSTRUCTION + "\nCandidate IDs:\n" + json.dumps({
        "source_candidate_ids": trace["source_candidate_ids"],
        "target_candidate_ids": trace["target_candidate_ids"],
    })
    images = [source_image, current_image,
              _candidate_panel(source_image, source_owner, source),
              _candidate_panel(current_image, current_owner, current)]
    response = client.chat_json(prompt, images=images)
    pairs = _validate_selection(response, source, current)
    return {"status": response["status"], "pairs": pairs, "selection": deepcopy(response),
            "trace": {**trace, "prompt": prompt, "vlm_calls": 1}}
# END reusable visible-vertex observations and ID association.
