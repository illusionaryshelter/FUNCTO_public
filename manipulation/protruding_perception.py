"""Select an observed backing surface near a protruding interaction region.

Only registered RGB-D, intrinsics and an already selected local region enter
this module. The surrounding window is a search area, not a whole-object mask.
Plane geometry comes from depth; the VLM selects its semantic role only.
"""

# BEGIN local protruding-region backing evidence without whole-owner detection.
from copy import deepcopy
import json
import math

import numpy as np
from PIL import Image

from .grounding import _client, _image, _mask
from .observations import _intrinsics
from .selection import _box, _highlight, _panels
from .support_grounding import _plane_candidates


_BACKING_INSTRUCTION = (
    "Identify the observed backing surface for the selected protruding interaction "
    "part. This is a geometric witness behind that part: the surface to which the "
    "part is attached or in front of which it protrudes. It is NOT the region the "
    "hand contacts, and neither the surface nor its surrounding window is a whole "
    "rigid-object reconstruction.\n"
    "Image 1 is the original scene. Image 2 enlarges the local context, marking the "
    "selected interaction part GREEN. Image 3 has numbered independent candidate "
    "panels: RED marks actual observed plane pixels and GREEN retains the selected "
    "interaction part wherever it is visible. Plane candidates contain only valid "
    "depth pixels outside that selected part, within its local context window. "
    "Unobserved pixels and holes have not been filled.\n"
    "Use the selected part and visible context to identify its backing surface. "
    "Do not select by largest area, numerical order or proximity alone. A nearby "
    "unrelated plane or an occluding foreground surface is not the requested "
    "backing surface. Coplanar pixels can belong to different entities; a plane "
    "fit alone does not establish attachment or rigid-body identity. Confirm the "
    "role even if only one candidate exists. If no candidate is appropriate or "
    "the visible evidence is ambiguous, return null instead of guessing.\n"
    "Select only an allowed candidate ID. Do not merge masks, invent a surface, "
    "or generate coordinates, normals, axes, object poses, contact points or "
    "grasp directions. Return exactly {\"candidate_id\": integer_ID_or_null, "
    "\"reason\": short_nonempty_string} explaining the visible evidence."
)


def observed_backing_planes(image, depth, K, region_mask, *, client=None):
    """Return one selected observed plane and its complete candidate evidence.

    The same function applies to source and current observations. ``depth`` is
    optical-z depth in metres, registered to ``image``; ``K`` describes that
    image. Nonfinite/nonpositive depth pixels remain unusable. The nonempty
    binary ``region_mask`` identifies the selected interaction part, not its
    containing owner. Its bounding box is expanded on all sides by the ceiling
    of 0.6 times its longest side, then clipped to the image. Plane extraction
    excludes the interaction mask and never grows beyond that window.

    Existing bounded Open3D extraction produces the candidates. One VLM call
    confirms the backing role even for a single candidate; missing candidates,
    abstention, invalid responses and transport errors fail explicitly, without
    another prompt, a larger window or a largest-plane fallback.

    The returned plane retains the extractor's ``mask``, ``origin``, ``normal``,
    ``first_axis`` and fit metadata. All geometry is in this camera's metric
    coordinates. Its normal is UNORIENTED: neither world-up nor an approach
    direction is supplied here. ``origin`` is a fit reference and ``first_axis``
    is a numerical tangent, not a physical centre or a functional direction.
    The caller must independently establish protrusion side, part geometry,
    functional frame and any rigid-owner association it needs. In particular,
    this function does not require valid depth inside the interaction region
    to select a surrounding plane, nor does it claim to recover that region's
    missing geometry. Diagnostics report that region's valid-depth coverage.

    Diagnostics preserve every candidate mask and geometry before semantic
    selection. The selected plane is copied so subsequent direction/sign work
    cannot silently alter those recorded candidates. No file path, simulator
    state, whole-owner mask, source identity or metric VLM output is accepted.
    """
    image = _image(image)
    region = _mask(region_mask, image, "Selected interaction region")
    raw_depth = np.asarray(depth)
    if raw_depth.shape != region.shape or raw_depth.dtype.kind not in "iuf":
        raise ValueError("Registered metric depth must be a numeric array matching the RGB image")
    depth = raw_depth.astype(float, copy=False)
    intrinsics = _intrinsics(K)
    positive_depth = np.isfinite(depth) & (depth > 0)

    region_box = _box(region)
    left, top, right, bottom = region_box
    padding = math.ceil(.6 * max(right - left, bottom - top))
    context_box = (max(0, left - padding), max(0, top - padding),
                   min(image.width, right + padding), min(image.height, bottom + padding))
    left, top, right, bottom = context_box
    valid = np.zeros(region.shape, dtype=bool)
    valid[top:bottom, left:right] = True
    valid &= positive_depth & ~region
    if np.count_nonzero(valid) < 24:
        raise ValueError("Local context has fewer than 24 valid depth pixels outside the interaction region")
    candidates, sample_count = _plane_candidates(depth, intrinsics, valid)
    if not candidates:
        raise ValueError("Local context has no sufficiently supported observed backing-plane candidate")

    # The coloured part identifies the semantic anchor in both local displays.
    # Red plane masks remain distinct from this green interaction region.
    marked = _highlight(image, region, (30, 225, 110))
    context = marked.crop(context_box)
    scale = 256 / max(context.size)
    context = context.resize((max(1, round(context.width * scale)),
                              max(1, round(context.height * scale))), Image.Resampling.NEAREST)
    prompt = _BACKING_INSTRUCTION + "\nObserved candidates:\n" + json.dumps({
        "context_box_xyxy": list(context_box),
        "allowed_ids": [candidate["candidate_id"] for candidate in candidates],
        "candidates": [{"candidate_id": candidate["candidate_id"], "bbox": candidate["bbox"]}
                       for candidate in candidates],
    })
    response = _client(client).chat_json(prompt, images=[image, context, _panels(marked, candidates)])
    if (not isinstance(response, dict) or set(response) != {"candidate_id", "reason"}
            or not isinstance(response["reason"], str) or not response["reason"].strip()):
        raise ValueError("Backing selection requires exactly candidate_id and a nonempty reason")
    number = response["candidate_id"]
    if number is None:
        raise ValueError("No unambiguous observed backing surface: " + response["reason"].strip())
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    if type(number) is not int or number not in by_id:
        raise ValueError("Backing selection must refer to an existing observed candidate ID")

    diagnostics = {
        "selected_candidate_id": number,
        "selection_reason": response["reason"].strip(),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "sampled_point_count": sample_count,
        "interaction_bbox_xyxy": list(region_box),
        "context_box_xyxy": list(context_box),
        "context_padding_pixels": padding,
        "context_valid_depth_pixels": int(np.count_nonzero(valid)),
        "interaction_pixels": int(np.count_nonzero(region)),
        "interaction_valid_depth_pixels": int(np.count_nonzero(region & positive_depth)),
        "coordinate_frame": "input RGB-D camera; metres",
        "normal_sign": "unoriented; physical protrusion side is not established here",
        "geometry_scope": "observed local backing-plane witness, not contact geometry or a whole owner",
        "vlm_used": True,
    }
    return deepcopy(by_id[number]), diagnostics
# END local protruding-region backing evidence without whole-owner detection.
