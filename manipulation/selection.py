"""Task-conditioned selection of observed owners or interaction regions.

The same request builder consumes compiled demonstration semantics and actual
detector candidates for any skill. It generates no prompts with a second LLM,
does not estimate geometry, and never substitutes a part mask for its owner.
"""

# BEGIN generic grounding selection from source evidence and current intent.
import json
import math

import numpy as np
from PIL import Image, ImageDraw

from .grounding import _client, _detection, _image, _mask


_SELECTION_INSTRUCTION = (
    "Select observed candidates for the CURRENT instruction using the demonstrated "
    "functional role and the supplied visual evidence. The source_evidence fields "
    "describe a demonstration, not a command overriding the current instruction. "
    "Source-only position, level, colour, camera direction or world orientation "
    "must not become current-instance requirements unless the current instruction "
    "actually requests them. Preserve the demonstrated functional contact role. "
    "Do not infer a new target from a skill name.\n"
    "selection_scope states exactly what is being selected: owner means the whole "
    "rigid entity or moving rigid part marked GREEN in the source image; it is not "
    "necessarily the largest enclosing object. interaction_region means the local "
    "functional contact region marked RED. A local part is not its whole owner; "
    "a whole object is not a precise interaction-region mask. Source query descriptions "
    "can mention surrounding geometry used for directions: these are contextual "
    "witnesses, not additional interaction targets and need not fit inside the red mask.\n"
    "Each current view has an original image and, when nonempty, numbered candidate "
    "panels. Panels enlarge existing pixels only. Use the original scene to determine "
    "which object or moving part a candidate belongs to. Candidate labels are detector "
    "queries, not proof of identity. Verify even a single candidate. Select only "
    "provided IDs within that view; equal IDs in different views do not establish "
    "physical correspondence. Images from other supplied views may provide context, "
    "but do not invent a mask in a view without a candidate.\n"
    "For each view return candidate_ids as: one ID for a uniquely supported match; "
    "multiple IDs only if EVERY listed candidate independently satisfies the current "
    "semantic request, so choosing any is allowed; [] when no supplied candidate "
    "matches; null when evidence cannot resolve which candidate meets the request. "
    "Uncertainty between possible targets is NOT permission to use all of them. "
    "If a view has no candidates, return [] for it. Do not force an answer from weak "
    "or occluded evidence. Do not merge masks, generate coordinates, geometry, poses "
    "or claim grasp feasibility. This is semantic selection only.\n"
    "Return exactly {\"views\": {view_id: {\"candidate_ids\": [integer_ID, ...] "
    "or null, \"reason\": short_nonempty_string}, ...}}, with exactly the supplied "
    "views. Explain the visible evidence or the reason for abstaining briefly."
)


def _box(mask):
    """Compute display bounds from an existing mask, without changing its pixels."""
    yy, xx = np.nonzero(mask)
    return (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)


def _highlight(image, mask, colour, *, outline=True):
    panel = Image.composite(Image.new("RGB", image.size, colour), image,
                            Image.fromarray(mask.astype(np.uint8) * 85))
    if outline:
        ImageDraw.Draw(panel).rectangle(_box(mask), outline=colour, width=1)
    return panel


def _context_crop(image, box):
    """Enlarge small candidate evidence for presentation, never for detection."""
    x0, y0, x1, y1 = box
    margin = max(10, math.ceil(max(x1 - x0, y1 - y0) * .6))
    crop = image.crop((max(0, math.floor(x0) - margin), max(0, math.floor(y0) - margin),
                       min(image.width, math.ceil(x1) + margin),
                       min(image.height, math.ceil(y1) + margin)))
    scale = min(256 / crop.width, 256 / crop.height)
    return crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))),
                       Image.Resampling.NEAREST)


def _panels(image, candidates):
    """Keep candidate masks independent and show a local crop of each one."""
    columns = min(3, len(candidates))
    rows = math.ceil(len(candidates) / columns)
    sheet = Image.new("RGB", (columns * 272, rows * 290), (22, 28, 35))
    for index, candidate in enumerate(candidates):
        panel = _highlight(image, candidate["mask"], (255, 70, 40))
        # The detector box need only overlap its mask. Show the entire actual
        # mask so an outlying component cannot be hidden from semantic review.
        crop = _context_crop(panel, _box(candidate["mask"]))
        left, top = index % columns * 272, index // columns * 290
        sheet.paste(crop, (left + (272 - crop.width) // 2, top + 26 + (256 - crop.height) // 2))
        # Only IDs are rendered; natural-language labels remain in JSON.
        ImageDraw.Draw(sheet).text((left + 8, top + 6), str(candidate["candidate_id"]), fill="white")
    return sheet


def _prepare(source, instruction, views, selection_scope):
    """Allowlist semantic evidence; source poses and simulator IDs are not sent."""
    if selection_scope not in {"owner", "interaction_region"}:
        raise ValueError("selection_scope must be owner or interaction_region")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A current instruction is required")
    if not isinstance(source, dict) or not {"image", "mask", "description"} <= source.keys():
        raise ValueError("Source requires image, mask and description")
    description = source["description"]
    queries = source.get("query_descriptions", [])
    if not isinstance(description, str) or not description.strip():
        raise ValueError("Source description must be nonempty")
    if (not isinstance(queries, list)
            or any(not isinstance(query, str) or not query.strip() for query in queries)):
        raise ValueError("Source query_descriptions must be a list of nonempty strings")
    source_image = _image(source["image"])
    interaction = _mask(source["mask"], source_image, "Source interaction mask")
    owner = None
    if "owner_mask" in source:
        owner = _mask(source["owner_mask"], source_image, "Source owner mask")
    if selection_scope == "owner" and owner is None:
        raise ValueError("Owner selection requires the source owner_mask")
    panel = _highlight(source_image, interaction, (255, 65, 65), outline=False)
    if owner is not None:
        # An outline gives source context without occluding the red contact area.
        ImageDraw.Draw(panel).rectangle(_box(owner), outline=(30, 225, 110), width=2)
    source_extent = owner if owner is not None else interaction
    images = [panel, _context_crop(panel, _box(source_extent))]
    image_order = ["source scene: red interaction region, green owner bounds when available",
                   "source detail: same annotation, enlarged existing pixels"]
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty view-ID mapping")
    prepared, descriptions = {}, {}
    for name, view in views.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("View IDs must be nonempty strings")
        if not isinstance(view, dict) or not {"image", "candidates"} <= view.keys():
            raise ValueError("Each view requires image and candidates")
        image = _image(view["image"])
        raw = view["candidates"]
        if not isinstance(raw, list):
            raise ValueError("Candidates must be a list of actual detections")
        if any(not isinstance(item, dict) or not isinstance(item.get("label"), str)
               or not item["label"].strip() for item in raw):
            raise ValueError("Each candidate needs a nonempty detector label")
        labels = list(dict.fromkeys(item["label"] for item in raw))
        candidates, used = [], set()
        for index, item in enumerate(raw, 1):
            number = item.get("candidate_id", index)
            if type(number) is not int or number < 1 or number in used:
                raise ValueError("Candidate IDs must be distinct positive integers within a view")
            used.add(number)
            candidates.append(_detection(item, image, labels, number))
        prepared[name] = candidates
        descriptions[name] = {
            "allowed_ids": [item["candidate_id"] for item in candidates],
            "candidates": [{"candidate_id": item["candidate_id"], "label": item["label"],
                            "bbox": item["bbox"]} for item in candidates],
        }
        images.append(image)
        image_order.append(f"view {name}: original current scene")
        if candidates:
            images.append(_panels(image, candidates))
            image_order.append(f"view {name}: numbered candidate details")
    context = {"selection_scope": selection_scope, "current_instruction": instruction.strip(),
               "source_evidence": {"description": description.strip(), "query_descriptions": queries},
               "views": descriptions,
               "image_order": [{"image": index + 1, "content": content}
                               for index, content in enumerate(image_order)]}
    prompt = _SELECTION_INSTRUCTION + "\nInput data:\n" + json.dumps(context, ensure_ascii=False)
    return prompt, images, prepared


def select_grounding_candidates(source, instruction, views, *, selection_scope, client=None):
    """Select semantic candidates using a fixed, automatically assembled prompt.

    ``source`` consumes existing ``image``, interaction ``mask``, ``description``
    and optional ``query_descriptions``. ``selection_scope='owner'`` additionally
    requires ``source.owner_mask``. These are compiled demonstration evidence;
    only ``instruction`` describes the current request. No source file path,
    skill-name branch, source pose or current oracle geometry is used.

    Each ``views[view_id]`` contains an image and detector candidates. Explicit
    positive integer candidate IDs are preserved; otherwise the input index
    starting at one is used. A normal single-view call supplies one mapping
    entry. Multiple supplied views share semantic context only: their IDs and
    masks are not fused into a physical or geometric association.

    Each result has a status: selected, alternatives (all semantically allowed),
    unresolved, no_match, or no_candidates. ``regions`` retains the separate
    selected masks. Owner and local interaction scopes are explicit in the
    returned record; this function never silently constructs an owner_mask.
    Geometry, correspondence, accessibility and grasp feasibility remain the
    responsibilities of the existing downstream consumers.
    """
    prompt, images, candidates = _prepare(source, instruction, views, selection_scope)
    called = any(candidates.values())
    if called:
        response = _client(client).chat_json(prompt, images=images)
    else:
        response = {"views": {name: {"candidate_ids": [], "reason": "No detector candidates were supplied."}
                              for name in candidates}}
    if (not isinstance(response, dict) or set(response) != {"views"}
            or not isinstance(response["views"], dict) or set(response["views"]) != set(candidates)):
        raise ValueError("Selection must return exactly the supplied view IDs")
    result = {}
    for name, items in candidates.items():
        decision = response["views"][name]
        if not isinstance(decision, dict) or set(decision) != {"candidate_ids", "reason"}:
            raise ValueError("Each view decision requires exactly candidate_ids and reason")
        ids, reason = decision["candidate_ids"], decision["reason"]
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("Selection reason must be a nonempty string")
        by_id = {item["candidate_id"]: item for item in items}
        if ids is not None:
            if (not isinstance(ids, list) or any(type(number) is not int or number not in by_id for number in ids)
                    or len(set(ids)) != len(ids)):
                raise ValueError("Selected IDs must be distinct integers from this view's candidates")
        if not items:
            if ids != []:
                raise ValueError("A view with no candidates must return an empty list")
            status = "no_candidates"
        elif ids is None:
            status = "unresolved"
        elif not ids:
            status = "no_match"
        else:
            status = "selected" if len(ids) == 1 else "alternatives"
        result[name] = {"status": status, "candidate_ids": ids, "reason": reason.strip(),
                        "regions": [by_id[number] for number in ids] if ids else []}
    return {"selection_scope": selection_scope, "views": result, "vlm_used": called}
# END generic grounding selection.
