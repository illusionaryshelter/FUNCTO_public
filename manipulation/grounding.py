"""Ground one manipulation stage without imposing tool/function/grasp points.

Gemini proposes semantic roles; SAM supplies instances; SD-DINO supplies only
two-image correspondences. None of these functions estimates an articulation
axis, reconstructs hidden geometry, or tracks a rigid pose through a video.
Models and network clients are imported only when a caller requests their use.
"""

# BEGIN task-adaptive grounding: the existing tool-use entry points stay separate.
import json

import numpy as np
from PIL import Image, ImageDraw


_CHOICES = {
    "relation": {"transfer", "on", "inside", "articulation", "tool_use"},
    "interaction": {"none", "grasp", "push", "pull", "tool"},
    "motion": {"free", "planar", "translation_plane", "revolute", "prismatic", "auto"},
}
_ROLES = ("carrier", "reference")


def _image(image):
    if not isinstance(image, Image.Image) or min(image.size) < 1:
        raise ValueError("Expected a nonempty PIL image")
    return image.convert("RGB")


def _task(task):
    """Reject extra fields, particularly ungrounded numeric geometry from a VLM."""
    required = set(_CHOICES) | set(_ROLES)
    if not isinstance(task, dict) or not required <= task.keys():
        raise ValueError("Task requires relation, interaction, motion, carrier and reference")
    if task.keys() - required - {"goal_description"}:
        raise ValueError("Task contains unsupported fields; geometry is not semantic output")
    result = {}
    for field, values in _CHOICES.items():
        if not isinstance(task[field], str) or task[field] not in values:
            raise ValueError(f"Invalid task {field}: {task[field]!r}")
        result[field] = task[field]
    for role in _ROLES:
        labels = task[role]
        if not isinstance(labels, dict) or set(labels) != {"demo_label", "scene_label"}:
            raise ValueError(f"{role} requires exactly demo_label and scene_label")
        if any(not isinstance(label, str) or not label.strip() for label in labels.values()):
            raise ValueError(f"{role} labels must be nonempty strings")
        result[role] = {key: value.strip() for key, value in labels.items()}
    if "goal_description" in task:
        goal = task["goal_description"]
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal_description must be a nonempty string")
        result["goal_description"] = goal.strip()
    return result


def _client(client):
    if client is None:
        from local_models.gemini import Gemini

        return Gemini()
    return client


def infer_task(instruction, demo_image, scene_image, *, demo_final_image=None, client=None):
    """Infer semantic candidates for one active-carrier stage, never metric geometry.

    The optional demonstration final image provides outcome context. This is not
    a multistage planner: stage splitting and subsequent state updates belong to
    the caller. Motion remains a proposal until validated against observations.
    """
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a nonempty string")
    images = [_image(demo_image), _image(scene_image)]
    image_order = "Image 1: demonstration initial scene. Image 2: destination initial scene."
    if demo_final_image is not None:
        images.append(_image(demo_final_image))
        image_order += " Image 3: demonstration final scene, showing the demonstrated outcome."
    prompt = (
        "Ground the supplied instruction for ONE demonstrated manipulation stage. "
        "The carrier is the object or moving rigid part whose trajectory transfers; "
        "the reference is a distinct object, fixed body, support surface or container. "
        "A moving part and its parent body may overlap in the image. They are distinct "
        "roles, and the same physical instance cannot fill both roles. Do not require "
        "a tool, function point or grasp point. Infer interaction from the instruction "
        # BEGIN requested-goal precedence: the same video can supply motion for
        # a different requested relation. Visible side effects must not replace
        # an explicit request to transfer only object position/orientation.
        "and visual demonstration, not solely from an object name. The instruction "
        "defines the requested success criterion; images identify entities and "
        "demonstrated motion. Do not add a functional effect merely because one is "
        "visible in the video. When the instruction explicitly asks only for object "
        "motion or final position/orientation, choose the corresponding object "
        "relation even if that demonstration also shows a functional side effect. "
        # END requested-goal precedence.
        "Relation meanings: "
        "transfer means relocating/reorienting the carrier itself is the goal; "
        "on means the carrier itself comes to rest on a support; inside means the "
        "carrier itself goes into a container's free interior; articulation means "
        "moving a mechanically constrained part relative to its parent body. "
        "tool_use means the carrier performs a function on the reference using "
        "a working region, including delivering its contained material into the "
        "reference or modifying/cleaning the reference. Its functional effect, "
        "not merely relocating the carrier, defines success. This functional "
        "relation, when its functional effect is requested, takes precedence over "
        "the transport/rotation needed to perform it; a vessel used to deliver "
        "its contents is a functional carrier even "
        "if it is also grasped. For tool_use choose interaction=tool. Merely "
        "moving a potential tool into storage is still on/inside/transfer. "
        # BEGIN motion evidence: task requirements and geometry are separate.
        "For motion, planar means the whole demonstrated stage is constrained to "
        "translation within a fixed plane and rotation about its normal (SE(2)), "
        "for example by a visible maintained support/contact relation or an explicit "
        "task requirement. A final placement on a surface does not constrain the "
        "preceding airborne carry to that plane. translation_plane additionally "
        "requires fixed orientation. Propose prismatic/revolute from evidence of "
        "a moving part relative to its parent or an explicit motion requirement, "
        "not because a free object happens to follow a line or arc. The geometric "
        "stage will verify each proposal against the complete pose sequence; "
        "choose auto when available evidence does not establish an allowed-motion "
        "constraint. Tool use and a planar/other motion constraint may coexist: "
        "retain both requirements instead of changing the relation to transfer "
        "or dropping the motion constraint to fit a narrower downstream capability. "
        # END motion evidence.
        "Propose semantics only: do not invent axes, distances, depths, angles, limits "
        "or unseen container geometry. Labels must be concise SAM text prompts and "
        "may differ between demonstration and destination. Include goal_description "
        "to retain visual/spatial instance-identifying details and the intended outcome. "
        "Return exactly these JSON fields: relation, interaction, motion, carrier, "
        "reference, goal_description. carrier/reference each contain only demo_label "
        "and scene_label. Allowed values: "
        + json.dumps({key: sorted(value) for key, value in _CHOICES.items()})
        + "\n" + image_order + "\nInstruction: " + json.dumps(instruction.strip())
    )
    return _task(_client(client).chat_json(prompt, images=images))


def _mask(mask, image, name):
    mask = np.asarray(mask)
    if mask.shape != (image.height, image.width):
        raise ValueError(f"{name} must match image height and width")
    if mask.dtype.kind not in "buif" or not np.isfinite(mask).all():
        raise ValueError(f"{name} must be a finite binary mask")
    if not np.logical_or(mask == 0, mask == 1).all() or not mask.any():
        raise ValueError(f"{name} must be binary and nonempty")
    return mask.astype(bool, copy=False)


def _detection(value, image, labels, number):
    if not isinstance(value, dict) or not {"label", "score", "bbox", "mask"} <= value.keys():
        raise ValueError("Detection requires label, score, bbox and mask")
    if value["label"] not in labels:
        raise ValueError("Detector returned an unrequested label")
    score = value["score"]
    if isinstance(score, (bool, np.bool_)) or not np.isscalar(score):
        raise ValueError("Detection score must be a finite probability")
    try:
        score = float(score)
        bbox = np.asarray(value["bbox"], dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError("Detection score and bbox must be numeric") from error
    if not np.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Detection score must be a finite probability")
    if bbox.shape != (4,) or not np.isfinite(bbox).all():
        raise ValueError("Detection bbox must contain four finite xyxy coordinates")
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
        raise ValueError("Detection bbox lies outside the image or has no area")
    mask = _mask(value["mask"], image, "Detection mask")
    yy, xx = np.nonzero(mask)
    if xx.max() + 1 <= x1 or xx.min() >= x2 or yy.max() + 1 <= y1 or yy.min() >= y2:
        raise ValueError("Detection bbox and mask do not overlap")
    return {**value, "score": score, "bbox": bbox.tolist(), "mask": mask,
            "candidate_id": number}


def _candidate_sheet(image, detections):
    """Number each mask in its own panel so nested parts cannot hide each other.

    Pillow avoids importing the legacy annotation stack (torchvision and its
    visualization dependencies) merely to prepare a semantic selection image.
    """
    panel_width, panel_height, columns = 256, 280, min(3, len(detections))
    rows = (len(detections) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * panel_width, rows * panel_height), "white")
    for index, item in enumerate(detections):
        alpha = Image.fromarray(item["mask"].astype(np.uint8) * 90)
        panel = Image.composite(Image.new("RGB", image.size, (255, 70, 40)), image, alpha)
        draw = ImageDraw.Draw(panel)
        draw.rectangle(item["bbox"], outline=(255, 255, 0), width=2)
        panel.thumbnail((panel_width, panel_width))
        left, top = index % columns * panel_width, index // columns * panel_height
        sheet.paste(panel, (left, top + 24))
        # Only numeric IDs are drawn: labels remain in the JSON prompt, avoiding
        # font-dependent failures for non-Latin category names.
        ImageDraw.Draw(sheet).text((left + 5, top + 5), str(item["candidate_id"]), fill="black")
    return sheet


def select_roles(image, task, domain="demo", *, detector=None, client=None):
    """Bind carrier/reference to actual SAM instances, selecting only if ambiguous.

    Both roles are required for this stage interface. Masks may overlap for a
    moving part and its body, but identical masks or one selected ID are rejected.
    This image-level check is not proof of physical identity in an occluded scene.
    """
    image, task = _image(image), _task(task)
    if domain not in {"demo", "scene"}:
        raise ValueError("domain must be demo or scene")
    requested = {role: task[role][f"{domain}_label"] for role in _ROLES}
    labels = list(dict.fromkeys(requested.values()))
    if detector is None:
        from local_models.sam31 import detect

        detector = detect
    values = detector(image, labels)
    if not isinstance(values, list):
        raise ValueError("Detector must return a list of detections")
    detections = [_detection(item, image, labels, index + 1) for index, item in enumerate(values)]
    candidates = {
        role: [item["candidate_id"] for item in detections if item["label"] == label]
        for role, label in requested.items()
    }
    for role, numbers in candidates.items():
        if not numbers:
            raise ValueError(f"No {domain} {role} instance for label {requested[role]!r}")
    if all(len(numbers) == 1 for numbers in candidates.values()):
        selected = {role: numbers[0] for role, numbers in candidates.items()}
    else:
        prompt = (
            "Select the carrier and reference for this ONE manipulation stage from "
            "the numbered SAM mask panels. Image 1 is the original scene; image 2 "
            "has one panel per candidate, showing its red mask and yellow bbox. "
            "Use the semantic task and visible context; a detection score alone is "
            "not an instance identity. Choose only an allowed candidate ID per role. "
            "Carrier and reference must be distinct physical instances; a moving "
            "part and its parent fixed body may have overlapping masks. Return "
            "exactly {\"carrier\": integer_ID, \"reference\": integer_ID}. If the "
            "evidence cannot distinguish the required instance, use null for that "
            "role instead of guessing (the caller will stop).\n"
            + json.dumps({"domain": domain, "task": task, "allowed_ids": candidates,
                          "candidates": [{"id": item["candidate_id"], "label": item["label"],
                                          "bbox": item["bbox"]} for item in detections]})
        )
        selected = _client(client).chat_json(prompt, images=[image, _candidate_sheet(image, detections)])
        if not isinstance(selected, dict) or set(selected) != set(_ROLES):
            raise ValueError("Role selection must contain exactly carrier and reference IDs")
    for role, number in selected.items():
        if type(number) is not int or number not in candidates[role]:
            raise ValueError(f"Unresolved or invalid {role} candidate ID: {number!r}")
    carrier, reference = (detections[selected[role] - 1] for role in _ROLES)
    if (selected["carrier"] == selected["reference"]
            or np.array_equal(carrier["mask"], reference["mask"])):
        raise ValueError("Carrier and reference cannot be the same instance or identical mask")
    return {"carrier": carrier, "reference": reference}


# BEGIN source-evidence query proposals: current interaction masks are inferred, not inputs.
def propose_query_regions(source_image, current_image, source_owner_mask,
                          current_owner_mask, source_region_mask, query_pixels, *,
                          description, query_descriptions, client=None):
    """Ask once for coarse current-owner cells corresponding to source queries.

    Images are owner-context crops; xy queries and returned integer xyxy boxes
    use their respective crop coordinates. Right/bottom box edges are exclusive.
    Source queries may lie outside the interaction region, because functional
    directions can require evidence elsewhere on the same owner. Descriptions
    explain that evidence; they are actually passed to the semantic selector.
    Noisy source region pixels outside the owner remain visible and are counted;
    nonempty overlap is required, while every query must belong to the owner.

    The current owner is partitioned by a 5x5 image grid, discarding empty cells.
    A selected cell gets a quarter-cell margin for the downstream dense search.
    These boxes are program-defined search windows, NOT predicted part masks or
    metric geometry. The caller must intersect each window with the current
    owner and valid depth, then assess the resulting correspondences. Null or
    invalid semantic selections fail explicitly rather than guessing a region.
    """
    source_image, current_image = _image(source_image), _image(current_image)
    source_owner = _mask(source_owner_mask, source_image, "source_owner_mask")
    current_owner = _mask(current_owner_mask, current_image, "current_owner_mask")
    source_region = _mask(source_region_mask, source_image, "source_region_mask")
    if not np.any(source_region & source_owner):
        raise ValueError("Source interaction region does not overlap its owner")
    source_spill = int(np.count_nonzero(source_region & ~source_owner))
    pixels = np.asarray(query_pixels)
    if (pixels.dtype.kind not in "iuf" or pixels.ndim != 2
            or pixels.shape[1] != 2 or not len(pixels)):
        raise ValueError("query_pixels must be a nonempty numeric Nx2 xy array")
    pixels = pixels.astype(float)
    if (not np.isfinite(pixels).all() or np.any(pixels < 0)
            or np.any(pixels > [source_image.width-1, source_image.height-1])):
        raise ValueError("Source queries must be finite and inside source_image")
    # Match the existing descriptor sampler/backprojection nearest-pixel rule.
    nearest = np.rint(pixels).astype(int)
    if not source_owner[nearest[:, 1], nearest[:, 0]].all():
        raise ValueError("Every source query must belong to the source owner")
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must explain the demonstrated interaction")
    if (not isinstance(query_descriptions, (list, tuple))
            or len(query_descriptions) != len(pixels)
            or any(not isinstance(item, str) or not item.strip()
                   for item in query_descriptions)):
        raise ValueError("query_descriptions must give one nonempty description per query")

    def owner_panel(image, owner):
        """Retain owner appearance and context while marking the chosen instance."""
        array = np.array(image, copy=True)
        array[~owner] = np.rint(array[~owner].astype(float) * .45).astype(np.uint8)
        padded = np.pad(owner, 1, constant_values=False)
        interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                    & padded[1:-1, :-2] & padded[1:-1, 2:])
        array[owner & ~interior] = (40, 255, 90)
        return Image.fromarray(array)

    source_panel = owner_panel(source_image, source_owner)
    # A translucent source-only overlay records the supplied interaction part;
    # it does not hide other owner locations needed to define an axis or plane.
    source_panel = Image.composite(
        Image.new("RGB", source_image.size, (255, 65, 45)), source_panel,
        Image.fromarray(source_region.astype(np.uint8) * 70),
    )

    def display_panel(panel):
        """Scale presentation only; small native crops otherwise lose their parts
        under grid labels. Pixel and edge mappings below stay explicit, and the
        unannotated images used by correspondence models are never resized here.
        """
        scale = 480 / max(panel.size)
        size = tuple(max(1, int(round(length*scale))) for length in panel.size)
        scales = np.asarray(size, dtype=float) / np.asarray(panel.size, dtype=float)
        return panel.resize(size, Image.Resampling.BICUBIC), scales

    source_panel, source_scale = display_panel(source_panel)
    source_draw = ImageDraw.Draw(source_panel)
    for number, (x, y) in enumerate(pixels, 1):
        # Queries are pixel centers; cell rectangles below are pixel edges.
        x, y = (np.array([x, y])+.5)*source_scale-.5
        source_draw.ellipse((x-3, y-3, x+3, y+3), outline=(255, 255, 0), width=1)
        source_draw.text((max(0, min(x+5, source_panel.width-14)),
                          max(0, min(y-14, source_panel.height-14))), str(number),
                         fill=(255, 255, 0), stroke_width=1, stroke_fill="black")

    current_panel, current_scale = display_panel(owner_panel(current_image, current_owner))
    current_draw = ImageDraw.Draw(current_panel)
    x_edges = np.linspace(0, current_image.width, 6, dtype=int)
    y_edges = np.linspace(0, current_image.height, 6, dtype=int)
    cells = {}
    for row in range(5):
        for column in range(5):
            x0, x1 = map(int, x_edges[column:column+2])
            y0, y1 = map(int, y_edges[row:row+2])
            yy, xx = np.nonzero(current_owner[y0:y1, x0:x1])
            if not len(xx):
                continue
            # Grid IDs stay row-major even when empty cells are absent. This
            # lets response validation reject cells without any owner pixels.
            number = row * 5 + column + 1
            cells[number] = [x0, y0, x1, y1]
            display_box = [x0*current_scale[0], y0*current_scale[1],
                           x1*current_scale[0]-1, y1*current_scale[1]-1]
            current_draw.rectangle(display_box, outline=(255, 210, 0), width=1)
            xx, yy = xx+x0, yy+y0
            index = int(np.argmin((xx-(x0+x1-1)/2)**2 + (yy-(y0+y1-1)/2)**2))
            tx, ty = (np.array([xx[index], yy[index]])+.5)*current_scale-.5
            current_draw.text((max(0, min(tx, current_panel.width-14)),
                               max(0, min(ty-5, current_panel.height-14))),
                              str(number), fill="white", stroke_width=1, stroke_fill="black")

    details = {
        "description": description.strip(),
        "source_interaction_pixels_outside_owner": source_spill,
        "queries": [{"id": index+1, "source_xy": point.tolist(),
                     "description": query_descriptions[index].strip()}
                    for index, point in enumerate(pixels)],
        "allowed_cells": [{"id": number, "box_xyxy": box} for number, box in cells.items()],
    }
    prompt = (
        "Find functional visual correspondences from ONE demonstration owner to ONE "
        "selected current owner. Image 1 keeps the demonstration owner context: its "
        "green outline identifies the owner, translucent red highlights the known "
        "interaction region, and yellow numbered points are the source queries. "
        "Some queries deliberately lie outside the red region because they define "
        "functional directions; use their descriptions and the demonstrated purpose. "
        "The supplied red region may have noisy boundary pixels outside the green "
        "owner. They are shown unchanged; the selected owner takes priority and "
        "every source query belongs to it. Do not interpret spilled red pixels as "
        "another interaction owner. "
        "Image 2 shows the selected current owner in green and numbered coarse grid "
        "cells. Only cells containing current-owner pixels are allowed. For every "
        "source query, identify the current-owner location with the same functional "
        "role and choose the cell containing that location. This is coarse semantic "
        "search guidance; do not estimate a precise mask, coordinate, depth, axis, "
        "pose or distance. Multiple queries may select the same cell. If a query's "
        "functional counterpart is absent, occluded or cannot be identified from "
        "these images, return null for that query instead of guessing or selecting "
        "a merely similar-looking unrelated part. Return exactly {\"cells\": "
        "[integer_cell_ID_or_null, ...]}, in source query ID order, one entry per "
        "query. Do not return metric geometry or additional keys.\n"
        + json.dumps(details, ensure_ascii=False)
    )
    response = _client(client).chat_json(prompt, images=[source_panel, current_panel])
    if (not isinstance(response, dict) or set(response) != {"cells"}
            or not isinstance(response["cells"], list)
            or len(response["cells"]) != len(pixels)):
        raise ValueError("Query proposal must contain exactly one cells list matching the query count")
    selected = response["cells"]
    boxes = []
    for index, number in enumerate(selected):
        if number is None:
            raise ValueError(f"Unresolved functional correspondence for source query {index+1}")
        if type(number) is not int or number not in cells:
            raise ValueError(f"Invalid current-owner cell for source query {index+1}: {number!r}")
        x0, y0, x1, y1 = cells[number]
        # Small fixed margins absorb coarse grid-boundary uncertainty. Expansion
        # never substitutes for the caller's owner/valid-depth intersection.
        dx, dy = max(1, (x1-x0+3)//4), max(1, (y1-y0+3)//4)
        boxes.append([max(0, x0-dx), max(0, y0-dy),
                      min(current_image.width, x1+dx), min(current_image.height, y1+dy)])
    return {"boxes": np.asarray(boxes, dtype=np.int64), "diagnostics": {
        "description": details["description"],
        "query_descriptions": [item["description"] for item in details["queries"]],
        "source_interaction_pixels_outside_owner": source_spill,
        "query_count": len(pixels), "grid_shape": [5, 5],
        "allowed_cells": details["allowed_cells"], "selected_cell_ids": selected.copy(),
        "cell_boxes": [cells[number].copy() for number in selected],
        "search_boxes": boxes, "margin_fraction": .25,
        "box_convention": "current-image xyxy; right and bottom exclusive",
        "display": {
            "source": {"native_size": list(source_image.size), "panel_size": list(source_panel.size),
                       "scale_xy": source_scale.tolist()},
            "current": {"native_size": list(current_image.size), "panel_size": list(current_panel.size),
                        "scale_xy": current_scale.tolist()},
            "pixel_center_mapping": "(native_xy + 0.5) * scale_xy - 0.5",
            "edge_mapping": "native_xy * scale_xy",
        },
        "scope": ("VLM-selected program-defined coarse search windows; "
                  "not a current interaction mask or metric correspondence"),
        "client_calls": 1,
    }}
# END source-evidence query proposals.


def _square(image):
    """Pad before extraction, avoiding different SD/DINO resize-rounding offsets."""
    size = max(image.size)
    left, top = (size - image.width) // 2, (size - image.height) // 2
    square = Image.new("RGB", (size, size), "black")
    square.paste(image, (left, top))
    return square, np.array([left, top], dtype=float)


# BEGIN dense masked correspondence: bounded descriptor batches replace cell representatives.
def _sample(features, pixels, size, offset):
    """Bilinear sampling with pixel centers and align_corners=False coordinates.

    Interpolation and subsequent cosine arithmetic use float64, while the
    extractor's feature maps stay in their original FP32 storage. Close
    original-image pixels can have much closer scores than feature-cell centers.
    Only the current small batch is promoted, never an image-sized feature map.
    """
    _, height, width = features.shape
    locations = (pixels + offset + 0.5) * np.array([width, height]) / size - 0.5
    locations = np.clip(locations, [0, 0], [width - 1, height - 1])
    x0, y0 = np.floor(locations).astype(int).T
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (locations - np.floor(locations)).T
    return (
        features[:, y0, x0].T * ((1 - wx) * (1 - wy))[:, None]
        + features[:, y0, x1].T * (wx * (1 - wy))[:, None]
        + features[:, y1, x0].T * ((1 - wx) * wy)[:, None]
        + features[:, y1, x1].T * (wx * wy)[:, None]
    )


def match_points(image1, image2, mask1, mask2, pixels1, *, extractor=None):
    """Transfer optional xy anchors by exhaustive masked original-pixel matching.

    Full images are padded to squares without resizing their original pixels.
    The existing extractor then applies its unchanged SD/DINO feature profile.
    Feature (j,i) spans the square under the half-pixel convention; padding is
    removed only when reporting original-image coordinates. Every original
    target-mask pixel is bilinearly sampled and considered, including pixels
    between descriptor-cell centers. This finds the cosine optimum over integer
    valid pixels, not a claim of subpixel or semantic ground-truth accuracy.

    At most 32 source and 512 target descriptors are sampled per batch. Larger
    source sets repeat the target scan, retaining that bound; no image-sized
    C-channel descriptor tensor or full source-by-target score matrix is built.
    Only mask indices and per-source winners/scores persist across batches.

    Returned diagnostics expose tied scores and many-to-one matches. No greedy
    reassignment fabricates distinct points. Callers must assess these diagnostics
    before using correspondences to infer geometry; scores are not probabilities.
    An empty Nx2 array means no anchors are needed and does not run the extractor.
    """
    image1, image2 = _image(image1), _image(image2)
    mask1, mask2 = _mask(mask1, image1, "mask1"), _mask(mask2, image2, "mask2")
    pixels = np.asarray(pixels1)
    if pixels.dtype.kind not in "iuf" or pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError("pixels1 must be a numeric Nx2 array in xy order")
    pixels = pixels.astype(float)
    if not np.isfinite(pixels).all() or np.any(pixels < 0) or np.any(
            pixels > [image1.width - 1, image1.height - 1]):
        raise ValueError("Source pixels must be finite and inside image1")
    # Match observations.backproject: nearest pixel center, with ties to even.
    # Using floor(x + .5) here would disagree at exact half-pixel boundaries.
    nearest = np.rint(pixels).astype(int)
    if not mask1[nearest[:, 1], nearest[:, 0]].all():
        raise ValueError("Every source pixel must lie inside mask1")
    if not len(pixels):
        return np.empty((0, 2), dtype=float), {"point_count": 0, "extractor_called": False}
    square1, offset1 = _square(image1)
    square2, offset2 = _square(image2)
    if extractor is None:
        from local_models.sd_dino import extract_pair

        extractor = extract_pair
    pair = extractor(square1, square2)
    if not isinstance(pair, (tuple, list)) or len(pair) != 2:
        raise ValueError("Extractor must return a pair of CPU feature arrays")
    features = []
    for feature in pair:
        value = np.asarray(feature)
        if (value.ndim != 4 or value.shape[0] != 1 or min(value.shape) < 1
                or value.dtype.kind != "f" or not np.isfinite(value).all()):
            raise ValueError("Features must be finite floating [1,C,Hf,Wf] arrays")
        value = value[0].astype(np.float32, copy=False)
        if not np.isfinite(value).all():
            raise ValueError("Feature values are not representable in FP32")
        features.append(value)
    if features[0].shape[0] != features[1].shape[0]:
        raise ValueError("Source and target descriptors must have equal channel counts")
    target_indices = np.flatnonzero(mask2)
    source_batch_size, target_batch_size = 32, 512
    best_indices = np.full(len(pixels), -1, dtype=np.int64)
    best_scores = np.full(len(pixels), -np.inf)
    runner_up_scores = np.full(len(pixels), -np.inf)
    candidate_count = 0
    for source_start in range(0, len(pixels), source_batch_size):
        source_end = min(source_start + source_batch_size, len(pixels))
        source = _sample(features[0], pixels[source_start:source_end], square1.width, offset1)
        norms = np.linalg.norm(source, axis=1)
        if np.any(norms == 0):
            index = source_start + int(np.flatnonzero(norms == 0)[0])
            raise ValueError(f"Source point {index} has a zero descriptor")
        source /= norms[:, None]
        best = best_scores[source_start:source_end]
        second = runner_up_scores[source_start:source_end]
        indices = best_indices[source_start:source_end]
        for target_start in range(0, len(target_indices), target_batch_size):
            flat = target_indices[target_start:target_start + target_batch_size]
            target_pixels = np.column_stack((flat % image2.width, flat // image2.width))
            target = _sample(features[1], target_pixels, square2.width, offset2)
            norms = np.linalg.norm(target, axis=1)
            valid = norms > 0
            if source_start == 0:
                candidate_count += int(valid.sum())
            if not valid.any():
                continue
            flat, target = flat[valid], target[valid]
            target /= norms[valid, None]
            similarities = np.clip(source @ target.T, -1.0, 1.0)
            winners = np.argmax(similarities, axis=1)
            block_best = similarities[np.arange(len(source)), winners]
            block_second = (np.partition(similarities, -2, axis=1)[:, -2]
                            if len(target) > 1 else np.full(len(source), -np.inf))
            # Merge the two maxima from each disjoint block. Strict improvement
            # keeps the first row-major pixel on equal scores, including ties
            # spanning blocks; negative maxima remain eligible throughout.
            merged = np.column_stack((best, second, block_best, block_second))
            better = block_best > best
            indices[better] = flat[winners[better]]
            best[:] = np.maximum(best, block_best)
            second[:] = np.partition(merged, -2, axis=1)[:, -2]
        if np.any(indices < 0):
            raise ValueError("No target candidate has a nonzero descriptor")
    scores = best_scores.tolist()
    second_scores = [float(value) if np.isfinite(value) else None for value in runner_up_scores]
    margins = [score - second if second is not None else None
               for score, second in zip(scores, second_scores)]
    ties = [margin is not None and margin <= 1e-6 for margin in margins]
    duplicate_groups = [np.flatnonzero(best_indices == index).tolist()
                        for index in np.unique(best_indices) if np.sum(best_indices == index) > 1]
    output = np.column_stack((best_indices % image2.width, best_indices // image2.width)).astype(float)
    diagnostics = {
        "point_count": len(pixels), "extractor_called": True,
        "candidate_count": candidate_count, "masked_pixel_count": len(target_indices),
        "excluded_zero_descriptors": len(target_indices) - candidate_count,
        "feature_shapes": [list(feature.shape) for feature in features],
        "padding_xy": [offset1.tolist(), offset2.tolist()],
        "square_sizes": [square1.width, square2.width],
        "mapping": "centered-square / half-pixel bilinear / all original mask pixels",
        "source_batch_size": source_batch_size, "target_batch_size": target_batch_size,
        "similarity_arithmetic": "float64 interpolation and cosine from stored FP32 features",
        "scores": scores, "second_scores": second_scores, "margins": margins,
        "numerical_ties": ties, "nonpositive_scores": [score <= 0 for score in scores],
        "duplicate_target_groups": duplicate_groups,
    }
    return output, diagnostics
# END dense masked correspondence.
# END task-adaptive grounding.
