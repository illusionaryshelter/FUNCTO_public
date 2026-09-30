"""Read compiled offline skills and observe their declared interaction reference.

The bundle owns both the demonstration semantics and the detection target.
Current instructions and images are runtime inputs; detector words are not.
This module stops at semantic selection. Existing geometry consumers recover
the functional correspondence and transfer the demonstrated grasp trajectory.
"""

# BEGIN portable offline skill artifacts and data-driven runtime detection.
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .grounding import _detection, _image, _mask
from .motion import _pose_array
from .observations import _intrinsics
from .selection import select_grounding_candidates


_SEMANTIC_FIELDS = {
    "object_description", "interaction_part_description", "interaction_reference",
    "description", "reason",
}
_QUERY_FIELDS = {"query_pixels", "query_descriptions", "frame_recipe", "compiled_source_frame"}


def _semantics(value):
    """Require the exact offline owner contract, without inventing missing words."""
    if not isinstance(value, dict) or set(value) != _SEMANTIC_FIELDS:
        raise ValueError("Skill semantics require the complete offline owner record")
    result = deepcopy(value)
    for field in _SEMANTIC_FIELDS - {"interaction_reference"}:
        if not isinstance(result[field], str) or not result[field].strip():
            raise ValueError(f"Skill semantic {field} must be a nonempty string")
    reference = result["interaction_reference"]
    if (not isinstance(reference, dict) or set(reference) != {"text", "scope"}
            or not isinstance(reference["scope"], str) or reference["scope"] not in {"object", "part"}
            or not isinstance(reference["text"], str) or not reference["text"].strip()):
        raise ValueError("interaction_reference requires text and scope object or part")
    if reference["scope"] == "object" and reference["text"] != result["object_description"]:
        raise ValueError("An object reference must use its object_description exactly")
    if any(character in reference["text"] for character in ("\n", "\r", "_")):
        raise ValueError("A detector reference must be a natural noun phrase, not an asset ID or prompt")
    return result


def _read_json(path):
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Skill JSON must contain an object: {path.name}")
    return value


def load_grounding_skill(directory):
    """Load a self-contained ``functo.grounding_skill.1`` bundle.

    ``skill.json`` declares ``owner_role``, ``geometry_kind`` and ``files``.
    Required file entries are semantic, template, program, observations and
    image. Protruding-line geometry also requires a geometry JSON attachment;
    query-frame and vertex consumers need no such attachment. File references
    must remain inside the bundle. Provenance strings are never opened.

    The source template stores geometry only: its runtime description and
    object/part semantics come exclusively from the offline semantic record.
    Query-frame templates retain their reviewed queries; other consumers must
    not inherit the discarded three-point recipe. NPZ loading forbids pickle.
    Nothing here reads simulator state or changes the demonstrated trajectory.
    """
    root = Path(directory).resolve()
    manifest = _read_json(root / "skill.json")
    if manifest.get("schema") != "functo.grounding_skill.1":
        raise ValueError("Unsupported grounding skill schema")
    if manifest.get("owner_role") not in {"carrier", "reference"}:
        raise ValueError("Grounding skill owner_role must be carrier or reference")
    kind = manifest.get("geometry_kind")
    if kind not in {"query_frame", "vertices", "protruding_line"}:
        raise ValueError("Unsupported grounding skill geometry_kind")
    files = manifest.get("files")
    required = {"semantic", "template", "program", "observations", "image"}
    if (not isinstance(files, dict) or not required <= files.keys()
            or files.keys() - required - {"geometry"}):
        raise ValueError("Grounding skill files must declare the complete bundle")
    if kind == "protruding_line" and "geometry" not in files:
        raise ValueError("protruding_line requires its compiled geometry attachment")
    paths = {}
    for name, relative in files.items():
        if not isinstance(relative, str) or not relative.strip() or Path(relative).is_absolute():
            raise ValueError("Skill file paths must be nonempty bundle-relative paths")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Skill file is missing or escapes its bundle")
        paths[name] = path
    semantic = _semantics(_read_json(paths["semantic"]))
    template = _read_json(paths["template"])
    if template.keys() & _SEMANTIC_FIELDS:
        raise ValueError("Source template duplicates semantics owned by semantic.json")
    if kind != "query_frame" and template.keys() & _QUERY_FIELDS:
        raise ValueError("Non-query geometry must not retain an obsolete query-frame recipe")
    if kind == "query_frame" and not {"query_pixels", "query_descriptions", "frame_recipe"} <= template.keys():
        raise ValueError("query_frame requires its reviewed functional queries and recipe")
    with Image.open(paths["image"]) as opened:
        image = opened.convert("RGB")
    with np.load(paths["observations"], allow_pickle=False) as archive:
        arrays = {key: archive[key].copy() for key in archive.files}
    if arrays.keys() & (template.keys() | _SEMANTIC_FIELDS | {"image"}):
        raise ValueError("Observation arrays cannot override template geometry or semantics")
    if not {"mask", "owner_mask", "depth", "K", "T_camera_owner"} <= arrays.keys():
        raise ValueError("Source observations require separate contact/owner masks and calibrated RGB-D")
    source = {**template, **arrays, **deepcopy(semantic), "image": image}
    source["mask"] = _mask(source["mask"], image, "Source contact mask")
    source["owner_mask"] = _mask(source["owner_mask"], image, "Source owner mask")
    if (np.asarray(source["depth"]).shape != (image.height, image.width)
            or not np.issubdtype(np.asarray(source["depth"]).dtype, np.number)):
        raise ValueError("Source depth must match the source image")
    source["K"] = _intrinsics(source["K"])
    source["T_camera_owner"] = _pose_array(source["T_camera_owner"], "Source camera-from-owner pose")
    if not isinstance(source.get("owner_id"), str) or not source["owner_id"].strip():
        raise ValueError("Source template must identify its demonstrated rigid owner")
    program = _read_json(paths["program"])
    demo = program.get("demo")
    if not isinstance(demo, dict):
        raise ValueError("Compiled program must contain its demonstrated trajectory")
    declared_owner = demo.get(manifest["owner_role"] + "_id")
    if declared_owner is not None and declared_owner != source["owner_id"]:
        raise ValueError("Source template and compiled program name different owners")
    geometry = _read_json(paths["geometry"]) if "geometry" in paths else None
    if kind == "protruding_line":
        front = np.asarray(source.get("front_mask"))
        if (front.dtype != np.bool_ or front.shape != source["mask"].shape
                or not front.any() or np.any(front & ~source["mask"])):
            raise ValueError("Protruding front_mask must be nonempty boolean source-contact evidence")
        anchor = np.asarray(geometry.get("anchor_pixel"), dtype=float)
        if (anchor.shape != (2,) or not np.isfinite(anchor).all()
                or not 0 <= anchor[0] < image.width or not 0 <= anchor[1] < image.height):
            raise ValueError("Protruding anchor_pixel must be a finite source-image pixel")
        _pose_array(geometry.get("source_frame"), "Compiled protruding source_frame")
    return {"manifest": manifest, "semantic": semantic, "program": program,
            "source_template": source, "geometry": geometry}


def observe_interaction_reference(skill, instruction, views, *, detector=None, client=None):
    """Detect and semantically select the reference chosen during extraction.

    Each view supplies an ``image`` and may supply separately observed
    ``selected_owner`` context for a part. The detector receives precisely the
    stored noun phrase. Object references select whole owners; part references
    select functional components, which are neither whole-owner masks nor exact
    demonstrated contact patches. No alternate query or score-based fallback is
    introduced when a view is occluded or ambiguous. Independent current owner
    evidence, if a downstream consumer needs it, remains a separate observation.
    """
    if not isinstance(skill, dict) or not {"manifest", "semantic", "source_template"} <= skill.keys():
        raise ValueError("Observe a loaded grounding skill")
    semantic = _semantics(skill["semantic"])
    source = skill["source_template"]
    if (not isinstance(source, dict)
            or any(source.get(key) != semantic[key] for key in _SEMANTIC_FIELDS)):
        raise ValueError("Source template semantic input differs from its offline artifact")
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A current instruction is required")
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty view-ID mapping")
    reference = semantic["interaction_reference"]
    scope = "owner" if reference["scope"] == "object" else "interaction_reference"
    images = {}
    for name, view in views.items():
        if (not isinstance(name, str) or not name.strip() or not isinstance(view, dict)
                or "image" not in view or "candidates" in view):
            raise ValueError("Each named view must supply an image, not caller-selected detections")
        images[name] = _image(view["image"])
        if "selected_owner" in view and scope != "interaction_reference":
            raise ValueError("selected_owner context is only meaningful for a part reference")
    if detector is None:
        from local_models.sam31 import detect

        detector = detect
    detections, selection_views = {}, {}
    labels = [reference["text"]]
    for name, image in images.items():
        raw = detector(image, labels.copy())
        if not isinstance(raw, list):
            raise ValueError("Detector must return a list of actual detections")
        rows, used = [], set()
        for index, row in enumerate(raw, 1):
            number = row.get("candidate_id", index) if isinstance(row, dict) else index
            if type(number) is not int or number < 1 or number in used:
                raise ValueError("Detector candidate IDs must be distinct positive integers")
            used.add(number)
            rows.append(_detection(row, image, labels, number))
        detections[name] = rows
        selection_views[name] = {"image": image, "candidates": rows}
        if "selected_owner" in views[name]:
            selection_views[name]["selected_owner"] = views[name]["selected_owner"]
    selection = select_grounding_candidates(
        source, instruction, selection_views, selection_scope=scope, client=client)
    return {"interaction_reference": deepcopy(reference),
            "owner_role": skill["manifest"]["owner_role"], "sam_queries": labels,
            "selection_scope": scope, "detections": detections, "selection": selection}
# END portable offline skill artifacts and data-driven runtime detection.
