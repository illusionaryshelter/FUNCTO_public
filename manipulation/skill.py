"""Self-contained PRD skill bundles and current semantic instance selection.

Version 2 records source observations, fixed functional frames and complete
relative motion. It is the sole production format; source provenance is never
opened when loading a bundle. File organization has no simulator dependency.
"""

# BEGIN version-2 portable representation production and consumption.
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .grounding import _detection, _image, _mask
from .grasp import validate_timeline
from .motion import _pose_array, _poses_array
from .observations import _intrinsics
from .selection import select_grounding_candidates


_SEMANTIC_FIELDS = {
    "object_description", "interaction_part_description", "interaction_reference",
    "description", "reason",
}
_FILES = {"metadata": "metadata.json", "source": "source.json",
          "observations": "source.npz", "image": "rgb.png",
          "representation": "representation.json", "motion": "motion.npz"}


def _semantics(value):
    """Read the actual semantic record; contact-free parts may be unnamed."""
    if not isinstance(value, dict) or set(value) != _SEMANTIC_FIELDS:
        raise ValueError("Skill semantics require the complete offline owner record")
    result = deepcopy(value)
    for field in _SEMANTIC_FIELDS - {"interaction_reference"}:
        if not isinstance(result[field], str) or (field != "interaction_part_description" and not result[field].strip()):
            raise ValueError(f"Skill semantic {field} must be text")
    reference = result["interaction_reference"]
    if (not isinstance(reference, dict) or set(reference) != {"text", "scope"}
            or reference["scope"] not in {"object", "part"}
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


def _json(path, value):
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(f"Unsupported skill metadata value: {type(item).__name__}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False, default=convert) + "\n", encoding="utf-8")


def _metadata(value):
    if (not isinstance(value, dict) or not {"skill_id", "owner_role", "semantic", "roles"} <= value.keys()
            or not isinstance(value["skill_id"], str) or not value["skill_id"].strip()
            or value["owner_role"] not in {"carrier", "reference"}
            or not isinstance(value["roles"], dict)):
        raise ValueError("Skill metadata needs skill_id, owner_role, semantic and physical/semantic roles")
    result = deepcopy(value)
    result["semantic"] = _semantics(value["semantic"])
    return result


def _source(value):
    """Check source identities and calibrated array dimensions, not task success."""
    # BEGIN region geometry is source evidence, explicitly declared by callers.
    required = {"image", "observation", "depth_m", "crop_xyxy", "timestamp_s", "frame_index",
                "owner_id", "T_world_body_at_image", "region_geometry"}
    if not isinstance(value, dict) or not required <= value.keys():
        raise ValueError("Source requires one calibrated image, point bank, time and fixed-body pose")
    result = dict(value)
    if value['region_geometry'] not in ('planar', 'surface'):
        raise ValueError('Source region_geometry must be planar or surface')
    # END no task-name inference or old-bundle geometry fallback.
    image = value["image"]
    if not isinstance(image, Image.Image):
        array = np.asarray(image)
        if array.dtype != np.uint8 or array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("Source image must be RGB uint8 or a PIL image")
        image = Image.fromarray(array)
    result["image"] = _image(image)
    shape = (image.height, image.width)
    depth = np.asarray(value["depth_m"])
    if depth.shape != shape or depth.dtype.kind not in "fiu":
        raise ValueError("Source metric optical depth must match RGB dimensions")
    result["depth_m"] = depth
    if (not isinstance(value["owner_id"], str) or not value["owner_id"].strip()
            or isinstance(value["frame_index"], (bool, np.bool_))
            or not isinstance(value["frame_index"], (int, np.integer)) or value["frame_index"] < 0
            or not np.isscalar(value["timestamp_s"]) or not np.isfinite(value["timestamp_s"])):
        raise ValueError("Source requires its original frame ID, seconds and rigid owner ID")
    result["T_world_body_at_image"] = _pose_array(value["T_world_body_at_image"], "source body pose")
    if "L_body_region" in value:
        result["L_body_region"] = _pose_array(value["L_body_region"], "source L_body_region")
    crop = np.asarray(value["crop_xyxy"])
    if (crop.shape != (4,) or crop.dtype.kind not in "iu"
            or not 0 <= crop[0] < crop[2] <= image.width
            or not 0 <= crop[1] < crop[3] <= image.height):
        raise ValueError("Source crop must use integer original-image xyxy bounds")
    result["crop_xyxy"] = crop
    observation = value["observation"]
    names = {"xyz", "pixels", "object_mask", "entity_member", "region_mask", "evidence_only", "K", "T_world_camera_cv"}
    if not isinstance(observation, dict) or set(observation) != names:
        raise ValueError("Source point bank requires the native observed-point fields")
    bank = {key: np.asarray(array) for key, array in observation.items()}
    xyz, pixels = bank["xyz"], bank["pixels"]
    if (xyz.ndim != 2 or xyz.shape[1:] != (3,) or not len(xyz) or not np.isfinite(xyz).all()
            or pixels.shape != (len(xyz), 2) or pixels.dtype.kind not in "iu"
            or np.any(pixels < 0) or np.any(pixels[:, 0] >= image.width)
            or np.any(pixels[:, 1] >= image.height)):
        raise ValueError("Source xyz and original pixel identities must align")
    bank["object_mask"] = _mask(bank["object_mask"], result["image"], "Source entity mask")
    for key in ("entity_member", "region_mask", "evidence_only"):
        if bank[key].dtype != np.bool_ or bank[key].shape != (len(xyz),):
            raise ValueError(f"Source {key} must be a point-aligned Boolean array")
    if not np.array_equal(bank["evidence_only"], bank["region_mask"] & ~bank["entity_member"]):
        raise ValueError("Source evidence-only identities disagree with region/entity membership")
    if not np.array_equal(bank["entity_member"], bank["object_mask"][pixels[:, 1], pixels[:, 0]]):
        raise ValueError("Source entity membership disagrees with the original image mask")
    bank["K"] = _intrinsics(bank["K"])
    bank["T_world_camera_cv"] = _pose_array(bank["T_world_camera_cv"], "source camera pose")
    if "region_image_mask" in value:
        mask = _mask(value["region_image_mask"], result["image"], "Source supplied region")
        if not np.array_equal(bank["region_mask"], mask[pixels[:, 1], pixels[:, 0]]):
            raise ValueError("Source region point identities disagree with the supplied image region")
        result["region_image_mask"] = mask
    elif bank["region_mask"].any():
        raise ValueError("A source region requires its original image mask")
    result["observation"] = bank
    result["missing"] = deepcopy(value.get("missing", {}))
    if "feature" in value:
        feature = np.asarray(value["feature"])
        if (feature.ndim != 4 or feature.shape[0] != 1 or min(feature.shape) < 1
                or feature.dtype != np.float32 or not np.isfinite(feature).all()
                or not value.get("feature_model")):
            raise ValueError("Source feature requires FP32 [1,C,H,W] data and its model record")
        result["feature"] = feature
        result["missing"].pop("feature", None)
    else:
        result["missing"].setdefault("feature", "Source spatial image feature has not been extracted")
    return result


def _representation(value):
    if not isinstance(value, dict) or value.get("kind") not in {"grasp", "manipulation"}:
        raise ValueError("Representation requires grasp/manipulation kind")
    result = dict(value)
    frames, times = validate_timeline(value["frame_indices"], value["timestamps"])
    result.update(frame_indices=frames, timestamps=times)
    for name in ("H_g", "H_m", "C_D", "T_world_carrier", "T_world_reference", "T_world_eef"):
        if name in value:
            poses = _poses_array(value[name], name)
            if len(poses) != len(times):
                raise ValueError(f"Representation {name} must align with original timestamps")
            result[name] = poses
    for name in ("L_G", "L_R"):
        if name in value:
            result[name] = _pose_array(value[name], name)
    if "progress" in value:
        progress = np.asarray(value["progress"], dtype=float)
        expected = np.zeros(1) if len(times) == 1 else (times-times[0])/(times[-1]-times[0])
        if not np.array_equal(progress, expected):
            raise ValueError("Representation progress must retain the original time parameterization")
        result["progress"] = progress
    return result


def _split_arrays(record):
    arrays = {key: value for key, value in record.items() if isinstance(value, np.ndarray)}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Skill arrays cannot require pickle")
    return {key: value for key, value in record.items() if key not in arrays}, arrays


def _source_motion_agreement(metadata, source, representation):
    """The image's frame and the frame used to form H must be the same one."""
    expected = ("carrier" if representation["kind"] == "grasp" and "H_g" in representation
                else "reference" if representation["kind"] == "manipulation" and "H_m" in representation else None)
    if expected is not None and metadata["owner_role"] != expected:
        raise ValueError("The source region owner role does not match the represented functional motion")
    row = np.flatnonzero(representation["frame_indices"] == source["frame_index"])
    if len(row) != 1 or not np.isclose(representation["timestamps"][row[0]], source["timestamp_s"], atol=1e-9, rtol=0):
        raise ValueError("Source image frame/time must belong to the original skill timeline")
    role = metadata["owner_role"]
    local = "L_G" if role == "carrier" else "L_R"
    track = "T_world_" + role
    if (local in representation and "L_body_region" in source
            and not np.allclose(source["L_body_region"], representation[local], atol=1e-7, rtol=0)):
        raise ValueError("Source image region frame differs from the representation's fixed frame")
    if (track in representation
            and not np.allclose(source["T_world_body_at_image"], representation[track][row[0]], atol=1e-7, rtol=0)):
        raise ValueError("Source image body pose differs from its original trajectory sample")


def save_grounding_skill(directory, *, metadata, source, representation):
    """Write one new ``functo.grounding_skill.2`` bundle, manifest last.

    ``metadata`` includes skill_id, owner_role, semantic and roles, and may carry
    independent task conditions, provenance and missing inputs. ``source`` is
    one actual image plus its native observation point bank, calibrated depth,
    crop, timestamp_s/frame_index, owner_id, T_world_body_at_image and explicit
    region_geometry (planar or surface). Optional
    L_body_region and region_image_mask describe the supplied source region;
    optional feature/feature_model record its spatial MARCO map. No source
    directory is opened. ``representation`` is build_representation's result.
    Arrays live in NPZ, source RGB in PNG; other records are strict JSON.
    """
    meta, src, motion = _metadata(metadata), _source(source), _representation(representation)
    _source_motion_agreement(meta, src, motion)
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    src["image"].save(root / _FILES["image"])
    source_meta, source_arrays = _split_arrays({key: value for key, value in src.items()
                                             if key not in {"image", "observation"}})
    source_arrays.update({"observation__" + key: value for key, value in src["observation"].items()})
    motion_meta, motion_arrays = _split_arrays(motion)
    _json(root / _FILES["metadata"], meta)
    _json(root / _FILES["source"], source_meta)
    _json(root / _FILES["representation"], motion_meta)
    np.savez_compressed(root / _FILES["observations"], **source_arrays)
    np.savez_compressed(root / _FILES["motion"], **motion_arrays)
    manifest = {"schema": "functo.grounding_skill.2", "files": dict(_FILES)}
    _json(root / "skill.json", manifest)
    return manifest


def _read_arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def load_grounding_skill(directory):
    """Read only this version-2 package's files; provenance paths stay opaque.

    Return ``manifest, metadata, source, representation``. The source's nested
    ``observation`` is directly usable by the dual-view adapter. Loading never
    invokes models, re-extracts regions or fabricates absent motion components.
    """
    root = Path(directory).resolve()
    manifest = _read_json(root / "skill.json")
    if manifest.get("schema") != "functo.grounding_skill.2":
        raise ValueError("Unsupported grounding skill schema; version 2 is required")
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(_FILES):
        raise ValueError("Version-2 skill files must name the complete bundle")
    paths = {}
    for name, relative in files.items():
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("Skill files must have bundle-relative paths")
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Skill file is missing or escapes its bundle")
        paths[name] = path
    meta = _metadata(_read_json(paths["metadata"]))
    source_meta, source_arrays = _read_json(paths["source"]), _read_arrays(paths["observations"])
    motion_meta, motion_arrays = _read_json(paths["representation"]), _read_arrays(paths["motion"])
    bank = {key.removeprefix("observation__"): value for key, value in source_arrays.items()
            if key.startswith("observation__")}
    observed = {key: value for key, value in source_arrays.items() if not key.startswith("observation__")}
    if (source_meta.keys() & (observed.keys() | {"image", "observation"})
            or motion_meta.keys() & motion_arrays.keys()):
        raise ValueError("Skill JSON and array records cannot override one another")
    with Image.open(paths["image"]) as opened:
        source = _source({**source_meta, **observed, "observation": bank, "image": opened.convert("RGB")})
    representation = _representation({**motion_meta, **motion_arrays})
    _source_motion_agreement(meta, source, representation)
    return {"manifest": manifest, "metadata": meta, "source": source, "representation": representation}
# END version-2 portable representation production and consumption.


# BEGIN semantic observation from the version-2 source evidence.
def observe_interaction_reference(skill, instruction, views, *, detector=None, client=None):
    """Detect and semantically select the reference chosen during extraction.

    Each view supplies an ``image`` and may supply separately observed
    ``selected_owner`` context for a part. The detector receives precisely the
    stored noun phrase. Object references select semantic entities, potentially
    containing several physical bodies; part references
    select functional components, which are neither whole-owner masks nor exact
    demonstrated contact patches. No alternate query or score-based fallback is
    introduced when a view is occluded or ambiguous. Independent current owner
    evidence, if a downstream consumer needs it, remains a separate observation.
    """
    if not isinstance(skill, dict) or not {"metadata", "source"} <= skill.keys():
        raise ValueError("Observe a loaded version-2 grounding skill")
    metadata = _metadata(skill["metadata"])
    semantic = metadata["semantic"]
    observed = skill["source"]
    reference = semantic["interaction_reference"]
    region = observed.get("region_image_mask")
    if region is None and reference["scope"] != "object":
        raise ValueError("A part reference requires its supplied source region evidence")
    # Semantic entity selection never asserts that a furniture mask is one
    # rigid body. Region ownership remains the explicit numeric owner record.
    scope = "entity" if reference["scope"] == "object" else "interaction_reference"
    source = {**semantic, "image": observed["image"],
              "owner_mask": observed["observation"]["object_mask"]}
    if region is not None:
        source["mask"] = region
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("A current instruction is required")
    if not isinstance(views, dict) or not views:
        raise ValueError("views must be a nonempty view-ID mapping")
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
            "owner_role": metadata["owner_role"], "sam_queries": labels,
            "selection_scope": scope, "detections": detections, "selection": selection}
# END semantic observation from the version-2 source evidence.
