# BEGIN current-owner grounding module documentation.
"""Ground source evidence using FUNCTO's existing local perception models.

Cached skills bind source semantic evidence to current owner RGB-D. Complete
functional frames use query recipes; on placement uses unordered support and
an explicitly bound carrier. The original raw-video path retains its explicit
tracking and rigid observations. Neither path supplies knowledge of hidden
joints, complete container interiors, actual Holding, or robot executability.
"""
# END current-owner grounding module documentation.

from pathlib import Path
import json

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from .grounding import infer_task, match_points, select_roles
from .motion import transfer_motion
from .observations import backproject, load_rgbd, rigid_registration, track_rgbd
from .pipeline import motion_kind, run_stage, write_json
from .support_grounding import infer_support
# BEGIN shared cached functional-evidence consumers.
from collections.abc import Mapping
from copy import deepcopy

from .goals import SupportSurface, _points
from .grounding import _image, _selected_interaction_evidence
from .motion import _errors, _pose_array, _tolerances
from .observations import Camera, _mask, _noncollinear
from .regions import (match_functional_region, sample_region_pixels,
                      source_function_geometry)
# END cached functional-evidence imports.


def _binding(demo_image, scene_image, demo_depth, scene_depth, demo_camera, scene_camera,
             demo_mask, scene_mask, extractor):
    # BEGIN share region sampling while retaining owner geometry for tracking.
    pixels, valid_demo = sample_region_pixels(demo_mask, demo_depth)
    # END shared calibrated sampling.
    valid_scene = np.asarray(scene_mask, bool) & np.isfinite(scene_depth) & (scene_depth > 0)
    def crop(image, mask):
        yy, xx = np.nonzero(mask)
        if not len(xx):
            raise ValueError("Selected entity has no valid depth")
        padding = max(2, int(max(xx.max()-xx.min()+1, yy.max()-yy.min()+1)*.05))
        x0, y0 = max(0, int(xx.min())-padding), max(0, int(yy.min())-padding)
        x1, y1 = min(image.width, int(xx.max())+padding+1), min(image.height, int(yy.max())+padding+1)
        return image.crop((x0, y0, x1, y1)), mask[y0:y1, x0:x1], np.array([x0, y0])
    first_image, first_mask, first_offset = crop(demo_image, valid_demo)
    second_image, second_mask, second_offset = crop(scene_image, valid_scene)
    targets, matching = match_points(first_image, second_image, first_mask, second_mask,
                                     pixels-first_offset, extractor=extractor)
    targets += second_offset
    matching["crop_offsets_xy"] = [first_offset.tolist(), second_offset.tolist()]
    scores = np.asarray(matching["scores"])
    selected, seen = [], set()
    for index in np.argsort(-scores):
        pixel_id = tuple(np.rint(targets[index]).astype(int))
        # Adjacent dense pixels can have almost equal scores on the same valid
        # correspondence. Keep that localization diagnostic; metric consensus
        # and distinct target pixels decide whether a rigid binding is usable.
        if scores[index] <= 0 or pixel_id in seen:
            continue
        seen.add(pixel_id)
        selected.append(int(index))
    if len(selected) < 6:
        raise ValueError("Descriptor binding is ambiguous or has too few distinct correspondences")
    first = backproject(demo_depth, demo_camera.K, pixels[selected], mask=valid_demo)
    second = backproject(scene_depth, scene_camera.K, targets[selected], mask=valid_scene)
    transform, geometry = rigid_registration(first, second, max_error=0.01)
    return transform, {"source_pixels": pixels[selected].tolist(),
                       "target_pixels": targets[selected].tolist(),
                       "matching": matching, "rigid_fit": geometry,
                       "source": "visible semantic correspondences + calibrated RGB-D"}


def _body_points(depth, camera, mask, initial_pose):
    cloud = backproject(depth, camera.K, mask=mask)
    if len(cloud) > 3000:
        cloud = cloud[np.linspace(0, len(cloud)-1, 3000).astype(int)]
    if len(cloud) < 6:
        raise ValueError("Selected carrier has insufficient observed geometry")
    return (cloud-initial_pose[:3, 3]) @ initial_pose[:3, :3]


# BEGIN cached skill grounding from current owners, independent of demo tracking.
def _current_owner(record, role):
    """Normalize one current RGB-D observation without estimating hidden state.

    ``binding_camera`` fixes the owner's coordinate gauge to this observation's
    camera at binding time. Its identity pose is a definition, not an estimate of
    a physical principal axis. Subsequent tracking must retain that fixed gauge.
    ``T_output_camera`` is mandatory even when explicitly identity: a gauge
    choice cannot replace camera-to-output calibration.

    Optional ``interaction_evidence`` is a separately selected visible part
    with mask, owner_id, description and source. It is retained for the coarse
    semantic request, not used to crop body geometry or require query coverage.
    """
    if not isinstance(record, Mapping):
        raise ValueError(f"Current {role} observation must be a mapping")
    required = {"image", "owner_mask", "depth", "K", "owner_id",
                "frame_convention", "T_output_camera"}
    if not required.issubset(record):
        raise ValueError(f"Current {role} observation is missing {sorted(required-set(record))}")
    if "mask" in record or "region_id" in record:
        raise ValueError("Current input is an owner observation, not a presegmented interaction region")
    if not isinstance(record["owner_id"], str) or not record["owner_id"].strip():
        raise ValueError("Current owner_id must be a nonempty string")
    image = _image(record["image"])
    shape = (image.height, image.width)
    owner_mask = _mask(record["owner_mask"], shape, "Current owner_mask")
    # BEGIN preserve selected observed parts separately from current owners.
    interaction = _selected_interaction_evidence(
        record.get("interaction_evidence"), image, owner_mask, owner_id=record["owner_id"])
    # END malformed ownership/provenance fails before any model invocation.
    depth = np.asarray(record["depth"], dtype=float)
    if depth.shape != shape:
        raise ValueError("Current RGB, owner mask and registered metric depth must share dimensions")
    # Validate calibration and positive-depth coverage before any model call.
    cloud = backproject(depth, record["K"], mask=owner_mask)
    if len(cloud) < 3:
        raise ValueError("Current owner has fewer than three valid metric depth samples")
    convention = record["frame_convention"]
    if convention == "binding_camera":
        if "T_camera_owner" in record:
            raise ValueError("binding_camera defines its identity gauge; use tracked_owner for a supplied pose")
        camera_owner = np.eye(4)
    elif convention == "tracked_owner":
        if "T_camera_owner" not in record:
            raise ValueError("tracked_owner requires its current T_camera_owner")
        camera_owner = _pose_array(record["T_camera_owner"], "Current T_camera_owner").copy()
    else:
        raise ValueError("frame_convention must be tracked_owner or binding_camera")
    output_camera = _pose_array(record["T_output_camera"], "Current T_output_camera").copy()
    normalized = {"image": image, "owner_mask": owner_mask, "depth": depth,
                  "K": np.asarray(record["K"], float), "owner_id": record["owner_id"],
                  "T_camera_owner": camera_owner}
    evidence = {"owner_id": record["owner_id"], "frame_convention": convention,
                "T_camera_owner": camera_owner, "T_output_camera": output_camera,
                "T_output_owner": output_camera @ camera_owner,
                "visible_depth_samples": len(cloud),
                "geometry_coverage": "current visible RGB-D surfaces; hidden geometry is unknown"}
    # BEGIN carry optional evidence through normalization rather than dropping it.
    if interaction is not None:
        normalized["interaction_evidence"] = interaction
        evidence["interaction_evidence"] = {
            **{key: interaction[key] for key in ("owner_id", "description", "source")},
            "mask_pixel_count": int(interaction["mask"].sum()),
            "pixels_outside_owner": int(np.count_nonzero(interaction["mask"] & ~owner_mask)),
            "scope": "selected visible part, separate from whole-owner geometry",
        }
    # END owner masks and their full visible point clouds remain unchanged.
    return normalized, evidence


def _set_observed_pose(scene, key, observed):
    """Reject contradictory caller state instead of silently rewriting its pose."""
    if key in scene:
        supplied = _pose_array(scene[key], f"scene.{key}")
        if not np.allclose(supplied, observed, atol=1e-6, rtol=0):
            raise ValueError(f"scene.{key} conflicts with the current owner observation/frame convention")
    scene[key] = observed.copy()


def _check_source_frame(geometry, event, role):
    """Check the recipe on source evidence before asking any current VLM/model."""
    reconstructed = _pose_array(geometry["source_frame"], "Reconstructed source function frame")
    compiled = _pose_array(event[role + "_frame"], "Compiled source function frame")
    values = [event["position_tolerance_m"], event["rotation_tolerance_rad"]]
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError("Source function frame tolerances must be numeric, not boolean")
    position_tolerance, rotation_tolerance = _tolerances(*values)
    position, rotation = _errors(reconstructed[None], compiled[None])
    if position[0] > position_tolerance or rotation[0] > rotation_tolerance:
        raise ValueError(f"Source {role} recipe frame disagrees with the compiled functional event")
    return {"position_error_m": float(position[0]), "rotation_error_rad": float(rotation[0]),
            "position_tolerance_m": position_tolerance, "rotation_tolerance_rad": rotation_tolerance}


def _interaction_samples(matched, minimum=1):
    """Keep contact queries separate from auxiliary direction witnesses."""
    points = _points(matched["target_points"], "Current functional query points")
    indices = np.asarray(matched["diagnostics"]["interaction_query_indices"])
    if (indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(points))):
        raise ValueError("Interaction query indices must select distinct current correspondences")
    selected = _points(points[indices], "Observed interaction samples", minimum=minimum)
    if minimum >= 3 and not _noncollinear(selected):
        raise ValueError("Observed interaction samples must be noncollinear")
    return selected


def _normal_matches(observed, expected, tolerance, *, signed):
    """Compare evidenced plane directions without inventing a physical side."""
    observed, expected = np.asarray(observed, float), np.asarray(expected, float)
    if (observed.shape != (3,) or expected.shape != (3,) or not np.isfinite(observed).all()
            or not np.isfinite(expected).all() or np.linalg.norm(observed) <= 1e-12
            or np.linalg.norm(expected) <= 1e-12):
        raise ValueError("Physical plane normals must be finite nonzero vectors")
    cosine = float(observed @ expected / (np.linalg.norm(observed) * np.linalg.norm(expected)))
    return np.arccos(np.clip(cosine if signed else abs(cosine), -1., 1.)) <= tolerance


# BEGIN unordered support binding and independently observed carrier contact.
def _support_direction(direction, source):
    """Validate physical side evidence, expressed in the declared output frame."""
    value = np.asarray(direction, float)
    if value.shape != (3,) or not np.isfinite(value).all() or not np.any(value):
        raise ValueError("support_direction must be a finite nonzero physical direction")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("support_direction_source must explain calibration and physical side")
    # Direction scale has no physical meaning. Scale first to avoid overflow
    # or underflow before taking the norm, as in the support estimator itself.
    scaled = value / np.max(np.abs(value))
    return scaled / np.linalg.norm(scaled)


def _contact_geometry(face):
    """Validate sourced owner-local contact samples without completing a face."""
    if not isinstance(face, Mapping) or not isinstance(face.get("source"), str) or not face["source"].strip():
        raise ValueError("Current on_geometry requires a physical geometry source")
    points = _points(face["carrier_region_points"], "Current carrier contact geometry", minimum=3)
    normal = np.asarray(face["carrier_outward_normal"], float)
    if not _normal_matches(normal, normal, np.pi, signed=True):
        raise ValueError("Invalid physical carrier normal")
    gap, angle = _tolerances(face["gap_tolerance"], face["normal_tolerance"])
    normal = normal / np.linalg.norm(normal)
    if not _noncollinear(points) or np.max(np.abs((points-points[0]) @ normal)) > gap:
        raise ValueError("Current carrier contact geometry must be a noncollinear planar face")
    return {**deepcopy(face), "carrier_region_points": points.copy(),
            "carrier_outward_normal": normal, "gap_tolerance": gap, "normal_tolerance": angle}


def bind_carrier_contact(template, current_owner, *, position_tolerance_m,
                         normal_tolerance_rad, client=None, extractor=None):
    """Observe a carrier's contact geometry for a subsequent placement binding.

    This is an explicit upstream operation when the relevant carrier surface
    is visible. It reuses functional correspondence and produces sparse current
    evidence, not a full hidden bottom or verified Holding. A known, already
    bound object model can instead supply these same numerical fields directly.
    Neither route infers current geometry by copying demonstration samples.
    """
    gap, angle = _tolerances(position_tolerance_m, normal_tolerance_rad)
    source = source_function_geometry(template)
    if source["geometry_kind"] != "plane" or template.get("normal_role") != "outward":
        raise ValueError("Carrier contact requires an evidenced outward plane normal")
    current, owner_evidence = _current_owner(current_owner, "carrier")
    matched = match_functional_region(template, current, client=client, extractor=extractor)
    if (matched["geometry_kind"] != source["geometry_kind"]
            or not np.allclose(matched["source_frame"], source["source_frame"], atol=1e-8, rtol=0)):
        raise ValueError("Current matching changed the prevalidated carrier source geometry")
    face = _contact_geometry({
        "carrier_region_points": _interaction_samples(matched, minimum=3),
        "carrier_outward_normal": matched["target_normal"],
        "gap_tolerance": gap, "normal_tolerance": angle,
        "source": "offline outward-normal semantics + current interaction-query correspondences",
        "normal_evidence": template["frame_recipe"]["source"],
        "coverage": "sparse visible contact correspondences; full face and hidden geometry are unknown",
    })
    camera = Camera(current["K"], current["image"].width, current["image"].height)
    body = _body_points(current["depth"], camera, current["owner_mask"], current["T_camera_owner"])
    return {"carrier_id": current["owner_id"], "source_owner_id": template["owner_id"],
            "carrier_mapping": _pose_array(matched["transform"], "Current carrier mapping"),
            "on_geometry": face, "initial_pose": owner_evidence["T_output_owner"],
            "body_points": body, "body_points_source": "current visible owner RGB-D; hidden geometry unknown",
            "source": "current carrier contact correspondence and calibrated owner coordinates",
            "region_binding_evidence": matched, "owner_observation_evidence": owner_evidence}


def _bind_support_scene(program, scene, source_templates, current_owners, *, client):
    """Bind an on goal from reference RGB-D, without reference point identities.

    Carrier pose, geometry, source-to-current mapping and (for EEF output)
    the declared attachment model are caller responsibilities. This function never assumes
    the carrier is identical to the demonstration, nor guesses identity for its
    mapping. Only the support role needs a current image here. Every event in
    this skill must refer to the same evidenced source support region, because
    its numerical consumer checks one finite current support surface.
    """
    if not isinstance(current_owners, Mapping) or set(current_owners) != {"reference"}:
        raise ValueError("An on binding requires only reference owner RGB-D and an already bound carrier")
    if "support" in scene:
        raise ValueError("Visual on binding estimates support; use the numerical consumer for supplied support")
    reference, owner_evidence = _current_owner(current_owners["reference"], "reference")
    for role in ("carrier", "reference"):
        if not isinstance(scene.get(role + "_id"), str) or not scene[role + "_id"]:
            raise ValueError("On binding requires explicit current carrier_id and reference_id")
    if scene["reference_id"] != reference["owner_id"] or scene["carrier_id"] == scene["reference_id"]:
        raise ValueError("Current on owners disagree with the selected reference instance")
    scene["initial_pose"] = _pose_array(scene["initial_pose"], "Current carrier pose")
    scene["body_points"] = _points(scene["body_points"], "Current carrier body points", minimum=3).copy()
    if not isinstance(scene.get("body_points_source"), str) or not scene["body_points_source"].strip():
        raise ValueError("Current carrier body_points_source is required")
    direction = _support_direction(scene.get("support_direction"), scene.get("support_direction_source"))
    events = program["demo"].get("functional_events")
    if not isinstance(events, list) or not events:
        raise ValueError("On binding requires explicit functional_events")
    names = [event.get("event_id") for event in events]
    if (any(not isinstance(name, str) or not name for name in names)
            or len(set(names)) != len(names) or not isinstance(source_templates, Mapping)
            or set(source_templates) != set(names)):
        raise ValueError("Source support semantics must cover every unique functional event")
    bindings = scene.get("functional_bindings", {})
    semantics, mapping = None, None
    for event in events:
        name = event["event_id"]
        pair = source_templates[name]
        if not isinstance(pair, Mapping) or set(pair) != {"reference"}:
            raise ValueError("Each on source event requires reference support semantics only")
        template = pair["reference"]
        required = ("owner_id", "region_id", "description", "source")
        if (not isinstance(template, Mapping) or any(
                not isinstance(template.get(key), str) or not template[key].strip() for key in required)
                or template.get("owner_id") != program["demo"].get("reference_id")
                or template.get("normal_role") != "support"):
            raise ValueError("Source on semantics require the selected owner, region and evidenced support side")
        if any(key in template for key in ("query_pixels", "query_descriptions", "frame_recipe")):
            raise ValueError("On reference semantics do not accept ordered queries or a frame recipe")
        selection = tuple(template[key] for key in ("owner_id", "region_id", "description"))
        if semantics is not None and selection != semantics:
            raise ValueError("One on skill must refer to one source support region")
        semantics = selection
        binding = bindings.get(name)
        if not isinstance(binding, Mapping) or not isinstance(binding.get("source"), str) or not binding["source"].strip():
            raise ValueError("On binding requires an evidenced current carrier_mapping")
        if any(binding.get(role + "_id") != scene[role + "_id"] for role in ("carrier", "reference")):
            raise ValueError("Current functional binding owners disagree with the scene")
        if "carrier_mapping" not in binding:
            raise ValueError("On binding requires an explicit current carrier_mapping")
        current_mapping = _pose_array(binding["carrier_mapping"], "Current carrier_mapping")
        if mapping is not None and not np.allclose(mapping, current_mapping, atol=1e-8, rtol=0):
            raise ValueError("One carrier requires a consistent source-to-current mapping")
        mapping = current_mapping
        binding["on_geometry"] = _contact_geometry(binding["on_geometry"])
    output_camera = owner_evidence["T_output_camera"]
    support_camera, support_evidence = infer_support(
        reference["image"], reference["depth"], reference["K"], reference["owner_mask"],
        support_direction=output_camera[:3, :3].T @ direction,
        instruction=semantics[2], client=client)
    support = SupportSurface(output_camera[:3, :3] @ support_camera.origin + output_camera[:3, 3],
                             output_camera[:3, :3] @ support_camera.basis,
                             output_camera[:3, :3] @ support_camera.normal,
                             support_camera.boundary.copy(), source=support_camera.source)
    scene["support"] = support.to_dict()
    scene["region_binding_evidence"] = {
        name: {"reference": {"source_owner_id": semantics[0], "source_region_id": semantics[1],
                              "target_owner_id": scene["reference_id"],
                              "source_semantics": source_templates[name]["reference"]["source"],
                              "support_direction_output": direction.copy(),
                              "support_direction_source": scene["support_direction_source"],
                              "binding_kind": "unordered_observed_support", **support_evidence},
               "on_geometry": deepcopy(bindings[name]["on_geometry"]), "support": support.to_dict(),
               "carrier_mapping_source": bindings[name]["source"],
               "orientation_source": "physical support normal; no observed reference tangent or target yaw"}
        for name in names}
    scene["owner_observation_evidence"] = {"reference": owner_evidence,
                                          "carrier": {"source": scene["body_points_source"],
                                                      "scope": "explicit upstream current carrier binding"}}
    return scene
# END unordered support binding and independently observed carrier contact.


def bind_current_scene(program, scene, source_templates, current_owners, *, client=None, extractor=None):
    """Bind a cached skill from source functional evidence and current owners.

    ``source_templates`` is ``{'grasp': template}`` for an independent grasp.
    Full-frame manipulation uses paired carrier/reference templates per event.
    On placement uses reference semantics only and an explicitly prebound
    carrier; see ``_bind_support_scene``. Full-frame templates use
    ``source_function_geometry``'s explicit query/recipe schema;
    auxiliary direction queries may extend beyond the source interaction mask
    within its owner. Current observations never require a part mask/region ID.

    ``current_owners`` uses role keys and calibrated owner RGB-D observations
    described by ``_current_owner``. Its owner poses and visible geometry are
    expressed in the named ``scene.reference_frame``. Caller EEF feedback,
    declared attachment, known geometry and mechanism evidence are retained; none
    are inferred from a source demonstration or from successful matching.

    All inputs are copied. Source recipes, owner IDs and coordinate mappings
    are checked before a current model call. Plane recipes cannot satisfy a
    full-frame manipulation or independent grasp. On support selection has no
    query recipe or reference object pose requirement.
    """
    program, result = deepcopy(program), deepcopy(scene)
    source_templates, current_owners = deepcopy(source_templates), deepcopy(current_owners)
    kind = program.get("skill_kind")
    if kind not in {"grasp", "manipulation"}:
        raise ValueError("Only grasp/manipulation programs have functional bindings")
    if not isinstance(result.get("reference_frame"), str) or not result["reference_frame"].strip():
        raise ValueError("Current scene must name its common reference_frame")
    # BEGIN bind only the physical relation needed by an on placement.
    if kind == "manipulation" and program["task"]["relation"] == "on":
        return _bind_support_scene(program, result, source_templates, current_owners, client=client)
    # END relation-specific unordered support dispatch.
    roles = ("carrier",) if kind == "grasp" else ("carrier", "reference")
    if not isinstance(current_owners, Mapping) or set(current_owners) != set(roles):
        raise ValueError(f"Current owner roles must be exactly {list(roles)}")
    if not isinstance(source_templates, Mapping):
        raise ValueError("source_templates must be an event/role mapping")

    current, owner_evidence = {}, {}
    for role in roles:
        current[role], owner_evidence[role] = _current_owner(current_owners[role], role)
        owner_id = current[role]["owner_id"]
        if role + "_id" in result and result[role + "_id"] != owner_id:
            raise ValueError("Current owner does not match the selected scene role")
        result[role + "_id"] = owner_id
    if len(roles) == 2 and result["carrier_id"] == result["reference_id"]:
        raise ValueError("Current carrier and reference must be distinct rigid owners")
    carrier_pose = owner_evidence["carrier"]["T_output_owner"]
    _set_observed_pose(result, "carrier_pose", carrier_pose)
    _set_observed_pose(result, "initial_pose", carrier_pose)
    if kind == "manipulation":
        _set_observed_pose(result, "reference_pose", owner_evidence["reference"]["T_output_owner"])
    carrier = current["carrier"]
    camera = Camera(carrier["K"], carrier["image"].width, carrier["image"].height)
    visible_body = _body_points(carrier["depth"], camera, carrier["owner_mask"], carrier["T_camera_owner"])
    result["observed_body_points"] = visible_body
    if "body_points" in result:
        # Retain a caller's richer current body model in this same fixed owner
        # frame; replacing it with a single-view cloud would discard geometry.
        result["body_points"] = _points(result["body_points"], "Current owner-local body_points").copy()
        result.setdefault("body_points_source", "caller-supplied geometry in the selected current owner frame")
    else:
        result["body_points"] = visible_body.copy()
        result["body_points_source"] = "current registered RGB-D visible surface in the fixed owner frame"
    owner_evidence["carrier"]["body_points_source"] = result["body_points_source"]
    owner_evidence["carrier"]["body_points_frame"] = result["carrier_id"]

    demo, jobs = program["demo"], []
    if kind == "grasp":
        if set(source_templates) != {"grasp"}:
            raise ValueError("Independent grasp requires exactly one grasp source template")
        jobs.append(("grasp", "carrier", source_templates["grasp"], None))
    else:
        events = demo.get("functional_events")
        if not isinstance(events, list) or not events:
            raise ValueError("Manipulation requires explicit functional_events")
        names = [event.get("event_id") for event in events]
        if (any(not isinstance(name, str) or not name for name in names)
                or len(set(names)) != len(names) or set(source_templates) != set(names)):
            raise ValueError("Source templates must cover each unique compiled functional event")
        for event in events:
            paired = source_templates[event["event_id"]]
            if not isinstance(paired, Mapping) or set(paired) != set(roles):
                raise ValueError("Each functional event requires carrier/reference source templates")
            jobs.extend((event["event_id"], role, paired[role], event) for role in roles)

    # Complete this entire pass first: a bad later event must not spend model
    # calls matching an earlier event whose source data already look valid.
    source_geometry, source_checks = {}, {}
    for name, role, template, event in jobs:
        expected = demo.get(role + "_id")
        if not isinstance(expected, str) or not expected:
            raise ValueError("Cached functional grounding requires explicit demonstration owner role IDs")
        if not isinstance(template, Mapping) or template.get("owner_id") != expected:
            raise ValueError("Demonstration region owner does not match the compiled source role")
        geometry = source_function_geometry(template)
        geometry_kind = geometry["geometry_kind"]
        if geometry_kind not in {"frame", "plane"}:
            raise ValueError("Functional recipe kind must be frame or plane")
        if geometry_kind == "plane":
            raise ValueError("A plane supplies insufficient direction evidence for this full-frame skill")
        source_geometry[(name, role)] = geometry
        if event is not None:
            source_checks[(name, role)] = _check_source_frame(geometry, event, role)

    matches, evidence = {}, {}
    for name, role, template, event in jobs:
        matched = match_functional_region(template, current[role], client=client, extractor=extractor)
        checked = source_geometry[(name, role)]
        if (matched["geometry_kind"] != checked["geometry_kind"]
                or not np.allclose(matched["source_frame"], checked["source_frame"], atol=1e-8, rtol=0)):
            raise ValueError("Current matching changed the prevalidated source functional geometry")
        matched["source_owner_id"], matched["target_owner_id"] = template["owner_id"], current[role]["owner_id"]
        matched["source_region_id"] = template["region_id"]
        if event is not None:
            matched["source_frame_check"] = source_checks[(name, role)]
        matches[(name, role)] = matched
        if kind == "grasp":
            evidence[name] = matched
        else:
            evidence.setdefault(name, {})[role] = matched
    if kind == "grasp":
        result["region_transform"] = _pose_array(matches[("grasp", "carrier")]["transform"],
                                                "Current grasp functional mapping").copy()
    else:
        bindings = deepcopy(result.get("functional_bindings", {}))
        for event in demo["functional_events"]:
            name = event["event_id"]
            binding = bindings.setdefault(name, {})
            for role in roles:
                binding[role + "_frame"] = _pose_array(matches[(name, role)]["target_frame"],
                                                        "Current functional frame").copy()
            kinds = {role: matches[(name, role)]["geometry_kind"] for role in roles}
            binding.update(carrier_id=result["carrier_id"], reference_id=result["reference_id"],
                           geometry_kind=kinds,
                           orientation_source={role: ("observed functional axis and plane" if kinds[role] == "frame"
                                                      else "observed plane normal; tangent is a numerical preference")
                                               for role in roles},
                           source="source query recipes reconstructed from current owner RGB-D correspondences")
        result["functional_bindings"] = bindings
    result["region_binding_evidence"], result["owner_observation_evidence"] = evidence, owner_evidence
    return result


def ground_current_program(program, scene, source_templates, current_owners, *, output="eef",
                           client=None, extractor=None):
    """Bind current observations then call exactly one existing numerical core.

    ``output='eef'`` requests a complete grasp/manipulation EEF reference and
    retains the explicitly declared attachment model. ``output='object'`` deliberately
    requests manipulation body geometry only. Missing feedback never triggers
    an automatic output downgrade. Both return geometric candidates, not robot
    execution certificates, with visual binding evidence for inspection.
    """
    from .program import ground_object_program, ground_program
    if output not in {"eef", "object"}:
        raise ValueError("output must be eef or object")
    if output == "object" and program.get("skill_kind") != "manipulation":
        raise ValueError("Object-only output is only defined for manipulation")
    bound = bind_current_scene(program, scene, source_templates, current_owners,
                               client=client, extractor=extractor)
    result = ground_program(program, bound) if output == "eef" else ground_object_program(program, bound)
    result["region_binding_evidence"] = bound["region_binding_evidence"]
    result["owner_observation_evidence"] = bound["owner_observation_evidence"]
    return result
# END cached skill grounding from current owner observations.


# BEGIN grasp candidates screened against actual tool and observed owner geometry.
def ground_grasp_candidates(program, candidate_scenes, current_views, tool, *,
                            current_joint_positions_m, contact_tolerance_m,
                            penetration_tolerance_m):
    """Generate complete functional grasp references and assess their geometry.

    ``candidate_scenes`` contains already perceived functional bindings for one
    selected owner, synchronized robot EEF pose, and common output frame. This
    entry deliberately preserves every generated reference, even when observed
    target geometry contradicts its grasp. A functional frame is not promoted
    to an exact whole-object rigid registration.

    ``tool`` is the caller's known robot geometry; current finger coordinates
    are explicit proprioception, never borrowed from source contact samples.
    ``current_views`` supplies actual owner RGB-D and optional perceived part
    masks. The checker evaluates visible owner/tool compatibility, not full-arm
    planning, complete scene clearance, dynamic closure, or actual Holding.
    These are alternative actions under one established owner/goal binding,
    not mutually exclusive scene hypotheses. A unique supported action can be
    selected while other actions remain unresolved; their uncertainty does not
    invalidate its own geometry. Multiple supported actions remain available
    without a path-cost ranking. Ambiguous identity/semantics belongs upstream.
    """
    from .program import ground_program
    from .grasp_validation import screen_grasp_candidates

    if program.get("skill_kind") != "grasp":
        raise ValueError("Tool contact screening consumes independent grasp programs")
    if not isinstance(candidate_scenes, Mapping) or not candidate_scenes:
        raise ValueError("A nonempty mapping of bound grasp candidates is required")
    scenes = list(candidate_scenes.values())
    frames = {scene.get("reference_frame") for scene in scenes}
    owners = {scene.get("carrier_id") for scene in scenes}
    if len(frames) != 1 or len(owners) != 1 or None in frames or None in owners:
        raise ValueError("Grasp candidates must share one output frame and selected owner")
    first_eef = _pose_array(scenes[0]["eef_pose"], "Current robot EEF feedback")
    for scene in scenes:
        if not np.allclose(_pose_array(scene["eef_pose"], "Current EEF"), first_eef,
                           atol=1e-9, rtol=0):
            raise ValueError("Grasp candidates must use synchronized current robot feedback")
        if scene.get("tool_id") != tool.tool_id or scene.get("tool_frame") != tool.tool_frame:
            raise ValueError("Tool geometry and the bound candidate use different tool conventions")
    if (not isinstance(current_views, Mapping) or not current_views
            or any(view.get("owner_id") not in owners for view in current_views.values())):
        raise ValueError("Current grasp observations must belong to the selected owner")
    trajectories = {name: ground_program(program, deepcopy(scene))
                    for name, scene in candidate_scenes.items()}
    validation = screen_grasp_candidates(
        trajectories, current_views, tool, reference_frame=next(iter(frames)),
        current_eef_pose=first_eef, current_joint_positions_m=current_joint_positions_m,
        contact_tolerance_m=contact_tolerance_m,
        penetration_tolerance_m=penetration_tolerance_m)
    for name, trajectory in trajectories.items():
        trajectory["grasp_geometry_validation"] = deepcopy(validation["candidates"][name])
    selected = validation["selected_candidate_id"]
    return {"status": validation["status"], "selected_candidate_id": selected,
            "selected_trajectory": trajectories[selected] if selected is not None else None,
            "trajectories": trajectories, "grasp_geometry_validation": validation,
            "robot_execution_verified": False}
# END grasp screening keeps functional references separate from execution facts.


# BEGIN screened rigid-owner candidate generation through the numerical core.
def _registered_owner_transform(program, scene, role):
    """Derive the tested mapping from the exact bindings consumed by the core.

    Do not accept a second caller-supplied validation transform: checking a
    good transform while generating motion from a different binding would
    defeat the gate. Full functional frames yield one rigid owner mapping;
    inconsistent event mappings or support/joint-only constraints do not.
    Those representations continue using their own grounding entry points.
    """
    if program["skill_kind"] == "grasp":
        if role != "carrier":
            raise ValueError("Independent grasp registers its carrier")
        return (_pose_array(scene["carrier_pose"], "carrier_pose")
                @ _pose_array(scene["region_transform"], "region_transform"))
    if program["skill_kind"] != "manipulation" or role not in {"carrier", "reference"}:
        raise ValueError("A manipulation registers its carrier or reference owner")
    events = program["demo"].get("functional_events")
    if not events or program["task"]["relation"] == "on":
        raise ValueError("Rigid registration screening requires full functional-frame bindings")
    pose_key = "initial_pose" if role == "carrier" else "reference_pose"
    owner_pose = _pose_array(scene[pose_key], pose_key)
    transforms = []
    for event in events:
        if event.get("constraint_kind", "functional_frame") != "functional_frame":
            raise ValueError("A joint-coordinate goal does not define a rigid owner registration")
        binding = scene["functional_bindings"][event["event_id"]]
        if binding.get(role + "_id") != scene[role + "_id"]:
            raise ValueError("Functional binding and registered owner must agree")
        source = _pose_array(event[role + "_frame"], "source functional frame")
        current = _pose_array(binding[role + "_frame"], "current functional frame")
        transforms.append(owner_pose @ current @ np.linalg.inv(source))
    if any(not np.allclose(value, transforms[0], atol=1e-7, rtol=0) for value in transforms[1:]):
        raise ValueError("Functional events do not share one rigid owner registration")
    return transforms[0]


def ground_registered_candidates(program, candidate_scenes, source, current_views, *,
                                 role, same_rigid_geometry, geometry_source, max_error=.01):
    """Screen bound rigid-owner candidates, then generate geometric references.

    candidate_scenes maps arbitrary unique candidate IDs to complete scenes for
    ground_program. source and current_views follow screen_registrations' owner
    RGB-D contract. All scenes bind the same current owner and output frame;
    all observations describe that same captured state. There is no dataset,
    camera name, candidate count, skill name, asset or point-order convention.

    Only supported/unresolved candidates reach the existing numerical consumer.
    Rejected bindings have no trajectory here. Return selected_trajectory only
    for a unique supported candidate with no unresolved competitors. A geometry
    applicability failure returns not_applicable and no unvalidated trajectory.
    Numeric consumer errors still propagate; there is no unchecked fallback.

    # BEGIN an explicit physical registration is stronger than a function frame.
    Use this entry only for bindings that actually claim physical rigid-owner
    registration. A full functional frame or unchanged instance shape alone
    does not establish that claim. Functional grasp candidates instead use
    ground_grasp_candidates to inspect tool/owner geometry. On-support and
    joint-coordinate goals likewise do not imply a complete rigid registration.
    Screening establishes neither actual Holding nor robot executability.
    # END check applicability from geometric meaning, not the skill label.
    """
    from .program import ground_program
    from .registration_validation import screen_registrations

    if not isinstance(candidate_scenes, Mapping) or not candidate_scenes:
        raise ValueError("A nonempty candidate-ID to scene mapping is required")
    if role not in {"carrier", "reference"}:
        raise ValueError("role must be carrier or reference")
    if (not isinstance(source.get("owner_id"), str) or not source["owner_id"]
            or source["owner_id"] != program["demo"].get(role + "_id")):
        raise ValueError("Source owner must match the compiled demonstration role")
    frames = {scene.get("reference_frame") for scene in candidate_scenes.values()}
    owners = {scene.get(role + "_id") for scene in candidate_scenes.values()}
    if len(frames) != 1 or len(owners) != 1 or None in frames or None in owners:
        raise ValueError("Candidates must share one output frame and registered owner")
    if (not isinstance(current_views, Mapping) or not current_views
            or any(view.get("owner_id") not in owners for view in current_views.values())):
        raise ValueError("Current views must belong to the candidates' registered owner")
    transforms = {name: _registered_owner_transform(program, scene, role)
                  for name, scene in candidate_scenes.items()}
    validation = screen_registrations(
        source, current_views, transforms, reference_frame=next(iter(frames)),
        same_rigid_geometry=same_rigid_geometry, geometry_source=geometry_source,
        max_error=max_error)
    trajectories = {}
    # Keep rejected/unresolved status in the returned evidence. Numerical
    # trajectory construction never upgrades uncertain registration to truth.
    for name in validation["supported_candidate_ids"] + validation["unresolved_candidate_ids"]:
        trajectory = ground_program(program, deepcopy(candidate_scenes[name]))
        evidence = validation["candidates"][name]
        trajectory["registration_validation"] = {
            "candidate_id": name, "status": evidence["status"], "role": role,
            "source_owner_id": source["owner_id"], "target_owner_id": next(iter(owners)),
            "reference_frame": validation["reference_frame"],
            "T_output_source_owner": evidence["T_output_source_owner"].copy(),
            "geometry_source": geometry_source, "same_rigid_geometry": bool(same_rigid_geometry),
            "max_error_m": validation["max_error_m"], "scope": validation["scope"],
            "views": {view: deepcopy(value["summary"]) for view, value in evidence["views"].items()},
        }
        trajectories[name] = trajectory
    selected = validation["selected_candidate_id"]
    return {"status": validation["status"], "selected_candidate_id": selected,
            "selected_trajectory": trajectories[selected] if selected is not None else None,
            "trajectories": trajectories, "registration_validation": validation,
            "robot_execution_verified": False}
# END screened rigid-owner candidate generation through the numerical core.


def ground_rgbd(demo_dir, scene_dir, instruction, *, client=None, detector=None,
                extractor=None, tracker=None, output_dir=None):
    """Produce a numerical stage from RGB-D and task semantics, without new labels.

    Optional ``scene_dir/geometry.json`` may supply known interior/support
    geometry from another system component. It is never synthesized from an
    outer bounding box. Raw depth and intrinsics are needed for metric output;
    no additional task-specific training or offline motion dataset is used.
    """
    # BEGIN deprecated combined demonstration/classification entry.
    # This raw-video path reruns infer_task and dispatches on its task schema.
    # Current skill grounding loads the preserved representation and binds its
    # declared entities/regions to observations. Shared perception APIs above
    # remain independently used; this deprecation is specific to ground_rgbd.
    # END deprecated combined demonstration/classification entry.
    demo_dir, scene_dir = Path(demo_dir), Path(scene_dir)
    demo_image, demo_depth, demo_camera = load_rgbd(demo_dir)
    scene_image, scene_depth, scene_camera = load_rgbd(scene_dir)
    indices = sorted({int(path.stem) for path in (demo_dir / "rgb").iterdir()
                      if path.stem.isdigit() and path.suffix.lower() in {".jpg", ".png", ".jpeg"}})
    if len(indices) < 2:
        raise ValueError("The demonstrated stage requires at least two RGB-D frames")
    final_image, _, _ = load_rgbd(demo_dir, indices[-1])
    task = infer_task(instruction, demo_image, scene_image,
                      demo_final_image=final_image, client=client)
    # BEGIN explicit physical support side, independent of mapped goal points.
    geometry_path = scene_dir / "geometry.json"
    geometry = json.loads(geometry_path.read_text()) if geometry_path.is_file() else {}
    if geometry and geometry.get("reference_frame") != "scene_camera":
        raise ValueError("Optional scene geometry must declare reference_frame=scene_camera")
    support_direction_camera = None
    if task["relation"] == "on" and "support" not in geometry:
        config_path = scene_dir / "config.json"
        config = json.loads(config_path.read_text()) if config_path.is_file() else {}
        support_direction_camera = _support_direction(
            config.get("support_direction"), config.get("support_direction_source"))
    # END calibrated support-side validation before tracking or instance models.
    # BEGIN one observation path for tools, transported objects and moving parts.
    # Compiled functional frames describe fixed local geometry, not subsequent
    # motion. Track its carrier/reference exactly as for the other task families.
    if task["relation"] == "tool_use" and task["motion"] not in {"auto", "free"}:
        raise ValueError("The tool-use optimizer cannot jointly enforce function events and "
                         "a constrained motion family; refusing to discard the motion constraint")
    demo_roles = select_roles(demo_image, task, "demo", detector=detector, client=client)
    masks = {}

    def role_masks(domain, roles):
        carrier = roles["carrier"]["mask"]
        # A parent cabinet mask may contain the moving drawer/door. Only its
        # remaining visible fixed part is allowed to serve as motion reference.
        reference = roles["reference"]["mask"] & ~carrier
        if reference.sum() < 12:
            raise ValueError(f"{domain} has insufficient visible fixed reference outside the moving part")
        masks[domain] = {"carrier": carrier, "reference": reference}
        if output_dir is not None:
            destination = Path(output_dir) / domain
            destination.mkdir(parents=True, exist_ok=True)
            (demo_image if domain == "demo" else scene_image).save(destination / "initial.png")
            for role, mask in masks[domain].items():
                Image.fromarray(mask.astype(np.uint8)*255).save(destination / f"{role}_mask.png")
    tracker = track_rgbd if tracker is None else tracker
    role_masks("demo", demo_roles)
    track = tracker(demo_dir, masks["demo"]["carrier"], reference_mask=masks["demo"]["reference"])
    demo_poses = np.asarray(track["poses"], float)
    if len(demo_poses) != len(indices) or not np.array_equal(track["frame_indices"], indices):
        raise ValueError("Tracked poses do not cover the selected demonstration frames")
    if task["relation"] == "tool_use":
        # The compiled tool scene supplies its functional region bindings.
        # Whole-owner matching here cannot replace those local correspondences.
        return {"tool_use": True, "task": task, "instruction": instruction, "tracking": track}
    scene_roles = select_roles(scene_image, task, "scene", detector=detector, client=client)
    role_masks("scene", scene_roles)
    # END common observation path; destination grounding remains task-specific.
    actual_extractor = extractor
    if actual_extractor is None:
        # BEGIN shared default MARCO pair features.
        from local_models.marco import extract_pair
        # END shared default feature provider.
        actual_extractor = extract_pair

    bindings, evidence = {}, {}
    for role in ("reference", "carrier"):
        bindings[role], evidence[role] = _binding(
            demo_image, scene_image, demo_depth, scene_depth, demo_camera, scene_camera,
            masks["demo"][role], masks["scene"][role], actual_extractor)
    reference = bindings["reference"]
    # The body frame is transferred from demo-visible body features. Scene
    # geometry is then expressed in that fixed frame, retaining shape changes
    # instead of pretending that the whole new object equals the demonstration.
    initial_pose = bindings["carrier"] @ demo_poses[0]
    body = _body_points(scene_depth, scene_camera, masks["scene"]["carrier"], initial_pose)
    goal = reference @ demo_poses[-1]
    scene = {"initial_pose": initial_pose, "reference_transform": reference,
             "body_points": body, "preferred_goal_pose": goal,
             "goal_source": "demonstrated endpoint bound through visible reference correspondences",
             "geometry_coverage": "visible RGB-D surface samples; hidden geometry is unknown"}
    for key in ("support", "interior", "entry_region", "entry_start_index", "margin"):
        if key in geometry:
            scene[key] = geometry[key]
    if task["relation"] == "on" and "support" not in scene:
        # BEGIN support-side direction from actual calibration/physical evidence.
        # A reference object's largest plane may be a side panel. Select an
        # observed plane by the task and retain only its observed support area.
        surface, surface_evidence = infer_support(
            scene_image, scene_depth, scene_camera.K, masks["scene"]["reference"],
            support_direction=support_direction_camera,
            instruction=instruction + "\n" + task.get("goal_description", ""), client=client)
        scene["support"] = surface.to_dict()
        evidence["support"] = surface_evidence
        evidence["support"]["support_direction_source"] = config["support_direction_source"]
        # END calibrated support side; no source goal or camera-facing default.
    # BEGIN demonstrated containment target without invented hidden geometry.
    # The reference-bound endpoint already specifies the demonstrated inside
    # motion. Missing cavity geometry limits containment verification, rather
    # than requiring another offline input before that motion can transfer.
    # run_stage preserves this distinction and still validates supplied regions.
    # END demonstrated containment target without invented hidden geometry.
    if task["relation"] == "articulation" and task["motion"] == "free":
        raise ValueError("Articulation cannot be silently represented by unconstrained free motion")
    # Noisy image bindings produce an estimated pose, not an exact six-DoF
    # semantic command. For constrained motion, make any within-tolerance model
    # projection explicit; incompatible binding remains an error.
    # BEGIN the pose-only perception fit declares geometric sampling explicitly.
    # No physical timestamps are available or fabricated in this shape query.
    fitted = transfer_motion(demo_poses, initial_pose, reference_transform=reference,
                             kind=motion_kind(task), progress=np.linspace(0., 1., len(demo_poses)))
    # END fitting a geometric mechanism does not bind execution timing.
    if fitted["model"].kind != "free":
        coordinates = fitted["model"].coordinates_of(goal[None])
        projected = fitted["model"].poses(coordinates)[0]
        evidence["goal_projection"] = {
            "position_correction_m": float(np.linalg.norm(goal[:3, 3]-projected[:3, 3])),
            "rotation_correction_rad": float(Rotation.from_matrix(goal[:3, :3] @ projected[:3, :3].T).magnitude()),
            "meaning": "projection of estimated correspondence geometry within motion-fit tolerances",
        }
        scene["preferred_goal_pose"] = projected
    evidence["tracking"] = {key: value for key, value in track.items() if key not in {"poses", "points"}}
    return {"task": task, "instruction": instruction, "demo_poses": demo_poses,
            "frame_indices": track.get("frame_indices", indices),
            "carrier_id": f"carrier:{scene_roles['carrier']['candidate_id']}",
            "reference_id": f"reference:{scene_roles['reference']['candidate_id']}",
            "reference_frame": "scene_camera", "scene": scene, "evidence": evidence}


def run_rgbd(demo_dir, scene_dir, instruction, output_dir, **dependencies):
    """Execute semantic grounding, observations and trajectory generation once."""
    # BEGIN deprecated raw-RGB-D orchestration.
    # This entry owns the old VLM relation/motion/interaction dispatch. It is
    # not called by the current compiled-skill grounding and planning path.
    # END deprecated raw-RGB-D orchestration.
    destination = Path(output_dir)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "instruction": instruction,
              "demo": str(Path(demo_dir).resolve()), "scene": str(Path(scene_dir).resolve())}
    write_json(destination / "run.json", report)
    try:
        observation = ground_rgbd(demo_dir, scene_dir, instruction, output_dir=destination, **dependencies)
        # BEGIN save the actual motion source before either downstream consumer.
        write_json(destination / "observations.json", observation)
        if observation.get("tool_use"):
            from .tool_use import run_tool_use
            result = run_tool_use(demo_dir, scene_dir, observation["task"], instruction,
                                  destination, tracking=observation["tracking"])
        else:
            result = run_stage(observation)
            write_json(destination / "trajectory.json", result)
        # END common motion provenance and task-specific trajectory generation.
        report["status"] = "passed"
        report["output_type"] = result["output_type"]
    except BaseException as error:
        # Preserve failure (including interruption) before propagating it.
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "run.json", report)
    return result
