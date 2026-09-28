# BEGIN current-owner grounding module documentation.
"""Ground source evidence using FUNCTO's existing local perception models.

Cached skills reconstruct functional geometry from source query recipes and
current owner RGB-D. The original raw-video path also retains its explicit
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
from .grounding import _image
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


def _bind_on_geometry(event, binding, paired_sources, paired_matches, current, owner_evidence, scene, client):
    """Recover sparse contact/support evidence; never require a target part mask.

    A recipe normal only gains a physical side through the offline declaration
    ``normal_role``. Only matched queries originally inside the source contact
    mask can become contact samples. These are sparse visible evidence, not a
    reconstructed full face or a certificate about an occluded bottom surface.
    A supplied richer current geometry remains usable and is checked for plane
    and normal consistency with these independent correspondences. Discrete
    carrier face samples do not establish a polygon boundary, so no such
    boundary is invented. The observed support has its own finite boundary.
    """
    from .pipeline import _surface
    gap, angle = _tolerances(event["position_tolerance_m"], event["rotation_tolerance_rad"])
    carrier_match, reference_match = paired_matches["carrier"], paired_matches["reference"]
    carrier_source, reference_source = paired_sources["carrier"], paired_sources["reference"]
    if "on_geometry" not in binding:
        if carrier_match["geometry_kind"] != "plane" or carrier_source.get("normal_role") != "outward":
            raise ValueError("Source on template needs an evidenced outward plane normal to infer contact geometry")
        face_points = _interaction_samples(carrier_match, minimum=3)
        normal = np.asarray(carrier_match["target_normal"], float)
        if np.any(np.abs((face_points-carrier_match["target_origin"]) @ normal) > gap):
            raise ValueError("Observed carrier contact samples are not on the declared functional plane")
        binding["on_geometry"] = {
            "carrier_region_points": face_points.copy(), "carrier_outward_normal": normal.copy(),
            "gap_tolerance": gap, "normal_tolerance": angle,
            "source": "offline outward-normal semantics + current interaction-query correspondences",
            "normal_evidence": carrier_source["frame_recipe"]["source"],
            "coverage": "sparse visible contact correspondences; full face and hidden geometry are unknown",
        }
    face = binding["on_geometry"]
    if not isinstance(face.get("source"), str) or not face["source"].strip():
        raise ValueError("Current on_geometry requires an independent physical geometry source")
    face_points = _points(face["carrier_region_points"], "Current carrier contact geometry")
    face_normal = np.asarray(face["carrier_outward_normal"], float)
    face_gap, face_angle = _tolerances(face["gap_tolerance"], face["normal_tolerance"])
    if not _normal_matches(face_normal, face_normal, np.pi, signed=True):
        raise ValueError("Invalid physical carrier normal")
    face_normal = face_normal / np.linalg.norm(face_normal)
    observed_face = _interaction_samples(carrier_match)
    if np.any(np.abs((observed_face-face_points[0]) @ face_normal) > face_gap):
        raise ValueError("Current contact model disagrees with the matched interaction surface")
    if carrier_match["geometry_kind"] == "plane" and not _normal_matches(
            carrier_match["target_normal"], face_normal, face_angle,
            signed=carrier_source.get("normal_role") == "outward"):
        raise ValueError("Matched carrier plane and physical contact normal disagree")

    reference_observation = current["reference"]
    if "support" not in scene:
        if reference_match["geometry_kind"] != "plane" or reference_source.get("normal_role") != "support":
            raise ValueError("Source on template needs an evidenced support plane normal to infer the current support")
        # Three independent interaction queries support the plane hypothesis.
        # Direction-only witnesses may lie elsewhere on the same owner and are
        # never substituted when the actual interaction surface is unobserved.
        observed_reference = _interaction_samples(reference_match, minimum=3)
        if np.any(np.abs((observed_reference-reference_match["target_origin"])
                         @ reference_match["target_normal"]) > gap):
            raise ValueError("Observed reference samples are not on the declared support plane")
        camera_owner = reference_observation["T_camera_owner"]
        normal_camera = camera_owner[:3, :3] @ reference_match["target_normal"]
        origin_camera = camera_owner[:3, :3] @ reference_match["target_origin"] + camera_owner[:3, 3]
        support_camera, support_evidence = infer_support(
            reference_observation["image"], reference_observation["depth"], reference_observation["K"],
            reference_observation["owner_mask"], origin_camera + .05 * normal_camera,
            instruction=reference_source["description"], client=client)
        output_camera = owner_evidence["reference"]["T_output_camera"]
        support = SupportSurface(
            output_camera[:3, :3] @ support_camera.origin + output_camera[:3, 3],
            output_camera[:3, :3] @ support_camera.basis,
            output_camera[:3, :3] @ support_camera.normal,
            support_camera.boundary.copy(), source=support_camera.source)
        scene["support"] = support.to_dict()
        owner_evidence["reference"]["support"] = support_evidence
    else:
        support = _surface(scene["support"])
    output_reference = owner_evidence["reference"]["T_output_owner"]
    observed_reference = _interaction_samples(reference_match)
    reference_points = observed_reference @ output_reference[:3, :3].T + output_reference[:3, 3]
    offsets = reference_points-support.origin
    uv = offsets @ support.basis
    boundary = support._hull.equations
    if (np.any(np.abs(offsets @ support.normal) > gap)
            or np.any(uv @ boundary[:, :2].T + boundary[:, 2] > gap)):
        raise ValueError("Selected support disagrees with matched interaction points or its finite visible boundary")
    if reference_match["geometry_kind"] == "plane" and not _normal_matches(
            output_reference[:3, :3] @ reference_match["target_normal"], support.normal, angle,
            signed=reference_source.get("normal_role") == "support"):
        raise ValueError("Selected support normal disagrees with the matched reference plane")
    binding["on_observation_scope"] = "matched contact samples and observed finite support; physical stability is unverified"


def bind_current_scene(program, scene, source_templates, current_owners, *, client=None, extractor=None):
    """Bind a cached skill from source functional evidence and current owners.

    ``source_templates`` is ``{'grasp': template}`` for an independent grasp,
    otherwise ``{event_id: {'carrier': template, 'reference': template}}``.
    Templates use ``source_function_geometry``'s explicit query/recipe schema;
    auxiliary direction queries may extend beyond the source interaction mask
    within its owner. Current observations never require a part mask/region ID.

    ``current_owners`` uses role keys and calibrated owner RGB-D observations
    described by ``_current_owner``. Its owner poses and visible geometry are
    expressed in the named ``scene.reference_frame``. Caller EEF feedback,
    actual Holding, known geometry and mechanism evidence are retained; none
    are inferred from a source demonstration or from successful matching.

    All inputs are copied. Every source recipe, owner ID, coordinate mapping
    and existing pose is checked before a current model call. Plane recipes
    provide only a normal and numerical tangent preference and are permitted
    for ``on``; they cannot satisfy full-frame manipulation or independent grasp.
    """
    program, result = deepcopy(program), deepcopy(scene)
    source_templates, current_owners = deepcopy(source_templates), deepcopy(current_owners)
    kind = program.get("skill_kind")
    if kind not in {"grasp", "manipulation"}:
        raise ValueError("Only grasp/manipulation programs have functional bindings")
    if not isinstance(result.get("reference_frame"), str) or not result["reference_frame"].strip():
        raise ValueError("Current scene must name its common reference_frame")
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
        if geometry_kind == "plane" and (kind == "grasp" or program["task"]["relation"] != "on"):
            raise ValueError("A plane supplies insufficient direction evidence for this full-frame skill")
        if kind == "manipulation" and program["task"]["relation"] == "on":
            existing = result.get("functional_bindings", {}).get(name, {})
            needs_normal = "on_geometry" not in existing if role == "carrier" else "support" not in result
            normal_role = "outward" if role == "carrier" else "support"
            if needs_normal and (geometry_kind != "plane" or template.get("normal_role") != normal_role):
                raise ValueError(f"Source on template needs an evidenced {normal_role} plane normal")
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
            if program["task"]["relation"] == "on":
                _bind_on_geometry(event, binding, source_templates[name], evidence[name], current,
                                  owner_evidence, result, client)
                # Preserve the sparse physical samples and the actually checked
                # support with their coverage; frame success alone does not
                # explain which contact geometry the downstream check used.
                evidence[name]["on_geometry"] = deepcopy(binding["on_geometry"])
                evidence[name]["support"] = deepcopy(result["support"])
                evidence[name]["orientation_source"] = deepcopy(binding["orientation_source"])
        result["functional_bindings"] = bindings
    result["region_binding_evidence"], result["owner_observation_evidence"] = evidence, owner_evidence
    return result


def ground_current_program(program, scene, source_templates, current_owners, *, output="eef",
                           client=None, extractor=None):
    """Bind current observations then call exactly one existing numerical core.

    ``output='eef'`` requests a complete grasp/manipulation EEF reference and
    retains all actual-Holding requirements. ``output='object'`` deliberately
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


def ground_rgbd(demo_dir, scene_dir, instruction, *, client=None, detector=None,
                extractor=None, tracker=None, output_dir=None):
    """Produce a numerical stage from RGB-D and task semantics, without new labels.

    Optional ``scene_dir/geometry.json`` may supply known interior/support
    geometry from another system component. It is never synthesized from an
    outer bounding box. Raw depth and intrinsics are needed for metric output;
    no additional task-specific training or offline motion dataset is used.
    """
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
        from local_models.sd_dino import extract_pair
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
    geometry_path = scene_dir / "geometry.json"
    if geometry_path.is_file():
        geometry = json.loads(geometry_path.read_text())
        if geometry.get("reference_frame") != "scene_camera":
            raise ValueError("Optional scene geometry must declare reference_frame=scene_camera")
        for key in ("support", "interior", "entry_region", "entry_start_index", "margin"):
            if key in geometry:
                scene[key] = geometry[key]
    if task["relation"] == "on" and "support" not in scene:
        # A reference object's largest plane may be a side panel. Select an
        # observed plane by the task and retain only its observed support area.
        # Orient its side from the demonstrated goal, since a carried object
        # may start below its destination support before being lifted onto it.
        # Use a material point average, not the arbitrary body-frame origin,
        # so a different valid body coordinate convention cannot flip the side.
        goal_center = goal[:3, :3] @ body.mean(axis=0) + goal[:3, 3]
        surface, surface_evidence = infer_support(
            scene_image, scene_depth, scene_camera.K, masks["scene"]["reference"], goal_center,
            instruction=instruction + "\n" + task.get("goal_description", ""), client=client)
        scene["support"] = surface.to_dict()
        evidence["support"] = surface_evidence
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
    fitted = transfer_motion(demo_poses, initial_pose, reference_transform=reference, kind=motion_kind(task))
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
