"""Transfer object/part demonstrations while preserving task and frame meaning.

The numerical entry accepts grounded observations, which can come from the
RGB-D adapter or an existing system's perception. No tool/grasp landmarks are
required. Each stage operates on one rigid carrier; ordered stages explicitly
pass updated entity poses instead of reusing the original static scene.
"""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .goals import (InteriorRegion, SupportSurface, check_inside, check_on,
                    place_inside, place_on)
# BEGIN separate observed structure from the applied task motion constraint.
from .motion import analyze_motion, transfer_motion
# END motion analysis import.


RELATIONS = {"transfer", "on", "inside", "articulation"}
INTERACTIONS = {"none", "grasp", "push", "pull", "tool"}


def motion_kind(task):
    # BEGIN deprecated semantic-enum adapter.
    # Only the old object-stage entry maps a VLM motion label here. Explicit
    # MotionModel geometry and planning requirements are separate live APIs.
    # END deprecated semantic-enum adapter.
    # BEGIN semantic constraint selection, independent of candidate diagnostics.
    """Select the applied family from semantic evidence, not trajectory rank.

    The existing semantic ``motion`` proposal can specify planar motion for a
    whole-stage support/contact requirement; complete poses must still fit it.
    ``auto`` for a general object task does not authorize a new mechanical
    restriction. ``run_stage`` separately reports every compatible observed
    family, so keeping a carry free does not hide its planar/linear geometry.
    An articulation task supplies mechanism evidence and permits automatic
    selection only of a geometrically supported joint family downstream.
    """
    if task["motion"] == "auto" and task["relation"] != "articulation":
        return "free"
    return task["motion"]
    # END semantic constraint selection.


def serializable(value):
    """Convert numeric outputs without unsafe pickle or silent NaN encoding."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(item) for item in value]
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(serializable(value), ensure_ascii=False,
                                    indent=2, allow_nan=False))


def _matrix(value, name):
    matrix = np.asarray(value, float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite 4x4 pose")
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(matrix[:3, :3]), 1, atol=1e-6)):
        raise ValueError(f"{name} must be a rigid SE(3) transform")
    return matrix


def _surface(value):
    return SupportSurface(np.asarray(value["origin"]), np.asarray(value["basis"]),
                          np.asarray(value["normal"]), np.asarray(value["boundary"]),
                          source=value.get("source", "provided_support_geometry"))


def _interior(value):
    if "source" not in value:
        raise ValueError("Interior geometry requires an evidence source")
    if "halfspaces" in value:
        return InteriorRegion(np.asarray(value["halfspaces"]), value["source"])
    if "vertices" in value:
        return InteriorRegion.from_vertices(value["vertices"], value["source"])
    raise ValueError("Known free interior needs halfspaces or vertices, not outer bounds")


def run_stage(observation):
    """Create and validate an object trajectory from a grounded demonstration.

    Required fields: ``task`` (relation/interaction/motion), ``demo_poses``,
    ``scene`` (initial_pose/body_points/reference_transform), ``carrier_id``,
    ``reference_id`` and ``reference_frame``. All scene geometry shares that
    reference frame; object points use a fixed body frame. Optional grasp or
    contact geometry is evidence only, never a fabricated rigid EE transform.
    """
    # BEGIN deprecated object-stage task dispatch.
    # This entry consumes the legacy relation/interaction/motion schema from
    # raw RGB-D or saved object-stage observations. Current skill experiments
    # bind explicit representation data and typed planning requirements.
    # Shared geometry helpers in this module remain used by those consumers.
    # END deprecated object-stage task dispatch.
    task = observation["task"]
    relation, interaction = task["relation"], task["interaction"]
    if relation not in RELATIONS:
        raise ValueError(f"Unsupported object relation: {relation!r}")
    if interaction not in INTERACTIONS:
        raise ValueError(f"Unsupported interaction: {interaction!r}")
    for key in ("carrier_id", "reference_id", "reference_frame"):
        if not isinstance(observation.get(key), str) or not observation[key].strip():
            raise ValueError(f"A nonempty {key} is required")
    if observation["carrier_id"] == observation["reference_id"]:
        raise ValueError("Carrier and reference must be distinct entities or parts")
    scene = observation["scene"]
    margin = scene.get("margin", 0.)
    if (isinstance(margin, (bool, np.bool_)) or not np.isscalar(margin)
            or not isinstance(margin, (int, float, np.number))
            or not np.isfinite(margin) or margin < 0):
        raise ValueError("Geometry margin must be finite and nonnegative")
    start = _matrix(scene["initial_pose"], "Initial object pose")
    transform = _matrix(scene["reference_transform"], "Reference binding")
    body = np.asarray(scene["body_points"], float)
    if (body.ndim != 2 or body.shape[1] != 3 or len(body) < 3
            or not np.isfinite(body).all()):
        raise ValueError("Scene body_points must contain finite local object geometry")
    demo = np.asarray(observation["demo_poses"], float)
    # BEGIN retain observed candidates even when the task permits free motion.
    # These fits use demonstration-reference coordinates, before scene binding
    # or endpoint correction. They are evidence, never a substitute for the
    # selected task constraint or a reason to silently downgrade that constraint.
    observed_structure = analyze_motion(demo)
    observed_structure["pose_frame"] = "demonstration_reference_from_fixed_body"
    # END observed candidate diagnostics.
    # BEGIN this geometry-only input declares its sample phase explicitly.
    # run_stage has no timestamp contract. Uniform geometric progress does not
    # claim a physical recording or execution clock, unlike compiled programs.
    options = {"kind": motion_kind(task), "reference_transform": transform,
               "progress": np.linspace(0., 1., len(demo))}
    # END both preliminary and goal-bound transfers consume the same phase.
    preliminary = transfer_motion(demo, start, **options)
    if relation == "articulation" and preliminary["model"].kind not in {"prismatic", "revolute"}:
        raise ValueError("Articulation observations do not establish a supported fixed-axis/direction model")
    # A bound goal encodes actual target-relative meaning. Without a supplied
    # explicit pose, map the demonstration's endpoint position through the
    # reference binding; simply copying displacement would miss a moved target.
    exact_goal = "goal_pose" in scene
    if exact_goal and "preferred_goal_pose" in scene:
        raise ValueError("Provide an exact goal or an estimated preferred goal, not both")
    if exact_goal:
        preferred = _matrix(scene["goal_pose"], "Bound goal pose")
    elif "preferred_goal_pose" in scene:
        preferred = _matrix(scene["preferred_goal_pose"], "Estimated preferred goal pose")
    elif relation == "articulation":
        raise ValueError("Articulation needs a geometrically bound goal pose; action labels are insufficient")
    else:
        preferred = np.asarray(preliminary["poses"][-1]).copy()
        preferred[:3, 3] = (transform @ demo[-1])[:3, 3]
    geometry = None
    if relation == "on":
        if "support" not in scene:
            raise ValueError("Placement on a surface needs a bounded support patch")
        geometry = _surface(scene["support"])
        if exact_goal:
            goal, goal_report = preferred, check_on(preferred, body, geometry)
            if (goal_report["support_gap_m"] > 1e-7
                    or goal_report["minimum_boundary_clearance_m"] < margin-1e-7):
                raise ValueError("Exact goal does not satisfy the support relation")
        else:
            goal, goal_report = place_on(preferred, body, geometry, margin=margin)
    elif relation == "inside":
        # BEGIN separate demonstrated inside motion from volume certification.
        # RGB-D often observes the destination relation without revealing the
        # whole cavity. Transfer that bound endpoint without inventing walls or
        # treating unknown clearance as zero. Supplied geometry is always used;
        # an invalid/incompatible region is never bypassed by this branch.
        if "interior" in scene:
            geometry = _interior(scene["interior"])
            if exact_goal:
                goal, goal_report = preferred, check_inside(preferred, body, geometry)
                if goal_report["minimum_clearance_m"] < margin-1e-7:
                    raise ValueError("Exact goal does not satisfy the interior relation")
            else:
                goal, goal_report = place_inside(preferred, body, geometry, margin=margin)
        else:
            if margin > 0 and "entry_region" not in scene:
                raise ValueError("An inside margin needs known interior or entry-region geometry")
            goal, goal_report = preferred, {
                "containment_status": "not_verified",
                "geometry_source": "not_observed_or_provided",
                "goal_source": scene.get("goal_source", "explicit_goal_pose" if exact_goal
                                         else "bound_demonstration_endpoint"),
                "reason": "A transferred endpoint does not certify unobserved free container volume",
            }
        # END separate demonstrated inside motion from volume certification.
    else:
        goal, goal_report = preferred, {"geometry_source": scene.get("goal_source", "bound_demonstration")}
    transferred = transfer_motion(demo, start, goal_pose=goal, **options)
    poses = np.asarray(transferred["poses"])
    if not np.allclose(poses[0], start, atol=1e-7):
        raise ValueError("Transfer changed the observed initial pose")
    if not np.allclose(poses[-1], goal, atol=1e-7):
        raise ValueError("Transfer failed the bound goal pose")
    if relation == "on":
        goal_report = check_on(poses[-1], body, geometry)
    # BEGIN verify known cavity geometry without claiming unknown containment.
    elif relation == "inside" and geometry is not None:
        goal_report = check_inside(poses[-1], body, geometry)
        goal_report["containment_status"] = "verified_on_supplied_points"
    # END verify known cavity geometry without claiming unknown containment.
    # A supplied convex entry corridor is checked over actual object extent.
    # Its start index is an observed/demo event, not a hard-coded fraction.
    if "entry_region" in scene:
        region = _interior(scene["entry_region"])
        index = scene.get("entry_start_index")
        if type(index) is not int or not 0 <= index < len(poses):
            raise ValueError("Entry-region checks need a valid demonstrated entry_start_index")
        checks = [check_inside(pose, body, region) for pose in poses[index:]]
        clearance = min(item["minimum_clearance_m"] for item in checks)
        if clearance < margin-1e-7:
            raise ValueError("Transferred object trajectory leaves the observed entry corridor")
        goal_report.update(entry_corridor_violation_m=max(0., -clearance),
                           entry_corridor_minimum_clearance_m=clearance,
                           entry_corridor_margin_m=float(margin),
                           entry_corridor_checked_nodes=len(checks),
                           entry_corridor_scope="saved nodes only; interpolation between nodes is not certified")
    frame_indices = np.asarray(observation.get("frame_indices", list(range(len(demo)))))
    if (frame_indices.shape != (len(poses),) or frame_indices.dtype.kind not in "iu"
            or (frame_indices < 0).any() or (frame_indices[1:] <= frame_indices[:-1]).any()):
        raise ValueError("Frame indices must be nonnegative strictly increasing integers matching the demonstration")
    model = transferred["model"]
    # BEGIN distinguish what was observed from what is actually enforced.
    transferred["diagnostics"]["observed_structure"] = observed_structure
    transferred["diagnostics"]["applied_constraint"] = {
        "requested_motion": task["motion"], "kind": model.kind,
        "source": ("articulation semantics and compatible observed motion"
                   if task["motion"] == "auto" and relation == "articulation"
                   else "no additional task motion constraint"
                   if task["motion"] == "auto"
                   else "task motion specification validated against complete poses"),
    }
    # END observed/applied distinction.
    return {
        "status": "passed", "output_type": "object_trajectory",
        "carrier_id": observation["carrier_id"], "reference_id": observation["reference_id"],
        "reference_frame": observation["reference_frame"],
        "pose_convention": "column-vector reference_from_fixed_body, metres, 4x4 SE(3)",
        "task": task, "poses": poses, "frame_indices": frame_indices,
        "body_points": body, "model": model.to_dict(), "goal_pose": goal,
        "goal_geometry": None if geometry is None else geometry.to_dict(),
        "goal_checks": goal_report, "motion_checks": transferred["diagnostics"],
        "interaction_geometry": observation.get("interaction_geometry"),
        "interaction_geometry_status": ("provided" if observation.get("interaction_geometry") is not None
                                        else "not_estimated_for_object_trajectory"),
        "observation_evidence": observation.get("evidence", {}),
        "geometry_coverage": scene.get("geometry_coverage", "supplied_points_only"),
        "path_scope": "demonstration transfer; no complete-scene collision or force guarantee",
    }


def _move_scene_geometry(scene, delta):
    """Move reference-attached goals when a previous stage moved that entity."""
    scene["reference_transform"] = delta @ _matrix(scene["reference_transform"], "Reference binding")
    for key in ("goal_pose", "preferred_goal_pose"):
        if key in scene:
            scene[key] = delta @ _matrix(scene[key], "Bound goal pose")
    if "support" in scene:
        surface = _surface(scene["support"])
        surface.origin = delta[:3, :3] @ surface.origin + delta[:3, 3]
        surface.basis = delta[:3, :3] @ surface.basis
        surface.normal = delta[:3, :3] @ surface.normal
        scene["support"] = surface.to_dict()
    for key in ("interior", "entry_region"):
        if key in scene:
            region = _interior(scene[key])
            normals = region.halfspaces[:, :3] @ delta[:3, :3].T
            offsets = region.halfspaces[:, 3] - normals @ delta[:3, 3]
            scene[key] = InteriorRegion(np.column_stack((normals, offsets)), region.source).to_dict()


def run_sequence(observations):
    """Run an already ordered stage list, updating carrier/reference states.

    Each stage's ``reference_pose`` describes the reference at capture time.
    These transforms and every initial carrier pose must share one world or
    otherwise explicitly registered reference frame. This is state handoff,
    not an autonomous task-order planner.
    """
    if not isinstance(observations, list) or not observations:
        raise ValueError("An ordered nonempty stage list is required")
    states, body_frames, placements, invalidated = {}, {}, {}, {}
    output, frame = [], observations[0]["reference_frame"]
    for original in observations:
        observation = deepcopy(original)
        if observation["reference_frame"] != frame:
            raise ValueError("Sequence stages must use a common registered reference frame")
        scene = observation["scene"]
        carrier, reference = observation["carrier_id"], observation["reference_id"]
        if carrier in invalidated or reference in invalidated:
            raise ValueError("A placed entity moved with an uncertain contact relationship; "
                             "a fresh registered observation is required before continuing")
        body_frame = observation.get("body_frame_id")
        if not isinstance(body_frame, str) or not body_frame:
            raise ValueError("Sequence stages must declare the carrier's registered body_frame_id")
        if carrier in body_frames and body_frames[carrier] != body_frame:
            raise ValueError("Repeated carrier stages must share a registered fixed body frame")
        body_frames[carrier] = body_frame
        if carrier in states:
            scene["initial_pose"] = states[carrier].copy()
        if reference in states:
            if observation.get("reference_body_frame_id") != body_frames[reference]:
                raise ValueError("Moved reference requires its registered reference_body_frame_id")
            if "reference_pose" not in scene:
                raise ValueError("A previously moved reference needs its capture-time reference_pose")
            captured = _matrix(scene["reference_pose"], "Captured reference pose")
            delta = states[reference] @ np.linalg.inv(captured)
            _move_scene_geometry(scene, delta)
        result = run_stage(observation)
        placements.pop(carrier, None)
        # A placed object is not a rigid attachment. If its support/container
        # moves, dependent world poses become unknown, including nested items.
        changed = [carrier]
        if not np.allclose(result["poses"], result["poses"][0], atol=1e-8):
            while changed:
                moved = changed.pop()
                for dependent, owner in list(placements.items()):
                    if owner == moved and dependent not in invalidated:
                        invalidated[dependent] = {"status": "needs_observation", "moved_reference": moved}
                        states.pop(dependent, None)
                        changed.append(dependent)
        states[carrier] = np.asarray(result["poses"][-1]).copy()
        if observation["task"]["relation"] in {"on", "inside"}:
            placements[carrier] = reference
        output.append(result)
    return {"status": "passed", "reference_frame": frame, "stages": output,
            "final_entity_poses": states, "unobserved_entity_states": invalidated}


def run_prepared(input_path, output_dir):
    """Replay grounded observations, preserving failures and fresh-run semantics."""
    path, destination = Path(input_path), Path(output_dir)
    inputs = json.loads(path.read_text())
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Use an empty output directory: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "input": str(path.resolve())}
    write_json(destination / "run.json", report)
    try:
        write_json(destination / "observations.json", inputs)
        # BEGIN consume compiled skills through the existing prepared entry.
        if "skill_kind" in inputs:
            from .program import ground_program
            demo = inputs["demo"]
            if "trajectory_file" in demo:
                array_path = (path.parent / demo["trajectory_file"]).resolve(strict=True)
                # The saved observations may be replayed from their new run
                # directory; preserve the actual array location explicitly.
                demo["trajectory_file"] = str(array_path)
                write_json(destination / "observations.json", inputs)
                with np.load(array_path, allow_pickle=False) as arrays:
                    for name in arrays.files:
                        if name in demo:
                            raise ValueError(f"Duplicate inline/NPZ demonstration field: {name}")
                        demo[name] = arrays[name].copy()
            result = ground_program(inputs, inputs["scene"])
        else:
            result = run_sequence(inputs["stages"]) if "stages" in inputs else run_stage(inputs)
        # END shared prepared entry; object-only stages retain their own scope.
        write_json(destination / "trajectory.json", result)
        report["status"] = "passed"
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        write_json(destination / "run.json", report)
    return result
