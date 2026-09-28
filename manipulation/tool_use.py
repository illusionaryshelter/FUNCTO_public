"""Consume compiled tool skills through the shared functional-region core.

The demonstrated functional frames and event times are explicit skill inputs.
This module neither selects a named tool action nor reconstructs obsolete
function/center/grasp annotations. Object motion can be consumed before a robot
attachment is known; no end-effector motion or execution success is fabricated.
"""

# BEGIN compiled functional-program tool interface.
from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path

import numpy as np

from .goals import _points
from .grasp import validate_timeline
from .motion import _poses_array
from .pipeline import write_json
from .program import ground_object_program


def _required(record, keys, name):
    if not isinstance(record, Mapping):
        raise ValueError(f"{name} must be a mapping of compiled metadata")
    missing = sorted(set(keys)-set(record))
    if missing:
        raise ValueError(f"{name} missing required compiled metadata: {', '.join(missing)}")


def _compiled_program(program, *, require_motion):
    """Check missing compilation products without inventing their semantics."""
    _required(program, ("skill_kind", "task", "demo"), "interaction_program")
    if program["skill_kind"] != "manipulation":
        raise ValueError("The compiled tool program must have skill_kind='manipulation'")
    task = program["task"]
    _required(task, ("relation", "motion"), "interaction_program.task")
    if task["relation"] != "tool_use":
        raise ValueError("The compiled tool program must declare task.relation='tool_use'")
    if task["motion"] not in {"free", "auto"}:
        raise ValueError("The tool function optimizer cannot discard a constrained motion family")
    demo = program["demo"]
    keys = ["frame_indices", "timestamps", "functional_events"]
    if require_motion:
        keys.append("T_reference_carrier")
    _required(demo, keys, "interaction_program.demo")
    frames, times = validate_timeline(demo["frame_indices"], demo["timestamps"])
    events = demo["functional_events"]
    if not isinstance(events, list) or not events:
        raise ValueError("interaction_program.demo.functional_events must explicitly identify functional events")
    event_fields = ("event_id", "frame_index", "carrier_frame", "reference_frame", "source",
                    "position_tolerance_m", "rotation_tolerance_rad")
    for index, event in enumerate(events):
        _required(event, event_fields, f"interaction_program.demo.functional_events[{index}]")
        frame = event["frame_index"]
        if (isinstance(frame, (bool, np.bool_)) or not isinstance(frame, (int, np.integer))
                or frame not in frames):
            raise ValueError(f"Functional event {index} frame_index must identify an original demonstration frame")
    if require_motion:
        motion = _poses_array(demo["T_reference_carrier"], "demo.T_reference_carrier")
        if len(motion) != len(frames):
            raise ValueError("Compiled T_reference_carrier must cover every original demonstration frame")
    return frames, times


def prepare_tracked_demo(program, tracking):
    """Return a copied program with exactly-once reference-motion removal.

    Tracking's carrier/reference body poses map their fixed local coordinates
    to camera zero. Compiled functional frames must already use those *same*
    owner coordinates; a numeric bridge cannot infer or repair different body
    origins. Source frame IDs must exactly match the compiled timeline, including
    sparse/nonzero IDs. No function event, direction or timestamp is re-inferred.

    The existing tracker exposes ``poses = reference[0] @ inv(reference[t]) @
    carrier[t]``. Verify that declared gauge, then store the pure owner-relative
    ``T_reference_carrier = inv(reference[t]) @ carrier[t]``. This conversion is
    performed once, with no synthetic point trajectory, dataset copy or symlink.
    Original dictionaries and arrays remain unmodified.
    """
    frames, times = _compiled_program(program, require_motion=False)
    _required(tracking, ("frame_indices", "body_poses", "poses"), "tracking")
    # Check IDs before borrowing compiled times for the generic validator: a
    # dropped tracking row is a frame mismatch, not a source timestamp error.
    if not np.array_equal(np.asarray(tracking["frame_indices"]), frames):
        raise ValueError("Tracking frame_indices must exactly match the compiled original demonstration frame_indices")
    tracked_frames, _ = validate_timeline(tracking["frame_indices"], times)
    if "timestamps" in tracking:
        _, tracked_times = validate_timeline(tracked_frames, tracking["timestamps"])
        if not np.allclose(tracked_times, times, atol=1e-9, rtol=0):
            raise ValueError("Tracking timestamps disagree with the compiled demonstration timeline")
    owners = tracking["body_poses"]
    _required(owners, ("carrier", "reference"), "tracking.body_poses")
    carrier = _poses_array(owners["carrier"], "tracking.body_poses.carrier")
    reference = _poses_array(owners["reference"], "tracking.body_poses.reference")
    recorded = _poses_array(tracking["poses"], "tracking.poses")
    if any(len(values) != len(frames) for values in (carrier, reference, recorded)):
        raise ValueError("Tracked carrier, reference and poses must cover the complete compiled frame_indices")
    relative = np.linalg.inv(reference) @ carrier
    if not np.allclose(recorded, reference[0] @ relative, atol=1e-8, rtol=0):
        raise ValueError("tracking.poses disagrees with its declared initial-reference gauge")
    result = deepcopy(program)
    result["demo"]["T_reference_carrier"] = relative
    return result


def _load_record(path, description):
    if not path.is_file():
        raise ValueError(f"Missing {description}: {path}; compile functional regions, frames and events before tool transfer")
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid {description} JSON: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain a JSON object: {path}")
    return value


def run_tool_use(demo_dir, scene_dir, task, instruction, output_dir, *, tracking=None):
    """Transfer a compiled tool skill without action-label or landmark dispatch.

    Read ``demo_dir/interaction_program.json`` and
    ``scene_dir/interaction_scene.json``. Current scene functional bindings must
    already have been grounded from its observations. The same object program
    core used by full skill execution generates and independently validates all
    functional-event targets. With ``tracking=None``, consume the compiled
    T_reference_carrier directly; an existing trajectory never forces another
    tracking/model/API pass.

    ``task`` is checked for relation and motion compatibility, not mapped to a
    fixed list of named actions. ``instruction`` is output provenance only;
    semantic compilation and VLM calls belong upstream. Output is an object
    geometry candidate, with no inferred grasp, EEF path or execution claim.
    """
    demo_dir, scene_dir, output_dir = Path(demo_dir), Path(scene_dir), Path(output_dir)
    program_path, scene_path = demo_dir / "interaction_program.json", scene_dir / "interaction_scene.json"
    _required(task, ("relation", "motion"), "requested task")
    if task["relation"] != "tool_use":
        raise ValueError("The tool interface requires requested task.relation='tool_use'")
    if task["motion"] not in {"free", "auto"}:
        raise ValueError("The tool function optimizer cannot discard the requested constrained motion family")
    program = _load_record(program_path, "compiled interaction program")
    _compiled_program(program, require_motion=tracking is None)
    if tracking is not None:
        program = prepare_tracked_demo(program, tracking)
    scene = _load_record(scene_path, "grounded interaction scene")
    _required(scene, ("functional_bindings", "initial_pose", "carrier_id", "reference_id",
                      "reference_frame", "reference_pose", "body_points"), "interaction_scene")
    _points(scene["body_points"], "interaction_scene.body_points", minimum=3)
    destination = output_dir / "trajectory.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite a previous tool trajectory: {destination}")
    result = ground_object_program(program, scene)
    result["input_evidence"] = {
        "interaction_program": str(program_path.resolve()),
        "interaction_scene": str(scene_path.resolve()),
        "motion_source": "current tracked demonstration" if tracking is not None else "compiled demonstration",
        "reference_motion": "removed once from tracked fixed-owner poses" if tracking is not None else "compiled T_reference_carrier",
        "instruction": instruction,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(destination, result)
    return result
# END compiled functional-program tool interface.
