"""Generate grasp alternatives and submit each to the existing skill planner.

This recorded-input adapter consumes the same manifest, perception, robot
feedback and finite contact declarations as ``run_recorded_skill``. Model
inference exits before any motion-planning subprocess starts. No simulated
object state, new gripper command or skill-specific parameter enters here.
Only a formal planner-accepted archive can be selected for later execution.
"""

# BEGIN serial generated-grasp integration with unchanged native validation.
import argparse
from copy import deepcopy
from dataclasses import fields
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))


def write_json(path, value):
    """Keep complete numerical evidence, including exact float64 priors."""
    def convert(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        if isinstance(item, np.generic):
            return item.item()
        raise TypeError(type(item).__name__)
    Path(path).write_text(json.dumps(value, default=convert, indent=2,
                                    allow_nan=False) + "\n")


def option(command, flag):
    """Read one explicit singleton from the existing recorded request."""
    if command.count(flag) != 1:
        raise ValueError(f"The recorded planning command requires exactly one {flag}")
    return command[command.index(flag) + 1]


def set_option(command, flag, value):
    """Change an explicitly named adapter field, never positional guesswork."""
    if flag in command:
        option(command, flag)
        command[command.index(flag) + 1] = str(value)
    else:
        command.extend((flag, str(value)))


def load_inputs(command_path):
    """Load ordinary frozen inputs without opening the source demonstration."""
    command = json.loads(Path(command_path).read_text())
    if not isinstance(command, list) or not all(isinstance(x, str) for x in command):
        raise ValueError("planning-command must be a JSON argument array")
    if "--" in command:
        command = command[command.index("--") + 1:]
    if Path(command[2]).resolve() != Path(__file__).with_name("run_recorded_skill.py"):
        raise ValueError("This adapter requires the existing run_recorded_skill command")
    if any(flag in command for flag in ("--continuation", "--rebind", "--request-ledger")):
        raise ValueError("New action alternatives cannot rewrite an in-flight execution request")
    manifest = json.loads(Path(option(command, "--manifest")).read_text())
    root = Path(manifest["workspace_root"])
    segment = next(row for row in manifest["segments"]
                   if row["id"] == option(command, "--segment"))
    candidate = next(row for row in segment["grounding"]["candidates"]
                     if row["id"] == option(command, "--candidate"))
    program = json.loads((root / segment["program_path"]).read_text())
    scene = json.loads((root / candidate["scene_path"]).read_text())
    if program["skill_kind"] != "grasp" or scene["reference_frame"] != "world":
        raise ValueError("Grasp generation requires a grasp skill in the gravity-aligned world frame")
    return {"command": command, "manifest": manifest, "segment": segment,
            "candidate": candidate, "program": program, "scene": scene,
            "fixture": json.loads(Path(option(command, "--robot-fixture")).read_text()),
            "feedback": json.loads(Path(option(command, "--feedback")).read_text()),
            "contacts": json.loads(Path(option(command, "--contact-bindings")).read_text())}


def region_inputs(inputs):
    """Calibrate material geometry only; no invented runtime sphere indices.

    The existing planner independently binds these same native faces to its
    actual sphere inventory. The candidate check only needs the finite face,
    its affine finger motion, the fixed owner patch and mechanical q limits.
    Numerical tolerances come from ContactRule's existing public defaults;
    every physical allowance remains the explicit declaration's value.
    """
    import mujoco
    from manipulation.contact_binding import calibrate_box_material_features
    from manipulation.planning_scene import ContactRule
    from prepare_recorded_task import prepare_candidate

    fixture, feedback = inputs["fixture"], inputs["feedback"]
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    velocities = dict(zip(feedback["joint_names"], feedback["start_dq"]))
    arm_names = feedback["arm_joint_names"]
    problem, _ = prepare_candidate(inputs["program"], inputs["scene"],
        {**feedback, "joint_names": arm_names,
         "start_q": [measured[name] for name in arm_names],
         "start_dq": [velocities[name] for name in arm_names]}, inputs["segment"]["id"])
    activations = [row for row in problem.conditions
                   if row.get("kind") == "grasp_contact_activation"]
    supplied = inputs["contacts"]["grasp_activations"]
    if set(supplied) != {row["condition_id"] for row in activations}:
        raise ValueError("Every original close requires its explicit finite material bindings")
    selected_ids = {name for names in supplied.values() for name in names}
    declarations = [row for row in inputs["contacts"]["contacts"]
                    if row["rule_id"] in selected_ids]
    if (not selected_ids or {row["rule_id"] for row in declarations} != selected_ids
            or any(row.get("actor", "robot") != "robot"
                   or row["target_owner"] != inputs["scene"]["carrier_id"] for row in declarations)):
        raise ValueError("Candidate contact regions must bind this robot and selected carrier")
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    materials = calibrate_box_material_features(model, tcp_site=fixture["tcp_site"],
        joint_positions=measured, finger_joint_names=feedback["finger_joint_names"],
        material_faces=inputs["contacts"]["material_faces"])
    numerical = {field.name: field.default for field in fields(ContactRule)
                 if field.name in {"numerical_position_tolerance_m", "numerical_angle_tolerance_rad"}}
    rules = []
    for declaration in declarations:
        material = materials[declaration["material_id"]]
        rule = {**numerical, **deepcopy(declaration), "actor": "robot"}
        for name in ("feature_kind", "feature_points", "feature_normal", "feature_joint_translation"):
            if name in declaration:
                raise ValueError("Material geometry must come from the isolated native robot")
            rule[name] = material[name]
        rules.append(rule)
    # Keep each original close's permission set separate. Two different
    # activations cannot lend each other material/patch permissions.
    requirements = []
    for activation in activations:
        goal = problem.pose_events[activation["arrival_goal_id"][1]]
        progress = [float(goal.progress)]
        if activation["interval"][1] == 1.:
            progress.append(1.)
        requirements.append({"condition_id": activation["condition_id"],
            "activation_event": activation["activation_event"],
            "rule_ids": list(supplied[activation["condition_id"]]),
            "goal_progress": sorted(set(progress))})
    return {"contact_rules": rules, "requirements": requirements,
            "T_reference_owner": inputs["scene"]["carrier_pose"],
            "finger_position_limits": [model.jnt_range[model.joint(name).id].tolist()
                                       for name in feedback["finger_joint_names"]],
            "finger_joint_names": feedback["finger_joint_names"],
            "source": inputs["contacts"]["source"],
            "scope": "Fixed declared permission geometry; neither measured target surface nor actual contact"}


def check_region(candidate, regions, requirement=None):
    from manipulation.grasp_region import check_grasp_region_compatibility
    selected = regions["requirements"] if requirement is None else [requirement]
    reports = []
    for item in selected:
        report = check_grasp_region_compatibility(
            [rule for rule in regions["contact_rules"] if rule["rule_id"] in item["rule_ids"]],
            T_reference_tool=np.asarray(candidate["T_reference_tool"]),
            T_reference_owner=np.asarray(regions["T_reference_owner"]),
            finger_position_limits=np.asarray(regions["finger_position_limits"]))
        reports.append({"condition_id": item["condition_id"], **report})
    states = {row["status"] for row in reports}
    return {"status": ("contradicted" if "contradicted" in states else
                       "unresolved" if "unresolved" in states else "compatible"),
            "requirements": reports, "actual_contact_verified": False}


def check_approach(tool_pose, geometry, approach, limit):
    """Apply only an explicit hard direction to this exact planned tool pose."""
    if limit is None:
        return {"status": "not_required"}
    direction = np.asarray(approach, float)
    direction /= np.linalg.norm(direction)
    axis = (np.asarray(tool_pose) @ np.asarray(geometry["T_tool_grasp"]))[:3, 2]
    axis /= np.linalg.norm(axis)
    angle = float(np.arctan2(np.linalg.norm(np.cross(axis, direction)), axis @ direction))
    return {"status": "compatible" if angle <= limit else "contradicted",
            "angle_rad": angle, "maximum_angle_rad": limit}


def generate(args):
    """Run the installed model against the current visible owner and native hand."""
    from manipulation.grasp_generation import (GraspGenerationRequest,
                                               GraspGenXBackend, generate_grasp_candidates)
    from manipulation.tool_geometry import load_robosuite_panda_tool
    from manipulation.motion import _pose_array

    inputs = load_inputs(args.planning_command)
    scene = inputs["scene"]
    points = np.asarray(scene["observed_body_points"], dtype=float)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or not scene.get("body_points_source")):
        raise ValueError("Grasp generation requires explicit visible owner points and their source")
    owner = _pose_array(scene["carrier_pose"], "current reference-from-owner")
    points_reference = points @ owner[:3, :3].T + owner[:3, 3]
    # Robot-specific asset selection is an explicit adapter input. Generic
    # generation consumes only its measured geometry and calibrated transform.
    tool = load_robosuite_panda_tool(tool_id=scene["tool_id"], tool_frame=scene["tool_frame"])
    geometry = tool.grasp_generation_geometry()
    regions = region_inputs(inputs)
    from generated_grasp_tool_binding import validate_tool_binding
    geometry["fixture_binding_check"] = validate_tool_binding(
        tool, regions["contact_rules"], regions["finger_position_limits"])
    write_json(args.output / "region-inputs.json", regions)
    write_json(args.output / "gripper-geometry.json", geometry)
    backend = GraspGenXBackend(checkpoints_dir=args.checkpoints, gripper_geometry=geometry)
    request = GraspGenerationRequest(model_points_reference=points_reference,
        T_reference_tool_prior=owner @ np.asarray(scene["grasp_goal"]["T_carrier_eef"]),
        T_tool_grasp=geometry["T_tool_grasp"],
        approach_reference=None if args.approach is None else np.asarray(args.approach),
        hard_max_approach_angle_rad=args.hard_approach_angle)
    result = generate_grasp_candidates(request, backend, seed=args.seed)
    for candidate in result["candidates"]:
        candidate["region_compatibility"] = check_region(candidate, regions)
    result["input_provenance"] = {
        "scene": inputs["candidate"]["scene_path"],
        "visible_owner_point_count": len(points), "body_points_source": scene["body_points_source"],
        "model_geometry_is_complete": False, "permitted_region_source": regions["source"],
        "region_is_observed_segmentation": False, "simulator_object_state_used": False,
        "planning_tested": False, "actual_holding_verified": False}
    write_json(args.output / "generation.json", result)


def run_child(command, log_path, timeout):
    """Bound a serial child and terminate its own process group on deadline."""
    with Path(log_path).open("x") as stream:
        process = subprocess.Popen(command, cwd=REPO, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise TimeoutError("The finite generated-grasp search wall budget was exhausted")


def candidate_manifest(inputs, candidate, output):
    """Rebuild this action's scene while leaving all sensing and source data intact."""
    from manipulation.grasp import bind_grasp_candidate
    scene = bind_grasp_candidate(inputs["scene"], np.asarray(candidate["T_reference_tool"]),
        source={"kind": "demonstration_guided_graspgenx_candidate",
                "generation_report": str(output.parent.parent / "generation.json"),
                "candidate_id": candidate["candidate_id"], "branch": candidate["source"],
                "quality": candidate["quality"], "scope": "Candidate choice; original task precision retained"})
    scene_path = output / "scene.json"
    write_json(scene_path, scene)
    manifest = deepcopy(inputs["manifest"])
    segment = next(row for row in manifest["segments"] if row["id"] == inputs["segment"]["id"])
    identity = "generated/" + candidate["candidate_id"]
    segment["grounding"]["candidates"] = [{**deepcopy(inputs["candidate"]),
        "id": identity, "scene_path": str(scene_path), "registration_status": "unchanged_owner_binding"}]
    segment["grounding"]["status"] = "generated_grasp_action"
    segment["grounding"]["selected_candidate"] = identity
    path = output / "input-manifest.json"
    write_json(path, manifest)
    return path, identity


def planned_region_checks(trajectory_path, inputs, regions, *, geometry=None,
                          approach=None, hard_angle=None):
    """Check exact planned goal FK without granting execution authorization.

    A planner may choose any member of the original metric/angle allowed set.
    Checking only the generator's nominal transform therefore is insufficient.
    The saved cubic and progress map are sampled without fitting or smoothing;
    only the isolated robot model supplies FK. Finger q remains existential
    here and is never written into the executed finger trajectory.
    """
    import mujoco
    from manipulation.planning import JointTrajectory, _ProgressMap

    with np.load(trajectory_path, allow_pickle=False) as saved:
        if str(saved["classification"].item()) != "planner_accepted":
            raise ValueError("Only a formal planner-accepted curve can reach candidate selection")
        curve = JointTrajectory(tuple(map(str, saved["joint_names"])), saved["times"].copy(),
            saved["position"].copy(), saved["velocity"].copy(), saved["progress_u"].copy(),
            saved["progress_s"].copy(), (), (), {})
    model = mujoco.MjModel.from_xml_path(inputs["fixture"]["robot_mjcf_path"])
    data = mujoco.MjData(model)
    for name, position in zip(inputs["feedback"]["joint_names"], inputs["feedback"]["start_q"]):
        data.qpos[model.joint(name).qposadr[0]] = position
    site = model.site(inputs["fixture"]["tcp_site"]).id
    mujoco.mj_kinematics(model, data)
    actual = np.eye(4)
    actual[:3, :3] = data.site_xmat[site].reshape(3, 3)
    actual[:3, 3] = data.site_xpos[site]
    if not np.allclose(actual, inputs["feedback"]["eef_pose"], atol=1e-8, rtol=0):
        raise ValueError("Independent FK must match the original synchronized entrance")
    mapping = _ProgressMap(curve.progress_u, curve.progress_s)
    samples = [(u, progress, requirement) for requirement in regions["requirements"]
               for u, progress in mapping.anchor_parameters(requirement["goal_progress"])]
    checks = []
    for u, progress, requirement in samples:
        query_time = float(u * curve.duration)
        q = curve.sample(np.array([query_time]))[0][0]
        for name, position in zip(curve.joint_names, q):
            data.qpos[model.joint(name).qposadr[0]] = position
        mujoco.mj_kinematics(model, data)
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = data.site_xmat[site].reshape(3, 3), data.site_xpos[site]
        region = check_region({"T_reference_tool": pose}, regions, requirement)
        direction = check_approach(pose, geometry, approach, hard_angle)
        status = "contradicted" if direction["status"] == "contradicted" else region["status"]
        checks.append({"time_s": query_time, "progress": float(progress),
                       "T_reference_tool": pose, "region": region,
                       "approach": direction, "status": status})
    return checks


def planning_outcome(report, diagnostics=None):
    """Separate bounded search rejection from input, runtime and resource errors."""
    if report.get("status") != "completed":
        return "error"
    if report.get("planning_success") is True:
        return "planning_accepted"
    if report.get("planning_success") is not False:
        raise ValueError("Completed planning must report an explicit boolean success")
    # plan_trajectory returns operational exceptions as success=False too.
    # Only documented finite search/observation outcomes may try another pose;
    # an unknown error category is an error, never an assumed geometric failure.
    diagnostics = {} if diagnostics is None else diagnostics
    normal_failures = {"no_feasible_solution_found", "cumulative_planning_budget_exhausted",
                       "deadline_exceeded", "needs_observation", "unknown_space_forbidden"}
    failure = diagnostics.get("failure", report.get("failure"))
    if (failure not in normal_failures or diagnostics.get("error_type") not in
            {None, "PlanningWorkExhausted", "MotionPlanningError"}):
        return "error"
    return "not_accepted"


def run(args):
    """Retain exact prior priority, then try the remaining finite candidate pool."""
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + args.search_wall_seconds
    summary = {"status": "running", "selected": None, "attempts": [],
               "max_candidate_attempts": args.max_candidate_attempts,
               "search_wall_seconds": args.search_wall_seconds,
               "actual_execution_tested": False, "scope": "Generation and full independent grasp planning"}
    try:
        child = [sys.executable, "-u", str(Path(__file__).resolve()), "--generate-only",
                 "--planning-command", str(args.planning_command), "--output", str(args.output),
                 "--checkpoints", str(args.checkpoints), "--tool-model", args.tool_model,
                 "--seed", str(args.seed)]
        if args.approach is not None:
            child += ["--approach", *map(str, args.approach)]
        if args.hard_approach_angle is not None:
            child += ["--hard-approach-angle", str(args.hard_approach_angle)]
        write_json(args.output / "generation-command.json", child)
        code = run_child(child, args.output / "generation.log", max(1., deadline - time.monotonic()))
        if code != 0:
            raise RuntimeError(f"Candidate generation failed with exit code {code}; see generation.log")
        generation = json.loads((args.output / "generation.json").read_text())
        regions = json.loads((args.output / "region-inputs.json").read_text())
        geometry = json.loads((args.output / "gripper-geometry.json").read_text())
        inputs = load_inputs(args.planning_command)
        attempts = 0
        for candidate in generation["candidates"]:
            # A rejected nominal centre cannot reject the complete original
            # goal tolerance set. Keep it eligible for the full motion planner,
            # then check the actual planned arrival member below.
            remaining = deadline - time.monotonic()
            if attempts >= args.max_candidate_attempts or remaining <= 0:
                summary["status"] = "search_budget_exhausted"
                break
            output = args.output / "attempts" / f"{attempts:03d}"
            output.mkdir(parents=True)
            manifest, identity = candidate_manifest(inputs, candidate, output)
            command = list(inputs["command"])
            for flag, value in (("--manifest", manifest), ("--candidate", identity),
                                ("--output", output / "planning"),
                                ("--max-host-rss-mib", args.max_host_rss_mib),
                                ("--max-device-used-mib", min(args.max_device_used_mib,
                                     int(option(inputs["command"], "--max-device-used-mib")))),
                                ("--host-process-root-pid", os.getpid())):
                set_option(command, flag, value)
            set_option(command, "--planning-seconds", min(float(option(command, "--planning-seconds")), remaining))
            write_json(output / "planning-command.json", command)
            row = {"candidate_id": candidate["candidate_id"], "source": candidate["source"],
                   "quality": candidate["quality"], "output": str(output), "status": "planning"}
            summary["attempts"].append(row)
            write_json(args.output / "result.json", summary)
            attempts += 1
            code = run_child(command, output / "planning.log", max(1., deadline - time.monotonic()))
            report = json.loads((output / "planning" / "run-report.json").read_text())
            # An input/runtime failure can occur before a PlanningProblem or
            # planning report exists. Preserve that original operation error.
            if code != 0 or report.get("status") != "completed":
                row.update(returncode=code, status="error", error=report.get("error"))
                raise RuntimeError(f"Motion planning failed as an operation: {report}")
            planning = json.loads((output / "planning" / "planning" / "planning-report.json").read_text())
            diagnostics = planning.get("diagnostics", {})
            row.update(returncode=code, status=planning_outcome(report, diagnostics))
            if code != 0 or row["status"] == "error":
                row.update(failure=diagnostics.get("failure"), error_type=diagnostics.get("error_type"),
                           reason=diagnostics.get("reason"))
                raise RuntimeError(f"Motion planning failed as an operation: {report.get('error') or row}")
            if row["status"] != "planning_accepted":
                # Native initialization/allocation errors return structured
                # planning failures; trying another grasp cannot repair them.
                row["failure"] = diagnostics.get("failure")
                continue
            trajectory = output / "planning" / "planning" / "trajectory.npz"
            checks = planned_region_checks(trajectory, inputs, regions, geometry=geometry,
                approach=args.approach, hard_angle=args.hard_approach_angle)
            write_json(output / "planned-region-checks.json", checks)
            if time.monotonic() >= deadline:
                summary["status"] = "search_budget_exhausted"
                break
            if any(check["status"] == "contradicted" for check in checks):
                row["status"] = "planned_region_contradiction"
                continue
            row["planned_region_status"] = ("compatible" if all(c["status"] == "compatible" for c in checks)
                                             else "unresolved")
            summary.update(status="planning_accepted", selected={
                "candidate_id": candidate["candidate_id"], "source": candidate["source"],
                "quality": candidate["quality"], "trajectory": str(trajectory),
                "manifest": str(manifest), "scene": str(output / "scene.json"),
                "T_reference_tool": candidate["T_reference_tool"],
                "region_status": row["planned_region_status"], "actual_holding_verified": False})
            break
        else:
            summary["status"] = "no_accepted_candidate_in_generated_pool"
    except TimeoutError as error:
        summary.update(status="search_budget_exhausted", error=str(error))
    except Exception as error:
        summary.update(status="error", error=f"{type(error).__name__}: {error}")
        (args.output / "exception.txt").write_text(traceback.format_exc())
    finally:
        summary["elapsed_seconds"] = time.monotonic() - started
        write_json(args.output / "result.json", summary)
    print(json.dumps({key: summary[key] for key in ("status", "selected", "elapsed_seconds")}), flush=True)
    return 0 if summary["status"] == "planning_accepted" else 1


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--planning-command", required=True, type=Path)
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--checkpoints", required=True, type=Path)
    result.add_argument("--tool-model", required=True, choices=("robosuite_panda",))
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--max-candidate-attempts", type=int, default=8)
    result.add_argument("--search-wall-seconds", type=float, default=600.)
    result.add_argument("--max-host-rss-mib", type=int, default=8192)
    result.add_argument("--max-device-used-mib", type=int, default=6144)
    result.add_argument("--approach", type=float, nargs=3)
    result.add_argument("--hard-approach-angle", type=float, help="Explicit hard angular bound in radians")
    result.add_argument("--generate-only", action="store_true")
    return result


if __name__ == "__main__":
    args = parser().parse_args()
    args.output = args.output.resolve()
    if (args.max_candidate_attempts <= 0 or not np.isfinite(args.search_wall_seconds)
            or args.search_wall_seconds <= 0 or not 0 < args.max_host_rss_mib <= 12288
            or not 0 < args.max_device_used_mib <= 10240):
        raise ValueError("Finite search and resource limits must fit this adapter's admitted local bounds")
    if args.generate_only:
        generate(args)
    else:
        raise SystemExit(run(args))
# END model scores never authorize execution or modify observed collision evidence.
