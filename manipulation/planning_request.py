"""Compile and solve one explicit grounded-reference planning request.

``bind`` consumes the public task compiler's program/scene contract and the
complete continuous reference produced by grounding. ``plan`` owns one scene,
official robot validator and direct backend. A caller supplies task goals and
contacts for each hypothesis as explicit data; representation kinds do not
select hidden task factories or numerical allowances in this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
import os
import time
import traceback

import numpy as np

from .planning_records import _write_diagnostic_json, plan_and_record


def reexpress_contact_bindings(contact_bindings, material_contacts, owner_bindings, *, source):
    """Bind source task patches to current charts and official robot features.

    ``owner_bindings`` maps each source contact owner/actor to its current id
    and ``T_current_source`` chart transform. Robot material rows contain the
    official current TCP features. Other material rows retain the task actor's
    source chart and use its own transform, independently of the target patch.
    ``build_material_contact_rules`` assembles the existing contact contract. Patch
    dimensions, gap/normal allowances, intervals and event activation remain
    the original source declarations. Raw source records remain caller-owned.
    """
    from .contact_binding import build_material_contact_rules
    result = deepcopy(contact_bindings)
    # Source material selectors and thickness are representation evidence.
    # ContactRule consumes their resolved explicit features; the complete raw
    # declarations remain in the persisted planning input beside this result.
    result.pop("material_faces", None)
    feature_keys = ("feature_kind", "feature_points", "feature_normal",
                    "feature_joint_translation")
    materials = dict(materials={}, owner_materials={})
    for row in material_contacts["contacts"]:
        feature = {key: deepcopy(row[key]) for key in (*feature_keys, "source")}
        if row["actor"] == "robot":
            feature["robot_body"] = row["robot_body"]
            materials["materials"][row["material_id"]] = feature
        else:
            actor = owner_bindings[row["actor"]]
            transform = np.asarray(actor["T_current_source"])
            feature["feature_points"] = (np.asarray(feature["feature_points"])
                @ transform[:3, :3].T + transform[:3, 3])
            feature["feature_normal"] = transform[:3, :3] @ feature["feature_normal"]
            if feature["feature_joint_translation"] is not None:
                feature["feature_joint_translation"] = np.einsum("ij,njk->nik",
                    transform[:3, :3], feature["feature_joint_translation"])
            materials["owner_materials"].setdefault(actor["owner_id"], {})[row["material_id"]] = feature
    for row in result["contacts"]:
        row.pop("minimum_thickness_m", None)
        for key in (*feature_keys, "robot_body"):
            row.pop(key, None)
        if row["actor"] != "robot":
            row["actor"] = owner_bindings[row["actor"]]["owner_id"]
        binding = owner_bindings[row["target_owner"]]
        relocation = np.asarray(binding["T_current_source"])
        row["target_owner"] = binding["owner_id"]
        row["patch_origin"] = relocation[:3, :3] @ row["patch_origin"] + relocation[:3, 3]
        row["patch_axes"] = relocation[:3, :3] @ row["patch_axes"]
        row["source"] = source
    result["contacts"] = [asdict(rule) for rule in
                           build_material_contact_rules(materials, result["contacts"])]
    result["source"] = source
    return result


def bind_task_inputs(hypothesis, request):
    """Express original typed task geometry in this hypothesis's current chart.

    F places the source functional region. The current owner's calibrated
    chart and source owner-from-region relation yield the source-to-current
    local mapping. The independent E target supplies the existing grasp goal
    contract; complete H_g placement belongs only to ``reference_phase``.
    Manipulation consumes the current typed scene's attachment, mechanism,
    finite support and functional bindings through the original compiler.
    """
    original, binding = request["task"], request["task_binding"]
    program, source_scene = original["program"], original["scene"]
    feedback = request["feedback"]
    chart = np.asarray(binding["T_world_owner"])
    function = np.asarray(hypothesis["T_world_function"])
    # Source functional events use the source skill owner chart. Task contact
    # patches may use a different source task chart, declared independently.
    # The same two transformations serve every representation and owner role.
    mapping = np.linalg.inv(chart) @ function @ np.linalg.inv(binding["L_source_owner_region"])
    relocation = mapping @ np.linalg.inv(binding["T_task_owner_source_owner"])
    owners = deepcopy(binding.get("other_contact_owner_bindings", {}))
    scene = deepcopy(binding["scene"])
    scene.update(reference_frame=feedback["reference_frame"], eef_pose=feedback["eef_pose"],
                 tool_id=feedback["tool_id"], tool_frame=feedback["tool_frame"])
    role = binding["owner_role"]
    scene[role+"_id"] = binding["owner_id"]
    owners[source_scene[role+"_id"]] = dict(owner_id=binding["owner_id"],
                                           T_current_source=relocation)
    if program["skill_kind"] == "grasp":
        goal = deepcopy(source_scene["grasp_goal"])
        goal.update(T_carrier_eef=np.linalg.inv(chart) @ np.asarray(hypothesis["T_world_tool"]),
            source=dict(kind="frozen_reference_interaction", hypothesis_id=hypothesis["id"],
                        source=binding["source"]))
        scene.update(carrier_pose=chart, initial_pose=chart,
                     region_transform=mapping, grasp_goal=goal)
    else:
        # The current scene already declares carrier correspondence, A, model
        # coordinates and observed support through their existing typed fields.
        # Region-owned full-frame event geometry uses that same local mapping.
        if role == "reference":
            scene["reference_pose"] = chart
        for event in program["demo"]["functional_events"]:
            current = scene["functional_bindings"][event["event_id"]]
            for owner_role in ("carrier", "reference"):
                current[owner_role+"_id"] = scene[owner_role+"_id"]
            frame_key = role+"_frame"
            if frame_key in event:
                current[frame_key] = mapping @ np.asarray(event[frame_key])
    # Candidate alignment would move the compiler's ordinary source reference.
    # The explicit continuous grounding reference already owns that placement.
    scene.pop("grasp_reference_prior", None)
    scene["region_binding_evidence"] = dict(source=binding["source"],
        hypothesis_id=hypothesis["id"], chart="Calibrated current observation chart")
    contacts = reexpress_contact_bindings(original["contact_bindings"],
        binding["current_material_contacts"], owners,
        source="Source task patches and official robot contact features; "+binding["source"])
    return dict(program=deepcopy(program), scene=scene, contact_bindings=contacts)


def bind(hypothesis, request):
    """Bind original task relations and one complete nominal reference.

    ``task`` contains the task compiler's typed source ``program`` and
    ``scene``; ``task_binding`` supplies current entity charts and resolved
    robot materials. The compiler retains task goals, events, release,
    attachments and process relations. ``reference_phase`` replaces only its
    nominal guidance path and preserves the compiler's task metadata.
    """
    from .direct_planner import bind_pose_objectives
    from .planning_problem import build_planning_problem, reference_phase
    from .planning_inputs import require_feedback_identity

    feedback = request["feedback"]
    require_feedback_identity(feedback)
    task = bind_task_inputs(hypothesis, request)
    names = tuple(feedback["arm_joint_names"])
    columns = [feedback["joint_names"].index(name) for name in names]
    problem = build_planning_problem(
        [dict(skill_id=request["skill_id"], program=deepcopy(task["program"]),
              scene=deepcopy(task["scene"]))],
        joint_names=names, start_q=np.asarray(feedback["start_q"])[columns],
        start_dq=np.asarray(feedback["start_dq"])[columns],
        **{key: feedback[key] for key in
           ("tool_link", "reference_frame", "base_pose", "tool_transform")})
    original_phase, = problem.phases
    source = dict(hypothesis_id=hypothesis["id"], source=hypothesis["source"],
                  region_frame=hypothesis["T_world_function"],
                  tool_reference_anchor=hypothesis["tool_reference_anchor_world"])
    phase = reference_phase(hypothesis["reference"], skill_id=request["skill_id"],
                            source=source)
    phase.metadata = {**original_phase.metadata, **phase.metadata}
    problem = replace(problem, phases=(phase,),
        source_timestamps=hypothesis["reference"]["eef_timestamps"],
        source_progress=phase.reference.progress)
    problem = bind_pose_objectives(problem, request["pose_objectives"])
    provenance = dict(builder="build_planning_problem + reference_phase", **source,
        task_geometry="Explicit typed source task declarations bound in current functional coordinates",
        reference_geometry="Complete source continuous path with its own tool-motion anchor",
        entity_geometry="Current RGB-D and semantic masks in declared calibrated owner charts",
        # BEGIN use the public measurement clock without a simulator-specific field.
        feedback_id=feedback["feedback_id"], start_time_s=feedback["timestamp_s"])
        # END coordinate, event and curve times retain their explicit source units.
    return problem, provenance


def plan(hypothesis, request):
    """Prepare one observed scene, call the direct backend once, and close it.

    Capture feedback places robot shapes for RGB-D self filtering. Current
    entrance feedback independently defines q/dq and the physical reference
    boundary. Both records retain their original identities and clocks. The
    supplied total host/device limits cover the explicit portfolio process
    root, including sibling workers; native scene admission uses that scope.
    """
    import torch
    import yaml
    from curobo.perception import MapperCfg
    from curobo.types import DeviceCfg
    from .direct_planner import DirectPreferenceBackend, ReferenceObjective
    from .execution import ExecutionLimits, FixedFingerTargets, continuation_execution_inputs
    from .planning import PlanningBudget
    from .planning_inputs import (finger_trajectory_factory, position_controlled_hand,
        owner_observation_inputs, save_observation_partitions, load_observed_owner,
        load_joint_reference)
    from .planning_observation import (build_scene_from_rgbd, scene_task_owner_ids,
                                       observation_resolutions)
    from .grounding_robot_geometry import (build_robot_validator, robot_self_filter_shapes,
                                            tool_geometry_names)
    from .robot_self_filter import classify_depth
    from .resource_admission import configure_host_process_root

    output = Path(request["output_directory"])
    output.mkdir(parents=True, exist_ok=True)
    report = dict(status="preparing", hypothesis_id=hypothesis["id"],
                  planning_called=False, execution_tested=False, wallclock={"events": {}})
    started = time.monotonic()

    def mark(name):
        report["wallclock"]["events"][name] = dict(
            unix_time_ns=time.time_ns(), elapsed_seconds=time.monotonic()-started)

    try:
        report["host_process_scope"] = configure_host_process_root(
            request.get("host_process_root_pid", os.getpid()))
        problem, provenance = bind(hypothesis, request)
        feedback, fixture = request["feedback"], request["robot_fixture"]
        task = bind_task_inputs(hypothesis, request)
        resources, mapper = request["resources"], request["mapper"]
        budget = PlanningBudget(**request["budget"])
        finger_targets = FixedFingerTargets.from_mapping(request["finger_targets"])
        execution_limits = ExecutionLimits.from_mapping(request["execution_limits"])
        continuation, rebind = request.get("continuation"), request.get("rebind")
        if continuation is not None:
            finger_targets, execution_limits = continuation_execution_inputs(
                continuation, finger_targets, execution_limits, rebind=rebind)
        measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
        mark("scene_input_read_started")
        observed_owners = tuple(load_observed_owner(path)
                                for path in request.get("observed_owners", ()))
        observations_in = request["observations"]
        views = [dict(id=identity) for identity in observations_in]
        mask_rows = [(owner, identity, mask) for identity, view in observations_in.items()
                     for owner, mask in view["owner_masks"].items()]
        owner_frames, masks = owner_observation_inputs(problem, task["scene"], views,
            mask_rows, source=str(hypothesis["source"]), observed_owners=observed_owners)
        task_owners = scene_task_owner_ids(task["scene"])
        report["owner_voxel_sizes_m"] = observation_resolutions(
            (*owner_frames, *(owner.owner_id for owner in observed_owners)),
            task_owner_ids=task_owners, environment_voxel_size=mapper["voxel_size"])

        # The fixed robot and installed hand retain their official reusable
        # shapes and component relationships. Only named feedback and calibrated
        # transforms place this robot; task entities use observed fields.
        robot = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
        mark("native_robot_load_started")
        validator = build_robot_validator(fixture,
            reference_frame=problem.reference_frame, T_reference_base=problem.base_pose,
            tool_frame=problem.tool_link, T_frame_tcp=problem.tool_transform)
        mark("native_robot_load_completed")
        report["independent_robot_model"] = validator.describe()
        robot["robot_cfg"]["kinematics"]["lock_joints"] = {
            name: measured[name] for name in feedback["finger_joint_names"]}
        factory = finger_trajectory_factory(problem, feedback, finger_targets,
            incoming_command=feedback["incoming_finger_command"],
            timestep=fixture["control_timestep_s"], ctrlrange=fixture["finger_control_ranges"],
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            acceleration_limits={name: float(robot["robot_cfg"]["kinematics"]["cspace"]["max_acceleration"])
                                 for name in feedback["finger_joint_names"]})
        torch.set_num_threads(resources["torch_cpu_threads"])
        torch.cuda.set_per_process_memory_fraction(resources["torch_memory_mib"]*1024**2 /
                                                  torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        hand, report["gripper"] = position_controlled_hand(robot, feedback, problem,
            device=DeviceCfg("cuda:0"), execution_limits=execution_limits,
            position_limits=fixture["position_limits"], velocity_limits=fixture["velocity_limits"],
            trajectory_factory=factory, fixture=fixture)
        tool_members = tool_geometry_names(fixture)
        from .planning_scene import ContactRule
        contact_input = task["contact_bindings"]
        contacts = tuple(ContactRule(**row) for row in contact_input["contacts"])
        if any(row.get("kind") == "grasp_contact_activation" for row in problem.conditions):
            from .contact_binding import bind_grasp_contact_rules
            bind_grasp_contact_rules(problem, contacts, contact_input["grasp_activations"])
        report["contact_binding"] = dict(source=contact_input["source"],
                                         rule_ids=[row.rule_id for row in contacts])
        report["requirement_consumption"] = problem.requirement_consumption
        capture = request["capture_feedback"]
        capture_fk = validator.checker.query(joint_names=capture["joint_names"],
            joint_positions=capture["start_q"], T_reference_base=capture["base_pose"],
            pairs=(), frames=())
        shapes = robot_self_filter_shapes(fixture, capture_fk["frame_poses"])
        observations = {}
        for identity, view in observations_in.items():
            filtered = classify_depth(observed_depth=view["depth"], K=np.asarray(view["K"]),
                T_world_camera=np.asarray(view["T_output_camera"]), geometries=shapes,
                padding_scale=mapper["self_padding_scale"],
                padding_offset_m=mapper["self_padding_offset_m"], source=mapper["self_filter_source"])
            folder = output/"observations"/identity
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder/"self-filter-labels.npy", filtered["self_filter"]["labels"])
            np.save(folder/"free-ray-depth-m.npy", filtered["self_filter"]["free_ray_depth_m"])
            _write_diagnostic_json(folder/"self-filter-report.json", dict(
                provenance=filtered["provenance"], observation_id=identity,
                feedback_source=capture["source"]))
            observations[identity] = dict(image=view["image"], depth=view["depth"],
                K=np.asarray(view["K"]), T_output_camera=np.asarray(view["T_output_camera"]),
                reference_frame=problem.reference_frame, self_filter=filtered["self_filter"],
                owner_masks={owner: masks[owner, identity] for owner in owner_frames})
            if "background_mask" in view:
                observations[identity]["background_mask"] = view["background_mask"]
        height, width = next(iter(observations.values()))["depth"].shape
        config = MapperCfg(extent_meters_xyz=tuple(mapper["extent"]), voxel_size=mapper["voxel_size"],
            extent_esdf_meters_xyz=tuple(mapper["extent"]), esdf_voxel_size=mapper["voxel_size"],
            grid_center=torch.tensor(mapper["center"]), truncation_distance=2*mapper["voxel_size"],
            image_height=height, image_width=width, depth_minimum_distance=.05,
            depth_maximum_distance=5., device="cuda:0")
        scene = build_scene_from_rgbd(observations, owner_frames, mapper_cfg=config,
            depth_error_m=mapper["depth_error"], reference_frame=problem.reference_frame,
            robot_model=validator.checker,
            fixed_geometry_names=tuple(row["geom"] for row in fixture["fixed_collision_geometries"]),
            tool_geometry_names=tool_members, gripper=hand, observed_owners=observed_owners,
            contacts=contacts, inherited_initial_state=request.get("inherited_initial_state"),
            task_owner_ids=task_owners, fine_regions=request.get("fine_regions", {}),
            unknown_policy=request["unknown_policy"],
            max_mapping_bytes=resources["mapping_memory_mib"]*1024**2,
            max_host_rss_bytes=resources["max_host_rss_bytes"],
            max_device_used_bytes=resources["max_device_used_bytes"])
        report["observation_build"] = scene.observation_report
        report["observation_partitions"] = save_observation_partitions(scene, output)
        np.savez_compressed(output/"observed-free.npz", known_free=scene.coverage.known_free,
            origin=scene.coverage.origin, voxel_size=np.asarray(scene.coverage.voxel_size),
            observation_ids=np.asarray(scene.coverage.observation_ids))
        scene.validate_problem(problem)
        # BEGIN preserve continuous path operands rather than sampled caches.
        from .execution_binding import plain
        _write_diagnostic_json(output/"problem.json", plain(problem))
        _write_diagnostic_json(output/"reference.json", plain(hypothesis["reference"]))
        # END composed rotation/translation expressions survive archival.
        _write_diagnostic_json(output/"provenance.json", provenance)
        mark("scene_preparation_completed")
        report["planning_called"] = True
        mark("planning_call_started")
        # BEGIN retain the caller's explicit solver CPU pool independently of mapping.
        torch.set_num_threads(resources["planner_cpu_threads"])
        # END both hypotheses receive the same thread and iteration policy.
        backend = DirectPreferenceBackend(robot, robot_validator=validator, budget=budget,
            reference_objective=ReferenceObjective(**request["reference_objective"]),
            initialization=request["initialization"])
        try:
            result = plan_and_record(problem, scene, robot, output/"planning",
                robot_validator=validator, budget=budget, backend=backend,
                execution_limits=execution_limits.describe(), provenance=provenance,
                joint_reference=load_joint_reference(request.get("joint_reference"), problem), continuation=continuation,
                rebind=rebind, request_ledger=request.get("request_ledger"))
            mark("planning_result_saved")
        finally:
            backend.close()
        report.update(status="completed", planning_success=result.success,
                      failure=result.diagnostics.get("failure"))
        return result
    except Exception as error:
        report.update(status="error", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
        raise
    finally:
        report["elapsed_seconds"] = time.monotonic()-started
        _write_diagnostic_json(output/"run-report.json", report)
