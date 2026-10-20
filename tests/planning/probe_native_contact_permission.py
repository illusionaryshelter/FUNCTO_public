"""Bounded source-backed robot trajectory with one declared local material permission.

All original source solids, collision spheres and self-collision rules remain.
Two additional spheres model a small synthetic rigid pad on the source TCP; only
its dedicated contact-feature sphere receives a material identity. This pad
is declared experiment geometry, not a claim about an actual robot tool.
The owner, patch and observation volume are fully specified synthetic inputs.

One complete V2 solve must pass with the finite material permission. The same
exported curve is then rejected without that permission, with a noncontact
palm obstruction owned by the same object, and with another explicit owner.
These are paired curve checks, not separate claims of global infeasibility.
No physical grasp, measured contact or actual Holding is supplied or inferred.
Future finger geometry is an explicit constant hypothesis; no actuator tracking is certified.
Run serially: two IK seeds, two trajectory seeds, N=8, 512 interpolation slots,
no search, at most two complete solves, and a 2 GiB Torch allocator ceiling.
The source plus added pad and native padding must fit the 640-sphere admission limit.
"""

# BEGIN native finite-material permission and paired export counterexamples.
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import resource
import sys
import time
import traceback

import numpy as np
from scipy.spatial.transform import Rotation

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def pose_vector(matrix):
    """Native cuboid/Pose ordering is xyz followed by wxyz."""
    q = Rotation.from_matrix(matrix[:3, :3]).as_quat()[[3, 0, 1, 2]]
    return [*matrix[:3, 3].tolist(), *q.tolist()]


# BEGIN one declared synthetic pad extends both source solids and native proxies.
def contact_pad_fixture(fixture, robot, validator):
    """Preserve every source solid/pair and append one exact TCP-frame box.

    This experiment requires an existing TCP frame with no original spheres,
    allowing new proxy indices to follow the source inventory unchanged. The
    pad is separated from all original solids by the complete new pair list.
    Its two spheres are optimizer/scene proxies, never a replacement solid.
    """
    from manipulation.robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker
    from manipulation.robot_validation import RobotStateValidator

    robot = deepcopy(robot)
    kin = robot["robot_cfg"]["kinematics"]
    tool = fixture["tool_link"]
    if tool in kin["collision_link_names"] or tool in kin["collision_spheres"]:
        raise ValueError("This added-pad experiment requires a source TCP frame without original spheres")
    additions = [{"center": [0., 0., .22], "radius": .005},
                 {"center": [0., 0., .226], "radius": .006}]
    tcp = validator.T_frame_tcp
    kin["collision_link_names"].append(tool)
    kin["collision_spheres"][tool] = [{"center": (tcp[:3, :3] @ row["center"]+tcp[:3, 3]).tolist(),
                                       "radius": row["radius"]} for row in additions]
    kin["self_collision_buffer"][tool] = 0.
    originals = []
    for row in validator.checker.model_record()["geometries"]:
        if row["kind"] == "convex":
            originals.append(NativeSolid(row["name"], row["frame"], np.asarray(row["vertices_frame"]), row["source"]))
        else:
            originals.append(NativePrimitive(row["name"], row["frame"], row["kind"], tuple(row["dimensions"]),
                                             np.asarray(row["T_frame_geometry"]), row["source"]))
    pad_name = "declared_synthetic_contact_pad"
    local = np.eye(4)
    local[:3, 3] = [0., 0., .221]
    source = "explicit synthetic TCP pad: exact 4 by 4 by 10 mm filled box"
    pad = NativePrimitive(pad_name, tool, "box", (.004, .004, .01), tcp @ local, source)
    checker = RobotGeometryChecker(validator.checker.urdf_path, base_frame=validator.checker.base_frame,
                                   geometries=[*originals, pad])
    pairs = (*validator.pairs, *(PairQuery(name, pad_name, "required", source)
                                for name in validator.checker.geometry_names))
    extended = RobotStateValidator(checker, joint_names=validator.joint_names,
        arm_joint_names=validator.arm_joint_names, finger_joint_names=validator.finger_joint_names,
        pairs=pairs, named_velocity_limits=validator.velocity_limits, reference_frame=validator.reference_frame,
        T_reference_base=validator.T_reference_base, tool_frame=validator.tool_frame, T_frame_tcp=tcp,
        fk_translation_tolerance_m=validator.fk_translation_tolerance_m,
        fk_rotation_tolerance_rad=validator.fk_rotation_tolerance_rad,
        source=f"{validator.source}; {source}")
    return robot, extended, additions, pad_name
# END original source geometry and physical pair permissions are unchanged.


def run(output, *, fixture_path):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "success": False,
              "scope": "complete source robot plus explicitly declared synthetic rigid contact pad",
              "production_environment_validation": False, "physical_contact_validation": False,
              "moving_gripper_validation": False, "actual_holding_evidence": False}
    try:
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.geom.types import Cuboid
        from manipulation.online_geometry import FrozenPrimitives, OnlinePlanningScene
        from manipulation.planning import PlanningBudget, plan_trajectory, _ProgressMap
        from manipulation.planning_problem import (MotionPhase, NumericalTolerance,
            PlanningProblem, PosePath, PoseProcessRequirement)
        from manipulation.planning_scene import (
            CollisionOwner, ContactRule, ObservedSpace, _local_spheres)
        from probe_robot_fixture import load_probe_fixture, fixed_probe_hand

        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires explicitly authorized CUDA access")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        # BEGIN explicit source fixture plus declared pad; the robot tree is unchanged.
        fixture, source_robot, source_validator = load_probe_fixture(fixture_path, reference_frame="robot_base")
        original_geometry = deepcopy(source_robot["robot_cfg"]["kinematics"]["collision_spheres"])
        robot, validator, additions, pad_name = contact_pad_fixture(fixture, source_robot, source_validator)
        kinematic_input = robot["robot_cfg"]["kinematics"]
        for link, original in original_geometry.items():
            if kinematic_input["collision_spheres"][link] != original:
                raise AssertionError("The experiment must preserve every original source sphere")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        names = tuple(kinematics.joint_names)
        if names != tuple(fixture["arm_joint_names"]):
            raise AssertionError("This fixed-finger hypothesis leaves only source arm joints active")
        initial = kinematics.default_joint_position.reshape(-1).clone()
        tool_index = kinematics.tool_frames.index(fixture["tool_link"])
        parameters = kinematics.config.kinematics_config
        pad_indices = parameters.get_sphere_index_from_link_name(fixture["tool_link"]).cpu().tolist()
        if len(pad_indices) != 2 or pad_indices != list(range(fixture["sphere_count"], fixture["sphere_count"]+2)):
            raise AssertionError("The declared pad must follow all unchanged source sphere indices")
        contact_index, backing_index = map(int, pad_indices)
        # END source installations and all original robot bodies retain their identities.

        def fk(coordinates):
            value = torch.as_tensor(coordinates, dtype=torch.float32, device=device.device)
            return kinematics.compute_kinematics(JointState.from_position(
                value[None].contiguous(), joint_names=list(names)))

        state = fk(initial[None])
        position = state.tool_poses.position[0, 0, tool_index].detach().cpu().numpy().astype(float)
        quaternion = state.tool_poses.quaternion[0, 0, tool_index].detach().cpu().numpy().astype(float)
        tool_pose = np.eye(4)
        tool_pose[:3, 3] = position
        tool_pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        tool_pose = tool_pose @ validator.T_frame_tcp
        position = tool_pose[:3, 3].copy()
        original_spheres = state.robot_spheres[0, 0].detach().cpu().numpy().astype(float)
        local_added = ((original_spheres[[contact_index, backing_index], :3]-position)
                       @ tool_pose[:3, :3])
        np.testing.assert_allclose(local_added, [row["center"] for row in additions], atol=2e-6, rtol=0)

        rigid_indices = [int(index) for link in original_geometry if link not in fixture["finger_link_names"]
            for index in parameters.get_sphere_index_from_link_name(link).cpu().tolist()]
        palm_index = min(rigid_indices, key=lambda index: np.linalg.norm(original_spheres[index, :3]-position))
        # The declared contact polygon is the small pad's bottom face. Its
        # outer sphere extends 1 mm past the plane; the separate backing
        # sphere covers noncontact tool volume. This limited proxy error is
        # what the permission models, rather than authorizing body overlap.
        # The declared 4x4x10 mm pad is inside the two-sphere union. A 2 mm
        # ball at the contact sphere's center lies inside that body, bounding
        # this sphere's outer-proxy error by 5-2=3 mm (plus a 0.1 mm guard).
        feature_points = [[x, y, .216] for x, y in ((-.002, -.002), (.002, -.002),
                                                   (.002, .002), (-.002, .002))]
        owner_pose = tool_pose.copy()
        owner_pose[:3, 3] = position+tool_pose[:3, :3]@np.array([0., 0., .206])
        palm_local = (original_spheres[palm_index, :3]-owner_pose[:3, 3])@owner_pose[:3, :3]
        second_pose = tool_pose.copy()
        second_pose[:3, 3] = original_spheres[contact_index, :3]

        # BEGIN explicit 1 mm contact cells bound this small synthetic patch.
        def query(identity, boxes):
            return FrozenPrimitives(primitives=[{"kind": "cuboid", "pose": box.pose,
                "dimensions": box.dims} for box in boxes], owner_ids=(identity,),
                contact_voxel_size_m=.001, source="explicit synthetic solid cuboids")
        # END paired scenes query the same frozen model and finite-cell rule.

        def owner(identity, pose, extra_palm_blocker=False):
            dimensions = [.016, .016, .02] if identity == "contact_owner" else [.002, .002, .002]
            boxes = [Cuboid(name=f"{identity}_body", pose=[0., 0., 0., 1., 0., 0., 0.], dims=dimensions)]
            if extra_palm_blocker:
                boxes.append(Cuboid(name="same_owner_noncontact_obstruction",
                                    pose=[*palm_local.tolist(), 1., 0., 0., 0.], dims=[.002, .002, .002]))
            return CollisionOwner(identity, query(identity, boxes),
                                  PosePath(np.array([0., 1.]), np.stack((pose, pose)), "declared static owner frame"),
                                  "fully specified synthetic cuboid geometry", True)

        background = query("far_background", [Cuboid(name="far_background", pose=[1.8, 1.8, 1.8, 1., 0., 0., 0.],
                                                       dims=[.02, .02, .02])])
        base_owner = owner("contact_owner", owner_pose)
        palm_owner = owner("contact_owner", owner_pose, extra_palm_blocker=True)
        other_owner = owner("other_owner", second_pose)
        rule = ContactRule(
            rule_id="declared_pad_bottom", actor="robot", target_owner="contact_owner", interval=(0., 1.),
            sphere_indices=(contact_index,), material_id="synthetic_pad_bottom",
            feature_points=feature_points, feature_normal=[0., 0., -1.],
            patch_origin=[0., 0., .01], patch_axes=np.eye(3), patch_half_size=[.008, .008],
            gap_bounds=(-5e-5, 5e-5), normal_tolerance_rad=.001,
            modeling_allowance_m=.0012, geometry_error_m=.0031, minimum_thickness_m=.004,
            source="declared pad material polygon, finite owner patch and bounded proxy error", required=True)
        materials = [None]*kinematics.total_spheres
        materials[contact_index] = "synthetic_pad_bottom"
        coverage = ObservedSpace(np.ones((81, 81, 81), bool), origin=[-2., -2., -2.], voxel_size=.05,
                                 observation_ids=("explicit synthetic known-space grid",), reference_frame="robot_base")

        # BEGIN the static contact fixture states its true pose process separately.
        contact_reference = PosePath(np.array([0., 1.]), np.stack((tool_pose, tool_pose)),
                                     "synthetic static contact reference")
        problem = PlanningProblem(
            names, initial.detach().cpu().numpy(), np.zeros(len(names)), fixture["tool_link"], "robot_base", np.eye(4), validator.T_frame_tcp,
            (MotionPhase("contact_reference", contact_reference),),
            numerical=NumericalTolerance(position_m=1e-5, rotation_rad=1e-4),
            pose_relations=(PoseProcessRequirement(0., 1., contact_reference, np.zeros((6, 2)),
                "Explicit fixture keeps one constant tool pose while independently changing contact permissions"),),
            source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))
        # END permission and required contact remain independent scene relations.
        full_current = np.asarray(kinematic_input["cspace"]["default_joint_position"], float)
        for name, value in zip(names, initial.cpu().numpy()):
            full_current[fixture["joint_names"].index(name)] = float(value)
        hand, installations, hand_report = fixed_probe_hand(
            fixture, robot, problem, device=device, full_current_positions=full_current)
        report["nominal_hand_hypothesis"] = hand_report

        def scene(owners, permissions):
            return OnlinePlanningScene(unknown_policy={"outside_regions": "forbid", "conditional_regions": []}, background=background, owners=(*installations, *owners), contacts=permissions, coverage=coverage,
                                 reference_frame="robot_base", robot_sphere_materials=materials,
                                 # BEGIN preserve the installation filters' exact source identity.
                                 robot_geometry_source=fixture["robot_geometry_source"],
                                 # END the complete validator record separately identifies the added pad.
                                 gripper=deepcopy(hand), native_geometry_materials={pad_name: "synthetic_pad_bottom"})

        scenes = {"permitted": scene((base_owner,), (rule,)),
                  "permission_removed": scene((base_owner,), ()),
                  "same_owner_noncontact_obstruction": scene((palm_owner,), (rule,)),
                  "different_owner_obstruction": scene((base_owner, other_owner), (rule,))}
        progress = torch.zeros((1, 1), dtype=torch.float32, device=device.device)
        owner_position, owner_quaternion = base_owner.predicted_pose(progress)
        raw = base_owner.query(_local_spheres(state.robot_spheres, owner_position, owner_quaternion), key="raw_fixture")
        raw_values = raw.detach().cpu().numpy()[0, 0]
        if not .0005 < raw_values[contact_index] < .0015:
            raise AssertionError("The declared original material proxy must actually penetrate the native owner field")
        if np.max(np.delete(raw_values, contact_index)) > 1e-7:
            raise AssertionError("The positive fixture must keep every original/noncontact sphere clear")
        report["declared_geometry"] = {
            "original_spheres_retained": sum(len(rows) for rows in original_geometry.values()),
            "native_total_spheres_including_disabled_slots": kinematics.total_spheres,
            "added_pad_spheres_tool_frame": additions, "contact_sphere_index": contact_index,
            "declared_pad_body_dimensions_m": [.004, .004, .01],
            "declared_pad_body_center_tool_frame": [0., 0., .221],
            "unmodified_backing_sphere_index": backing_index, "noncontact_palm_sphere_index": palm_index,
            "feature_points_tool_frame": feature_points, "owner_pose": pose_vector(owner_pose),
            "original_penetration_m": float(raw_values[contact_index]),
            "modeling_allowance_m": rule.modeling_allowance_m,
            "geometry_error_m": rule.geometry_error_m, "minimum_thickness_m": rule.minimum_thickness_m}

        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=2, knots=8, ik_iterations=80,
                                trajectory_iterations=150, interpolation_buffer_size=512,
                                # BEGIN include both added pad spheres and native disabled slots.
                                max_robot_spheres=kinematics.total_spheres,
                                # END allocator and solver workspace limits remain unchanged.
                                maximum_trajectory_solves=2, search_seconds=0.)
        report["budget"] = asdict(budget)
        result = plan_trajectory(problem, scenes["permitted"], robot, budget=budget, robot_validator=validator)
        report["diagnostics"] = result.diagnostics
        if not result.success or result.trajectory is None:
            raise AssertionError("Complete native optimization and exported validation must accept the permitted fixture")
        curve = result.trajectory
        times = np.linspace(0., curve.duration, 401)
        # BEGIN explicit acceleration slot keeps progress in its own channel.
        q, dq, _, s = curve.sample(times)
        # END permission queries retain their original shared clock.
        native = fk(q)
        progress = device.to_device(s)[None]
        report["paired_export_checks"] = {}
        for label, world in scenes.items():
            # Each paired scene uses the exported clock and the same constant
            # nominal fingers; binding cannot turn that hypothesis into feedback.
            world.gripper.bind_clock(curve.duration, _ProgressMap(curve.progress_u, curve.progress_s))
            world.validate_problem(problem)
            model = result.execution_binding["scene"]
            world.bind_online_robot(**{key: model[key] for key in (
                "robot_sphere_geometry", "robot_self_collision_pairs", "robot_self_collision_padding", "robot_model")})
            tool_p, tool_r = problem.transform_tool_pose(native.tool_poses.position[:, :, tool_index],
                                                         native.tool_poses.quaternion[:, :, tool_index])
            terms = world.components(problem.transform_robot_spheres(native.robot_spheres), tool_p, tool_r, progress,
                                     joint_position=device.to_device(q)[None], joint_names=names,
                                     trajectory_time=device.to_device(times)[None])
            total = torch.stack(tuple(terms.values())).sum(0)
            maxima = {name: float(value.max().detach().cpu()) for name, value in terms.items()}
            report["paired_export_checks"][label] = {"maximum_by_component": maxima,
                                                     "minimum_violation": float(total.min().detach().cpu()),
                                                     "maximum_violation": float(total.max().detach().cpu())}
            if label == "permitted" and bool((total > 0).any()):
                raise AssertionError("The same exported curve must retain its material permission at every queried point")
            if label != "permitted" and float(total.min().detach().cpu()) <= .0004:
                raise AssertionError(f"The paired changed scene must reject the same complete curve: {label}")
        report["exported_motion"] = {"duration_s": curve.duration, "joint_names": list(curve.joint_names),
                                     "residuals": curve.residuals, "paired_query_points": len(times),
                                     "events": list(curve.events), "conditions": list(curve.conditions)}
        np.savez(output/"trajectory.npz", times=curve.times, position=curve.position, velocity=curve.velocity,
                 progress_u=curve.progress_u, progress_s=curve.progress_s)
        report.update(status="passed", success=True,
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved())
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (output/"exception.txt").write_text(traceback.format_exc())
        traceback.print_exc()
    finally:
        report.update(elapsed_seconds=time.monotonic()-started, peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        (output/"report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    options = parser.parse_args()
    run(options.output, fixture_path=options.fixture)
# END native contact-permission integration experiment.
