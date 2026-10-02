"""Bounded native Franka trajectory with one declared local material permission.

All original Franka collision spheres and native self-collision rules remain.
Two additional spheres model a small synthetic rigid pad on panda_hand; only
its dedicated contact-feature sphere receives a material identity. This pad
is declared experiment geometry, not a claim about an actual Franka tool.
The owner, patch and observation volume are fully specified synthetic inputs.

One complete V2 solve must pass with the finite material permission. The same
exported curve is then rejected without that permission, with a noncontact
palm obstruction owned by the same object, and with another explicit owner.
These are paired curve checks, not separate claims of global infeasibility.
No physical grasp, measured contact or actual Holding is supplied or inferred.
The native fingers stay mechanically locked; no moving-gripper claim is made.
Run serially: two IK seeds, two trajectory seeds, N=8, 512 interpolation slots,
no search, at most two complete solves, and a 2 GiB Torch allocator ceiling.
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


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = {"status": "running", "success": False,
              "scope": "complete native Franka plus explicitly declared synthetic rigid contact pad",
              "production_environment_validation": False, "physical_contact_validation": False,
              "moving_gripper_validation": False, "actual_holding_evidence": False}
    try:
        import torch
        from curobo.content import get_robot_configs_path
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import DeviceCfg, JointState
        from curobo._src.util.config_io import load_yaml, join_path
        from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
        from curobo._src.geom.types import Cuboid, SceneCfg
        from manipulation.planning import PlanningBudget, plan_trajectory
        from manipulation.planning_problem import (MotionPhase, NumericalTolerance,
            PlanningProblem, PosePath, PoseProcessRequirement)
        from manipulation.planning_scene import (
            CollisionOwner, ContactRule, CuroboSphereQuery, ObservedSpace, PlanningScene, _local_spheres)

        if not torch.cuda.is_available():
            raise RuntimeError("This experiment requires explicitly authorized CUDA access")
        torch.set_num_threads(2)
        torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        device = DeviceCfg("cuda:0")
        robot = deepcopy(load_yaml(join_path(get_robot_configs_path(), "franka.yml")))
        kinematic_input = robot["robot_cfg"]["kinematics"]
        original_geometry = deepcopy(kinematic_input["collision_spheres"])
        original_hand_count = len(original_geometry["panda_hand"])
        additions = [{"center": [0., 0., .22], "radius": .005},
                     {"center": [0., 0., .226], "radius": .006}]
        kinematic_input["collision_spheres"]["panda_hand"].extend(deepcopy(additions))
        # Keep the complete original model verbatim. The extra noncontact
        # backing sphere remains unmodified even against the permitted owner.
        for link, original in original_geometry.items():
            if kinematic_input["collision_spheres"][link][:len(original)] != original:
                raise AssertionError("The experiment must not remove or alter original Franka spheres")
        kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(deepcopy(robot), device_cfg=device))
        names = tuple(kinematics.joint_names)
        if len(names) != 7 or kinematic_input["lock_joints"] != {
                "panda_finger_joint1": .04, "panda_finger_joint2": .04}:
            raise AssertionError("This rigid-tool experiment must retain the explicit native locked fingers")
        initial = kinematics.default_joint_position.reshape(-1).clone()
        hand_indices = kinematics.config.kinematics_config.get_sphere_index_from_link_name("panda_hand").cpu().tolist()
        if len(hand_indices) != original_hand_count+2:
            raise AssertionError("Native sphere ownership must preserve the declared contact-pad additions")
        contact_index, backing_index = map(int, hand_indices[-2:])
        palm_index = int(hand_indices[0])

        def fk(coordinates):
            value = torch.as_tensor(coordinates, dtype=torch.float32, device=device.device)
            return kinematics.compute_kinematics(JointState.from_position(
                value[None].contiguous(), joint_names=list(names)))

        state = fk(initial[None])
        position = state.tool_poses.position[0, 0, 0].detach().cpu().numpy().astype(float)
        quaternion = state.tool_poses.quaternion[0, 0, 0].detach().cpu().numpy().astype(float)
        tool_pose = np.eye(4)
        tool_pose[:3, 3] = position
        tool_pose[:3, :3] = Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix()
        original_spheres = state.robot_spheres[0, 0].detach().cpu().numpy().astype(float)
        local_added = ((original_spheres[[contact_index, backing_index], :3]-position)
                       @ tool_pose[:3, :3])
        np.testing.assert_allclose(local_added, [row["center"] for row in additions], atol=2e-6, rtol=0)

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

        def query(identity, boxes):
            native = SceneCollision.from_config(SceneCollisionCfg(
                device_cfg=device, scene_model=SceneCfg(cuboid=boxes), cache={"cuboid": len(boxes)}))
            return CuroboSphereQuery(native, owner_ids=(identity,), source="explicit synthetic solid cuboids")

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

        def scene(owners, permissions):
            return PlanningScene(background=background, owners=owners, contacts=permissions, coverage=coverage,
                                 reference_frame="robot_base", robot_sphere_materials=materials,
                                 robot_geometry_source="unchanged complete Franka model plus two declared rigid pad spheres",
                                 has_gripper=False)

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

        # BEGIN the static contact fixture states its true pose process separately.
        contact_reference = PosePath(np.array([0., 1.]), np.stack((tool_pose, tool_pose)),
                                     "synthetic static contact reference")
        problem = PlanningProblem(
            names, initial.detach().cpu().numpy(), np.zeros(len(names)), "panda_hand", "robot_base", np.eye(4), np.eye(4),
            (MotionPhase("contact_reference", contact_reference),),
            numerical=NumericalTolerance(position_m=1e-5, rotation_rad=1e-4),
            pose_relations=(PoseProcessRequirement(0., 1., contact_reference, np.zeros((6, 2)),
                "Explicit fixture keeps one constant tool pose while independently changing contact permissions"),),
            source_timestamps=np.array([0., 1.]), source_progress=np.array([0., 1.]))
        # END permission and required contact remain independent scene relations.
        budget = PlanningBudget(ik_seeds=2, trajectory_seeds=2, knots=8, ik_iterations=80,
                                trajectory_iterations=150, interpolation_buffer_size=512,
                                maximum_trajectory_solves=2, search_seconds=0.)
        report["budget"] = asdict(budget)
        result = plan_trajectory(problem, scenes["permitted"], robot, budget=budget)
        report["diagnostics"] = result.diagnostics
        if not result.success or result.trajectory is None:
            raise AssertionError("Complete native optimization and exported validation must accept the permitted fixture")
        curve = result.trajectory
        times = np.linspace(0., curve.duration, 401)
        q, dq, s = curve.sample(times)
        native = fk(q)
        progress = device.to_device(s)[None]
        report["paired_export_checks"] = {}
        for label, world in scenes.items():
            terms = world.components(native.robot_spheres, native.tool_poses.position[:, :, 0],
                                     native.tool_poses.quaternion[:, :, 0], progress,
                                     joint_position=device.to_device(q)[None], joint_names=names)
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
    run(parser.parse_args().output)
# END native contact-permission integration experiment.
