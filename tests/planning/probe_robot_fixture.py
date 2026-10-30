"""Source-backed robot inputs for the bounded planning experiments.

The fixture adapter verifies native solids and pair policy. This module only
loads those explicit inputs, calibrates their existing sphere proxies, and
declares the fixed-finger hypothesis used by synthetic rigid-tool experiments.
It does not infer a filled solid from a sphere cover or a triangle surface.
"""

# BEGIN shared test adapters reuse the recorded source model and hand calibration.
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import yaml


def load_probe_fixture(fixture_path, *, reference_frame, T_reference_base=None,
                       full_current_positions=None):
    """Return ``fixture, robot, validator`` for one explicit source manifest.

    A supplied full position vector uses ``fixture['joint_names']`` order and
    is retained exactly. Only synthetic probes may omit it to declare the
    source configuration's default posture as their constructed initial state.
    The returned native configuration locks each independent finger at that
    posture; a separate explicit nominal hand model must accompany planning.
    """
    from robot_fixture import _pose
    from robot_validation_fixture import build_robot_validator

    fixture = json.loads(Path(fixture_path).read_text())
    robot = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
    kin = robot["robot_cfg"]["kinematics"]
    names = tuple(fixture["joint_names"])
    if tuple(kin["cspace"]["joint_names"]) != names:
        raise ValueError("Probe source configuration and fixture must retain the same named joint order")
    positions = np.asarray(kin["cspace"]["default_joint_position"]
                           if full_current_positions is None else full_current_positions, dtype=float)
    if positions.shape != (len(names),) or not np.isfinite(positions).all():
        raise ValueError("Probe entrance requires one finite position per complete source joint")
    base = np.eye(4) if T_reference_base is None else np.asarray(T_reference_base, dtype=float)
    transform = fixture["tool_transform"]
    validator = build_robot_validator(fixture, reference_frame=reference_frame,
        T_reference_base=base, tool_frame=fixture["tool_link"],
        T_frame_tcp=_pose(transform["position"], transform["quaternion_wxyz"]))
    limits = np.asarray([validator.checker.joint_limits[name] for name in names])
    if np.any(positions < limits[:, 0]) or np.any(positions > limits[:, 1]):
        raise ValueError("Constructed probe entrance must stay inside source mechanical limits")
    kin["lock_joints"] = {name: float(positions[names.index(name)]) for name in fixture["finger_joint_names"]}
    kin["cspace"]["default_joint_position"] = positions.tolist()
    return fixture, robot, validator


def calibrate_native_hand(robot, feedback, problem, *, device, position_limits, fixture):
    """Measure the existing native hand's affine sphere geometry by named FK.

    This is the recorded runner's calibration: query both actual source limits,
    derive each independent prismatic translation, then check the actual named
    entrance. No symmetric-width assumption or finger-name convention enters
    the geometry. Returned arrays own CPU copies of reusable native buffers.
    """
    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import JointState
    from robot_fixture import bind_fixture_gripper_geometry, bind_fixture_prismatic_self_geometry

    unlocked = deepcopy(robot)
    unlocked["robot_cfg"]["kinematics"]["lock_joints"] = None
    native = Kinematics(KinematicsCfg.from_robot_yaml_file(unlocked, device_cfg=device))
    names = tuple(native.joint_names)
    measured = dict(zip(feedback["joint_names"], feedback["start_q"]))
    finger_names = tuple(feedback["finger_joint_names"])
    columns = [names.index(name) for name in finger_names]
    if len(columns) != 2:
        raise ValueError("This parallel-jaw calibration requires two explicitly named native fingers")
    parameters = native.config.kinematics_config
    indices_by_link = {link: [int(index) for index in
        parameters.get_sphere_index_from_link_name(link).cpu().tolist()]
        for link in robot["robot_cfg"]["kinematics"]["collision_link_names"]}
    # BEGIN include the complete verified hand subtree, including fixed palm.
    binding = bind_fixture_gripper_geometry(fixture, robot, indices_by_link, finger_names)
    indices = tuple(binding['sphere_indices'])
    # END native solid replacement is never based on only convenient fingers.
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Native finger links must have disjoint nonempty sphere sets")
    limits = np.asarray([position_limits[name] for name in finger_names], dtype=float)
    initial = np.array([measured[name] for name in names])
    values = np.repeat(initial[None], 3, axis=0)
    values[:, columns] = limits[:, 0]
    for index, column in enumerate(columns):
        values[index+1, column] = limits[index, 1]
    query = device.to_device(values)[None].contiguous()
    with torch.no_grad():
        state = native.compute_kinematics(JointState.from_position(query, joint_names=list(names)))
        balls = problem.transform_robot_spheres(state.robot_spheres)[0].cpu().numpy().copy()
        tool_index = native.tool_frames.index(problem.tool_link)
        tool_p, tool_q = problem.transform_tool_pose(
            state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
        p, q = tool_p[0].cpu().numpy().copy(), tool_q[0].cpu().numpy().copy()
    local = np.stack([Rotation.from_quat(r[[1, 2, 3, 0]]).inv().apply(row[:, :3]-x)
                      for row, x, r in zip(balls, p, q)])[:, indices]
    spans = limits[:, 1]-limits[:, 0]
    if np.any(spans <= 0):
        raise ValueError("Calibrated prismatic fingers need positive native mechanical ranges")
    affine = np.stack([(local[index+1]-local[0])/spans[index]
                       for index in range(len(columns))], axis=-1)
    zero = local[0]-np.einsum("ncj,j->nc", affine, limits[:, 0])
    local_spheres = np.c_[zero, balls[0, indices, 3]]
    with torch.no_grad():
        state = native.compute_kinematics(JointState.from_position(
            device.to_device(initial)[None, None], joint_names=list(names)))
        current = problem.transform_robot_spheres(state.robot_spheres)[0, 0].cpu().numpy().copy()
        tool_p, tool_q = problem.transform_tool_pose(
            state.tool_poses.position[:, :, tool_index], state.tool_poses.quaternion[:, :, tool_index])
        p, q = tool_p[0, 0].cpu().numpy().copy(), tool_q[0, 0].cpu().numpy().copy()
    current_local = Rotation.from_quat(q[[1, 2, 3, 0]]).inv().apply(current[list(indices), :3]-p)
    predicted = zero+np.einsum("ncj,j->nc", affine, initial[columns])
    error = float(np.max(np.abs(current_local-predicted)))
    if error > 2e-6:
        raise ValueError(f"Native finger geometry is not affine in the declared TCP: {error} m")
    pairs = native.get_self_collision_config().collision_pairs.cpu().numpy()
    self_pairs = tuple(tuple(int(i) for i in pair) for pair in pairs if set(pair)&set(indices))
    result = {"joint_names": names, "finger_joint_names": finger_names,
        "current_positions": initial[columns], "joint_limits": limits,
        "sphere_indices": indices, "local_spheres": local_spheres,
        "sphere_joint_translation": affine, "self_collision_pairs": self_pairs,
        "affine_fk_max_error_m": error, "robot_spheres": current,
        "tool_position": p, "tool_quaternion": q,
        "sphere_indices_by_link": indices_by_link}
    binding = bind_fixture_prismatic_self_geometry(fixture, robot, result, feedback)
    result['self_collision_geometry'] = binding['geometry']
    result['local_spheres'] = binding['local_spheres']
    result['sphere_joint_translation'] = binding['sphere_joint_translation']
    result['native_geometry_binding'] = binding['audit']
    # BEGIN canonical source coefficients must still reproduce measured GPU FK.
    predicted = binding['local_spheres'][:, :3]+np.einsum(
        'ncj,j->nc', binding['sphere_joint_translation'], initial[columns])
    result['affine_fk_max_error_m'] = float(np.max(np.abs(current_local-predicted)))
    if result['affine_fk_max_error_m'] > 2e-6:
        raise ValueError('Canonical native hand model differs from measured GPU FK')
    # END the original metre-error contract is unchanged.
    return result


def fixed_probe_hand(fixture, robot, problem, *, device, full_current_positions,
                     control_timestep_s=.002):
    """Declare constant future finger geometry and preserve native installations.

    Existing command intentions and the control period remain explicit inputs.
    An arm-only experiment has no finger command event; the nominal geometry
    hypothesis does not manufacture an actuator action merely to bind a clock.
    The constant nominal trajectory is a hypothesis, including for the mounted
    measured-state probe; it makes no claim of actual actuator tracking or
    mechanically fixed hardware. Source mechanical ranges remain unchanged.
    """
    from manipulation.gripper_trajectory import FingerTrajectory, stationary_finger_command
    from manipulation.planning_scene import PrismaticGripper
    from robot_fixture import create_fixed_installation_owners

    names = tuple(fixture["joint_names"])
    values = np.asarray(full_current_positions, dtype=float).copy()
    if values.shape != (len(names),) or not np.isfinite(values).all():
        raise ValueError("Fixed-hand hypothesis requires the complete finite named initial posture")
    fingers = tuple(fixture["finger_joint_names"])
    columns = [names.index(name) for name in fingers]
    current = values[columns]
    kin = robot["robot_cfg"]["kinematics"]
    if kin["lock_joints"] != dict(zip(fingers, current.tolist())):
        raise ValueError("Native lock parameters and declared fixed-hand entrance must agree exactly")
    feedback = {"joint_names": names, "start_q": values, "finger_joint_names": fingers,
                "finger_link_names": fixture["finger_link_names"]}
    geometry = calibrate_native_hand(robot, feedback, problem, device=device,
                                     position_limits=fixture["position_limits"], fixture=fixture)
    commands = tuple({"event_id": event.event_id, "skill_id": event.skill_id,
        "occurrence": event.occurrence, "command": event.kind, "progress": event.progress}
        for event in problem.events if event.kind in {"hold", "open", "close"})
    speed = np.asarray([fixture["velocity_limits"][name] for name in fingers], dtype=float)
    acceleration = np.asarray(kin["cspace"]["max_acceleration"], dtype=float)
    if acceleration.ndim == 0:
        acceleration = np.full(len(names), float(acceleration))
    acceleration = acceleration[[kin["cspace"]["joint_names"].index(name) for name in fingers]]
    source = "explicit constant nominal finger geometry hypothesis; actual actuator tracking is not certified"

    def nominal(duration, progress_map, command_reference):
        # This continuous hypothesis has no future command-derived transitions.
        # Any actual command intentions remain in the separate formal event clock.
        reference = FingerTrajectory.build(joint_names=fingers, measured_initial_position=current,
            measured_initial_velocity=np.zeros(len(fingers)),
            incoming_command=stationary_finger_command(joint_names=fingers, position=current,
                timestamp_s=0., feedback_id="explicit named initial posture for a fixed-geometry probe",
                source="Explicit constant synthetic geometry coordinates"), commands=(), duration=duration,
            executable_until=duration, timestep=control_timestep_s,
            position_limits=geometry["joint_limits"].T, velocity_limits=np.stack((-speed, speed)),
            acceleration_limits=np.stack((-acceleration, acceleration)), source=source,
            feedback_id="explicit named initial posture for a fixed-geometry probe", timestamp_s=0.)
        # BEGIN this synthetic fixed shape is explicitly modeled for the full probe.
        from manipulation.gripper_trajectory import NominalFingerGeometry
        return NominalFingerGeometry.from_reference(reference, valid_until_s=duration, source=source)
        # END no default whole-clock geometry contract is inferred for other hands.

    hand = PrismaticGripper(controller="binary", joint_names=fingers, current_positions=current,
        joint_limits=geometry["joint_limits"], joint_bounds=None,
        coupling_matrix=np.empty((0, len(fingers))), coupling_offset=np.empty(0),
        sphere_indices=geometry["sphere_indices"], local_spheres=geometry["local_spheres"],
        sphere_joint_translation=geometry["sphere_joint_translation"],
        self_collision_pairs=geometry["self_collision_pairs"], commands=commands,
        source=source, control_timestep_s=control_timestep_s, nominal_finger_trajectory_factory=nominal,
        self_collision_geometry=geometry['self_collision_geometry'])
    original_links = {row["link"] for row in fixture["collision_geometries"]}
    owners, owner_report = create_fixed_installation_owners(fixture, device_cfg=device,
        T_output_base=problem.base_pose, sphere_indices_by_link={name: indices for name, indices in
            geometry["sphere_indices_by_link"].items() if name in original_links})
    report = {"source": source, "current_positions": current.tolist(),
        "joint_names": list(fingers), "control_timestep_s": float(control_timestep_s),
        "affine_fk_max_error_m": geometry["affine_fk_max_error_m"],
        "native_geometry_binding": geometry['native_geometry_binding'],
        "fixed_installation_owners": owner_report}
    return hand, owners, report
# END no source geometry or collision policy is invented by these probe adapters.


# BEGIN declared empty online background for synthetic planning probes only.
def empty_background(coverage):
    """No observed occupied sites inside this explicitly declared free fixture."""
    from manipulation.online_geometry import FrozenESDF
    # BEGIN the synthetic free fixture explicitly supplies no observed material.
    from manipulation.online_observed_geometry import FrozenObservedGeometry
    from manipulation.tsdf_geometry import tsdf_geometry_from_samples
    empty = tsdf_geometry_from_samples(np.ones((1, 1, 1), np.float32), np.zeros((1, 1, 1), bool),
        origin=coverage.origin, voxel_size=coverage.voxel_size, name='empty', source='synthetic empty source')
    return FrozenESDF.from_source(FrozenObservedGeometry(empty['solids'][0]),
        origin=coverage.origin, shape=coverage.known_free.shape, voxel_size=coverage.voxel_size, owner_ids=(),
        source="declared synthetic empty background; no recorded observation claim")
    # END neither an arbitrary field value nor coverage mask creates a solid.
# END physical unknown-space policy remains the separate finite coverage grid.
