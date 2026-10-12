"""Bounded CPU representation oracle for the fixed robot-collision suffix.

SciPy constructs the actual native uniform cubic basis; yourdfpy evaluates
the already installed Franka URDF without meshes, CUDA, or IK. The declared
suffix and numerical limits match probe_robot_collision.py. This witness
only establishes suffix representation capacity, not collision-free global
planning, native float32 execution, or a new production initializer.
"""

# BEGIN independent native-basis suffix-capacity oracle.
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial.transform import Rotation, Slerp
from yourdfpy import URDF
import yaml

# BEGIN resolve imports from this FUNCTO checkout, independent of its directory name.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
# END project-local imports.


def run(output):
    import curobo
    from manipulation.planning import _ProgressMap, _progress_for_cubic_boundaries

    contents = Path(curobo.__file__).resolve().parent / "content"
    configuration = yaml.safe_load((contents / "configs/robot/franka.yml").read_text())["robot_cfg"]["kinematics"]
    model = URDF.load(contents / "assets" / configuration["urdf_path"],
                      build_scene_graph=True, build_collision_scene_graph=False,
                      load_meshes=False, load_collision_meshes=False)
    joint_names = configuration["cspace"]["joint_names"][:7]
    measured = np.asarray(configuration["cspace"]["default_joint_position"][:7], dtype=float)

    def fk(joints):
        model.update_cfg(dict(zip(joint_names, joints)))
        return model.get_transform("panda_hand", frame_from="panda_link0").copy()

    # These are the unchanged synthetic fixture's two required endpoints.
    # Its preceding approach is free and is deliberately not certified here.
    entrance = measured.copy()
    entrance[0] += 1.5
    finish = entrance.copy()
    finish[0] += .1
    finish[1] -= .02
    reference = np.stack((fk(entrance), fk(finish)))
    difference = finish - entrance
    count = 12
    parameter_controls = np.arange(1, count + 1, dtype=float) / count
    actions = entrance + (2 * parameter_controls[:, None] - 1) * difference
    # At u=.5 only free controls 5..8 contribute. Continuing the suffix's
    # affine q(z) relation from control 5 makes every later cubic piece share
    # that same relation. The earlier four controls connect to measured q0;
    # their obstacle clearance is outside this representation-only claim.
    actions[:4] = measured + np.arange(1, 5)[:, None] / 5 * (actions[4] - measured)
    coefficients = np.concatenate((np.repeat(measured[None], 4, axis=0), actions,
                                   np.repeat(actions[-1, None], 3, axis=0)))
    curve = BSpline(np.arange(-3., count + 8.), coefficients, 3)
    mapping, warp = _progress_for_cubic_boundaries(
        _ProgressMap(np.array([0., 1.]), np.array([0., 2.])), count,
        stop_progress=(), anchors=(0., 1., 2.))
    query = np.linspace(.5, 1., 2001)
    progress = np.interp(query, mapping.u, mapping.s) - 1
    joints = curve(query * (count + 4))
    actual = np.stack([fk(value) for value in joints])
    ref_position = reference[0, :3, 3] + progress[:, None] * (reference[1, :3, 3] - reference[0, :3, 3])
    ref_rotation = Slerp([0., 1.], Rotation.from_matrix(reference[:, :3, :3]))(progress)
    position_error = ref_rotation.inv().apply(actual[:, :3, 3] - ref_position)
    rotation_error = (ref_rotation.inv() * Rotation.from_matrix(actual[:, :3, :3])).as_rotvec()

    # A conservative continuous bound closes the gaps between CPU samples.
    # Every revolute Jacobian translation column is bounded by the sum of
    # rigid offsets along the tool chain; its angular column has norm one.
    # On this suffix, q(z) is affine and z'(native_t)<=1/N because derivative
    # B-spline weights are nonnegative and sum to one. The triangle inequality
    # then bounds the error between each sample and its nearest neighbour.
    child_joint = {joint.child: joint for joint in model.joint_map.values()}
    link, reach = "panda_hand", 0.
    while link != "panda_link0":
        joint = child_joint[link]
        if joint.type not in ("fixed", "revolute", "continuous"):
            raise ValueError("This analytic reach bound is for the supplied revolute tool chain")
        if joint.origin is not None:
            reach += float(np.linalg.norm(joint.origin[:3, 3]))
        link = joint.parent
    max_q_l1_rate = 2 * float(np.abs(difference).sum()) * (count + 4) / count
    max_progress_rate = float(np.max(np.diff(mapping.s) / np.diff(mapping.u)))
    reference_angle = float((Rotation.from_matrix(reference[0, :3, :3]).inv()
                             * Rotation.from_matrix(reference[1, :3, :3])).magnitude())
    half_step = float(np.diff(query).max()) / 2
    position_gap_bound = half_step * (reach * max_q_l1_rate
                                     + np.linalg.norm(reference[1, :3, 3] - reference[0, :3, 3]) * max_progress_rate)
    rotation_gap_bound = half_step * (max_q_l1_rate + reference_angle * max_progress_rate)
    position_norm = float(np.linalg.norm(position_error, axis=1).max())
    rotation_norm = float(np.linalg.norm(rotation_error, axis=1).max())
    continuous_position_bound = position_norm + position_gap_bound
    continuous_rotation_bound = rotation_norm + rotation_gap_bound
    report = {
        "scope": "CPU continuous suffix-capacity witness; no global collision, native execution, or production seed claim",
        "native_free_controls": count,
        "sample_count": len(query),
        "reference_position_component_tolerance_m": .001,
        "reference_rotation_component_tolerance_rad": .005,
        "maximum_position_component_error_m": float(np.abs(position_error).max()),
        "maximum_rotation_component_error_rad": float(np.abs(rotation_error).max()),
        "sampled_position_norm_error_m": position_norm,
        "sampled_rotation_norm_error_rad": rotation_norm,
        "maximum_joint_affine_relation_error_rad": float(np.abs(joints - (entrance + progress[:, None] * difference)).max()),
        "continuous_position_norm_upper_bound_m": float(continuous_position_bound),
        "continuous_rotation_norm_upper_bound_rad": float(continuous_rotation_bound),
        "bound_details": {"tool_chain_offset_sum_m": reach,
                          "joint_l1_rate_per_normalized_time": max_q_l1_rate,
                          "progress_rate_per_normalized_time": max_progress_rate,
                          "position_gap_bound_m": float(position_gap_bound),
                          "rotation_gap_bound_rad": float(rotation_gap_bound)},
        "suffix_capacity_certified_within_original_component_bounds": bool(continuous_position_bound < .001
                                                                            and continuous_rotation_bound < .005),
        "full_approach_collision_certified": False,
        "production_joint_seed_injection": False,
        "joint_names": joint_names,
        "free_controls": actions.tolist(),
        "reference_endpoint_poses": reference.tolist(),
        "progress_clock_keys": len(warp.u),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
# END independent native-basis suffix-capacity oracle.
