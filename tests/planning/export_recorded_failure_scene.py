"""Export static native scene queries for already saved IK configurations.

This reader reuses the original bounded scene builder, intercepts its planning
boundary, and never invokes an optimizer or physics. Saved outcomes remain
authoritative; their scene residuals must match the independently exported
per-sphere values before a visualization artifact can be published.
"""

# BEGIN bounded diagnostic export of frozen states, without a new solve.
import argparse
import json
from pathlib import Path
import resource
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from probe_recorded_grasp_feasibility import (
    cpu_array, native_fk, observed_scene, parse_args, read_json,
)


def export(args):
    started = time.monotonic()
    original = read_json(args.probe_report)
    journal = read_json(args.command_record)
    command = journal['command']
    script, = [i for i, value in enumerate(command)
               if Path(value).name == 'probe_recorded_grasp_feasibility.py']
    with patch.object(sys, 'argv', command[script:]):
        source_args = parse_args()
    if (source_args.output.resolve() != args.probe_report.resolve().parent
            or original['status'] != 'completed'
            or original['trajectory_planning_called'] or original['simulation_called']
            or original['candidate_id'] != source_args.candidate
            or source_args.torch_memory_mib != 2048):
        raise ValueError('Explicit command and frozen completed static probe must match')
    args.output.mkdir(parents=True, exist_ok=False)
    source_args.output = args.output
    rows = [{
        'kind': 'entrance', 'mode': 'measured', 'source_frame': None, 'rank': None,
        'q': original['seed_policy']['seed_pool'][0][0], 'progress': 0.,
        'target': original['measured_q0_fk_pose'],
        'tool_pose': original['measured_q0_fk_pose'],
        'native_success': None, 'pose_limits_passed': True,
        'postcheck': original['measured_q0_collision_postcheck'],
    }]
    for mode, mode_report in original['modes'].items():
        for target in mode_report['targets']:
            with np.load(target['artifact'], allow_pickle=False) as artifact:
                for solution in target['solutions']:
                    rank = solution['solution_rank']
                    if not np.array_equal(artifact['joint_position'][rank], solution['joint_position']):
                        raise ValueError('Frozen joint arrays and their report disagree')
                    rows.append({
                        'kind': 'ik', 'mode': mode, 'source_frame': target['source_frame'],
                        'rank': rank, 'q': solution['joint_position'], 'progress': target['progress'],
                        'target': target['T_reference_tool'],
                        'tool_pose': artifact['actual_tool_pose'][rank].tolist(),
                        'native_success': solution['native_success'],
                        'pose_limits_passed': solution['fk_pose_and_limits_satisfied'],
                        'position_error_m': solution['fk_position_error_m'],
                        'rotation_error_rad': solution['fk_rotation_error_rad'],
                        'postcheck': solution['original_collision_postcheck'],
                    })
    if not 1 <= len(rows) <= 65:
        raise ValueError('This diagnostic permits at most 65 explicit saved states per scene')

    import torch
    from curobo.kinematics import Kinematics, KinematicsCfg
    from curobo.types import DeviceCfg

    prepared = observed_scene(source_args)
    problem, scene = prepared['problem'], prepared['scene']
    if scene.contacts or any(owner.moving for owner in scene.owners):
        raise ValueError('This frozen grasp review requires static owners and no contact permissions')
    device = DeviceCfg('cuda:0')
    native = Kinematics(KinematicsCfg.from_robot_yaml_file(prepared['robot'], device_cfg=device))
    names = tuple(native.joint_names)
    if names != tuple(original['joint_names']) or native.total_spheres > 480:
        raise ValueError('Native geometry and active joint order changed')
    if scene.estimated_workspace_bytes(8, 1) > 1024**3:
        raise ValueError('Static queries exceed the existing 1 GiB workspace budget')
    with np.load(args.probe_report.parent/'native-collision-model.npz', allow_pickle=False) as model:
        sphere_links = model['source_sphere_links'].copy()
        native_pairs = model['collision_pairs'].copy()
    if not np.array_equal(native.get_self_collision_config().collision_pairs.cpu().numpy(), native_pairs):
        raise ValueError('Static export must retain the exact saved native sphere pairs')
    hand = scene.gripper
    hand_pairs = np.asarray(hand.self_collision_pairs, dtype=np.int64)
    pairs_tensor = torch.as_tensor(hand_pairs, device='cuda', dtype=torch.long)
    arrays = {key: [] for key in ('actual_spheres', 'scene_spheres', 'tool_poses',
                                  'unknown', 'background', 'gripper_overlap', 'native_overlap')}
    maximum_scene_error = maximum_fk_error = 0.
    for start in range(0, len(rows), 8):
        batch = rows[start:start+8]
        values = np.asarray([row['q'] for row in batch])
        with torch.no_grad():
            query, state, position, quaternion, poses = native_fk(native, problem, values, names, device)
            spheres = problem.transform_robot_spheres(state.robot_spheres)
            progress = device.to_device(np.asarray([row['progress'] for row in batch]))[:, None]
            coordinates, half, _ = hand.state(progress, query, names)
            expanded = hand.spheres(spheres, position, quaternion, coordinates, half)
            unknown = scene.coverage.violation(expanded)
            background = scene.background(expanded, key=('robot', 'background'))
            components = scene.components(spheres, position, quaternion, progress, query, names)
            # Expose the individual terms of the production positive-overlap
            # sum. No alternate penalty or contact allowance is introduced.
            first = expanded.index_select(-2, pairs_tensor[:, 0])
            second = expanded.index_select(-2, pairs_tensor[:, 1])
            hand_penetration = cpu_array(torch.relu(first[..., 3]+second[..., 3]
                - torch.linalg.vector_norm(first[..., :3]-second[..., :3], dim=-1)))[:, 0]
            actual = cpu_array(spheres)[:, 0]
            planned = cpu_array(expanded)[:, 0]
        for index, row in enumerate(batch):
            maximum_fk_error = max(maximum_fk_error, float(np.max(np.abs(poses[index]-row['tool_pose']))))
            for key, tensor in components.items():
                actual_term = float(cpu_array(tensor)[index].max())
                saved_term = row['postcheck']['residuals']['scene:'+key]
                maximum_scene_error = max(maximum_scene_error, abs(actual_term-saved_term))
            background_sum = float(cpu_array(background)[index].sum())
            unknown_max = float(cpu_array(unknown)[index].max())
            if (abs(background_sum-row['postcheck']['residuals']['scene:collision/robot/background']) > 2e-6
                    or abs(unknown_max-row['postcheck']['residuals']['scene:unknown/space/robot']) > 2e-6
                    or abs(float(hand_penetration[index].sum())-
                           row['postcheck']['residuals']['scene:gripper/self']) > 2e-5):
                raise ValueError('Per-sphere reductions disagree with the original scene result')
            overlaps = np.zeros(native.total_spheres, dtype=float)
            np.maximum.at(overlaps, hand_pairs[:, 0], hand_penetration[index])
            np.maximum.at(overlaps, hand_pairs[:, 1], hand_penetration[index])
            native_penetration = np.maximum(0., actual[index, native_pairs[:, 0], 3]
                + actual[index, native_pairs[:, 1], 3]
                - np.linalg.norm(actual[index, native_pairs[:, 0], :3]
                                 - actual[index, native_pairs[:, 1], :3], axis=1))
            native_overlap = np.zeros(native.total_spheres, dtype=float)
            np.maximum.at(native_overlap, native_pairs[:, 0], native_penetration)
            np.maximum.at(native_overlap, native_pairs[:, 1], native_penetration)
            positive = np.flatnonzero(hand_penetration[index] > 0)
            top = positive[np.argsort(hand_penetration[index, positive])[-16:][::-1]]
            row['gripper_pairs'] = {
                'positive_count': len(positive),
                'maximum_overlap_m': float(hand_penetration[index].max()),
                'top': [{'indices': hand_pairs[i].tolist(),
                         'links': sphere_links[hand_pairs[i]].tolist(),
                         'overlap_m': float(hand_penetration[index, i])} for i in top]}
            arrays['gripper_overlap'].append(overlaps)
            arrays['native_overlap'].append(native_overlap)
        for key, value in (('actual_spheres', actual), ('scene_spheres', planned),
                           ('tool_poses', poses), ('unknown', cpu_array(unknown)[:, 0]),
                           ('background', cpu_array(background)[:, 0])):
            arrays[key].extend(value)
    if maximum_fk_error > 2e-6 or maximum_scene_error > 2e-5:
        raise ValueError(f'Frozen evidence changed: FK {maximum_fk_error}, scene {maximum_scene_error}')
    np.savez_compressed(args.output/'states.npz', **{k: np.asarray(v) for k, v in arrays.items()},
                        sphere_links=sphere_links, hand_pairs=hand_pairs, native_pairs=native_pairs,
                        coverage_clearance=scene.coverage._distance.cpu().numpy()[0, 0],
                        coverage_origin=scene.coverage.origin,
                        coverage_voxel_size=np.asarray(scene.coverage.voxel_size))
    report = {'status': 'exported', 'candidate_id': original['candidate_id'],
              'probe_report': str(args.probe_report.resolve()),
              'command_record': str(args.command_record.resolve()),
              'manifest_path': str(source_args.manifest.resolve()),
              'fixture_path': str(source_args.robot_fixture.resolve()),
              'feedback_path': str(source_args.feedback.resolve()),
              'segment_id': source_args.segment, 'states': rows,
              'sphere_count': native.total_spheres, 'native_pair_count': len(native_pairs),
              'maximum_fk_matrix_difference': maximum_fk_error,
              'maximum_original_scene_term_difference': maximum_scene_error,
              'ik_solver_called': False, 'trajectory_planning_called': False,
              'simulation_called': False, 'maximum_query_batch': 8,
              'elapsed_seconds': time.monotonic()-started,
              'peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'torch_peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    (args.output/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'states'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('probe-report', 'command-record', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    export(parser.parse_args())
# END bounded diagnostic export of frozen states, without a new solve.
