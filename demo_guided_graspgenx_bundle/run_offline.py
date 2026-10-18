"""Offline candidate-only GraspGen-X integration for current recovery environment.

This CLI does not send commands to any robot and cannot claim execution safety.
Replace/add validators using DemoGraspPlanner in your robot controller.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from demo_guided_grasp import (
    GraspGenXBackend, GraspRequest, DemoGraspPlanner, PlannerOptions
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-npz', required=True)
    p.add_argument('--checkpoints', required=True, help='root containing gen/ and dis/')
    p.add_argument('--assets-dir', help='GraspGenX repo assets root (optional if gripper_descriptions present)')
    p.add_argument('--gripper', default='robotiq_2f_85')
    p.add_argument('--output-npz', default='grasp_plan_result.npz')
    p.add_argument('--preferred-approach-deg', type=float, default=30.0)
    p.add_argument('--hard-max-approach-deg', type=float, default=110.0)
    p.add_argument('--num-grasps', type=int, default=200)
    p.add_argument('--env-collision-check', action='store_true', help='final-pose ENVIRONMENT mesh/PC collision only')
    args = p.parse_args()

    with np.load(args.input_npz, allow_pickle=False) as data:
        arrays = {k:data[k] for k in data.files}
    for key in ['model_pc_W','object_pc_W','allowed_pc_W','scene_without_target_pc_W']:
        if key not in arrays:
            p.error(f'missing {key} in NPZ')
    req = GraspRequest(
        model_pc_W=arrays['model_pc_W'],
        object_pc_W=arrays['object_pc_W'],
        allowed_pc_W=arrays['allowed_pc_W'],
        scene_without_target_pc_W=arrays['scene_without_target_pc_W'],
        T_WG_prior=arrays.get('T_WG_prior'),
        approach_W=arrays.get('approach_W'),
        preferred_approach_deg=args.preferred_approach_deg,
        hard_max_approach_deg=args.hard_max_approach_deg,
    )
    backend=GraspGenXBackend(
        args.checkpoints, args.gripper, assets_dir=args.assets_dir,
        num_grasps=args.num_grasps,
    )
    scene_collision = backend.make_scene_collision_callback() if args.env_collision_check else None
    # No fake contact/arm validators. Status stays CANDIDATE_ONLY by design.
    planner=DemoGraspPlanner(backend,scene_collision_free=scene_collision,
                             opts=PlannerOptions(fast_path_enabled=False))
    result=planner.plan(req)
    print('STATUS:',result.status)
    print('SOURCE:',result.source,'QUALITY:',result.quality,'UTILITY:',result.utility)
    print('DIAGNOSTICS:',json.dumps(result.diagnostics,ensure_ascii=False,indent=2))
    target=Path(args.output_npz)
    target.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(target,
        T_WG=(result.T_WG if result.T_WG is not None else np.zeros((0,4,4))),
        T_WG_pre=(result.T_WG_pre if result.T_WG_pre is not None else np.zeros((0,4,4))),
        status=np.array(result.status), source=np.array(result.source),
        quality=np.array(result.quality), utility=np.array(result.utility),
        diagnostics_json=np.array(json.dumps(result.diagnostics,ensure_ascii=False)))
    print('Saved:',target)

if __name__=='__main__':
    main()
