# Demonstration-conditioned GraspGen-X adapter (prototype, Oct 2026)

This repository contains a *training-free* three-source candidate planner for GraspGen-X:

1. World-frame 6D demonstration prior and multi-scale local SE(3) perturbations
2. GraspGen-X diffusion global candidates
3. GraspGen-X OBB geometric candidates

All candidates are scored by GraspGen-X's existing discriminator. They are filtered
by an **allowed grasp patch**, a **hard approach angular bound**, external collision
checks, and robot-specific skill checks. A *soft* approach preference and
*soft* demonstration preference select among feasible candidates.

**Never send `CANDIDATE_ONLY` results directly to a real robot.**
`EXECUTION_READY` is possible only when three callbacks are supplied and return
positive results: full-scene collision, target-contact validation, and entire
robot skill trajectory feasibility. The included CLI supplies at most the
*approximate final-pose environment-collision* check, so it can never produce
an `EXECUTION_READY` result. Physical safety is dependent on your actual robot,
calibration, finger-pad contact and motion-planning implementations.

## Installation

Use the same `recovery` environment where GraspGen-X runs; no ZMQ is used.

```bash
conda activate recovery
# Verify GraspGen-X has already been installed and its demo works.
python -m pip install numpy scipy
export GRASPGENX_CHECKPOINT_DIR="$HOME/recovery/models/graspgenx_checkpoints"
export GRASPGENX_GRIPPER_CFG_DIR="$HOME/recovery/models/gripper_descriptions"
```

Place this package as `~/recovery/demo_graspgenx/`, for example. Checkpoint path
passed to `--checkpoints` must end with `/release` and contain the `gen/` and
`dis/` folders.

## Frames and input data

- World frame `W` is **gravity-aligned**; axes and units match across both RGB-D
  cameras, POIL, and robot calibration. Point clouds in **meters**.
- GraspGen-X frame `G`: +Z is gripper approach direction **into the object**;
  +X is the finger closing direction; source pose must be converted to this.
- `T_WG_prior`: 4x4 pose of *gripper base*, not the functional frame or TCP.
  If the transferred source grasp relative to the functional frame is
  `T_FG_source`, then `T_WG_prior = T_WF_target @ T_FG_source`. Human-hand
  demonstrations normally **do not** directly supply a robot grasp-frame prior.
- `model_pc_W` is model input: for a small movable object it can be the entire
  object; for a door/cabinet it may be a graspable handle **with context**. OBB
  heuristics assume world Z-up.
- `object_pc_W` is the entire segmented *target* object, used for local ROI
  contact support; `allowed_pc_W` is the permitted **graspable patch**, not
  necessarily the tool's functional interaction tip.
- `scene_without_target_pc_W` is the rest of the scene; it must include the
  table, other objects and fixed obstacles. Empty cloud is accepted for tests,
  but is not proof that the physical scene is obstacle-free.

```python
import numpy as np
np.savez_compressed(
    "scene_for_grasp.npz",
    model_pc_W=model_pc_W.astype(np.float32),
    object_pc_W=object_pc_W.astype(np.float32),
    allowed_pc_W=allowed_pc_W.astype(np.float32),
    scene_without_target_pc_W=scene_without_target_pc_W.astype(np.float32),
    T_WG_prior=T_WG_prior.astype(np.float32),  # optional
    approach_W=np.array([0., 0., -1.], dtype=np.float32), # example world direction, optional
)
```

`approach_W` is a **unit vector in W** pointing the desired direction of gripper
+Z; choose it from the actual task, not from a hardcoded camera direction.

## Run candidate-only smoke test (no real robot)

```bash
python run_offline.py \
  --input-npz scene_for_grasp.npz \
  --checkpoints "$GRASPGENX_CHECKPOINT_DIR/release" \
  --assets-dir "$HOME/recovery/GraspGenX/assets" \
  --gripper robotiq_2f_85 \
  --num-grasps 200 \
  --preferred-approach-deg 30 \
  --hard-max-approach-deg 110 \
  --env-collision-check \
  --output-npz grasp_plan_result.npz
```

**For a truly strict approach constraint**, set both angles equal (e.g., 30°).
For an environmental fallback, keep the 30° preference and a wider *hard*
ceiling, if task physics allows such a change. The algorithm never relaxes a
specified hard ceiling or the allowed grasp region.

The NPZ contains `T_WG`, `T_WG_pre`, `status`, `source`, `quality`,
`utility`, and `diagnostics_json`.

## Integrating contact and robot execution validators

```python
from demo_guided_grasp import GraspGenXBackend, DemoGraspPlanner, GraspRequest

backend = GraspGenXBackend(checkpoints_dir, "robotiq_2f_85", assets_dir=assets)

scene_collision = backend.make_scene_collision_callback(
    collision_threshold=0.008, num_collision_samples=768, batch_size=2,
)

# Implement these with your actual robot URDF/SDF/kinematics/physics and task.
def validate_contact(T_WG, request):
    # Check actual finger-pad contact patches lie inside allowed_pc_W.
    # Check collisions with *target* outside intended contact pads.
    # Check jaw width, contact normals, closure sweep, friction stability.
    raise NotImplementedError

def validate_skill(T_WG, T_WG_pre, request):
    # Convert GraspGen-X grasp frame to robot TCP/tool0 with calibrated T_G_TCP.
    # Check IK and continuous collision-free approach, grasp closure and retreat.
    # For pull/door/tool-use, check the entire *post-grasp* task trajectory,
    # robot reachability, joint limits and contact/wrench feasibility.
    raise NotImplementedError

planner = DemoGraspPlanner(
    backend, scene_collision_free=scene_collision,
    target_contact_valid=validate_contact,
    robot_skill_feasible=validate_skill,
)
result = planner.plan(request)
if result.status != "EXECUTION_READY":
    raise RuntimeError(f"Not safe to execute: {result.status} / {result.diagnostics}")
```

### What the built-in ROI filter does (and cannot guarantee)

Transforms object points into each candidate gripper frame and verifies that
sufficient *allowed* target points overlap the gripper's open swept-volume box.
This is much more informative than only comparing the gripper base translation
to the functional patch, but **it is still only a proxy**. It cannot prove
finger contacts, collision-free closing, task mechanics or force closure.
`validate_contact` must enforce true contact membership and target collisions.

### What built-in environment collision checks do (and cannot guarantee)

`graspgenx.utils.collision_filter.filter_colliding_grasps` compares sampled
open-gripper collision-mesh surface to the **non-target** scene PC, only at the
final grasp pose. It does not check arm links, the target itself, swept approach
or the grasped object's later trajectory. For a 5070 Ti 16 GB, reducing
`batch_size` and `num_collision_samples` limits the `torch.cdist` memory, but
also makes collision checking approximate. Execute only after a proper 3D/URDF
or distance-field motion planner verifies the complete trajectory.

## Algorithm and fallback policy

1. Validate input frames and ROI geometry.
2. Generate multi-scale `T_WG_prior` perturbations; gate on hard ROI/angle;
   score surviving poses using GraspGen-X's discriminator.
3. Optional fast path only if all three validators exist and model quality
   clears a configurable threshold.
4. Generate GraspMoE diffusion+OBB global alternatives, retain full set (no
   premature top-k based purely on model score), combine and evaluate.
5. If no feasible candidate, retry with denser OBB + additional diffusion.
6. If still none, return `NO_FEASIBLE_GRASP`; **never** relax the permitted
   grasp patch, user hard angle limit or safety/kinematic constraints silently.

The demonstration prior is a *soft preference*: candidates that deviate from
it can win when the environment makes the original pose infeasible. The
quality score is a learned discriminator score, **not a calibrated probability
of successful skill execution**. Score thresholds and ROI overlap must be
validated with task data. The branch labels are `demo`, `diff`, `obb`.

## Testing and validation boundaries

```bash
python -m pip install pytest
python -m pytest -q test_demo_guided_grasp.py
```

These tests cover local sampling/SE(3), symmetric rotations, strict angular
limits, ROI alignment, environmental fallback and explicit status semantics.
They use a MOCK `CandidateBackend`, **not a real GraspGen-X GPU checkpoint**.
To validate full feasibility in `recovery`, run the GraspGen-X official demo
first, then `run_offline.py` on real fused RGB-D point clouds, and finally add
robot-specific validators. No synthetic test substitutes for physical trials.

Sources verified 2026-10-10:
- https://github.com/NVlabs/GraspGenX/blob/main/graspgenx/samplers/planner.py
- https://github.com/NVlabs/GraspGenX/blob/main/graspgenx/samplers/graspmoe.py
- https://github.com/NVlabs/GraspGenX/blob/main/graspgenx/utils/collision_filter.py
- https://github.com/NVlabs/GraspGenX/blob/main/graspgenx/x_grippers.py
- https://github.com/NVlabs/GraspGenX/blob/main/end2end/robot_profiles.py
