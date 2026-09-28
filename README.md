# An Official Code Implementation of MimicFunc: Imitating Tool Manipulation from a Single Human Video via Functional Correspondence (CoRL 2025)
[[Preprint]](https://arxiv.org/abs/2502.11744) [[website]](https://sites.google.com/view/mimicfunc) [[video]](https://youtu.be/H0sEzJufpeE)

[Chao Tang](https://mkt1412.github.io/), Anxing Xiao, Yuhong Deng, Tianrun Hu, Wenlong Dong, Hanbo Zhang, David Hsu, and Hong Zhang  

If you find this work useful, please cite:
```
@article{tang2025functo,
  title={Functo: Function-centric one-shot imitation learning for tool manipulation},
  author={Tang, Chao and Xiao, Anxing and Deng, Yuhong and Hu, Tianrun and Dong, Wenlong and Zhang, Hanbo and Hsu, David and Zhang, Hong},
  journal={arXiv preprint arXiv:2502.11744},
  year={2025}
}
  ```

## Local pipeline

This workspace uses one `recovery` Python environment for local **SAM3.1 +
SD-DINO** perception. Gemini performs semantic keypoint selection and optional
pose selection through the existing s3e gateway backend. There are no OWLv2,
SAM, SD-DINO or LLaVA model servers to start.

- `local_models/sam31.py`: SAM3.1 text detection and segmentation from the local
  multiplex checkpoint. Boxes and masks always come from the same instance.
- `local_models/sd_dino.py`: SD v1.3, timestep 0, DINOv2 ViT-B/14 token layer 11,
  native shared two-image co-PCA (256 dimensions per SD scale), branch L2
  normalization and equal-weight concatenation. SD960 and DINO840 produce two
  FP32 `[1,1536,60,60]` descriptors. The old remote feature service's implementation
  was not provided; this is an explicit local feature definition.

<!-- BEGIN explicit model-pose adapter. -->

- `local_models/foundationpose.py`: rigid-instance registration and tracking for
  the raw object/part path, using calibrated RGB-D and explicitly identified
  local meshes. It validates rendered geometry against the observations before
  returning model poses; finite SE(3) matrices alone do not establish accuracy.

<!-- END explicit model-pose adapter. -->

- `local_models/runtime.py`: sequential, short-lived local worker processes
  release model RAM, GPU allocations and autocast state before dense matching.
  Network access is disabled inside perception workers; missing weights fail
  explicitly. The worker CUDA allocator is capped at 8 GiB, combined caller and
  worker RSS at 12 GiB, with at least 1.5 GiB host memory kept available.
  <!-- BEGIN host allocator reclamation before resource rejection. -->
  On host-memory pressure, the Linux worker first collects unreachable objects
  and returns unused glibc allocator pages, then rechecks the same limits.
  Live tensors, model calls and numerical settings are unchanged.
  <!-- END host allocator reclamation. -->
- `local_models/gemini.py`: reuses the sibling `s3e` source backend. It reads
  `SUB2API_API_KEY` only when called. `SUB2API_BASE_URL` defaults to the user's
  local gateway `http://127.0.0.1:18080/antigravity`; `GEMINI_MODEL` defaults to
  `gemini-3.7-flash-tiered`. Override these environment variables if needed.
  Annotated images are sent to that authorized gateway. API failures and invalid
  candidate indices raise errors instead of using old predictions.

<!-- BEGIN explicit local rigid-pose dependency. -->
Expected sibling repositories are `sam3`, `sd-dino`, `s3e`, and `FoundationPose`
for raw object/part motion tracking. SAM weights live at
`../models/sam3.1/sam3.1_multiplex.pt`; SD/ODISE/OpenCLIP and DINO use the existing
local caches checked by the adapter. No entry point installs dependencies or
downloads weights. The geometry stages require Open3D, SciPy, OpenCV and CasADi
with IPOPT, in addition to the existing model dependencies.
<!-- END explicit local rigid-pose dependency. -->

The raw SD path retains the accepted unused-component GPU optimization. It does
not share the two different CLIP models, change precision, cache pair features,
or change FUNCTO's interpolation, cosine similarity, mask multiplication or
global argmax.

## Demo

`main.py` implements released steps 8--12: it consumes the prepared demonstration
annotations and trajectory, detects the test scene, transfers points and poses,
then optimizes a trajectory. It does not implement raw human-video preparation
or robot execution.

The task configuration is `utils_IL/config/config_pour.yaml`. SD-DINO is enabled
by default. `vp_flag: false` disables only optional pose selection; semantic
function/grasp point selection still uses Gemini.

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python main.py --output-dir ./runs/pour-local
```

The output directory must be empty. Only the initial test RGB-D observation and
its metadata are copied there, and all predictions are generated anew. Without
`--output-dir`, each invocation creates a timestamped directory under `runs/`.
`--data-root` selects a root containing `demo_data/<task>_demo` and
`test_data/<task>_test`; `--config` selects another YAML task configuration.
Paths to prompts and default configuration are independent of the current
working directory. The bundled RGB-D example uses the original Azure Kinect
intrinsics and millimeter depth units in `utils_IL/params.py`; custom recordings
need matching calibration before geometric processing.

After success, `run.json` records stage completion and validates finite 2D/3D
points and trajectory arrays. The final files are
`tool_traj_transfer_output/test_tool_trajectory.json` and
`tool_traj_transfer_output/test_tool_traj_pc.ply` inside that run directory.
Historical results under `test_data/` remain example data, never substitutes for
a completed new run.

<!-- BEGIN offline visualization of saved local predictions. -->
## Visual validation

Render a completed run without model inference or Gemini calls:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  python visualize_run.py ./runs/pour-local
```

Open `visualization/index.html` inside the run. It links the mask/keypoint
overlays, saved semantic selections, full trajectory MP4, an offline interactive
3D replay with a frame slider, and numerical diagnostics. `--demo-dir` overrides
the demonstration path recorded in `run.json` if the data has moved.

The renderer reuses the original point-cloud reconstruction and compares its
poses against every tool snapshot in the saved PLY. Only displayed point counts
are reduced. The static original tool is removed from the background; hidden
surfaces stay unknown. White crosses in the image show the actual 3D points
reprojected through the saved calibration, exposing depth-neighbor adjustments
to the 2D matches. Trajectory translations are **function-point** positions;
video FPS is illustrative because the saved poses have no timestamps. These
views support semantic and geometric inspection, not collision, dynamics or
robot-execution certification.
<!-- END offline visualization of saved local predictions. -->

## Contract checks

The focused tests use synthetic CPU data and mock model/API boundaries. They
check instance association, semantic response parsing, fresh output isolation,
and result validation; they do not download weights or initialize GPU models.

```bash
PYTHONNOUSERSITE=1 MPLBACKEND=Agg PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  python -m pytest -q tests cloud_services/test_scene_graph_contract.py
```

<!-- BEGIN task-adaptive object/part manipulation. -->
## Object and moving-part manipulation

`manipulate.py` adds a separate path for object transport, placement, containment
and articulated parts. Motion, goal relation and interaction are independent:
an object does not need a tool function point, and a pushed part does not need a
grasp point. The existing `main.py` tool-use reproduction remains available.

<!-- BEGIN compiled region/grasp/function-event consumer. -->
### Compiled skills and functional regions

`manipulation.program.ground_program(program, scene)` consumes independent
`skill_kind="grasp"` and `"manipulation"` skills. It returns EEF geometry,
explicit gripper intentions and original frame IDs/timestamps. A grasp is a
candidate until current observations verify stable holding; lifting belongs to
the subsequent manipulation. Held-object motion uses the caller's measured
`scene.holding.T_carrier_eef`, not the preceding grasp candidate. At release,
the EEF continues independently and the released object requires observation.

`regions.function_frame` constructs directions from evidenced functional
geometry. `regions.ground_relation` preserves both sides of the relation:
`D_new = F_reference_new @ inverse(F_reference_demo) @ D_demo @ F_carrier_demo
@ inverse(F_carrier_new)`. Explicit internal function events prevent initial-pose
normalization from cancelling the carrier's functional offset. Region clouds do
not have to describe globally congruent objects. RGB-D local correspondence is
one supported estimator, not a replacement for functional semantics.

Owner-local functional frames and current output coordinates are distinct:
`scene.reference_pose` maps the current reference owner into the named output
`scene.reference_frame`; use an explicit identity only when they coincide.
`perception.bind_current_scene(program, scene, source_templates, current_owners)`
binds cached demonstration evidence without rerunning source tracking. Source
templates contain the known interaction region, explanatory query pixels and
their geometric construction. **Current input contains owner RGB-D only; no
current part mask, part ID, target query or functional frame is supplied.**
Gemini uses the source evidence to propose coarse search cells; the existing
SD-DINO matcher searches those windows inside the current owner. One descriptor
pair is reused for the queries. The source/current frames are reconstructed
independently, so changed part dimensions do not have to pass a rigid-shape fit.

The source template fields are `image`, `mask` (offline interaction region),
`owner_mask`, registered metric `depth`, `K`, `T_camera_owner`, `owner_id`,
`region_id`, `description`, `query_pixels`, `query_descriptions` and
`frame_recipe`. Direction queries may be elsewhere on the same owner: a small
contact patch does not necessarily contain every directional cue. The recipe
stores `kind` (`frame` or `plane`), `origin_weights`, `axis_weights`,
`plane_weights`, and an evidence `source`. Weights have one entry per query;
their sums are respectively 1, 0, and 0. `plane_weights` defines the second
in-plane hint, not a normal. The origin and two directed hints reconstruct a
right-handed frame. These weights express evidenced functional geometry, not
an automatic semantic interpretation of arbitrary sampled points. Source
reconstruction is checked against compiled event frames before model inference.

Each current owner supplies `image`, `owner_mask`, registered metric `depth`,
`K`, `owner_id`, `T_output_camera` and a `frame_convention`: `tracked_owner`
uses its existing `T_camera_owner`; `binding_camera` explicitly defines the
binding-time camera as its fixed initial local coordinates. Identity in the
latter case is a coordinate definition, not an inferred physical orientation.
Subsequent tracking must preserve that definition. Camera calibration cannot
be replaced by the coordinate choice. A grasp needs only the carrier owner;
manipulation needs both carrier and reference.

`perception.ground_current_program(..., output="eef")` calls this binder and
the existing complete skill consumer. Choose `output="object"` explicitly
for manipulation body geometry. Missing actual Holding never silently switches
an EEF request to object-only output. Both return candidates with correspondence
and coordinate evidence, not certificates of physical execution. The existing
raw-video `--demo` path keeps its separate extraction/tracking responsibility.

For an `on` event, a source plane recipe additionally declares `normal_role` as
`outward` (carrier) or `support` (reference), backed by its offline evidence.
Matched queries originally inside the source interaction mask can generate
sparse current contact samples; auxiliary direction queries cannot stand in
for contact samples. The existing support estimator recovers a finite visible
surface from current owner RGB-D and checks it against the correspondence.
Richer current model geometry may be supplied, but current part geometry is
not a universal external prerequisite. Sparse samples and visible body points
do not certify hidden geometry or a complete contact face. `ground_on` checks
the contact samples, signed normal, supplied body extent and finite support.
Plane-frame tangents are numerical preferences, not measured functional twist;
plane evidence is rejected where a full functional orientation is required.
Constrained parts require a sourced current `motion_model`,
including available coordinate limits. Required function poses cannot be
silently projected off that mechanism. Supplied entry corridors remain checked.

`function_motion.transfer_function_motion` consumes function-frame SE(3) poses
directly, keeps the original function-origin/Euler objective and internal hard
events, and independently verifies all saved-node increments. Its fixed strict
constraint mode does not retry by weakening angular limits. The compiled tool
object path uses the same core; missing actual holding never produces a fake EEF.

Existing `--prepared` accepts a JSON object containing `skill_kind`, `demo`,
`task` when needed, and the current `scene`. Source numeric arrays may live in
`demo.trajectory_file` (NPZ, relative to the input JSON); field duplication is
rejected and saved run metadata retains an absolute reference for replay.

```bash
PYTHONNOUSERSITE=1 python manipulate.py --prepared /path/to/prepared.json \
  --output-dir runs/compiled-skill
```

The read-only offline exporter is `../collect/export_functo.py`. Actual eight-clip
numeric examples and full field conventions are documented in
[the implementation and mathematical review](../sim-version-plans/2026-09-28-mimicfunc-region-math-and-implementation.md).
Those are same-instance offline GT fixtures, not proof of SAM localization,
cross-instance grounding or robot execution. The current semantic mask gaps
remain explicit. No environment installation or model download is required for
the numeric consumers.
<!-- END compiled region/grasp/function-event consumer. -->

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python manipulate.py --demo /path/to/demonstrated-stage --scene /path/to/current-scene \
  --instruction "push the drawer closed" --output-dir runs/drawer-close
```

Each RGB-D directory contains `rgb/00000.jpg` (or PNG), registered
`depth/00000.png` (or NPY), and `intrinsic.json`. The demonstration contains the
whole stage's numbered frames. Open3D's column-major `intrinsic_matrix`, width
and height are read from calibration. `depth_unit_m` explicitly specifies metres
per stored value; Open3D `depth_scale` specifies its divisor. If neither is
provided, the original millimetre convention applies. Geometry is no longer
restricted to the original one-metre range or hard-coded camera intrinsics.

<!-- BEGIN TAPIP3D object motion and fixed semantic ownership. -->
The raw path uses Gemini for role/relationship selection, SAM3.1 for rigid-part
masks, SD-DINO for cross-instance correspondence and local TAPIP3D for calibrated
RGB-D material tracks. It samples up to 64 spread, valid-depth points per body,
tracks carrier and reference together in one bounded GPU worker, and fits
first-frame-to-current rigid motion with the existing correspondence RANSAC
and least-squares refit. Visibility is a filter, not an independent accuracy
score. Raw tracks, query identities, residuals and reference motion remain
available in `observations.json`. No mesh or monocular depth model is required.

Camera coordinates must be calibrated. For a stationary recording, declare
`"stationary_camera": true` in the existing demonstration `config.json`.
Programmatic `track_rgbd(..., extrinsics=...)` instead accepts known
camera-from-world SE(3) matrices for every selected frame. An object's reference
motion is compensated separately and never substitutes for camera calibration.
The bundled pour recording uses a stationary camera; older runs can be replayed
through `main.py` with their original annotated trajectories.

Optional semantic points and directions bind once to an explicit owner using
`manipulation.anchors.bind_anchors` and follow that owner's fixed body poses
through `propagate_anchors`. Empty anchor lists are valid for object motion.
A sliding contact is not automatically treated as a fixed material point.
The tool path now consumes `demo/interaction_program.json` (compiled functional
frames, original timeline and internal events) and `scene/interaction_scene.json`
(current functional bindings and geometry). Source tracking converts once to
`T_reference_carrier`; an already compiled source trajectory can also be consumed
directly. This production interface has no mandatory function/center/grasp triple
or fixed list of tool-action names. The original `main.py` reproduction still
uses its released annotations; it is not a fallback for missing region evidence.

Observed compatible motion families and applied task constraints are separate
in `motion_checks`. A straight carry does not automatically become a slider;
a declared whole-stage planar task is validated and transferred in its tilted
plane. Invalid observations or incompatible known constraints fail explicitly.
The released tool optimizer cannot jointly enforce a mechanical/planar family
and its function event; such combinations are explicitly rejected, never
silently converted into free motion.
<!-- END TAPIP3D object motion and fixed semantic ownership. -->

An `on` goal selects a task-relevant observed plane with Gemini, then uses a
bounded patch that remains inside its observed mask, including any holes. It
does not treat the largest face of a reference object as its support surface.
The supplied object extent must fit the patch. An `inside` goal means placing
an object into a cabinet, drawer or another container. Its demonstrated endpoint
is mapped through the observed container reference, without requiring extra
offline geometry. The output marks containment as `not_verified` when no free
interior is observed or supplied: it is a transferred object trajectory, not a
claim that hidden walls or clearance have been checked. If known **free interior
geometry** is supplied, the complete supplied object point set must fit it; an
invalid region or infeasible placement remains an error. The container's outer
box is never used as its free interior. An optional `geometry.json` can carry
geometry produced by another component:

```json
{
  "reference_frame": "scene_camera",
  "interior": {
    "vertices": [[0,0,0.5], [0.2,0,0.5], [0,0.2,0.5], [0.2,0.2,0.5],
                 [0,0,0.7], [0.2,0,0.7], [0,0.2,0.7], [0.2,0.2,0.7]],
    "source": "known free cavity geometry"
  }
}
```

This example describes a convex free region in metres. Partial RGB-D does not
reveal an occluded cavity, and neither a semantic label nor an outer mask is used
to invent it. Optional `entry_region` and `entry_start_index` check the supplied
object geometry against a known entry corridor at the generated trajectory
nodes, using the supplied `margin` in metres. They do not certify interpolation
between those nodes. These local checks are not complete-scene collision or force planning.

The numerical API `manipulation.pipeline.run_stage` accepts already grounded
observations. `run_sequence` accepts an ordered list and updates carrier and
reference states. Repeated entities require registered, fixed `body_frame_id`
values; moving references require `reference_body_frame_id` and their capture
pose. A placed object is not assumed rigidly attached to its support/container:
if that reference later moves, its world state becomes `needs_observation`.
This sequence API consumes a provided order; it does not discover a new task
plan from a long instruction.

`scene.goal_pose` is an exact target and is never projected into a convenient
answer. `scene.preferred_goal_pose` is an estimate that placement may adjust
within the requested geometric relation. Goal geometry and motion must both
pass. The core supports free SE(3), planar SE(2), fixed-orientation planar
translation, fixed-axis rotation, and fixed-direction translation.

To replay a saved grounding without loading models or calling an API:

```bash
PYTHONNOUSERSITE=1 python manipulate.py --prepared runs/drawer-close/observations.json \
  --output-dir runs/drawer-close-replay
```

Every new run writes `run.json`, grounded `observations.json` for the object path,
and `trajectory.json` containing fixed-body-to-reference 4×4 poses in metres,
entity IDs, frame conventions, model/goal checks and available interaction
evidence. These are object/part trajectories, not end-effector commands. The
offline `trajectory.png` and `coordinate_checks.json` visualize bounded
keyframes and verify the underlying point transformations without invented
function or grasp markers. A successful geometry run does not certify contact
forces, hidden surfaces, physical task completion or robot execution.

Run the analytic examples without models to inspect all five spaces, placement
and an open/place/close sequence:

```bash
PYTHONNOUSERSITE=1 MPLCONFIGDIR=/tmp/functo-mpl OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python examples/manipulation_geometry.py --output-dir runs/manipulation-geometry
```

These examples use known synthetic geometry. Actual new task coverage still
requires its own demonstrated stage and scene; the bundled pouring example
alone cannot verify real drawer, door or container operation.
<!-- END task-adaptive object/part manipulation. -->
