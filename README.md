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
geometry. For tasks that require a complete functional orientation,
`regions.ground_relation` preserves both sides of the relation:
`D_new = F_reference_new @ inverse(F_reference_demo) @ D_demo @ F_carrier_demo
@ inverse(F_carrier_new)`. Explicit internal function events prevent initial-pose
normalization from cancelling the carrier's functional offset. Region clouds do
not have to describe globally congruent objects. RGB-D local correspondence is
one supported estimator, not a replacement for functional semantics.

Owner-local functional frames and current output coordinates are distinct.
Every result names its common output coordinates in `scene.reference_frame`.
For a complete-frame relation, `scene.reference_pose` maps the current reference
owner into those coordinates; use an explicit identity only when they coincide.
The `on` relation instead binds a finite support region directly and does not
require the reference object's complete pose or a reference functional frame.

<!-- BEGIN generic semantic selection before geometric binding. -->
`manipulation.selection.select_grounding_candidates` automatically assembles one
VLM request from an existing demonstration template, the **current instruction**,
and actual detector candidates. Its fixed instructions are shared across skills;
it does not ask another model to write a prompt, branch on a skill name, or send
source poses, asset IDs or detector scores to the semantic decision.

```python
from manipulation.selection import select_grounding_candidates

selection = select_grounding_candidates(
    source_template,             # image, interaction mask, description; optional query descriptions
    current_instruction,         # current task intent, separate from demonstration evidence
    {"camera": {"image": rgb, "candidates": detections}},
    selection_scope="owner",     # requires source_template["owner_mask"]
)
decision = selection["views"]["camera"]
```

Choose the scope from the consumer's responsibility: `owner` selects the whole
rigid entity or moving part represented by the demonstration owner (for example,
the grasped object or the reference furniture); `interaction_region` selects a
local functional part. The same source interaction mask and semantic description
explain the function in either case. Source-only colour, level or world orientation
does not override the current request. Placement can supply its existing support
description with no ordered queries or frame recipe.

Even a single candidate is checked by the VLM. Results distinguish `selected`,
`alternatives` (every returned candidate is semantically permitted), `unresolved`,
`no_match`, and `no_candidates`. Empty input does not trigger a model call. No mask
is generated or merged: `decision["regions"]` contains the selected original
candidates, and a caller must explicitly resolve multiple allowed candidates
before using a consumer that requires one. IDs are local to each supplied view;
multiple views do not establish a three-dimensional correspondence.

Only a selection with `selection_scope == "owner"` can supply a current
`owner_mask`. A selected interaction mask remains local evidence and cannot stand
in for the object's visible geometry. This semantic selection does not certify
mask completeness, accessibility, grasp orientation, Holding, or execution. The
existing functional-correspondence and observed-support solvers retain their
separate geometric responsibilities.
<!-- END generic semantic selection before geometric binding. -->

`perception.bind_current_scene(program, scene, source_templates, current_owners)`
binds cached demonstration evidence without rerunning source tracking. For a
grasp or a complete-frame relation, source templates contain the known
interaction region, explanatory query pixels and their geometric construction.
**Current visual input contains owner RGB-D; it requires no current part mask,
part ID, target query or functional frame.**
Gemini uses the source evidence to propose coarse search cells; the existing
SD-DINO matcher searches those windows inside the current owner. One descriptor
pair is reused for the queries. The source/current frames are reconstructed
independently, so changed part dimensions do not have to pass a rigid-shape fit.

The query-based source template fields are `image`, `mask` (offline interaction region),
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
complete-frame manipulation needs both carrier and reference. An `on` binding
needs only the reference observation because the held carrier is already bound,
as described below.

`perception.ground_current_program(..., output="eef")` calls this binder and
the existing complete skill consumer. Choose `output="object"` explicitly
for manipulation body geometry. Missing actual Holding never silently switches
an EEF request to object-only output. Both return candidates with correspondence
and coordinate evidence, not certificates of physical execution. The existing
raw-video `--demo` path keeps its separate extraction/tracking responsibility.

<!-- BEGIN unordered support binding and explicit held-carrier contract. -->
For `task.relation="on"`, each `source_templates[event_id]` contains only a
`reference` semantic record with `owner_id`, `region_id`, `description`,
`normal_role="support"` and `source`. Its owner is the demonstrated reference
owner. The record identifies the demonstrated supporting part; it contains no
ordered corner queries or reference frame recipe. `current_owners` contains only
the selected `reference` owner's calibrated RGB-D observation. Gemini selects
among planes measured from that owner, or abstains; it does not generate metric
geometry. All events in one `on` skill name the same source support region and
use one consistent carrier mapping. The chosen finite patch stays within
observed support pixels.

The caller supplies the already bound carrier and actual holding:

- `scene.carrier_id`, `initial_pose`, `body_points` and `body_points_source`
  identify the current body, its pose in the common output coordinates, and
  its geometry in the current carrier's fixed local coordinates.
- `scene.functional_bindings[event_id]` contains `carrier_id`, `reference_id`,
  `carrier_mapping`, `on_geometry` and `source`. `carrier_mapping` maps source
  carrier coordinates into current carrier coordinates. `on_geometry` declares
  `carrier_region_points`, `carrier_outward_normal`, an evidence `source`, and
  `gap_tolerance`/`normal_tolerance`, all geometry in carrier-local coordinates.
  Identity is permitted only for an
  explicitly established common carrier coordinate system, never inferred
  from the action name or matching object category.
- `scene.support_direction` and `support_direction_source` provide the
  physically justified support-side direction in `scene.reference_frame`.
  For a top bearing surface this can be calibrated world up. A camera viewing
  direction, arbitrary carrier starting point or fitted PCA axis cannot replace
  that evidence. The visual binder rejects a pre-supplied `support`; an
  independently established signed support observation instead goes directly
  to `ground_program` or `ground_object_program`, the numerical consumers.
- EEF output retains the synchronized `eef_pose`, actual
  `holding.T_carrier_eef`, and the existing tool identity requirements. The
  support binder neither estimates nor verifies that holding relationship.

This does not require current simulator object GT. When the carrier's contact
surface is visible, the upstream helper
`perception.bind_carrier_contact(template, current_owner, *, position_tolerance_m,
normal_tolerance_rad, client=None, extractor=None)` reuses the source-region
visual matcher. It returns `carrier_mapping`, `on_geometry`, `initial_pose`,
`body_points` and their coordinate/evidence records, which the caller combines
with actual holding feedback. Its source plane template declares
`normal_role="outward"`; only matched source interaction queries become
contact samples. Sparse observations do not establish an occluded complete
face or full body coverage. A previously known carrier model may supply richer
geometry in the same bound coordinates. Neither route fabricates holding.

`ground_on` checks the declared contact face, opposing signed normals, supplied
body extent and finite support. Plane tangent axes are computational coordinates,
not a measured cup-handle yaw or a required placement point. The motion preference
comes from the current carrier and the demonstrated motion; changing only a
support plane's coordinate basis cannot rotate the carrier or its trajectory.
Internal function events, opening intentions, the release boundary and the
independent post-release retreat remain distinct and retain the original
timeline. Satisfying a function event does not certify subsequent contact,
force stability, collision-free robot motion or execution.

An isolated placement experiment may explicitly assume a fixed known carrier
and exact Holding. This tests support grounding and conditional motion transfer;
it does not establish online holding estimation or successful execution of a
preceding grasp. Offline demonstration GT may label source geometry and events;
current hidden reference-object pose, masks, contact or surface geometry must
not enter a visual grounding experiment.
<!-- END unordered support binding and explicit held-carrier contract. -->

Plane-only evidence is rejected where a full functional orientation is required.
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
The supplied object extent must fit the patch. When the raw RGB-D path infers
this support, the current scene's existing `config.json` must provide
`support_direction` and `support_direction_source`; the direction is expressed
in **scene camera coordinates**. For example, if independent calibration places
the physical upward direction along camera negative Y:

```json
{
  "depth_scale": 1000,
  "support_direction": [0, -1, 0],
  "support_direction_source": "calibrated upward direction in this camera frame"
}
```

The vector in this example is not a default. Missing direction evidence is an
error when inferring support; the demonstrated endpoint or object center is not
used to guess the physical side. An independently supplied signed `support` in
`geometry.json` remains usable without separately providing this direction.

An `inside` goal means placing
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

<!-- BEGIN observed local protrusion geometry for independent grasp grounding. -->
### Local protruding regions and independent grasp candidates

An observed elongated region can supply grasp direction without three ordered
material-point correspondences or a complete current owner mask:

```python
from manipulation.protruding_perception import observed_backing_planes
from manipulation.protruding_geometry import observe_protruding_line, protruding_frames

plane, evidence = observed_backing_planes(rgb, depth_m, K, selected_region_mask)
geometry = observe_protruding_line(depth_m, K, selected_region_mask, plane)
# Match an existing source contact anchor into geometry["front_mask"] with
# manipulation.grounding.match_points. Keep the returned original-image pixel.
frames, evidence = protruding_frames(depth_m, K, matched_pixel, geometry)
```

`selected_region_mask` is a perception result, such as a SAM candidate confirmed
by the generic task-conditioned selector. A surrounding image window proposes
backing-plane candidates; Gemini confirms the backing role. The window and local
part do not become a complete owner mask. The same geometric definition must be
used offline and at runtime. Missing or inconsistent evidence is an error.

The plane-to-protrusion direction gives the signed outward normal. A reliable
observed front band supplies an **unsigned** line; fitting pixels and valid
anchor pixels remain distinct observed masks. The functional origin is the
explicit matched surface pixel with its measured depth, never the region mean.
Returned frames use `y = inward`, `z = +/-line`, and `x = y cross z`. The two
orientations record unresolved line sign; they do not certify physically
interchangeable or collision-free grasps. The estimator requires an elongated,
approximately plane-parallel protrusion and sufficient metric separation at its
5 mm surface tolerance; it is not the geometry model for every graspable object.

For a source frame `Fd` in demonstration owner coordinates and each current
camera-local frame `Fn`, pass `region_transform = Fn @ inverse(Fd)` and
`carrier_pose = T_output_camera` to the existing independent `ground_program`.
This camera pose is a coordinate gauge, not a recovered object pose. Keep both
complete candidates when line sign is unresolved. The original timeline,
closure events and protected suffix remain governed by the grasp consumer;
actual holding, IK, collision and execution still require current verification.
<!-- END observed local protrusion geometry for independent grasp grounding. -->
