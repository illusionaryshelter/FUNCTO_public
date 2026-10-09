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

<!-- BEGIN native articulation coordinate goals. -->
An `articulation` relation can instead use explicit `joint_coordinate` events
with a caller-bound `prismatic` or `revolute` `scene.motion_model`. This expresses
a functional joint state directly, without constructing artificial two-sided
region frames or supplying `scene.reference_pose`:

- `demo.motion_coordinates`: the full original `N` or `N×1` signed scalar
  sequence, with nonempty `motion_coordinates_source`. Distances are metres;
  revolute angles are unwrapped radians. Keep release/retreat observations.
- `demo.functional_events`: each event retains `event_id`, `frame_index`,
  `source`, and declares `constraint_kind="joint_coordinate"`, a length-one
  `coordinate_value` matching that source observation, and positive
  `coordinate_tolerance` in the same unit.
- `scene.functional_bindings[event_id]`: `current_coordinates` is a length-one
  target in the **current model's** signed coordinate convention, accompanied
  by `carrier_id`, `reference_id`, and its evidence `source`.
- `scene.carrier_mapping` and `carrier_mapping_source`: an explicit SE(3) map
  from source carrier coordinates to current carrier coordinates, used when
  transporting the released-hand reference. Identity requires evidence of
  the same coordinate convention; it is never assumed.

<!-- BEGIN native prismatic progress and post-event displacement contract. -->
The soft coordinate reference is `q_demo - q_demo[0] + q_current_start`;
functional targets are enforced by current FK. Between native prismatic hard
events, source progress `(q_demo[i]-q_demo[a])/(q_demo[b]-q_demo[a])` maps to
the bound current endpoints. This scales each source displacement by the
interval's target/source displacement ratio, retaining its pauses and, for a
positive ratio, the signs of its increments. Equal-amplitude targets retain
the translated source sequence. A zero-net source interval retains its entire
internal motion when the current endpoints also coincide; binding it to a
nonzero displacement is rejected because source progress is undefined.
After the last functional event, source metric increments remain unchanged.
Timestamps, gripper events and the separately transferred released-hand retreat
are not rescaled. `motion_checks.coordinate_transfer` records the interval
scales; it does not claim that retargeted source increments retain their size.

Source/current signed directions must already correspond. Revolute transfer
keeps its existing unwrapped-coordinate and explicit winding contract. The
initial revolute coordinate is the principal angle relative to the model's
reference pose; supplied targets and limits must use that same gauge, since a
static pose cannot establish past full revolutions.
<!-- END native prismatic progress and post-event displacement contract. -->

This numerical consumer does not identify a new object's hidden joint or
its travel limits. A learned same-instance mechanism can be localized using
robot FK and a separately established fixed Holding. Transferring that prior
to a different instance needs evidence. `coordinate_limits`, when supplied,
are checked at controlled trajectory nodes; otherwise the result explicitly
reports `limit_status="not_supplied"`. Neither case certifies collisions or
execution. The existing Holding, gripper events, release boundary, independent
retreat and optional entry-corridor checks remain in force. Source action
intervals are retained as demonstration evidence, not returned as transferable
robot commands.
<!-- END native articulation coordinate goals. -->

<!-- BEGIN offline interaction references drive runtime semantic observation. -->
Compiled grounding skills live under `../collect/skills/<episode>/<segment>/`.
They contain the demonstrated program, calibrated source evidence, reviewed
functional geometry and the actual offline VLM semantic record. The runtime
entry reads its SAM query from that record; experiment scripts do not choose a
different object/part phrase or rewrite the skill description.

```python
from manipulation.skill import load_grounding_skill, observe_interaction_reference

skill = load_grounding_skill(skill_directory)
observation = observe_interaction_reference(
    skill, current_instruction,
    {"camera": {"image": current_rgb}},
)
decision = observation["selection"]["views"]["camera"]
# Preserve unresolved/alternative decisions. Pass selected masks to the
# existing geometry consumer declared by skill["manifest"]["geometry_kind"].
```

`skill.json` uses schema `functo.grounding_skill.1`, `owner_role` (`carrier` or
`reference`), and `geometry_kind` (`query_frame`, `vertices` or
`protruding_line`). Its `files` mapping names bundle-relative `semantic`,
`template`, `program`, `observations` and `image` files. `protruding_line` also
requires a `geometry` JSON attachment; its observed source front-surface mask
can be stored in the NPZ observations. All payloads stay inside the bundle.
Provenance paths are retained as records and are never opened by this loader.

`semantic.json` is the corresponding offline owner's unmodified record:
`object_description`, `interaction_part_description`, `description`, `reason`
and `interaction_reference={"text": noun_phrase, "scope": "object" | "part"}`.
Object scope selects an owner mask. Part scope selects a functional component
with the explicit runtime scope `interaction_reference`; its mask can exceed
the demonstrated contact patch and does not become an owner mask. Current
owner evidence remains a separate observation when a later consumer needs it.
No missing part detection triggers an automatic object-query substitution.

`load_grounding_skill` returns `manifest`, `semantic`, `program`,
`source_template` and `geometry` (or `None`). It injects offline semantics into
the in-memory source template and rejects duplicate handwritten semantics in
the geometry file. Only query-frame geometry retains reviewed query pixels,
their descriptions and the frame recipe. Vertex and protruding-line consumers
must not inherit discarded three-point roles. The demonstrated grasp poses,
timeline and numerical geometry consumers are unchanged. This integration does
not claim that reviewed geometric annotations have become automatic extraction.
<!-- END offline artifact to runtime observation contract. -->

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

<!-- BEGIN partial rigid-owner evidence without complete-object segmentation. -->
`selection_scope="owner_evidence"` instead selects **visible geometric samples
belonging to one target rigid entity**. Here `source_template["mask"]` is known
source-owner evidence, not a contact mask; a source `owner_mask` is optional.
Current candidates may be partial surfaces, without covering the complete owner.
The automatically assembled request requires `same_owner=true` for a nonempty
selection. Several masks then form a `selected` evidence set, never alternative
target instances; empty or unresolved decisions carry `same_owner=null`.
The caller may explicitly combine these separately returned masks as positive
owner evidence for observed-point correspondence or `screen_registrations`.
This semantic co-ownership statement does not establish measured motion rigidity.

An `owner` selection supplies object-scoped geometry. Neither a selected
interaction region nor an `owner_evidence` subset may be relabelled as a complete
object segmentation. The registration validator accepts partial owner evidence:
outside its mask ownership stays unknown, although measured free space may still
contradict a projected surface. No scope certifies mask completeness,
accessibility, grasp orientation, Holding, or execution. The existing geometric
consumers retain their separate responsibilities.
<!-- END explicit partial-owner evidence semantics. -->
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

<!-- BEGIN optional perceived part evidence with explicit current-owner context. -->
When a detector does provide a useful current part, pass
`views[view_id]["selected_owner"] = {"owner_id": ..., "mask": ..., "description":
..., "source": ...}` to `select_grounding_candidates(...,
selection_scope="interaction_region")`. The request shows the actual selected
owner separately and asks the VLM to verify part membership; pixel overlap is
not an identity test. Uncertain membership remains unresolved.

The selected observation may then be supplied as
`current_owners[role]["interaction_evidence"]`, with the same four field names.
Here `mask` is the actual selected **part** mask and `owner_id` names the already
selected owner. This optional evidence reaches the coarse VLM localization as a
separate blue overlay and structured provenance. Owner-context RGB and dense
descriptor inputs remain unchanged. Attachment points and direction witnesses
may lie outside the part; no fixed query count or part-containment rule is added.
The binding records the original mask pixels and actual model requests.
<!-- END optional part evidence does not replace whole-owner observations. -->

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

<!-- BEGIN observed orthogonal surfaces for functional-frame grounding. -->
### Functional directions from two observed surfaces

When a demonstrated functional frame is defined by two approximately
orthogonal exterior faces, observed RGB-D normals can supply its directions.
Existing semantic correspondence still identifies the two faces and the anchor:

```python
from manipulation.surface_geometry import observe_orthogonal_surface_frame

geometry = observe_orthogonal_surface_frame(
    depth_m, K, owner_mask,
    axis_pixel=matched_axis_surface_pixel,
    anchor_pixel=matched_anchor_surface_pixel,
    surface_side_source="Evidence that these are visible opaque exterior faces",
)
Fn = geometry["frame"]  # camera coordinates
```

This is an explicit geometric assumption, not a detector for every graspable
object. The two semantic roles must follow the same convention offline and at
runtime. The observer cannot recognize a role swap from depth alone. It does
not require a complete current interaction mask, ideal object dimensions or
material corner identities. It does require a perceived owner mask and two
valid semantic queries on sufficiently observed, distinct planar surfaces.

The existing 5 mm RANSAC candidates provide coarse mask membership. Each query
must uniquely belong to a different candidate and remain inside its core after
all candidate overlaps are removed and one 3×3 erosion is applied. SVD refits
the actual core points, with at least 24 noncollinear samples per face. The
caller’s visible opaque exterior-surface declaration signs normals toward the
camera; near-grazing observations are rejected. The default maximum deviation
from orthogonality is 5 degrees; this is an admissibility tolerance, not an
accuracy guarantee. All residuals, selected masks and normal corrections are
returned. Missing, ambiguous or inconsistent evidence raises an error.

For signed normals `a` (axis face) and `b` (anchor face), the frame uses
`x = a`, `z = normalize(b cross a)`, and `y = z cross x`. Its origin is the
anchor pixel’s original measured-depth backprojection, never a fitted centroid,
ideal centre or plane projection. Thus improving the direction measurement
does not remove semantic anchor localization error. Both source and current
frames must use this definition; replacing only the current frame would change
the meaning of the transferred skill.

Convert the offline camera frame into its demonstration owner coordinates to
obtain `Fd`. Pass `region_transform = Fn @ inverse(Fd)` and
`carrier_pose = T_output_camera` to the existing `ground_program`. The original
EEF relative motion, gripper intentions and protected suffix retain their
existing contract. A returned frame does not establish symmetric-face material
identity, collision clearance or actual holding.
<!-- END observed orthogonal surfaces for functional-frame grounding. -->

<!-- BEGIN unordered visible-vertex correspondence and stable consensus. -->
### Unordered vertex correspondences and joint rigid estimation

Visible vertices can provide rigid correspondences without assigning fixed
function/grasp/centre roles or requiring exactly three semantic points. The
RGB-D observer and visual association are independent of the rigid estimator:

```python
from manipulation.vertices import (
    observe_vertex_candidates, associate_vertex_candidates,
)
from manipulation.consensus import rigid_correspondence_hypotheses

source_vertices = observe_vertex_candidates(source_rgb, source_depth, source_K, source_mask)
current_vertices = observe_vertex_candidates(current_rgb, current_depth, current_K, current_mask)
association = associate_vertex_candidates(
    source_rgb, current_rgb, source_mask, current_mask,
    source_vertices, current_vertices, client=client,
)
# source_points and current_points are the measured 3D coordinates from the
# returned explicit pairs, expressed in the desired source/current frames.
consensus = rigid_correspondence_hypotheses(
    source_points, current_points, pair_ids=correspondence_ids,
    max_error=0.01,
)
```

The caller obtains the current owner mask from perception. The source can use
offline demonstration annotations. The observer retains RGB corner locations
and the actual nearby depth pixels separately; a sampled surface point is not
certified to be the exact physical vertex. Duplicate corner/depth observations
cannot add multiple votes. A shared geometry-only prompt proposes a variable
number of source/current ID pairs, without skill-name branches, asset names,
metric geometry from the VLM, fixed point roles, or image-order assumptions.
Ambiguous or insufficient visual association must remain explicit.

Three noncollinear corresponding measurements form a minimal rigid hypothesis.
The sparse estimator enumerates these subsets, checks every original pair,
and jointly refits each consensus until its fitted and accepted sets agree.
It returns all stable candidates with maximum support; lower-support outcomes
remain in diagnostics. Commonly reordering paired measurements does not assign
new axes or change their roles. Explicit correspondence is still necessary:
geometric consistency cannot disambiguate every symmetric object.

The 10 mm default is a geometric consistency gate inherited from the existing
rigid-registration interface. It is **not** a noise estimate, covariance,
calibrated confidence interval, or terminal-pose accuracy guarantee. This path
does not require a noise model or calibration dataset, and uses no uncertainty
weights or local surface fitting. All observed pairs and final residuals remain
available for review. A candidate supported by only three points is marked as
weak evidence because it has no additional supporting correspondence.

If source points are in the demonstration owner frame and current points in
the binding camera frame, each hypothesis maps source owner to current camera.
Pass it as `region_transform`, with the calibrated camera pose as
`carrier_pose`, in a candidate scene for the grasp consumer. For full-frame
manipulation, bind the reference functional frames in the same declared output
coordinates. The camera pose is a coordinate gauge, not current object GT.
Where the same rigid source surface remains applicable, submit all bound
candidate scenes together through the screened entry below. The original
timeline, closure intent and protected suffix stay with the numerical consumer;
geometric support does not establish actual holding or execution feasibility.
<!-- END unordered visible-vertex correspondence and stable consensus. -->

<!-- BEGIN actual tool versus observed-owner grasp screening. -->
### Check functional grasp candidates against the known gripper

Use `manipulation.perception.ground_grasp_candidates` for complete, already
bound functional grasp candidates. It first generates every EEF reference with
the existing consumer, then checks the actual known tool against current owner
RGB-D. Rejected and unresolved references remain in `trajectories` for inspection;
they are not relabelled as executable motions. No whole-owner registration is
inferred from the functional frame.

```python
from manipulation.perception import ground_grasp_candidates
from manipulation.tool_geometry import load_robosuite_panda_tool

tool = load_robosuite_panda_tool(tool_id=feedback["tool_id"], tool_frame=feedback["tool_frame"])
decision = ground_grasp_candidates(
    program, candidate_scenes, current_views, tool,
    current_joint_positions_m=feedback["current_joint_positions_m"],
    contact_tolerance_m=0.001,
    penetration_tolerance_m=0.00025,
)
```

The Panda factory reads the installed standalone robot asset; it never reads a
live scene or object mesh. Its external tool frame must explicitly denote that
asset's `grip_site`. Native finger coordinates are `[q1, q2]` with open values
`[0.04, -0.04]` metres, not two positive distances. Actual pad faces, palm/finger
convex collision geometry and their FK are used; the tool site is not assumed
to be a pad centre. Another robot supplies its own tool adapter, not a skill-name
branch in the checker.

Current views use `owner_id`, `reference_frame`, `depth`, `K`, `owner_mask` and
`T_output_camera`, with optional actual `interaction_mask`. Owner pixels remain
available to detect palm/finger interference even outside the selected part.
Current finger q is synchronized robot feedback, not demonstration contact q.
The original open/close intent and protected EEF suffix remain unchanged.

Finite owner depth samples are retained as position observations. Local surface
normals use only continuous, same-owner neighborhoods, including one-sided
neighborhoods at thin parts; failure to estimate a normal does not erase a depth
sample. Uncertain edge membership cannot by itself establish a hard collision,
and missing opposed surface evidence cannot establish bilateral contact.

The two explicit tolerances have different meanings: the first is an observed
near-contact band; the second permits only shallow geometric penetration before
reporting a solid conflict. The values above are the Panda experiment's declared
1 mm contact and 0.25 mm penetration settings, not camera noise estimates or
universal tolerances for every gripper. They are fixed before outcome evaluation.

The checker reports visible local grasp geometry and retains missing evidence
as unresolved. It does not certify unseen surfaces, full-scene/full-arm collision,
dynamic closing, forces or stable Holding. A freely replannable approach conflict
does not prove that the terminal grasp itself is impossible; inspect phase
diagnostics. `selected_trajectory` returns the unique supported action. Multiple
supported actions remain ambiguous; this checker does not rank path costs.
Other unresolved actions are
retained and do not veto that candidate: these are alternative grasps of one
already established owner/goal, not conflicting hypotheses about scene identity.
Resolve ambiguous object/goal bindings upstream. The motion backend and
synchronized execution feedback retain their separate responsibilities.
<!-- END functional grasp generation and physical execution remain distinct. -->

<!-- BEGIN production projective registration screening and selected trajectory. -->
### Select registered trajectories using current depth

`manipulation.perception.ground_registered_candidates` is the shared entry for
competing, fully bound rigid-owner registrations. It checks all candidates
against the same current views, then calls the existing `ground_program` for
supported or unresolved bindings. Clearly rejected bindings produce no
trajectory. A unique supported candidate with no unresolved competitors yields
`selected_trajectory`; ambiguous, insufficient or inapplicable evidence yields
`None`. Candidate names, order and path length do not choose a winner.

```python
from manipulation.perception import ground_registered_candidates

decision = ground_registered_candidates(
    program,
    candidate_scenes,  # {candidate_id: complete_scene_for_ground_program}
    source_owner_rgbd,
    current_views,
    role="reference",  # "carrier" only for an explicit physical carrier registration
    same_rigid_geometry=geometry_is_established,
    geometry_source=geometry_evidence,
    max_error=0.01,
)
trajectory = decision["selected_trajectory"]  # None means no unique selection.
```

Inputs use existing owner/frame conventions:

| Input | Required fields and meaning |
|---|---|
| `source_owner_rgbd` | `owner_id`, `depth`, `K`, `owner_mask`, `T_camera_owner`; metric camera-z depth and the observed source whole-owner surface, which may use offline annotations |
| `current_views` | Arbitrary nonempty view-ID mapping; each value has `owner_id`, `reference_frame`, `depth`, `K`, `owner_mask`, `T_output_camera`; all views observe the same current rigid owner/state in one calibrated output frame |
| `candidate_scenes` | Arbitrary nonempty candidate-ID mapping to numerical scenes; same bound owner and output frame, bindings that actually claim physical rigid registration |
| Geometry applicability | Source geometry must remain valid and the correspondence must claim the same material rigid mapping. An unchanged instance alone does not turn a local functional frame into an exact body pose; a new shape must not inherit rigid congruence |

The checked transform is derived from the **actual numerical binding**. For
grasp it is `carrier_pose @ region_transform`; for a manipulation reference it
is `reference_pose @ current_reference_frame @ inverse(source_reference_frame)`.
All functional events must imply the same rigid mapping. A separately supplied
"validation pose" cannot pass while a different pose generates the motion.
Source owner IDs must match the compiled role; current view IDs bind to the
same owner as every candidate scene. No skill/asset-name branches, fixed camera
labels or fixed candidate/semantic-point counts are used.

The lower-level `manipulation.registration_validation.screen_registrations`
accepts arbitrary candidate-ID -> output-from-source-owner transforms with the
same source/view contract. It returns per-candidate/per-view summaries and
actual pixel-ray evidence. No models, simulation or motion planners are called.
The source surface is projected using each hypothesis. A predicted surface
clearly **before** the nearest measured surface contradicts observed free space;
one **behind** it may be occluded and remains unknown. A z-buffer counts each
image ray once. Owner-mask erosion and depth-continuous 3x3 neighborhoods avoid
boundary/background comparisons without inventing a noise model or filling
hidden geometry. Positive support must be noncollinear in every supplied view;
insufficient coverage stays unresolved instead of passing on zero conflicts.

The 10 mm default ray gate and fixed 10 mm depth-continuity gate are geometric
consistency parameters, not sensor variance, permitted penetration or task
accuracy. A known geometry mismatch returns `not_applicable` and no unchecked
trajectory. An `on` support patch or a joint-coordinate goal does not by itself
define a complete rigid registration; those distinct representations retain
their existing grounding entries. This check must not impose whole-object
congruence on functional transfer between different shapes.

Selected/unresolved references carry `registration_validation` with the tested
transform, owner IDs, evidence source and per-view summaries. Frame IDs,
timestamps, grasp closure, functional events, actual-Holding requirements and
release/retreat semantics are preserved. `robot_execution_verified` remains
false: this module does not add cuRobo/OMPL, whole-robot collision, trajectory
optimization or proof of complete `inside`/physical execution.
<!-- END production projective registration screening and selected trajectory. -->

<!-- BEGIN shared resident motion backend and fixed online scene. -->
## Robot motion planning and execution

The motion contract is defined in [motion_planner_prd.md](motion_planner_prd.md).
Skills supply typed goals, process requirements, current object bindings,
finite material permissions and original event identities. Ordinary `PosePath`
samples provide initialization guidance; they do not create mandatory approach
poses, stops, motion requests or additional validation capacity.

`MotionBackend` owns a robot's native cuRobo IK and trajectory optimizer. Its
first binding initializes the fixed-capacity native models; later requests
replace values and clear optimizer/request state without rebuilding those
models. Concurrent use of the same backend is rejected while its buffers are
owned by a request. Cancellation or deadline expiry revokes a late result's
execution authorization.

```python
from manipulation.planning import MotionBackend, PlanningBudget

backend = MotionBackend(robot_configuration, robot_validator=robot_validator,
                        budget=PlanningBudget())
try:
    result = backend.plan(problem, online_scene)
    # Reuse backend.plan(...) for later, freshly observed motion requests.
finally:
    backend.close()
```

<!-- BEGIN bounded recovery and shared signed functional-event constraints. -->
Each request evaluates its initialization candidate first. A feasible candidate
proceeds directly to delivery. Otherwise it receives one primary solve and at
most one recovery: add collision check times for dense positive intervals without
an existing positive sample in the same collision channel, or use bounded OMPL
initialization when no such interval needs refinement. An already detected
collision does not claim that additional sampling will remove it. Mixed intervals
retain refinement for independently missed collisions.
Functional-multiplier updates use the existing inner iteration blocks; they do
not create another trajectory solve. Pose, coordinate and finite-support events
share this signed-inequality mechanism. Support keeps separate gap, body-height,
footprint and normal requirements, normalized by its declared metric scales;
normalization never changes their hard allowed set. Support events have no
duplicate ordinary square penalty and retain free tangent/yaw motion. There is no
post-feasibility quality stage, per-candidate export validation, witness-driven
map refinement, or solver reconstruction loop. A recorded joint reference is
explicit guidance; when native fitting changes its representation, diagnostics
say so rather than claiming the original curve was preserved.
<!-- END final acceptance still checks the original physical requirements. -->

The native optimizer, native success metrics and OMPL checks
use the same `OnlinePlanningScene` in `manipulation/online_geometry.py`:

<!-- BEGIN non-gradient search preserves the existing source-geometry authority. -->
OMPL state validity, entrance checks and final candidate checks can confirm
registered world-collision sphere proposals against the same frozen source
solids. Each search branch owns its confirmation context and release history;
unknown-space, contact and task requirements remain independent hard checks.
Native self-collision states use the bound finger clock. Differentiable optimizer
queries retain their existing sphere costs and do not run CPU confirmation.
<!-- END source confirmation does not supply observations or contact permissions. -->

- `FrozenESDF` retains the current Mapper field and source-validity bits.
  <!-- BEGIN direct complete-owner admission without discarded observation maps. -->
  Each observed background/owner partition has one fixed field. Explicit
  complete `CollisionOwner` geometry may enter `build_scene_from_rgbd` through
  `complete_owners`; it retains its current frame, native materials and semantic
  mask partition without allocating an observation field that would be discarded.
  Its RGB-D rays still contribute only their original observed-free evidence.
  Per-owner resolution remains explicit, including the complete owner's contact
  grid spacing. Owner support is
  prepared from the observed extent and registered robot sphere radius before
  solving; requests cannot enlarge it after a failed query.
  <!-- END direct complete-owner admission. -->
  <!-- BEGIN semantic ownership uncertainty is handled before fusion. -->
  Online background fusion excludes a one-pixel, eight-neighbor outer ring
  around admitted observed owner pixels. These endpoints have uncertain
  ownership: they are added to neither an owner mask nor the background.
  Existing valid measured free-ray prefixes remain independent and unchanged.
  This bounded image-grid assumption does not repair arbitrary segmentation
  errors or certify excluded geometry as free; the existing unknown-space
  policy still applies. Every request builds fresh per-partition Mappers from
  its RGB-D snapshot. Rebuilding with corrected partitions removes old fusion
  pollution; restoring an old frozen scene binding does not repair its map.
  Observation diagnostics retain the actual background and boundary masks.
  <!-- END no collision-result erasure or object-specific mask growth. -->
- `FrozenPrimitives` retains calibrated complete boxes/cylinders and their
  declared transforms. Robot sphere layout and self-collision exclusions are
  bound to the same registered robot model.
- Unknown-space policy explicitly lists finite regions where conditional
  attempts are allowed. Known occupied geometry remains active there, and an
  out-of-field query requires a scene update. Conditional permission never
  creates observed-free evidence or a successful task fact.
- `ContactRule` names a finite material feature, target owner, target region,
  interval, normal and gap allowance. Contact subtraction preserves every
  represented cell outside that finite permission, including cells covered by
  the same robot sphere. A whole hand/owner pair exemption is not accepted.
  <!-- BEGIN exact target-field storage sharing. -->
  Rules with exactly equal target regions may share their immutable distance
  arrays. Material identity, activation intervals and per-sphere rule subsets
  remain independent; restored repeated fields must agree exactly. CPU and GPU
  storage count each shared field once under the existing contact-bank limit.
  <!-- END no permission or geometric tolerance is merged by storage sharing. -->

<!-- BEGIN native confirmation of registered actor/world sphere candidates. -->
Positive registered actor/world sphere costs are collision candidates. Final candidate
acceptance uses `online_native_confirmation.OnlineCollisionConfirmation`:
only those candidates are
checked with registered native robot solids or complete known-owner box unions,
against the same frozen observed TSDF source components (or analytic frozen
primitives). Owner registration proves every box cell is covered by its assigned
sphere and binds finite material geometry to that complete solid at scene
admission, even when no positive candidate exists. Confirmation uses the same
query's attachment/release pose. One spatial index is prepared per request.
Effective finite contact rules remove only wholly contained source components
for the exact registered material; unknown-space, field support,
contact conditions and self-collision constraints retain their own verdicts.
Numerical boundary or unsupported native results keep the sphere rejection.
Optimization still uses the differentiable sphere costs. This is a discrete
observed-model check, not a continuous or unobserved-space safety certificate.
Native finite materials support full planar extrusions and true box edges.
An edge retains its two endpoints, supporting normal cone and complete box;
a conservative depth bound preserves its original numerical lateral tolerance.
Optional contacts constrain their normal only where the complete body can
intersect removed target material. Required contacts and task support intervals
retain their original normal requirements. A grasp handoff rigidly re-expresses
its finite patches and binds the new request's Holding condition; it does not
copy the previous close-event latch.
<!-- END native confirmation of registered actor/world sphere candidates. -->

<!-- BEGIN current support is distinct from future placement material. -->
`initial_support.InitialSupportRelation` describes an existing finite support
between a complete attached owner and an independently observed static plane.
It does not reuse a future placement's material edges, gap or normal targets.
The target's entire observed source must belong to this independently selected
support layer; supplying that partition is the caller's responsibility.
Preparation freezes only source components intersecting the owner's entrance
native geometry. This inventory is finite without a future material footprint
or an ideal-plane clipping condition on the reconstructed source vertices.
The differentiable field and native confirmation remove the same component
IDs for that actor/target pair. A separate whole-owner minimum-height constraint
prevents nominal penetration from increasing beyond the approximate entrance
model; it does not certify physical penetration depth. Separation, rotation and
return require no new phase or latch. Other owners, background, source components
not intersecting at entrance, and final placement conditions retain their checks.
The declaration cannot identify obstacles incorrectly mixed into a plane layer.
Frozen bindings store the relation and prepared
field, and workspace admission includes both field and vertex-query memory.
<!-- END current support does not redefine grasp or placement success. -->

`planning_observation.build_scene_from_rgbd` creates this production model.
`build_audit_scene_from_rgbd`, source TSDF geometry, complete-volume coverage
and detailed source-scene component checkers are separate audit tools. The
bounded native confirmation above queries only local source components for
positive candidates; it does not reconstruct or scan the whole source scene.

`PlanningProblem.bound_motion_block` uses actual evidence dependencies to bound
one request while preserving original progress and event identities. Coupled
process/velocity requirements cannot be split by inserting a new stop. Actual
T evidence at a boundary permits a newly bound request; it does not authorize
an unplanned suffix of an old curve. Validated forward progress starts another
request allowance while retaining cumulative work and physical-step history.

Delivery checks the exact native-to-Hermite representation, measured entry,
continuous named joint position/velocity/acceleration and bound clock once.
<!-- BEGIN planning geometry and physical execution have distinct duties. -->
`PlanningResult.execution_binding` retains the exact curve and frozen planning
scene for authorization and provenance. `TrajectoryExecutor` does not rebuild
or query that frozen world during preflight or physical execution.
`RobotStateValidator` checks measured joint limits, model/FK identity and the
registered required native robot/fixed-installation collision pairs only.
Tracking, event timing, real arrival before close, fresh observations, current
Holding dependencies and effect requirements remain execution duties. An
attachment-dependent owner needs current Holding evidence or a current owner
pose; entrance facts cannot silently persist into later samples. Commanded
fingers remain separate from measured finger positions. Frozen occupied cells
and their conservative envelopes cannot establish actual physical contact.

Array records admit at most 64 MiB per array. Contact fields retain their
independent 256 MiB aggregate limit; overall planning, archive and process
resource limits remain independent checks.
<!-- END physical outcomes require current evidence, not frozen geometry. -->

CPU contracts live under `tests/test_motion_backend_contract.py`,
`tests/test_online_geometry.py`, and the affected planning/execution tests.
The recorded-task CLI in `tests/planning/run_recorded_skill.py` owns the backend
lifetime and requires explicit finite-contact and unknown-policy inputs.
Its result recorder saves only returned artifacts and never initiates another
solve or validation. Dataset-specific selections and user-run experiments
belong in the workspace's `sim-version-plans` directories.
<!-- END shared resident motion backend and fixed online scene. -->
