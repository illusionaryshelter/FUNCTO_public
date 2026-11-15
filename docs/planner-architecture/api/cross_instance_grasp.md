# `manipulation/cross_instance_grasp.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../16-cross-instance-grasp.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `exact_native_rows`

[定义位置](../../../manipulation/cross_instance_grasp.py#L22)。

输入签名：

```python
def exact_native_rows(rows):
```

实现说明：

> Group equal full native rows without rounding, NMS or score filtering.
>
> Every original field participates, including quality and object identity.
> First-occurrence order preserves the existing stable ranking winner. The
> inverse map reconstructs the original row order from the retained rows.

显式输出分支：

```python
(first[order], sorted_to_stable[inverse], counts[order])
```

读取的对象属性：

```python
np.unique
np.argsort
np.empty_like
np.arange
```

写入目标：

```python
_
first
inverse
counts
order
sorted_to_stable
sorted_to_stable[order]
```

直接调用符号：

```python
np.unique
np.argsort
np.empty_like
np.arange
len
```

## `NoGraspCandidates`

[定义位置](../../../manipulation/cross_instance_grasp.py#L36)。

```python
class NoGraspCandidates(RuntimeError):
```

类字段：

无显式条目。

实现说明：

> A completed native prediction exposed no candidates to relation ranking.

## `NoGraspCandidates.__init__`

[定义位置](../../../manipulation/cross_instance_grasp.py#L39)。

输入签名：

```python
def __init__(self, diagnostics):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
super().__init__
```

写入目标：

```python
self.diagnostics
```

直接调用符号：

```python
super().__init__
super
```

## `generate_and_rank`

[定义位置](../../../manipulation/cross_instance_grasp.py#L43)。

输入签名：

```python
def generate_and_rank(points_camera_m, source_G, evidence, backend, seed=0, *, T_world_camera, region_mask, approach_thresh):
```

实现说明：

> Bind each distinct native row once in first-occurrence order.
>
> ``points_camera_m`` contains observed owner points in the declared CV
> camera frame, stored as float32. It is passed directly to the SDK, so the
> recorded input domain and actual quantized input share the same coordinates.
> Camera calibration determines the world binding. The SDK supplies
> each native prediction's score, insertion depth, height and opening.
> ``region_mask`` is a nonempty bool array in exactly this input-point order.
> It retains the original region identities in the supplied owner order.
> ``approach_thresh`` is radians.
> The existing source relation supplies the G+Z approach direction.
>
> The proxy collision filter is disabled: official robot geometry and observed
> TSDF queries remain separate consumers with explicit finger joint inputs.
>
> Native pi-periodic labels are bound to two distinct actual Panda poses.
> Each retains the original native label-class score. The complete prior E
> enters soft position/orientation ranking after those actual poses exist.
> The prior remains separate from all native model predictions.

显式输出分支：

```python
dict(grasps_world=grasps, tool_poses_world=tools, quality=quality, sources=sources, candidate_ids=identifiers, candidate_records=records, rejected_candidates=[], context_proposal=proposal, prior_origin='preserved_region_relation_approach_steering', pool_diagnostics=diagnostics, prepared_points_world=prepared_world, preprocessing=preprocessing, generation=generation, ranking=ranking, binding=backend.T_tool_grasp.copy(), model_sources=backend.model_sources, native_label_rotations=label_rotations, native_row_indices=native_indices, native_label_indices=label_indices, region_anchor_points_world=region_anchors, native_first_row_indices=first_rows, native_raw_to_unique=raw_to_unique, native_duplicate_counts=duplicate_counts, device=backend.device, geometry_source=backend.geometry_source, candidate_joint_positions_m=joints, candidate_widths_m=widths, candidate_depths_m=depths, candidate_heights_m=heights, native_grasp_rows=native_rows, native_poses_camera=native_poses, native_tip_positions_world_m=tips_world, model_input_points_camera_m=points_camera, steering_region_mask=region_mask, steering_prior_grasp_world=proposal, steering_approach_camera=approach_camera, steering_prior_tool_world=tool_prior, position_scale_m=position_scale, model_region_mask=model_mask, model_seed_indices=seed_indices)
```

读取的对象属性：

```python
np.asarray
np.float32
camera[:3, :3].T
backend.predict
camera.copy
region_mask.sum
backend.prepared_points_camera
backend.model_sources
backend.geometry_source
backend.prepared_region_mask
backend.sampled_seed_indices
model_mask.sum
np.unique
model_mask[seed_indices].all
np.linalg.inv
np.linalg
backend.T_tool_grasp
np.flatnonzero
NATIVE_LABEL_ROTATIONS.copy
np.repeat
np.tile
np.arange
native[native_indices].copy
native[native_indices, 4:13].reshape(-1, 3, 3).copy
native[native_indices, 4:13].reshape
native[native_indices, 13:16].copy
np.eye
native[native_indices, 1].copy
native[native_indices, 2].copy
native[native_indices, 3].copy
native[native_indices, 0].copy
backend.tip_anchor_grasp
backend.joint_positions_for_widths
np.full
np.asarray(source_G)[:3, :3].T
diagnostics.update
backend.T_tool_grasp.copy
backend.device
```

写入目标：

```python
started
camera
points_camera
region_mask
proposal
approach_camera
controls
backend.prepared_points_camera
backend.prepared_region_mask
backend.sampled_seed_indices
native
preprocessing
prepared_world
count
diagnostics
seed_indices
model_mask
sampling
diagnostics['steering']
native_count
first_rows
raw_to_unique
duplicate_counts
unique_count
label_rotations
label_count
native_indices
label_indices
native_rows
native_rotations
native_centers
native_poses
native_poses[:, :3, :3]
native_poses[:, :3, 3]
widths
heights
depths
quality
rotations_world_native
centers_world_native
tips_world
grasps
grasps[:, :3, :3]
grasps[:, :3, 3]
tools
joints
identifiers
row
label
sources
tool_prior
position_scale
ranking
source_anchor
region_anchors
records
index
generation
diagnostics['generation']
```

直接调用符号：

```python
perf_counter
np.asarray
context_grasp_proposal
dict
float
backend.predict
len
camera.copy
int
region_mask.sum
NoGraspCandidates
model_mask.sum
np.unique
bool
model_mask[seed_indices].all
np.linalg.inv
np.flatnonzero
exact_native_rows
NATIVE_LABEL_ROTATIONS.copy
np.repeat
np.tile
np.arange
native[native_indices].copy
native[native_indices, 4:13].reshape(-1, 3, 3).copy
native[native_indices, 4:13].reshape
native[native_indices, 13:16].copy
np.eye
native[native_indices, 1].copy
native[native_indices, 2].copy
native[native_indices, 3].copy
native[native_indices, 0].copy
backend.joint_positions_for_widths
zip
np.full
rank_candidates
str
range
diagnostics.update
backend.T_tool_grasp.copy
```

## `source_capture_grasp`

[定义位置](../../../manipulation/cross_instance_grasp.py#L207)。

输入签名：

```python
def source_capture_grasp(skill, source_frame_world, *, T_world_eef):
```

实现说明：

> Return the demonstrated tool at the source image's owner placement.
>
> H_g is the complete source functional-frame-from-tool motion. Sampling its
> declared interaction time and composing source F supplies the same nominal
> image-capture tool used by native grasp steering and orientation transfer.
> The source owner pose and C_D provide a separately recorded comparison.
> Both tracks are legitimate source skill representation.

显式输出分支：

```python
(relative, capture_tool, dict(T_source_capture_tool=capture_tool, T_source_capture_recorded_tool=recorded_capture_tool, T_source_capture_body=source['T_world_body_at_image'], source_image_time_s=float(source['timestamp_s']), source_interaction_time_s=interaction, source='complete source skill F_s H_g(t_interaction) at image capture', nominal_capture_max_matrix_error=float(np.max(np.abs(recorded_capture_tool - capture_tool)))))
```

读取的对象属性：

```python
np.eye
np.flatnonzero
np.asarray
np.max
np.abs
```

写入目标：

```python
representation
source
interaction
source_reference
index
relative
capture_tool
owner_relation
recorded_capture_tool
```

直接调用符号：

```python
float
generate_reference
np.eye
int
np.flatnonzero
np.asarray
sample_poses
dict
np.max
np.abs
```

## `record_owner_model_input`

[定义位置](../../../manipulation/cross_instance_grasp.py#L240)。

输入签名：

```python
def record_owner_model_input(arrays, T_world_camera, input_camera_view_id):
```

实现说明：

> Map region elements into the independent owner input by exact identity.
>
> ``target_element_ids`` indexes the region-reference domain;
> ``target_owner_element_ids`` is paired with the owner point input in its
> original order. A mapped region row outside that owner evidence retains
> its element ID, source-row ID and unsupported state. Coordinates never
> establish this association, including for coincident measured points.

显式输出分支：

```python
(points_camera, region_mask, provenance)
```

读取的对象属性：

```python
np.asarray
owner_ids.ndim
owner_ids.tolist
np.intp
region_ids.shape
source_ids.shape
np.fromiter
lookup.get
region_ids.tolist
((owner - camera[:3, 3]) @ camera[:3, :3]).astype
np.float32
np.zeros
arrays.update
owner.copy
owner_ids.copy
region_ids.copy
source_ids.copy
region_ids[~supported].copy
source_ids[~supported].copy
region_mask.sum
supported.sum
(~supported).sum
```

写入目标：

```python
camera
owner
owner_ids
lookup
region_ids
source_ids
owner_rows
identity
supported
points_camera
region_mask
region_mask[owner_rows[supported]]
provenance
```

直接调用符号：

```python
np.asarray
len
ValueError
dict
zip
owner_ids.tolist
range
np.fromiter
lookup.get
region_ids.tolist
((owner - camera[:3, 3]) @ camera[:3, :3]).astype
np.zeros
arrays.update
owner.copy
owner_ids.copy
region_ids.copy
source_ids.copy
region_ids[~supported].copy
source_ids[~supported].copy
int
region_mask.sum
str
supported.sum
(~supported).sum
```

## `transport_orientation`

[定义位置](../../../manipulation/cross_instance_grasp.py#L290)。

输入签名：

```python
def transport_orientation(rotation_query_reference, rotation_world_reference_camera, rotation_world_query_camera, rotation_source_tool):
```

实现说明：

> Compose raw relative object rotation, camera calibration and source E.

显式输出分支：

```python
object_rotation @ np.asarray(rotation_source_tool)
```

读取的对象属性：

```python
Rotation.from_matrix(rotation_query_reference).as_matrix
Rotation.from_matrix
np.asarray
np.asarray(rotation_world_reference_camera).T
```

写入目标：

```python
camera_rotation
object_rotation
```

直接调用符号：

```python
Rotation.from_matrix(rotation_query_reference).as_matrix
Rotation.from_matrix
np.asarray
```

## `_generation_arrays`

[定义位置](../../../manipulation/cross_instance_grasp.py#L301)。

输入签名：

```python
def _generation_arrays(generated, selected_index):
```

实现说明：

> Publish every native candidate and its original scores in array form.

显式输出分支：

```python
arrays
```

读取的对象属性：

```python
fields.items
arrays.update
np.asarray
```

写入目标：

```python
fields
arrays
output
source
ranking
name
```

直接调用符号：

```python
dict
fields.items
arrays.update
np.asarray
```

## `run_cross_instance_grasp`

[定义位置](../../../manipulation/cross_instance_grasp.py#L333)。

输入签名：

```python
def run_cross_instance_grasp(skill, region_result, views, *, feedback, robot_fixture, owner_id, mapping_settings, max_tilt_rad, field_margin_m, approach_thresh_rad, checkpoint_path=None):
```

实现说明：

> Return every final E/B hypothesis with native prediction and field evidence.
>
> ``skill`` is already loaded. ``region_result`` is the completed public
> cross-instance region result; its F and identity-preserving assignments
> supply the grasp prior and native region steering mask. ``views`` are the
> public calibrated views in that same order. Feedback is the named robot
> measurement at image capture. Robot fixture and mapping settings explicitly
> declare the official model and observed-background integration parameters.
>
> The original equal-weight native-quality/full-tool-rotation/position score
> selects one actual AnyGrasp E. Each real UNOPose batch row replaces only
> its rotation. Final two-axis swing preserves the same origin. All final
> view hypotheses remain available for complete independent planning; task F
> stays distinct from B = E @ inverse(H_g(t_interaction)). The caller uses
> the shared reference generator to compose B with the complete source H_g.

显式输出分支：

```python
dict(status='completed', hypotheses=hypotheses, arrays=arrays, report=report)
```

读取的对象属性：

```python
arrays['view_ids'].tolist
geometry.grasp_generation_geometry
source_provenance.update
Path(__file__).resolve().parents
Path(__file__).resolve
error.diagnostics.update
error.diagnostics
arrays.items
name.startswith
generated['tool_poses_world'][selected_index].copy
generated['candidate_joint_positions_m'][selected_index].copy
np.asarray
arrays.update
np.repeat
np.stack
source['image'].convert
view['image'].convert
np.eye
original_tool.copy
initial_poses.append
orientation_rows.append
np.count_nonzero
predicted.items
prepared['arrays'].items
row.update
refinement_rows.append
hypotheses.append
np.linalg.inv
np.linalg
```

写入目标：

```python
started
arrays
context
observations
view
relative_goal
source_tool
source_provenance
geometry
model_geometry
source_G
weights
backend
points_camera
region_mask
model_input
model_input['owner_evidence']
generated
name
value
rank
selected_index
candidate_id
original_tool
native_fingers
measured
entrance_fingers
selection
generation_seconds
phase
static_backgrounds
_
static_background_admission
static_arrays
static_reports
static_seconds
source
bank
reference_observation
queries
predicted
orientation_rows
initial_poses
index
raw
raw[:3, :3]
raw[:3, 3]
tool
tool[:3, :3]
arrays['unopose_tool_poses_world']
unopose_seconds
prepared
scene_seconds
hypotheses
refinement_rows
source_row
initial
row
report
```

直接调用符号：

```python
ValueError
perf_counter
dict
list
arrays['view_ids'].tolist
source_capture_grasp
load_robosuite_panda_tool
geometry.grasp_generation_geometry
source_provenance.update
Path(__file__).resolve
Path
AnyGraspBackend
record_owner_model_input
generate_and_rank
error.diagnostics.update
arrays.items
name.startswith
int
str
generated['tool_poses_world'][selected_index].copy
generated['candidate_joint_positions_m'][selected_index].copy
zip
np.asarray
arrays.update
_generation_arrays
background_model_masks
query_tool_candidates
np.repeat
np.stack
source['image'].convert
view['image'].convert
predict_batch
enumerate
np.eye
original_tool.copy
transport_orientation
initial_poses.append
orientation_rows.append
float
np.count_nonzero
predicted.items
prepare_grasp_scene
prepared['arrays'].items
refine_pose
row.update
group_state_checks
refinement_rows.append
hypotheses.append
np.linalg.inv
```
