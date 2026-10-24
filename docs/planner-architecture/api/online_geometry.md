# `manipulation/online_geometry.py` 实现契约

代码基线：`13fefbc`。本次文档引用提交仅回填实际实现基线，源码与该代码提交相同。组件正文描述当前实现；阶段验证位于项目外 `sim-version-plans`。
职责和数据语义见[组件说明](../02-observation-and-collision.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_plain`

[定义位置](../../../manipulation/online_geometry.py#L33)。

输入签名：

```python
def _plain(value):
```

显式输出分支：

```python
value.tolist()
[_plain(row) for row in value]
{key: _plain(row) for (key, row) in value.items()}
value.item()
value
```

读取的对象属性：

```python
np.ndarray
value.tolist
value.items
np.generic
value.item
```

写入目标：

```python
row
key
```

直接调用符号：

```python
isinstance
value.tolist
_plain
value.items
value.item
```

## `_matrices`

[定义位置](../../../manipulation/online_geometry.py#L45)。

输入签名：

```python
def _matrices(pose):
```

实现说明：

> Snapshot the same differentiable rigid poses for interval queries.

显式输出分支：

```python
result
```

读取的对象属性：

```python
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy
np.broadcast_to
np.eye
position.shape
Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape
Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.reshape
```

写入目标：

```python
position
quaternion
value
result
result[..., :3, 3]
result[..., :3, :3]
```

直接调用符号：

```python
value.detach().cpu().numpy
value.detach().cpu
value.detach
np.broadcast_to(np.eye(4), (*position.shape[:-1], 4, 4)).copy
np.broadcast_to
np.eye
Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix().reshape
Rotation.from_quat(quaternion.reshape(-1, 4)[:, [1, 2, 3, 0]]).as_matrix
Rotation.from_quat
quaternion.reshape
```

## `OnlinePlanningScene`

[定义位置](../../../manipulation/online_geometry.py#L55)。

```python
class OnlinePlanningScene():
```

类字段：

```python
binding_record = to_execution_binding
```

实现说明：

> Known robot collision entities and original observed task fields.
>
> Robot geometry is shared with the state validator and placed by complete
> joint FK. Unknown-space decisions for moving scene owners retain the
> supplied policy and the original measured free-ray coverage.

## `OnlinePlanningScene.__init__`

[定义位置](../../../manipulation/online_geometry.py#L64)。

输入签名：

```python
def __init__(self, *, background, owners, coverage, reference_frame, robot_model, fixed_geometry_names, tool_geometry_names, contacts=(), gripper=None, has_gripper=True, unknown_policy, initial_support_relations=(), numerical_position_tolerance_m=1e-06, numerical_rotation_tolerance_rad=1e-06):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
owner.query
robot_model.geometry_names
self.fixed_geometry_names
self.tool_geometry_names
owner.owner_id
self.owners
dict.fromkeys
robot_model._geometries.values
robot_model._geometries
robot_model._geometries.items
np.eye
bounds.flags
self._native_local_bounds
np.asarray
lower.shape
upper.shape
np.any
rule.rule_id
self.contacts
rule.target_owner
rule.actor
rule.robot_body
rule.interval
owner.attachments
relation.interval
self.initial_support_relations
row.relation_id
row.actor
row.target_owner
self.numerical_position_tolerance_m
```

写入目标：

```python
owner
self.background
self.owners
self.coverage
self.reference_frame
self.robot_model
self.fixed_geometry_names
self.tool_geometry_names
robot_ids
self.required_robot_links
frame
_
self._native_local_bounds
name
geometry
bounds
bounds.flags.writeable
self._native_local_bounds[name]
self.contacts
self.gripper
self.has_gripper
self.unknown_policy
region
lower
upper
owner_ids
rule
self.numerical_position_tolerance_m
self.numerical_rotation_tolerance_rad
self.required_progress_anchors
x
relation
self.initial_support_relations
self._initial_support_models
by_owner
row
self._fixed_release_poses
self.observation_report
self.observation_partition_masks
self.grasp_contact_activations
self.robot_pairs
self.robot_validator
self.kinematics
self.check_request
self.last_query_report
self._objective_material_samples
```

直接调用符号：

```python
isinstance
any
TypeError
tuple
set
len
ValueError
dict.fromkeys
robot_model._geometries.values
robot_model._geometries.items
_native_bounds
_transform
np.eye
deepcopy
np.asarray
np.any
float
sorted
PreparedInitialSupport
```

## `OnlinePlanningScene.bind_online_robot`

[定义位置](../../../manipulation/online_geometry.py#L142)。

输入签名：

```python
def bind_online_robot(self, *, validator, check_request=None):
```

实现说明：

> Reuse the validator's complete model and calibrated Pinocchio FK.

显式输出分支：

无显式条目。

读取的对象属性：

```python
validator.checker
self.robot_model
validator.pairs
validator.checker.model
self.required_robot_links
validator.T_reference_base
```

写入目标：

```python
self.robot_validator
self.robot_pairs
self.kinematics
self.check_request
```

直接调用符号：

```python
ValueError
DifferentiableRobotKinematics
```

## `OnlinePlanningScene.clone_for_rollout`

[定义位置](../../../manipulation/online_geometry.py#L154)。

输入签名：

```python
def clone_for_rollout(self):
```

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.owners
self._fixed_release_poses
self.gripper
```

写入目标：

```python
result
result.owners
owner
result._fixed_release_poses
result.gripper
result.last_query_report
```

直接调用符号：

```python
copy
tuple
dict
```

## `OnlinePlanningScene._release_pose`

[定义位置](../../../manipulation/online_geometry.py#L163)。

输入签名：

```python
def _release_pose(relation, position, quaternion, progress):
```

实现说明：

> Gather exact release FK; a missing anchor is an explicit violation.
>
> The first occurrence is the physical release. Later samples on the
> same progress plateau may move the hand while the owner stays fixed.
> When absent, the gathered numeric value is not accepted as
> a release prediction: ``found`` is false and subsequent samples fail.
> Keeping arithmetic finite avoids poisoning an earlier, still-attached
> query's derivatives when it does not need a release state yet.

显式输出分支：

```python
((p + quaternion_rotate(q, _tensor(relation.position_tool_owner, p)), quaternion_multiply(q, _tensor(relation.quaternion_tool_owner, q))), matches.any(-1))
```

读取的对象属性：

```python
relation.release_condition_id
(progress - relation.interval[1]).abs
relation.interval
matches.to(torch.int64).argmax
matches.to
torch.int64
torch.gather
index[:, None, None].expand
relation.position_tool_owner
relation.quaternion_tool_owner
matches.any
```

写入目标：

```python
matches
index
p
q
```

直接调用符号：

```python
ValueError
(progress - relation.interval[1]).abs
matches.to(torch.int64).argmax
matches.to
torch.gather
index[:, None, None].expand
quaternion_rotate
_tensor
quaternion_multiply
matches.any
```

## `OnlinePlanningScene.with_release_history`

[定义位置](../../../manipulation/online_geometry.py#L184)。

输入签名：

```python
def with_release_history(self, position, quaternion, progress):
```

实现说明：

> Copy release FK from an already selected prefix into a child branch.
>
> Future releases remain variables. The first reached occurrence of a
> release owns its pose, including on a progress plateau; extending a
> branch cannot replace it with the later hand pose. Full optimization
> still uses the original scene, where releases remain shared decisions.

显式输出分支：

```python
result
```

读取的对象属性：

```python
self.clone_for_rollout
self.owners
owner.attachments
owner.owner_id
relation.release_condition_id
result._fixed_release_poses
(progress >= relation.interval[1] - 1e-07).any
relation.interval
self._release_pose
found.all
```

写入目标：

```python
result
owner
index
relation
key
pose
found
result._fixed_release_poses[key]
```

直接调用符号：

```python
self.clone_for_rollout
enumerate
bool
(progress >= relation.interval[1] - 1e-07).any
self._release_pose
found.all
ValueError
```

## `OnlinePlanningScene._permission`

[定义位置](../../../manipulation/online_geometry.py#L206)。

输入签名：

```python
def _permission(self, rule, actor_pose, target_pose, progress, coordinates, *, uncertain_hand=None):
```

显式输出分支：

```python
(policy, residual * active.to(progress.dtype), (active & ~certain).to(progress.dtype))
```

读取的对象属性：

```python
rule.feature_points
rule.feature_joint_translation
torch.einsum
torch.ones_like
torch.bool
quaternion_conjugate(target_pose[1]).unsqueeze
target_pose[0].unsqueeze
rule.feature_normal
rule.patch_axes
torch.atan2
torch.linalg.vector_norm
torch.linalg
torch.linalg.cross
(actor_normal * target_normal).sum
torch.relu
rule.normal_tolerance_rad
rule.numerical_angle_tolerance_rad
rule.interval
actor_pose[1].unsqueeze
rule._feature_edge_normals
inverse_target.unsqueeze
rule.numerical_position_tolerance_m
(edge_normals * origin.unsqueeze(-2)).sum
origin.unsqueeze
rule._feature_edge_offsets
(material_normal.unsqueeze(-2) * target_local).sum
material_normal.unsqueeze
rows.append
bounds.append
(sign * material_heights).max(-1).values.unsqueeze
(sign * material_heights).max(-1).values
(sign * material_heights).max
rule.modeling_allowance_m
rule.patch_origin
rule.patch_half_size
axis.expand
progress.shape
(axis @ patch_origin + width + tolerance).expand
rule.feature_kind
rule._feature_tangent
(tangent * patch_normal).sum(-1).unsqueeze
(tangent * patch_normal).sum
(origin * (patch_normal - gap_axis)).sum
(material_normal * patch_normal).sum
cosine.unsqueeze
rule.gap_bounds
(sign * gap_axis).unsqueeze
(limit - sign * gap_offset + tolerance).unsqueeze
torch.cat
rule.robot_body
active.to
progress.dtype
(active & ~certain).to
```

写入目标：

```python
points
mapping
certain
world
target_local
actor_normal
target_normal
angle
normal
active
inverse_target
material_normal
edges_world
edge_normals
origin
tolerance
rows
bounds
material_heights
sign
patch_axes
patch_origin
column
width
axis
patch_normal
tangent_world
tangent
gap_axis
gap_offset
cosine
limit
region_normals
region_bounds
residual
policy
```

直接调用符号：

```python
_tensor
torch.einsum
ValueError
torch.ones_like
_world_points
quaternion_rotate
quaternion_conjugate(target_pose[1]).unsqueeze
quaternion_conjugate
target_pose[0].unsqueeze
torch.atan2
torch.linalg.vector_norm
torch.linalg.cross
(actor_normal * target_normal).sum
torch.relu
_interval
actor_pose[1].unsqueeze
inverse_target.unsqueeze
(edge_normals * origin.unsqueeze(-2)).sum
origin.unsqueeze
(material_normal.unsqueeze(-2) * target_local).sum
material_normal.unsqueeze
rows.append
bounds.append
(sign * material_heights).max(-1).values.unsqueeze
(sign * material_heights).max
enumerate
axis.expand
(axis @ patch_origin + width + tolerance).expand
(tangent * patch_normal).sum(-1).unsqueeze
(tangent * patch_normal).sum
(origin * (patch_normal - gap_axis)).sum
(material_normal * patch_normal).sum
cosine.unsqueeze
(sign * gap_axis).unsqueeze
(limit - sign * gap_offset + tolerance).unsqueeze
torch.cat
active.to
(active & ~certain).to
```

## `OnlinePlanningScene.optimizer_preferences`

[定义位置](../../../manipulation/online_geometry.py#L275)。

输入签名：

```python
def optimizer_preferences(self, progress, joint_position, joint_names):
```

实现说明：

> Squared tracking error for optimizer use only; no acceptance change.
>
> A tracking drive already specifies one executable command at every
> progress. Its hard numerical band remains in components(). This small
> objective keeps optimization attracted to that command after entering
> the feasible band, instead of stopping at a float32-sensitive boundary.
> Binary drives supply no command width and therefore no such objective.
> Material contact and functional pose freedoms are never centered here.

显式输出分支：

```python
torch.zeros_like(progress)
((joint_position[..., indices] - command) ** 2).sum(-1)
```

读取的对象属性：

```python
self.gripper
hand.controller
torch.zeros_like
hand.joint_bounds
lower.shape
progress.shape
hand.joint_names
upper.shape
tuple(joint_names).index
torch.long
((joint_position[..., indices] - command) ** 2).sum
```

写入目标：

```python
hand
lower
upper
indices
name
command
```

直接调用符号：

```python
torch.zeros_like
ValueError
hand.joint_bounds
len
tuple(joint_names).index
tuple
constant_tensor
((joint_position[..., indices] - command) ** 2).sum
```

## `OnlinePlanningScene.project_residual`

[定义位置](../../../manipulation/online_geometry.py#L299)。

输入签名：

```python
def project_residual(self, progress, joint_position, joint_names):
```

实现说明：

> Independent executable finger-drive equalities for OMPL projection.
>
> Output has J-rank(A) rows for tracking and zero rows otherwise. The
> caller must include the same native A q=b coupling in its problem;
> duplicate coupling rows are intentionally not introduced here. These
> are exact equalities, with projection convergence handled separately
> from the native controller command or task's permissible motion.

显式输出分支：

```python
joint_position[..., :0]
(joint_position[..., indices] - command) @ _tensor(hand._drive_basis.T, joint_position)
```

读取的对象属性：

```python
self.gripper
self.gripper.controller
tuple(joint_names).index
hand.joint_names
hand.joint_bounds
torch.long
hand._drive_basis.T
hand._drive_basis
```

写入目标：

```python
hand
indices
name
lower
upper
command
```

直接调用符号：

```python
tuple(joint_names).index
tuple
hand.joint_bounds
constant_tensor
_tensor
```

## `OnlinePlanningScene.validate_problem`

[定义位置](../../../manipulation/online_geometry.py#L321)。

输入签名：

```python
def validate_problem(self, problem, *, goal_identities=None):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
problem.reference_frame
self.reference_frame
problem.conditions
row.get
problem.pose_events
self.contacts
used.intersection
problem.events
event.kind
pose_ids.index
requirement.actual_relation
requirement.progress
commands[0].progress
used.update
self.owners
owner.attachments
relation.condition_id
declaration.get
owner.owner_id
relation.interval
np.linalg.inv
np.linalg
np.asarray
np.eye
relation.position_tool_owner
Rotation.from_quat(relation.quaternion_tool_owner[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
relation.quaternion_tool_owner
np.allclose
relation.release_condition_id
release.get
self.gripper
self.gripper.commands
row.skill_id
row.event_id
row.occurrence
row.kind
row.progress
```

写入目标：

```python
conditions
row
activations
pose_ids
used
ids
rule
original
commands
event
requirement
self.grasp_contact_activations
owner
relation
declaration
expected
actual
actual[:3, 3]
actual[:3, :3]
release
```

直接调用符号：

```python
ValueError
row.get
list
range
len
set
grasp_contact_rule_ids
asdict
used.intersection
event_occurrence_key
pose_ids.index
validate_grasp_contact_arrival
used.update
tuple
deepcopy
declaration.get
np.linalg.inv
np.asarray
np.eye
Rotation.from_quat(relation.quaternion_tool_owner[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
np.allclose
release.get
```

## `OnlinePlanningScene._owner_poses`

[定义位置](../../../manipulation/online_geometry.py#L378)。

输入签名：

```python
def _owner_poses(self, position, quaternion, progress, owner_poses, costs):
```

显式输出分支：

```python
poses
```

读取的对象属性：

```python
self.owners
owner.owner_id
np.asarray
position.new_tensor(matrix[:3, 3]).expand
position.new_tensor
progress.shape
quaternion.new_tensor
Rotation.from_matrix(matrix[:3, :3]).as_quat
Rotation.from_matrix
q.expand
owner.attachments
relation.release_condition_id
releases.append
self._fixed_release_poses
self._release_pose
((progress >= relation.interval[1]) & ~found[:, None]).to
relation.interval
progress.dtype
owner.pose
```

写入目标：

```python
poses
owner
matrix
p
q
poses[owner.owner_id]
releases
index
relation
pose
found
costs[f'relationship/missing_release/{owner.owner_id}/{index}']
```

直接调用符号：

```python
np.asarray
position.new_tensor(matrix[:3, 3]).expand
position.new_tensor
quaternion.new_tensor
Rotation.from_matrix(matrix[:3, :3]).as_quat
Rotation.from_matrix
q.expand
enumerate
releases.append
self._release_pose
((progress >= relation.interval[1]) & ~found[:, None]).to
owner.pose
```

## `OnlinePlanningScene._robot_self_components`

[定义位置](../../../manipulation/online_geometry.py#L402)。

输入签名：

```python
def _robot_self_components(self, complete_q, complete_names, robot_poses, robot_matrices, hand_state, zero, *, record_pair, record_nonclear, include_bounds):
```

实现说明：

> Query the fixed robot's required entity pairs at prepared states.
>
> Entity shapes and fixed link transforms are the official model of
> this unchanged robot/end-effector configuration, reusable across
> tasks. Named feedback/nominal hand coordinates and calibrated FK
> determine poses; task fields and camera coverage do not enter here.
> The caller supplies differentiable poses and their same-state matrix
> snapshots. Coal distances own the classification. Penetrating-pair
> witnesses supply the existing local derivative; numerical boundaries
> remain separately reported hard evidence.

显式输出分支：

```python
({'collision/robot/self': torch.stack(self_costs).reshape(shape), 'unresolved/robot/self': torch.stack(self_unresolved).reshape(shape)}, report)
```

读取的对象属性：

```python
complete_q.shape
np.zeros
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
hand_state.uncertain
self.robot_model.geometry_depending_on
self.robot_model
self.gripper.joint_names
self.gripper
self.tool_geometry_names
complete_q.detach().cpu().numpy
complete_q.detach().cpu
complete_q.detach
np.ndindex
self.robot_model.query
self.robot_validator.T_reference_base
self.robot_validator
self.robot_validator.required_pairs
self.check_request
self.robot_model._geometries
np.asarray
report['status_counts'].get
((_world(b, pb) - _world(a, pa)) * normal).sum
signed.detach
self_costs.append
self_unresolved.append
torch.stack(self_costs).reshape
torch.stack
torch.stack(self_unresolved).reshape
```

写入目标：

```python
shape
uncertain
unmodelled
report
q_cpu
self_costs
self_unresolved
index
evidence
value
undecided
row
first_id
second_id
entity_key
status
lower
report['pair_queries']
report['status_counts'][status]
a
b
pa
item
pb
normal
signed
```

直接调用符号：

```python
tuple
np.zeros
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
set
self.robot_model.geometry_depending_on
list
complete_q.detach().cpu().numpy
complete_q.detach().cpu
complete_q.detach
np.ndindex
self.robot_model.query
record_pair
native_pair_lower_bound
np.asarray
report['status_counts'].get
inverse_points
_tensor
((_world(b, pb) - _world(a, pa)) * normal).sum
_world
signed.detach
record_nonclear
self_costs.append
self_unresolved.append
torch.stack(self_costs).reshape
torch.stack
torch.stack(self_unresolved).reshape
```

## `OnlinePlanningScene.components`

[定义位置](../../../manipulation/online_geometry.py#L481)。

输入签名：

```python
def components(self, robot_spheres, position, quaternion, progress, joint_position=None, joint_names=None, *, trajectory_time, active_contact_rule_ids=None, contact_permission_progress=None, owner_poses=None, geometry_evidence_out=None, geometry_scope='full'):
```

实现说明：

> Return same-state residuals and retain distance/coverage diagnostics.
>
> ``robot_spheres`` is an empty native-kinematics output. Registered
> native entities carry all robot geometry. This API returns original
> entity, contact, relationship and coverage conditions. The independent
> finite optimization function is ``optimization_objective``.
> ``geometry_scope='robot'`` computes the same official self pairs and
> hand state without constructing task-owner poses or querying fields.

显式输出分支：

```python
costs
costs
```

读取的对象属性：

```python
self.kinematics
value.detach().cpu().numpy
value.detach().cpu
value.detach
self.gripper
self.gripper.state
hand_state.coordinates
hand_state.half
hand_state.controller_residual
hand_state.uncertain.to
hand_state.uncertain
progress.dtype
self.gripper.joint_names
torch.cat
self.gripper.joint_names.index
self.kinematics.poses
self.robot_model._geometries.items
self.robot_model._geometries
self.robot_model
Rotation.from_matrix(local[:3, :3]).as_quat
Rotation.from_matrix
robot_poses.items
complete_q.sum
position.sum
robot_poses.values
pose[0].sum
self._robot_self_components
costs.update
self._owner_poses
torch.zeros_like
owner.owner_id
owner.query
self.owners
self.background
poses.items
poses.values
self.contacts
rule.actor
self._permission
rule.target_owner
rule.rule_id
policy['normals'].detach().cpu().numpy
policy['normals'].detach().cpu
policy['normals'].detach
policy['bounds'].detach().cpu().numpy
policy['bounds'].detach().cpu
policy['bounds'].detach
active.detach().cpu().numpy
active.detach().cpu
active.detach
rule.required
policy['active'].detach().cpu().numpy
policy['active'].detach().cpu
policy['active'].detach
np.ndindex
progress.shape
target_matrix[:3, :3].T
self.check_request
gaps.append
undecided.append
policy['active'].to
normal.dtype
torch.stack(gaps).reshape
torch.stack
progress.new_tensor(undecided).reshape
progress.new_tensor
permission_progress.detach().cpu().numpy
permission_progress.detach().cpu
permission_progress.detach
np.zeros
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
self.robot_validator.checker.geometry_depending_on
self.robot_validator.checker
self.robot_validator
owner.moving
fields.items
self.fixed_geometry_names
pairs.append
first.moving
second.moving
first.owner_id
second.owner_id
first.query
second.query
progress.detach().cpu().numpy
progress.detach().cpu
progress.detach
owner.attachments
np.logical_or.reduce
np.logical_or
relation.active
attachment_progress.shape
rule.robot_body
self._initial_support_models.values
self._initial_support_models
model.relation.actor
model.relation
model.relation.target_owner
unmodelled_entity_keys.intersection
entity_key.values
residuals.append
self.tool_geometry_names
regions.append
model.relation.interval
model.permission
second.root
self._native_local_bounds
report['status_counts'].get
torch.stack(residuals).reshape
costs.get
model.components
self.coverage
self.unknown_policy
unknown.append
unresolved.append
rows.append
progress.new_tensor(unknown).reshape
progress.new_tensor(unresolved).reshape
```

写入目标：

```python
costs
permission_progress
evidence_time
evidence_progress
evidence_permission
value
selected_nonclear
hand_state
coordinates
half
complete_q
complete_names
costs['gripper/controller']
costs['conditional/hand_geometry']
missing_names
name
link_poses
robot_poses
frame
_
local
link_position
link_quaternion
local_position
local_quaternion
robot_poses[name]
robot_matrices
pose
zero
robot_costs
report
report['geometry_scope']
report['nonclear_pairs']
self.last_query_report
report['coverage_scope']
poses
identity
identity[1][..., 0]
poses['background']
fields
owner
fields['background']
matrices
policies
rule
actor_pose
policy
normal
uncertain
policy['active']
active
policies[rule.rule_id]
gaps
undecided
normals_cpu
bounds_cpu
interval_active
target_field
target_matrices
index
evidence
target_matrix
world_normals
world_bounds
gap
normal_cost
costs[f'contact/{rule.rule_id}']
costs[f'unresolved/contact/{rule.rule_id}']
costs[f'conditional/contact/{rule.rule_id}']
progress_cpu
uncertain_cpu
unmodelled
unmodelled_entity_keys
pairs
moving_owners
geometry
target
field
first
second
attachment_progress
attached
relation
native_world_bounds
component
first_id
second_id
first_pose
second_pose
fm
sm
actor
residuals
entity_key
rules
support
model
state_zero
held_internal
report['internal_assembly_pair_count']
regions
normals
bounds
region
permission
native_bounds
key
native_world_bounds[key]
report['pair_queries']
status
report['status_counts'][status]
costs[component]
support_components
report['owner_coverage']
unknown
unresolved
rows
conditional
prefix
costs[f'{prefix}unknown/forbidden/{owner.owner_id}']
costs[f'{prefix}unresolved/coverage/{owner.owner_id}']
report['owner_coverage'][owner.owner_id]
```

直接调用符号：

```python
ValueError
value.detach().cpu().numpy
value.detach().cpu
value.detach
tuple
self.gripper.state
hand_state.uncertain.to
torch.cat
self.gripper.joint_names.index
self.kinematics.poses
self.robot_model._geometries.items
_tensor
Rotation.from_matrix(local[:3, :3]).as_quat
Rotation.from_matrix
quaternion_rotate
quaternion_multiply
_matrices
robot_poses.items
complete_q.sum
position.sum
robot_poses.values
pose[0].sum
self._robot_self_components
costs.update
nonclear_diagnostics
self._owner_poses
torch.zeros_like
poses.items
poses.values
self._permission
policy['normals'].detach().cpu().numpy
policy['normals'].detach().cpu
policy['normals'].detach
policy['bounds'].detach().cpu().numpy
policy['bounds'].detach().cpu
policy['bounds'].detach
active.detach().cpu().numpy
active.detach().cpu
active.detach
policy['active'].detach().cpu().numpy
policy['active'].detach().cpu
policy['active'].detach
np.ndindex
query_region_intersection
float
gaps.append
undecided.append
policy['active'].to
torch.stack(gaps).reshape
torch.stack
progress.new_tensor(undecided).reshape
progress.new_tensor
permission_progress.detach().cpu().numpy
permission_progress.detach().cpu
permission_progress.detach
np.zeros
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
set
self.robot_validator.checker.geometry_depending_on
fields.items
pairs.append
combinations
progress.detach().cpu().numpy
progress.detach().cpu
progress.detach
np.logical_or.reduce
relation.active
self._initial_support_models.values
self.check_request
unmodelled_entity_keys.intersection
entity_key.values
residuals.append
record_pair
regions.append
PairPermission
model.permission
_native_bounds
_transform
pair_residual
report['status_counts'].get
record_nonclear
torch.stack(residuals).reshape
costs.get
model.components
query_coverage
unknown.append
unresolved.append
len
rows.append
list
progress.new_tensor(unknown).reshape
progress.new_tensor(unresolved).reshape
```

## `OnlinePlanningScene.components.record_pair`

[定义位置](../../../manipulation/online_geometry.py#L508)。

输入签名：

```python
def record_pair(index, component, first, second, evidence, *, geometry_entity_key, permission_kind='none'):
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
geometry_evidence_out.append
```

写入目标：

无显式条目。

直接调用符号：

```python
geometry_evidence_out.append
list
float
```

## `OnlinePlanningScene.components.record_nonclear`

[定义位置](../../../manipulation/online_geometry.py#L527)。

输入签名：

```python
def record_nonclear(index, component, first, second, evidence, residual):
```

显式输出分支：

```python
None
```

读取的对象属性：

无显式条目。

写入目标：

```python
row
selected_nonclear[component]
selected
order
selected['first']
maximum
selected['maximum']
```

直接调用符号：

```python
list
float
order
```

## `OnlinePlanningScene.components.nonclear_diagnostics`

[定义位置](../../../manipulation/online_geometry.py#L546)。

输入签名：

```python
def nonclear_diagnostics():
```

显式输出分支：

```python
rows
```

读取的对象属性：

```python
selected_nonclear.values
rows.append
rows.extend
```

写入目标：

```python
rows
selected
first
maximum
```

直接调用符号：

```python
selected_nonclear.values
rows.append
rows.extend
```

## `OnlinePlanningScene.optimization_objective`

[定义位置](../../../manipulation/online_geometry.py#L801)。

输入签名：

```python
def optimization_objective(self, position, quaternion, progress, joint_position, joint_names, *, trajectory_time, robot_link_poses, contact_permission_progress=None, report_out=None):
```

实现说明：

> Return a scalar objective using original geometry and current poses.
>
> Inputs have batch/time leading dimensions and each row is stably
> ordered by physical time, retaining the event's declared side order
> at equal times. Quaternions are unit wxyz rotations produced by FK.
> ``robot_link_poses`` maps
> every required reference-frame robot link to differentiable position
> and wxyz quaternion tensors from the caller's calibrated FK. Entity
> fixed transforms, owner attachment/release relations, hand certainty,
> finite contact regions and initial-support cell identities retain
> their existing scene definitions. Material samples remain fixed for
> the whole solve. Original ``components`` and interval queries retain
> all collision, missing-contact and unknown-space decisions.
>
> The finite sample objective can miss an overlap between its material
> points. Original support/permission branch switches and closest-feature
> ties are nondifferentiable. A constant zero field contributes no TSDF
> direction. These are objective-scope facts reported to the caller;
> they do not change any scene acceptance criterion.

显式输出分支：

```python
result
```

读取的对象属性：

```python
owner.owner_id
owner.query
self.owners
self.background
self._objective_material_samples
fields.items
self.robot_model._geometries.items
self.robot_model._geometries
self.robot_model
{id(geometry): geometry for geometry in geometries.values()}.items
geometries.values
geometries.items
item.points
samples.items
self.gripper
self.gripper.state
hand_state.coordinates
np.zeros
progress.shape
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
hand_state.uncertain
self.robot_model.geometry_depending_on
self.gripper.joint_names
Rotation.from_matrix(local[:3, :3]).as_quat
Rotation.from_matrix
robot_poses.items
self._owner_poses
torch.zeros_like
poses.items
position.sum
quaternion.sum
joint_position.sum
self.robot_validator.required_pairs
self.robot_validator
np.ndindex
pair.first
pair.second
values.append
self.contacts
rule.actor
self._permission
rule.target_owner
rule.rule_id
policy['normals'].detach().cpu().numpy
policy['normals'].detach().cpu
policy['normals'].detach
policy['bounds'].detach().cpu().numpy
policy['bounds'].detach().cpu
policy['bounds'].detach
(policy['active'] & policy['normal_valid']).detach().cpu().numpy
(policy['active'] & policy['normal_valid']).detach().cpu
(policy['active'] & policy['normal_valid']).detach
rule.required
target_samples.zero_points
report['contact_without_zero_samples'].append
normal[index].square
owner.moving
self.fixed_geometry_names
pairs.append
pairs.extend
first.moving
second.moving
first.owner_id
second.owner_id
progress.detach().cpu().numpy
progress.detach().cpu
progress.detach
permission_progress.detach().cpu().numpy
permission_progress.detach().cpu
permission_progress.detach
owner.attachments
np.logical_or.reduce
np.logical_or
relation.active
rule.robot_body
self._initial_support_models.values
self._initial_support_models
model.relation.actor
model.relation
model.relation.target_owner
self.check_request
self.tool_geometry_names
target_matrix[:3, :3].T
regions.append
model.relation.interval
model.permission
model.components
values.extend
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square().reshape(-1).unbind
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square().reshape
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square
model.relation.relation_id
model.actor_field.voxel_size
model.actor_field
torch.stack(values).sum
torch.stack
progress.numel
value.detach().cpu().numpy().tolist
value.detach().cpu().numpy
value.detach().cpu
value.detach
relationship_components.items
report_out.update
```

写入目标：

```python
fields
owner
fields['background']
geometries
name
field
row
prepared
identity
geometry
self._objective_material_samples
key
samples
report
kind
item
permission_progress
hand_state
coordinates
uncertain
unmodelled
robot_poses
frame
_
local
link_position
link_quaternion
local_quaternion
robot_poses[name]
robot_matrices
pose
relationship_components
poses
identity_quaternion
identity_quaternion[..., 0]
poses['background']
matrices
zero
values
pair
index
report['unmodelled_pair_count']
report['native_pair_state_count']
policies
rule
actor_pose
policy
normal
conditional
policies[rule.rule_id]
target_samples
moving
pairs
target
first
second
attachment_progress
permission_cpu
attached
relation
first_pose
actor
rules
support
model
report['internal_assembly_pair_count']
regions
normals
bounds
active
target_matrix
world_normals
region
permission
report['pair_state_count']
components
result
report['relationship_components']
value
```

直接调用符号：

```python
fields.items
self.robot_model._geometries.items
MaterialSamples
{id(geometry): geometry for geometry in geometries.values()}.items
id
geometries.values
geometries.items
len
samples.items
self.gripper.state
np.zeros
tuple
hand_state.uncertain.detach().cpu().numpy
hand_state.uncertain.detach().cpu
hand_state.uncertain.detach
set
self.robot_model.geometry_depending_on
_tensor
Rotation.from_matrix(local[:3, :3]).as_quat
Rotation.from_matrix
quaternion_rotate
quaternion_multiply
_matrices
robot_poses.items
self._owner_poses
torch.zeros_like
poses.items
position.sum
quaternion.sum
joint_position.sum
np.ndindex
values.append
native_pair_objective
self._permission
policy['normals'].detach().cpu().numpy
policy['normals'].detach().cpu
policy['normals'].detach
policy['bounds'].detach().cpu().numpy
policy['bounds'].detach().cpu
policy['bounds'].detach
(policy['active'] & policy['normal_valid']).detach().cpu().numpy
(policy['active'] & policy['normal_valid']).detach().cpu
(policy['active'] & policy['normal_valid']).detach
report['contact_without_zero_samples'].append
bool
normal[index].square
contact_objective
pairs.append
pairs.extend
combinations
progress.detach().cpu().numpy
progress.detach().cpu
progress.detach
permission_progress.detach().cpu().numpy
permission_progress.detach().cpu
permission_progress.detach
np.logical_or.reduce
relation.active
self._initial_support_models.values
self.check_request
regions.append
PairPermission
model.permission
pair_objective
model.components
values.extend
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square().reshape(-1).unbind
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square().reshape
(components[f'initial_support/{model.relation.relation_id}'] / model.actor_field.voxel_size).square
torch.stack(values).sum
torch.stack
progress.numel
value.detach().cpu().numpy().tolist
value.detach().cpu().numpy
value.detach().cpu
value.detach
relationship_components.items
report_out.update
```

## `OnlinePlanningScene.evaluate`

[定义位置](../../../manipulation/online_geometry.py#L971)。

输入签名：

```python
def evaluate(self, *args, **kwargs):
```

显式输出分支：

```python
torch.stack([value for (name, value) in components.items() if not name.startswith('conditional/')]).sum(0)
```

读取的对象属性：

```python
kwargs.pop
self.components
output.update
torch.stack([value for (name, value) in components.items() if not name.startswith('conditional/')]).sum
torch.stack
components.items
name.startswith
```

写入目标：

```python
output
components
name
value
```

直接调用符号：

```python
kwargs.pop
self.components
output.update
torch.stack([value for (name, value) in components.items() if not name.startswith('conditional/')]).sum
torch.stack
components.items
name.startswith
```

## `OnlinePlanningScene.estimated_workspace_allocation`

[定义位置](../../../manipulation/online_geometry.py#L981)。

输入签名：

```python
def estimated_workspace_allocation(self, batch, horizon):
```

实现说明：

> Separate measured-RSS inventory from still-unallocated scene work.

显式输出分支：

```python
dict(cpu_resident=resident, cpu_pending_shared=pending, cpu_live=live + temporary, gpu_shared=0, gpu_live=transfer, gpu_retained=0, gpu_graph_live=0)
```

读取的对象属性：

```python
self.background
owner.query
self.owners
{id(value): value for value in fields}.values
field.storage_nbytes
self.robot_model._native_distance_batches
self.robot_model
self.robot_validator
pair.first
pair.second
self.robot_validator.required_pairs
self.robot_model.native_query_storage_nbytes
field.occupied_points
field.nodes
node.cell
field.cells
self._objective_material_samples
self.robot_model.geometry_names
{id(value): value for value in self._objective_material_samples.values()}.values
self._objective_material_samples.values
value.points.nbytes
value.points
value.cells.nbytes
value.cells
value.zero_points.nbytes
value.zero_points
value.point_bounds
value.point_bounds.nbytes
value.negative_bounds
value.negative_bounds.nbytes
rule.feature_points
self.contacts
self.robot_model.joint_names
self.required_robot_links
```

写入目标：

```python
fields
owner
unique_fields
value
resident
field
pending
schema
expected
pair
material_counts
node
zero_counts
count
zeros
samples
rows
rule
temporary
entities
states
live
transfer
```

直接调用符号：

```python
tuple
{id(value): value for value in fields}.values
id
sum
len
field.occupied_points
zip
{id(value): value for value in self._objective_material_samples.values()}.values
self._objective_material_samples.values
max
int
dict
```

## `OnlinePlanningScene.describe`

[定义位置](../../../manipulation/online_geometry.py#L1032)。

输入签名：

```python
def describe(self):
```

显式输出分支：

```python
{'model': 'native_robot_observed_scene_v1', 'reference_frame': self.reference_frame, 'robot_geometry_source': 'registered_native_model', 'owners': [owner.owner_id for owner in self.owners], 'robot_entities': list(self.robot_model.geometry_names), 'fixed_geometry_names': list(self.fixed_geometry_names), 'tool_geometry_names': list(self.tool_geometry_names), 'unknown_policy': deepcopy(self.unknown_policy), 'robot_view_coverage_status': 'not_queried', 'collision_rule': 'nonpenetration', 'required_progress_anchors': self.required_progress_anchors, 'observation_report': deepcopy(self.observation_report)}
```

读取的对象属性：

```python
self.reference_frame
owner.owner_id
self.owners
self.robot_model.geometry_names
self.robot_model
self.fixed_geometry_names
self.tool_geometry_names
self.unknown_policy
self.required_progress_anchors
self.observation_report
```

写入目标：

```python
owner
```

直接调用符号：

```python
list
deepcopy
```

## `OnlinePlanningScene.to_execution_binding`

[定义位置](../../../manipulation/online_geometry.py#L1044)。

输入签名：

```python
def to_execution_binding(self):
```

显式输出分支：

```python
{'schema': 'native_robot_observed_scene_v1', **self.describe(), 'owners': owners, 'robot_model': native, 'background': self.background.record(), 'contacts': _plain([asdict(row) for row in self.contacts]), 'tool_geometry_names': list(self.tool_geometry_names), 'grasp_contact_activations': _plain(self.grasp_contact_activations), 'initial_support_relations': [row.record() for row in self.initial_support_relations], 'gripper': hand_record, 'has_gripper': self.has_gripper, 'numerical_position_tolerance_m': self.numerical_position_tolerance_m, 'numerical_rotation_tolerance_rad': self.numerical_rotation_tolerance_rad, 'coverage': {'known_free': encode_array(self.coverage.known_free), 'origin': self.coverage.origin.tolist(), 'voxel_size': self.coverage.voxel_size, 'observation_ids': list(self.coverage.observation_ids)}}
```

读取的对象属性：

```python
self.owners
owner.pose_path
owners.append
owner.owner_id
owner.source
owner.moving
owner.pose_interval
path.progress
path.poses
path.source
path.motion_model
path.coordinates
path.attachment
owner.attachments
owner.query.record
owner.query
self.gripper
hand.nominal_finger_trajectory
hand.nominal_finger_trajectory.describe
self.robot_model.model_record
self.robot_model
self.robot_model.base_frame
self.robot_validator
self.robot_validator.T_reference_base.tolist
self.robot_validator.T_reference_base
self.robot_pairs
self.describe
self.background.record
self.background
self.contacts
self.tool_geometry_names
self.grasp_contact_activations
row.record
self.initial_support_relations
self.has_gripper
self.numerical_position_tolerance_m
self.numerical_rotation_tolerance_rad
self.coverage.known_free
self.coverage
self.coverage.origin.tolist
self.coverage.origin
self.coverage.voxel_size
self.coverage.observation_ids
```

写入目标：

```python
owners
owner
path
row
hand
hand_record
name
hand_record['nominal_model']
native
pair
```

直接调用符号：

```python
owners.append
list
_plain
asdict
owner.query.record
getattr
hand.nominal_finger_trajectory.describe
self.robot_model.model_record
self.robot_validator.T_reference_base.tolist
self.describe
self.background.record
row.record
encode_array
self.coverage.origin.tolist
```
