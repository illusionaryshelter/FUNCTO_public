# `transfer/cross_instance/context.py` 实现契约

代码基线：`96ee2dc-working-tree`。
职责和数据语义见[组件说明](../../../transfer/cross_instance/ARCHITECTURE.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_center`

[定义位置](../../../transfer/cross_instance/context.py#L21)。

输入签名：

```python
def _center(points):
```

实现说明：

> Centre after removing one point so constant clouds stay exactly zero.

显式输出分支：

```python
(points[0] + relative_mean, relative - relative_mean)
```

读取的对象属性：

```python
np.asarray
relative.mean
```

写入目标：

```python
points
relative
relative_mean
```

直接调用符号：

```python
np.asarray
relative.mean
```

## `_proper_rotation`

[定义位置](../../../transfer/cross_instance/context.py#L29)。

输入签名：

```python
def _proper_rotation(covariance, sample_count):
```

实现说明：

> Proper Procrustes rotation and its numerical uniqueness diagnostic.
>
> A rank-two paired covariance can determine the proper rotation. The
> smallest rotation curvature is sigma_2 + parity*sigma_3; checking each
> cloud's rank independently would not establish this pairing property.

显式输出分支：

```python
(rotation, singular, bool(curvature > numerical_zero))
```

读取的对象属性：

```python
np.linalg.svd
np.linalg
np.linalg.det
np.array
np.finfo(float).eps
np.finfo
```

写入目标：

```python
left
singular
right
parity
rotation
curvature
numerical_zero
```

直接调用符号：

```python
np.linalg.svd
np.linalg.det
np.array
np.finfo
max
bool
```

## `context_vector`

[定义位置](../../../transfer/cross_instance/context.py#L44)。

输入签名：

```python
def context_vector(owner, region_center):
```

实现说明：

> Bound the observed owner-to-region relation without normalizing its tip.
>
> The denominator squared equals owner variance plus squared centre offset,
> so ||vector|| <= 1. Coincident means therefore yield zero evidence exactly.
> This empirical observation relation is neither a physical surface normal
> nor a declaration of the object's unobserved centre or occupied volume.

显式输出分支：

```python
dict(vector=(region_center - owner_center) / rms, owner_center=owner_center, context_rms_m=rms, count=len(owner))
```

读取的对象属性：

```python
np.asarray
np.sqrt
np.mean
np.sum
```

写入目标：

```python
owner
region_center
owner_center
_
rms
```

直接调用符号：

```python
np.asarray
_center
float
np.sqrt
np.mean
np.sum
dict
len
```

## `fit_rigid_with_context`

[定义位置](../../../transfer/cross_instance/context.py#L60)。

输入签名：

```python
def fit_rigid_with_context(source_normalized, paired_target_normalized, d_source, d_target, kappa=1.0):
```

实现说明：

> Exactly fit one fixed correspondence with one fixed context relation.
>
> Minimize mean(||R*x_i+t-y_i||²)+kappa*||R*d_source-d_target||².
> The caller freezes both context vectors throughout correspondence search;
> changing d_target when J changes would define another assignment block.
> The rotation is proper, the translation uses the complete paired means,
> and the returned uniqueness flag is diagnostic rather than a gate.

显式输出分支：

```python
dict(rotation=rotation, translation=translation, covariance_singular_values=singular, rotation_unique=unique, context_squared_error=float(np.sum((rotation @ ds - dt) ** 2)))
```

读取的对象属性：

```python
np.asarray
y.T
np.outer
np.sum
```

写入目标：

```python
source_mean
x
target_mean
y
ds
dt
covariance
rotation
singular
unique
translation
```

直接调用符号：

```python
_center
np.asarray
len
np.outer
_proper_rotation
dict
float
np.sum
```

## `read_region_frame`

[定义位置](../../../transfer/cross_instance/context.py#L81)。

输入签名：

```python
def read_region_frame(source, mapped, source_frame):
```

实现说明：

> Summarize complete identity pairs by their least-squares similarity.
>
> This geometric summary transports the original functional origin with
> the fitted scale and translation, and rotates its unit axes. It does not
> scale the robot, its tool anchor, or the source skill's metric H_g offsets.
> Source variance must be positive. Nonunique rotations still admit an SVD
> minimizer: the returned frame is one such numerical summary, accompanied
> by rotation_unique=False, rather than a uniquely observed direction.
>
> Independent grasp evidence consumes the complete mapped rows directly.
> This summary's uniqueness therefore supplies a diagnostic, not a gate on
> grasp generation or a substitute for the observed owner context.

显式输出分支：

```python
dict(status='computed', matrix=matrix, rotation=rotation, translation=translation, scale=scale, T_world_region=frame, source_center=source_center, target_center=target_center, rms_residual_m=float(np.sqrt(np.mean(residuals ** 2))), residuals_m=residuals, paired_points=len(source), covariance_singular_values=fit['covariance_singular_values'], rotation_unique=fit['rotation_unique'], construction='least_squares_similarity_of_complete_source_identities', origin_centroid_offset_m=float(np.linalg.norm(original[:3, 3] - source_center)))
```

读取的对象属性：

```python
np.asarray
np.zeros
rotation.T
np.sum
np.eye
np.linalg.norm
np.linalg
np.sqrt
np.mean
```

写入目标：

```python
source
mapped
original
source_center
p
target_center
q
fit
rotation
rotated
scale
translation
matrix
matrix[:3, :3]
matrix[:3, 3]
frame
frame[:3, :3]
frame[:3, 3]
residuals
```

直接调用符号：

```python
np.asarray
_center
fit_rigid_with_context
np.zeros
float
np.sum
np.eye
np.linalg.norm
dict
np.sqrt
np.mean
len
```

## `orientation_evidence`

[定义位置](../../../transfer/cross_instance/context.py#L123)。

输入签名：

```python
def orientation_evidence(source_region, mapped_region, source_owner, target_owner, kappa=1.0):
```

实现说明：

> Combine full identity-preserving region pairs and visible owner context.
>
> Both region arrays use the same original source rows. Each is centred and
> divided by its own RMS, giving mean squared norm one. Consequently the
> local relation cost has constant 2. Context contributes its actual bounded
> vector norms. The returned bound is a uniform upper bound on this complete
> direction cost, making its normalized value lie in [0,1].
>
> OAReg's target measurements and its mapped source rows have different
> roles: this function consistently uses mapped rows for the target centre,
> RMS, covariance and context relation. No uniqueness of a physical frame
> is inferred from numerical rotation uniqueness.

显式输出分支：

```python
dict(matrix=matrix, source_center=source_center, target_center=target_center, source_rms_m=source_rms, target_rms_m=target_rms, d_source=ds, d_target=dt, direction_constant=direction_constant, direction_bound=direction_bound, rotation_wahba=rotation, rotation_unique=unique, source_context=source_context, target_context=target_context, covariance_singular_values=singular)
```

读取的对象属性：

```python
np.sqrt
np.mean
np.sum
(q / target_rms).T
np.outer
np.linalg.norm
np.linalg
```

写入目标：

```python
source_center
p
target_center
q
source_rms
target_rms
source_context
target_context
ds
dt
h
matrix
rotation
singular
unique
direction_constant
direction_bound
```

直接调用符号：

```python
_center
float
np.sqrt
np.mean
np.sum
context_vector
len
np.outer
_proper_rotation
np.linalg.norm
dict
```

## `_source_grasp_anchor`

[定义位置](../../../transfer/cross_instance/context.py#L158)。

输入签名：

```python
def _source_grasp_anchor(evidence, source_grasp):
```

实现说明：

> Original region centre in the measured source grasp G, in metres.

显式输出分支：

```python
source_grasp[:3, :3].T @ (evidence['source_center'] - source_grasp[:3, 3])
```

读取的对象属性：

```python
source_grasp[:3, :3].T
```

写入目标：

无显式条目。

直接调用符号：

无显式条目。

## `context_grasp_proposal`

[定义位置](../../../transfer/cross_instance/context.py#L163)。

输入签名：

```python
def context_grasp_proposal(evidence, source_G):
```

实现说明：

> One ordinary rigid proposal with optimal relation rotation and anchor.
>
> Its position places the unchanged metric source anchor at the mapped
> region centre. The region scale does not scale the robot or anchor. Model
> quality is evaluated later by the same scorer used for all other poses.

显式输出分支：

```python
proposal
```

读取的对象属性：

```python
np.asarray
np.eye
```

写入目标：

```python
source_grasp
anchor
rotation
proposal
proposal[:3, :3]
proposal[:3, 3]
```

直接调用符号：

```python
np.asarray
_source_grasp_anchor
np.eye
```

## `rank_candidates`

[定义位置](../../../transfer/cross_instance/context.py#L180)。

输入签名：

```python
def rank_candidates(tool_poses_world, quality, tool_prior_world, position_scale_m):
```

实现说明：

> Rank actual controlled E poses against one complete E pose prior.
>
> Inputs are a nonempty Nx4x4 array of proper rigid tool poses, one native
> quality value per pose, and a 4x4 prior for the same controlled tool origin
> and axes. All positions share the target world and metre unit. The positive
> position_scale_m is the saved region RMS length, a dimensional normalization
> rather than an orientation confidence or an acceptance threshold.
>
> The three unit-weight terms are ||p-p_prior||²/position_scale_m²,
> ||R-R_prior||_F²/8, and 1-quality. For relative rotation angle theta, the
> orientation term equals sin²(theta/2), spanning [0,1]. It is a soft cost on
> the complete physical pose, including rotation about the approach axis.
> Alternative grasp labels enter as explicitly bound physical E poses before
> this call; every row is scored exactly as supplied, without label changes.
>
> Returned angle and position errors describe this frozen candidate/prior
> comparison. They do not control execution. Stable sorting preserves the
> supplied order of exact ties. The SO(3) angle uses atan2 of the relative
> matrix's skew magnitude and trace, including both zero and pi endpoints.

显式输出分支：

```python
dict(costs_total=total, costs_quality=quality_cost, costs_orientation=orientation_cost, costs_position=position_cost, orientation_error_rad=angles, position_error_m=np.sqrt(position_squared), order=order, selected_index=int(order[0]))
```

读取的对象属性：

```python
np.asarray
np.sum
prior[:3, :3].T
np.stack
np.linalg.norm
np.linalg
np.trace
np.arctan2
np.argsort
np.sqrt
```

写入目标：

```python
poses
quality
prior
position_squared
position_cost
orientation_cost
relative
skew
sine
cosine
angles
quality_cost
total
order
```

直接调用符号：

```python
np.asarray
np.sum
float
np.stack
np.linalg.norm
np.trace
np.arctan2
np.argsort
dict
np.sqrt
int
```
