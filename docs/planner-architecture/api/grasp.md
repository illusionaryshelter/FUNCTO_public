# `manipulation/grasp.py` 实现契约

代码基线：`本次 A–D 实现工作树，实际提交引用待补`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `validate_timeline`

[定义位置](../../../manipulation/grasp.py#L22)。

输入签名：

```python
def validate_timeline(frame_indices, timestamps):
```

实现说明：

> Return independent arrays with original observation IDs and seconds.
>
> Frame IDs need not be consecutive or begin at zero. They are identifiers,
> not array offsets. Times must be finite and strictly increasing; this
> validation never replaces them with the geometry kernel's sample phase.
> A one-observation sequence is valid for an already established boundary.

显式输出分支：

```python
(frames.copy(), times.astype(float, copy=True))
```

读取的对象属性：

```python
np.bool_
np.asarray
frames.ndim
frames.dtype.kind
frames.dtype
np.any
times.shape
frames.shape
times.dtype.kind
times.dtype
np.isfinite(times).all
np.isfinite
frames.copy
times.astype
```

写入目标：

```python
value
frames
times
```

直接调用符号：

```python
any
isinstance
ValueError
np.asarray
len
np.any
np.isfinite(times).all
np.isfinite
frames.copy
times.astype
```

## `_name`

[定义位置](../../../manipulation/grasp.py#L44)。

输入签名：

```python
def _name(record, key):
```

显式输出分支：

```python
value
```

读取的对象属性：

```python
record.get
value.strip
```

写入目标：

```python
value
```

直接调用符号：

```python
record.get
isinstance
value.strip
ValueError
```

## `_gripper_events`

[定义位置](../../../manipulation/grasp.py#L51)。

输入签名：

```python
def _gripper_events(value, frames):
```

实现说明：

> Check explicit intentions without deriving commands from measured qpos.
>
> # BEGIN source end ownership is independent of device execution coverage.
> An explicitly owned command may use either source boundary. Requiring an
> observed frame avoids silently snapping an event after decimation. The
> shared physical clock must still provide a real activation tick and the
> required post-command coverage; accepting a source ID does not prove that.
> # END no last source observation is silently converted into a command.

显式输出分支：

```python
events
```

读取的对象属性：

```python
event.get
np.bool_
np.integer
ids.add
events.append
```

写入目标：

```python
frame_set
events
ids
previous
event
event_id
frame
command
```

直接调用符号：

```python
isinstance
ValueError
set
map
_name
event.get
int
ids.add
events.append
deepcopy
any
```

## `_grasp_goal`

[定义位置](../../../manipulation/grasp.py#L93)。

输入签名：

```python
def _grasp_goal(value):
```

实现说明：

> Validate the shared explicit grasp relation and task precision contract.

显式输出分支：

```python
{'T_carrier_eef': _pose_array(value['T_carrier_eef'], 'grasp_goal.T_carrier_eef'), 'position_tolerance_m': position_tolerance, 'rotation_tolerance_rad': rotation_tolerance, 'source': _source(value['source'], 'Grasp task goal and allowed set')}
```

读取的对象属性：

```python
np.bool_
```

写入目标：

```python
required
name
position_tolerance
rotation_tolerance
```

直接调用符号：

```python
isinstance
set
ValueError
any
_tolerances
_pose_array
_source
```

## `_reference_prior`

[定义位置](../../../manipulation/grasp.py#L109)。

输入签名：

```python
def _reference_prior(value):
```

实现说明：

> Read the prior in the same carrier coordinates as the selected goal.

显式输出分支：

```python
{'T_carrier_eef': _pose_array(value['T_carrier_eef'], 'grasp_reference_prior.T_carrier_eef'), 'source': _source(value['source'], 'Original grasp reference anchor')}
```

读取的对象属性：

无显式条目。

写入目标：

无显式条目。

直接调用符号：

```python
isinstance
set
ValueError
_pose_array
_source
```

## `bind_grasp_candidate`

[定义位置](../../../manipulation/grasp.py#L117)。

输入签名：

```python
def bind_grasp_candidate(scene, T_reference_tool, *, source):
```

实现说明：

> Return a scene whose new grasp goal also aligns its complete reference.
>
> ``T_reference_tool`` uses the scene's ``reference_frame`` and the existing
> ``tool_frame`` control point, not a generator's gripper base. Its producer
> must first perform that calibrated frame conversion. This binding only
> creates an action candidate; it establishes neither clearance nor Holding.
>
> Keep the original independent goal as ``grasp_reference_prior``. Grounding
> then moves every ordinary reference sample by the rigid displacement from
> that prior to the selected goal. An existing prior remains the anchor when
> another candidate is bound to an already bound scene, so changes never
> accumulate. The demonstration, functional correspondence, contact domains,
> current feedback and task tolerances are not altered.

显式输出分支：

```python
result
```

读取的对象属性：

```python
scene.get
goal['T_carrier_eef'].copy
np.array_equal
prior['T_carrier_eef'].copy
```

写入目标：

```python
key
carrier
target
goal
prior
selected
result
result['grasp_reference_prior']
result['grasp_goal']
```

直接调用符号：

```python
isinstance
ValueError
_name
_pose_array
_grasp_goal
scene.get
_reference_prior
goal['T_carrier_eef'].copy
deepcopy
np.array_equal
prior['T_carrier_eef'].copy
_inverse
_source
```

## `ground_grasp`

[定义位置](../../../manipulation/grasp.py#L156)。

输入签名：

```python
def ground_grasp(demo, scene):
```

实现说明：

> Bind the original same-tool reference and its independent grasp goal.
>
> Region correspondence maps the complete source motion into the current
> scene. The actual robot state belongs to the planner; no selected source
> frame creates an approach waypoint, Cartesian suffix lock, or stop. The
> grasp relation is explicitly supplied in ``scene.grasp_goal`` as the
> current carrier-local ``T_carrier_eef``, task ``position_tolerance_m`` and
> ``rotation_tolerance_rad``, and their ``source``. It is independent of
> ordinary reference samples and never observed Holding. Numerical solver
> tolerances do not supply missing task precision. ``protected_start_frame``
> retains contact evidence identity only.
>
> # BEGIN candidate alignment is explicit rather than inferred from a goal change.
> ``grasp_reference_prior`` opts into a generated candidate's full-reference
> alignment. For bound source samples D, original goal P and selected goal E
> in the output frame, the new reference is E @ inverse(P) @ D. This preserves
> the source's relative deviations even when D[-1] differs from P. Ordinary
> scenes without this record keep their independent goal/reference contract.
> # END reference alignment adds no waypoint, contact permission or command.

显式输出分支：

```python
result
```

读取的对象属性：

```python
np.bool_
np.integer
scene.get
np.array_equal
grasp_goal['T_carrier_eef'].copy
```

写入目标：

```python
key
relative
frames
times
events
protected
event
carrier
grasp_goal
mapping
bound_relative
prior
result
result['grasp_reference_prior']
```

直接调用符号：

```python
_name
ValueError
_poses_array
validate_timeline
len
_gripper_events
isinstance
any
_pose_array
_grasp_goal
scene.get
_reference_prior
np.array_equal
_inverse
int
grasp_goal['T_carrier_eef'].copy
```

## `attached_eef_reference`

[定义位置](../../../manipulation/grasp.py#L228)。

输入签名：

```python
def attached_eef_reference(carrier_poses, actual_T_carrier_eef):
```

实现说明：

> Apply the caller's declared attachment during modeled rigid holding.
>
> The caller obtains ``actual_T_carrier_eef = inverse(T_world_carrier) @
> T_world_eef`` from synchronized current evidence after checking stability.
> This function only composes geometry; passing a matrix is not evidence that
> the object is held. Stop using it when that relation is no longer valid,
> and never extrapolate it after release.

显式输出分支：

```python
carrier @ actual
```

读取的对象属性：

无显式条目。

写入目标：

```python
carrier
actual
```

直接调用符号：

```python
_poses_array
_pose_array
```
