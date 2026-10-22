# `tests/planning/observed_robot_fixture.py` 实现契约

代码基线：`a56c3e4`。
职责和数据语义见[组件说明](../05-acceptance-and-execution.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `load_observed_owner`

[定义位置](../../../tests/planning/observed_robot_fixture.py#L12)。

输入签名：

```python
def load_observed_owner(path):
```

实现说明：

> Restore the same pre-relationship field and the declared rigid motion.

显式输出分支：

```python
CollisionOwner(row['owner_id'], field, PosePath(np.array([0.0, 1.0]), np.stack((pose, pose)), row['source']), row['source'], moving=bool(row['attachments']), attachments=tuple((Attachment(**relation) for relation in row['attachments'])))
```

读取的对象属性：

```python
json.loads
Path(path).read_text
FrozenObservedField.load
np.asarray
np.array
np.stack
```

写入目标：

```python
row
field
pose
relation
```

直接调用符号：

```python
json.loads
Path(path).read_text
Path
FrozenObservedField.load
np.asarray
CollisionOwner
PosePath
np.array
np.stack
bool
tuple
Attachment
```

## `problem_owner_binding`

[定义位置](../../../tests/planning/observed_robot_fixture.py#L26)。

输入签名：

```python
def problem_owner_binding(problem, owner_id):
```

实现说明：

> Copy the exact current owner pose and deduplicated held spans.
>
> The inverse attachment maps owner-local points into the tool frame.
> Release IDs and progress come from the current compiler's held metadata,
> never a source frame guessed to be the executable release time.

显式输出分支：

```python
(initial, attachments)
```

读取的对象属性：

```python
problem.phases
phase.metadata.get
phase.metadata
phases[0].metadata
np.allclose
metadata.get
np.linalg.inv
np.linalg
inverse[:3, 3].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat
Rotation.from_matrix
attachments.append
```

写入目标：

```python
phases
phase
initial
attachments
metadata
inverse
relation
same_span
item
```

直接调用符号：

```python
phase.metadata.get
ValueError
_pose_array
np.allclose
metadata.get
np.linalg.inv
list
inverse[:3, 3].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]].tolist
Rotation.from_matrix(inverse[:3, :3]).as_quat
Rotation.from_matrix
attachments.append
```
