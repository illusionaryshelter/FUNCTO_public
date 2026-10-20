# `tests/planning/recorded_owner_geometry.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `load_catalog_body_boxes`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L26)。

输入签名：

```python
def load_catalog_body_boxes(catalog_path, *, body_id):
```

输出分支的源码表达式：

```python
(boxes, source)
```

直接调用：

```python
Path
catalog_path.stat
ValueError
json.loads
catalog_path.read_text
next
any
len
deepcopy
str
catalog_path.resolve
```

## `_box_pose`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L59)。

输入签名：

```python
def _box_pose(box):
```

输出分支的源码表达式：

```python
(pose, halfsize)
```

直接写入的状态或数组位置：`pose[:3, :3]`、`pose[:3, 3]`。

直接调用：

```python
np.asarray
np.isfinite(position).all
np.isfinite
np.isfinite(quaternion).all
np.isfinite(halfsize).all
np.any
np.isclose
np.linalg.norm
ValueError
np.eye
Rotation.from_quat(quaternion[[1, 2, 3, 0]]).as_matrix
Rotation.from_quat
```

## `_problem_binding`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L73)。

输入签名：

```python
def _problem_binding(problem, owner_id):
```

直接读取的对象属性：`problem.phases`。

输出分支的源码表达式：

```python
(initial, attachments)
```

直接调用：

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

## `prepare_box_owner`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L106)。

输入签名：

```python
def prepare_box_owner(problem, owner_id, *, boxes, T_owner_source, geometry_source, mapping_source, cell_edge_m, max_spheres):
```

输出分支的源码表达式：

```python
{'owner_id': owner_id, 'cuboids': cuboids, 'local_spheres': spheres, 'initial_pose': initial.tolist(), 'attachments': attachments, 'geometry_complete': True, 'sphere_cover_error_m': 0.0, 'sphere_materials': [None] * len(spheres), 'source': f"Offline native box body {geometry_source['source_body_name']} from {geometry_source['catalog_path']}; explicit current carrier mapping and compiled attachment", 'provenance': {'geometry_source': deepcopy(geometry_source), 'T_owner_source': mapping.tolist(), 'mapping_source': deepcopy(mapping_source), 'cell_edge_m': float(cell_edge_m), 'max_spheres': max_spheres, 'box_count': len(boxes), 'sphere_count': len(spheres), 'coverage_cells': proof, 'coverage_proof': 'Every complete native box is tiled by Cartesian cells; each serialized ball encloses all eight vertices of its convex cell', 'occupied_space_evidence_supplied': False, 'contact_permissions_supplied': False, 'current_object_state_read': False}}
```

直接调用：

```python
_pose_array
isinstance
geometry_source.get
len
set
ValueError
np.isfinite
type
_problem_binding
_box_pose
np.maximum
np.ceil(2 * size / cell_edge_m).astype
np.ceil
sum
int
np.prod
np.asarray
list
product
zip
cuboids.append
Rotation.from_matrix(current_pose[:3, :3]).as_quat
Rotation.from_matrix
(2 * halfsize).tolist
range
(center_box @ current_pose[:3, :3].T + current_pose[:3, 3]).astype(np.float32).astype
(center_box @ current_pose[:3, :3].T + current_pose[:3, 3]).astype
float
np.linalg.norm(vertices - center, axis=1).max
np.linalg.norm
np.nextafter
np.float32
spheres.append
margins.append
proof.append
count.tolist
min
initial.tolist
deepcopy
mapping.tolist
```

## `_box_material_feature`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L169)。

输入签名：

```python
def _box_material_feature(row, pose, halfsize):
```

输出分支的源码表达式：

```python
(points, normal, thickness, audit)
```

直接写入的状态或数组位置：`vertices[:, axis]`、`vertices[:, tangents]`、`vertices[:, fixed]`。

直接调用：

```python
set
ValueError
type
range
np.zeros
np.asarray
list
product
np.eye
float
isinstance
len
any
np.isfinite(hint).all
np.isfinite
np.isclose
np.linalg.norm
nnls
np.finfo
np.arctan2
np.cross
np.min
hint.tolist
adjacent.tolist
coefficients.tolist
abs
audit.update
vertices.tolist
pose.tolist
np.linalg.norm(recovered - vertices, axis=1).max
np.abs((points - points[0]) @ normal).max
np.abs
```

## `bind_box_owner_materials`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L244)。

输入签名：

```python
def bind_box_owner_materials(prepared):
```

输出分支的源码表达式：

```python
{'owner_id': owner_id, 'materials': materials, 'sphere_materials': labels, 'native_geometry_materials': native}
```

直接写入的状态或数组位置：`groups[name]`、`materials[identity]`、`native[owner_id + '/' + name]`、`labels[index]`。

直接调用：

```python
prepared.get('provenance', {}).get
prepared.get
source.get
ValueError
isinstance
owner_id.strip
np.asarray
len
set
covers.values
np.isfinite(spheres).all
np.isfinite
np.any
covers.items
any
type
list
range
owned.extend
sorted
identity.strip
row['source'].strip
_box_pose
_box_material_feature
points.tolist
normal.tolist
float
np.max
model.update
pose.tolist
halfsize.tolist
materials[identity].update
json.dumps
```

## `create_box_owner`

[定义位置](../../../tests/planning/recorded_owner_geometry.py#L318)。

输入签名：

```python
def create_box_owner(prepared, *, contact_voxel_size_m):
```

输出分支的源码表达式：

```python
CollisionOwner(prepared['owner_id'], query, PosePath(np.array([0.0, 1.0]), np.stack((initial, initial)), prepared['source']), prepared['source'], geometry_complete=True, local_spheres=prepared['local_spheres'], sphere_materials=tuple(prepared['sphere_materials']), attachments=tuple((Attachment(**relation) for relation in prepared['attachments'])), sphere_cover_error_m=prepared['sphere_cover_error_m'])
```

直接调用：

```python
prepared.get('provenance', {}).get
prepared.get
source.get
ValueError
FrozenPrimitives
list
range
np.asarray
CollisionOwner
PosePath
np.array
np.stack
tuple
Attachment
```
