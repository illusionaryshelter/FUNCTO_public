# `manipulation/contact_binding.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `bind_grasp_contact_rules`

[定义位置](../../../manipulation/contact_binding.py#L22)。

输入签名：

```python
def bind_grasp_contact_rules(problem, contacts, activations):
```

直接读取的对象属性：`problem.conditions`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`row['contact_rule_ids']`、`replacements[row['condition_id']]`、`problem.conditions`。

直接调用：

```python
row.get
isinstance
len
set
ValueError
asdict
deepcopy
grasp_contact_rule_ids
used.intersection
used.update
tuple
replacements.get
```

## `calibrate_box_material_features`

[定义位置](../../../manipulation/contact_binding.py#L55)。

输入签名：

```python
def calibrate_box_material_features(model, *, tcp_site, joint_positions, finger_joint_names, material_faces):
```

输出分支的源码表达式：

```python
{row['material_id']: {'material_id': row['material_id'], 'native_geometry': row['native_geometry'], 'feature_kind': 'polygon', 'feature_points': zero[index].tolist(), 'feature_normal': normals[index].tolist(), 'feature_joint_translation': mapping[index].tolist(), 'finger_joint_names': list(names), 'native_thickness_m': float(2 * model.geom_size[local_faces[index][0], row['face_axis']]), 'online_material': {'kind': 'native_planar_extrusion_v1', 'thickness_m': float(2 * model.geom_size[local_faces[index][0], row['face_axis']]), 'native_geometry': row['native_geometry'], 'source': f"{row['source']}; complete native box face and inward extrusion"}, 'affine_fk_max_error_m': error, 'source': row['source']} for (index, row) in enumerate(declarations)}
```

直接写入的状态或数组位置：`data.qpos[model.jnt_qposadr[joint]]`、`vertices[:, axis]`、`vertices[:, tangents]`、`probe[index]`、`mapping[..., index]`。

直接调用：

```python
tuple
len
set
ValueError
model.joint
range
mujoco.MjData
joint_positions.items
np.isfinite
model.site
np.asarray
np.isfinite(bound).all
addresses.append
limits.append
list
any
isinstance
model.geom
type
np.zeros
product
np.eye
local_faces.append
query
np.empty
limits[:, 0].copy
np.allclose
np.einsum
limits.mean
max
float
np.max
np.abs
zero[index].tolist
normals[index].tolist
mapping[index].tolist
enumerate
```

## `calibrate_box_material_features.query`

[定义位置](../../../manipulation/contact_binding.py#L111)。

输入签名：

```python
def calibrate_box_material_features.query(coordinates):
```

输出分支的源码表达式：

```python
(np.asarray(points), np.asarray(normals))
```

直接写入的状态或数组位置：`data.qpos[addresses]`。

直接调用：

```python
mujoco.mj_kinematics
data.site_xmat[site].reshape
data.geom_xmat[geom].reshape
points.append
normals.append
np.asarray
```

## `build_material_contact_rules`

[定义位置](../../../manipulation/contact_binding.py#L160)。

输入签名：

```python
def build_material_contact_rules(material_binding, declarations):
```

直接读取的对象属性：`result.append`。

输出分支的源码表达式：

```python
tuple(result)
```

直接写入的状态或数组位置：`row[name]`、`row['sphere_indices']`、`row['online_material']`、`row['source']`。

直接调用：

```python
dict
row.setdefault
material_binding.get('owner_materials', {}).get
material_binding.get
ValueError
any
tuple
deepcopy
material.get
isinstance
sphere_source.strip
model.update
list
result.append
ContactRule
```
