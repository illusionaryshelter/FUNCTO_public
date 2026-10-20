# `manipulation/online_contact.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../03-attachment-contact-and-hand.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `_plain`

[定义位置](../../../manipulation/online_contact.py#L37)。

输入签名：

```python
def _plain(value):
```

输出分支的源码表达式：

```python
value.tolist()
value.item()
{str(k): _plain(v) for (k, v) in value.items()}
[_plain(v) for v in value]
value
```

直接调用：

```python
isinstance
value.tolist
value.item
str
_plain
value.items
```

## `require_native_material`

[定义位置](../../../manipulation/online_contact.py#L49)。

输入签名：

```python
def require_native_material(rule):
```

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
isinstance
material.get
len
set
tuple
ValueError
np.isfinite
```

## `_readonly`

[定义位置](../../../manipulation/online_contact.py#L65)。

输入签名：

```python
def _readonly(value, dtype=None):
```

直接读取的对象属性：`result.flags`。

输出分支的源码表达式：

```python
result
```

直接写入的状态或数组位置：`result.flags.writeable`。

直接调用：

```python
np.array
```

## `_field_layout`

[定义位置](../../../manipulation/online_contact.py#L71)。

输入签名：

```python
def _field_layout(rules):
```

输出分支的源码表达式：

```python
(grouped, sorted(needed))
```

直接调用：

```python
memberships.setdefault((rule.actor, index), []).append
memberships.setdefault
memberships.items
grouped.setdefault((actor, tuple(sorted(names))), []).append
grouped.setdefault
tuple
sorted
set
len
ValueError
needed.update
range
combinations
```

## `_field_geometry_keys`

[定义位置](../../../manipulation/online_contact.py#L91)。

输入签名：

```python
def _field_geometry_keys(rules, subsets):
```

输出分支的源码表达式：

```python
{names: frozenset((regions[name] for name in names)) for names in subsets}
```

直接调用：

```python
tuple
np.asarray(rule.patch_origin, dtype=float).reshape
np.asarray
np.asarray(rule.patch_axes, dtype=float).reshape
np.asarray(rule.patch_half_size, dtype=float).reshape
float
frozenset
```

## `PreparedContactFields`

[定义位置](../../../manipulation/online_contact.py#L110)。

```python
class PreparedContactFields:
    ...
    ...
```

该类型的直接方法：`__init__`、`prepare`、`validate_binding`、`constant`、`_group_constants`、`query`、`required_contact`、`record`、`from_record`。

## `PreparedContactFields.__init__`

[定义位置](../../../manipulation/online_contact.py#L122)。

输入签名：

```python
def PreparedContactFields.__init__(self, *, origin, voxel_size, shape, rules, maximum_radii, regions, channels, groups, source):
```

直接读取的对象属性：`self.origin`、`self.shape`、`self.voxel_size`、`self.channels`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接写入的状态或数组位置：`self.origin`、`self.voxel_size`、`self.shape`、`self.upper`、`self.rule_records`、`self.maximum_radii`、`self.regions`、`self.channels`、`self.groups`、`self.source`、`self._device`、`self.storage_nbytes`。

直接调用：

```python
_readonly
float
tuple
int
np.asarray
_plain
asdict
maximum_radii.items
str
sum
{id(row['distance']): row['distance'] for row in self.channels if row['kind'] == 'field'}.values
id
```

## `PreparedContactFields.prepare`

[定义位置](../../../manipulation/online_contact.py#L140)。

输入签名：

```python
@classmethod
def PreparedContactFields.prepare(cls, *, distance, origin, voxel_size, rules, maximum_radii, source, source_geometry, occupied=None):
```

输出分支的源码表达式：

```python
cls(origin=origin, voxel_size=h, shape=shape, rules=rules, maximum_radii=maximum_radii, regions=regions, channels=channels, groups=groups, source=source)
```

直接写入的状态或数组位置：`selected[start:start + len(local)]`、`mask[tuple(chosen.T)]`、`masks[rule.rule_id]`、`regions[rule.rule_id]`、`lookup[names]`、`prepared_geometry[geometry_key]`。

直接调用：

```python
tuple
len
ValueError
require_native_material
set
any
np.isfinite
maximum_radii.values
np.asarray
float
min
np.isfinite(origin).all
_field_layout
_field_geometry_keys
geometry_keys.values
require_host_memory
int
np.argwhere
np.abs
np.empty
source_geometry.prepare_rules
np.array
np.any
patch_corners.min
patch_corners.max
np.abs(axes).sum
np.zeros
range
(np.abs(local) + half_support <= widths + tolerance).all
patch_points.min
patch_points.max
np.flatnonzero
source_geometry.rule_region
np.asarray(source_geometry.surface_points(rule.rule_id), dtype=float).reshape
source_geometry.surface_points
np.clip
((xy / rule.patch_half_size + 1) * (_WITNESS_BINS / 2)).astype
witnesses.append
np.argmin
_readonly
bool
np.asarray(witnesses).reshape
math.sqrt
channels.append
source_geometry.distance_field
removed.any
remaining.any
distance_transform_edt
np.nextafter
transformed.astype
np.float32
sorted
grouped.items
enumerate
table.append
groups.append
cls
```

## `PreparedContactFields.validate_binding`

[定义位置](../../../manipulation/online_contact.py#L297)。

输入签名：

```python
def PreparedContactFields.validate_binding(self, rules, maximum_radii):
```

直接读取的对象属性：`self.rule_records`、`self.maximum_radii`。

输出方式：函数体没有显式 `return`；正常结束返回 `None`。

直接调用：

```python
tuple
_plain
asdict
float
maximum_radii.items
ValueError
```

## `PreparedContactFields.constant`

[定义位置](../../../manipulation/online_contact.py#L302)。

输入签名：

```python
def PreparedContactFields.constant(self, name, value, like):
```

直接读取的对象属性：`self._device`。

输出分支的源码表达式：

```python
self._device[key]
```

直接写入的状态或数组位置：`self._device[key]`。

直接调用：

```python
torch.tensor
```

## `PreparedContactFields._group_constants`

[定义位置](../../../manipulation/online_contact.py#L309)。

输入签名：

```python
def PreparedContactFields._group_constants(self, index, points):
```

直接读取的对象属性：`self._device`、`self.channels`、`self.groups`。

输出分支的源码表达式：

```python
self._device[key]
```

直接写入的状态或数组位置：`shared_indices[identity]`、`field_indices[channel_index]`、`self._device[bank_key]`、`self._device[key]`。

直接调用：

```python
enumerate
id
len
arrays.append
channel['distance'].transpose
torch.as_tensor
np.stack
kinds.append
mapping.append
torch.tensor
```

## `PreparedContactFields.query`

[定义位置](../../../manipulation/online_contact.py#L343)。

输入签名：

```python
def PreparedContactFields.query(self, spheres, policies, ordinary, intersects_geometry):
```

直接读取的对象属性：`self.groups`、`self._group_constants`、`self.constant`、`self.origin`、`self.voxel_size`、`self.shape`、`self.upper`。

输出分支的源码表达式：

```python
(ordinary, torch.zeros_like(ordinary))
(values, unsupported)
```

直接调用：

```python
torch.zeros_like
len
ValueError
next
iter
ordinary.clone
enumerate
set
self._group_constants
spheres.index_select
intersects_geometry.index_select
by_rule[name]['active'].to
ordinary.index_select
torch.where
(kind == 1).unsqueeze
self.constant
np.asarray
F.grid_sample
grid.reshape
sampled.reshape
mapping[code].unsqueeze(0).unsqueeze(-1).expand
mapping[code].unsqueeze(0).unsqueeze
mapping[code].unsqueeze
sampled.gather(0, take).squeeze
sampled.gather
torch.relu
(kind == 2).unsqueeze
selected[..., 3:4].clamp_min
((selected[..., :3] - radius >= self.constant('origin', self.origin, spheres)) & (selected[..., :3] + radius <= self.constant('upper', self.upper, spheres))).all
values.index_copy
unsupported.index_copy
missing.to
```

## `PreparedContactFields.required_contact`

[定义位置](../../../manipulation/online_contact.py#L393)。

输入签名：

```python
def PreparedContactFields.required_contact(self, rule, policy):
```

直接读取的对象属性：`self.regions`、`self.constant`。

输出分支的源码表达式：

```python
torch.ones_like(policy['active'], dtype=normals.dtype)
torch.relu(excess).amax(-1).amin(-1)
```

直接调用：

```python
len
torch.ones_like
self.constant
torch.einsum
bounds.unsqueeze
torch.relu(excess).amax(-1).amin
torch.relu(excess).amax
torch.relu
```

## `PreparedContactFields.record`

[定义位置](../../../manipulation/online_contact.py#L403)。

输入签名：

```python
def PreparedContactFields.record(self):
```

直接读取的对象属性：`self.channels`、`self.origin`、`self.voxel_size`、`self.shape`、`self.source`、`self.rule_records`、`self.maximum_radii`、`self.regions`、`self.groups`。

输出分支的源码表达式：

```python
{'model': MODEL, 'origin': self.origin.tolist(), 'voxel_size': self.voxel_size, 'shape': list(self.shape), 'source': self.source, 'rules': _plain(self.rule_records), 'maximum_radii': dict(self.maximum_radii), 'regions': {name: {'lower_patch': row['lower_patch'].tolist(), 'upper_patch': row['upper_patch'].tolist(), 'has_removed_material': row['has_removed_material'], 'removed_elements': row['removed_elements'], 'element_kind': row['element_kind'], 'surface_witnesses': encode_array(row['surface_witnesses'])} for (name, row) in self.regions.items()}, 'channels': [{'rule_ids': list(row['rule_ids']), 'kind': row['kind'], **({'distance': encoded[id(row['distance'])]} if row['kind'] == 'field' else {})} for row in self.channels], 'groups': _plain(self.groups), 'numerics': {'cell_cover_margin_m': math.sqrt(3.0) * self.voxel_size / 2, 'interpolation_margin_m': math.sqrt(3.0) * self.voxel_size / 2, 'maximum_surface_witnesses': _WITNESS_BINS ** 2}, 'scope': 'source components filtered before conservative forbidden-region fields, or registered analytic primitive cells; complete native planar or box-edge material; bounded source surface witnesses may miss contact between points; no Boolean equivalence claim'}
```

直接写入的状态或数组位置：`encoded[id(row['distance'])]`。

直接调用：

```python
id
encode_array
self.origin.tolist
list
_plain
dict
row['lower_patch'].tolist
row['upper_patch'].tolist
self.regions.items
math.sqrt
```

## `PreparedContactFields.from_record`

[定义位置](../../../manipulation/online_contact.py#L428)。

输入签名：

```python
@classmethod
def PreparedContactFields.from_record(cls, row):
```

输出分支的源码表达式：

```python
cls(origin=row['origin'], voxel_size=row['voxel_size'], shape=row['shape'], rules=rules, maximum_radii=row['maximum_radii'], regions=regions, channels=channels, groups=row['groups'], source=row['source'])
```

直接写入的状态或数组位置：`channel['distance']`、`restored_geometry[geometry_key]`、`regions[name]`。

直接调用：

```python
row.get
len
ValueError
np.asarray
isinstance
any
type
math.prod
np.isfinite(origin).all
np.isfinite
np.isscalar
row['source'].strip
tuple
ContactRule
set
row['maximum_radii'].values
require_native_material
_field_layout
enumerate
sorted
grouped.items
range
expected_groups.append
list
_field_geometry_keys
restored_geometry.get
decode_array
np.isfinite(field).all
np.array_equal
channels.append
row['regions'].items
np.isfinite(witnesses).all
np.isfinite([low, high]).all
np.any
_readonly
cls
```

## `native_material_conditions`

[定义位置](../../../manipulation/online_contact.py#L522)。

输入签名：

```python
def native_material_conditions(rule, policy, material_normal, prepared):
```

输出分支的源码表达式：

```python
_box_edge_material_conditions(rule, policy, material_normal, prepared)
({'gap': zero, 'material_band': zero}, torch.zeros_like(policy['active']))
({'gap': gap, 'material_band': penetration}, possible)
```

直接调用：

```python
_box_edge_material_conditions
_tensor
torch.einsum
torch.cat
float
a.unsqueeze
torch.zeros_like
prepared.constant
body.amin
body.amax
((body_upper >= lower) & (body_lower <= upper)).all
rectangle
(a * feature[..., 0, :]).sum
a[..., 2].clamp_max
plane.unsqueeze
(face_xy * a[..., None, :2]).sum
denominator.unsqueeze
torch.relu
heights.amin
heights.amax
torch.where
(body_xy * a[..., None, :2]).sum
coordinate.amin
```

## `native_material_conditions.rectangle`

[定义位置](../../../manipulation/online_contact.py#L563)。

输入签名：

```python
def native_material_conditions.rectangle(points, low, high):
```

输出分支的源码表达式：

```python
(corners, overlap)
```

直接调用：

```python
torch.maximum
points[..., :2].amin
torch.minimum
points[..., :2].amax
(first <= last).all
torch.stack
```

## `_box_edge_material_conditions`

[定义位置](../../../manipulation/online_contact.py#L591)。

输入签名：

```python
def _box_edge_material_conditions(rule, policy, material_normal, prepared):
```

输出分支的源码表达式：

```python
({'gap': zero, 'material_band': zero}, torch.zeros_like(policy['active']))
({'gap': gap, 'material_band': penetration}, possible)
```

直接调用：

```python
_tensor
torch.linalg.vector_norm
torch.linalg.cross
prepared.constant
tangent.unsqueeze
lateral.unsqueeze
normal.unsqueeze
torch.einsum
torch.zeros_like
body.amin
body.amax
((body_upper >= lower) & (body_lower <= upper)).all
torch.ones_like
torch.where
torch.maximum
torch.minimum
start[..., axis].abs
torch.stack
first.clamp
last.clamp
torch.relu
heights.amin
heights.amax
(a * (torch.where(a >= 0, lo, hi) - feature[..., 0, :])).sum
min
torch.as_tensor
torch.nextafter
```
