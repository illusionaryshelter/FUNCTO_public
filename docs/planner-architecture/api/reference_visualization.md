# `manipulation/reference_visualization.py` 实现契约

代码基线：`c059334`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_project`

[定义位置](../../../manipulation/reference_visualization.py#L23)。

输入签名：

```python
def _project(points, calibration):
```

显式输出分支：

```python
[None if point[2] <= 0 else (point[:2] / point[2]).tolist() for point in pixels]
```

读取的对象属性：

```python
np.asarray
K.T
(point[:2] / point[2]).tolist
```

写入目标：

```python
camera
K
local
pixels
point
```

直接调用符号：

```python
np.asarray
(point[:2] / point[2]).tolist
```

## `_axes`

[定义位置](../../../manipulation/reference_visualization.py#L30)。

输入签名：

```python
def _axes(frame, calibration):
```

显式输出分支：

```python
None
_project(points, calibration)
```

读取的对象属性：

```python
np.asarray
np.vstack
pose[:3, :3].T
```

写入目标：

```python
pose
points
```

直接调用符号：

```python
np.asarray
np.vstack
_project
```

## `_pose_track`

[定义位置](../../../manipulation/reference_visualization.py#L39)。

输入签名：

```python
def _pose_track(poses, times, calibration, *, label, source):
```

实现说明：

> Project a sampled pose track and retain its numerical orientation.
>
> Each pose maps the named moving frame into the panel's world frame.
> Axis endpoints are four centimetres from its origin. RPY is extrinsic
> xyz in degrees; the complete rotation matrix is the primary orientation.

显式输出分支：

```python
dict(pixels=_project(poses[:, :3, 3], calibration), times=np.asarray(times).tolist(), axes=[_axes(pose, calibration) for pose in poses], poses=poses.tolist(), rpy_deg=Rotation.from_matrix(poses[:, :3, :3]).as_euler('xyz', degrees=True).tolist(), label=label, source=source)
```

读取的对象属性：

```python
np.asarray
np.asarray(times).tolist
poses.tolist
Rotation.from_matrix(poses[:, :3, :3]).as_euler('xyz', degrees=True).tolist
Rotation.from_matrix(poses[:, :3, :3]).as_euler
Rotation.from_matrix
```

写入目标：

```python
poses
pose
```

直接调用符号：

```python
np.asarray
dict
_project
np.asarray(times).tolist
_axes
poses.tolist
Rotation.from_matrix(poses[:, :3, :3]).as_euler('xyz', degrees=True).tolist
Rotation.from_matrix(poses[:, :3, :3]).as_euler
Rotation.from_matrix
```

## `_entity_outline`

[定义位置](../../../manipulation/reference_visualization.py#L53)。

输入签名：

```python
def _entity_outline(image, mask):
```

实现说明：

> Retain the selected observed entity extent around its functional region.

显式输出分支：

```python
Image.fromarray(pixels)
```

读取的对象属性：

```python
np.array
np.ones
Image.fromarray
```

写入目标：

```python
pixels
boundary
pixels[boundary]
```

直接调用符号：

```python
np.array
binary_erosion
np.ones
Image.fromarray
```

## `write_reference_html`

[定义位置](../../../manipulation/reference_visualization.py#L62)。

输入签名：

```python
def write_reference_html(directory, skill, views, transfer, reference, report):
```

实现说明：

> Show source motion and transferred full poses in calibrated scene images.
>
> A shared source-time sample drives one pose per displayed track. Source
> poses are recorded demonstration evidence; current paths are generated
> nominal references. A grounding report has no current execution trajectory.

显式输出分支：

无显式条目。

读取的对象属性：

```python
np.asarray
source.get
np.zeros
original.shape
Image.fromarray
np.any
ImageDraw.Draw
draw.ellipse
source_image.save
np.linspace
clocks.append
np.unique
np.concatenate
transfer.get
outputs['result'].get
view.get
picture.save
path.sample_numpy
reference['eef_path'].sample_numpy
markers.extend
markers.append
panels.append
image.shape
json.dumps
''.join
json.dumps(panels, ensure_ascii=False).replace
html.replace
report.get
'；'.join
(directory / 'region.json').is_file
reference.get
(directory / 'reference.npz').is_file
(directory / 'reference-paths.json').is_file
html.replace('CARDS', cards).replace('DETAILS', escape(details)).replace
html.replace('CARDS', cards).replace
(directory / 'index.html').write_text
```

写入目标：

```python
directory
source
original
region_mask
source_picture
source_image
indices
membership
draw
x
y
source_frame
representation
source_times
clocks
name
display_times
source_paths
field
label
poses
source_paths[name]
panels
outputs
frame
index
view
image
picture
modes
_
filename
paths
path
original_times
times
progress
paths[name]
markers
declaration
initial
nominal
goal
point
details
cards
panel
data
html
failure
warnings
warning_text
```

直接调用符号：

```python
Path
np.asarray
source.get
np.zeros
overlay
Image.fromarray
np.any
source_queries
ImageDraw.Draw
draw.ellipse
_entity_outline
source_image.save
np.linspace
clocks.append
np.unique
np.concatenate
sample_poses
_pose_track
dict
_axes
transfer.get
enumerate
outputs['result'].get
region_picture
view.get
picture.save
len
path.sample_numpy
_project
reference['eef_path'].sample_numpy
markers.extend
markers.append
float
panels.append
json.dumps
''.join
escape
json.dumps(panels, ensure_ascii=False).replace
html.replace
str
report.get
'；'.join
(directory / 'region.json').is_file
reference.get
(directory / 'reference.npz').is_file
(directory / 'reference-paths.json').is_file
html.replace('CARDS', cards).replace('DETAILS', escape(details)).replace
html.replace('CARDS', cards).replace
(directory / 'index.html').write_text
```
