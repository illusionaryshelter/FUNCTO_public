# `manipulation/reference_visualization.py` 实现契约

代码基线：`ae2603f`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `_project`

[定义位置](../../../manipulation/reference_visualization.py#L20)。

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

[定义位置](../../../manipulation/reference_visualization.py#L27)。

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

## `write_reference_html`

[定义位置](../../../manipulation/reference_visualization.py#L35)。

输入签名：

```python
def write_reference_html(directory, skill, views, transfer, reference, report):
```

实现说明：

> Show observed regions and nominal source-time paths with explicit goals.

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
transfer.get
outputs['result'].get
picture.save
np.unique
np.r_
path.progress
np.linspace
np.array
path.sample_numpy
times.tolist
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
name
path
original_times
progress
poses
times
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
source_image.save
dict
_axes
transfer.get
enumerate
outputs['result'].get
region_picture
picture.save
len
np.unique
np.linspace
np.array
path.sample_numpy
_project
times.tolist
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
html.replace('CARDS', cards).replace('DETAILS', escape(details)).replace
html.replace('CARDS', cards).replace
(directory / 'index.html').write_text
```
