# `manipulation/mapper_surface.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../02-observation-and-collision.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `extract_tsdf_surface`

[定义位置](../../../manipulation/mapper_surface.py#L11)。

输入签名：

```python
def extract_tsdf_surface(mapper):
```

输出分支的源码表达式：

```python
(vertices, faces)
```

直接调用：

```python
mapper.extract_mesh
mesh.vertices.detach().cpu().numpy().astype
mesh.vertices.detach().cpu().numpy
mesh.vertices.detach().cpu
mesh.vertices.detach
mesh.faces.detach().cpu().numpy().astype
mesh.faces.detach().cpu().numpy
mesh.faces.detach().cpu
mesh.faces.detach
float
```
