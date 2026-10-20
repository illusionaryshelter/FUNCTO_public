# `tests/planning/contact_fixture.py` 实现契约

代码基线：`bc79a78071e464c35ed9839d62595f735ad85465`。

本模块的阶段职责与数据语义见[组件说明](../05-acceptance-and-execution.md)和[API 契约](../06-api-contracts.md)。[返回实现目录](../07-implementation-index.md)。

## `native_model_text`

[定义位置](../../../tests/planning/contact_fixture.py#L55)。

输入签名：

```python
def native_model_text(*, pad_width=0.1):
```

输出分支的源码表达式：

```python
(xml, _URDF)
```

直接调用：

```python
ValueError
_XML.replace
```
