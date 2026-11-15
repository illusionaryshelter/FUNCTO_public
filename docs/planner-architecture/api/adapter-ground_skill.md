# `config/ground_skill.py` 实现契约

代码基线：`c568b75`。
职责和数据语义见[组件说明](../01-inputs-and-task-model.md)及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。

## `ground_skill`

[定义位置](../../../config/ground_skill.py#L21)。

输入签名：

```python
def ground_skill(skill_directory, scene_directory, output, *, instruction, relations=None, view_ids=None, parameters=None):
```

实现说明：

> Read one capture and call the same public API used by programmatic callers.
>
> parameters supplies explicit region/grasp model settings, current
> relation/model declarations and prerequisite context. view_ids names
> the actual sensor folders; when omitted the capture's calibrated folders
> define their deterministic order. Native simulation files are not opened.

显式输出分支：

```python
save(result, request, output)
```

读取的对象属性：

```python
output.mkdir
folder.name
scene_directory.iterdir
folder.is_dir
(folder / 'calibration.json').is_file
Path(skill_directory).resolve
parameters.get
```

写入目标：

```python
skill
scene_directory
output
parameters
view_ids
folder
capture
binding_id
context
entity_id
request
result
```

直接调用符号：

```python
load_grounding_skill
Path
output.mkdir
dict
sorted
scene_directory.iterdir
folder.is_dir
(folder / 'calibration.json').is_file
load_scene
str
Path(skill_directory).resolve
parameters.get
ground_loaded_skill
save
```

## `main`

[定义位置](../../../config/ground_skill.py#L61)。

输入签名：

```python
def main():
```

显式输出分支：

无显式条目。

读取的对象属性：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
arguments.relations
json.loads
arguments.relations.read_text
arguments.parameters
arguments.parameters.read_text
arguments.skill
arguments.scene
arguments.output
arguments.instruction
arguments.view_ids
json.dumps
```

写入目标：

```python
parser
arguments
relations
parameters
record
key
```

直接调用符号：

```python
argparse.ArgumentParser
parser.add_argument
parser.parse_args
json.loads
arguments.relations.read_text
arguments.parameters.read_text
ground_skill
print
json.dumps
SystemExit
```
