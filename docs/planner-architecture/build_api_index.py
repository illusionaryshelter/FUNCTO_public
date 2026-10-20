"""Regenerate the documented current component API from its Python syntax.

The module table supplies the component scope. Definitions, signatures,
returns, attributes and direct calls are read from source, never imported.
"""
# BEGIN reproducible API inventory for the current observed-field components.
import ast
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASELINE = sys.argv[1]
CHAPTERS = {
 '01': '01-inputs-and-task-model.md', '02': '02-observation-and-collision.md',
 '03': '03-attachment-contact-and-hand.md', '04': '04-solver-and-search.md',
 '05': '05-acceptance-and-execution.md',
}
# These modules describe retired task-field conversion and contact bindings.
# Known robot geometry adapters remain part of the current component call graph.
RETIRED = {
 'tsdf_geometry', 'online_observed_geometry', 'tsdf_spatial_query',
 'scene_surface_batch', 'native_voxel_sources', 'voxel_layers', 'mapper_surface',
 'online_contact', 'contact_geometry', 'native_owner_geometry',
 'prismatic_self_collision', 'collision_path', 'holding_contact',
 'robot_coverage', 'geometry_motion_bound', 'online_native_confirmation',
 'adapter-prepare_recorded_task', 'adapter-recorded_owner_geometry',
 'adapter-recorded_observation', 'adapter-robot_coverage_fixture',
 'adapter-contact_fixture', 'adapter-run_recorded_simulation',
 'adapter-recorded_simulation', 'adapter-recorded_skill_evaluation',
 'adapter-continuous_simulation',
}


def definitions(body, parent=''):
    for item in body:
        if isinstance(item, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            name = f'{parent}.{item.name}' if parent else item.name
            yield name, item
            yield from definitions(item.body, name)


def local_nodes(node):
    """Walk this definition while leaving nested definitions their own entry."""
    for child in ast.iter_child_nodes(node):
        if not isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            yield child
            yield from local_nodes(child)


def ordered(values):
    return list(dict.fromkeys(values))


def code_block(values):
    return '\n```python\n'+'\n'.join(values)+'\n```\n' if values else '\n无显式条目。\n'


def render(path, chapter):
    tree = ast.parse((ROOT/path).read_text())
    rows = list(definitions(tree.body))
    text = [f'# `{path}` 实现契约\n\n代码基线：`{BASELINE}`。\n',
        f'职责和数据语义见[组件说明](../{CHAPTERS[chapter]})及[API 契约](../06-api-contracts.md)。[返回目录](../07-implementation-index.md)。\n']
    for name, node in rows:
        text.append(f'\n## `{name}`\n\n[定义位置](../../../{path}#L{node.lineno})。\n')
        if isinstance(node, ast.ClassDef):
            bases = ', '.join(ast.unparse(x) for x in node.bases)
            text.append(code_block([f'class {node.name}({bases}):']))
            fields = [ast.unparse(x) for x in node.body if isinstance(x, (ast.AnnAssign, ast.Assign))]
            text.append('\n类字段：\n'+code_block(fields))
        else:
            signature = f'def {node.name}({ast.unparse(node.args)})'
            if node.returns:
                signature += f' -> {ast.unparse(node.returns)}'
            text.append('\n输入签名：\n'+code_block([signature+':']))
        doc = ast.get_docstring(node)
        if doc:
            # BEGIN blank blockquote lines contain no trailing whitespace.
            text.append('\n实现说明：\n\n'+'\n'.join(('> '+line).rstrip() for line in doc.splitlines())+'\n')
            # END retain the original docstring paragraph boundaries.
        if not isinstance(node, ast.ClassDef):
            nodes = list(local_nodes(node))
            returns = [ast.unparse(n.value) if n.value else 'None' for n in nodes
                       if isinstance(n, (ast.Return, ast.Yield, ast.YieldFrom))]
            attrs = ordered(ast.unparse(n) for n in nodes
                            if isinstance(n, ast.Attribute) and isinstance(n.ctx, ast.Load))
            calls = ordered(ast.unparse(n.func) for n in nodes if isinstance(n, ast.Call))
            writes = ordered(ast.unparse(n) for n in nodes
                             if isinstance(n, (ast.Name, ast.Attribute, ast.Subscript))
                             and isinstance(n.ctx, ast.Store))
            text.append('\n显式输出分支：\n'+code_block(returns))
            text.append('\n读取的对象属性：\n'+code_block(attrs))
            text.append('\n写入目标：\n'+code_block(writes))
            text.append('\n直接调用符号：\n'+code_block(calls))
    return ''.join(text), len(rows)


def main():
    index = HERE/'07-implementation-index.md'
    pattern = r'\| \[([^\]]+\.py)\]\(api/([^\)]+)\.md\) \| \d+ \| \[(\d+)\]'
    modules = [(p, page, chapter) for p, page, chapter in re.findall(pattern, index.read_text())
               if page not in RETIRED]
    for module, chapter in [('implicit_distance','02'), ('observed_mapping','02'),
                            ('observed_robot','02'), ('observed_collision','02'),
                            ('observed_coverage','02'), ('grasp_region','03'),
                            ('native_field_distance','02'), ('tool_geometry','02')]:
        row = (f'manipulation/{module}.py', module, chapter)
        if row not in modules:
            modules.append(row)
    row = ('tests/planning/observed_robot_fixture.py', 'adapter-observed_robot_fixture', '05')
    if row not in modules:
        modules.append(row)
    # BEGIN index the direct consumers of the reusable native robot asset.
    for module in ('robot_validation_fixture', 'robot_fixture'):
        row = (f'tests/planning/{module}.py', f'adapter-{module}', '02')
        if row not in modules:
            modules.append(row)
    # END observed task owners and known robot models retain separate adapters.
    total, table = 0, []
    for path, page, chapter in modules:
        content, count = render(path, chapter)
        (HERE/'api'/f'{page}.md').write_text(content)
        total += count
        table.append(f'| [{path}](api/{page}.md) | {count} | [{chapter}]({CHAPTERS[chapter]}) |')
    for page in RETIRED:
        (HERE/'api'/f'{page}.md').unlink(missing_ok=True)
    index.write_text(f'''# 实现符号索引

代码基线：`{BASELINE}`。本索引覆盖下表所列当前组件及实验适配器；各阶段职责、必要输入和调用关系见[API 契约](06-api-contracts.md)。

`build_api_index.py` 从源码语法树生成每个类、函数、方法和局部定义。签名、显式返回、属性读取、写入目标和直接调用均保留源码表达式。属性可能在函数内部建立；调用前置条件由组件正文解释。正常执行至函数末尾时隐式返回 None。

覆盖 **{len(modules)} 个文件、{total} 个定义**。

| 模块 | 定义数 | 组件说明 |
|---|---:|---|
'''+ '\n'.join(table)+'\n')
    print(f'{len(modules)} modules; {total} definitions')


if __name__ == '__main__':
    main()
# END generated pages reflect actual source definitions.
