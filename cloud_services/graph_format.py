"""Parse the literal scene-graph response shared by both graph interfaces."""

# BEGIN shared literal-only graph response contract.
import ast
import re


GRAPH_INSTRUCTION = """Generate a scene graph of the image. Nodes describe objects;
edges describe spatial relations ON or INSIDE. Give every object a unique name
formed from its class and an instance number. For each node, include ALL matching
predicates from the user's request as separate string fields; omit predicate
fields if none match or the requested predicate is None.
Return ONLY two Python literal assignments, nodes = [...] and edges = [...],
with no Markdown fences, explanation, ellipses, imports, or executable code.
Each node is a tuple (name, predicate1, predicate2, ...), or (name,) if it has no
predicate. Each edge is a tuple (relation, source_name, destination_name).
For example:
nodes = [('object1', 'red', 'marked'), ('container1',)]
edges = [('INSIDE', 'object1', 'container1')]
"""


def strip_literal_fence(text: str) -> str:
    """Unwrap one complete Python/plain fence, never extract surrounding prose."""
    text = text.strip()
    match = re.fullmatch(r"```(?:python)?[ \t]*\r?\n(.*?)\r?\n```", text, re.S | re.I)
    return match.group(1).strip() if match else text


def parse_graph_assignments(text: str):
    """Read nodes/edges as data without executing any model-generated code."""
    assignments = {}
    for statement in ast.parse(strip_literal_fence(text), mode="exec").body:
        if not (isinstance(statement, ast.Assign) and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)):
            raise ValueError("Scene graph must contain only nodes/edges assignments")
        name = statement.targets[0].id
        if name not in {"nodes", "edges"} or name in assignments:
            raise ValueError("Scene graph requires exactly one nodes and one edges assignment")
        value = ast.literal_eval(statement.value)
        if not isinstance(value, list):
            raise ValueError(f"Scene graph {name} must be a literal list")
        assignments[name] = value
    if set(assignments) != {"nodes", "edges"}:
        raise ValueError("Scene graph requires nodes and edges assignments")

    nodes, names = [], set()
    for node in assignments["nodes"]:
        # Parentheses around a lone name produce a string rather than a tuple.
        fields = (node,) if isinstance(node, str) else node
        if not (isinstance(fields, (list, tuple)) and fields
                and isinstance(fields[0], str) and fields[0].strip()
                and all(field is None or isinstance(field, str) for field in fields[1:])):
            raise ValueError(f"Invalid scene graph node: {node!r}")
        if fields[0] in names:
            raise ValueError(f"Duplicate scene graph node name: {fields[0]}")
        names.add(fields[0])
        # Both public parsers already use comma-separated predicate strings.
        nodes.append((fields[0], ", ".join(field for field in fields[1:] if field is not None)))

    edges = assignments["edges"]
    for edge in edges:
        if not (isinstance(edge, (list, tuple)) and len(edge) == 3
                and all(isinstance(field, str) and field.strip() for field in edge)):
            raise ValueError(f"Invalid scene graph edge: {edge!r}")
    return nodes, [tuple(edge) for edge in edges]
# END shared literal-only graph response contract.
