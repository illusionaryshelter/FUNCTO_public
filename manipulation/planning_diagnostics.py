"""Owned evidence for native acceptance, before modes and anchors are folded.

These records describe evaluated discrete states. They never authorize motion,
change a residual, or reconstruct a geometry query at a different precision.
"""

from copy import deepcopy

import numpy as np


def scene_component_failures(components, times, progress, permission, joints,
                             position, quaternion, *, modes):
    """Retain every positive/nonfinite hard component in its original mode.

    Unknown observation coverage remains informational under the existing
    scene law. Missing release anchors are hard requirements under that same
    law. A mode index refers to the pre-fold query, never to its maximum.
    """
    cpu = lambda value: value.detach().cpu().numpy().copy()
    hard = [(name, value) for name, value in components.items()
            if not name.startswith("unknown/") or name.startswith("unknown/release_anchor/")]
    result = []
    state = None
    width = times.shape[1] // len(modes)
    for name, tensor in hard:
        values = cpu(tensor)
        for batch, column in np.argwhere((values > 0) | ~np.isfinite(values)):
            if state is None:
                state = [cpu(value) for value in (times, progress, permission, joints, position, quaternion)]
            time_s, phase, permitted, q, p, rotation = (value[batch, column] for value in state)
            result.append({"component": name, "raw_residual": float(values[batch, column]),
                "batch_index": int(batch), "ordered_query_index": int(column % width),
                "expanded_query_index": int(column),
                "mode": modes[column // width], "time_s": float(time_s),
                "progress": float(phase), "permission_progress": float(permitted),
                "joint_position": q.tolist(), "tool_position": p.tolist(),
                "tool_quaternion_wxyz": rotation.tolist()})
    return result


def bind_trace_to_native_seed(trace, state, seed_index, joint_names):
    """Bind by complete typed q/dq bytes, independently of native seed ranking.

    Native result sorting can permute seeds after metrics evaluation. Comparing
    the original named states avoids attributing another seed's scene witness.
    Identical q/dq rows share the same fixed-clock scene evaluation.
    """
    if trace is None:
        return {"status": "unavailable", "reason": "No source metrics trace was retained"}
    if "error" in trace:
        return {"status": "error", "error": trace["error"]}
    columns = [state.joint_names.index(name) for name in joint_names]
    arrays = [getattr(state, name)[0, seed_index].detach().cpu().numpy()[:, columns]
              for name in ("position", "velocity")]
    matches = [index for index in range(len(trace["input_position"]))
        if all(actual.dtype == stored[index].dtype and actual.shape == stored[index].shape
               and actual.tobytes() == stored[index].tobytes()
               for actual, stored in zip(arrays, (trace["input_position"], trace["input_velocity"])))]
    if not matches:
        return {"status": "unmatched", "reason": "No exact named q/dq input matches this result seed"}
    index = matches[0]
    failures = deepcopy([row for row in trace["component_failures"] if row["batch_index"] == index])
    witnesses = trace["world_geometry"].get("failure_witnesses", [])
    references = {(row["batch_index"], row["expanded_query_index"]): row["witness_index"]
                  for row in trace["world_geometry"].get("failure_witness_references", [])}
    for row in failures:
        if row["component"] == "source_world_collision":
            reference = references[(row["batch_index"], row["expanded_query_index"])]
            row["geometry_witnesses"] = [deepcopy(witnesses[reference])]
    return {"status": "recorded", "source_batch_index": index,
        "component_failures": failures,
        "world_geometry": deepcopy(trace["world_geometry"]),
        "world_geometry_scope": "Worst witness across this metrics call; joint/time identity is explicit",
        "scope": "Original source components before mode maximum, constraint weighting and anchor folding"}
