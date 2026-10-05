"""Owned evidence for native acceptance, before modes and anchors are folded.

These records describe evaluated discrete states. They never authorize motion,
change a residual, or reconstruct a geometry query at a different precision.
"""

from copy import deepcopy

import numpy as np


# BEGIN attempt-wide failure witnesses, independent of candidate ranking.
class WitnessCapacityError(ValueError):
    """All known failure times cannot fit the admitted refinement workspace."""


class AttemptWitnesses:
    """Keep every discovered failure time until this planning attempt ends.

    The ledger records normalized physical time, not source progress: a pause
    can map several distinct trajectory times to one progress value. Exact
    repeated times share one query. Nearby but distinct times remain distinct
    because neither a numerical tolerance nor a previous passing query proves
    the intervening source geometry safe.
    """

    def __init__(self):
        self._parameters = set()

    @staticmethod
    def _validated_parameters(values, *, source):
        parameters = np.asarray(values, dtype=float)
        if (parameters.ndim != 1 or not np.isfinite(parameters).all()
                or np.any(parameters < 0.) or np.any(parameters > 1.)):
            raise ValueError(f"{source} requires finite normalized physical times in [0, 1]")
        return parameters

    @property
    def parameters(self):
        """Return an owned sorted snapshot; callers cannot erase the ledger."""
        return np.asarray(sorted(self._parameters), dtype=float)

    def add_native(self, native_metrics, *, duration):
        """Union exact times from each bound native source-constraint trace.

        Traces can carry the original normalized query parameter or their
        physical ``time_s`` and frozen duration; the latter ratio is used
        exactly, without clipping or
        epsilon merging. Unavailable/unmatched observer traces contain no
        attributed witness and are left to their existing diagnostic report.
        """
        duration = float(duration)
        if not np.isfinite(duration) or duration <= 0.:
            raise ValueError("Native failure witnesses require a finite positive physical duration")
        parameters = []
        for label in ("rollout", "interpolated"):
            metrics = native_metrics.get(label, {})
            # Unfolded native joint constraints can carry their own original
            # time. An index without that explicit time cannot identify an
            # anchor folded into the last cost column and is never guessed.
            parameters.extend(float(row["normalized_time"])
                              for row in metrics.get("violating_constraint_samples", ())
                              if "normalized_time" in row)
            trace = metrics.get("source_trace", {})
            if trace.get("status") != "recorded":
                continue
            for row in trace.get("component_failures", ()):
                # A recorded component failure is already selected using its
                # original hard residual, including nonfinite rejection.
                parameters.append(float(row["normalized_time"]) if "normalized_time" in row
                                  else float(row["time_s"])/duration)
        parameters = self._validated_parameters(parameters, source="Native failure witnesses")
        self._parameters.update(parameters.tolist())

    def add_validation(self, validation):
        """Union every failed time from one completed independent curve check."""
        parameters = self._validated_parameters(validation["u"], source="Independent validation")
        violations = np.asarray(validation["violations"], dtype=float)
        if violations.shape != parameters.shape:
            raise ValueError("Independent validation needs one hard residual per physical query time")
        failed = (violations > 0.) | ~np.isfinite(violations)
        self._parameters.update(parameters[failed].tolist())

    def merged_parameters(self, *, fixed_parameters, existing_parameters, regular_parameters,
                          max_points):
        """Admit the complete union before a caller mutates solver workspaces.

        Fixed anchors and regular optimizer samples already evaluate those
        exact times. Only newly witnessed, uncovered times require extra
        refinement storage. Previously active refinements are retained even
        if a caller supplies a changed regular grid; the method never removes
        an active constraint or truncates the union to fit a finite budget.
        """
        if type(max_points) is not int or max_points < 0:
            raise ValueError("The refinement witness capacity must be a nonnegative integer")
        fixed = self._validated_parameters(fixed_parameters, source="Fixed anchors")
        existing = self._validated_parameters(existing_parameters, source="Existing refinements")
        regular = self._validated_parameters(regular_parameters, source="Regular optimizer samples")
        covered = set(fixed.tolist()) | set(regular.tolist())
        merged = set(existing.tolist()) | (self._parameters-covered)
        if len(merged) > max_points:
            raise WitnessCapacityError(
                "The complete failure-witness union exceeds the finite refinement-point budget: "
                f"requires {len(merged)} points, allowed {max_points}; no failure time was discarded")
        return np.asarray(sorted(merged), dtype=float)


def capture_component_failures(components, times, progress, joints, position, quaternion, *,
                               normalized_time=None, mode="exact"):
    """Capture process/self residual times before anchor columns are folded.

    Inputs use the already ordered native query layout ``[batch, time, ...]``.
    Each component's trailing dimensions are reduced only within that same
    state. The resulting rows share the existing scene-trace schema, so the
    caller can attach them to the exact named joint-state trace without
    repeating FK, process evaluation or a source-geometry query.
    """
    cpu = lambda value: value.detach().cpu().numpy().copy()
    shape = tuple(times.shape)
    if len(shape) != 2:
        raise ValueError("Failure capture requires native [batch, time] query times")
    rows, state, parameters = [], None, None
    for name, tensor in components.items():
        values = cpu(tensor)
        if values.ndim < 2 or values.shape[:2] != shape:
            raise ValueError("Hard component samples must match their physical query times")
        values = values.reshape(*shape, -1).max(axis=-1)
        for batch, column in np.argwhere((values > 0.) | ~np.isfinite(values)):
            if state is None:
                state = [cpu(value) for value in (times, progress, joints, position, quaternion)]
                if any(value.shape[:2] != shape for value in state):
                    raise ValueError("Failure state samples must match their physical query times")
                if normalized_time is not None:
                    parameters = cpu(normalized_time)
                    if parameters.shape != shape:
                        raise ValueError("Normalized failure times must match the native query layout")
            time_s, phase, q, p, rotation = (value[batch, column] for value in state)
            row = {"component": name, "raw_residual": float(values[batch, column]),
                "batch_index": int(batch), "ordered_query_index": int(column),
                "expanded_query_index": int(column), "mode": mode,
                "time_s": float(time_s), "progress": float(phase),
                "permission_progress": float(phase), "joint_position": q.tolist(),
                "tool_position": p.tolist(), "tool_quaternion_wxyz": rotation.tolist()}
            if parameters is not None:
                row["normalized_time"] = float(parameters[batch, column])
            rows.append(row)
    return rows
# END all discovered physical failure times survive candidate and stage changes.


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
