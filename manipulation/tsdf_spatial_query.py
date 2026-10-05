"""Bounded spatial scheduling for immutable TSDF source components.

The native tree supplies conservative geometric bounds; Coal still supplies
every retained raw distance. This module owns only one pair query's scheduling
and work counters. Contact permission and decisive-failure policy remain with
the caller, which may stop this generator immediately after a near result.
"""

# BEGIN spatial certificates change leaf work, never component consumption order.
import math

import coal
import numpy as np


def _transform(pose):
    return coal.Transform3s(pose[:3, :3], pose[:3, 3])


def iter_component_distances(geometry, actor_shape, actor_pose, owner_pose, bounds,
                             stop_distance, request, *, previous_component, statistics):
    """Yield ``(flat ID, native component, world pose, result, measured count)``.

    A single early seed distance initializes the geometric upper bound. Its
    result or exception is consumed only at that leaf's original position.
    All later work remains in bounded original-order runs; an accepted contact
    can resume the generator, while a rejected pair need not visit another leaf.

    Bounds certify the represented geometry, not all digits of an unexecuted
    finite-precision Coal iteration. No library tolerance is used as an error
    estimate or subtracted from a speculative bound.
    """
    from ._coal_scene_batch import certified_distance_upper_bound, scan_separated_prefix

    candidates = geometry.component_indices(bounds)
    statistics["query_count"] += 1
    statistics["candidate_count"] += len(candidates)
    statistics["completed"] = False
    if not len(candidates):
        statistics["completed_queries"] += 1
        statistics["completed"] = statistics["completed_queries"] == statistics["query_count"]
        return
    actor_transform, owner_transform = _transform(actor_pose), _transform(owner_pose)
    search = geometry._spatial_index.query(
        actor_shape, actor_transform, owner_transform, candidates, stop_distance)
    statistics["workspace_nbytes_upper"] = max(
        statistics["workspace_nbytes_upper"], int(search.workspace_nbytes)+len(candidates)*32)
    queried, pruned = np.zeros(len(candidates), bool), np.zeros(len(candidates), bool)
    upper = math.inf
    last_bound_count = 0
    seed_id = int(search.seed_id(previous_component))
    errors = {}
    seed = None

    def record_queries(indices):
        # Reusing the seed at its original position does not count a second
        # physical call. Native batches never contain that already queried ID.
        positions = np.searchsorted(candidates, indices)
        queried[positions] = True
        statistics["exact_leaf_query_count"] += len(indices)

    def certify(component, pose, result):
        nonlocal upper
        # A near/possibly permitted component cannot supply an upper bound for
        # the still-required unpermitted separated-component minimum. Keeping
        # all near leaves also preserves signed penetrations and patch evidence.
        if result.min_distance <= stop_distance:
            return False
        value = float(certified_distance_upper_bound(
            actor_shape, actor_transform, component[2], _transform(pose), result))
        if math.isfinite(value) and 0. <= value < upper:
            upper = value
            return True
        return False

    try:
        component = geometry.collision_component(seed_id)
        pose = owner_pose @ component[3]
        result = coal.DistanceResult()
        record_queries(np.asarray([seed_id], dtype=np.intp))
        statistics["seed_query_count"] += 1
        statistics["prior_hint_query_count"] += int(seed_id == previous_component)
        coal.distance(actor_shape, actor_transform, component[2], _transform(pose), request, result)
        if not math.isfinite(float(result.min_distance)):
            raise RuntimeError("Independent scene distance returned nonfinite evidence")
        seed = component, pose, result
        certify(component, pose, result)
    except Exception as error:
        # Even a shape-construction error on the hint is postponed: an earlier
        # component can already reject this pair and make the later work moot.
        errors[seed_id] = error

    def retained():
        nonlocal last_bound_count
        # Distinguish tree revisits from actual bound calculations: the native
        # query may reuse a previously certified bound at a smaller cutoff.
        statistics["retained_pass_count"] += 1
        selected, report = search.retained_ids(upper)
        # Include the seed traversal as well as all later re-filtering. The
        # per-call native delta alone would omit bounds evaluated by seed_id.
        total = int(report["total_node_bound_count"])
        statistics["node_bound_count"] += total-last_bound_count
        last_bound_count = total
        # The actual seed result and any already discovered error must remain
        # consumable in source order, independently of a later tighter bound.
        for forced in {seed_id, *errors}:
            position = int(np.searchsorted(selected, forced))
            if position == len(selected) or selected[position] != forced:
                selected = np.insert(selected, position, forced).astype(np.intp, copy=False)
        kept = np.zeros(len(candidates), bool)
        kept[np.searchsorted(candidates, selected)] = True
        newly_pruned = ~kept & ~queried & ~pruned
        statistics["bound_pruned_count"] += int(np.count_nonzero(newly_pruned))
        pruned[newly_pruned] = True
        return selected

    selected = retained()
    cursor = -1
    while True:
        begin = int(np.searchsorted(selected, cursor, side="right"))
        if begin == len(selected):
            break
        identity = int(selected[begin])
        if identity in errors:
            raise errors[identity]
        if identity == seed_id:
            cursor = identity
            component, pose, result = seed
            yield identity, component, pose, result, 1
            continue

        # Do not duplicate the hinted leaf or prepare across it. Components
        # beyond an earlier near result may be prepared but never queried;
        # preparation errors are retained until their original ID is consumed.
        end = min(begin+4096, len(selected))
        seed_position = int(np.searchsorted(selected, seed_id))
        if begin < seed_position < end:
            end = seed_position
        batch, poses, identities = [], [], []
        for flat_index in selected[begin:end]:
            flat_index = int(flat_index)
            if flat_index in errors:
                break
            try:
                component = geometry.collision_component(flat_index)
                pose = owner_pose @ component[3]
            except Exception as error:
                errors[flat_index] = error
                break
            batch.append(component)
            poses.append(pose)
            identities.append(flat_index)
        if not batch:
            # The next required leaf itself failed preparation. Earlier near
            # results have already been consumed, so this exception is now due.
            raise errors[identity]
        poses = np.asarray(poses, dtype=np.float64)
        identities = np.asarray(identities, dtype=np.intp)
        shapes = [component[2] for component in batch]
        start, tightened = 0, False
        while start < len(batch):
            count, minimum, result, near, near_result = scan_separated_prefix(
                actor_shape, actor_transform, shapes[start:], poses[start:], request, stop_distance)
            consumed = int(count)+int(near >= 0)
            record_queries(identities[start:start+consumed])
            cursor = int(identities[start+consumed-1])
            if minimum >= 0:
                index = start+int(minimum)
                tightened |= certify(batch[index], poses[index], result)
                yield int(identities[index]), batch[index], poses[index], result, int(count)
            if near >= 0:
                index = start+int(near)
                yield int(identities[index]), batch[index], poses[index], near_result, 1
            start += consumed
            if tightened:
                # Refresh only after the consumer has seen the complete queried
                # prefix (including a near leaf). It may have stopped already.
                break
        del batch, poses, identities, shapes
        if tightened:
            selected = retained()
    statistics["completed_queries"] += 1
    statistics["completed"] = statistics["completed_queries"] == statistics["query_count"]
# END query disposal releases only scheduling scratch, never source-map shapes.
