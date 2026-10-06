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


# BEGIN bound component concurrency independently of the completed surface batch.
COMPONENT_QUERY_WORKERS = 8


def component_batch_workspace_bytes(request):
    """Reserve one 4096-slot batch and its native worker stacks/solvers.

    Persistent shape pointers and prepared world matrices belong to the source
    geometry's separate admission. Python miss preparation, result slots and
    deferred exceptions fit the existing 5-KiB per-item batch allowance; each
    worker additionally owns its native stack and a cold GJK/EPA workspace.
    """
    solver = 32768+512*int(request.epa_max_iterations)
    return 4096*(4096+1024)+COMPONENT_QUERY_WORKERS*(8*1024**2+solver)
# END the bound is independent of trajectory length and selected component count.


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
    from ._coal_scene_batch import certified_distance_upper_bound

    candidates = geometry.component_indices(bounds)
    statistics["query_count"] += 1
    statistics["candidate_count"] += len(candidates)
    statistics["completed"] = False
    # BEGIN report physical work even for a pair with no retained candidates.
    execution = statistics.setdefault("execution", {
        "physical_leaf_query_count": 0, "consumed_leaf_query_count": 0,
        "unconsumed_leaf_query_count": 0, "prepared_component_count": 0,
        "reused_component_count": 0, "native_batch_count": 0, "worker_count_max": 0})
    # END zero work is explicit, without allocating a prepared target table.
    if not len(candidates):
        statistics["completed_queries"] += 1
        statistics["completed"] = statistics["completed_queries"] == statistics["query_count"]
        return
    # BEGIN only target-side exact operands survive between independent states.
    prepared = geometry.prepared_components(owner_pose)
    # END the existing tree still depends on this actor and current state.
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
        # Count evidence admitted by the original sequential schedule. A seed
        # is admitted once, never again at its original position. Parallel
        # suffix work is measured separately even when the caller rejects near.
        positions = np.searchsorted(candidates, indices)
        queried[positions] = True
        statistics["exact_leaf_query_count"] += len(indices)
        execution["consumed_leaf_query_count"] += len(indices)
        execution["unconsumed_leaf_query_count"] = (
            execution["physical_leaf_query_count"]-execution["consumed_leaf_query_count"])

    # BEGIN Python prepares only newly encountered slots in the original segment.
    def prepare(indices):
        """Return the valid original prefix and newly created metadata only.

        A construction error still truncates preparation at its original ID.
        Native conversion errors remain in their slots and are raised only
        when batch.prefix() reaches them, after any preceding near decision.
        """
        missing = prepared.missing(indices)
        new_components, new_ids, shapes, poses = {}, [], [], []
        count = len(indices)
        for value in missing:
            identity = int(value)
            try:
                component = geometry.collision_component(identity)
                pose = owner_pose @ component[3]
            except Exception as error:
                errors[identity] = error
                count = int(np.searchsorted(indices, identity))
                break
            new_components[identity] = component
            new_ids.append(identity)
            shapes.append(component[2])
            poses.append(pose)
        if new_ids:
            prepared.register(np.asarray(new_ids, dtype=np.intp), shapes,
                              np.asarray(poses, dtype=np.float64))
        execution["prepared_component_count"] += len(new_ids)
        execution["reused_component_count"] += count-len(new_ids)
        return indices[:count], new_components

    def evidence(identity, created):
        component = created.get(identity)
        if component is None:
            component = geometry.collision_component(identity)
        return component, prepared.pose(identity)
    # END a hot segment performs no per-component Python lookup or matrix product.

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
        # BEGIN the seed retains its original first physical call and exception.
        indices, created = prepare(np.asarray([seed_id], dtype=np.intp))
        if not len(indices):
            raise errors[seed_id]
        component, pose = evidence(seed_id, created)
        result = coal.DistanceResult()
        execution["physical_leaf_query_count"] += 1
        record_queries(np.asarray([seed_id], dtype=np.intp))
        statistics["seed_query_count"] += 1
        statistics["prior_hint_query_count"] += int(seed_id == previous_component)
        coal.distance(actor_shape, actor_transform, component[2], _transform(pose), request, result)
        if not math.isfinite(float(result.min_distance)):
            raise RuntimeError("Independent scene distance returned nonfinite evidence")
        seed = component, pose, result
        certify(component, pose, result)
        # END previous IDs never reuse an earlier distance or request history.
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
        # BEGIN reuse immutable prepared slots without moving the old segment edge.
        for failed in errors:
            position = int(np.searchsorted(selected, failed))
            if begin <= position < end:
                end = position
        identities, created = prepare(selected[begin:end])
        if not len(identities):
            # The next required leaf itself failed preparation. Earlier near
            # results have already been consumed, so this exception is now due.
            raise errors[identity]
        batch = prepared.scan(actor_shape, actor_transform, identities, request, COMPONENT_QUERY_WORKERS)
        execution["physical_leaf_query_count"] += int(batch.physical_query_count)
        execution["unconsumed_leaf_query_count"] = (
            execution["physical_leaf_query_count"]-execution["consumed_leaf_query_count"])
        execution["native_batch_count"] += 1
        execution["worker_count_max"] = max(execution["worker_count_max"], int(batch.worker_count))
        start, tightened = 0, False
        while start < len(batch):
            count, minimum, result, near, near_result = batch.prefix(start, stop_distance)
            consumed = int(count)+int(near >= 0)
            record_queries(identities[start:start+consumed])
            cursor = int(identities[start+consumed-1])
            if minimum >= 0:
                index = start+int(minimum)
                identity = int(identities[index])
                component, pose = evidence(identity, created)
                tightened |= certify(component, pose, result)
                yield identity, component, pose, result, int(count)
            if near >= 0:
                index = start+int(near)
                identity = int(identities[index])
                component, pose = evidence(identity, created)
                yield identity, component, pose, near_result, 1
            start += consumed
            if tightened:
                # Refresh only after the consumer has seen the complete queried
                # prefix (including a near leaf). It may have stopped already.
                break
        del batch, identities, created
        # END unconsumed speculative suffixes never tighten the spatial bound.
        if tightened:
            selected = retained()
    statistics["completed_queries"] += 1
    statistics["completed"] = statistics["completed_queries"] == statistics["query_count"]
# END query disposal releases only scheduling scratch, never source-map shapes.
