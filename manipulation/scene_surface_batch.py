"""Prefetch one state's independent zero-surface distances in original order.

This module owns neither signed TSDF components nor collision/contact policy.
It only prepares the same cold Coal calls earlier, then supplies their evidence
at each original consumption point. The real last-result cache changes there,
never while independent jobs execute.
"""

# BEGIN independent surface jobs preserve scene enumeration and cache history.
import coal
import numpy as np

from ._coal_scene_batch import ordered_distances
from .tsdf_geometry import ObservedTsdfGeometry


# BEGIN use the user's eight-worker default, including the calling thread.
SURFACE_QUERY_WORKERS = 8
# END component batches run later, so the two worker groups never nest.


def selected_body_pairs(actor, targets, decisions, visited):
    """Share the exact whole-body encounter order with the component iterator.

    Non-surface pairs also enter ``visited``. In particular an earlier TSDF
    actor/analytic target can suppress the later reverse surface encounter.
    The selected layer is tested first, so an unselected coarse layer cannot
    hide the same named physical body in the selected fine layer.
    """
    for target in targets:
        if actor["owner"] == target["owner"]:
            continue
        selected = decisions.get((actor["owner"], actor["name"], target["owner"]))
        if target["layer"] is not None and (selected is None or target["layer"] != selected["layer"]):
            continue
        key = tuple(sorted(((actor["owner"], actor["name"]), (target["owner"], target["name"]))))
        if key in visited:
            continue
        visited.add(key)
        yield target, key, selected


def surface_batch_workspace_bytes(pair_bound, request):
    """Reserve descriptors/results and the selected private worker workspaces.

    Shared native shapes/BVHs and the persistent last-result cache are already
    charged by the scene. The new 16-KiB per-pair reserve includes Python slots,
    shadow entries, pose/signature bytes, native containers and delayed errors.
    Native container bytes remain separately observable; this conservative
    admission reserve does not pretend to measure third-party peak allocation.
    """
    # BEGIN admission follows the production worker count, not an obsolete cap.
    workers = min(pair_bound, SURFACE_QUERY_WORKERS)
    # END benchmark overrides change both dispatch and its conservative reserve.
    solver = 32768 + 512 * int(request.epa_max_iterations)
    return int(pair_bound * 16384 + workers * (8 * 1024**2 + solver))


class SurfaceDistanceBatch:
    """Bounded jobs for one state, consumed in its original pair order.

    The shadow cache models only the last signature for each native shape pair:
    A->A shares one job, while A->B->A still requests three original cold calls.
    A slot is keyed by physical body identity, not the possibly aliased shapes.
    """

    def __init__(self, actors, targets, decisions, request, surface_cache):
        self._request, self._cache = request, surface_cache
        self._slots, shadow, visited = {}, {}, set()
        self._first_shapes, self._first_transforms = [], []
        self._second_shapes, self._second_transforms = [], []
        self._batch = None
        self._cache_hit_count = 0
        # Preparation must not replace an earlier component/permission error.
        # Invalid enumeration itself will be repeated by selected_body_pairs
        # at its original position; retain only the already prepared prefix.
        try:
            for actor in actors:
                for target, key, _selected in selected_body_pairs(actor, targets, decisions, visited):
                    if (isinstance(actor["shape"], ObservedTsdfGeometry)
                            or not isinstance(target["shape"], ObservedTsdfGeometry)
                            or target["shape"].surface_shape is None):
                        continue
                    try:
                        self._slots[key] = self._prepare(actor, target, shadow)
                    except Exception as error:
                        self._slots[key] = error
        except Exception:
            # No alternate query backend is installed for an invalid suffix.
            # The original enumeration raises before reaching a missing slot.
            pass
        if self._first_shapes:
            try:
                self._batch = ordered_distances(self._first_shapes, self._first_transforms,
                    self._second_shapes, self._second_transforms, request, SURFACE_QUERY_WORKERS)
            except Exception as error:
                # Global native input errors belong to the first required miss,
                # after any earlier signed-component or contact-rule checks.
                self._batch = error

    def _prepare(self, actor, target, shadow):
        surface = target["shape"].surface_shape
        cache_key = (id(actor["shape"]), id(surface))
        signature = (self._request.saveToString(), actor["pose"].dtype.str,
            actor["pose"].tobytes(), target["pose"].dtype.str, target["pose"].tobytes())
        previous = shadow.get(cache_key)
        if previous is None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                previous = (cached[0], cached[1:])
        if previous is not None and previous[0] == signature:
            token = previous[1]
            self._cache_hit_count += 1
            miss = False
        else:
            # Construct both transforms before extending any aligned job list.
            transforms = [coal.Transform3s(row["pose"][:3, :3], row["pose"][:3, 3])
                          for row in (actor, target)]
            token = len(self._first_shapes)
            self._first_shapes.append(actor["shape"])
            self._first_transforms.append(transforms[0])
            self._second_shapes.append(surface)
            self._second_transforms.append(transforms[1])
            miss = True
        shadow[cache_key] = signature, token
        return cache_key, signature, token, miss

    def consume(self, actor, target):
        """Return raw evidence or raise only at the original surface position."""
        key = tuple(sorted(((actor["owner"], actor["name"]), (target["owner"], target["name"]))))
        slot = self._slots[key]
        if isinstance(slot, Exception):
            raise slot
        cache_key, signature, token, miss = slot
        if isinstance(token, int):
            if isinstance(self._batch, Exception):
                raise self._batch
            result = self._batch.result(token)
            value = float(result.min_distance)
            if not np.isfinite(value):
                raise RuntimeError("Observed TSDF surface distance returned nonfinite evidence")
            witness_first = np.asarray(result.getNearestPoint1()).copy()
            witness_second = np.asarray(result.getNearestPoint2()).copy()
        else:
            value, witness_first, witness_second = token
        if miss:
            # Retain the existing guard against any unexpected request write.
            # Production requests are cold; native rejects both cached-guess
            # switches before dispatch instead of silently replacing history.
            if self._request.saveToString() == signature[0]:
                self._cache[cache_key] = signature, value, witness_first, witness_second
            else:
                self._cache.pop(cache_key, None)
        return value, witness_first, witness_second

    @property
    def scheduled_query_count(self):
        return len(self._first_shapes)

    @property
    def cache_hit_count(self):
        return self._cache_hit_count

    @property
    def native_worker_count(self):
        return 0 if self._batch is None or isinstance(self._batch, Exception) else self._batch.worker_count

    @property
    def storage_nbytes(self):
        return 0 if self._batch is None or isinstance(self._batch, Exception) else self._batch.storage_nbytes
# END signed components, permissions and cross-state hints stay with the caller.
