"""Bounded region preparation shared only within one synchronous inference.

Every candidate recomputes its normal eligibility. A sample can be reused only
for identical eligibility bytes on the same seed, graph and sampling contract.
"""
from copy import deepcopy
import numpy as np

CACHE_BUDGET_BYTES = 64 * 1024**2


class RegionScope:
    """One immutable surface/domain binding, destroyed when its infer returns.

    Keys include the seed, both sampling parameters and the entire frozen
    TransferConfig. Thus normals/positions/validity/entity/spacing/CSR are fixed
    under a key; mapped center is not an input to this private sampler at all.
    Surface/domain identity alone is not a persistent cache invalidation rule:
    callers may replace or mutate them after this one synchronous infer ends.
    """

    def __init__(self, sampler):
        self.sampler = sampler
        self.binding = None
        self.prepared = {}
        self.counts = dict(calls=0, preparation_hits=0, preparation_misses=0,
            sample_hits=0, sample_misses=0, reused_samples_with_changed_normal=0,
            unsampleable_sample_misses=0, retention_budget_skips=0,
            retained_bytes_accounted=0, peak_retained_bytes_accounted=0,
            max_local_member_count=0, retained_preparations=0, retained_samples=0,
            cache_budget_bytes=CACHE_BUDGET_BYTES, full_surface_masks_retained=0)

    def _retain(self, byte_count):
        if self.counts["retained_bytes_accounted"] + byte_count > CACHE_BUDGET_BYTES:
            self.counts["retention_budget_skips"] += 1
            return False
        self.counts["retained_bytes_accounted"] += byte_count
        self.counts["peak_retained_bytes_accounted"] = self.counts["retained_bytes_accounted"]
        return True

    def _prepare(self, surface, domain, seed, radius, spacing, config):
        """Repeat exactly the original enclosing-ball and sorted-CSR preparation.

        Neither mapped normal nor mapped center participates in these lines.
        The outward numeric guard and sorted global IDs are intentionally the
        original expressions: no radius, topology or floating rule is relaxed.
        """
        thickness = config.surface_height_spacing_factor * surface["spacing"][seed]
        tangent_limit = radius * (1 + 1e-12)
        outer = np.nextafter(np.hypot(tangent_limit, thickness)
                             * (1 + 8*np.finfo(float).eps), np.inf)
        columns = domain["tree"].query_ball_point(surface["xyz"][seed], outer, workers=1)
        members = np.sort(domain["surface_ids"][np.asarray(columns, dtype=np.int64)])
        local_seed = int(np.searchsorted(members, seed))
        if local_seed == len(members) or members[local_seed] != seed:
            raise RuntimeError("The enclosing region query lost its own seed")
        bounded = {key: surface[key][members]
                   for key in ("xyz", "normals", "spacing", "valid", "entity_member")}
        bounded["adjacency"] = surface["adjacency"][members][:, members]
        graph = bounded["adjacency"]
        # Array payloads plus conservative metadata allowance per preparation;
        # these are small bounded subsets, never complete-surface copies.
        byte_count = members.nbytes + sum(value.nbytes for key, value in bounded.items()
            if key != "adjacency") + graph.data.nbytes + graph.indices.nbytes + graph.indptr.nbytes + 4096
        retained = self._retain(byte_count)
        self.counts["max_local_member_count"] = max(self.counts["max_local_member_count"], len(members))
        return dict(members=members, local_seed=local_seed, bounded=bounded,
                    samples={}, retained=retained)

    @staticmethod
    def _compact(mode, members):
        """Own only compact global membership and the original small result."""
        compact = {key: deepcopy(value) for key, value in mode.items() if key != "mask"}
        compact["member_indices"] = members[mode["mask"]]
        compact["seed_index"] = int(members[mode["seed_index"]])
        compact["sample_indices"] = members[mode["sample_indices"]]
        return compact

    @staticmethod
    def _restore(compact, surface_count):
        """A fresh tree prevents later candidate diagnostics from leaking."""
        mode = deepcopy(compact)
        members = mode.pop("member_indices")
        mask = np.zeros(surface_count, bool)
        mask[members] = True
        mode["mask"] = mask
        return mode

    def sample(self, surface, domain, normal, seed, *, sampling_radius_m,
               sampling_spacing_m, config):
        if self.binding is None:
            self.binding = surface, domain
        elif surface is not self.binding[0] or domain is not self.binding[1]:
            raise ValueError("One region scope cannot mix infer surface/domain bindings")
        self.counts["calls"] += 1
        # TransferConfig is frozen and hashable. Include all its fields rather
        # than guessing which present/future sampling thresholds matter.
        key = (int(seed), float(sampling_radius_m), float(sampling_spacing_m), config)
        prepared = self.prepared.get(key)
        if prepared is None:
            self.counts["preparation_misses"] += 1
            prepared = self._prepare(surface, domain, seed, sampling_radius_m,
                                     sampling_spacing_m, config)
            if prepared["retained"]:
                self.prepared[key] = prepared
                self.counts["retained_preparations"] += 1
        else:
            self.counts["preparation_hits"] += 1
        bounded = prepared["bounded"]
        # This candidate-specific predicate MUST run even when the seed repeats:
        # removing a normal-ineligible bridge can change connected members/FPS.
        eligible = bounded["normals"] @ normal >= config.normal_min_agreement
        eligibility_bytes = eligible.tobytes()
        cached = prepared["samples"].get(eligibility_bytes)
        if cached is not None:
            self.counts["sample_hits"] += 1
            if cached["normal_bytes"] != np.asarray(normal).tobytes():
                self.counts["reused_samples_with_changed_normal"] += 1
            compact = cached["mode"]
        else:
            self.counts["sample_misses"] += 1
            # Original sampler remains intact: seed plane, tangent radius,
            # induced graph, connected component, FPS, rank and all diagnostics.
            mode = self.sampler(bounded, eligible, prepared["local_seed"],
                sampling_radius_m=sampling_radius_m, sampling_spacing_m=sampling_spacing_m,
                config=config)
            diagnostic = mode["diagnostics"]
            if (diagnostic["neighborhood_point_count"] < max(6, config.min_neighborhood_points)
                    or diagnostic["neighborhood_rank"] < 2):
                self.counts["unsampleable_sample_misses"] += 1
            compact = self._compact(mode, prepared["members"])
            normal_bytes = np.asarray(normal).tobytes()
            byte_count = sum(value.nbytes for value in compact.values() if isinstance(value, np.ndarray))
            byte_count += len(eligibility_bytes) + len(normal_bytes) + 4096
            if prepared["retained"] and self._retain(byte_count):
                prepared["samples"][eligibility_bytes] = dict(mode=compact, normal_bytes=normal_bytes)
                self.counts["retained_samples"] += 1
        return self._restore(compact, len(surface["xyz"]))

    def close(self):
        """No retained arrays/graphs survive to the next timed inference."""
        self.prepared.clear()
        self.binding = None
