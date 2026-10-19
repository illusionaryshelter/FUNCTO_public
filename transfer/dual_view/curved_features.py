"""MARCO responses at actual curved-surface measurements, without plane gates.

Only geometry qualification differs from the planar response field. Native
descriptor coordinates, positive bilinear weights, fixed query identities and
raw dot products are shared. An unknown corner is never dropped/renormalized.
"""

import numpy as np

from .curved_geometry import bounded_members, project_measurements
from .features import EmpiricalSupport, ResponseField, _sampling_corners


class SurfaceResponseField(ResponseField):
    """Response maps qualified by measured points and one-camera graph paths.

    A candidate first associates to a deterministic real depth measurement.
    Its appearance is sampled at that measurement's original pixel, and the
    geometric residual is returned separately. This discrete association must
    not be described as evaluating a continuous reconstructed depth surface.
    """

    def __init__(self, surface, queries, views):
        if surface.get('region_geometry') != 'surface':
            raise ValueError('SurfaceResponseField requires explicit curved geometry')
        super().__init__(surface, queries, views)
        # This cache belongs to this immutable observation/feature field only.
        # It stores observed paths, never scores or candidate-dependent gates.
        self._footprint_cache = {}
        self._footprint_cache_bytes = 0
        self._footprint_cache_limit_bytes = 64*1024**2

    def _footprint(self, view_index, references, corners, positive):
        """Require every positive feature corner in the same observed patch."""
        lookup = self.surface['pixel_to_observation'][view_index]
        result = np.zeros(positive.shape, bool)
        for row, reference in enumerate(references):
            key = (view_index, int(reference))
            members = self._footprint_cache.get(key)
            if members is None:
                members, _ = bounded_members(self.surface, int(reference),
                    self.surface['sampling_radius_m'], view_index=view_index)
                # A full cache changes only reuse, never the geometry result.
                # Include a conservative per-entry allowance for Python keys.
                cost = members.nbytes+256
                if self._footprint_cache_bytes+cost <= self._footprint_cache_limit_bytes:
                    self._footprint_cache[key] = members
                    self._footprint_cache_bytes += cost
            selected = np.flatnonzero(positive[row])
            xy = corners[row, selected]
            inside = (np.all(xy >= 0, axis=1) & (xy[:, 0] < lookup.shape[1])
                      & (xy[:, 1] < lookup.shape[0]))
            columns, xy = selected[inside], xy[inside]
            ids = lookup[xy[:, 1], xy[:, 0]]
            result[row, columns] = np.isin(ids, members, assume_unique=False)
        return result

    # BEGIN candidates and frozen appearance references share native sampling.
    def _sample_observations(self, view_index, observation_ids, rows=None):
        """Sample actual measurements in one camera with complete RGB support.

        Callers supply only actual observation identities belonging to this
        camera. Candidate geometry has already qualified those identities;
        references use the camera's entire observed entity domain directly.
        Both paths therefore use identical original pixels, resize metadata,
        unrenormalized bilinear weights and positive-corner graph predicates.
        """
        from local_models.marco import RESOLUTION

        view = self.views[view_index]
        shape = (len(self.queries), len(observation_ids)) if rows is None else (len(observation_ids),)
        values = np.full(shape, np.nan)
        valid = np.ones(len(observation_ids), bool)
        state = np.full(len(observation_ids), 'appearance_pending', object)
        if view['missing_reason'] is not None:
            state[:] = view['missing_reason']
            valid[:] = False
            return values, valid, state

        response, crop = view['response'], view['crop_xyxy']
        pixel = self.surface['observation_pixels'][observation_ids]
        local = pixel-crop[:2]
        inside = np.all(local >= 0, axis=1) & np.all(local <= crop[2:]-crop[:2]-1, axis=1)
        state[~inside] = 'outside_feature_crop'
        valid &= inside
        grid = local*view['resize_scale']*(np.array(response.shape[:0:-1])-1)/(RESOLUTION-1)
        inside, corners, weights = _sampling_corners(grid, response.shape[-2:])
        state[valid & ~inside] = 'outside_feature_grid'
        valid &= inside
        active = np.flatnonzero(valid)
        if len(active):
            positive = weights[active] > 0
            pixel_corners = np.rint(corners[active]*(RESOLUTION-1)
                /(np.array(response.shape[:0:-1])-1)/view['resize_scale']+crop[:2]).astype(np.int64)
            corner_inside = np.all(pixel_corners >= crop[:2], axis=-1)
            corner_inside &= np.all(pixel_corners < crop[2:], axis=-1)
            connected = self._footprint(view_index, observation_ids[active], pixel_corners, positive)
            supported = connected & corner_inside
            complete = np.all(~positive | supported, axis=1)
            any_support = np.any(positive & supported, axis=1)
            state[active[~complete & any_support]] = 'partial_feature_support'
            state[active[~complete & ~any_support]] = 'no_feature_support'
            valid[active[~complete]] = False
            chosen = active[complete]
            if len(chosen):
                # Rewrite the same bilinear sum as base + sum(w*(value-base)).
                # A strictly positive, already supported corner supplies base;
                # casting it before subtraction keeps arithmetic in float64.
                # This preserves nonzero constants exactly, so empirical ranks
                # cannot turn summation roundoff into appearance support. No
                # corner is dropped and the original weights are unchanged.
                first = np.argmax(weights[chosen] > 0, axis=1)
                base_xy = corners[chosen, first]
                if rows is None:
                    base = response[:, base_xy[:, 1], base_xy[:, 0]].astype(float)
                else:
                    base = response[rows[chosen], base_xy[:, 1], base_xy[:, 0]].astype(float)
                sampled = base.copy()
                for corner in range(4):
                    weight = weights[chosen, corner]
                    positive_here = weight > 0
                    ids = chosen[positive_here]
                    xy = corners[ids, corner]
                    if rows is None:
                        delta = response[:, xy[:, 1], xy[:, 0]]-base[:, positive_here]
                        sampled[:, positive_here] += delta*weight[positive_here]
                    else:
                        delta = response[rows[ids], xy[:, 1], xy[:, 0]]-base[positive_here]
                        sampled[positive_here] += delta*weight[positive_here]
                if rows is None:
                    values[:, chosen] = sampled
                else:
                    values[chosen] = sampled
                state[chosen] = 'appearance_valid_orientation_unknown'
        return values, valid, state

    def prepare_support(self):
        """Freeze separate empirical appearance references for both cameras.

        Each camera contributes every actual entity observation once. Only
        measurements with complete native feature footprints enter its sorted
        reference distributions. A missing camera stays empty; responses from
        the other camera are never borrowed. References are built before the
        candidate loop, so their domain cannot depend on a candidate's score.
        The fixed block size limits temporary interpolation arrays without
        subsampling or changing the reference domain.
        """
        supports, diagnostics = [], []
        for view_index, view_id in enumerate(self.surface['view_ids']):
            ids = np.flatnonzero(self.surface['observation_entity_member']
                & (self.surface['observation_view_index'] == view_index))
            reference = np.full((len(self.queries), len(ids)), np.nan)
            known = np.zeros(len(ids), bool)
            counts = {}
            for start in range(0, len(ids), 256):
                stop = min(start+256, len(ids))
                values, valid, state = self._sample_observations(view_index, ids[start:stop])
                reference[:, start:stop] = values
                known[start:stop] = valid
                names, number = np.unique(state, return_counts=True)
                for name, count in zip(names.tolist(), number.tolist()):
                    counts[name] = counts.get(name, 0)+count
            support = EmpiricalSupport(reference, np.broadcast_to(known, reference.shape))
            supports.append(support)
            diagnostics.append(dict(view_id=view_id, observation_count=len(ids),
                reason_counts=counts, **support.diagnostics))
            # EmpiricalSupport owns sorted rows. Release the unsorted Q-by-N
            # array before allocating the next camera's complete reference.
            del reference
        return tuple(supports), diagnostics
    # END one sampling convention governs candidate and reference appearance.

    def sample(self, points, normals, query_rows=None, *, registration_error_m=0.,
               query_position_error_m=0.):
        """Return fixed-Q scores and independent per-view position evidence.

        The normal argument preserves the shared point-field call shape only;
        no direction is inferred from it or used to accept curved positions.
        Registration error is zero for target reference-domain measurements
        and is explicitly supplied when evaluating a fitted transform.
        """
        points, normals = np.asarray(points, float), np.asarray(normals, float)
        if points.ndim != 2 or points.shape[1:] != (3,) or normals.shape != points.shape:
            raise ValueError('World positions and optional normal records must align [N,3]')
        rows = None if query_rows is None else np.asarray(query_rows)
        if rows is not None and (rows.shape != (len(points),) or rows.dtype.kind not in 'iu'
                                or np.any(rows < 0) or np.any(rows >= len(self.queries))):
            raise ValueError('query_rows must name one frozen source descriptor per point')
        shape = (len(self.queries), len(points)) if rows is None else (len(points),)
        values = np.full((*shape, 2), np.nan)
        valid_views = np.zeros((len(points), 2), bool)
        states, projections, counts = [], [], []
        for view_index in range(len(self.views)):
            projection = project_measurements(self.surface, points, view_index,
                registration_error_m=registration_error_m,
                query_position_error_m=query_position_error_m)
            projections.append(projection)
            state = np.asarray(projection['state_names'], object)[projection['states']].copy()
            valid = projection['supported'].copy()
            # BEGIN geometry chooses measurements; the shared helper samples RGB.
            active = np.flatnonzero(valid)
            if len(active):
                selected_rows = None if rows is None else rows[active]
                sampled, observed, appearance_state = self._sample_observations(
                    view_index, projection['observation_indices'][active], selected_rows)
                if rows is None:
                    values[:, active, view_index] = sampled
                else:
                    values[active, view_index] = sampled
                valid[active] = observed
                state[active] = appearance_state
            # END unsupported geometry retains its original diagnostic state.
            valid_views[:, view_index] = valid
            unique, number = np.unique(state, return_counts=True)
            counts.append(dict(zip(unique.tolist(), number.tolist())))
            states.append(state)
        known = np.broadcast_to(valid_views.any(axis=1), shape).copy()
        scores = np.max(np.where(np.isfinite(values), values, -np.inf), axis=-1)
        return dict(scores=np.where(known, scores, np.nan), known=known,
            view_scores=values, view_valid=valid_views, view_state=np.stack(states, axis=1),
            view_orientation_known=np.zeros_like(valid_views), geometry=projections,
            reason_counts=dict(zip(self.surface['view_ids'], counts)))
