"""Choose one observed distance field for a complete physical body.

Layer choice reads source support and a conservative body envelope, never a
collision value. The returned decision is also the input to material contact
and independent solid checking. Resolution is independent of measurement
allowance; neither a positive ESDF nor a finer voxel proves observed freedom.
"""
from copy import copy, deepcopy
from dataclasses import dataclass
import math

import numpy as np
import torch

# BEGIN compact archival records retain exact native selections.
from .collision_records import encode_array
# END codec storage does not alter any layer-selection arithmetic.
from .optimizer_device_checks import require_tensor

# BEGIN finite local refinement capacity, independently admitted by Mapper bytes.
# Four million float32 samples need 16 MB; the padded int32 prefix is counted
# from its actual shape. Mapping scratch has a separate one-GiB admission.
MAX_FINE_CELLS = 4_000_000
# END this is a local inventory bound, never permission to refine the workspace.


def _measured_spheres(spheres, margin):
    """Round r+measured allowance outward in the native query's actual dtype.

    A float32 addition may round the requested allowance down, or above a
    separately computed float64 selector envelope. Construct the same outward
    radius for both consumers. The detached rounding correction preserves the
    ordinary derivative with respect to the supplied physical sphere radius.
    """
    if margin == 0:
        return spheres
    radius = spheres[..., 3:4]
    target = radius.detach().to(torch.float64)+margin
    # BEGIN device-created scalars avoid repeated host transfers in queries.
    target = torch.nextafter(target, torch.full((), torch.inf, dtype=target.dtype, device=target.device))
    rounded = target.to(radius.dtype)
    rounded = torch.where(rounded.to(torch.float64) < target,
        torch.nextafter(rounded, torch.full((), torch.inf, dtype=rounded.dtype, device=rounded.device)), rounded)
    # END infinity is exact; radius arithmetic and its rounding are unchanged.
    return torch.cat((spheres[..., :3], rounded+(radius-radius.detach())), dim=-1)


def validate_body_groups(groups, sphere_count):
    """Bind every sphere exactly once to its complete native physical body.

    Native geometry names let an independent entity check consume the same
    decision. Materials cannot identify bodies: many arm solids have none.
    """
    result, indices, names, geometry_names = [], [], set(), set()
    for supplied in groups:
        row = deepcopy(supplied)
        identity = row["body_id"]
        selected = row["sphere_indices"]
        geometries = row["geometry_names"]
        if (not isinstance(identity, str) or not identity or identity in names
                or not selected or any(type(i) is not int for i in selected)
                or not geometries or any(not isinstance(n, str) or not n for n in geometries)
                or len(set(geometries)) != len(geometries)
                or geometry_names.intersection(geometries)):
            raise ValueError("Bodies need unique identities, complete sphere groups and native geometry names")
        names.add(identity)
        geometry_names.update(geometries)
        indices.extend(selected)
        result.append(row)
    if sorted(indices) != list(range(sphere_count)):
        raise ValueError("Body groups must partition the complete collision sphere model exactly once")
    return tuple(result)


@dataclass(frozen=True)
class LayerSelection:
    """Immutable reference to one selection, before native collision queries."""

    sphere_fine: torch.Tensor
    body_fine: torch.Tensor
    influence_aabb: torch.Tensor
    index_lower: torch.Tensor
    index_upper: torch.Tensor
    body_ids: tuple

    def record(self):
        """Archive the actual selection; consumers must not independently reselect."""
        # BEGIN bounded binary arrays avoid Python objects for every body/state.
        return {"body_ids": list(self.body_ids),
                "body_fine": encode_array(self.body_fine.detach().cpu()),
                "influence_aabb_grid_m": encode_array(self.influence_aabb.detach().cpu()),
                "index_lower": encode_array(self.index_lower.detach().cpu()),
                "index_upper": encode_array(self.index_upper.detach().cpu())}
        # END only final-validation callers request these immutable records.


class FineRegion:
    """Source-supported fine region with an exact bounded box-sum lookup.

    Each uint8 sample retains a nearest-site bit (1), observed TSDF sign (2),
    and negative TSDF sign (4). A raw whole-cell free ray may supply missing
    sign evidence, but a negative-sign/free-ray contradiction is inadmissible.
    All samples in the body's complete interpolation envelope must qualify.
    This is local source support, not a global nearest-surface certificate.
    """

    def __init__(self, record):
        self.record = deepcopy(record)
        supplied_shape = np.asarray(record["shape_xyz"])
        if supplied_shape.shape != (3,) or supplied_shape.dtype.kind not in "iu":
            raise ValueError("Fine field dimensions must be three explicit integers")
        shape = tuple(int(value) for value in supplied_shape)
        pose = np.asarray(record["pose"], dtype=float)
        size = float(record["voxel_size_m"])
        support = float(record["numerical_support_m"])
        margin = float(record["measurement_allowance_m"])
        if (min(shape) < 3 or math.prod(shape) > MAX_FINE_CELLS
                or pose.shape != (7,) or not np.isfinite(pose).all()
                or not np.array_equal(pose[3:], [1., 0., 0., 0.])
                or not np.isfinite([size, support, margin]).all()
                or size <= 0 or support < 2*size or margin < 0
                or not record["measurement_evidence"] or not record["source"]):
            raise ValueError("Fine region needs bounded geometry, numerical support and measured error evidence")
        bits = np.frombuffer(bytes.fromhex(record["source_bits_hex"]), dtype=np.uint8)
        packed = np.frombuffer(bytes.fromhex(record["raw_free_packbits_hex"]), dtype=np.uint8)
        if len(bits) != math.prod(shape) or len(packed) != math.ceil(len(bits)/8) or np.any(bits > 7):
            raise ValueError("Fine region must retain every source/sign and raw-ray validity sample")
        free = np.unpackbits(packed)[:len(bits)].astype(bool)
        if np.any(((bits & 4) != 0) & ((bits & 2) == 0)):
            raise ValueError("A negative sign cannot exist without an observed TSDF sample")
        valid = ((bits & 1) != 0) & (((bits & 2) != 0) | free) & ~(free & ((bits & 4) != 0))
        # The finite cell bound keeps every exact prefix count in int32.
        invalid = (~valid).reshape(tuple(shape)).astype(np.int32)
        prefix = np.pad(invalid, ((1, 0),)*3)
        for axis in range(3):
            prefix = prefix.cumsum(axis=axis, dtype=np.int32)
        self._prefix = torch.from_numpy(prefix)
        self._device_prefix = {}
        # BEGIN repeated layer queries share immutable device constants.
        self._device_constants = {}
        self._device_groups = {}
        # END no input state or layer decision is cached here.
        self.shape = shape
        # BEGIN own the immutable pose that supplies cached device constants.
        self.pose = pose.copy()
        self.pose.setflags(write=False)
        self.size, self.support, self.margin = size, support, margin
        # END caller-owned ndarray mutations cannot change this bound region.
        # BEGIN fixed, source-certified optimizer overlap regions.
        self.guidance_boxes = np.empty((0, 2, 3), dtype=np.float64)
        self._device_guidance_boxes = {}
        # END source acceptance continues to use select(), independently.

    @property
    def storage_bytes(self):
        # BEGIN include one device's int shape, FP64 shape/pose and infinities.
        return self._prefix.numel()*self._prefix.element_size() + (3+3+7+2)*8 + self.guidance_boxes.nbytes
        # END each planner binds this region to its admitted single GPU.

    # BEGIN immutable tensors retain their original dtype and exact conversion.
    def _constants(self, device):
        if device not in self._device_constants:
            self._device_constants[device] = (
                torch.tensor(self.shape, dtype=torch.int64, device=device),
                torch.tensor(self.shape, dtype=torch.float64, device=device),
                torch.tensor(self.pose, dtype=torch.float64, device=device),
                torch.tensor(-torch.inf, dtype=torch.float64, device=device),
                torch.tensor(torch.inf, dtype=torch.float64, device=device))
        return self._device_constants[device]
    # END each device reuses its original layer constants across queries.

    # BEGIN Python advanced-index lists otherwise create a host index per call.
    def _group_indices(self, device, groups):
        partition = tuple(tuple(group["sphere_indices"]) for group in groups)
        key = (device, partition)
        if key not in self._device_groups:
            self._device_groups[key] = tuple(torch.tensor(indices, device=device, dtype=torch.int64)
                                              for indices in partition)
        return self._device_groups[key]
    # END the scene's fixed partitions preserve all sphere and reduction orders.

    def _box_count(self, lower, upper):
        """Count unsupported inclusive sample boxes using eight prefix corners."""
        key = str(lower.device)
        if key not in self._device_prefix:
            self._device_prefix[key] = self._prefix.to(lower.device)
        prefix = self._device_prefix[key]
        shape = self._constants(lower.device)[0]
        lo = torch.maximum(torch.zeros_like(lower), torch.minimum(lower, shape))
        hi = torch.maximum(torch.zeros_like(upper), torch.minimum(upper+1, shape))
        count = torch.zeros(lower.shape[:-1], dtype=torch.int32, device=lower.device)
        for x in (0, 1):
            for y in (0, 1):
                for z in (0, 1):
                    corner = [hi[..., i] if side else lo[..., i] for i, side in enumerate((x, y, z))]
                    count = count + ((-1)**(3-x-y-z))*prefix[tuple(corner)]
        return count

    def _envelope_support(self, envelope):
        """Evaluate the one shared source-support predicate on saved bounds.

        Arithmetic on stored metric values is outward rounded. The usable
        grid is rounded inward, and interpolation index intervals outward.
        This predicate can verify a recorded choice without choosing again.
        """
        if envelope.shape[-2:] != (2, 3) or envelope.dtype != torch.float64:
            raise ValueError("Source-support envelopes require float64 [...,2,3] grid-local bounds")
        # BEGIN validate measured bounds before integer index construction.
        low, high = envelope[..., 0, :], envelope[..., 1, :]
        require_tensor(torch.isfinite(envelope).all() & ~(low > high).any(),
                       "Source-support envelopes must be finite ordered boxes")
        _, shape, _, negative, positive = self._constants(low.device)
        # END all valid FP64 bounds enter the identical outward-rounded formula.
        # Integer shape and half-integer offsets are represented exactly.
        lo_coordinate = torch.nextafter(torch.nextafter(low/self.size, negative)+(shape-1)/2, negative)
        hi_coordinate = torch.nextafter(torch.nextafter(high/self.size, positive)+(shape-1)/2, positive)
        lo = torch.floor(lo_coordinate).to(torch.int64)-1
        hi = torch.ceil(hi_coordinate).to(torch.int64)+1
        half = torch.nextafter(shape*self.size/2, negative)
        allowed_low = torch.nextafter(-half+self.support, positive)
        allowed_high = torch.nextafter(half-self.support, negative)
        bounds = ((low > allowed_low) & (high < allowed_high)).all(-1)
        inside = ((lo >= 0) & (hi < shape)).all(-1)
        return bounds & inside & (self._box_count(lo, hi) == 0), lo, hi

    def validate_selection_envelope(self, influence_aabb_grid_m, index_lower, index_upper):
        """Verify saved support evidence; never replace its coarse/fine choice."""
        supported, lo, hi = self._envelope_support(influence_aabb_grid_m)
        if (index_lower.shape != lo.shape or index_upper.shape != hi.shape
                or index_lower.dtype != torch.int64 or index_upper.dtype != torch.int64):
            raise ValueError("Saved stencil indices must be aligned int64 [...,3] arrays")
        return {"indices_match": ((index_lower == lo) & (index_upper == hi)).all(-1),
                "fine_supported": supported}

    def select(self, spheres, groups):
        """Select by the whole sphere union, already certified to cover each solid.

        Inputs are actual post-gripper spheres in the target owner's frame.
        Each body's AABB includes every complete sphere, plus the measured
        allowance. One extra sample on each side encloses all interpolation
        stencils. Strict support boundaries prevent ROI-edge substitution.
        """
        if spheres.ndim != 4 or spheres.shape[-1] != 4:
            raise ValueError("Layer selection requires [B,H,S,4] complete body spheres")
        groups = validate_body_groups(groups, spheres.shape[-2])
        # BEGIN validate the complete supplied cover before selecting a layer.
        require_tensor(torch.isfinite(spheres).all() & ~(spheres[..., 3] < 0).any(),
                       "Layer selection cannot omit invalid or disabled body-cover spheres")
        # END every selected sphere belongs to the original valid input.
        # Selection is a discrete, shared model choice; collision gradients
        # continue through the original spheres in the selected native field.
        # The Mapper produces axis-aligned grids in each owner's local frame.
        # Use the SAME quantized native sphere inputs, promoted to float64;
        # independently recomputed double FK would be a different query.
        # Outward rounding at every +/- operation encloses exact arithmetic
        # on those stored sphere values, including strict ROI-edge cases.
        detached = _measured_spheres(spheres.detach(), self.margin).to(dtype=torch.float64)
        _, _, pose, toward_low, toward_high = self._constants(detached.device)
        local = detached[..., :3]-pose[:3]
        local_low = torch.nextafter(local, toward_low)
        local_high = torch.nextafter(local, toward_high)
        lower, upper = [], []
        indices = self._group_indices(detached.device, groups)
        for idx in indices:
            radius = detached[..., idx, 3:4]
            low = torch.nextafter(local_low[..., idx, :]-radius, toward_low).amin(-2)
            high = torch.nextafter(local_high[..., idx, :]+radius, toward_high).amax(-2)
            lower.append(low)
            upper.append(high)
        low, high = torch.stack(lower, -2), torch.stack(upper, -2)
        envelope = torch.stack((low, high), -2)
        chosen, lo, hi = self._envelope_support(envelope)
        per_sphere = torch.empty(spheres.shape[:-1], dtype=torch.bool, device=spheres.device)
        for i, idx in enumerate(indices):
            per_sphere[..., idx] = chosen[..., i, None]
        return LayerSelection(per_sphere, chosen, envelope, lo, hi,
                              tuple(group["body_id"] for group in groups))

    # BEGIN smooth guidance uses boxes proved entirely supported before solving.
    def prepare_guidance_boxes(self, envelopes_world, *, maximum_boxes=16):
        """Grow bounded supported witness boxes using exact prefix counts.

        No distance value establishes validity. A missing sign/site anywhere
        in a witness's complete interpolation box prevents its admission.
        Greedy face expansion is deterministic and uses only the frozen map.
        The resulting metric boxes include guards for interpolation stencils
        and the numerical support band. They remain fixed for the whole solve.
        """
        envelopes = np.asarray(envelopes_world, dtype=np.float64)
        if (envelopes.ndim != 3 or envelopes.shape[1:] != (2, 3)
                or len(envelopes) > 32768+16 or not np.isfinite(envelopes).all()
                or np.any(envelopes[:, 0] > envelopes[:, 1])
                or type(maximum_boxes) is not int or not 0 < maximum_boxes <= 16):
            raise ValueError("Guidance witnesses need bounded finite complete-body envelopes")
        local = envelopes-self.pose[:3]
        selected, lower, upper = self._envelope_support(torch.from_numpy(local))
        prefix = self._prefix.numpy()
        shape = np.asarray(self.shape)
        def supported(lo, hi):
            count = 0
            for x in (0, 1):
                for y in (0, 1):
                    for z in (0, 1):
                        corner = tuple(int(hi[i]+1 if side else lo[i])
                                       for i, side in enumerate((x, y, z)))
                        count += (-1)**(3-x-y-z)*int(prefix[corner])
            return count == 0
        boxes = []
        # BEGIN capacity counts distinct final cuboids, never repeated witnesses.
        box_keys = set()
        duplicate_boxes = 0
        # END different witnesses still run the same supported face expansion.
        origin = -(shape-1)*self.size/2
        # Native query coordinates are float32. This inward numerical guard
        # makes positive double-precision clearance imply the strict selector
        # predicate even where rounded grid faces coincide with sample centres.
        guard = 32*np.finfo(np.float32).eps*(1.+float(np.abs(self.pose[:3]).max())
                                              +float((shape*self.size).max()))
        allowed = np.stack((-shape*self.size/2+self.support+guard,
                            shape*self.size/2-self.support-guard))
        for index in np.flatnonzero(selected.numpy()):
            # BEGIN an already-covered witness can still grow a different box.
            # Its narrower starting stencil can admit a different first face
            # expansion. Only equal final coordinates establish redundancy.
            # END no containment shortcut discards another supported cuboid.
            if len(boxes) == maximum_boxes:
                break
            lo, hi = lower[index].numpy().copy(), upper[index].numpy().copy()
            for axis in range(3):
                left, right = 0, int(lo[axis])
                while left < right:
                    middle = (left+right)//2
                    trial = lo.copy(); trial[axis] = middle
                    if supported(trial, hi): right = middle
                    else: left = middle+1
                lo[axis] = left
                left, right = int(hi[axis]), int(shape[axis]-1)
                while left < right:
                    middle = (left+right+1)//2
                    trial = hi.copy(); trial[axis] = middle
                    if supported(lo, trial): left = middle
                    else: right = middle-1
                hi[axis] = left
            # BEGIN invert the selector's one-sample interpolation padding once.
            # _envelope_support uses floor(low_coordinate)-1 and
            # ceil(high_coordinate)+1. Thus a metric envelope strictly above
            # sample lo+1 and below sample hi-1 stays within these supported
            # indices. The existing inward guard preserves those strict
            # inequalities after native float32 sphere arithmetic; an extra
            # sample of erosion would discard already-proved witness space.
            box = np.stack((np.maximum(origin+(lo+1)*self.size+guard, allowed[0]),
                            np.minimum(origin+(hi-1)*self.size-guard, allowed[1])))
            # END source validity and the numerical support band are unchanged.
            if np.all(box[0] < box[1]):
                # BEGIN deduplicate only identical results after full expansion.
                # A boundary witness may not fit its own inward-guarded box.
                # Repeating that result must not consume another finite slot;
                # every different final cuboid remains eligible for admission.
                key = tuple(box.ravel())
                if key in box_keys:
                    duplicate_boxes += 1
                    continue
                box_keys.add(key)
                boxes.append(box)
                # END no witness coordinates are rounded or epsilon-merged.
        self.guidance_boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 2, 3)
        self.guidance_boxes.setflags(write=False)
        self._device_guidance_boxes = {}
        covered = np.zeros(len(local), dtype=bool)
        for box in boxes:
            covered |= np.all(local[:, 0] > box[0], axis=-1) & np.all(local[:, 1] < box[1], axis=-1)
        return {"witnesses": len(envelopes), "source_supported_witnesses": int(selected.sum()),
                "unsupported_witnesses": int((~selected).sum()), "cuboids": len(boxes),
                "duplicate_cuboid_candidates": duplicate_boxes,
                "supported_uncovered_envelopes": int((selected.numpy() & ~covered).sum()),
                "cuboid_capacity_reached": len(boxes) == maximum_boxes,
                "overlap_m": 4*self.size,
                "cuboids_grid_m": self.guidance_boxes.tolist()}

    def guidance_weights(self, spheres, groups):
        """Return one differentiable weight per complete body and sphere.

        Smoothstep has zero value and derivative at the source-safe boundary.
        Its derivative is retained: dropping it would give the gradient of a
        different objective in the overlap band. Maximum over fixed boxes is
        continuous and piecewise differentiable, without stateful hysteresis.
        """
        groups = validate_body_groups(groups, spheres.shape[-2])
        expanded = _measured_spheres(spheres, self.margin).to(torch.float64)
        device = spheres.device
        if device not in self._device_guidance_boxes:
            self._device_guidance_boxes[device] = torch.tensor(
                self.guidance_boxes.copy(), device=device, dtype=torch.float64)
        boxes = self._device_guidance_boxes[device]
        pose = self._constants(device)[2]
        centers = expanded[..., :3]-pose[:3]
        weights = []
        indices = self._group_indices(device, groups)
        for idx in indices:
            low = (centers[..., idx, :]-expanded[..., idx, 3:4]).amin(-2)
            high = (centers[..., idx, :]+expanded[..., idx, 3:4]).amax(-2)
            if len(self.guidance_boxes):
                clearance = torch.minimum(low[..., None, :]-boxes[:, 0],
                                           boxes[:, 1]-high[..., None, :]).amin(-1)
                phase = (clearance/(4*self.size)).clamp(0., 1.)
                weight = (phase.square()*(3.-2.*phase)).amax(-1)
            else:
                weight = low.sum(-1)*0.
            weights.append(weight.to(spheres.dtype))
        body_weights = torch.stack(weights, -1)
        sphere_weights = torch.empty(spheres.shape[:-1], dtype=spheres.dtype, device=device)
        for index, idx in enumerate(indices):
            sphere_weights[..., idx] = body_weights[..., index, None]
        return sphere_weights, body_weights
    # END fixed supported boxes affect only differentiable collision guidance.


class LayeredVoxelQuery:
    """Two native fields, one externally frozen complete-body choice per state."""

    def __init__(self, coarse, fine, region):
        if coarse.owner_ids != fine.owner_ids:
            raise ValueError("Both resolutions must represent precisely the same owner partition")
        from .contact_geometry import query_geometry
        self.coarse, self.fine, self.region = coarse, fine, FineRegion(region)
        self.owner_ids = coarse.owner_ids
        self.source = coarse.source+"; source-supported local fine region"
        self.independent_geometry = {"layers": {"coarse": query_geometry(coarse), "fine": query_geometry(fine)},
                                     "fine_region": deepcopy(region)}
        self.refinement_source = None
        self.mapping_report = None

    def clone_for_rollout(self):
        result = copy(self)
        result.coarse, result.fine = self.coarse.clone_for_rollout(), self.fine.clone_for_rollout()
        return result

    def inflated_spheres(self, spheres):
        """Apply the separately measured sensor allowance to collision radii."""
        return _measured_spheres(spheres, self.region.margin)

    def __call__(self, spheres, *, key, selection):
        if selection.sphere_fine.shape != spheres.shape[:-1]:
            raise ValueError("Collision must consume the aligned complete-body layer decision")
        # Both fields retain their own bounded native buffers. where selects
        # one value and one gradient; no coarse/fine maximum or sum is formed.
        expanded = self.inflated_spheres(spheres)
        coarse = self.coarse(expanded, key=key)
        fine = self.fine(expanded, key=key)
        return torch.where(selection.sphere_fine, fine, coarse)

    # BEGIN background optimizer blending never changes source layer evidence.
    def background_guidance(self, spheres, *, key, groups):
        expanded = self.inflated_spheres(spheres)
        coarse = self.coarse(expanded, key=key)
        fine = self.fine(expanded, key=key)
        weights, _ = self.region.guidance_weights(spheres, groups)
        return coarse+weights*(fine-coarse)
    # END both native values and the overlap-weight derivative remain in graph.
