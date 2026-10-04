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
    # BEGIN device-created scalars avoid repeated host transfers in capture.
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
        if (min(shape) < 3 or math.prod(shape) > 1_000_000
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
        # At most one million cells keeps every exact prefix count in int32.
        invalid = (~valid).reshape(tuple(shape)).astype(np.int32)
        prefix = np.pad(invalid, ((1, 0),)*3)
        for axis in range(3):
            prefix = prefix.cumsum(axis=axis, dtype=np.int32)
        self._prefix = torch.from_numpy(prefix)
        self._device_prefix = {}
        # BEGIN stable immutable layer constants shared with captured queries.
        self._device_constants = {}
        self._device_groups = {}
        # END no input state or layer decision is cached here.
        self.shape = shape
        # BEGIN own the immutable pose that supplies cached device constants.
        self.pose = pose.copy()
        self.pose.setflags(write=False)
        self.size, self.support, self.margin = size, support, margin
        # END caller-owned ndarray mutations cannot change this bound region.

    @property
    def storage_bytes(self):
        # BEGIN include one device's int shape, FP64 shape/pose and infinities.
        return self._prefix.numel()*self._prefix.element_size() + (3+3+7+2)*8
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
    # END warmup creates constants before graph replay can use their addresses.

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
        # BEGIN keep unsafe integer indexing out of a deferred-failure block.
        low, high = envelope[..., 0, :], envelope[..., 1, :]
        valid = require_tensor(torch.isfinite(envelope).all() & ~(low > high).any(),
                               "Source-support envelopes must be finite ordered boxes")
        if valid is not None:
            envelope = torch.where(valid, envelope, torch.zeros_like(envelope))
            low, high = envelope[..., 0, :], envelope[..., 1, :]
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
        # BEGIN record dynamic faults on-device inside an optimizer graph.
        valid = require_tensor(torch.isfinite(spheres).all() & ~(spheres[..., 3] < 0).any(),
                               "Layer selection cannot omit invalid or disabled body-cover spheres")
        if valid is not None:
            spheres = torch.where(valid, spheres, torch.zeros_like(spheres))
        # END a failed block cannot export this placeholder selection.
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
