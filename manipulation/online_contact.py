"""Frozen forbidden-region fields and a bounded native planar contact model.

Preparation may visit source grid sites once. Online work has dimensions only
for states, registered spheres, rule combinations and a few material vertices
or surface witnesses; there is no occupied-cell dimension in an optimizer's
forward or backward graph. This is an explicitly different model from exact
sphere/material-cell Boolean subtraction used by the independent audit.
"""

# BEGIN finite registered contact fields and planar material constraints.
from dataclasses import asdict
from itertools import combinations
import math

import numpy as np
from scipy.ndimage import distance_transform_edt
import torch
from torch.nn import functional as F

from .collision_records import MAX_ARRAY_BYTES, decode_array, encode_array
from .planning_scene import _tensor
from .resource_admission import DEFAULT_MAX_HOST_RSS_BYTES, require_host_memory


MODEL = "frozen_planar_contact_v2"
_MAX_RULES = 16
_MAX_FIELDS = 16
_MAX_STORED_BYTES = 8 * MAX_ARRAY_BYTES
_WITNESS_BINS = 4
_PREPARE_CHUNK = 65_536


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def require_planar_material(rule):
    """Admit an explicit source-bound full feature extrusion, never a guess."""
    material = rule.online_material
    if (not isinstance(material, dict) or material.get("kind") != "native_planar_extrusion_v1"
            or rule.feature_kind != "polygon" or len(rule.feature_points) > 16
            or len(set(rule.sphere_indices)) != len(rule.sphere_indices)
            or tuple(material.get("sphere_indices", ())) != tuple(rule.sphere_indices)):
        raise ValueError("UNSUPPORTED_MODEL: online finite contact requires a registered complete native planar extrusion")
    if (not np.isfinite(material["thickness_m"]) or material["thickness_m"] <= 0
            or rule.modeling_allowance_m >= material["thickness_m"] / 2):
        raise ValueError("UNSUPPORTED_MODEL: online material band must be smaller than half its native extrusion thickness")


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.flags.writeable = False
    return result


def _field_layout(rules):
    """Derive exact sphere membership and every realizable permission subset."""
    memberships = {}
    for rule in rules:
        for index in rule.sphere_indices:
            memberships.setdefault((rule.actor, index), []).append(rule.rule_id)
    grouped = {}
    for (actor, index), names in memberships.items():
        grouped.setdefault((actor, tuple(sorted(names))), []).append(index)
    needed = set()
    for _, names in grouped:
        if len(names) > 4:
            raise ValueError("UNSUPPORTED_MODEL: one material sphere exceeds the bounded simultaneous-rule field bank")
        needed.update(subset for count in range(1, len(names) + 1) for subset in combinations(names, count))
    if len(needed) > _MAX_FIELDS:
        raise ValueError("UNSUPPORTED_MODEL: finite contact field combinations exceed the fixed storage budget")
    return grouped, sorted(needed)


class PreparedContactFields:
    """Request-owned frozen fields, bounded witnesses and immutable binding.

    A nonempty forbidden field stores distance to forbidden site centres minus
    one cell half-diagonal and one trilinear interpolation half-diagonal. The
    latter follows the 1-Lipschitz distance bound and the maximum weighted
    corner offset. Thus sampled values conservatively bound distance to the
    retained cell union. They are not exact ESDF distances. Empty and unchanged
    sets have explicit branches; neither relies on EDT's outside-array rules.
    """

    def __init__(self, *, origin, voxel_size, shape, rules, maximum_radii,
                 regions, channels, groups, source):
        self.origin = _readonly(origin, float)
        self.voxel_size = float(voxel_size)
        self.shape = tuple(int(v) for v in shape)
        self.upper = self.origin + (np.asarray(self.shape) - 1) * self.voxel_size
        self.rule_records = tuple(_plain(asdict(rule)) for rule in rules)
        self.maximum_radii = {name: float(value) for name, value in maximum_radii.items()}
        self.regions, self.channels, self.groups = regions, tuple(channels), tuple(_plain(groups))
        self.source = str(source)
        self._device = {}
        self.storage_nbytes = sum(row["distance"].nbytes for row in self.channels if row["kind"] == "field")

    @classmethod
    def prepare(cls, *, distance, site_valid, observed_sign, origin, voxel_size,
                rules, maximum_radii, source, occupied=None):
        """Build full-owner forbidden fields once, before any rollout graph."""
        rules = tuple(rules)
        if not 0 < len(rules) <= _MAX_RULES or len({r.rule_id for r in rules}) != len(rules):
            raise ValueError("UNSUPPORTED_MODEL: online contact requires one to sixteen distinct rules per owner")
        for rule in rules:
            require_planar_material(rule)
        if set(maximum_radii) != {r.rule_id for r in rules} or any(
                not np.isfinite(v) or v <= 0 for v in maximum_radii.values()):
            raise ValueError("Online contact must bind every rule's actual maximum sphere radius")
        values = np.asarray(distance)
        shape = tuple(values.shape)
        h, origin = float(voxel_size), np.asarray(origin, float)
        if (len(shape) != 3 or min(shape) < 2 or values.nbytes > MAX_ARRAY_BYTES
                or origin.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(h) or h <= 0):
            raise ValueError("UNSUPPORTED_MODEL: finite contact grid exceeds the bounded source-field layout")
        # Group spheres by their exact actor/rule membership. All simultaneous
        # subsets are represented, including the true union of multiple patches.
        grouped, combinations_needed = _field_layout(rules)
        if len(combinations_needed) > _MAX_FIELDS or len(combinations_needed) * values.size * 4 > _MAX_STORED_BYTES:
            raise ValueError("UNSUPPORTED_MODEL: finite contact field combinations exceed the fixed storage budget")
        # Includes site indices/coordinates, masks, EDT internal index/distance
        # arrays, retained channels and one outward conversion buffer.
        require_host_memory(int(values.size * (96 + len(rules) + 4 * len(combinations_needed))),
                            pending_existing_bytes=0, max_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES)
        valid, sign = np.asarray(site_valid), np.asarray(observed_sign)
        if valid.shape != shape or sign.shape != shape or valid.dtype != bool or sign.dtype != bool:
            raise ValueError("Contact preparation requires the same source site/sign validity masks")
        occupied = (valid & ((np.abs(values) <= h * .51) | ((values < 0) & sign))
                    if occupied is None else np.asarray(occupied, dtype=bool))
        if occupied.shape != shape:
            raise ValueError("Registered occupied material must share its source grid")
        indices = np.argwhere(occupied)
        centres = origin + indices * h
        surface = np.abs(values[tuple(indices.T)]) <= h * .51 if len(indices) else np.empty(0, bool)
        upper = origin + (np.asarray(shape) - 1) * h
        masks, regions = {}, {}
        for rule in rules:
            tolerance = rule.numerical_position_tolerance_m
            axes = np.asarray(rule.patch_axes)
            widths = np.r_[rule.patch_half_size, rule.modeling_allowance_m]
            # A permission outside the field is not silently clipped. A whole
            # cell must fit both the finite permission and the source layout.
            corners = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
            patch_corners = rule.patch_origin + (corners * widths) @ axes.T
            if np.any(patch_corners.min(0) < origin) or np.any(patch_corners.max(0) > upper):
                raise ValueError(f"UNSUPPORTED_MODEL: contact patch {rule.rule_id} extends beyond its frozen owner field")
            half_support = np.abs(axes).sum(axis=0) * h / 2
            selected = np.zeros(len(indices), bool)
            for start in range(0, len(indices), _PREPARE_CHUNK):
                local = (centres[start:start + _PREPARE_CHUNK] - rule.patch_origin) @ axes
                selected[start:start + len(local)] = (np.abs(local) + half_support <= widths + tolerance).all(-1)
            mask = np.zeros(shape, bool)
            chosen = indices[selected]
            if len(chosen):
                mask[tuple(chosen.T)] = True
                patch_points = (centres[selected] - rule.patch_origin) @ axes
                lower = patch_points.min(0) - half_support
                high = patch_points.max(0) + half_support
            else:
                lower, high = np.zeros(3), np.zeros(3)
            # Witnesses remain actual selected source sites; choosing at most
            # one best zero-site per fixed patch bin is bounded and conservative.
            witnessed = np.flatnonzero(selected & surface)
            witnesses = []
            if len(witnessed):
                xy = (centres[witnessed] - rule.patch_origin) @ axes[:, :2]
                bins = np.clip(((xy / rule.patch_half_size + 1) * (_WITNESS_BINS / 2)).astype(int),
                               0, _WITNESS_BINS - 1)
                identities = bins[:, 0] * _WITNESS_BINS + bins[:, 1]
                observed_distance = np.abs(values[tuple(indices[witnessed].T)])
                for identity in range(_WITNESS_BINS ** 2):
                    candidates = np.flatnonzero(identities == identity)
                    if len(candidates):
                        selected_index = candidates[np.argmin(observed_distance[candidates])]
                        witnesses.append(centres[witnessed[selected_index]])
            if rule.required and not witnesses:
                raise ValueError(f"UNSUPPORTED_MODEL: required contact {rule.rule_id} has no observed surface witness in its finite patch")
            masks[rule.rule_id] = mask
            regions[rule.rule_id] = {"lower_patch": _readonly(lower), "upper_patch": _readonly(high),
                "has_removed_material": bool(len(chosen)), "removed_cells": int(len(chosen)),
                "surface_witnesses": _readonly(np.asarray(witnesses).reshape(-1, 3))}
        channels, lookup = [], {}
        margin = math.sqrt(3.) * h  # cell half-diagonal + interpolation half-diagonal
        for names in combinations_needed:
            removed = np.zeros(shape, bool)
            for name in names:
                removed |= masks[name]
            if not removed.any():
                channel = {"rule_ids": names, "kind": "unchanged"}
            else:
                remaining = occupied & ~removed
                if not remaining.any():
                    channel = {"rule_ids": names, "kind": "empty"}
                else:
                    transformed = distance_transform_edt(~remaining, sampling=h) - margin
                    # Float32 conversion must not make the conservative field
                    # larger. Interpolation has its own separately included margin.
                    field = np.nextafter(transformed.astype(np.float32), np.float32(-np.inf))
                    field.flags.writeable = False
                    channel = {"rule_ids": names, "kind": "field", "distance": field}
            lookup[names] = len(channels)
            channels.append(channel)
        groups = []
        for (actor, names), sphere_indices in sorted(grouped.items()):
            table = [-1]
            for code in range(1, 2 ** len(names)):
                subset = tuple(name for bit, name in enumerate(names) if code & (1 << bit))
                table.append(lookup[subset])
            groups.append({"actor": actor, "rule_ids": names, "sphere_indices": tuple(sorted(sphere_indices)),
                           "channel_lookup": tuple(table)})
        return cls(origin=origin, voxel_size=h, shape=shape, rules=rules,
                   maximum_radii=maximum_radii, regions=regions, channels=channels, groups=groups, source=source)

    def validate_binding(self, rules, maximum_radii):
        if (tuple(_plain(asdict(rule)) for rule in rules) != self.rule_records
                or {name: float(value) for name, value in maximum_radii.items()} != self.maximum_radii):
            raise ValueError("UNSUPPORTED_MODEL: frozen contact fields disagree with the exact material/rule/radius binding")

    def constant(self, name, value, like):
        """Own small immutable model tensors without writable NumPy aliases."""
        key = ("constant", name, like.device, like.dtype)
        if key not in self._device:
            self._device[key] = torch.tensor(value, device=like.device, dtype=like.dtype)
        return self._device[key]

    def _group_constants(self, index, points):
        # Each frozen channel is uploaded once per device/dtype, even when
        # several disjoint sphere-membership groups can select that channel.
        bank_key = ("fields", points.device, points.dtype)
        if bank_key not in self._device:
            arrays, field_indices = [], {}
            for channel_index, channel in enumerate(self.channels):
                if channel["kind"] == "field":
                    field_indices[channel_index] = len(arrays)
                    arrays.append(channel["distance"].transpose(2, 1, 0))
            bank = (torch.as_tensor(np.stack(arrays), device=points.device, dtype=points.dtype)[None]
                    if arrays else None)
            self._device[bank_key] = bank, field_indices
        bank, field_indices = self._device[bank_key]
        key = ("group", index, points.device, points.dtype)
        if key not in self._device:
            group = self.groups[index]
            kinds, mapping = [], []
            for channel_index in group["channel_lookup"]:
                channel = None if channel_index < 0 else self.channels[channel_index]
                kind = "unchanged" if channel is None else channel["kind"]
                kinds.append({"unchanged": 0, "empty": 1, "field": 2}[kind])
                mapping.append(field_indices[channel_index] if kind == "field" else 0)
            self._device[key] = (bank,
                torch.tensor(kinds, device=points.device, dtype=torch.long),
                torch.tensor(mapping, device=points.device, dtype=torch.long),
                torch.tensor(group["sphere_indices"], device=points.device, dtype=torch.long))
        return self._device[key]

    def query(self, spheres, policies, ordinary, intersects_geometry):
        """Only batched field sampling and rule-bit selection, never source cells."""
        by_rule = {rule.rule_id: policy for rule, policy in policies}
        if not policies:
            return ordinary, torch.zeros_like(ordinary)
        actors = {rule.actor for rule, _ in policies}
        if len(actors) != 1:
            raise ValueError("One contact query must identify one actor's sphere array")
        actor = next(iter(actors))
        values, unsupported = ordinary.clone(), torch.zeros_like(ordinary)
        for index, group in enumerate(self.groups):
            if group["actor"] != actor:
                continue
            if not set(group["rule_ids"]) <= set(by_rule):
                raise ValueError("Contact query must supply every frozen rule for its actor")
            bank, kinds, mapping, indices = self._group_constants(index, spheres)
            selected = spheres.index_select(-2, indices)
            intersects = intersects_geometry.index_select(-1, indices)
            code = torch.zeros_like(selected[..., 0, 3], dtype=torch.long)
            for bit, name in enumerate(group["rule_ids"]):
                code = code + by_rule[name]["active"].to(torch.long) * (1 << bit)
            kind = kinds[code]
            ordinary_selected = ordinary.index_select(-1, indices)
            result = torch.where((kind == 1).unsqueeze(-1), 0., ordinary_selected)
            if bank is not None:
                coordinate = (selected[..., :3] - self.constant("origin", self.origin, spheres)) / self.voxel_size
                grid = 2 * coordinate / self.constant("shape_minus_one", np.asarray(self.shape) - 1, spheres) - 1
                sampled = F.grid_sample(bank, grid.reshape(1, 1, 1, -1, 3),
                    align_corners=True, padding_mode="zeros", mode="bilinear")
                sampled = sampled.reshape(bank.shape[1], *selected.shape[:-1])
                take = mapping[code].unsqueeze(0).unsqueeze(-1).expand(1, *selected.shape[:-1])
                distance = sampled.gather(0, take).squeeze(0)
                collision = torch.relu(selected[..., 3] - distance)
                result = torch.where((kind == 2).unsqueeze(-1), collision, result)
            radius = selected[..., 3:4].clamp_min(0)
            support = ((selected[..., :3] - radius >= self.constant("origin", self.origin, spheres)) &
                       (selected[..., :3] + radius <= self.constant("upper", self.upper, spheres))).all(-1)
            # A far-away optional contact does not require a local owner map.
            # Explicit empty sets need no distance lookup at all; the scene's
            # independent unknown-space policy still applies everywhere.
            missing = (kind == 2).unsqueeze(-1) & intersects & ~support & (selected[..., 3] > 0)
            # Missing field support is a failure, never a border-clamped pass.
            # Keep the ordinary value too; no unsupported state may erase it.
            result = torch.where(missing, ordinary_selected, result)
            result = torch.where(intersects, result, ordinary_selected)
            result = torch.where(selected[..., 3] > 0, result, 0.)
            values = values.index_copy(-1, indices, result)
            unsupported = unsupported.index_copy(-1, indices, missing.to(spheres.dtype))
        return values, unsupported

    def required_contact(self, rule, policy):
        """A bounded actual source witness must satisfy all conditions together."""
        points = self.regions[rule.rule_id]["surface_witnesses"]
        normals, bounds = policy["normals"], policy["bounds"]
        if not len(points):
            return torch.ones_like(policy["active"], dtype=normals.dtype)
        witnesses = self.constant((rule.rule_id, "witnesses"), points, normals)
        excess = torch.einsum("...pc,vc->...vp", normals, witnesses) - bounds.unsqueeze(-2)
        return torch.relu(excess).amax(-1).amin(-1)

    def record(self):
        return {"model": MODEL, "origin": self.origin.tolist(), "voxel_size": self.voxel_size,
            "shape": list(self.shape), "source": self.source, "rules": _plain(self.rule_records),
            "maximum_radii": dict(self.maximum_radii),
            "regions": {name: {"lower_patch": row["lower_patch"].tolist(),
                "upper_patch": row["upper_patch"].tolist(), "has_removed_material": row["has_removed_material"],
                "removed_cells": row["removed_cells"], "surface_witnesses": encode_array(row["surface_witnesses"])}
                for name, row in self.regions.items()},
            "channels": [{"rule_ids": list(row["rule_ids"]), "kind": row["kind"],
                **({"distance": encode_array(row["distance"])} if row["kind"] == "field" else {})}
                for row in self.channels], "groups": _plain(self.groups),
            "numerics": {"cell_cover_margin_m": math.sqrt(3.) * self.voxel_size / 2,
                         "interpolation_margin_m": math.sqrt(3.) * self.voxel_size / 2,
                         "maximum_surface_witnesses": _WITNESS_BINS ** 2},
            "scope": "fixed forbidden-region cells plus registered native planar extrusion; conservative projection enclosures; bounded observed witnesses may miss contact between sites; not exact cell Boolean equivalence"}

    @classmethod
    def from_record(cls, row):
        """Restore exact frozen fields; execution does not rerun EDT or sampling."""
        from .planning_scene import ContactRule
        if (row.get("model") != MODEL or not 0 < len(row["channels"]) <= _MAX_FIELDS
                or not 0 < len(row["rules"]) <= _MAX_RULES):
            raise ValueError("UNSUPPORTED_MODEL: exact frozen planar-contact binding is required")
        shape, origin, h = row["shape"], np.asarray(row["origin"], float), row["voxel_size"]
        if (not isinstance(shape, (list, tuple)) or len(shape) != 3
                or any(type(value) is not int or value < 2 for value in shape)
                or math.prod(shape) * 4 > MAX_ARRAY_BYTES or origin.shape != (3,)
                or not np.isfinite(origin).all() or not np.isscalar(h) or not np.isfinite(h) or h <= 0
                or not isinstance(row["source"], str) or not row["source"].strip()):
            raise ValueError("Frozen contact grid must retain finite bounded layout and provenance")
        rules = tuple(ContactRule(**value) for value in row["rules"])
        names = {rule.rule_id for rule in rules}
        if (len(names) != len(rules) or set(row["maximum_radii"]) != names or set(row["regions"]) != names
                or any(not np.isscalar(v) or not np.isfinite(v) or v <= 0 for v in row["maximum_radii"].values())):
            raise ValueError("Frozen contact rules, radii and finite region inventory must agree exactly")
        for rule in rules:
            require_planar_material(rule)
        grouped, combinations_needed = _field_layout(rules)
        if [tuple(item["rule_ids"]) for item in row["channels"]] != combinations_needed:
            raise ValueError("Frozen contact channels must retain every exact permitted rule union")
        lookup = {value: index for index, value in enumerate(combinations_needed)}
        expected_groups = []
        for (actor, ids), indices in sorted(grouped.items()):
            table = [-1] + [lookup[tuple(name for bit, name in enumerate(ids) if code & (1 << bit))]
                           for code in range(1, 2 ** len(ids))]
            expected_groups.append({"actor": actor, "rule_ids": list(ids),
                "sphere_indices": sorted(indices), "channel_lookup": table})
        if row["groups"] != expected_groups:
            raise ValueError("Frozen contact groups must preserve exact material sphere membership and field lookup")
        channels = []
        stored = 0
        for item in row["channels"]:
            kind = item["kind"]
            if kind not in {"field", "empty", "unchanged"}:
                raise ValueError("Unknown frozen contact field kind")
            channel = {"rule_ids": tuple(item["rule_ids"]), "kind": kind}
            if kind == "field":
                field = decode_array(item["distance"])
                if tuple(field.shape) != tuple(row["shape"]) or field.dtype != np.float32 or not np.isfinite(field).all():
                    raise ValueError("Frozen contact field arrays must preserve their full layout and dtype")
                stored += field.nbytes
                if stored > _MAX_STORED_BYTES:
                    raise ValueError("Frozen contact fields exceed their storage bound")
                channel["distance"] = field
            channels.append(channel)
        regions = {}
        rule_by_name = {rule.rule_id: rule for rule in rules}
        for name, item in row["regions"].items():
            witnesses = decode_array(item["surface_witnesses"])
            if witnesses.ndim != 2 or witnesses.shape[1] != 3 or len(witnesses) > _WITNESS_BINS ** 2 or not np.isfinite(witnesses).all():
                raise ValueError("Frozen observed contact witnesses exceed their finite scope")
            low, high = np.asarray(item["lower_patch"], float), np.asarray(item["upper_patch"], float)
            if (low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all()
                    or np.any(low > high) or type(item["has_removed_material"]) is not bool
                    or type(item["removed_cells"]) is not int or not 0 <= item["removed_cells"] <= math.prod(shape)
                    or item["has_removed_material"] != (item["removed_cells"] > 0)):
                raise ValueError("Frozen material region bounds and source-cell count must remain finite and consistent")
            rule = rule_by_name[name]
            widths = np.r_[rule.patch_half_size, rule.modeling_allowance_m]
            tolerance = rule.numerical_position_tolerance_m + 1e-12
            if (np.any(low < -widths - tolerance) or np.any(high > widths + tolerance)
                    or (not item["has_removed_material"] and (len(witnesses) or np.any(low != 0) or np.any(high != 0)))
                    or (rule.required and not len(witnesses))):
                raise ValueError("Frozen removed material and required evidence must stay within the declared finite patch")
            witness_local = (witnesses - rule.patch_origin) @ rule.patch_axes
            if np.any(witness_local < low - tolerance) or np.any(witness_local > high + tolerance):
                raise ValueError("Frozen observed witness lies outside its prepared material region")
            regions[name] = {**item, "lower_patch": _readonly(item["lower_patch"], float),
                "upper_patch": _readonly(item["upper_patch"], float), "surface_witnesses": witnesses}
        return cls(origin=row["origin"], voxel_size=row["voxel_size"], shape=row["shape"], rules=rules,
            maximum_radii=row["maximum_radii"], regions=regions, channels=channels,
            groups=row["groups"], source=row["source"])


def planar_material_conditions(rule, policy, material_normal, prepared):
    """Few conservative supports cover complete native material, including tilt.

    The full native body is F + [-thickness,0] * outward_normal. Its projected
    XY box, intersected with the removed target's projected box, encloses every
    possible material intersection. The minimum material-plane coordinate of
    that box occurs at one of four XY corners and the highest target material
    height (opposing normals). Requiring it >= -allowance rules out side/back
    penetration without adding patch thickness to the permitted allowance.

    Gap bounds use the analogous front-feature projection enclosure. These
    enclosures may reject feasible tilted/rotated contacts; they never require
    an entire larger pad to fit a smaller patch. Optional rules impose these
    conditions only when complete native-body/removed-material bounds overlap.
    """
    front = policy["feature_points"]
    axes, origin = _tensor(rule.patch_axes, front), _tensor(rule.patch_origin, front)
    a = torch.einsum("...c,cd->...d", material_normal, axes)
    feature = torch.einsum("...vc,cd->...vd", front - origin, axes)
    body = torch.cat((feature, feature - float(rule.online_material["thickness_m"]) * a.unsqueeze(-2)), dim=-2)
    region = prepared.regions[rule.rule_id]
    zero = torch.zeros_like(policy["active"], dtype=front.dtype)
    if not region["has_removed_material"]:
        return {"gap": zero, "material_band": zero}
    lower = prepared.constant((rule.rule_id, "lower_patch"), region["lower_patch"], front)
    upper = prepared.constant((rule.rule_id, "upper_patch"), region["upper_patch"], front)
    body_lower, body_upper = body.amin(-2), body.amax(-2)
    possible = ((body_upper >= lower) & (body_lower <= upper)).all(-1) & policy["active"]
    half = _tensor(rule.patch_half_size, front)

    def rectangle(points, low, high):
        first = torch.maximum(points[..., :2].amin(-2), low)
        last = torch.minimum(points[..., :2].amax(-2), high)
        overlap = (first <= last).all(-1)
        # Unordered/empty bounds are still finite; the boolean controls use.
        corners = torch.stack((first, torch.stack((first[..., 0], last[..., 1]), -1),
                               last, torch.stack((last[..., 0], first[..., 1]), -1)), -2)
        return corners, overlap

    face_xy, face_overlap = rectangle(feature, -half, half)
    plane = (a * feature[..., 0, :]).sum(-1)
    # Invalid normals carry their own nonzero hard residual. Never divide by
    # a zero/reversed normal while the optimizer explores such a state.
    denominator = a[..., 2].clamp_max(-1e-6)
    heights = (plane.unsqueeze(-1) - (face_xy * a[..., None, :2]).sum(-1)) / denominator.unsqueeze(-1)
    tolerance = rule.numerical_position_tolerance_m
    gap = torch.relu(rule.gap_bounds[0] - heights.amin(-1) - tolerance)
    gap = gap + torch.relu(heights.amax(-1) - rule.gap_bounds[1] - tolerance)
    gap = torch.where(possible & face_overlap, gap, 0.)
    body_xy, body_overlap = rectangle(body, lower[:2], upper[:2])
    coordinate = (body_xy * a[..., None, :2]).sum(-1) + a[..., 2:3] * upper[2] - plane.unsqueeze(-1)
    penetration = torch.relu(-coordinate.amin(-1) - rule.modeling_allowance_m - tolerance)
    penetration = torch.where(possible & body_overlap, penetration, 0.)
    return {"gap": gap, "material_band": penetration}
# END online contact work is independent of the number of source occupied cells.
