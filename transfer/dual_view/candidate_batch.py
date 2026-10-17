"""Per-call CUDA geometry owner and dense oriented candidate projection.

Only current observations are uploaded. The caller keeps the original query,
check and center grouping; neither predictions nor previous uploads are reused.
"""
import numpy as np
import torch


BOOL_FIELDS = ("supported", "conflict", "unknown", "position_supported",
               "plane_valid", "oriented_supported", "entity_supported",
               "query_normal_valid")


class DenseProjector:
    """Upload one unchanged full-depth surface and classify dense FP64 rows.

    Host shape checks are static metadata only. Every data-dependent predicate
    stays on the device. Invalid IDs use a single inert sentinel observation;
    their masks prevent that sentinel from supplying any geometric evidence.
    The equations and state priority match project_observations(tolerance=0)
    and project_oriented, including owner bypass and entity corner weights.
    """

    def __init__(self, surface, *, device="cuda"):
        self.device = device
        self.factor = float(surface["surface_height_spacing_factor"])
        self.agreement = float(surface["normal_min_agreement"])
        self.radius = float(surface["sampling_radius_m"])
        self.K = [self.tensor(value, torch.float64) for value in surface["K"]]
        self.T = [self.tensor(value, torch.float64) for value in surface["T_world_camera_cv"]]
        self.lookup = [self.tensor(value, torch.int64) for value in surface["pixel_to_observation"]]
        self.observation_count = len(surface["observation_xyz"])
        self.observed = self.padded(surface["observation_xyz"], torch.float64, 0)
        self.normals = self.padded(surface["observation_normals"], torch.float64, 0)
        self.spacing = self.padded(surface["observation_spacing"], torch.float64, 0)
        self.valid = self.padded(surface["observation_valid"], torch.bool, False)
        self.entity = self.padded(surface["observation_entity_member"], torch.bool, False)

    def tensor(self, value, dtype):
        return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=self.device)

    def padded(self, value, dtype, fill):
        value = self.tensor(value, dtype)
        padding = torch.full((1, *value.shape[1:]), fill, dtype=dtype, device=self.device)
        return torch.cat((value, padding), dim=0)

    def observation_ids(self, ids):
        return torch.where(ids >= 0, ids, self.observation_count)

    def project(self, points, query_normals, view):
        """Return packed device results; all returned values are GPU computed."""
        K, T, lookup = self.K[view], self.T[view], self.lookup[view]
        height, width = lookup.shape
        camera = (points - T[:3, 3]) @ T[:3, :3]
        homogeneous = camera @ K.T
        finite = torch.isfinite(homogeneous).all(1) & (homogeneous[:, 2] != 0)
        uv = homogeneous[:, :2] / homogeneous[:, 2, None]
        uv = torch.where(finite[:, None], uv, torch.nan)
        rounded = torch.round(uv)
        numeric = 64 * np.finfo(np.float64).eps * max(height, width)
        uv = torch.where(torch.abs(uv - rounded) <= numeric, rounded, uv)
        front = finite & (camera[:, 2] > 0)
        inside = front & (uv >= 0).all(1) & (uv[:, 0] <= width-1) & (uv[:, 1] <= height-1)
        states = torch.full((len(points),), 10, dtype=torch.int64, device=self.device)
        states = torch.where(finite & ~front, 2, states)
        states = torch.where(front & ~inside, 3, states)

        # Clamp only lookup operands. Original UV, masks and returned IDs retain
        # the original invalid/outside states; a clamped pixel is never evidence.
        safe_uv = torch.nan_to_num(uv, nan=0., posinf=0., neginf=0.)
        safe_uv = torch.stack((safe_uv[:, 0].clamp(0, width-1),
                               safe_uv[:, 1].clamp(0, height-1)), dim=1)
        nearest_xy = torch.round(safe_uv).to(torch.int64)
        nearest = lookup[nearest_xy[:, 1], nearest_xy[:, 0]]
        present = inside & (nearest >= 0)
        ids = torch.where(present, nearest, -1)
        near = self.observation_ids(ids)
        states = torch.where(inside & ~present, 4, states)
        reference = self.observed[near]
        delta = points - reference
        exact = present & (torch.abs(delta) <= (
            64*np.finfo(np.float64).eps * torch.maximum(torch.ones_like(reference),
                                                       torch.abs(reference)))).all(1)
        states = torch.where(exact, 0, states)
        active = present & ~exact

        low, high = torch.floor(safe_uv).to(torch.int64), torch.ceil(safe_uv).to(torch.int64)
        corners = torch.stack((low, torch.stack((high[:, 0], low[:, 1]), dim=1),
                               torch.stack((low[:, 0], high[:, 1]), dim=1), high), dim=1)
        corner_ids = lookup[corners[:, :, 1], corners[:, :, 0]]
        corner_present = (corner_ids >= 0).all(1)
        safe_corners = self.observation_ids(corner_ids)
        states = torch.where(active & ~corner_present, 4, states)
        active = active & corner_present
        plane_valid = present & self.valid[near]
        planes = self.valid[near] & self.valid[safe_corners].all(1)
        states = torch.where(active & ~planes, 5, states)
        active = active & planes
        normal = self.normals[near]
        thickness = self.factor * self.spacing[near]
        corner_delta = self.observed[safe_corners] - reference[:, None]
        same = (torch.abs((corner_delta * normal[:, None]).sum(2)) <= thickness[:, None]).all(1)
        same = same & ((self.normals[safe_corners] * normal[:, None]).sum(2) >= self.agreement).all(1)
        states = torch.where(active & ~same, 6, states)
        active = active & same
        signed_height = (delta * normal).sum(1)
        in_slab = torch.abs(signed_height) <= thickness
        local = torch.sqrt((delta * delta).sum(1)) <= self.radius
        states = torch.where(active & in_slab & local, 1, states)
        states = torch.where(active & in_slab & ~local, 7, states)
        states = torch.where(active & ~in_slab & (signed_height > 0), 8, states)
        states = torch.where(active & ~in_slab & (signed_height <= 0), 9, states)

        normal_valid = torch.isfinite(query_normals).all(1) & (
            torch.abs(torch.sqrt((query_normals * query_normals).sum(1)) - 1) <= 1e-6)
        oriented = plane_valid & normal_valid & ((query_normals * normal).sum(1) >= self.agreement)
        fraction = safe_uv - low
        fx, fy = fraction[:, 0], fraction[:, 1]
        weights = torch.stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy), dim=1)
        labeled = (corner_ids >= 0) & self.entity[safe_corners]
        entity = present & ((weights <= 0) | labeled).all(1)
        position = (states == 0) | (states == 1)
        conflict = states == 8
        supported = position & plane_valid & oriented & entity
        boolean = torch.stack((supported, conflict, ~(supported | conflict), position,
                               plane_valid, oriented, entity, normal_valid), dim=1)
        return uv, torch.stack((ids, states), dim=1), boolean
