"""Freeze raw Mapper observations without changing their implicit geometry.

Every entity uses its own rigid frame. Camera poses are expressed in that
frame before integration; the resulting samples remain fixed there when a
declared relationship subsequently moves the entity. Identity masks determine
which measured endpoints enter an entity, independently of its motion.
"""

# BEGIN one RGB-D-to-implicit-field producer for all registered entities.
from pathlib import Path
import json
import math

import numpy as np
import torch

from .collision_records import encode_array, decode_array, MAX_ARRAY_BYTES
from .implicit_distance import ObservedField
from .resource_admission import (
    DEFAULT_MAX_HOST_RSS_BYTES, DEFAULT_MAX_DEVICE_USED_BYTES, require_host_memory,
    require_device_memory,
)


class FrozenObservedField(ObservedField):
    """Raw weighted observations, support and entity identity in one frame."""

    def __init__(self, values, observed, *, weights, origin, voxel_size, name, source):
        super().__init__(values, observed, origin=origin, voxel_size=voxel_size)
        self.weights = np.asarray(weights, dtype=np.float32).copy()
        if (self.weights.shape != self.values.shape or not np.isfinite(self.weights).all()
                or np.any(self.weights < 0) or not name or not source):
            raise ValueError("A frozen field needs nonnegative sample weights and provenance")
        self.weights.flags.writeable = False
        self.name, self.source = str(name), str(source)
        self.owner_ids = frozenset((self.name,))
        self.storage_nbytes += self.weights.nbytes

    def clone_for_rollout(self):
        """A rollout changes rigid poses while sharing immutable observations."""
        return self

    def record(self):
        """Store the samples defining the set; indexes are reconstructed."""
        return {
            "kind": "observed_trilinear_field_v1", "name": self.name, "source": self.source,
            "values": encode_array(self.values), "observed": encode_array(self.observed),
            "weights": encode_array(self.weights), "origin": self.origin.tolist(),
            "voxel_size": self.voxel_size,
        }

    @classmethod
    def from_record(cls, record):
        if record["kind"] != "observed_trilinear_field_v1":
            raise ValueError("A frozen field requires raw trilinear sample records")
        return cls(decode_array(record["values"]), decode_array(record["observed"]),
                   weights=decode_array(record["weights"]), origin=record["origin"],
                   voxel_size=record["voxel_size"], name=record["name"], source=record["source"])

    def save(self, path):
        Path(path).write_text(json.dumps(self.record(), allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls.from_record(json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self):
        return {
            "name": self.name, "source": self.source, "shape_xyz": self.shape.tolist(),
            "origin_m": self.origin.tolist(), "voxel_size_m": self.voxel_size,
            "observed_samples": int(np.count_nonzero(self.observed)),
            "supported_cells": int(np.count_nonzero(self.valid_cells)),
            "occupied_candidate_cells": len(self.cells),
            "representation": "nonpositive_trilinear_values_on_fully_observed_cells",
            "storage_bytes": self.storage_nbytes,
        }


def freeze_mapper(mapper, *, name, source,
                  max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES):
    """Decode the native weighted TSDF block export into XYZ sample arrays.

    Native block coordinates are centered, and local samples are indexed as
    z*B*B+y*B+x. The origin denotes the first sample center, not a voxel face.
    """
    config = mapper.config
    if config.enable_static:
        raise ValueError("A frozen observation requires the dynamic TSDF channel")
    shape = np.asarray(config.grid_shape[::-1], dtype=np.int64)
    if math.prod(shape) * 4 > MAX_ARRAY_BYTES:
        raise ValueError("Raw TSDF samples exceed the collision-record array budget")
    block_size = int(mapper.tsdf.block_size)
    export_bytes = int(config.max_blocks) * (block_size ** 3 * 8 + 24)
    require_host_memory(int(2 * export_bytes + math.prod(shape) * 64),
                        pending_existing_bytes=0, max_rss_bytes=max_host_rss_bytes)
    size = float(config.voxel_size)
    center = config.grid_center.detach().cpu().numpy().astype(float)
    origin = center - (shape - 1) * size / 2
    exported = mapper.tsdf.export_blocks()
    keys = exported["active_block_coords"].detach().cpu().numpy()
    data = exported["block_data"].detach().cpu().numpy()
    del exported
    block_offset = np.ceil(shape / block_size).astype(np.int64) // 2
    values = np.zeros(tuple(shape), dtype=np.float32)
    weights = np.zeros(tuple(shape), dtype=np.float32)
    for key, payload in zip(keys, data):
        base = (np.asarray(key, dtype=np.int64) + block_offset) * block_size
        lower, upper = np.maximum(base, 0), np.minimum(base + block_size, shape)
        if np.any(lower >= upper):
            continue
        region = tuple(slice(int(a), int(b)) for a, b in zip(lower, upper))
        local = tuple(slice(int(a), int(b)) for a, b in zip(lower - base, upper - base))
        weighted = np.asarray(payload, dtype=np.float32).reshape(
            block_size, block_size, block_size, 2).transpose(2, 1, 0, 3)[local]
        weights[region] = weighted[..., 1]
        np.divide(weighted[..., 0], weighted[..., 1], out=values[region],
                  where=weighted[..., 1] > 0)
    observed = weights >= float(config.minimum_tsdf_weight)
    return FrozenObservedField(values, observed, weights=weights, origin=origin,
                               voxel_size=size, name=name, source=source)


def integrate_entity(observations, *, config, name, source,
                     max_host_rss_bytes=DEFAULT_MAX_HOST_RSS_BYTES,
                     max_device_used_bytes=DEFAULT_MAX_DEVICE_USED_BYTES,
                     block_checkpoint=None):
    """Integrate an iterable of calibrated, identity-partitioned RGB-D views.

    Each row provides depth (camera-z meters), an entity mask, RGB, K with
    integer pixel centers and T_entity_camera in the OpenCV camera convention.
    Rows may describe different times: their poses already express the camera
    in the same entity-local frame. Unknown pixels contribute no depth.
    """
    from curobo.perception import Mapper
    from curobo.types import CameraObservation, DeviceCfg, Pose
    from .planning_observation import _mapper_memory_bound, _mapper_intrinsics

    bound = _mapper_memory_bound(config)
    require_host_memory(bound, pending_existing_bytes=0, max_rss_bytes=max_host_rss_bytes)
    require_device_memory(torch, config.device, native_bytes=bound,
                          max_device_used_bytes=max_device_used_bytes)
    device = DeviceCfg(device=config.device)
    mapper = Mapper(config)
    count = 0
    for count, row in enumerate(observations, 1):
        depth = np.where(row["mask"], row["depth"], 0.).astype(np.float32)
        if depth.shape != (config.image_height, config.image_width):
            raise ValueError("Every observation must match the configured camera dimensions")
        observation = CameraObservation(
            name=f"{name}:{count}", depth_image=device.to_device(depth)[None],
            rgb_image=torch.as_tensor(np.asarray(row["image"]).copy(),
                                      dtype=torch.uint8, device=device.device)[None],
            intrinsics=device.to_device(_mapper_intrinsics(row["K"]))[None],
            pose=Pose.from_matrix(device.to_device(row["T_entity_camera"])[None]),
            resolution=list(depth.shape), depth_to_meter=1.,
        )
        mapper.integrate(camera_observation=observation)
    if not count:
        raise ValueError("An entity field requires at least one observation")
    stats = mapper.get_stats(scan_pool=True)
    if stats["allocation_failures"]:
        raise ValueError("TSDF allocation failed while integrating observed endpoints")
    if block_checkpoint is not None:
        # The native checkpoint retains the same weighted samples for the
        # maintained TSDF raycaster used by the image self filter. Collision
        # queries consume the raw field below, independently of rendering.
        mapper.save_blocks(block_checkpoint)
    result = freeze_mapper(mapper, name=name, source=source,
                           max_host_rss_bytes=max_host_rss_bytes)
    return result, {"observation_count": count, "mapper": stats, **result.summary()}
# END moving a field changes its pose, never its samples or observation support.
