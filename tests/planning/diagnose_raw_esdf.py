"""Read an archived V2 ESDF and report distances before collision activation.

This is a diagnostic, never a source of free-space permission. Queries specify
``center_in_parent_frame_m`` and ``radius_m``: the parent is the frame containing
the serialized voxel record's pose (world for the archived background). A moving
owner's world-to-owner transform must be applied by the caller at the query time.
The voxel record's pose is then applied exactly once by this module.

The installed cuRobo Warp sampler and activation function run on CPU. No distance
is reconstructed from a clipped cost, no map is regenerated, and no field sign,
resolution or occupancy sample is changed. GPU visibility should be disabled by
the caller. The small native kernel is compiled into an explicit temporary cache.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import warp as wp
from curobo._src.geom.data.data_voxel import sample_voxel_sdf_with_grad
from curobo._src.geom.collision.wp_collision_common import apply_collision_activation


@wp.kernel
def _query_native(field: wp.array(dtype=wp.float16),
                  points: wp.array(dtype=wp.vec3),
                  radius: wp.array(dtype=wp.float32),
                  activation: wp.array(dtype=wp.float32),
                  weight: wp.array(dtype=wp.float32),
                  dims: wp.vec3i, size: wp.float32,
                  output: wp.array(dtype=wp.vec3),
                  raw_gradient: wp.array(dtype=wp.vec3)):
    """Use the installed sampler and the installed cost activation on CPU."""
    i = wp.tid()
    sample = sample_voxel_sdf_with_grad(field, 0, points[i], dims, size, 100.0)
    raw = sample[0]
    raw_gradient[i] = wp.vec3(sample[1], sample[2], sample[3])
    # compute_local_sdf_with_grad caps the sampler result at max_dist=100 m.
    # Preserve raw separately, including the mapper's +10000 m missing-site value.
    collision_sdf = wp.min(raw, 100.0)
    penetration = radius[i] + activation[i] - collision_sdf
    activated = apply_collision_activation(penetration, activation[i])[0]
    output[i] = wp.vec3(raw, raw - radius[i], weight[i] * activated)


def find_voxel_record(snapshot, name):
    """Select one named record without assuming a task, actor or dataset."""
    matches = []
    def visit(value):
        if isinstance(value, dict):
            if value.get("kind") == "voxel" and value.get("name") == name:
                matches.append(value)
            else:
                for item in value.values():
                    visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)
    visit(snapshot)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one voxel record {name!r}, found {len(matches)}")
    return matches[0]


def diagnose(record, queries):
    """Return raw samples and native CPU interpolation with bounded allocations."""
    raw_shape = record["shape_xyz"]
    if (not isinstance(raw_shape, list) or len(raw_shape) != 3
            or any(type(n) is not int or n < 2 for n in raw_shape)
            or math.prod(raw_shape) > 8_000_000):
        raise ValueError("Diagnostic supports complete 3-D fields of 2..8M samples")
    # Python integer product is checked before int64 conversion: malformed
    # dimensions cannot overflow into an apparently small allocation.
    shape = np.asarray(raw_shape, dtype=np.int64)
    sample_count = math.prod(raw_shape)
    if not isinstance(queries, list) or not 0 < len(queries) <= 1024 or any(not isinstance(q, dict) for q in queries):
        raise ValueError("Diagnostic needs 1..1024 explicit queries")
    dtype = np.dtype(record["field_dtype"])
    if dtype != np.dtype("<f2"):
        raise ValueError("Installed native voxel query requires the original float16 field")
    encoded = record["field_bytes_hex"]
    if not isinstance(encoded, str) or len(encoded) != 2*sample_count*dtype.itemsize:
        raise ValueError("Hex field length does not match the complete native grid")
    field = np.frombuffer(bytes.fromhex(encoded), dtype=dtype).reshape(shape)
    if not np.isfinite(field).all():
        raise ValueError("The archived native field contains nonfinite samples")
    occupied = np.flatnonzero(field.reshape(-1) <= 0)
    if record.get("occupancy_rule") != "ESDF_sample_nonpositive" or len(occupied) > 250_000:
        raise ValueError("Diagnostic needs the bounded nonpositive-sample occupancy contract")
    if not np.array_equal(occupied, record["occupied_flat_indices"]):
        raise ValueError("Archived occupancy does not match every nonpositive field sample")
    if len(occupied) * len(queries) > 100_000_000:
        raise ValueError("Pairwise filled-cell diagnostic exceeds bounded CPU work")
    supplied_size = float(record["voxel_size"])
    native_max = float(np.finfo(np.float32).max)
    if not math.isfinite(supplied_size) or not 1./native_max <= supplied_size <= native_max:
        raise ValueError("Voxel size and its reciprocal must be positive finite native float32")
    size = np.float32(supplied_size)
    pose = np.asarray(record["pose"], dtype=float)
    if (pose.shape != (7,) or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1., rtol=0, atol=1e-5)):
        raise ValueError("Field pose must contain finite xyz and a unit wxyz quaternion")
    rotation = Rotation.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
    centres = np.asarray([q["center_in_parent_frame_m"] for q in queries], dtype=np.float32)
    radii = np.asarray([q["radius_m"] for q in queries], dtype=np.float32)
    activations = np.asarray([q.get("activation_distance_m", 0.) for q in queries], dtype=np.float32)
    weights = np.asarray([q.get("weight", 1.) for q in queries], dtype=np.float32)
    if (centres.shape != (len(queries), 3) or not np.isfinite(centres).all()
            or radii.shape != (len(queries),) or not np.isfinite(radii).all() or np.any(radii < 0)
            or activations.shape != (len(queries),) or not np.isfinite(activations).all() or np.any(activations < 0)
            or weights.shape != (len(queries),) or not np.isfinite(weights).all() or np.any(weights < 0)):
        raise ValueError("Queries require finite xyz, nonnegative radii, activation and weight")
    points = np.asarray((centres-pose[:3]) @ rotation, dtype=np.float32)
    if not np.isfinite(points).all():
        raise ValueError("Transformed query coordinates must fit native float32")
    # Native floor/index conversion uses int32. A full factor-of-two margin
    # also protects against float32 rounding at the signed integer limits.
    bounded_coordinates = points.astype(np.float64)/float(size)+shape/2-.5
    if not np.isfinite(bounded_coordinates).all() or np.any(np.abs(bounded_coordinates) >= 2**30):
        raise ValueError("Native continuous grid coordinates exceed safe int32 indexing")
    wp.config.kernel_cache_dir = "/tmp/curobo-esdf-audit-warp-cache"
    wp.init()
    cpu = "cpu"
    output = wp.empty(len(queries), dtype=wp.vec3, device=cpu)
    gradient = wp.empty(len(queries), dtype=wp.vec3, device=cpu)
    wp.launch(_query_native, len(queries), inputs=[
        wp.array(field.reshape(-1), dtype=wp.float16, device=cpu),
        wp.array(points, dtype=wp.vec3, device=cpu),
        wp.array(radii, device=cpu), wp.array(activations, device=cpu),
        wp.array(weights, device=cpu), wp.vec3i(*map(int, shape)), float(size), output, gradient], device=cpu)
    values = output.numpy()
    gradients = gradient.numpy()
    occupied_ijk = np.column_stack(np.unravel_index(occupied, tuple(shape)))
    cell_centres = (occupied_ijk + .5 - shape/2) * float(size)
    rows = []
    for q, point, radius, eta, weight, value, grad in zip(queries, points, radii, activations, weights, values, gradients):
        # Metadata reproduces the sampler's float32 coordinate equation. The
        # interpolation value itself is from installed native code, not this loop.
        coordinates = point*(np.float32(1.)/size) + shape.astype(np.float32)*np.float32(.5) - np.float32(.5)
        base = np.floor(coordinates).astype(np.int64)
        fraction = coordinates-base
        corners = []
        for offset in itertools.product((0, 1), repeat=3):
            index = base+offset
            valid = bool(np.all(index >= 0) and np.all(index < shape))
            corner_weight = float(np.prod(np.where(np.asarray(offset), fraction, 1-fraction)))
            sample = float(field[tuple(index)]) if valid else None
            corners.append({"index_xyz": index.tolist(), "in_bounds": valid,
                            "raw_esdf_m": sample, "raw_signbit": bool(np.signbit(sample)) if valid else None,
                            "trilinear_weight_before_boundary_normalization": corner_weight})
        # Signed distance to each closed cell cube, then subtract radius. This
        # is the minimum sphere/cell pair clearance; for a sphere centre inside
        # the union it is not a global union penetration-depth certificate.
        cube_q = np.abs(cell_centres-point)-float(size)/2
        cell_sdf = (np.linalg.norm(np.maximum(cube_q, 0), axis=1)
                    + np.minimum(np.max(cube_q, axis=1), 0))
        pair_clearance = float(np.min(cell_sdf)-radius) if len(cell_sdf) else None
        support_valid = all(row["in_bounds"] for row in corners)
        rows.append({"query": q, "query_local_m": point.tolist(),
            "continuous_center_index_xyz": coordinates.tolist(), "corners": corners,
            "all_interpolation_corners_in_bounds": support_valid,
            "raw_interpolated_esdf_m": float(value[0]),
            "sphere_radius_m": float(radius), "sphere_surface_clearance_m": float(value[1]),
            "activation_distance_m": float(eta), "weight": float(weight),
            "native_activated_collision_cost": float(value[2]),
            "raw_sampler_gradient_local": grad.tolist(),
            "raw_sampler_gradient_norm": float(np.linalg.norm(grad)),
            "eight_raw_corners_all_zero": support_valid and all(row["raw_esdf_m"] == 0 for row in corners),
            "minimum_sphere_cell_pair_clearance_m": pair_clearance,
            "free_space_certified": False})
    return {"status": "complete", "device": cpu, "voxel_name": record["name"],
        "shape_xyz": shape.tolist(), "voxel_size_m": float(size),
        "field_counts": {"negative": int((field < 0).sum()), "zero": int((field == 0).sum()),
                         "positive": int((field > 0).sum())},
        "distance_convention": "metres; ESDF negative inside, positive outside; sphere clearance = ESDF(center)-radius",
        "cost_convention": "installed V2 activation(weight, radius+eta-min(ESDF,100)); at eta=0 and weight=1: max(0,radius-min(ESDF,100))",
        "collision_sdf_max_distance_m": 100.,
        "collision_sdf_limit_source": "default SceneCollision/SceneData/VoxelData construction used by the archived observation builder; custom limits are not serialized by this archive",
        "scope": "archived field only; native CPU sampler; no GPU, map rebuild, planning, physics or free-space certification",
        "numerical_scope": "installed native formulas run on CPU; float32 query coordinates; not a bitwise replay of GPU FK, pose inversion or kernels",
        "queries": rows}


def read_bounded_json(path, limit_bytes):
    """Bound bytes before JSON parsing, including a file changed after stat."""
    with Path(path).open("rb") as stream:
        payload = stream.read(limit_bytes+1)
    if len(payload) > limit_bytes:
        raise ValueError(f"JSON input exceeds {limit_bytes} bytes: {path}")
    return json.loads(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--voxel-name", required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    snapshot = read_bounded_json(args.scene, 64*1024**2)
    result = diagnose(find_voxel_record(snapshot, args.voxel_name), read_bounded_json(args.queries, 4*1024**2))
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"status": result["status"], "queries": len(result["queries"]), "output": str(args.output)}))


if __name__ == "__main__":
    main()
