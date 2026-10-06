"""Locate the confirmed TSDF mesh center/corner mismatch with one known plane.

The analytic tilted plane has nonzero x/y/z normal components. One depth image
is fused into one small map, then the public raw surface and FUNCTO-corrected
surface are compared against that plane. No planning or real-scene rerun occurs.
"""

# BEGIN one bounded localization for the newly identified mesh-coordinate fault.
import argparse
import gc
import json
from pathlib import Path
import sys

import numpy as np
import torch
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.mapper_surface import extract_tsdf_surface
from manipulation.planning_observation import _mapper_memory_bound, _require_mapping_memory
from curobo.perception import Mapper, MapperCfg
from curobo.types import CameraObservation, DeviceCfg, Pose


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(512*1024**2/torch.cuda.get_device_properties(0).total_memory)
    wp.config.kernel_cache_dir = '/tmp/curobo-small-roi-warp-cache'
    device = DeviceCfg('cuda:0')
    normal = np.array([1., 2., 3.])
    normal /= np.linalg.norm(normal)
    point = np.array([0., 0., 1.])
    offset = float(normal@point)
    height = width = 64
    intrinsic = np.array([[120., 0., 31.5], [0., 120., 31.5], [0., 0., 1.]])
    # BEGIN isolate mesh coordinates from the integrator's pixel lookup.
    # The library samples depth at integer pixel coordinates. A tilted image
    # would mix that projective discretization into this regression. Face the
    # camera toward the plane so every pixel has exactly the same z-depth;
    # the surface remains tilted along all three axes of the Mapper frame.
    x_axis = np.cross([0., 1., 0.], normal)
    x_axis /= np.linalg.norm(x_axis)
    camera_pose = np.eye(4)
    camera_pose[:3, :3] = np.column_stack((x_axis, np.cross(normal, x_axis), normal))
    camera_pose[:3, 3] = point-normal
    depth = np.ones((height, width), dtype=np.float32)
    # END constant depth removes pixel-center and interpolation assumptions.
    cfg = MapperCfg(extent_meters_xyz=(.04, .04, .04), voxel_size=.002,
        extent_esdf_meters_xyz=(.04, .04, .04), esdf_voxel_size=.002,
        grid_center=torch.tensor(point, dtype=torch.float32), truncation_distance=.004,
        image_height=height, image_width=width, depth_minimum_distance=.05,
        depth_maximum_distance=5., device='cuda:0', seeding_method='scatter')
    # BEGIN this direct native probe uses the shared total-resource admission.
    _require_mapping_memory(_mapper_memory_bound(cfg), cfg.device,
        max_host_rss_bytes=8*1024**3, max_device_used_bytes=6*1024**3)
    # END no native Mapper is created before host and device admission.
    mapper = Mapper(cfg)
    camera = CameraObservation(name='analytic_plane', depth_image=device.to_device(depth)[None],
        rgb_image=torch.zeros((1, height, width, 3), dtype=torch.uint8, device=device.device),
        intrinsics=device.to_device(intrinsic)[None],
        pose=Pose.from_matrix(device.to_device(camera_pose)[None]),
        resolution=[height, width], depth_to_meter=1.)
    mapper.integrate(camera_observation=camera)
    raw = mapper.extract_mesh(refine_iterations=0, surface_only=True)
    raw_vertices = raw.vertices.detach().cpu().numpy().astype(np.float64)
    vertices, faces = extract_tsdf_surface(mapper)
    if not len(vertices):
        raise AssertionError('The fully observed analytic plane must export triangles')
    raw_error = raw_vertices@normal-offset
    corrected_error = vertices@normal-offset
    # Different export launches may reorder triangles. Plane residuals test
    # actual geometry, without depending on that atomic output ordering.
    result = {'normal': normal.tolist(), 'offset_m': offset, 'voxel_size_m': cfg.voxel_size,
        'camera_pose': camera_pose.tolist(), 'constant_camera_depth_m': 1.,
        'triangles': len(faces), 'raw_plane_error_m': {
            'min': float(raw_error.min()), 'max': float(raw_error.max()), 'mean': float(raw_error.mean())},
        'corrected_plane_error_m': {'min': float(corrected_error.min()),
            'max': float(corrected_error.max()), 'mean': float(corrected_error.mean())},
        'expected_raw_shift_along_normal_m': -cfg.voxel_size*float(normal.sum())/2,
        # The plane is affine in this camera's z-depth. Allow a small absolute
        # tolerance for float32 projection and fp16 TSDF weighted accumulators.
        'plane_tolerance_m': 3e-5, 'tsdf_integration_passes': 1,
        'planner_called': False, 'physics_steps': 0}
    result['passed'] = bool(np.max(np.abs(corrected_error)) < result['plane_tolerance_m']
        and abs(raw_error.mean()-result['expected_raw_shift_along_normal_m']) < result['plane_tolerance_m'])
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, allow_nan=False), flush=True)
    if not result['passed']:
        raise AssertionError('TSDF surface disagrees with the independently specified plane')
    del mapper, raw, camera
    gc.collect()


if __name__ == '__main__':
    main()
# END one image supplies an analytic reference independent of marching cubes.
