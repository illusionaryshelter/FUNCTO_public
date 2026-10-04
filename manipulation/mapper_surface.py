"""Observed TSDF triangle surfaces in the Mapper's integration coordinates.

These surfaces retain only the observed zero level set. They do not establish
an occupied volume, obstacle containment, or freedom in unobserved space.
"""

import numpy as np


# BEGIN FUNCTO owns the installed Mapper mesh-coordinate correction.
def extract_tsdf_surface(mapper):
    """Return owned CPU vertices/faces without changing the Mapper or its TSDF.

    The installed cuRobo ``builder_coord.voxel_to_world`` puts TSDF samples at
    ``index + 0.5``, as used by camera integration and occupied-voxel export.
    Its ``builder_mesh.generate_mesh_kernel`` instead places those same sample
    values at ``index``. With zero refinement, restoring each coordinate by
    half the TSDF voxel size recovers the sample-center zero-level surface.
    This is a coordinate correction, not an obstacle or clearance allowance.

    Use the public Mapper API, whose underlying level already defaults to 0.
    Refinement is explicitly zero: Newton projection would use the mismatched
    positions before this correction, so the same translation would no longer
    describe that output. Preserve the original triangle topology; no convex
    hull, welding, hole filling, or surface thickening occurs here.

    Vertices are in the frame used by Mapper.integrate(). Background vertices
    already include grid_center; owner-local vertices need the owner pose once
    when queried. Do not additionally apply the ESDF grid's pose or lower bound.
    """
    mesh = mapper.extract_mesh(refine_iterations=0, surface_only=True)
    vertices = mesh.vertices.detach().cpu().numpy().astype(np.float64, copy=True)
    faces = mesh.faces.detach().cpu().numpy().astype(np.int64, copy=True)
    vertices += float(mapper.config.voxel_size) / 2
    return vertices, faces
# END this surface export does not alter any planning acceptance predicate.
