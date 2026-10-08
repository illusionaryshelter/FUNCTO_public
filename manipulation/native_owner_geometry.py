"""Validate registered known-owner solids independently of collision candidates.

Only explicit local primitive geometry, conservative proxy coverage and finite
material identities enter these checks. They read no simulator state and query
no world geometry. Both scene admission and native acceptance use this proof.
"""
from itertools import product
import math

import numpy as np
from scipy.spatial.transform import Rotation


# BEGIN complete registered owner solids share the existing native pair checker.
def bind_native_owner_geometry(owner):
    """Verify a full box union and its exact Cartesian-cell sphere ownership.

    A name or a material label cannot prove a proxy covers a native solid.
    Every box is tiled into the declared cells, each associated sphere must
    enclose all eight cell corners, and the resulting sphere groups must
    partition the owner's complete proxy array. These bounded CPU checks run
    at scene admission, including zero-candidate scenes, never in rollouts.
    """
    if (not owner.geometry_complete or not hasattr(owner.query, 'primitives')
            or owner.local_spheres is None):
        raise ValueError('Native owner confirmation requires complete registered primitive geometry')
    spheres = np.asarray(owner.local_spheres)
    geometry, sphere_geometry, groups = {}, {}, {}
    signs = np.asarray(list(product((-1., 1.), repeat=3)))
    for row in owner.query.primitives:
        binding = row.get('native_binding')
        if (row['kind'] not in {'box', 'cuboid'} or not isinstance(binding, dict)
                or set(binding) != {'geometry_name', 'sphere_indices', 'subdivisions', 'source'}):
            raise ValueError('Native owner confirmation requires explicit box/cell binding')
        name, ids, counts = binding['geometry_name'], binding['sphere_indices'], binding['subdivisions']
        if (not isinstance(name, str) or not name.strip() or name in geometry
                or not isinstance(binding['source'], str) or not binding['source'].strip()
                or len(counts) != 3 or any(type(n) is not int or n < 1 for n in counts)
                or math.prod(counts) != len(ids) or not ids
                or any(type(i) is not int or not 0 <= i < len(spheres) for i in ids)
                or len(set(ids)) != len(ids) or set(ids) & sphere_geometry.keys()):
            raise ValueError('Native owner spheres must form distinct complete Cartesian covers')
        half = np.asarray(row['dimensions']) / (2 if row['kind'] == 'cuboid' else 1)
        xyz_wxyz = np.asarray(row['pose'])
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_quat(xyz_wxyz[[4, 5, 6, 3]]).as_matrix()
        pose[:3, 3] = xyz_wxyz[:3]
        vertices = signs * half
        step = 2 * half / counts
        cells = np.asarray(list(product(*(range(n) for n in counts))), dtype=float)
        corners = -half + (cells[:, None, :] + .5) * step + signs[None] * step / 2
        corners = corners @ pose[:3, :3].T + pose[:3, 3]
        balls = spheres[ids]
        if np.any(np.linalg.norm(corners - balls[:, None, :3], axis=-1).max(-1) > balls[:, 3]):
            raise ValueError('Native owner sphere does not enclose its entire source box cell')
        geometry[name] = (row, vertices, pose)
        groups[name] = frozenset(ids)
        sphere_geometry.update((i, (name,)) for i in ids)
    if set(sphere_geometry) != set(range(len(spheres))):
        raise ValueError('Native owner boxes must cover every registered owner sphere')
    return geometry, sphere_geometry, groups


def check_native_owner_material(rule, native):
    """Tie a permission's complete body back to the actual registered solid."""
    _, vertices, pose = native
    body = vertices @ pose[:3, :3].T + pose[:3, 3]
    material = rule.online_material
    if material['kind'] == 'native_box_edge_v1':
        local = np.asarray(list(product((-1., 1.), repeat=3))) * material['half_size_m']
        bound = np.asarray(material['T_actor_box'])
        declared = local @ bound[:3, :3].T + bound[:3, 3]
    else:
        front = np.asarray(rule.feature_points)
        declared = np.concatenate((front, front - material['thickness_m'] * rule.feature_normal))
    tolerance = 256 * np.finfo(float).eps * max(1., float(np.abs(body).max()))
    distances = np.linalg.norm(body[:, None] - declared[None], axis=-1)
    if (len(declared) != len(body) or max(distances.min(0).max(), distances.min(1).max()) > tolerance):
        raise ValueError('Native owner contact material differs from its registered complete solid')
# END material declarations cannot replace or shrink an attached physical body.
