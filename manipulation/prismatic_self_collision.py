"""Prepared native convex self pairs for a rigid, prismatic hand subtree.

Source identity and complete sphere partitions are checked once at binding.
ConvexHull supplies faces/edges once; online queries contain only affine joint
projections. All solids share the TCP rotation, so neither that rotation nor
arm FK occurs in the pair-distance arithmetic.
"""
# BEGIN exact native hand-pair zero sets replace only registered proxy pairs.
from copy import deepcopy
from itertools import combinations, product

import numpy as np
from scipy.spatial import ConvexHull
import torch

class PreparedPrismaticSelfCollision:
    """Frozen separating-axis model, with explicit complete proxy replacement.

    A positive residual is the least overlap over all unit separating axes;
    it is not claimed to be Euclidean penetration depth. Zero means at least
    one projection separates (touching is the mathematical zero boundary).
    No numerical epsilon is subtracted as a physical penetration permission.
    Near-parallel axes are retained; only an exactly zero cross product is
    omitted. Extra triangulation edges are valid axes and need no heuristic
    deduplication. Query arithmetic uses the input floating-point dtype.
    """
    MAX_GEOMETRIES = 32
    MAX_VERTICES = 256
    MAX_AXES = 16384

    def __init__(self, geometry, *, joint_names, sphere_indices, local_spheres,
                 sphere_joint_translation, collision_pairs):
        self.joint_names = tuple(joint_names)
        count = len(self.joint_names)
        if not isinstance(geometry, (tuple, list)) or not 1 <= len(geometry) <= self.MAX_GEOMETRIES:
            raise ValueError('UNSUPPORTED_MODEL: bounded complete native hand geometry is required')
        hand_indices = tuple(sphere_indices)
        sphere_rows = {index: row for row, index in enumerate(hand_indices)}
        enabled = {tuple(sorted(map(int, pair))) for pair in np.asarray(collision_pairs).reshape(-1, 2)}
        rows, seen, ids = [], [], set()
        for record in geometry:
            if set(record) != {'geometry_id', 'sphere_indices', 'vertices', 'joint_translation', 'source'}:
                raise ValueError('Native hand solids require exact geometry, sphere and prismatic source bindings')
            identity, source = record['geometry_id'], record['source']
            selected = record['sphere_indices']
            vertices = np.asarray(record['vertices'], dtype=float)
            mapping = np.asarray(record['joint_translation'], dtype=float)
            if (not isinstance(identity, str) or not identity or identity in ids
                    or not isinstance(source, str) or not source
                    or not isinstance(selected, (tuple, list)) or not selected
                    or any(type(index) is not int or index not in sphere_rows for index in selected)
                    or len(set(selected)) != len(selected)
                    or vertices.ndim != 2 or vertices.shape[1:] != (3,)
                    or not 4 <= len(vertices) <= self.MAX_VERTICES
                    or mapping.shape != (3, count)
                    or not np.isfinite(vertices).all() or not np.isfinite(mapping).all()):
                raise ValueError('UNSUPPORTED_MODEL: finite complete convex prismatic solids are required')
            # The adapter validates metre-position errors over full mechanical
            # limits, then binds canonical source coefficients to both models.
            # A dimensionless coefficient tolerance would mix physical units.
            native_mapping = np.asarray(sphere_joint_translation)[[sphere_rows[i] for i in selected]]
            if not np.array_equal(native_mapping, np.broadcast_to(mapping, native_mapping.shape)):
                raise ValueError('Native hand solids and their complete spheres have different joint translations')
            hull = ConvexHull(vertices)
            edges = np.unique(np.sort(np.concatenate((hull.simplices[:, [0, 1]],
                hull.simplices[:, [1, 2]], hull.simplices[:, [0, 2]])), axis=1), axis=0)
            directions = vertices[edges[:, 1]]-vertices[edges[:, 0]]
            directions /= np.linalg.norm(directions, axis=1)[:, None]
            rows.append(dict(identity=identity, indices=tuple(selected), vertices=vertices,
                             mapping=mapping, normals=hull.equations[:, :3], edges=directions))
            seen.extend(selected)
            ids.add(identity)
        if sorted(seen) != sorted(hand_indices):
            raise ValueError('Native prismatic geometry must cover every declared hand sphere exactly once')
        self.geometry = deepcopy(list(geometry))
        self.pair_geometry_ids, self.replaced_pairs = [], []
        gaps, coefficients, axis_pairs, moving = [], [], [], []
        total_axes = 0
        for first, second in combinations(rows, 2):
            complete = {tuple(sorted(pair)) for pair in product(first['indices'], second['indices'])}
            selected = complete & enabled
            if not selected:
                continue
            if selected != complete:
                raise ValueError('UNSUPPORTED_MODEL: a native solid pair requires its complete enabled sphere cross product')
            # A fixed relative orientation makes this finite SAT axis set
            # complete for every admissible prismatic coordinate, not merely
            # for the posture used during preparation.
            cross = np.cross(first['edges'][:, None], second['edges'][None]).reshape(-1, 3)
            norms = np.linalg.norm(cross, axis=1)
            cross = cross[norms > 0]/norms[norms > 0, None]
            axes = np.concatenate((first['normals'], second['normals'], cross))
            axes /= np.linalg.norm(axes, axis=1)[:, None]
            total_axes += len(axes)
            if total_axes > self.MAX_AXES:
                raise ValueError('UNSUPPORTED_MODEL: prepared prismatic separating-axis capacity exceeded')
            a, b = first['vertices'] @ axes.T, second['vertices'] @ axes.T
            gaps.append(np.column_stack((b.min(0)-a.max(0), a.min(0)-b.max(0))))
            coefficient = axes @ (second['mapping']-first['mapping'])
            coefficients.append(coefficient)
            axis_pairs.extend([len(self.pair_geometry_ids)]*len(axes))
            moving.append(bool(np.any(second['mapping'] != first['mapping'])))
            self.pair_geometry_ids.append((first['identity'], second['identity']))
            self.replaced_pairs.extend(sorted(complete))
        self.gaps = np.concatenate(gaps) if gaps else np.empty((0, 2))
        self.coefficients = np.concatenate(coefficients) if coefficients else np.empty((0, count))
        self.axis_pairs = np.asarray(axis_pairs, dtype=np.int64)
        self.moving_pairs = np.asarray(moving, dtype=bool)
        self.replaced_pairs = tuple(sorted(self.replaced_pairs))
        self.pair_geometry_ids = tuple(self.pair_geometry_ids)
        self._device = {}
        for value in (self.gaps, self.coefficients, self.axis_pairs, self.moving_pairs):
            value.flags.writeable = False

    def filter_pairs(self, pairs, padding):
        """Remove only exact native pairs represented by this prepared model."""
        pairs = np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
        present = {tuple(sorted(map(int, pair))) for pair in pairs}
        replaced = set(self.replaced_pairs)
        if not replaced <= present:
            raise ValueError('Native hand-pair replacement differs from the runtime self-pair policy')
        ids = sorted({index for pair in replaced for index in pair})
        if ids and np.any(np.asarray(padding)[ids] != 0):
            raise ValueError('UNSUPPORTED_MODEL: padded proxy pairs have no exact native-solid replacement contract')
        return [list(map(int, pair)) for pair in pairs if tuple(sorted(map(int, pair))) not in replaced]

    def components(self, state):
        """Return hard and explicitly conditional native-solid pair overlaps."""
        coordinates, half = state.coordinates, state.half
        count = len(self.pair_geometry_ids)
        if not count:
            zero = coordinates.new_zeros(coordinates.shape[:-1])
            return zero, zero
        key = (coordinates.device, coordinates.dtype)
        if key not in self._device:
            # Device tensors own their storage; read-only preparation arrays
            # are never aliased into Torch or copied again by a hot query.
            self._device[key] = (torch.tensor(self.coefficients.copy(), device=coordinates.device, dtype=coordinates.dtype),
                torch.tensor(self.gaps.copy(), device=coordinates.device, dtype=coordinates.dtype),
                torch.tensor(self.axis_pairs.copy(), device=coordinates.device, dtype=torch.long),
                torch.tensor(self.moving_pairs.copy(), device=coordinates.device, dtype=torch.bool))
        coefficient, gaps, index, moving = self._device[key]
        shift = coordinates @ coefficient.T
        spread = half @ coefficient.abs().T
        axis_separation = torch.maximum(gaps[:, 0]+shift, gaps[:, 1]-shift)-spread
        separation = coordinates.new_full((*coordinates.shape[:-1], count), -torch.inf)
        separation = separation.scatter_reduce(-1, index.expand_as(axis_separation),
                                                axis_separation, reduce='amax', include_self=True)
        residual = torch.relu(-separation)
        conditional = state.uncertain[..., None] & moving
        return torch.where(conditional, 0., residual).sum(-1), torch.where(conditional, residual, 0.).sum(-1)

    @property
    def storage_nbytes(self):
        return sum(value.nbytes for value in (self.gaps, self.coefficients, self.axis_pairs, self.moving_pairs))

    def record(self):
        return {'kind': 'prepared_prismatic_self_v1', 'geometry': deepcopy(self.geometry),
                'pair_geometry_ids': [list(pair) for pair in self.pair_geometry_ids],
                'replaced_pairs': [list(pair) for pair in self.replaced_pairs],
                'gaps': self.gaps.tolist(), 'coefficients': self.coefficients.tolist(),
                'axis_pairs': self.axis_pairs.tolist(), 'moving_pairs': self.moving_pairs.tolist()}

    def validate_record(self, record):
        # Deserialization reconstructs only the small, supplied convex records
        # once outside online queries, then checks every frozen coefficient.
        # No external source asset or acceptance-time mesh audit is consulted.
        if record != self.record():
            raise ValueError('Serialized prismatic model differs from its complete native geometry binding')
# END all other self pairs and all external collisions remain untouched.
