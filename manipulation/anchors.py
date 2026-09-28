"""Bind optional semantic geometry to fixed rigid-body frames.

An anchor belongs to exactly one owner. Its coordinates are measured once in
the same external reference as that owner's poses, then retained in the owner's
fixed body frame. This module does not track images, infer contact persistence,
or decide which entity owns a semantic point. In particular, a sliding contact
must not be declared a fixed material point without independent evidence.

Points use metres. Directions are nonzero free vectors: translation does not
act on them, and their supplied length is preserved rather than normalized.
"""

# BEGIN owned fixed-anchor binding and propagation.
from collections.abc import Mapping, Sequence

import numpy as np

from .motion import _poses_array


def _owners_and_frames(owner_poses, frame_indices):
    """Validate one shared frame sequence and each owner's proper SE(3) poses.

    Array rows are synchronized by the supplied original frame IDs. A numerical
    pose check cannot establish tracking accuracy or detect a caller mixing
    camera, world and reference coordinates; a common reference is a contract.
    """
    indices = np.asarray(frame_indices)
    if (indices.ndim != 1 or not len(indices) or indices.dtype.kind not in "iu"
            or (indices < 0).any() or (indices[1:] <= indices[:-1]).any()
            or any(isinstance(value, (bool, np.bool_))
                   for value in np.asarray(frame_indices, dtype=object))):
        raise ValueError("frame_indices must be nonempty, nonnegative strictly increasing integers")
    if not isinstance(owner_poses, Mapping):
        raise ValueError("owner_poses must map owner names to pose sequences")
    poses = {}
    for owner, sequence in owner_poses.items():
        if not isinstance(owner, str) or not owner.strip():
            raise ValueError("Each pose owner must have a nonempty string name")
        poses[owner] = _poses_array(sequence, f"owner_poses[{owner!r}]")
        if len(poses[owner]) != len(indices):
            raise ValueError(f"Owner {owner!r} poses must match the shared frame_indices")
    return poses, indices.copy()


def _records(anchors, owners, value_key):
    """Check explicit ownership and geometry without inventing missing anchors."""
    if not isinstance(anchors, Sequence) or isinstance(anchors, (str, bytes)):
        raise ValueError("Anchors must be a sequence of records; an empty sequence is valid")
    expected = {"name", "owner", "kind", "frame_index", value_key}
    names = set()
    result = []
    for anchor in anchors:
        if not isinstance(anchor, Mapping) or set(anchor) != expected:
            raise ValueError(f"Each anchor must contain exactly {sorted(expected)}")
        name, owner = anchor["name"], anchor["owner"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Each anchor name must be a nonempty string")
        if not isinstance(owner, str) or owner not in owners:
            raise ValueError(f"Unknown anchor owner: {owner!r}")
        identity = (owner, name)
        if identity in names:
            raise ValueError(f"Duplicate anchor name {name!r} for owner {owner!r}")
        names.add(identity)
        kind, index = anchor["kind"], anchor["frame_index"]
        if kind not in ("point", "direction"):
            raise ValueError("Anchor kind must be 'point' or 'direction'")
        if (isinstance(index, (bool, np.bool_))
                or not isinstance(index, (int, np.integer)) or index < 0):
            raise ValueError("Anchor frame_index must be a nonnegative integer frame ID")
        value = np.array(anchor[value_key], dtype=float, copy=True)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Anchor {value_key} must be a finite three-vector")
        if kind == "direction" and not np.any(value != 0):
            raise ValueError("A direction anchor must be nonzero")
        result.append({"name": name, "owner": owner, "kind": kind,
                       "frame_index": int(index), value_key: value})
    return result


def bind_anchors(anchors, owner_poses, frame_indices):
    """Convert 0..N observed anchors into fixed owner-local coordinates once.

    ``owner_poses`` maps owner IDs to arrays of shape ``[T,4,4]``. Every pose
    maps that owner's *same fixed body frame* into one common reference frame.
    ``frame_indices`` gives the original nonnegative frame ID at every row.
    IDs may start above zero and skip frames; they are never array offsets.

    Each input record contains exactly ``name``, ``owner``, ``kind`` (``point``
    or ``direction``), ``frame_index`` and ``value`` (a three-vector in the common
    reference at that frame). The observation frame must be present, but need
    not be the first frame. Returned records replace ``value`` with
    ``local_value`` and retain the original observation frame as provenance.

    For owner pose ``T_k = [R_k, t_k]``, bind a point as
    ``p_body = R_k.T @ (p_k - t_k)`` and a direction as ``d_body = R_k.T @ d_k``.
    No identity-first-pose assumption, temporal integration or depth resampling
    enters this operation. No owner trajectory is needed for an empty list.
    """
    poses, indices = _owners_and_frames(owner_poses, frame_indices)
    records = _records(anchors, poses, "value")
    rows = {int(index): row for row, index in enumerate(indices)}
    result = []
    for anchor in records:
        index = anchor["frame_index"]
        if index not in rows:
            raise ValueError(f"Anchor frame_index {index} is absent from frame_indices")
        pose = poses[anchor["owner"]][rows[index]]
        value = anchor.pop("value")
        if anchor["kind"] == "point":
            value = value - pose[:3, 3]
        anchor["local_value"] = pose[:3, :3].T @ value
        if not np.isfinite(anchor["local_value"]).all():
            raise ValueError("Anchor binding produced non-finite coordinates")
        result.append(anchor)
    return result


def propagate_anchors(bindings, owner_poses, frame_indices):
    """Evaluate previously bound anchors along synchronized owner trajectories.

    ``bindings`` is the result of :func:`bind_anchors`. The owner body frames
    must retain their binding-time definitions. A new external reference or
    a later frame subsequence is allowed: binding frame IDs are historical
    metadata and need not occur in this output's ``frame_indices``.

    Return ``{"frame_indices": [T], "anchors": [...]}``. Each anchor preserves
    its binding record and adds ``values`` with shape ``[T,3]`` in the external
    reference. Points follow ``R_t @ p_body + t_t``; directions follow
    ``R_t @ d_body``. The input bindings are never modified or rebound.
    """
    poses, indices = _owners_and_frames(owner_poses, frame_indices)
    records = _records(bindings, poses, "local_value")
    for anchor in records:
        owner = poses[anchor["owner"]]
        values = owner[:, :3, :3] @ anchor["local_value"]
        if anchor["kind"] == "point":
            values = values + owner[:, :3, 3]
        if not np.isfinite(values).all():
            raise ValueError("Anchor propagation produced non-finite coordinates")
        anchor["values"] = values
    return {"frame_indices": indices, "anchors": records}
# END owned fixed-anchor binding and propagation.
