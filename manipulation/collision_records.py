"""Bounded, typed collision-query records without per-element Python objects.

The codec preserves the exact stored values. Geometry consumers remain
responsible for their physical and finite-value contracts. Decoding never
accepts a previous nested-list representation or infers a missing dtype.
"""

# BEGIN bounded binary records for the actual collision-layer decisions.
import base64
import binascii
import math

import numpy as np
import torch


MAX_ARRAY_BYTES = 32 * 1024**2
_MAX_ARRAY_DIMENSIONS = 5
_DTYPES = {name: np.dtype(name) for name in ("|b1", "<i8", "<f4", "<f8")}
_ARRAY_KEYS = {"shape", "dtype", "data_base64"}
_LAYER_KEYS = {"body_ids", "body_fine", "influence_aabb_grid_m", "index_lower", "index_upper"}


def _array_layout(shape, dtype):
    """Check integer dimensions and byte size before any payload allocation."""
    if (not isinstance(shape, list) or len(shape) > _MAX_ARRAY_DIMENSIONS
            or any(type(value) is not int or value < 0 or value > MAX_ARRAY_BYTES for value in shape)
            or not isinstance(dtype, str) or dtype not in _DTYPES):
        raise ValueError("Collision arrays need at most five bounded integer dimensions and an explicit supported little-endian dtype")
    # Python integers cannot wrap a huge product into an admitted byte count.
    # Bound individual dimensions too, including shapes with a zero axis.
    byte_count = math.prod(shape) * _DTYPES[dtype].itemsize
    if byte_count > MAX_ARRAY_BYTES:
        raise ValueError("One collision record array cannot exceed 32 MiB")
    return _DTYPES[dtype], byte_count


def encode_array(value):
    """Encode a NumPy array or an already detached CPU Torch tensor.

    Payload length is exactly ``4 * ceil(raw_bytes / 3)`` ASCII bytes, where
    ``raw_bytes = prod(shape) * dtype.itemsize``. Shape and dtype metadata add
    only a small per-array cost. GPU transfer and graph detachment must occur
    explicitly at the recording call site, before using this CPU codec.
    """
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or value.requires_grad:
            raise ValueError("Collision record tensors must already be detached and on CPU")
        try:
            value = value.numpy()
        except (TypeError, RuntimeError) as error:
            raise ValueError("Collision record tensor dtype or layout is unsupported") from error
    if not isinstance(value, np.ndarray):
        raise ValueError("Collision records require an explicit array, not nested lists")
    shape = list(value.shape)
    dtype = value.dtype.newbyteorder("<").str
    target_dtype, _ = _array_layout(shape, dtype)
    # astype converts a big-endian input numerically; tobytes writes a dense
    # C-order payload even for a strided view. Admission precedes both copies.
    raw = value.astype(target_dtype, copy=False).tobytes(order="C")
    return {"shape": shape, "dtype": dtype,
            "data_base64": base64.b64encode(raw).decode("ascii")}


def decode_array(record):
    """Decode a strictly bounded record as a read-only NumPy array.

    The array shares its immutable decoded bytes, avoiding an additional full
    payload copy. A caller intentionally editing values must request copy().
    Exact encoded length is checked before base64 decoding, then exact raw
    length before reshape. No metadata-driven large allocation is performed.
    """
    if not isinstance(record, dict) or set(record) != _ARRAY_KEYS:
        raise ValueError("Collision arrays require exactly shape, dtype and data_base64")
    dtype, byte_count = _array_layout(record["shape"], record["dtype"])
    encoded = record["data_base64"]
    if not isinstance(encoded, str) or len(encoded) != 4*((byte_count+2)//3):
        raise ValueError("Collision array base64 length differs from its declared shape and dtype")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("Collision array payload is not valid standard base64") from error
    if len(raw) != byte_count:
        raise ValueError("Decoded collision array byte length differs from its declared shape and dtype")
    return np.frombuffer(raw, dtype=dtype).reshape(record["shape"])


def slice_layer_selection(records, index=(0, 0)):
    """Retain one [batch, time] state for each actor/target layer record.

    Input and output map actor/target names to LayerSelection.record() values.
    Singleton batch/time axes are preserved, so independent geometry checking
    uses the same schema for a full bounded chunk and an archived single state.
    Each array is decoded separately; only its small selected slice is encoded
    into the output. No full-horizon decision record is copied into each state.
    """
    if (not isinstance(records, dict) or not isinstance(index, tuple) or len(index) != 2
            or any(type(value) is not int or value < 0 for value in index)):
        raise ValueError("Layer selection slicing needs a target mapping and nonnegative integer (batch,time)")
    batch, time = index
    result = {}
    for target, record in records.items():
        if (not isinstance(target, str) or not target or not isinstance(record, dict)
                or set(record) != _LAYER_KEYS):
            raise ValueError("Each target needs one complete typed layer-selection record")
        body_ids = record["body_ids"]
        if (not isinstance(body_ids, list) or not body_ids
                or any(not isinstance(value, str) or not value for value in body_ids)
                or len(set(body_ids)) != len(body_ids)):
            raise ValueError("Layer selection body_ids must be unique nonempty strings")
        fine = decode_array(record["body_fine"])
        if (fine.dtype != np.dtype(bool) or fine.ndim != 3 or fine.shape[-1] != len(body_ids)
                or batch >= fine.shape[0] or time >= fine.shape[1]):
            raise ValueError("Layer selection must provide the requested state for every body")
        state_shape = fine.shape
        sliced = {"body_ids": list(body_ids),
                  "body_fine": encode_array(fine[batch:batch+1, time:time+1])}
        del fine
        for name, suffix, kind in (("influence_aabb_grid_m", (2, 3), "f"),
                                   ("index_lower", (3,), "i"), ("index_upper", (3,), "i")):
            array = decode_array(record[name])
            if array.shape != (*state_shape, *suffix) or array.dtype.kind != kind:
                raise ValueError("Layer selection envelope and indices must align with every body/state")
            sliced[name] = encode_array(array[batch:batch+1, time:time+1])
            del array
        result[target] = sliced
    return result
# END exact typed records remain finite before encoding, decoding or state slicing.
