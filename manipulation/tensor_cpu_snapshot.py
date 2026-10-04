"""One exact, masked CPU snapshot of aligned dense tensor fields.

This is a data-movement boundary, not a numerical implementation of a tensor
operation. Every field retains its dtype and stored bits. A shared leading
boolean mask selects complete rows once, before any row reaches the CPU.
"""

# BEGIN a single device transfer preserves independent field representations.
import math

import numpy as np
import torch


_NUMPY_DTYPES = {
    torch.bool: np.dtype(np.bool_),
    torch.uint8: np.dtype(np.uint8),
    torch.int8: np.dtype(np.int8),
    torch.int16: np.dtype(np.int16),
    torch.int32: np.dtype(np.int32),
    torch.int64: np.dtype(np.int64),
    torch.float16: np.dtype(np.float16),
    torch.float32: np.dtype(np.float32),
    torch.float64: np.dtype(np.float64),
    torch.complex64: np.dtype(np.complex64),
    torch.complex128: np.dtype(np.complex128),
}


def masked_cpu_snapshot(tensors, valid):
    """Copy ``tensor.detach()[valid]`` fields together without numeric casts.

    All fields share a device and the mask's leading dimensions; their tail
    shapes and dtypes may differ. Byte concatenation performs no arithmetic,
    including for signed zeros and NaN payloads. Each returned NumPy array
    owns only its field's selected bytes, not a view retaining the whole pack.

    Let P be the unmasked payload size and K the selected payload size, K<=P.
    Contiguous source copies plus the pack use at most 2P temporary bytes.
    The masked gather uses P+K plus its row indices; discard P before the CPU
    copy. At unpacking, discard the device selection before producing K field
    bytes from K host bytes. Thus peak extra payload is at most 2P, rather than
    retaining all three full-size representations together. The scene caller
    admits one additional P plus mask indices beyond its existing P allowance.
    """
    tensors = tuple(tensors)
    if not isinstance(valid, torch.Tensor) or valid.dtype != torch.bool:
        raise ValueError("A CPU snapshot needs one boolean leading-dimension mask")
    if not tensors:
        return ()
    device = tensors[0].device
    leading, rows = tuple(valid.shape), valid.numel()
    metadata = []
    for value in tensors:
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.device != device or tuple(value.shape[:valid.ndim]) != leading):
            raise ValueError("CPU snapshot fields must share a device and the mask's leading dimensions")
        if value.dtype not in _NUMPY_DTYPES:
            raise TypeError("CPU snapshot field dtype has no supported NumPy representation")
        tail = tuple(value.shape[valid.ndim:])
        width = math.prod(tail)
        metadata.append((tail, width, value.element_size(), _NUMPY_DTYPES[value.dtype]))
    # One contiguous copy per field also materializes logical conjugate or
    # negative views, as the original masked tensor copy does. The temporary
    # tuple is released as soon as cat returns; no source copies survive it.
    packed = torch.cat(tuple(value.detach().clone(memory_format=torch.contiguous_format)
        .reshape(rows, width).view(torch.uint8)
        for value, (_, width, _, _) in zip(tensors, metadata)), dim=1)
    selected = packed[valid.reshape(-1)]
    del packed
    host = selected.cpu().numpy()
    del selected
    count, offset, result = len(host), 0, []
    for tail, width, element_size, dtype in metadata:
        end = offset+width*element_size
        # Copy each byte field before viewing its dtype. Row packing may put
        # it at an unaligned offset; the owned copy restores normal contiguous
        # layout and does not keep the shared host pack alive after return.
        field = host[:, offset:end].copy().view(dtype).reshape((count, *tail))
        result.append(field)
        offset = end
    return tuple(result)
# END filtering and field order match the original independent masked copies.
