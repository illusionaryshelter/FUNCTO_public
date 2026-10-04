"""Bitwise data-movement checks for the source geometry CPU boundary."""

# BEGIN field packing must not become a floating-point conversion.
from unittest.mock import patch

import numpy as np
import pytest
import torch

from manipulation.tensor_cpu_snapshot import masked_cpu_snapshot


def mixed_fields():
    """Include raw NaN payloads, signed zeros and integers beyond FP53."""
    words32 = np.array([0x80000000, 0x00000000, 0x7fc01234, 0xff801234,
                        0x00000001, 0x80000001, 0x7f800000, 0xff800000,
                        0x3f800001, 0xbf800001, 0x7f7fffff, 0xff7fffff], dtype=np.uint32)
    words64 = np.array([0x8000000000000000, 0, 0x7ff8000000001234, 0xfff0000000001234,
                        1, 0x8000000000000001, 0x7ff0000000000000, 0xfff0000000000000,
                        0x3ff0000000000001, 0xbff0000000000001,
                        0x7fefffffffffffff, 0xffefffffffffffff], dtype=np.uint64)
    values32 = torch.from_numpy(words32.view(np.float32)).reshape(2, 3, 2)
    values64 = torch.from_numpy(words64.view(np.float64)).reshape(2, 3, 2)
    integers = torch.tensor([-(2**63), 2**63-1, 2**53+1, -(2**53+1), 0, -1]).reshape(2, 3)
    flags = torch.tensor([[True, False, True], [False, True, False]])
    empty = torch.empty((2, 3, 0, 4), dtype=torch.float32)
    half = torch.arange(24, dtype=torch.float16).reshape(2, 3, 2, 2)
    return values32, values64, integers, flags, empty, half


def independent_reference(fields, valid):
    return tuple(value.detach()[valid].cpu().numpy() for value in fields)


def assert_same_fields(actual, expected):
    assert len(actual) == len(expected)
    for value, reference in zip(actual, expected):
        assert value.dtype == reference.dtype
        assert value.shape == reference.shape
        assert value.tobytes() == reference.tobytes()
        assert value.flags.c_contiguous


@pytest.mark.parametrize("selection", ("all", "mixed", "none"))
@pytest.mark.parametrize("strided", (False, True))
def test_dtypes_tail_shapes_order_and_special_bits(selection, strided):
    fields = mixed_fields()
    valid = torch.tensor([[True, False, True], [False, True, False]])
    if selection != "mixed":
        valid.fill_(selection == "all")
    if strided:
        fields = tuple(value.transpose(0, 1) for value in fields)
        valid = valid.t()
    expected = independent_reference(fields, valid)
    assert_same_fields(masked_cpu_snapshot(fields, valid), expected)


def test_strided_tail_and_logical_views_retain_original_masked_values():
    base = torch.arange(48, dtype=torch.float64).reshape(2, 3, 8)
    complex_base = torch.complex(base, -base)
    fields = (base[..., ::2], base._neg_view(), complex_base.conj())
    valid = torch.tensor([[True, False, True], [True, True, False]])
    assert_same_fields(masked_cpu_snapshot(fields, valid), independent_reference(fields, valid))


@pytest.mark.parametrize("shape", ((0, 3), (2, 0), (0,)))
def test_zero_states_keep_every_typed_tail(shape):
    valid = torch.empty(shape, dtype=torch.bool)
    fields = (torch.empty((*shape, 4), dtype=torch.float32),
              torch.empty(shape, dtype=torch.int64),
              torch.empty((*shape, 2, 0), dtype=torch.float64))
    assert_same_fields(masked_cpu_snapshot(fields, valid), independent_reference(fields, valid))


def test_one_mask_gather_and_cpu_call_without_scalar_reads():
    fields = mixed_fields()
    valid = torch.tensor([[True, False, True], [True, True, False]])
    expected = independent_reference(fields, valid)
    original_getitem, original_cpu = torch.Tensor.__getitem__, torch.Tensor.cpu
    gathers, transfers = [], []

    def getitem(value, index):
        if isinstance(index, torch.Tensor) and index.dtype == torch.bool:
            gathers.append((value.dtype, tuple(index.shape)))
        return original_getitem(value, index)

    def cpu(value, *args, **kwargs):
        transfers.append(value.dtype)
        return original_cpu(value, *args, **kwargs)

    with patch.object(torch.Tensor, "__getitem__", getitem), \
            patch.object(torch.Tensor, "cpu", cpu), \
            patch.object(torch.Tensor, "item", side_effect=AssertionError("device scalar read")), \
            patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("device boolean read")):
        actual = masked_cpu_snapshot(fields, valid)
    assert gathers == [(torch.uint8, (6,))]
    assert transfers == [torch.uint8]
    assert_same_fields(actual, expected)


def test_unpacked_fields_do_not_retain_shared_pack_or_alias_inputs():
    fields = (torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_(),
              torch.arange(6, dtype=torch.int64).reshape(2, 3),
              torch.ones((2, 3, 1), dtype=torch.bool))
    valid = torch.ones((2, 3), dtype=torch.bool)
    expected = independent_reference(fields, valid)
    actual = masked_cpu_snapshot(fields, valid)
    for index, value in enumerate(actual):
        base = value
        while isinstance(base.base, np.ndarray):
            base = base.base
        assert base.flags.owndata and base.nbytes == value.nbytes
        for other in actual[index+1:]:
            assert not np.shares_memory(value, other)
    actual[0][0, 0] = -25.
    assert_same_fields(independent_reference(fields, valid), expected)
    assert fields[0].grad is None


def test_invalid_schema_fails_before_device_transfer():
    with patch.object(torch.Tensor, "cpu", side_effect=AssertionError("unexpected transfer")):
        with pytest.raises(ValueError, match="boolean"):
            masked_cpu_snapshot((torch.zeros((2, 3)),), torch.zeros((2, 3)))
        with pytest.raises(ValueError, match="leading dimensions"):
            masked_cpu_snapshot((torch.zeros((2, 3)),), torch.ones((3, 2), dtype=torch.bool))
        with pytest.raises(TypeError, match="NumPy"):
            masked_cpu_snapshot((torch.zeros((2, 3), dtype=torch.bfloat16),),
                                torch.ones((2, 3), dtype=torch.bool))
# END the native query still receives separate arrays in its original order.
