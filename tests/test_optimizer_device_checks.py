"""Eager input validation and reusable polynomial tensor storage."""

# BEGIN invalid inputs fail before their first indexing or curve evaluation.
from unittest.mock import patch

import pytest
import torch

from manipulation.gripper_trajectory import FingerTrajectory
from manipulation.voxel_layers import FineRegion
from test_gripper_trajectory import _arguments
from test_voxel_layers import GROUPS, region


def same_bits(first, second):
    assert first.dtype == second.dtype and first.shape == second.shape
    assert torch.equal(first.detach().contiguous().view(torch.uint8),
                       second.detach().contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_finger_cache_preserves_values_gradients_and_device_storage(dtype):
    model = FingerTrajectory.build(**_arguments())
    times = torch.tensor([[0., .151, .302, .551, 1.]], dtype=dtype, requires_grad=True)
    eager = model.sample_torch(times)
    gradient = torch.autograd.grad(eager[0].sum(), times)[0]
    cache = model._torch_cache[(times.device, dtype)]
    # Once warmed, sampling must not rebuild either host-backed constant.
    with patch.object(torch, "as_tensor", side_effect=AssertionError("unexpected constant copy")):
        actual = model.sample_torch(times)
        actual_gradient = torch.autograd.grad(actual[0].sum(), times)[0]
    for expected, result in zip((*eager, gradient), (*actual, actual_gradient)):
        same_bits(expected, result)
    assert all(a is b for a, b in zip(cache, model._torch_cache[(times.device, dtype)]))
    # Both precision-specific copies remain available to alternating queries.
    other = torch.float64 if dtype == torch.float32 else torch.float32
    model.sample_torch(times.detach().to(other))
    assert all(a is b for a, b in zip(cache, model._torch_cache[(times.device, dtype)]))
    stored = sum(t.numel()*t.element_size() for values in model._torch_cache.values() for t in values)
    assert stored <= model.torch_storage_bytes
    assert FingerTrajectory.from_mapping(model.describe()).describe() == model.describe()


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -.01, 1.01))
def test_invalid_finger_times_raise_before_polynomial_indexing(invalid):
    model = FingerTrajectory.build(**_arguments())
    times = torch.tensor([[invalid]])
    with patch("torch.searchsorted", side_effect=AssertionError("invalid time reached indexing")):
        with pytest.raises(ValueError, match="finite trajectory duration"):
            model.sample_torch(times)

@pytest.mark.parametrize("column,value", ((0, float("nan")), (1, float("inf")), (3, -.01)))
def test_invalid_spheres_never_reach_unsafe_prefix_indices(column, value):
    fine = FineRegion(region())
    spheres = torch.tensor([[[[0., 0., 0., .01], [.03, 0., 0., .01], [0., .02, 0., .01]]]])
    spheres[..., 0, column] = value
    with patch.object(fine, "_box_count", side_effect=AssertionError("invalid sphere reached indexing")):
        with pytest.raises(ValueError, match="body-cover spheres"):
            fine.select(spheres, GROUPS)
# END valid polynomial queries retain their existing values and derivatives.
