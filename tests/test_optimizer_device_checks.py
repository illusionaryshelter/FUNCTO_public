"""Fault isolation and exact values for the optimizer's deferred checks."""

# BEGIN exercise eager/public errors and the distinct optimizer block boundary.
from unittest.mock import patch

import pytest
import torch
import numpy as np

from manipulation.gripper_trajectory import FingerTrajectory
from manipulation.optimizer_device_checks import (
    DeviceChecks, require_tensor, constant_tensor, _host_constant_key,
)
from manipulation.voxel_layers import FineRegion
from test_gripper_trajectory import _arguments
from test_voxel_layers import GROUPS, region


def same_bits(first, second):
    assert first.dtype == second.dtype and first.shape == second.shape
    assert torch.equal(first.detach().contiguous().view(torch.uint8),
                       second.detach().contiguous().view(torch.uint8))


def test_eager_exceptions_and_nested_scopes_remain_independent():
    first, second = DeviceChecks("cpu"), DeviceChecks("cpu")
    with pytest.raises(ValueError, match="outside"):
        require_tensor(torch.tensor(False), "outside")
    assert require_tensor(torch.tensor(True), "valid") is None
    with first.scope():
        require_tensor(torch.tensor(False), "first", error_type=RuntimeError)
        with second.scope():
            require_tensor(torch.tensor(True), "second")
        require_tensor(torch.tensor(True), "first", error_type=RuntimeError)
    second.raise_if_failed()
    with pytest.raises(RuntimeError, match="first"):
        first.raise_if_failed()
    first.reset()
    first.raise_if_failed()
    assert first.storage_bytes == 8
    with pytest.raises(ValueError, match="outside again"):
        require_tensor(torch.tensor(False), "outside again")


def test_scoped_constants_keep_original_casts_values_and_tensor_gradients():
    checks = DeviceChecks("cpu")
    like = torch.empty((), dtype=torch.float32)
    hosts = ([1., -0., np.nextafter(1., 2.)], np.array([1., -0., 1e-40]),
             np.array([2**54+1, -3], dtype=np.int64), np.array([False, True]))
    for value in hosts:
        expected = torch.as_tensor(value, dtype=like.dtype)
        with checks.scope():
            actual = constant_tensor(value, like)
            same_bits(constant_tensor(value, like), actual)
        same_bits(expected, actual)
    value = torch.tensor([2.], requires_grad=True)
    with checks.scope():
        copied = constant_tensor(value, like)
    assert copied is value
    copied.square().sum().backward()
    assert value.grad.item() == 4.
    checks.clear_constants()
    assert checks.storage_bytes == 8


def test_constant_keys_do_not_round_mixed_lists_or_merge_distinct_layouts():
    checks = DeviceChecks("cpu")
    like = torch.empty((), dtype=torch.int64)
    assert _host_constant_key([2**53, 1.], like.dtype) != _host_constant_key([2**53+1, 1.], like.dtype)
    with checks.scope():
        a = constant_tensor([2**53, 1.], like)
        b = constant_tensor([2**53+1, 1.], like)
        assert a[0].item() != b[0].item()
        source = np.arange(12., dtype=np.float64).reshape(3, 4)
        assert _host_constant_key(source, like.dtype) != _host_constant_key(np.asfortranarray(source), like.dtype)
        first = constant_tensor(source, like, dtype=torch.float64)
        second = constant_tensor(np.asfortranarray(source), like, dtype=torch.float64)
    assert first.stride() != second.stride()
    same_bits(first, second)
    # CPU ndarray aliases are never retained in a content-addressed cache.
    source = np.array([1.])
    with checks.scope():
        constant_tensor(source, like, dtype=torch.float64)
        source[0] = 2.
        fresh = constant_tensor(np.array([1.]), like, dtype=torch.float64)
    assert fresh.item() == 1.


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_finger_cache_preserves_values_gradients_and_device_storage(dtype):
    model = FingerTrajectory.build(**_arguments())
    times = torch.tensor([[0., .151, .302, .551, 1.]], dtype=dtype, requires_grad=True)
    eager = model.sample_torch(times)
    gradient = torch.autograd.grad(eager[0].sum(), times)[0]
    cache = model._torch_cache[(times.device, dtype)]
    checks = DeviceChecks("cpu")
    # Once warmed, sampling must not rebuild either host-backed constant.
    with patch.object(torch, "as_tensor", side_effect=AssertionError("unexpected constant copy")):
        with checks.scope():
            actual = model.sample_torch(times)
            actual_gradient = torch.autograd.grad(actual[0].sum(), times)[0]
    checks.raise_if_failed()
    for expected, result in zip((*eager, gradient), (*actual, actual_gradient)):
        same_bits(expected, result)
    assert all(a is b for a, b in zip(cache, model._torch_cache[(times.device, dtype)]))
    # The two precision-specific copies never evict an address held by a graph.
    other = torch.float64 if dtype == torch.float32 else torch.float32
    model.sample_torch(times.detach().to(other))
    assert all(a is b for a, b in zip(cache, model._torch_cache[(times.device, dtype)]))
    stored = sum(t.numel()*t.element_size() for values in model._torch_cache.values() for t in values)
    assert stored <= model.torch_storage_bytes
    assert FingerTrajectory.from_mapping(model.describe()).describe() == model.describe()


@pytest.mark.parametrize("invalid", (float("nan"), float("inf"), -.01, 1.01))
def test_invalid_finger_times_raise_eagerly_or_fail_the_complete_block(invalid):
    model = FingerTrajectory.build(**_arguments())
    times = torch.tensor([[invalid]])
    with pytest.raises(ValueError, match="finite trajectory duration"):
        model.sample_torch(times)
    checks = DeviceChecks("cpu")
    with checks.scope():
        values = model.sample_torch(times)
    assert all(torch.isfinite(value).all() for value in values)
    with pytest.raises(ValueError, match="finite trajectory duration"):
        checks.raise_if_failed()


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_layer_selection_preserves_every_bound_index_and_choice(dtype):
    fine = FineRegion(region())
    spheres = torch.tensor([[[[0., 0., 0., .01], [.074, 0., 0., .01],
                               [-0., .02, 0., .01]]]], dtype=dtype)
    checks = DeviceChecks("cpu")
    for direction in (-torch.inf, torch.inf):
        candidate = spheres.clone()
        candidate[..., 1, 0] = torch.nextafter(candidate[..., 1, 0], spheres.new_tensor(direction))
        expected = fine.select(candidate, GROUPS)
        with checks.scope():
            actual = fine.select(candidate, GROUPS)
        checks.raise_if_failed()
        for field in ("sphere_fine", "body_fine", "influence_aabb", "index_lower", "index_upper"):
            same_bits(getattr(expected, field), getattr(actual, field))


@pytest.mark.parametrize("column,value", ((0, float("nan")), (1, float("inf")), (3, -.01)))
def test_invalid_spheres_never_reach_unsafe_prefix_indices(column, value):
    fine = FineRegion(region())
    spheres = torch.tensor([[[[0., 0., 0., .01], [.03, 0., 0., .01], [0., .02, 0., .01]]]])
    spheres[..., 0, column] = value
    with pytest.raises(ValueError, match="body-cover spheres"):
        fine.select(spheres, GROUPS)
    checks = DeviceChecks("cpu")
    with checks.scope():
        selected = fine.select(spheres, GROUPS)
    assert torch.isfinite(selected.influence_aabb).all()
    with pytest.raises(ValueError, match="body-cover spheres"):
        checks.raise_if_failed()
# END no allclose tolerance substitutes for exact valid-input comparisons.
