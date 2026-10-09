"""Checkpoint execution must preserve the full discrete objective and VJP."""

# BEGIN execution-only memory changes preserve all mathematical derivatives.
import pytest
import torch

import manipulation.collision_path as collision_path
from manipulation.collision_path import SphereCollisionPath


def streams(*, residual_grad=True, centre_grad=True):
    """Two distinct sphere inventories expose stale closures during recompute."""
    generator = torch.Generator().manual_seed(413)
    result = {}
    for name, count in (("robot/background", 5), ("payload/owner", 7)):
        residual = .2 + torch.rand(2, 6, count, generator=generator, dtype=torch.float64)
        centres = torch.randn(2, 6, count, 3, generator=generator, dtype=torch.float64)
        # Equal maxima test the original amax subgradient; an inactive NaN
        # column must contribute neither a value nor an input derivative.
        residual[:, 0] = 2.
        residual[:, 2] = 2.
        residual[:, 5] = float("nan")
        centres[:, 5] = float("nan")
        centres[:, 4] = centres[:, 0]
        residual.requires_grad_(residual_grad)
        centres.requires_grad_(centre_grad)
        broadcast_mode = torch.tensor([[[0], [0], [0], [1], [1], [9]],
                                       [[1], [0], [1], [0], [0], [9]]], dtype=torch.bool)
        sphere_mode = torch.zeros(2, 6, count, dtype=torch.long)
        sphere_mode[:, 4, ::2] = 1
        result[name] = SphereCollisionPath(residual, centres,
            torch.linspace(.2, .8, count, dtype=torch.float64), (broadcast_mode, sphere_mode))
    times = torch.tensor([[1., 0., .5, .5, 2., float("nan")],
                          [2., .5, 1., 0., .5, float("nan")]], dtype=torch.float64)
    valid = torch.tensor([[True]*5+[False]]).expand(2, -1)
    order = torch.tensor([[1, 2, 3, 0, 4, 5], [3, 1, 4, 2, 0, 5]])
    return result, times, valid, order


def complete_formula(paths, times, valid, order):
    """Unchunked reference formula retains all spheres for comparison only."""
    def ordered(value):
        indices = order.reshape(*order.shape, *((1,)*(value.ndim-2))).expand_as(value)
        return value.gather(1, indices)
    times, valid = ordered(times), ordered(valid)
    edge = (valid[:, :-1] & valid[:, 1:] & (times[:, 1:] > times[:, :-1]))[..., None]
    values = []
    for path in paths.values():
        residual = torch.where(valid[..., None], ordered(path.residual), 0.)
        centres = ordered(path.centres)
        connected = edge.expand_as(residual[:, :-1])
        for key in path.mode_keys:
            key = ordered(key)
            connected = connected & (key[:, 1:] == key[:, :-1])
        first = torch.where(connected[..., None], centres[:, :-1], 0.)
        second = torch.where(connected[..., None], centres[:, 1:], 0.)
        costs = torch.where(connected, (residual[:, :-1]+residual[:, 1:])*.5, 0.)
        integral = (costs*torch.linalg.vector_norm(second-first, dim=-1)).sum(1)
        values.append((residual.amax(1)+integral/path.radii).sum(-1))
    return torch.stack(values).sum(0)


@pytest.mark.parametrize("residual_grad,centre_grad", [(True, True), (True, False), (False, True)])
def test_multiple_checkpoint_chunks_preserve_value_and_every_input_gradient(
        monkeypatch, residual_grad, centre_grad):
    # Twelve time/batch states under this small budget permit only two
    # spheres per block, so both unequal inventories require several blocks.
    monkeypatch.setattr(collision_path, "_COLLISION_PATH_SAMPLE_ELEMENTS", 32)
    paths, times, valid, order = streams(residual_grad=residual_grad, centre_grad=centre_grad)
    assert collision_path._collision_path_chunk_size(times.numel(), 7) == 2
    inputs = tuple(value for path in paths.values() for value in (path.residual, path.centres)
                   if value.requires_grad)
    expected = complete_formula(paths, times, valid, order)
    expected_gradients = torch.autograd.grad(expected.sum(), inputs)
    actual = collision_path.collision_path_objective(paths, times, valid, order=order)
    # autograd.grad is the optimizer's vector-Jacobian entry point and must
    # remain supported; reentrant checkpoint would reject this operation.
    actual_gradients = torch.autograd.grad(actual.sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients):
        assert torch.isfinite(actual_gradient).all()
        assert torch.count_nonzero(actual_gradient[:, 5]) == 0
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-13, atol=1e-13)


def test_no_grad_multiblock_execution_has_identical_values(monkeypatch):
    monkeypatch.setattr(collision_path, "_COLLISION_PATH_SAMPLE_ELEMENTS", 32)
    paths, times, valid, order = streams()
    with torch.no_grad():
        expected = complete_formula(paths, times, valid, order)
        actual = collision_path.collision_path_objective(paths, times, valid, order=order)
    assert not actual.requires_grad
    torch.testing.assert_close(actual, expected, rtol=1e-13, atol=1e-13)
# END resource control changes neither soft geometry nor hard scene acceptance.
