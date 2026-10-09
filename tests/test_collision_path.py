"""CPU mathematical counterexamples for the shared geometric collision cost."""
# BEGIN objective values and derivatives, without robot or task fixtures.
import torch

from manipulation.collision_path import SphereCollisionPath, collision_path_objective


def evaluate(x, c, *, times=None, valid=None, modes=(), radius=1.):
    x = torch.as_tensor(x, dtype=torch.float64)
    if x.ndim == 1:
        x = torch.stack((x, x * 0, x * 0), -1)[None, :, None]
    c = torch.as_tensor(c, dtype=torch.float64).reshape(1, -1, 1)
    times = torch.arange(c.shape[1], dtype=c.dtype)[None] if times is None else torch.tensor([times], dtype=c.dtype)
    valid = torch.ones_like(times, dtype=torch.bool) if valid is None else torch.tensor([valid])
    path = SphereCollisionPath(c, x, c.new_tensor([radius]), tuple(torch.tensor([m])[..., None] for m in modes))
    return collision_path_objective({"pair": path}, times, valid)


def test_timing_and_collinear_sampling_do_not_change_piecewise_linear_arc():
    first = evaluate([0., 1., 2.], [0., 1., 0.], times=[0., .1, 1.])
    second = evaluate([0., .5, 1., 1.5, 2.], [0., .5, 1., .5, 0.], times=[0., .7, .8, .9, 1.])
    torch.testing.assert_close(first, first.new_tensor([2.]))
    torch.testing.assert_close(first, second)


def test_free_travel_does_not_dilute_existing_collision():
    torch.testing.assert_close(evaluate([0., 1., 2.], [0., 1., 0.]),
                               evaluate([0., 1., 2., 100., 2.], [0., 1., 0., 0., 0.]))


def test_stationary_collision_and_zero_length_gradients_are_finite():
    x = torch.zeros(1, 3, 1, 3, dtype=torch.float64, requires_grad=True)
    c = torch.full((1, 3, 1), .2, dtype=torch.float64, requires_grad=True)
    value = collision_path_objective({"pair": SphereCollisionPath(c, x, c.new_ones(1))},
        c.new_tensor([[0., .3, 1.]]), torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(value, c.new_tensor([.2]))
    gx, gc = torch.autograd.grad(value.sum(), (x, c))
    assert torch.isfinite(gx).all() and torch.isfinite(gc).all()
    assert gc.sum() == 1 and gx.abs().sum() == 0


def test_duplicate_time_and_mode_switch_never_become_motion():
    # Neither the same-time displacement nor the different-mode edge is real.
    value = evaluate([0., 1., 100., 200.], [0., 2., 3., 1.],
        times=[0., 1., 1., 2.], modes=([0, 0, 1, 2],))
    torch.testing.assert_close(value, value.new_tensor([4.]))
    assert evaluate([0.], [.4]).item() == .4


def test_invalid_padding_is_masked_before_norm_and_peak():
    actual = evaluate([0., 1., float("nan")], [0., 2., float("nan")], valid=[True, True, False])
    torch.testing.assert_close(actual, actual.new_tensor([3.]))
    assert torch.isnan(evaluate([0., 1.], [0., float("nan")])).all()


def test_objective_matches_exact_autograd_of_discrete_formula():
    x = torch.tensor([[[[0., 0., 0.]], [[.4, .2, 0.]], [[1., .3, 0.]]]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[[.1], [.3], [.2]]], dtype=torch.float64, requires_grad=True)
    def function(centres, residual):
        return collision_path_objective({"pair": SphereCollisionPath(residual, centres, residual.new_tensor([.2]))},
            residual.new_tensor([[0., .2, 1.]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.autograd.gradcheck(function, (x, c))


def test_each_stream_sphere_and_batch_contributes_independently():
    # Sphere 0 crosses a mode boundary; sphere 1 retains one continuous model.
    c = torch.tensor([[[1., 2.], [3., 4.]], [[0., 1.], [0., 1.]]], dtype=torch.float64)
    x = torch.zeros(2, 2, 2, 3, dtype=c.dtype)
    x[:, 1, :, 0] = 2.
    modes = torch.tensor([[[0, 0], [1, 0]], [[0, 0], [0, 0]]])
    path = SphereCollisionPath(c, x, c.new_tensor([1., 2.]), (modes,))
    result = collision_path_objective({"a": path, "b": path},
        c.new_tensor([[0., 1.], [0., 1.]]), torch.ones(2, 2, dtype=torch.bool))
    # Batch 0: peak 3+4, arc 0+3. Batch 1: peak 0+1, arc 0+1.
    torch.testing.assert_close(result, c.new_tensor([20., 4.]))
# END numerical tests do not change original collision acceptance.
