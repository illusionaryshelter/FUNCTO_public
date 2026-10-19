"""Semantic query compaction retains exact witnesses and max-tie derivatives."""

# BEGIN compare compact geometry batches with the unchanged three-mode contract.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.planning import _ProgressMap, _scene_boundary_evaluate


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_boundary_compaction_keeps_batched_plateaus_release_witness_and_tie_gradients(dtype):
    mapping = _ProgressMap(np.array([0., .25, .75, 1.]), np.array([0., .5, .5, 1.]))
    # Batch rows deliberately have different exact arrival columns. The middle
    # of row zero's plateau must not inherit its entrance's expired left mode.
    u = torch.tensor([[0., .25, .5, .75, 1.], [0., .125, .25, .75, 1.]], dtype=dtype)
    progress = torch.tensor([[0., .5, .5, .5, 1.], [0., .25, .5, .5, 1.]], dtype=dtype)
    left = torch.tensor([[False, True, False, False, False], [False, False, True, False, False]])
    right = torch.tensor([[False, False, False, True, False], [False, False, False, True, False]])
    lower = torch.where(left, torch.nextafter(progress, torch.full_like(progress, -float('inf'))), progress)
    upper = torch.where(right, torch.nextafter(progress, torch.full_like(progress, float('inf'))), progress)
    seen = []

    def evaluate(spheres, position, quaternion, s, joints, names, *, trajectory_time,
                 contact_permission_progress):
        seen.append((s.detach().clone(), joints.detach().clone(), trajectory_time.detach().clone(),
                     contact_permission_progress.detach().clone()))
        # A released owner needs an exact release pose in the same evaluation;
        # separate sparse side queries cannot supply this witness reliably.
        matching = s == .5
        index = matching.to(torch.int64).argmax(1)
        release = joints[..., 0].gather(1, index[:, None])
        missing = (~matching.any(1)).to(s.dtype)[:, None]*100.
        coefficient = torch.where(s < .5, 2., torch.where(s > .5, 3., 1.))
        # At q=1 the mode costs tie but their derivatives differ. Removing
        # duplicate exact scalar terms from amax would change the gradient.
        return coefficient*(joints[..., 0]-1.)+1.+(release-1.)+missing

    scene = SimpleNamespace(required_progress_anchors=(.5,),
        contacts=(SimpleNamespace(actor='robot', interval=(.5, 1.)),), evaluate=evaluate)
    problem = SimpleNamespace(start_progress=0., end_progress=1.)
    q = torch.ones((2, 5, 1), dtype=dtype, requires_grad=True)
    counts = {}
    actual = _scene_boundary_evaluate(scene, problem, mapping,
        torch.zeros((2, 5, 1, 4), dtype=dtype), torch.zeros((2, 5, 3), dtype=dtype),
        torch.zeros((2, 5, 4), dtype=dtype), progress, q, ('q',),
        trajectory_time=u*4, normalized_time=u, query_counts=counts)
    gradient = torch.autograd.grad(actual.sum(), q)[0]
    assert len(seen) == 1
    last = {'exact_states': 10, 'left_states': 4, 'right_states': 2, 'evaluated_states': 16}
    assert counts == {**last, 'query_calls': 1, **{'total_'+key: value for key, value in last.items()}}
    compact_s, compact_q, compact_time, compact_permission = seen[0]
    columns = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3])
    torch.testing.assert_close(compact_q, q.detach().index_select(1, columns), rtol=0, atol=0)
    torch.testing.assert_close(compact_time, (u*4).index_select(1, columns), rtol=0, atol=0)
    torch.testing.assert_close(compact_s[:, :5], progress, rtol=0, atol=0)
    # Exact arrival grants the finite material permission even when attachment
    # mode is queried immediately to its left, with no shift of physical time.
    assert compact_s[0, 5] < .5 and compact_permission[0, 5] == .5
    assert compact_s[1, 6] < .5 and compact_permission[1, 6] == .5

    q_reference = q.detach().clone().requires_grad_(True)
    tripled = lambda value: torch.cat((value, value, value), dim=1)
    reference_terms = evaluate(torch.zeros((2, 15, 1, 4), dtype=dtype),
        torch.zeros((2, 15, 3), dtype=dtype), torch.zeros((2, 15, 4), dtype=dtype),
        torch.cat((progress, lower, upper), 1), tripled(q_reference), ('q',),
        trajectory_time=tripled(u*4), contact_permission_progress=torch.cat((progress, progress, upper), 1))
    expected = torch.stack(reference_terms.split(5, dim=1)).amax(0)
    expected_gradient = torch.autograd.grad(expected.sum(), q_reference)[0]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(gradient, expected_gradient, rtol=1e-6, atol=1e-7)


def test_declared_boundary_absent_from_samples_adds_no_side_geometry():
    mapping = _ProgressMap(np.array([0., 1.]), np.array([0., 1.]))
    q = torch.tensor([[[.2], [.8]]], requires_grad=True)
    called = []
    def evaluate(spheres, position, quaternion, s, joints, names, **kwargs):
        called.append(s.shape[1])
        return joints[..., 0].square()
    scene = SimpleNamespace(required_progress_anchors=(.5,), contacts=(), evaluate=evaluate)
    progress = torch.tensor([[.2, .8]])
    actual = _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
        mapping, torch.zeros((1, 2, 1, 4)), torch.zeros((1, 2, 3)), torch.zeros((1, 2, 4)),
        progress, q, ('q',), trajectory_time=progress, normalized_time=progress)
    assert called == [2]
    torch.testing.assert_close(actual, q[..., 0].square())
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), q)[0], 2*q)


def test_query_totals_accumulate_across_different_batch_layouts():
    counts = {}
    scene = SimpleNamespace(required_progress_anchors=(), evaluate=lambda *args, **kwargs: args[3]*0.)
    for batch, horizon in ((1, 3), (2, 4)):
        s = torch.zeros((batch, horizon))
        _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
            None, None, None, None, s, None, (), trajectory_time=s, normalized_time=s, query_counts=counts)
    assert counts['query_calls'] == 2
    assert counts['evaluated_states'] == 8
    assert counts['total_evaluated_states'] == counts['total_exact_states'] == 11
# END only geometry work shrinks; task witnesses and scalar max semantics remain.


# BEGIN a stopped request still reports the actual compact query layout.
def test_timeout_returns_query_counts_without_authorizing_partial_candidate():
    from manipulation.planning import MotionPlanningError
    from test_motion_backend_contract import Native, run
    native = Native([False])
    counts = {'process_mode': 'trajectory_optimizer', 'anchor_count': 3,
              'anchor_capacity': 256, 'exact_states': 40, 'left_states': 4,
              'right_states': 4, 'evaluated_states': 48}
    native._manager_type = SimpleNamespace(query_diagnostics=lambda: [dict(counts)])
    def stop(seed):
        native.calls.append('solve')
        raise MotionPlanningError('deadline_exceeded', 'actual step consumed request deadline')
    native.optimize = stop
    result, backend = run(native)
    try:
        assert not result.success and result.trajectory is None and result.execution_binding is None
        assert result.diagnostics['failure'] == 'deadline_exceeded'
        assert result.diagnostics['rollout_query_counts'] == [counts]
        assert 'export' not in native.calls
    finally:
        backend.close()
# END counters preserve failure evidence, never execution permission.


# BEGIN unsupported dynamic capture fails at the shared query boundary.
def test_dynamic_boundary_capture_is_rejected_before_geometry_or_tensor_work(monkeypatch):
    from manipulation.planning import MotionPlanningError
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    def forbidden(*args, **kwargs):
        raise AssertionError("Unsupported capture reached geometry evaluation")
    scene = SimpleNamespace(required_progress_anchors=(.5,), evaluate=forbidden)
    # A metadata-only stand-in deliberately has no tensor operations. The
    # capability check must reject before nonzero/FK/geometry can be invoked.
    progress = SimpleNamespace(is_cuda=True)
    with pytest.raises(MotionPlanningError, match="require eager execution") as error:
        _scene_boundary_evaluate(scene, SimpleNamespace(start_progress=0., end_progress=1.),
            None, None, None, None, progress, None, (), trajectory_time=None, normalized_time=None)
    assert error.value.reason == "unsupported_model"
# END no hidden fallback or CUDA environment setup is needed for the rejection.
