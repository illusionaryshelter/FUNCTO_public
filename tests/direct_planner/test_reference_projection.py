"""Independent Torch objectives and original cubic bounds check initialization."""

# BEGIN compare the compiled residuals with the maintained physical producers.
from dataclasses import replace

import numpy as np
import pytest
import torch

from manipulation.planning_problem import _first_event_query
from manipulation.preference_planner.pose_roles import bind_pose_objectives
from manipulation.direct_planner.reference_projection import ReferenceProjection
from manipulation.direct_planner.support_linearization import normalized_pose_event
from .test_direct_integration import robot_case


@pytest.mark.parametrize('yaw', [None, 0., .2])
def test_projection_preserves_reference_motion_and_declared_event_geometry(yaw):
    planner, initial = robot_case()
    computation = planner.computation
    original_event = computation.problem.pose_events[0]
    event = replace(original_event, yaw_axis=None if yaw is None else np.array([0., 1., 0.]),
        yaw_tolerance_rad=yaw)
    computation.problem.pose_events = (event,)
    computation.problem = bind_pose_objectives(computation.problem, [dict(
        goal_id=['pose_events', 0], role='required',
        scales=dict(position=.02, rotation=.1, yaw=.1) if yaw is not None else dict(position=.02, rotation=.1),
        scale_source='original analytical fixture event')])
    event = computation.problem.pose_events[0]
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    context = computation.shared_context(controls)
    fit = ReferenceProjection(computation, context, check_request=lambda: None)
    actual = dict(zip(fit.component_names, np.asarray(fit.component_costs(initial.ravel())).ravel()))
    original = computation.objective_components(controls, context=context)
    for name in ('reference_prior', 'tool_angular_motion', 'motion_quality'):
        assert actual[name] == pytest.approx(float(original[name].detach()), rel=2e-12, abs=1e-9)
    mask = _first_event_query(computation.progress, event.progress, computation.times)
    sample = int(mask[0].to(dtype=torch.int64).argmax())
    tool = tuple(value[:, sample] for value in context['robot_state'][4])
    rows = normalized_pose_event(event, *tool)
    expected_event = sum(float(value.detach())+1 for value in rows.values())
    if yaw == 0.:
        expected_event += float(event.error_components(*tool)['yaw'].square().detach())
    assert actual['event_initialization'] == pytest.approx(expected_event, rel=2e-12, abs=1e-9)
    # A fitted numerical proposal retains the actual measured entrance and
    # remains within the original position bounds throughout its cubic.
    fitted, record = fit.fit(initial, maximum_evaluations=10)
    assert record['cost'] <= float(np.square(np.asarray(fit.residual(
        np.clip(initial.ravel(), *fit.bounds)))).sum())
    full = computation.terminal.expand(fitted)
    curve = computation.full_domain._curve_map
    rows, offset = curve.query(np.linspace(0., 1., 251))
    positions = rows@full+offset
    bounds = computation.bounds.numpy()[0]
    assert np.all(positions >= bounds[0]-1e-14) and np.all(positions <= bounds[1]+1e-14)
    np.testing.assert_allclose(positions[0], curve.start_q, rtol=0., atol=1e-14)
# END scene qualification remains in the real original-problem integration tests.
