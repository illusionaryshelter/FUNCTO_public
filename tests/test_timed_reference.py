"""Physical pose/twist boundaries against independently evaluated source motion."""
# BEGIN complete output derivatives, source rotations and actual clocks are checked.
from types import SimpleNamespace
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.motion import MotionModel
from manipulation.planning_problem import (
    MotionPhase, PosePath, quaternion_conjugate, quaternion_multiply,
)
from manipulation.timed_reference import BoundToolReference


def pose(position=(0., 0., 0.), rotation=(0., 0., 0.)):
    result = np.eye(4)
    result[:3, 3] = position
    result[:3, :3] = Rotation.from_rotvec(rotation).as_matrix()
    return result


def phase(path, initial, goals):
    return MotionPhase("declared interval", path, {"reference_boundaries": {
        "initial_pose": initial, "goals": [dict(progress=s, pose=target, source="declared join") for s, target in goals],
        "source": "explicit boundary declaration"}})


def bind(path, initial, twist, *, goals=None, u=(0., 1.), s=(0., 1.), duration=4.):
    if goals is None:
        goals = [(.7, path.sample_numpy(.7))]
    task = SimpleNamespace(phases=(phase(path, initial, goals),))
    return BoundToolReference(task, u, s, duration, initial, twist)


def twist_at(reference, times):
    values = torch.as_tensor(np.asarray(times), dtype=torch.float64).requires_grad_()
    p, q = reference.sample(values)
    # Seven ordinary reverse derivatives independently check the production
    # forward-mode boundary derivative, without asking for Hessians at q=I.
    derivatives = [torch.autograd.grad(component.sum(), values, retain_graph=True)[0]
                   for component in torch.cat((p, q), -1).unbind(-1)]
    jacobian = torch.stack(derivatives, -1)
    v, dq = jacobian[..., :3], jacobian[..., 3:]
    omega = 2*quaternion_multiply(dq, quaternion_conjugate(q))[..., 1:]
    return torch.cat((v, omega), -1).detach().numpy()


def source_with_rotating_offset():
    source = PosePath([0., 1.], np.stack((pose(), pose((.4, -.1, .2), (0., 0., 1.2)))),
                      source="nominal relation", attachment=pose((.23, -.08, .04), (.1, -.2, .3)))
    parent = PosePath([0., 1.], np.stack((pose((.1, .2, .3)), pose((.3, -.1, .4), (0., .4, 0.)))),
                      source="independent parent motion")
    source.parent = parent
    return source


def test_entry_hold_and_event_hold_use_measured_and_one_sided_physical_velocities():
    nominal = source_with_rotating_offset()
    initial = pose((-.3, .4, .8), (.3, -.2, -.4))
    entrance = np.array([.13, -.07, .05, -.12, .09, .16])
    reference = bind(nominal, initial, entrance, goals=[(.6, nominal.sample_numpy(.6))],
                     u=(0., .1, .4, .7, 1.), s=(0., 0., .6, .6, 1.), duration=5.)
    np.testing.assert_allclose(reference.sample_numpy(0.), initial, atol=7e-16, rtol=0)
    np.testing.assert_allclose(twist_at(reference, [0.])[0], entrance, atol=2e-15, rtol=0)
    # First arrival is 2 s, not the hold's end at 3.5 s. The right nominal
    # physical velocity is zero; the left correction cancels nonzero motion.
    assert reference.diagnostics["final_join_time_s"] == 2.
    join = reference.diagnostics["joins"][-1]
    assert np.linalg.norm(join["nominal_left_twist"]) > .1
    np.testing.assert_allclose(join["output_twist"], np.zeros(6), atol=0, rtol=0)
    np.testing.assert_allclose(twist_at(reference, [np.nextafter(2., -np.inf), 2., np.nextafter(2., np.inf)]),
                               0., atol=2e-14, rtol=0)
    np.testing.assert_allclose(reference.sample_numpy(2.), nominal.sample_numpy(.6), atol=0, rtol=0)
    queries = torch.tensor([2., 2.7, 3.5, 4.3, 5.], dtype=torch.float64)
    actual, original = reference.sample(queries), reference._nominal(queries)
    for first, second in zip(actual, original):
        torch.testing.assert_close(first, second, atol=0, rtol=0)


def test_actual_clock_scale_changes_correction_but_keeps_the_same_entrance_twist():
    nominal = source_with_rotating_offset()
    initial, entrance = pose((-.2, .1, .4), (.2, .1, -.3)), [.2, .1, -.05, .1, -.2, .3]
    short, long = (bind(nominal, initial, entrance, duration=T) for T in (2., 8.))
    for reference in (short, long):
        np.testing.assert_allclose(twist_at(reference, [0.])[0], entrance, atol=1e-15, rtol=0)
    assert np.linalg.norm(short.sample_numpy(.5)-long.sample_numpy(2.)) > .05
    np.testing.assert_allclose(short.sample_numpy(1.4), long.sample_numpy(5.6), atol=0, rtol=0)


def test_non_nominal_terminal_goal_survives_at_the_exact_last_physical_time():
    nominal = source_with_rotating_offset()
    initial = pose((-.2, -.1, .3), (.4, -.2, .1))
    goal = pose((.7, -.4, .6), (.1, .2, -.3))
    reference = bind(nominal, initial, [.1, .2, -.1, .2, -.1, .3], goals=[(1., goal)], duration=3.)
    np.testing.assert_allclose(reference.sample_numpy(3.), goal, atol=1e-15, rtol=0)
    np.testing.assert_allclose(twist_at(reference, [3.])[0], reference._nominal_boundary(3., "right")[1], atol=3e-15, rtol=0)


@pytest.mark.parametrize("angle", [np.pi-1e-8, np.pi, np.pi+1e-8])
def test_near_pi_correction_has_finite_continuous_output_and_declared_endpoint_derivatives(angle):
    nominal = source_with_rotating_offset()
    initial = pose((-.3, .1, .7), (angle, 0., 0.))
    initial[:3, :3] = initial[:3, :3] @ nominal.sample_numpy(0.)[:3, :3]
    entrance = np.array([.1, -.2, .3, -.2, .1, .15])
    reference = bind(nominal, initial, entrance)
    values = torch.linspace(0., 4., 129, dtype=torch.float64, requires_grad=True)
    position, quaternion = reference.sample(values)
    derivative = torch.autograd.grad(position.square().sum()+quaternion.square().sum(), values)[0]
    assert torch.isfinite(position).all() and torch.isfinite(quaternion).all() and torch.isfinite(derivative).all()
    np.testing.assert_allclose(twist_at(reference, [0.])[0], entrance, atol=3e-15, rtol=0)
    np.testing.assert_allclose(twist_at(reference, [np.nextafter(2.8, -np.inf)])[0],
        twist_at(reference, [np.nextafter(2.8, np.inf)])[0], atol=2e-14, rtol=0)


def test_multi_turn_mechanism_and_rotating_attachment_remain_the_original_nominal_motion():
    coordinates = np.array([[0.], [4*np.pi]])
    model = MotionModel("revolute", coordinates, pose((.2, 0., .1)), np.eye(3), np.zeros(3),
                        1e-4, 1e-3, {"source": "declared two-turn mechanism"})
    nominal = PosePath([0., 1.], model.poses(coordinates), "unwrapped skill coordinates",
                      motion_model=model, coordinates=coordinates, attachment=pose((.1, .05, .02)))
    initial = nominal.sample_numpy(0.)
    omega = np.array([0., 0., np.pi])  # 4 pi over the explicitly supplied 4 s.
    entrance = np.r_[np.cross(omega, initial[:3, 3]), omega]
    reference = bind(nominal, initial, entrance, goals=[(.1, nominal.sample_numpy(.1))])
    query = np.linspace(0., 4., 401)
    np.testing.assert_allclose(reference.sample_numpy(query), nominal.sample_numpy(query/4), atol=2e-15, rtol=0)
    np.testing.assert_array_equal(reference.nominal_phases[0].coordinates, coordinates)
    rotation = reference.sample_numpy(query)[:, :3, :3]
    total = np.unwrap(np.arctan2(rotation[:, 1, 0], rotation[:, 0, 0]))
    assert total[-1]-total[0] == pytest.approx(4*np.pi, abs=1e-14)


def test_multiple_phase_boundaries_share_output_twist_on_their_actual_clock():
    first = PosePath([0., 1.], np.stack((pose(), pose((1., 0., 0.)))), "first supplied phase")
    second = PosePath([0., 1.], np.stack((pose((1., 0., 0.)), pose((3., 0., 0.)))), "second supplied phase")
    initial = pose((-.2, 0., 0.))
    phases = (phase(first, initial, [(.7, first.sample_numpy(.7))]),
              phase(second, second.sample_numpy(0.), [(.5, second.sample_numpy(.5))]))
    reference = BoundToolReference(SimpleNamespace(phases=phases), [0., .25, 1.], [0., 1., 2.], 4., initial,
                                   [.1, 0., 0., 0., 0., 0.])
    np.testing.assert_allclose(reference.sample_numpy(1.), second.sample_numpy(0.), atol=1e-15, rtol=0)
    np.testing.assert_allclose(twist_at(reference, [np.nextafter(1., -np.inf), np.nextafter(1., np.inf)])[:, 0],
                               2/3, atol=2e-14, rtol=0)
    assert [row["time_s"] for row in reference.boundaries] == [0., .7, 1., 2.5]


def test_nominal_internal_corner_after_declared_join_keeps_its_supplied_derivatives():
    nominal = PosePath([0., .6, 1.], np.stack((pose(), pose((.2, 0., 0.)), pose((1., 0., 0.)))), "source corner")
    reference = bind(nominal, pose((-.2, 0., 0.)), [.1, 0., 0., 0., 0., 0.],
                     goals=[(.3, nominal.sample_numpy(.3))], duration=2.)
    before, after = twist_at(reference, [1.2-1e-7, 1.2+1e-7])[:, 0]
    assert before == pytest.approx(1/6) and after == pytest.approx(1.)
    assert "other nominal corners" in reference.diagnostics["continuity_scope"]


def test_plain_phases_keep_nominal_values_without_inventing_boundary_requirements():
    nominal = source_with_rotating_offset()
    request = SimpleNamespace(phases=(MotionPhase("source", nominal),))
    reference = BoundToolReference(request, [0., .5, 1.], [0., .3, 1.], 3., pose(), np.ones(6))
    times = torch.tensor([[0., 1., 2., 3.]], dtype=torch.float64)
    for actual, expected in zip(reference.sample(times), reference._nominal(times)):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert reference.boundaries == []


def test_event_at_entrance_retains_source_data_and_suffix_for_changed_pose_and_twist():
    nominal = PosePath([0., 1.], np.stack((pose(), pose((1., 0., 0.)))), "declared linear motion")
    initial = nominal.sample_numpy(0.)
    reference = bind(nominal, initial, [.25, 0., 0., 0., 0., 0.], goals=[(0., initial)])
    np.testing.assert_array_equal(reference.sample_numpy([0., .7, 4.]), nominal.sample_numpy([0., .175, 1.]))
    assert reference.diagnostics["status"] == "nominal_reference"
    changed = bind(nominal, pose((.1, -.2, .3)), [.1, 0., 0., 0., 0., .4], goals=[(0., initial)])
    np.testing.assert_array_equal(changed.sample_numpy([0., 2., 4.]), nominal.sample_numpy([0., .5, 1.]))
    assert not changed.diagnostics["entrance_twist_bound"]
    np.testing.assert_array_equal(changed.source_boundaries[0]["declaration"]["goals"][0]["pose"], initial)


def test_continuation_after_last_declared_join_consumes_the_nominal_suffix():
    nominal = source_with_rotating_offset()
    initial = pose((-.3, .1, .2))
    reference = bind(nominal, initial, np.ones(6), goals=[(.4, nominal.sample_numpy(.4))],
                     s=(.6, 1.))
    np.testing.assert_array_equal(reference.sample_numpy([0., 2., 4.]), nominal.sample_numpy([.6, .8, 1.]))
    assert reference.diagnostics["status"] == "nominal_reference"
    assert reference.diagnostics["declared_boundaries_in_interval"] == 0
    assert not reference.diagnostics["entrance_twist_bound"]


# BEGIN archives preserve the same correction at interior times and clock holds.
def test_record_roundtrip_preserves_entry_twist_continuous_motion_and_nominal_suffix():
    from manipulation.execution_binding import plain
    nominal = source_with_rotating_offset()
    initial = pose((-.3, .4, .8), (.3, -.2, -.4))
    velocity = [.13, -.07, .05, -.12, .09, .16]
    reference = bind(nominal, initial, velocity, goals=[(.6, nominal.sample_numpy(.6))],
        u=(0., .1, .4, .7, 1.), s=(0., 0., .6, .6, 1.), duration=5.)
    record = json.loads(json.dumps(plain(reference.to_record())))
    restored = BoundToolReference.from_record(record)
    query = np.unique(np.r_[np.linspace(0., 5., 101), .137, 1.983, 2., 3.5])
    np.testing.assert_array_equal(restored.sample_numpy(query), reference.sample_numpy(query))
    np.testing.assert_array_equal(twist_at(restored, query), twist_at(reference, query))
    np.testing.assert_allclose(restored.sample_numpy(0.), initial, atol=1e-14)
    np.testing.assert_allclose(twist_at(restored, [0.])[0], velocity, atol=1e-14)
    assert plain(restored.to_record()) == record
# END reloading a sampled pose cache cannot satisfy this continuous invariant.
# END each continuity claim is tied to explicit input geometry and physical time.
