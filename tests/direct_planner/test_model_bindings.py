"""Small CPU invariants for local bindings to maintained model interfaces."""

# BEGIN compare affine spaces and halfspace extrema, not implementation text.
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.cubic_constraints import CubicAffineMap
from manipulation.preference_planner.terminal import TerminalHold
from manipulation.direct_planner.material_hull import material_hull_indices
from manipulation.direct_planner.terminal_binding import bind_terminal_hold
from manipulation.direct_planner.nominal_hand_geometry import bind_position_controlled_finger_geometry
from manipulation.gripper_trajectory import (
    FingerTrajectory, _extrema, bind_nominal_finger_geometry,
)
from manipulation.planning_problem import event_occurrence_key
from test_gripper_trajectory import _arguments


@pytest.mark.parametrize("controls", [2, 3, 7, 21])
@pytest.mark.parametrize("velocity", [[0., 0.], [.02, -.03]])
def test_sparse_terminal_preserves_native_affine_projection_and_curve(controls, velocity):
    curve = CubicAffineMap([.11, -.31], velocity, controls, .43)
    rng = np.random.default_rng(22031)
    stops = sorted(set([1.] + [k / curve.intervals for k in range(curve.intervals)]
        + [(k + .37) / curve.intervals for k in range(curve.intervals)]))
    for stop in stops:
        u, z = ([0., 1.], [1., 1.]) if stop == 0. else (
            ([0., 1.], [0., 1.]) if stop == 1. else ([0., stop, 1.], [0., 1., 1.]))
        interval = np.searchsorted(np.arange(curve.intervals + 1) / curve.intervals,
                                   stop, side="right") - 1
        if interval == 0 and np.any(curve.start_dq != 0.):
            with pytest.raises(ValueError, match="nonzero entrance velocity"):
                bind_terminal_hold(curve, u, z, source="same clock")
            continue
        native = TerminalHold.from_clock(curve, u, z, source="same clock")
        sparse = bind_terminal_hold(curve, u, z, source="same clock")
        seed = rng.normal(size=(2, controls, 2))
        expected, actual = native.expand(native.reduce(seed)), sparse.expand(sparse.reduce(seed))
        np.testing.assert_allclose(actual, expected, rtol=0., atol=8e-15)
        # A shared-column dot product sums up to `controls` floating terms;
        # its arithmetic error scales with that length, not a physical bound.
        np.testing.assert_allclose(sparse.matrix.T @ sparse.matrix,
            np.eye(sparse.matrix.shape[1]), rtol=0., atol=controls * np.finfo(float).eps)
        for order in range(4):
            np.testing.assert_allclose(curve.evaluate(actual, np.linspace(0., 1., 37), order),
                curve.evaluate(expected, np.linspace(0., 1., 37), order), rtol=0., atol=2e-12)
        for key, value in native.report.items():
            assert sparse.report[key] == value


def test_sparse_terminal_torch_gradient_and_native_pullback_agree():
    curve = CubicAffineMap([.1, -.2], [.03, -.04], 7, .4)
    terminal = bind_terminal_hold(curve, [0., .71, 1.], [0., 1., 1.], source="interior stop")
    reduced = torch.tensor(np.random.default_rng(13).normal(size=(terminal.matrix.shape[1], 2)),
                           dtype=torch.float64, requires_grad=True)
    weights = np.random.default_rng(14).normal(size=(3, 7, 2))
    output = torch.einsum("rnj,nj->r", torch.tensor(weights), terminal.expand(reduced))
    actual = torch.stack([torch.autograd.grad(value, reduced, retain_graph=True)[0].ravel()
                          for value in output]).detach().numpy()
    expected = terminal.pullback(weights.reshape(3, -1))
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-15)
    assert not terminal.matrix.flags.writeable and not terminal.offset.flags.writeable


def material_cases():
    line = np.array([[-1., 0., 0.], [0., 0., 0.], [1., 0., 0.], [0., 0., 0.]])
    plane = np.array([[-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.],
                      [1., 1., 0.], [0., 0., 0.]])
    cube = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    rotation = Rotation.from_rotvec([.4, -.2, .7]).as_matrix()
    near_plane = np.vstack((plane, [[0., 0., 1e-18]])) @ rotation.T
    return [np.array([[.2, .1, .3]]), line, plane, np.vstack((cube, [[0., 0., 0.]])),
            line @ rotation.T, plane @ rotation.T, near_plane]


@pytest.mark.parametrize("points", material_cases())
def test_material_indexes_preserve_original_halfspace_extrema(points):
    original = points.copy()
    indices = material_hull_indices(points)
    assert indices.dtype == np.int64 and np.all(np.diff(indices) > 0)
    np.testing.assert_array_equal(points, original)
    # Rigid transforms followed by every halfspace are still affine in the
    # original material point. Include all coordinate directions as witnesses.
    rng = np.random.default_rng(81)
    directions = np.vstack((rng.normal(size=(100, 3)), np.eye(3), -np.eye(3)))
    for _ in range(5):
        transform = Rotation.random(random_state=rng).as_matrix()
        moved = points @ transform.T + rng.normal(size=3)
        np.testing.assert_allclose((moved @ directions.T).max(axis=0),
            (moved[indices] @ directions.T).max(axis=0), rtol=0., atol=2e-14)


def test_unresolved_affine_rank_keeps_real_thickness_without_projection():
    points = np.array([[-1., -1., 0.], [-1., 1., 0.], [1., -1., 0.],
                       [1., 1., 0.], [0., 0., 1e-18]])
    indices = material_hull_indices(points)
    np.testing.assert_array_equal(indices, np.arange(len(points)))
    assert points[indices, 2].max() == 1e-18
    np.testing.assert_array_equal(material_hull_indices([[0., 0., 0.], [1., 0., 0.],
        [.5, 0., 0.], [1., 0., 0.]]), [0, 1])


def hand_case(*, open_time=.3005, contact_time=1., release_time=None):
    """Build separate measured and commanded shapes with original event data."""
    arguments = _arguments()
    arguments.update(duration=1.4, executable_until=1.4)
    arguments["incoming_command"].update(position=[.005, -.008], velocity=[0., 0.],
        target_position=[.005, -.008], remaining_ramp_s=0.)
    arguments["commands"] = [
        dict(arguments["commands"][0], time=open_time, progress=.3),
        dict(arguments["commands"][1], time=contact_time, progress=.8),
    ]
    reference = FingerTrajectory.build(**arguments)
    clock = {event_occurrence_key(row): {"time": row["time"],
        "activation_time_s": row["applied_time_s"]} for row in reference.events}
    conditions = [dict(kind="declared_attachment", condition_id="current-held-relation",
        source="Declared model at the measured entrance"),
        dict(kind="grasp_contact_activation", interval=[.8, 1.], condition_id="future-contact",
            activation_event={key: reference.events[-1][key]
                              for key in ("skill_id", "event_id", "occurrence")})]
    if release_time is not None:
        conditions.append(dict(kind="predicted_release", progress=.2, skill_id="s",
            event_id="released", condition_id="original-release"))
        clock[("s", "released", 0)] = {"time": release_time}
    return reference, conditions, clock


def bind_hand(reference, conditions, clock):
    return bind_position_controlled_finger_geometry(reference, conditions, clock,
        progress_interval=(0., 1.), source="Explicit fixed position-control nominal response")


def test_held_open_model_uses_predicted_endpoint_and_keeps_command_measurements_separate():
    reference, conditions, clock = hand_case()
    original = reference.describe()
    prefix = bind_nominal_finger_geometry(reference, conditions, clock, progress_interval=(0., 1.))
    model = bind_hand(reference, conditions, clock)
    transition = reference.events[0]["applied_time_s"]
    assert transition == prefix.valid_until_s == .302
    assert model.valid_until_s == reference.events[1]["applied_time_s"] == 1.
    np.testing.assert_array_equal(model.sample([0., transition])[0],
                                  np.tile(reference.measured_initial_position, (2, 1)))
    assert not model.sample(transition)[1].any()
    assert not np.array_equal(reference.sample(transition)[0], model.sample(transition)[0])
    np.testing.assert_allclose(model.sample(.8)[0], reference.events[0]["finger_positions"], atol=1e-15)
    assert _extrema(model._curve, model.joint_names, reference._limits())["bounds_satisfied"]
    assert model.valid_mask([1., np.nextafter(1., np.inf)]).tolist() == [True, False]
    assert "predicted endpoint" in model.source
    assert reference.describe() == original
    # The original archive interface retains the same complete polynomial and validity.
    from manipulation.gripper_trajectory import NominalFingerGeometry
    assert NominalFingerGeometry.from_mapping(model.describe()).describe() == model.describe()


def test_release_before_open_follows_the_active_target_from_predicted_shape():
    reference, conditions, clock = hand_case(open_time=.6, release_time=.2)
    model = bind_hand(reference, conditions, clock)
    np.testing.assert_array_equal(model.sample(.2)[0], reference.measured_initial_position)
    assert model.sample(.3)[0][0] < reference.measured_initial_position[0]
    np.testing.assert_allclose(model.sample(.5)[0], reference.incoming_command["target_position"], atol=1e-15)
    np.testing.assert_allclose(model.sample(.9)[0], reference.events[0]["finger_positions"], atol=1e-15)
    assert model.valid_until_s == 1.
    assert _extrema(model._curve, model.joint_names, reference._limits())["bounds_satisfied"]


def test_initial_free_model_retains_the_original_polynomial_and_contact_boundary():
    reference = FingerTrajectory.build(**_arguments())
    clock = {event_occurrence_key(row): {"time": row["time"],
        "activation_time_s": row["applied_time_s"]} for row in reference.events}
    conditions = [dict(kind="grasp_contact_activation", interval=[.6, 1.], condition_id="contact",
        activation_event={key: reference.events[1][key]
                          for key in ("skill_id", "event_id", "occurrence")})]
    original = bind_nominal_finger_geometry(reference, conditions, clock, progress_interval=(0., 1.))
    model = bind_hand(reference, conditions, clock)
    for name in ("times", "position", "velocity"):
        np.testing.assert_array_equal(getattr(model, name), getattr(original, name))
    assert model.valid_until_s == original.valid_until_s == .302


def test_contact_ending_at_original_open_does_not_limit_the_free_side():
    reference, conditions, clock = hand_case()
    conditions.append(dict(kind="grasp_contact_activation", interval=[0., .3], condition_id="old-contact",
        activation_event=dict(skill_id="previous", event_id="contact", occurrence=0)))
    assert bind_hand(reference, conditions, clock).valid_until_s == 1.


@pytest.mark.parametrize("has_occurrence", [False, True])
def test_contact_spanning_free_start_remains_unmodelled_after_held_prefix(has_occurrence):
    reference, conditions, clock = hand_case()
    identity = dict(skill_id="original", event_id="contact", occurrence=0)
    conditions.append(dict(kind="grasp_contact_activation", interval=[0., .7],
        condition_id="still-active-contact", activation_event=identity))
    if has_occurrence:
        clock[event_occurrence_key(identity)] = {"time": .1, "activation_time_s": .1}
    model = bind_hand(reference, conditions, clock)
    assert model.valid_until_s == reference.events[0]["applied_time_s"] == .302
    assert not model.valid_mask(np.nextafter(.302, np.inf))


def test_same_progress_release_and_later_open_do_not_clear_contact_early():
    reference, conditions, clock = hand_case(open_time=.6, release_time=.2)
    conditions[-1]["progress"] = reference.events[0]["progress"]  # Same progress, different times.
    conditions.append(dict(kind="grasp_contact_activation", interval=[0., .3], condition_id="old-contact",
        activation_event=dict(skill_id="previous", event_id="contact", occurrence=0)))
    model = bind_hand(reference, conditions, clock)
    assert model.valid_until_s == .2 < reference.events[0]["applied_time_s"]
    assert not model.valid_mask(.6)  # The maintained interface is one valid prefix, not disjoint intervals.


def test_missing_future_contact_occurrence_keeps_the_existing_input_error():
    reference, conditions, clock = hand_case()
    del clock[event_occurrence_key(reference.events[1])]
    with pytest.raises(ValueError, match="original bound event occurrence"):
        bind_hand(reference, conditions, clock)
# END maintained curve and complete material declarations remain the owners.
