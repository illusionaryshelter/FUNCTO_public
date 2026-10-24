"""Continuous path records preserve geometry and derivatives between knots."""

# BEGIN records traverse the production binding serializer and JSON boundary.
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.execution_binding import plain
from manipulation.motion import MotionModel
from manipulation.planning_problem import PosePath, reference_phase
from manipulation.representation import generate_reference
from manipulation.timed_reference import BoundToolReference
from test_initial_relative_rotation_reference import fixture, pose


def restore(path):
    return PosePath.from_record(json.loads(json.dumps(plain(path))))


@pytest.mark.parametrize("side", ["left", "right"])
def test_moving_functional_rotation_keeps_nested_paths_and_progress_derivatives(side):
    source, arguments = fixture(moving_reference=True)
    source["is_initial_relative_rotation_rpy"] = [False, False, True]
    reference = generate_reference(source, **arguments)
    path = reference["eef_path"]
    rebuilt = restore(path)
    # Query interior locations as well as joins: knot caches alone would
    # retain the endpoints but alter the rotated tool's lever-arm trajectory.
    query = np.unique(np.r_[path.progress, .137, .381, .694])
    np.testing.assert_array_equal(rebuilt.sample_numpy(query), path.sample_numpy(query))
    for value in (path, rebuilt):
        progress = torch.tensor(query, dtype=torch.float64, requires_grad=True)
        p, q = value.sample(progress, side=side)
        scalar = (p*torch.tensor([1., 2., 3.])).sum() + (q*torch.tensor([4., 5., 6., 7.])).sum()
        derivative, = torch.autograd.grad(scalar, progress)
        if value is path:
            expected = derivative
        else:
            torch.testing.assert_close(derivative, expected, rtol=0., atol=0.)
    # The complete expression, rather than its inherited pose cache, is bound.
    record = plain(path)
    operation = record
    while operation["kind"] == "pose":
        operation = operation["parent"]
    assert operation["kind"] == "functional_rotation_offset"
    assert set(operation) == {"kind", "nominal", "functional", "rotation", "source"}
    assert plain(rebuilt) == record


def test_mechanism_record_preserves_multiple_turns_and_composed_parent():
    coordinates = np.array([[0.], [4*np.pi]])
    model = MotionModel("revolute", coordinates, pose((.2, 0., .1)), np.eye(3), np.zeros(3),
        1e-4, 1e-3, {"source": "declared two-turn mechanism"})
    parent = PosePath([0., 1.], np.stack((pose(), pose((.1, -.05, .03)))), "moving reference")
    path = PosePath([0., 1.], model.poses(coordinates), "unwrapped mechanism",
        motion_model=model, coordinates=coordinates, attachment=pose((.1, .05, .02)), parent=parent)
    rebuilt = restore(path)
    query = np.linspace(0., 1., 101)
    expected = path.sample_numpy(query)
    actual = rebuilt.sample_numpy(query)
    np.testing.assert_array_equal(actual, expected)
    angle = np.unwrap(np.arctan2(actual[:, 1, 0], actual[:, 0, 0]))
    assert angle[-1]-angle[0] == pytest.approx(4*np.pi, abs=1e-14)


def test_complete_phase_record_restores_the_same_physical_time_reference():
    source, arguments = fixture(moving_reference=True)
    source["is_initial_relative_rotation_rpy"] = [True, False, True]
    reference = generate_reference(source, **arguments)
    phase = reference_phase(reference, skill_id="declared_motion", source="explicit reference geometry")
    record = json.loads(json.dumps(plain(phase)))
    rebuilt = type(phase)(record["name"], PosePath.from_record(record["reference"]), record["metadata"])
    arguments = ([0., .35, 1.], [0., .6, 1.], 3.6,
                 reference["eef_boundaries"]["initial_pose"], [.03, -.01, .02, .04, -.03, .01])
    original = BoundToolReference(SimpleNamespace(phases=(phase,)), *arguments)
    restored = BoundToolReference(SimpleNamespace(phases=(rebuilt,)), *arguments)
    query = np.linspace(0., 3.6, 401)
    np.testing.assert_array_equal(restored.sample_numpy(query), original.sample_numpy(query))
# END runtime truth and robot model loading are not inputs to these tests.
