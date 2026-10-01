"""CPU checks for independently controlled fingers under one actual-state contract."""
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent / "planning"))
from manipulation.planning_scene import PrismaticGripper
from recorded_simulation import ExecutionLimits
from run_recorded_skill import event_finger_bounds


def limits():
    return ExecutionLimits(source="Synthetic interval regression; not physical calibration",
        joint_names=("arm", "left", "right"), arm_joint_names=("arm",),
        finger_joint_names=("left", "right"), arm_position_error=[.01],
        arm_velocity_error=[.02], tcp_position_error_m=.003, tcp_rotation_error_rad=.01,
        stop_arm_velocity=[.01],
        initial_fingers={"position_bounds": [[.03, .04], [-.03, -.02]], "maximum_speed": [.1, .1]},
        events=({"skill_id": "s", "event_id": "close", "position_bounds": [[.005, .03], [-.025, -.004]],
                 "maximum_speed": [.1, .1]},
                {"skill_id": "s", "event_id": "hold", "position_bounds": [[.012, .016], [-.02, -.011]],
                 "maximum_speed": [.01, .01]}))


def events():
    return [SimpleNamespace(skill_id="s", event_id="close", kind="close", progress=.5),
            SimpleNamespace(skill_id="s", event_id="hold", kind="hold", progress=.8)]


def test_original_progress_boundaries_cover_both_actual_intervals():
    bounds = event_finger_bounds(limits(), events())
    lower, upper = bounds(torch.tensor([[0., .49, .5, .6, .8, .9]], dtype=torch.float64))
    np.testing.assert_allclose(lower[0], [[.03, -.03], [.03, -.03], [.005, -.03],
                                        [.005, -.025], [.005, -.025], [.012, -.02]])
    np.testing.assert_allclose(upper[0], [[.04, -.02], [.04, -.02], [.04, -.004],
                                        [.03, -.004], [.03, -.004], [.016, -.011]])
    # Repeating a boundary at a later clock time must not narrow its geometry.
    repeat = bounds(torch.tensor([[.5, .5]], dtype=torch.float64))
    torch.testing.assert_close(repeat[0][0, 0], repeat[0][0, 1])
    with pytest.raises(ValueError, match="original planner command events"):
        event_finger_bounds(limits(), events()[:1])


def test_position_bounds_do_not_add_finger_solver_coordinates_and_cover_asymmetry():
    bound = event_finger_bounds(limits(), events())
    hand = PrismaticGripper(controller="position_bounds", joint_names=("left", "right"),
        current_positions=[.035, -.028], joint_limits=[[0., .04], [-.04, 0.]],
        joint_bounds=bound, coupling_matrix=np.empty((0, 2)), coupling_offset=np.empty(0),
        sphere_indices=(0, 1), local_spheres=[[0., 0., 0., .002], [0., .1, 0., .003]],
        sphere_joint_translation=[[[1., 0.], [0., 0.], [0., 0.]],
                                  [[0., 0.], [0., 1.], [0., 0.]]],
        self_collision_pairs=(),
        commands=tuple({"skill_id": e.skill_id, "event_id": e.event_id,
                        "command": e.kind, "progress": e.progress} for e in events()),
        source="Native prismatic axes and explicit test execution limits")
    progress = torch.tensor([[0., .5, .6, .9]], dtype=torch.float64)
    coordinates, half, violations = hand.state(progress, torch.zeros((1, 4, 1)), ("arm",))
    assert torch.all(violations == 0)
    lower, upper = bound(progress)
    torch.testing.assert_close(coordinates-half, lower)
    torch.testing.assert_close(coordinates+half, upper)
    assert not torch.allclose(coordinates[..., 0], -coordinates[..., 1])
    proxy = hand.spheres(torch.zeros((1, 4, 2, 4), dtype=torch.float64),
                        torch.zeros((1, 4, 3), dtype=torch.float64),
                        torch.tensor([[[1., 0., 0., 0.]]], dtype=torch.float64).expand(1, 4, 4),
                        coordinates, half)
    # Both endpoints of each independent affine translation remain covered.
    for endpoint in (lower, upper):
        centers = torch.zeros((1, 4, 2, 3), dtype=torch.float64)
        centers[..., 0, 0] = endpoint[..., 0]
        centers[..., 1, 1] = .1+endpoint[..., 1]
        residual = torch.linalg.vector_norm(centers-proxy[..., :3], dim=-1)
        residual += torch.tensor([.002, .003])
        assert torch.all(residual <= proxy[..., 3]+1e-9)
