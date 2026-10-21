"""Scene query work is scoped to immutable entities and exact batch states."""

# BEGIN count production operations while retaining all returned evidence.
import numpy as np
import pytest
import torch

import manipulation.native_field_distance as native
import manipulation.online_geometry as online
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import CollisionOwner, ContactRule
from test_native_scene_contract import make_scene


def owner(name):
    values = np.indices((2, 2, 2))[0].astype(float) - .5
    field = FrozenObservedField(values, np.ones_like(values, bool), weights=np.ones_like(values),
        origin=[1., 1., 1.], voxel_size=.1, name=name, source="synthetic observed samples")
    path = PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "fixed relation")
    return CollisionOwner(name, field, path, "synthetic observed samples", moving=False)


def evaluate(scene, count):
    q = torch.tensor([[[.04, -.02]]], dtype=torch.float64).expand(1, count, 2).clone().requires_grad_()
    position = torch.zeros((1, count, 3), dtype=torch.float64)
    quaternion = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, count, 4)
    progress = torch.full((1, count), .5, dtype=torch.float64)
    return scene.components(None, position, quaternion, progress, q,
        ("left_slide", "right_slide"), trajectory_time=progress)


@pytest.mark.parametrize("count", [1, 8, 32])
def test_missing_required_contact_converts_each_actor_batch_once(tmp_path, monkeypatch, count):
    scene = make_scene(tmp_path)
    scene.owners = (owner("target"),)
    scene.contacts = (ContactRule(rule_id="required_feature", actor="robot", target_owner="target",
        interval=(0., 1.), material_id="feature",
        feature_points=[[-.001, -.001, 0.], [.001, -.001, 0.],
                        [.001, .001, 0.], [-.001, .001, 0.]],
        feature_normal=[0., 0., -1.], patch_origin=[0., 0., 0.], patch_axes=np.eye(3),
        patch_half_size=[.01, .01], gap_bounds=(-.001, .001), normal_tolerance_rad=.1,
        modeling_allowance_m=0., geometry_error_m=0., source="declared finite feature",
        required=True, robot_body="first"),)
    original = online._matrices
    batches = []

    def counted(pose):
        batches.append(int(np.prod(pose[0].shape[:-1])))
        return original(pose)

    monkeypatch.setattr(online, "_matrices", counted)
    result = evaluate(scene, count)
    assert batches == [count] * 5
    assert sum(batches) == 5 * count
    torch.testing.assert_close(result["contact/required_feature"], torch.ones((1, count), dtype=torch.float64))


def test_local_bounds_survive_calls_but_world_bounds_are_rebuilt_per_batch(tmp_path, monkeypatch):
    original = online._native_bounds
    calls = []

    def counted(shape, transform):
        calls.append(shape)
        return original(shape, transform)

    monkeypatch.setattr(online, "_native_bounds", counted)
    scene = make_scene(tmp_path)
    assert len(calls) == 2
    assert all(not bounds.flags.writeable for bounds in scene._native_local_bounds.values())
    scene.owners = (owner("first_target"), owner("second_target"))

    def unexpected(*args, **kwargs):
        raise AssertionError("Every nonempty native field query receives scene-owned bounds")

    monkeypatch.setattr(native, "_native_bounds", unexpected)
    first = evaluate(scene, 4)
    assert len(calls) == 2 + 2 * 4
    second = evaluate(scene, 4)
    assert len(calls) == 2 + 2 * 4 * 2
    assert first.keys() == second.keys()
    for key in first:
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)


def test_empty_background_needs_no_world_range(tmp_path, monkeypatch):
    scene = make_scene(tmp_path)

    def unexpected(*args, **kwargs):
        raise AssertionError("An empty field has no native world-range work")

    monkeypatch.setattr(online, "_native_bounds", unexpected)
    monkeypatch.setattr(native, "_native_bounds", unexpected)
    result = evaluate(scene, 3)
    assert all(torch.count_nonzero(value) == 0 for value in result.values())
# END no result depends on approximate pose reuse or another query's history.
