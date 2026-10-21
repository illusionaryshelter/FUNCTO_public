"""Fixed geometry objectives and independent same-query hard evidence."""

import numpy as np
import pytest
import torch

from manipulation.local_geometry_model import LocalGeometryModels, LocalPairModel
from manipulation.observed_mapping import FrozenObservedField
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import CollisionOwner, Attachment, ContactRule
from test_native_scene_contract import make_scene


def _quaternion(angles):
    """Smooth unit quaternions for three independent rotation components."""
    half = angles / 2
    c, s = torch.cos(half), torch.sin(half)
    return torch.stack((c[..., 0]*c[..., 1]*c[..., 2]+s[..., 0]*s[..., 1]*s[..., 2],
                        s[..., 0]*c[..., 1]*c[..., 2]-c[..., 0]*s[..., 1]*s[..., 2],
                        c[..., 0]*s[..., 1]*c[..., 2]+s[..., 0]*c[..., 1]*s[..., 2],
                        c[..., 0]*c[..., 1]*s[..., 2]-s[..., 0]*s[..., 1]*c[..., 2]), -1)


def test_two_moving_entities_have_the_derivative_of_their_fixed_scalar():
    normal = np.array([.3, .4, .5]); normal /= np.linalg.norm(normal)
    model = LocalPairModel((.02, -.01, .03), (.01, .03, -.02), tuple(normal), True, .01)
    q = torch.tensor([[.01, -.02, .03, .2, -.1, .05, .04, .01, .02, -.1, .15, -.2],
                      [-.03, .04, .01, -.3, .2, .1, .02, -.02, .05, .3, -.1, .2]],
                     dtype=torch.float64, requires_grad=True)
    def value(data):
        return model.value((data[:, :3], _quaternion(data[:, 3:6])),
                           (data[:, 6:9], _quaternion(data[:, 9:12])))
    assert torch.autograd.gradcheck(value, q, eps=1e-6, atol=1e-8, rtol=1e-5)
    derivative = torch.autograd.functional.jacobian(value, q)
    assert torch.count_nonzero(derivative[0, 1]) == 0
    assert torch.count_nonzero(derivative[1, 0]) == 0


def _plane(name="observed_plane"):
    values = np.broadcast_to(np.array([-.04, .01, .06])[:, None, None], (3, 2, 2)).copy()
    return FrozenObservedField(values, np.ones_like(values, bool), weights=np.ones_like(values),
        origin=[-.04, .975, -.025], voxel_size=.05, name=name, source="RGB-D")


def _query(scene, x, models=None, *, role="optimization", components=None):
    position = torch.stack((x, x*0+1, x*0), -1)[:, None]
    quaternion = torch.zeros((*position.shape[:-1], 4), dtype=x.dtype)
    quaternion[..., 0] = 1
    q = torch.stack((x*0+.04, x*0-.02), -1)[:, None]
    progress = torch.full(position.shape[:-1], .5, dtype=x.dtype)
    return scene.evaluate(None, position, quaternion, progress, q,
        ("left_slide", "right_slide"), trajectory_time=progress,
        query_role=role, local_models=models, components_out=components)


def _moving_scene(tmp_path):
    scene = make_scene(tmp_path, fixed=("first", "second"))
    scene.background = _plane()
    field = FrozenObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        weights=np.ones((2, 2, 2)), origin=[-.01]*3, voxel_size=.02,
        name="material", source="RGB-D")
    scene.owners = (CollisionOwner("material", field,
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed material", moving=True,
        attachments=(Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], "declared", None),)),)
    scene.unknown_policy = {"conditional_regions": [], "outside_regions": "allow_attempt"}
    return scene


def test_clear_trial_keeps_same_local_cost_and_new_collision_keeps_hard_evidence(tmp_path):
    scene, models = _moving_scene(tmp_path), LocalGeometryModels()
    base = torch.tensor([-.005, -.005, .04, .04], dtype=torch.float64)
    models.begin_capture(2, 2)
    _query(scene, base, models)
    assert models.finish_capture().tolist() == [True, True]
    trial = torch.tensor([-.004, .04, -.005, .03], dtype=torch.float64, requires_grad=True)
    hard = {}
    cost = _query(scene, trial, models, components=hard)
    collision = hard["collision/material/background"][:, 0]
    assert collision.tolist() == [1., 0., 1., 0.]
    # The first seed keeps its positive separation preference after clearing.
    assert cost[1, 0] > 0
    # The second seed's new collision is detected without inventing a local
    # target during a trial. The next actual-base refresh can capture it.
    assert cost[2, 0] == 0
    derivative = torch.autograd.grad(cost.sum(), trial)[0]
    assert derivative[1] < 0
    epsilon = 1e-6
    delta = torch.zeros_like(trial); delta[1] = epsilon
    numerical = ((_query(scene, trial+delta, models)-_query(scene, trial-delta, models))
                 / (2*epsilon))[1, 0]
    torch.testing.assert_close(derivative[1], numerical, atol=1e-8, rtol=1e-5)
    models.begin_capture(2, 2)
    _query(scene, torch.tensor([.04, .04, -.005, -.005], dtype=torch.float64), models)
    assert models.finish_capture().tolist() == [True, True]
    assert _query(scene, trial, models)[2, 0] > 0


def test_capture_lifecycle_distinguishes_empty_models_and_relation_sides():
    models = LocalGeometryModels()
    with pytest.raises(RuntimeError, match="explicit base capture"):
        models.state_key(0, 0., .5, .5)
    models.begin_capture(1, 2)
    key = models.state_key(0, 1., .5, .5)
    models.state(key)
    assert models.state_key(1, 1., .5, .5) == key
    assert models.finish_capture().tolist() == [True]
    assert models.state(key).pairs == {}
    with pytest.raises(RuntimeError, match="absent from its base capture"):
        models.state(models.state_key(0, 1., np.nextafter(.5, 0.), .5))
    models.begin_capture(1, 2)
    models.state(key)
    assert models.finish_capture().tolist() == [False]
    # A cancelled refresh releases temporary records and keeps the complete
    # previous layout. The caller still propagates its original interruption.
    models.begin_capture(2, 3)
    models.state(models.state_key(3, 2., .75, .75))
    models.abort_capture()
    assert (models.num_seeds, models.num_particles) == (1, 2)
    assert not models.capturing
    assert models.state(key).pairs == {}


def test_required_contact_uses_the_fixed_zero_point_region_excess(tmp_path):
    scene = make_scene(tmp_path, fixed=("first", "second"))
    scene.owners = (CollisionOwner("target", _plane("target"),
        PosePath(np.array([0., 1.]), np.stack((np.eye(4), np.eye(4))), "declared path"),
        "observed target", moving=False),)
    rule = ContactRule("feature", "robot", "target", (0., 1.), "declared_material",
        [[0., -.005, -.005], [0., .005, -.005], [0., .005, .005], [0., -.005, .005]],
        [-1., 0., 0.], [0., 1., 0.], [[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]],
        [.02, .02], (-.001, .001), .01, 0., 0., "source feature",
        required=True, robot_body="first")
    scene.contacts = (rule,)
    models = LocalGeometryModels()
    models.begin_capture(1, 1)
    _query(scene, torch.tensor([.03], dtype=torch.float64), models)
    models.finish_capture()
    point = torch.tensor([.02], dtype=torch.float64, requires_grad=True)
    hard = {}
    value = _query(scene, point, models, components=hard)
    assert hard["contact/feature"].item() == 1.
    assert 0 < value.item() < .1
    derivative = torch.autograd.grad(value.sum(), point)[0]
    numerical = (_query(scene, point+1e-6, models)-_query(scene, point-1e-6, models))/(2e-6)
    torch.testing.assert_close(derivative, numerical.reshape(-1), atol=1e-8, rtol=1e-5)
    assert derivative.item() > 0


def test_optimization_requires_an_explicit_block_model(tmp_path):
    with pytest.raises(ValueError, match="explicitly captured"):
        _query(_moving_scene(tmp_path), torch.tensor([.02], dtype=torch.float64))
