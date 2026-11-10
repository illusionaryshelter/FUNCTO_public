"""Accepted sequence boundaries, later collisions and exact robot identity."""

# BEGIN exercise the actual scene, native models and serialized pair evidence.
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from manipulation.pair_evidence import PairBatch
from test_native_scene_contract import make_scene


def declaration():
    """A completed execution contributes measured robot coordinates only."""
    return dict(joint_names=['left_slide', 'right_slide'], q=[.04, -.02], dq=[0., 0.],
        feedback_id='accepted-boundary', timestamp_s=2., source='predecessor/final-feedback.json')


def test_initial_scene_is_inherited_but_every_positive_time_is_queried(tmp_path):
    scene = make_scene(tmp_path, occupied_background=True)
    scene.inherited_initial_state = declaration()
    # The smallest positive binary64 time is still a future query: inheritance
    # has no time window, margin, object identity or task-name switch.
    times = torch.tensor([[0., np.nextafter(0., 1.), 1.]], dtype=torch.float64)
    q = torch.tensor([[[.04, -.02]]], dtype=torch.float64).expand(1, 3, 2)
    position = torch.zeros((1, 3, 3), dtype=torch.float64)
    rotation = torch.tensor([1., 0., 0., 0.], dtype=torch.float64).expand(1, 3, 4)
    evidence = PairBatch()
    costs = scene.components(None, position, rotation, torch.zeros_like(times), q,
        ('left_slide', 'right_slide'), trajectory_time=times, geometry_evidence_out=evidence)
    assert costs['collision/robot/background'][0, 0] == 0.
    assert torch.all(costs['collision/robot/background'][0, 1:] > 0.)
    assert torch.all(costs['collision/robot/self'] == 0.)
    observed = next(index for index, pair in enumerate(evidence.descriptors)
                    if pair.component == 'collision/robot/background')
    assert evidence.materialize(0, observed)['status'] == 'inherited_initial_state'
    assert evidence.materialize(1, observed)['status'] == 'intersection'


def test_inheritance_never_waives_official_robot_self_collision(tmp_path):
    from test_native_scene_contract import evaluate
    scene = make_scene(tmp_path, gap=-.001, occupied_background=True)
    scene.inherited_initial_state = declaration()
    costs = evaluate(scene)
    assert costs['collision/robot/self'].item() > 0.
    assert costs['collision/robot/background'].item() == 0.


@pytest.mark.parametrize('changed', [None, 'q', 'dq'])
def test_inherited_state_binds_exact_feedback_and_survives_serialization(tmp_path, changed):
    scene = make_scene(tmp_path)
    scene.inherited_initial_state = declaration()
    state = declaration()
    if changed is not None:
        state[changed][0] = np.nextafter(state[changed][0], 1.)
    problem = SimpleNamespace(reference_frame='world', joint_names=tuple(state['joint_names']),
        start_q=np.array(state['q']), start_dq=np.array(state['dq']), conditions=(),
        pose_events=(), events=())
    if changed is None:
        scene.validate_problem(problem)
        record = scene.to_execution_binding()
        assert record['inherited_initial_state'] == declaration()
        scene.inherited_initial_state['q'][0] = .2
        assert record['inherited_initial_state']['q'][0] == .04
    else:
        with pytest.raises(ValueError, match='same measured q/dq'):
            scene.validate_problem(problem)
# END inherited geometry and actual robot checks keep distinct responsibilities.
