"""Original scene relationship values, derivatives, events and hand certainty.

Maintained generic fields, ContactRule, PreparedInitialSupport and nominal
finger APIs provide the numerical evidence. No recorded task scene is loaded.
"""

# BEGIN relationship-only composition is checked against the original APIs.
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from geometry_objective_fixtures import _moving_scene
from manipulation.initial_support import PreparedInitialSupport
from manipulation.planning_scene import Attachment, ContactRule
from manipulation.sampled_geometry_objective import MaterialSamples
from manipulation.direct_planner.scene_relationships import scene_relationship_objective
import test_initial_support as support_fixture


def contact_rule():
    """A finite observed patch and opposing material face in metre units."""
    return ContactRule(rule_id='surface', actor='material', target_owner='background',
        interval=(0., 1.), material_id='declared_face', required=True,
        feature_points=[[0., -.04, -.04], [0., .04, -.04],
                        [0., .04, .04], [0., -.04, .04]],
        feature_normal=[-1., 0., 0.], patch_origin=[0., 1., 0.],
        patch_axes=np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]]),
        patch_half_size=[.04, .04], gap_bounds=(-.001, .001),
        normal_tolerance_rad=.1, modeling_allowance_m=.001, geometry_error_m=.001,
        source='declared finite contact fixture')


def contact_inputs(value, progress):
    """Independent x/rotation rows preserve the caller's full event layout."""
    x, angle = value.unbind(-1)
    position = torch.stack((x, x*0+1., x*0), -1)[None]
    quaternion = torch.stack((torch.cos(angle/2), angle*0, angle*0,
                              torch.sin(angle/2)), -1)[None]
    q = torch.stack((x*0+.04, x*0-.02), -1)[None]
    return position, quaternion, progress, q, ('left_slide', 'right_slide')


def original_contact_part(scene, arguments):
    """Subtract identical collision terms by changing only the required flag."""
    rules = scene.contacts
    links = scene.kinematics.poses(arguments[3], arguments[4])
    with_contact = scene.optimization_objective(*arguments,
        trajectory_time=arguments[2], robot_link_poses=links)
    scene.contacts = tuple(replace(rule, required=False) for rule in rules)
    collision = scene.optimization_objective(*arguments,
        trajectory_time=arguments[2], robot_link_poses=links)
    scene.contacts = rules
    return with_contact-collision


@pytest.mark.parametrize('release', [False, True])
def test_required_contact_matches_original_value_gradient_and_first_release(tmp_path, release):
    scene = _moving_scene(tmp_path)
    scene.contacts = (contact_rule(),)
    if release:
        scene.owners[0].attachments = (Attachment((0., .5), [0., 0., 0.],
            [1., 0., 0., 0.], 'declared', 'release'),)
    progress = torch.tensor([[.25, .5, .5, .75]], dtype=torch.float64)
    values = torch.tensor([[.03, .27], [.04, .31], [.08, .4], [.09, .5]],
                           dtype=torch.float64, requires_grad=True)
    arguments = contact_inputs(values, progress)
    expected = original_contact_part(scene, arguments)
    inventory = scene._objective_material_samples
    report = {}
    actual = scene_relationship_objective(scene, *arguments,
        trajectory_time=progress, material_samples=inventory, report_out=report)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    expected_gradient, = torch.autograd.grad(expected, values, retain_graph=True)
    gradient, = torch.autograd.grad(actual, values)
    torch.testing.assert_close(gradient, expected_gradient, atol=1e-11, rtol=1e-11)
    assert actual > 0. and torch.count_nonzero(gradient[:, 1]) > 0
    assert report['active_contact_state_count'] == 4
    assert scene._objective_material_samples is inventory
    if release:
        assert torch.count_nonzero(gradient[2:]) == 0
        assert report['relationship_components']['relationship/missing_release/material/0'] == [[0.]*4]

    def objective(value):
        return scene_relationship_objective(scene, *contact_inputs(value, progress),
            trajectory_time=progress, material_samples=inventory)

    assert torch.autograd.gradcheck(objective, (values.detach().requires_grad_(),),
                                    eps=1e-6, atol=1e-7, rtol=1e-5)


def test_initial_support_matches_original_relationship_mean_and_gradient(tmp_path, monkeypatch):
    import manipulation.sampled_geometry_objective as original
    scene = _moving_scene(tmp_path)
    actor, target = support_fixture.owners()
    scene.owners = (actor, target)
    model = PreparedInitialSupport(support_fixture.relation(), actor, target, 1e-6)
    scene._initial_support_models = {model.relation.relation_id: model}
    progress = torch.tensor([[.1, .5, 1.]], dtype=torch.float64)
    height = torch.tensor([-.001, -.002, -.003], dtype=torch.float64, requires_grad=True)
    position = torch.stack((height*0, height*0, height), -1)[None]
    quaternion = height.new_tensor([1., 0., 0., 0.]).expand(1, 3, 4)
    q = height.new_tensor([.04, -.02]).expand(1, 3, 2)
    names = ('left_slide', 'right_slide')
    arguments = position, quaternion, progress, q, names
    # Isolate the existing relation branch inside the original complete
    # objective. The maintained support model and its derivatives run intact.
    monkeypatch.setattr(original, 'pair_objective', lambda *args, **kwargs: position.sum()*0)
    monkeypatch.setattr(original, 'native_pair_objective', lambda *args, **kwargs: position.sum()*0)
    expected = scene.optimization_objective(*arguments, trajectory_time=progress,
        robot_link_poses=scene.kinematics.poses(q, names))
    report = {}
    actual = scene_relationship_objective(scene, *arguments,
        trajectory_time=progress, material_samples={}, report_out=report)
    torch.testing.assert_close(actual, expected, rtol=0., atol=1e-16)
    expected_gradient, = torch.autograd.grad(expected, height, retain_graph=True)
    gradient, = torch.autograd.grad(actual, height)
    torch.testing.assert_close(gradient, expected_gradient, rtol=0., atol=1e-12)
    assert actual > 0. and gradient[0] < 0. and gradient[1] < 0. and gradient[2] == 0.
    assert report['initial_support_count'] == 1 and report['state_count'] == 3


def test_nominal_hand_certainty_retains_conditional_and_missing_surface_diagnostics(tmp_path):
    from manipulation.gripper_trajectory import FingerTrajectory, NominalFingerGeometry
    from manipulation.planning_scene import PrismaticGripper
    from test_gripper_trajectory import _arguments
    scene = _moving_scene(tmp_path)
    reference = FingerTrajectory.build(**_arguments())
    nominal = NominalFingerGeometry.from_reference(reference, valid_until_s=.3,
                                                    source='declared finite fixture')
    scene.gripper = PrismaticGripper(controller='binary', joint_names=nominal.joint_names,
        current_positions=nominal.position[0], joint_limits=[[0., .04], [-.04, 0.]],
        joint_bounds=None, coupling_matrix=np.empty((0, 2)), coupling_offset=[],
        commands=(), source='declared nominal hand fixture',
        control_timestep_s=.002)
    scene.gripper.nominal_finger_trajectory = nominal
    mapping = np.zeros((4, 3, 2)); mapping[:, 0, 0] = 1.
    scene.contacts = (replace(contact_rule(), feature_joint_translation=mapping),)
    progress = torch.tensor([[.1, .2, .5]], dtype=torch.float64)
    values = torch.tensor([[.03, .2], [.04, .3], [.05, .4]],
                           dtype=torch.float64, requires_grad=True)
    arguments = contact_inputs(values, progress)
    report = {}
    value = scene_relationship_objective(scene, *arguments,
        trajectory_time=progress,
        material_samples={('field', 'background'): MaterialSamples(scene.background)},
        report_out=report)
    gradient, = torch.autograd.grad(value, values)
    assert value > 0. and report['active_contact_state_count'] == 2
    assert report['conditional_contact_state_count'] == 1
    assert torch.count_nonzero(gradient[2]) == 0
    empty_surface = MaterialSamples(scene.owners[0].query)
    assert len(empty_surface.zero_points) == 0
    report = {}
    scene_relationship_objective(scene, *arguments, trajectory_time=progress,
        material_samples={('field', 'background'): empty_surface}, report_out=report)
    assert report['contact_without_zero_samples'] == ['surface']
# END complete scene qualification retains its independent original evidence.
