"""CPU physical counterexamples for the shared signed scene interface."""
# BEGIN clear-side derivatives, fixed contacts, unknown fields and release inputs.
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.online_geometry import FrozenESDF
from manipulation.planning_scene import Attachment, ContactRule
from test_native_collision import model, body, owner_pose
from test_online_geometry import native_box_scene
from test_online_native_self import make_scene, rebind


def signed_native(current, q, matrices, variable, enabled=None, *, owner_sources=None):
    if enabled is None:
        enabled = torch.ones((*q.shape[:2], len(current.pairs)), dtype=torch.bool)
    if owner_sources is None:
        owner_sources = torch.full(matrices.shape[:3], -1, dtype=torch.long, device=q.device)
    return current.signed(q, matrices, enabled, variable, owner_sources=owner_sources)


@pytest.mark.parametrize('target,clearance', [(0.04, 0.02), (0.015, -0.005)])
def test_native_distance_keeps_both_signed_sides_and_same_query_derivatives(target, clearance):
    current = model()
    q = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)
    matrices = owner_pose(target).requires_grad_()
    variable = torch.ones((1, 1, len(current.pairs)), dtype=torch.bool)
    values, valid = signed_native(current, q, matrices, variable)
    group = current.groups.index('collision/robot/obstacle')
    assert not valid[0, 0, group, 0]
    assert valid[0, 0, group, 1]
    value = values[0, 0, group, 1]
    assert value.item() == pytest.approx(clearance, abs=2e-8)
    value.backward()
    assert q.grad[0, 0, 1].item() == pytest.approx(-1., abs=1e-8)
    assert matrices.grad[0, 0, 0, 0, 3].item() == pytest.approx(1., abs=1e-8)
    h = 1e-6
    positive, negative = q.detach().clone(), q.detach().clone()
    positive[..., 1] += h
    negative[..., 1] -= h
    difference = (signed_native(current, positive, matrices.detach(), variable)[0][0, 0, group, 1]
                  - signed_native(current, negative, matrices.detach(), variable)[0][0, 0, group, 1])/(2*h)
    assert q.grad[0, 0, 1].item() == pytest.approx(difference.item(), abs=2e-7)


# BEGIN registered frame bindings share values and derivatives at every scale.
@pytest.mark.parametrize('offset', [.015, .0203708059])
def test_native_bound_owner_keeps_same_rigid_gap_for_large_joint_angles(offset):
    current = model(owner_frames={'obstacle': 'slider'})
    q = torch.tensor([[[.23, .07], [80787.123456, -.02]],
                      [[-.32, .1], [-80786.234567, .03]]],
                     dtype=torch.float64, requires_grad=True)
    matrices = owner_pose(offset).expand(2, 2, 1, 4, 4).clone()
    sources = torch.arange(2).reshape(1, 2, 1).expand(2, 2, 1)
    variable = torch.zeros((2, 2, len(current.pairs)), dtype=torch.bool)
    values, valid = signed_native(current, q, matrices, variable, owner_sources=sources)
    group = current.groups.index('collision/robot/obstacle')
    assert valid[..., group, 0].all() and not valid[..., group, 1].any()
    torch.testing.assert_close(values[..., group, 0], q.new_full((2, 2), offset-.02),
                               rtol=0, atol=2e-8)
    values[..., group, 0].sum().backward()
    # The owner is built from the exact same registered frame as the robot
    # box. Even a large trial angle cannot alter their relative geometry.
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=2e-8)


def test_native_bound_release_derivatives_follow_source_samples_in_each_batch():
    base = np.eye(4)
    base[:3, :3] = Rotation.from_rotvec([.15, -.1, .22]).as_matrix()
    base[:3, 3] = [.11, -.07, .03]
    current = model(base=base, owner_frames={'obstacle': 'slider'})
    q = torch.tensor([[[.1, .04], [.2, .05], [.35, .06]],
                      [[-.2, .03], [-.3, .04], [-.15, .05]]],
                     dtype=torch.float64, requires_grad=True)
    local = np.eye(4)
    local[:3, 3] = [.045, .006, .003]
    local[:3, :3] = Rotation.from_rotvec([.13, -.17, .31]).as_matrix()
    matrices = q.new_tensor(local).expand(2, 3, 1, 4, 4).clone()
    sources = torch.tensor([[[0], [1], [1]], [[0], [1], [1]]])
    variable = torch.ones((2, 3, len(current.pairs)), dtype=torch.bool)
    group = current.groups.index('collision/robot/obstacle')

    def objective(coordinates):
        values, _ = signed_native(current, coordinates, matrices, variable, owner_sources=sources)
        return values[0, 2, group, 1]+2*values[1, 2, group, 1]

    objective(q).backward()
    torch.testing.assert_close(q.grad[:, 0], torch.zeros_like(q.grad[:, 0]), rtol=0, atol=1e-12)
    h = 1e-6
    # Rotation as well as translation must reach q(release), while the robot
    # still consumes q(current). Distinct batches cannot share either input.
    for batch in range(2):
        for sample in (1, 2):
            for coordinate in range(2):
                positive, negative = q.detach().clone(), q.detach().clone()
                positive[batch, sample, coordinate] += h
                negative[batch, sample, coordinate] -= h
                difference = (objective(positive)-objective(negative))/(2*h)
                assert q.grad[batch, sample, coordinate].item() == pytest.approx(difference.item(), abs=2e-7)


def test_native_registered_owner_can_consume_an_independent_world_pose():
    current = model(owner_frames={'obstacle': 'slider'})
    q = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)
    matrices = owner_pose(.04).requires_grad_()
    variable = torch.ones((1, 1, len(current.pairs)), dtype=torch.bool)
    sources = torch.full((1, 1, 1), -1, dtype=torch.long)
    values, _ = signed_native(current, q, matrices, variable, owner_sources=sources)
    group = current.groups.index('collision/robot/obstacle')
    assert values[0, 0, group, 1].item() == pytest.approx(.02, abs=2e-8)
    values[0, 0, group, 1].backward()
    assert q.grad[0, 0, 1].item() == pytest.approx(-1., abs=1e-8)
    assert matrices.grad[0, 0, 0, 0, 3].item() == pytest.approx(1., abs=1e-8)
# END source -1 is a physical world input, not a second attachment FK path.


def test_fixed_zero_contact_cannot_hide_another_pairs_variable_clearance():
    offset = np.eye(4)
    offset[0, 3] = .3
    current = model(shapes=[body('fixed', 'base'), body('moving', 'slider', pose=offset)])
    q = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)
    variable = torch.tensor([[[pair['first_geometry'] == 'moving' for pair in current.pairs]]])
    values, valid = signed_native(current, q, owner_pose(.02), variable)
    group = current.groups.index('collision/robot/obstacle')
    assert valid[0, 0, group].tolist() == [True, True]
    assert values[0, 0, group, 0].item() == pytest.approx(0., abs=2e-8)
    assert values[0, 0, group, 1].item() == pytest.approx(.26, abs=2e-8)
    values[0, 0, group, 1].backward()
    assert q.grad[0, 0, 1].item() == pytest.approx(1., abs=1e-8)


def test_declared_margin_is_included_in_aabb_minimum_ordering():
    def placed(name, x):
        pose = np.eye(4)
        pose[0, 3] = x
        return body(name, 'base', dimensions=(.02, .02, .02), pose=pose)
    current = model(fixed=True, owners={}, shapes=[placed('a', 0), placed('b', .05), placed('c', .2)],
        self_pairs=[dict(first='a', second='b', margin=0.), dict(first='a', second='c', margin=.3)])
    q = torch.empty((1, 1, 0), dtype=torch.float64)
    matrices = torch.empty((1, 1, 0, 4, 4), dtype=torch.float64)
    values, valid = signed_native(current, q, matrices, torch.zeros((1, 1, 2), dtype=torch.bool))
    assert valid[0, 0, 0].tolist() == [True, False]
    # The farther solid pair has smaller signed slack because the policy
    # contains a larger original margin; raw-gap sorting could prune it.
    assert values[0, 0, 0, 0].item() == pytest.approx(-.12, abs=2e-8)


def scene_query(current, q, *, progress=None, variable=None, sphere_offsets=None):
    batch, times, _ = q.shape
    progress = q.new_tensor(np.linspace(0, 1, times)[None]) if progress is None else progress
    position = torch.cat((q, q.new_zeros((batch, times, 2))), -1)
    count = len(current.robot_sphere_materials)
    centers = position[..., None, :].expand(batch, times, count, 3)
    if sphere_offsets is not None:
        centers = centers+q.new_tensor(sphere_offsets)
    spheres = torch.cat((centers, q.new_full((batch, times, count, 1), .09)), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(batch, times, 4)
    variable = torch.ones_like(q, dtype=torch.bool) if variable is None else variable
    return current.signed_constraints(spheres, position, quaternion, progress, q, ('slide',),
        joint_variable_mask=variable, trajectory_time=progress)


def test_scene_entry_is_fixed_but_later_collision_remains_variable_and_signed():
    current = native_box_scene()
    q = torch.tensor([[[0.], [.025]]], dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q, variable=torch.tensor([[[False], [True]]]))
    key = 'collision/robot/target'
    assert result.valid[key].tolist() == [[[True, False], [False, True]]]
    torch.testing.assert_close(result.slack[key], q.new_tensor([[[.02, 0.], [0., -.005]]]), rtol=0, atol=2e-8)
    assert result.variable[key].tolist() == [[[False, True], [False, True]]]
    result.slack[key][..., 1].sum().backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[0.], [-1.]]]), rtol=0, atol=1e-8)


def test_complete_contact_permission_produces_invalid_slot_not_zero_clearance():
    contact = ContactRule('attempt', 'robot', 'target', (.25, .75), (0,), 'whole registered component')
    current = native_box_scene(contact=contact)
    q = torch.full((1, 2, 1), .025, dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q, progress=q.new_tensor([[.5, .9]]))
    key = 'collision/robot/target'
    assert result.valid[key].tolist() == [[[False, False], [False, True]]]
    assert result.slack[key][0, 0].tolist() == [0., 0.]
    assert result.slack[key][0, 1, 1].item() == pytest.approx(-.005, abs=2e-8)


def test_unknown_esdf_rows_keep_fixed_shape_and_do_not_hide_valid_positive_distance():
    current = native_box_scene(target_x=.8, sphere_count=2)
    q = torch.tensor([[[0.], [.2]]], dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q, sphere_offsets=[[0., 0., 0.], [.3, 0., 0.]])
    key = 'collision/robot/background'
    assert result.slack[key].shape == result.valid[key].shape == (1, 2, 2)
    assert result.valid[key].tolist() == [[[False, True], [False, False]]]
    assert result.slack[key][0, 0, 1].item() == pytest.approx(.91)
    assert result.slack[key][0, 1].tolist() == [0., 0.]
    result.slack[key].sum().backward()
    assert torch.isfinite(q.grad).all()


def test_same_tcp_attachment_is_fixed_without_inspecting_its_zero_gradient():
    current = native_box_scene(target_x=.015, attached=True)
    q = torch.tensor([[[.02], [-.03]]], dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q)
    key = 'collision/robot/target'
    assert result.valid[key].tolist() == [[[True, False], [True, False]]]
    torch.testing.assert_close(result.slack[key][..., 0], q.new_full((1, 2), -.005), rtol=0, atol=2e-8)
    result.slack[key].sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=2e-8)


# BEGIN scene compilation preserves calibrated frame binding across FK inputs.
def test_native_scene_attachment_uses_registered_frame_and_calibrated_tcp_offset():
    current = native_box_scene(target_x=.015, attached=True)
    current.robot_model['T_frame_tcp'][0][3] = .003
    q = torch.tensor([[[.02], [80787.123456]]], dtype=torch.float64, requires_grad=True)
    progress = q.new_tensor([[0., 1.]])
    # Deliberately supplied lower-precision tool coordinates belong to the
    # independent ESDF sphere path. Native registered solids use named q.
    position = torch.cat((q.float().double(), q.new_zeros((1, 2, 2))), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(1, 2, 4)
    spheres = torch.cat((position[..., None, :], q.new_full((1, 2, 1, 1), .09)), -1)
    result = current.signed_constraints(spheres, position, quaternion, progress, q, ('slide',),
        joint_variable_mask=torch.ones_like(q, dtype=torch.bool), trajectory_time=progress)
    key = 'collision/robot/target'
    assert result.valid[key].tolist() == [[[True, False], [True, False]]]
    torch.testing.assert_close(result.slack[key][..., 0], q.new_full((1, 2), -.002), rtol=0, atol=2e-8)
    result.slack[key].sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=2e-8)
# END no collision margin or source primitive was changed to obtain invariance.


def test_released_owner_keeps_the_release_variables_at_later_query_times():
    current = native_box_scene(target_x=.1, attached=True)
    current.owners[0].attachments = (Attachment((0., .5), [.1, 0., 0.], [1., 0., 0., 0.], 'held', 'released'),)
    q = torch.tensor([[[0.], [.02], [.04]]], dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q, variable=torch.tensor([[[False], [True], [True]]]))
    key = 'collision/robot/target'
    # The release instant itself shares the same TCP input. Later the robot
    # consumes q(t), but the payload retains q(release), so that pair varies.
    assert result.valid[key].tolist() == [[[True, False], [True, False], [False, True]]]
    assert result.slack[key][0, 2, 1].item() == pytest.approx(.06, abs=2e-8)
    result.slack[key][0, 2, 1].backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[0.], [1.], [-1.]]]), rtol=0, atol=1e-8)
    assert not result.fixed_failures['unsupported/release_anchor/target/0'].any()


def test_missing_release_anchor_is_a_fixed_failure_not_an_optimizer_inequality():
    current = native_box_scene(target_x=.1, attached=True)
    current.owners[0].attachments = (Attachment((0., .5), [.1, 0., 0.], [1., 0., 0., 0.], 'held', 'released'),)
    q = torch.tensor([[[0.], [.04]]], dtype=torch.float64)
    result = scene_query(current, q)
    name = 'unsupported/release_anchor/target/0'
    assert result.fixed_failures[name].tolist() == [[0., 1.]]
    assert name not in result.slack


def hand_query(current, q, *, measured=False, variable=None, signed=True):
    """Use identical named coordinates for both public scene consumers."""
    progress = q.new_tensor([[0., .5]])
    position = torch.stack((q[..., 0], q[..., 0]*0, q[..., 0]*0), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(*q.shape[:2], 4)
    spheres = q.new_tensor([[0., 0., 0., np.sqrt(3)*.02], [.01, 0., 0., np.sqrt(3)*.02],
                           [.035, 0., 0., np.sqrt(3)*.02]]).expand(*q.shape[:2], 3, 4)
    kwargs = dict(trajectory_time=progress, active_contact_rule_ids=() if measured else None)
    if signed:
        kwargs['joint_variable_mask'] = torch.ones_like(q, dtype=torch.bool) if variable is None else variable
    method = current.signed_constraints if signed else current.components
    return method(spheres, position, quaternion, progress, q, ('arm', 'finger'), **kwargs)


@pytest.mark.parametrize('controller', ['binary', 'position_tracking'])
def test_measured_hand_uses_mechanical_limits_instead_of_a_command_band(controller):
    current = make_scene(independent_hand=True)
    current.robot_model['tool_frame'] = 'tool'
    hand = current.gripper
    hand.controller = controller
    if controller == 'position_tracking':
        hand.joint_bounds = lambda progress: (progress[..., None]*0-.01, progress[..., None]*0-.01)
    q = torch.tensor([[[0., .1], [0., .25]]], dtype=torch.float64, requires_grad=True)
    result = hand_query(current, q, measured=True)
    key, tolerance = 'gripper/controller', hand.numerical_joint_tolerance
    assert result.valid[key].tolist() == [[[False, True], [False, True]]]
    torch.testing.assert_close(result.slack[key][..., 1], q.new_tensor([[.1+tolerance, -.05+tolerance]]),
                               rtol=0, atol=2e-8)
    expected = hand_query(current, q, measured=True, signed=False)[key]
    torch.testing.assert_close(torch.relu(-result.slack[key][..., 1]), expected)
    result.slack[key][0, 1, 1].backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[0., 0.], [0., -1.]]]), rtol=0, atol=1e-8)


def test_tracking_hand_keeps_fixed_command_consistency_separate_from_variable_coordinates():
    current = make_scene(independent_hand=True)
    current.robot_model['tool_frame'] = 'tool'
    hand = current.gripper
    hand.controller = 'position_tracking'
    hand.joint_bounds = lambda progress: (progress[..., None]*0-.01, progress[..., None]*0-.01)
    tolerance = hand.numerical_joint_tolerance
    q = torch.tensor([[[0., -.01+.5*tolerance], [0., -.01+2*tolerance]]],
                     dtype=torch.float64, requires_grad=True)
    result = hand_query(current, q)
    key = 'gripper/controller'
    assert result.valid[key].all()
    torch.testing.assert_close(result.slack[key][..., 0], q.new_full((1, 2), tolerance))
    torch.testing.assert_close(result.slack[key][..., 1], q.new_tensor([[.5*tolerance, -tolerance]]))
    expected = hand_query(current, q, signed=False)[key]
    torch.testing.assert_close(torch.relu(-result.slack[key]).sum(-1), expected)
    result.slack[key][0, 1, 1].backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[0., 0.], [0., -1.]]]), rtol=0, atol=1e-8)
    # A non-singleton tracking command is an independent fixed input error;
    # actual coordinates at its midpoint cannot hide or optimize it away.
    hand.joint_bounds = lambda progress: (progress[..., None]*0-.02, progress[..., None]*0)
    inconsistent = hand_query(current, q.detach())
    assert torch.all(inconsistent.slack[key][..., 0] < 0)


def test_shared_ancestor_motion_cancels_for_a_native_self_pair_structurally():
    current = make_scene()
    rebind(current, [{'first': 'finger_solid', 'second': 'palm_solid', 'margin': 0.}])
    q = torch.tensor([[[0., -.01], [.1, -.01]]], dtype=torch.float64, requires_grad=True)
    variable = torch.tensor([[[True, False], [True, False]]])
    result = hand_query(current, q, variable=variable)
    key = 'collision/robot/self'
    assert result.valid[key].tolist() == [[[True, False], [True, False]]]
    torch.testing.assert_close(result.slack[key][..., 0], q.new_full((1, 2), -.02), rtol=0, atol=2e-8)
    result.slack[key].sum().backward()
    assert torch.all(q.grad[..., 0] == 0)


def test_released_owner_esdf_retains_anchor_gradient_and_fixed_history_removes_it():
    current = native_box_scene(target_x=.05, attached=True)
    current.owners[0].attachments = (Attachment((0., .5), [.05, 0., 0.], [1., 0., 0., 0.], 'held', 'released'),)
    axis = np.linspace(-.5, .5, 11, dtype=np.float32)
    values = np.broadcast_to(axis[:, None, None]+.2, (11, 11, 11)).copy()
    current.background = FrozenESDF(values, origin=[-.5]*3, voxel_size=.1,
                                   owner_ids=('background',), source='affine signed field')
    q = torch.tensor([[[0.], [.02], [.04]]], dtype=torch.float64, requires_grad=True)
    result = scene_query(current, q, variable=torch.tensor([[[False], [True], [True]]]))
    key = 'collision/target/background'
    assert result.valid[key].tolist() == [[[True, False], [False, True], [False, True]]]
    assert result.slack[key][0, 2, 1].item() == pytest.approx(.22, abs=2e-8)
    result.slack[key][0, 2, 1].backward()
    torch.testing.assert_close(q.grad, q.new_tensor([[[0.], [1.], [0.]]]), rtol=0, atol=1e-7)
    position = torch.cat((q.detach(), q.new_zeros((1, 3, 2))), -1)
    quaternion = q.new_tensor([1., 0., 0., 0.]).expand(1, 3, 4)
    fixed = current.with_release_history(position, quaternion, q.new_tensor([[0., .5, 1.]]))
    later = scene_query(fixed, q.detach()[:, :2], progress=q.new_tensor([[.6, .8]]))
    assert later.valid[key].tolist() == [[[True, False], [True, False]]]
    torch.testing.assert_close(later.slack[key][..., 0], q.new_full((1, 2), .22), rtol=0, atol=2e-8)
    assert not any(name.startswith('unsupported/release_anchor') for name in later.fixed_failures)
# END all fixtures are generic model/permission contracts, not skill exceptions.
