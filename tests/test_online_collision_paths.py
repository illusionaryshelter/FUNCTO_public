"""Same-query world sphere paths preserve hard scene laws and pose gradients."""

# BEGIN CPU regressions for optimizer-only per-sphere scene exports.
import numpy as np
import pytest
import torch

from manipulation.online_geometry import FrozenPrimitives, OnlinePlanningScene
from manipulation.planning_problem import PosePath
from manipulation.planning_scene import Attachment, CollisionOwner, ObservedSpace
from test_online_self_pair_cache import make_scene as hand_scene
from test_initial_support import scene as initial_support_scene
from test_online_contact import field as contact_field, rule as contact_rule


def owner(name='part', *, attachments=(), poses=None, excluded=()):
    query = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.2]*3,
        'pose': [0., 0., 0., 1., 0., 0., 0.]}], owner_ids=(name,), source='fixed test geometry')
    return CollisionOwner(name, query, PosePath(np.array([0., 1.]),
        np.stack((np.eye(4), np.eye(4))) if poses is None else poses, 'declared owner motion'),
        'test model', True, local_spheres=[[.03, 0., 0., .04]],
        attachments=attachments, robot_excluded_spheres=excluded,
        robot_exclusion_source='fixed installation' if excluded else None,
        robot_exclusion_geometry_source='test robot' if excluded else None)


def model(*, owners=(), contacts=(), radii=(.02, .03), materials=None):
    background = FrozenPrimitives(primitives=[{'kind': 'cuboid', 'dimensions': [.2]*3,
        'pose': [0., 0., 0., 1., 0., 0., 0.]}], owner_ids=('background',), source='test background')
    result = OnlinePlanningScene(background=background, owners=owners,
        coverage=ObservedSpace(np.ones((11,)*3, bool), origin=[-.5]*3, voxel_size=.1,
            observation_ids=('test',), reference_frame='world'), reference_frame='world',
        robot_sphere_materials=tuple(None for _ in radii) if materials is None else materials,
        robot_geometry_source='test robot', has_gripper=False, contacts=contacts,
        native_geometry_materials={'pad_box': 'pad'},
        unknown_policy={'outside_regions': 'forbid', 'conditional_regions': []})
    result.bind_online_robot(robot_sphere_geometry=[{'link_name': 'link',
        'sphere_indices': list(range(len(radii))), 'centers_local': [[0., 0., 0.]]*len(radii),
        'radii': list(radii)}], robot_self_collision_pairs=[])
    return result


def states(radii=(.02, .03), *, progress=(.1, .5, .9)):
    step = torch.tensor([progress], dtype=torch.float64)
    body = torch.zeros((1, len(progress), len(radii), 4), dtype=torch.float64)
    body[..., 0] = .08
    body[..., 3] = torch.tensor(radii, dtype=torch.float64)
    body.requires_grad_()
    p = torch.zeros((1, len(progress), 3), dtype=torch.float64, requires_grad=True)
    q = torch.zeros((1, len(progress), 4), dtype=torch.float64)
    q[..., 0] = 1.
    return body, p, q, step


def assert_hard_unchanged(scene, args, **kwargs):
    expected = scene.components(*args, **kwargs)
    paths = {}
    actual = scene.components(*args, **kwargs, collision_paths_out=paths)
    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.equal(expected[name], actual[name]), name
    for name, path in paths.items():
        torch.testing.assert_close(path.residual.sum(-1), actual[name], rtol=0, atol=0)
    return actual, paths


def test_export_keeps_installation_masks_disabled_slots_and_exact_hard_gradients():
    scene = model(owners=(owner(excluded=(0,)),), radii=(.02, -.1, .03))
    args = states((.02, -.1, .03))
    terms, paths = assert_hard_unchanged(scene, args)
    assert set(paths) == {'collision/robot/background', 'collision/robot/part'}
    target = paths['collision/robot/part']
    assert target.residual.shape == (1, 3, 2)
    assert not target.residual[..., 0].any()
    torch.testing.assert_close(target.radii, args[0].new_tensor([.02, .03]))
    assert not target.radii.requires_grad
    torch.testing.assert_close(target.centres, args[0][..., [0, 2], :3])
    actual_gradient = torch.autograd.grad(target.residual.sum(), args[0], retain_graph=True)[0]
    expected_gradient = torch.autograd.grad(terms['collision/robot/part'].sum(), args[0])[0]
    assert torch.equal(actual_gradient, expected_gradient)


def test_moving_target_inverse_rotation_and_translation_keep_their_gradients():
    held = Attachment((0., 1.), [0., 0., 0.], [1., 0., 0., 0.], 'held', None)
    scene = model(owners=(owner(attachments=(held,)),))
    body, p, q, step = states()
    angle = torch.tensor([[0., .4, .8]], dtype=torch.float64, requires_grad=True)
    q = torch.stack((torch.cos(angle/2), torch.zeros_like(angle),
                     torch.zeros_like(angle), torch.sin(angle/2)), -1)
    paths = {}
    scene.components(body, p, q, step, collision_paths_out=paths)
    relative = paths['collision/robot/part'].centres
    torch.testing.assert_close(relative[..., 0], .08*torch.cos(angle)[..., None].expand(1, 3, 2))
    torch.testing.assert_close(relative[..., 1], -.08*torch.sin(angle)[..., None].expand(1, 3, 2))
    angle_gradient, translation_gradient = torch.autograd.grad(relative[..., 1].sum(), (angle, p))
    torch.testing.assert_close(angle_gradient, -.16*torch.cos(angle))
    assert translation_gradient.abs().sum() > 0
    # Pure tool rotation moves an offset carried sphere despite zero TCP translation.
    payload = paths['collision/part/background']
    assert torch.linalg.vector_norm(payload.centres[:, 1:]-payload.centres[:, :-1]).item() > 0


def test_attachment_release_modes_and_pose_override_follow_the_actual_pose_law():
    relation = Attachment((.2, .8), [0., 0., 0.], [1., 0., 0., 0.], 'held', 'released')
    scene = model(owners=(owner(attachments=(relation,)),))
    args = states(progress=(.1, .2, .8, .9))
    _, paths = assert_hard_unchanged(scene, args)
    assert paths['collision/part/background'].mode_keys[0].squeeze(-1).tolist() == [[0, 1, 2, 2]]
    assert paths['collision/robot/part'].mode_keys[0].squeeze(-1).tolist() == [[0, 1, 2, 2]]
    overridden = {}
    scene.components(*args, owner_poses={'part': np.eye(4)}, collision_paths_out=overridden)
    # Externally measured owner poses bypass the nominal attachment branches.
    assert len(overridden['collision/part/background'].mode_keys) == 1


def test_uncertain_fingers_use_fixed_radius_and_do_not_change_known_arm_modes():
    scene = hand_scene(moving=True)
    # The calibrated gripper replaces a disabled native placeholder. Its
    # physical radius must remain available independently of that sentinel.
    scene.bind_online_robot(robot_sphere_geometry=[{'link_name': 'link',
        'sphere_indices': [0, 1, 2], 'centers_local': [[0., 0., 0.]]*3,
        'radii': [-1., .02, .02]}], robot_self_collision_pairs=[[1, 2], [0, 1]])
    scene.background = model().background
    args = states((.02, .02, .02), progress=(0., .5))
    terms, paths = assert_hard_unchanged(scene, args, trajectory_time=args[-1])
    path = paths['collision/robot/background']
    assert path.residual[0, 1, 0] == 0
    assert terms['conditional/collision/robot/background'][0, 1] > 0
    torch.testing.assert_close(path.radii, args[0].new_tensor([.02, .02, .02]))
    assert path.mode_keys[0].tolist() == [[[False, False, False], [True, False, False]]]
    assert 'collision/robot/self' not in paths
    # A native disabled runtime sphere still keeps its bound physical scale;
    # its mode changes and its residual vanishes, without inventing motion.
    changed = args[0].detach().clone()
    changed[0, 1, 1, 3] = -1.
    _, changed_paths = assert_hard_unchanged(scene, (changed, *args[1:]), trajectory_time=args[-1])
    assert not changed_paths['collision/robot/background'].mode_keys[-1][0, 1, 1]
    assert changed_paths['collision/robot/background'].radii[1] == .02


def test_contact_side_permission_and_initial_support_have_distinct_mode_keys():
    contact = contact_rule(sphere_indices=(0,))
    target = owner()
    target.query = contact_field(points=((-0.006, 0., 0.),))
    scene = model(owners=(target,), contacts=(contact,), materials=('pad', None))
    args = states(progress=(.1, .5, .9))
    _, paths = assert_hard_unchanged(scene, args)
    permission = paths['collision/robot/part'].mode_keys[0]
    assert permission.tolist() == [[[False, False], [True, False], [False, False]]]
    # The boundary helper may grant exact-arrival material permission while
    # the owner pose uses the left mode; export the actual selector it used.
    altered = {}
    scene.components(*args, contact_permission_progress=torch.full_like(args[-1], .5),
                     collision_paths_out=altered)
    assert altered['collision/robot/part'].mode_keys[0][0, :, 0].all()
    support = initial_support_scene()
    support_args = states((.001,), progress=(.5, 1.))
    _, support_paths = assert_hard_unchanged(support, support_args)
    keys = support_paths['collision/payload/support'].mode_keys
    assert keys[0].squeeze(-1).tolist() == [[1, 1]]  # Continued holding has no release.
    assert keys[1].squeeze(-1).tolist() == [[True, False]]


def test_export_requires_bound_geometry_and_cannot_mix_native_confirmation():
    scene = hand_scene(bound=False)
    args = states((.02, .02, .02))
    with pytest.raises(ValueError, match='bound calibrated'):
        scene.components(*args, collision_paths_out={})
    scene = model()
    with pytest.raises(ValueError, match='native confirmation'):
        scene.components(*states(), collision_paths_out={}, native_confirmation=lambda *args: {})


def test_path_memory_is_explicit_and_scales_with_actor_target_sphere_pairs():
    scene = model(owners=(owner(),))
    ordinary = scene.estimated_workspace_allocation(2, 3)
    exported = scene.estimated_workspace_allocation(2, 3, collision_paths=True)
    larger = scene.estimated_workspace_allocation(2, 6, collision_paths=True)
    ordinary_larger = scene.estimated_workspace_allocation(2, 6)
    for name in ('cpu_live', 'gpu_live'):
        assert exported[name] > ordinary[name]
        # Raw exports/complete gradients grow with the sample count, while
        # checkpoint scratch stops growing when its bounded block is full.
        assert exported[name]-ordinary[name] <= larger[name]-ordinary_larger[name]
        assert larger[name]-ordinary_larger[name] <= 2*(exported[name]-ordinary[name])
# END every test preserves execution's original scalar component values.
