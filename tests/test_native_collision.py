"""CPU counterexamples for one registered-solid value and witness gradient."""
# BEGIN proxy false positives, true collisions and both moving-body derivatives.
from itertools import product

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
import torch

from manipulation.native_collision import NativeCollisionModel


URDF = '''<robot name="generic_articulated_probe">
  <link name="base"/><link name="rotating"/><link name="slider"/>
  <joint name="turn" type="revolute"><parent link="base"/><child link="rotating"/>
    <axis xyz="0 0 1"/><limit lower="-2" upper="2" effort="10" velocity="1"/></joint>
  <joint name="slide" type="prismatic"><parent link="rotating"/><child link="slider"/>
    <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/></joint>
</robot>'''


def body(name='probe', frame='slider', dimensions=(.02, .16, .025), pose=None):
    return dict(name=name, frame=frame, kind='box', dimensions=list(dimensions),
                T_frame_geometry=(np.eye(4) if pose is None else pose).tolist(), source='synthetic registered box')


def primitive(dimensions=(.02, .08, .04), kind='cuboid', pose=None):
    return dict(kind=kind, dimensions=list(dimensions),
                pose=[0., 0., 0., 1., 0., 0., 0.] if pose is None else pose)


def model(*, shapes=None, owners=None, self_pairs=(), moving_owners=(), fixed=False,
          names=('turn', 'slide'), base=None, owner_frames=None):
    return NativeCollisionModel(robot_model=dict(
        urdf_xml='<robot name="fixed"><link name="base"/></robot>' if fixed else URDF,
        geometries=[body(frame='base' if fixed else 'slider')] if shapes is None else shapes),
        joint_names=() if fixed else names, T_reference_base=np.eye(4) if base is None else base,
        self_pairs=self_pairs, owners={'obstacle': [primitive()]} if owners is None else owners,
        moving_owners=moving_owners, owner_frames=owner_frames)


def query(current, q, matrices, enabled=None, *, owner_sources=None):
    if enabled is None:
        enabled = torch.ones((*q.shape[:2], len(current.pairs)), dtype=torch.bool)
    if owner_sources is None:
        owner_sources = torch.full(matrices.shape[:3], -1, dtype=torch.long, device=q.device)
    return current(q, matrices, enabled, owner_sources)


def owner_pose(x, *, rotation=None, count=1):
    value = np.repeat(np.eye(4)[None], count, axis=0)
    value[:, 0, 3] = x
    if rotation is not None:
        value[:, :3, :3] = rotation
    return torch.tensor(value, dtype=torch.float64)[None, None]


def test_enclosing_spheres_overlap_but_solids_and_gradients_are_clear():
    current = model()
    # Both source boxes have long y extents, so circumspheres intersect even
    # with a genuine x gap. The query intentionally has no proxy sphere input.
    separation = .04
    assert np.linalg.norm([.01, .08, .0125])+np.linalg.norm([.01, .04, .02]) > separation
    q = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)
    poses = owner_pose(separation).requires_grad_()
    values = query(current, q, poses)
    assert values.sum().item() == 0.
    values.sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))
    torch.testing.assert_close(poses.grad, torch.zeros_like(poses))


def test_true_overlap_has_same_query_joint_and_owner_witness_gradient():
    current = model()
    q = torch.zeros((1, 1, 2), dtype=torch.float64, requires_grad=True)
    poses = owner_pose(.015).requires_grad_()
    values = query(current, q, poses)
    assert values[0, 0, current.groups.index('collision/robot/obstacle')].item() == pytest.approx(.005, abs=2e-8)
    values.sum().backward()
    assert q.grad[0, 0, 1].item() == pytest.approx(1., abs=1e-8)
    assert poses.grad[0, 0, 0, 0, 3].item() == pytest.approx(-1., abs=1e-8)


def test_fixed_robot_without_owners_or_pairs_has_a_zero_self_group():
    current = model(fixed=True, owners={})
    q = torch.empty((2, 3, 0), dtype=torch.float64, requires_grad=True)
    matrices = torch.empty((2, 3, 0, 4, 4), dtype=torch.float64, requires_grad=True)
    values = query(current, q, matrices)
    assert values.shape == (2, 3, 1)
    assert torch.count_nonzero(values) == 0
    values.sum().backward()
    assert q.grad.shape == q.shape and matrices.grad.shape == matrices.shape


def test_registered_self_pair_preserves_its_positive_numerical_margin():
    displaced = np.eye(4)
    displaced[0, 3] = .02005
    current = model(fixed=True, owners={}, shapes=[body('first', 'base'), body('second', 'base', pose=displaced)],
        self_pairs=[dict(first='first', second='second', margin=.0001)])
    q = torch.empty((1, 1, 0), dtype=torch.float64)
    matrices = torch.empty((1, 1, 0, 4, 4), dtype=torch.float64)
    # Positive AABB separation below the declared self margin must still be
    # queried. The same geometry as a world pair has zero margin and is clear.
    assert query(current, q, matrices).item() == pytest.approx(.00005, abs=2e-9)
    world = model(fixed=True)
    assert query(world, q, owner_pose(.02005)).sum().item() == 0.


def test_owner_pairs_are_generic_and_enabled_is_scoped_to_one_pair_and_state():
    remote = np.eye(4)
    remote[1, 3] = 2.
    current = model(fixed=True, shapes=[body(frame='base', pose=remote)],
        owners={'payload': [primitive()], 'wall': [primitive()]}, moving_owners=('payload',))
    q = torch.empty((1, 2, 0), dtype=torch.float64)
    matrices = owner_pose(0., count=2).expand(1, 2, 2, 4, 4).clone()
    matrices[:, :, 1, 0, 3] = .015
    matrices.requires_grad_()
    enabled = torch.ones((1, 2, len(current.pairs)), dtype=torch.bool)
    component = [i for i, pair in enumerate(current.pairs) if pair['group'] == 'collision/payload/wall']
    enabled[0, 1, component] = False
    values = query(current, q, matrices, enabled)
    group = current.groups.index('collision/payload/wall')
    assert values[0, 0, group].item() == pytest.approx(.005, abs=2e-8)
    assert values[0, 1, group].item() == 0.
    values.sum().backward()
    assert matrices.grad[0, 0, 0, 0, 3].item() == pytest.approx(1., abs=1e-8)
    assert matrices.grad[0, 0, 1, 0, 3].item() == pytest.approx(-1., abs=1e-8)
    assert torch.count_nonzero(matrices.grad[0, 1]) == 0


@pytest.mark.parametrize('kind,dimensions', [
    ('cuboid', [.02, .08, .04]), ('box', [.01, .04, .02]),
    ('sphere', [.01]), ('cylinder', [.01, .02]), ('capsule', [.01, .02]),
])
def test_all_registered_primitive_shapes_use_their_analytic_solid(kind, dimensions):
    current = model(fixed=True, owners={'shape': [primitive(dimensions, kind)]})
    q = torch.empty((1, 1, 0), dtype=torch.float64)
    matrices = owner_pose(.015).requires_grad_()
    cost = query(current, q, matrices).sum()
    assert cost.item() == pytest.approx(.005, abs=2e-6)
    cost.backward()
    assert matrices.grad[0, 0, 0, 0, 3].item() == pytest.approx(-1., abs=2e-5)


def test_convex_robot_model_preserves_registered_vertices():
    vertices = np.asarray(list(product((-.01, .01), (-.08, .08), (-.0125, .0125))))
    current = model(fixed=True, shapes=[dict(name='source_convex', frame='base', kind='convex',
        vertices_frame=vertices.tolist(), source='registered native convex source')])
    q = torch.empty((1, 1, 0), dtype=torch.float64)
    assert query(current, q, owner_pose(.04)).sum().item() == 0
    assert query(current, q, owner_pose(.015)).sum().item() == pytest.approx(.005, abs=2e-8)


def test_articulated_rotated_robot_and_owner_pose_gradients_match_finite_difference():
    # Nontrivial base and geom-local transforms catch an incorrectly shifted
    # Pinocchio frame Jacobian and a missing reference-frame rotation.
    base = np.eye(4)
    base[:3, :3] = Rotation.from_rotvec([.15, -.1, .22]).as_matrix()
    base[:3, 3] = [.11, -.07, .03]
    local = np.eye(4)
    local[:3, 3] = [.023, -.012, .006]
    local[:3, :3] = Rotation.from_rotvec([.05, -.06, .12]).as_matrix()
    current = model(shapes=[body(pose=local)], names=('slide', 'turn'), base=base)
    q = torch.tensor([[[.06, .2]]], dtype=torch.float64, requires_grad=True)
    robot_pose = np.eye(4)
    robot_pose[:3, :3] = Rotation.from_rotvec([0., 0., .2]).as_matrix()
    robot_pose[:3, 3] = robot_pose[:3, :3] @ [.06, 0., 0.]
    target = base @ robot_pose @ local
    target[:3, 3] += target[:3, :3] @ [.015, .006, .003]
    # Parallel faces have a set-valued rotational derivative. Rotate the
    # target independently so central differences probe one smooth feature.
    target[:3, :3] = target[:3, :3] @ Rotation.from_rotvec([.13, -.17, .31]).as_matrix()
    matrices = torch.tensor(target, dtype=torch.float64)[None, None, None].requires_grad_()
    cost = query(current, q, matrices).sum()
    assert cost.item() > 0
    cost.backward()
    h = 1e-5
    for coordinate in range(2):
        positive, negative = q.detach().clone(), q.detach().clone()
        positive[..., coordinate] += h
        negative[..., coordinate] -= h
        derivative = (query(current, positive, matrices.detach()).sum()
                      - query(current, negative, matrices.detach()).sum())/(2*h)
        assert q.grad[0, 0, coordinate].item() == pytest.approx(derivative.item(), abs=1e-7)
    for coordinate in range(3):
        positive, negative = matrices.detach().clone(), matrices.detach().clone()
        axis = np.zeros(3)
        axis[coordinate] = h
        positive[0, 0, 0, :3, :3] = torch.tensor(Rotation.from_rotvec(axis).as_matrix()) @ matrices.detach()[0, 0, 0, :3, :3]
        negative[0, 0, 0, :3, :3] = torch.tensor(Rotation.from_rotvec(-axis).as_matrix()) @ matrices.detach()[0, 0, 0, :3, :3]
        derivative = (query(current, q.detach(), positive).sum()-query(current, q.detach(), negative).sum())/(2*h)
        tangent = (positive-negative)/(2*h)
        assert torch.sum(matrices.grad*tangent).item() == pytest.approx(derivative.item(), abs=1e-7)


def attached_owner(q, offset):
    """A rigid attachment from the same articulated transform as robot FK."""
    angle, slide = q[..., 0], q[..., 1]
    c, s, z, one = angle.cos(), angle.sin(), torch.zeros_like(angle), torch.ones_like(angle)
    row0 = torch.stack((c, -s, z, c*(slide+offset)), -1)
    row1 = torch.stack((s, c, z, s*(slide+offset)), -1)
    row2 = torch.stack((z, z, one, z), -1)
    row3 = torch.stack((z, z, z, one), -1)
    return torch.stack((row0, row1, row2, row3), -2).unsqueeze(-3)


def test_same_rigid_attachment_cancels_both_pinocchio_and_torch_derivative_paths():
    current = model()
    q = torch.tensor([[[.23, .07], [-.32, -.02]], [[.7, .1], [-.4, .03]]],
                     dtype=torch.float64, requires_grad=True)
    values = query(current, q, attached_owner(q, .015))
    values.sum().backward()
    collision = values[..., current.groups.index('collision/robot/obstacle')]
    torch.testing.assert_close(collision, torch.full_like(collision, .005), rtol=0, atol=2e-8)
    # An unpermitted overlap is retained, but moving the entire assembly cannot
    # resolve its internal relation and therefore has zero total derivative.
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=2e-8)


def test_alias_deduplication_keeps_extra_components_and_payload_obstacles():
    installation = body('mount', 'base', dimensions=(.04, .04, .04))
    dynamic = body('arm', 'slider')
    alias = primitive((.04, .04, .04))
    alias['native_binding'] = {'geometry_name': 'mount'}
    extra = primitive()
    current = NativeCollisionModel(robot_model=dict(urdf_xml=URDF, geometries=[installation, dynamic]),
        joint_names=('turn', 'slide'), T_reference_base=np.eye(4),
        self_pairs=[dict(first='mount', second='arm', margin=1e-6)],
        owners={'fixture': [alias, extra], 'payload': [primitive()]}, moving_owners=('payload',),
        owner_aliases={'fixture': ('mount',)})
    assert any(p['group'] == 'collision/robot/self' and p['first_geometry'] == 'mount' for p in current.pairs)
    assert not any(p['first_owner'] == 'robot' and p['first_geometry'] == 'mount'
                   and p['second_owner'] != 'robot' for p in current.pairs)
    assert not any(p['first_owner'] == 'robot' and p['second_owner'] == 'fixture'
                   and p['second_geometry'] == 'mount' for p in current.pairs)
    assert any(p['group'] == 'collision/robot/fixture' and p['second_geometry'] == 'primitive:1'
               for p in current.pairs)
    assert any(p['group'] == 'collision/payload/fixture' and p['second_geometry'] == 'mount'
               for p in current.pairs)
# END no skill name, Holding relation or contact permission is inferred here.
