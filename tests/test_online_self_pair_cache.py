"""Semantic and lifecycle checks for binding-owned self-collision indices."""
# BEGIN fixed indices cannot alter collision values, derivatives or bindings.
import numpy as np
import torch

from manipulation.online_geometry import FrozenESDF, OnlinePlanningScene
from manipulation.planning_scene import ObservedSpace, PrismaticGripper, _self_pair_sum


def make_scene(*, moving=None, bound=True):
    hand = None if moving is None else PrismaticGripper(
        controller='binary', joint_names=('finger',), current_positions=[-.01],
        joint_limits=[[-.2, .2]], joint_bounds=None, coupling_matrix=np.empty((0, 1)),
        coupling_offset=[], sphere_indices=(0,), local_spheres=[[0., 0., 0., .02]],
        sphere_joint_translation=[[[float(moving)], [0.], [0.]]],
        self_collision_pairs=((0, 1),), commands=(), source='calibrated linear test finger',
        control_timestep_s=.01)
    valid = np.ones((21, 21, 21), bool)
    field = FrozenESDF(np.ones(valid.shape, np.float32), origin=[-.5]*3, voxel_size=.05,
        site_valid=valid, observed_sign=valid, owner_ids=('background',), source='free test field')
    coverage = ObservedSpace(valid, origin=[-.5]*3, voxel_size=.05,
        observation_ids=('current',), reference_frame='world')
    result = OnlinePlanningScene(background=field, owners=(), coverage=coverage,
        reference_frame='world', robot_sphere_materials=(None,)*3, robot_geometry_source='three test links',
        gripper=hand, unknown_policy={'outside_regions': 'forbid', 'conditional_regions': []})
    if bound:
        rebind(result, [[1, 2], [0, 1]])
    return result


def rebind(scene, pairs):
    scene.bind_online_robot(robot_sphere_geometry=[
        {'link_name': f'link{i}', 'sphere_indices': [i], 'centers_local': [[0., 0., 0.]], 'radii': [.02]}
        for i in range(3)], robot_self_collision_pairs=pairs)


def evaluate(scene, dtype=torch.float64):
    balls = torch.tensor([[[[0., 0., 0., .02], [.01, 0., 0., .02], [.035, 0., 0., .02]],
                           [[0., 0., 0., .02], [.01, 0., 0., .02], [.035, 0., 0., .02]]]],
                         dtype=dtype, requires_grad=True)
    value = scene.components(balls, balls.new_zeros((1, 2, 3)),
        balls.new_tensor([[[1., 0., 0., 0.], [1., 0., 0., 0.]]]), balls.new_tensor([[0., .5]]),
        trajectory_time=balls.new_tensor([[0., .5]]))
    return balls, value


def test_prepared_indices_keep_known_and_uncertain_penetrations_and_gradients():
    scene = make_scene(moving=True)
    balls, terms = evaluate(scene)
    hard, conditional = terms['collision/robot/self'], terms['conditional/collision/robot/self']
    # At entrance, the finger at x=-.01 penetrates link1 by .02 m and
    # links1/2 penetrate by .015 m. Later only that .015 m remains hard;
    # the uncertain finger envelope contributes .23 m conditionally.
    torch.testing.assert_close(hard, hard.new_tensor([[.035, .015]]), rtol=0, atol=1e-15)
    torch.testing.assert_close(conditional, conditional.new_tensor([[0., .23]]), rtol=0, atol=1e-15)
    hard_gradient = torch.autograd.grad(hard.sum(), balls, retain_graph=True)[0]
    conditional_gradient = torch.autograd.grad(conditional.sum(), balls)[0]
    torch.testing.assert_close(hard_gradient[..., 0], balls.new_tensor([[[0., 0., -1.], [0., 1., -1.]]]),
                               rtol=0, atol=0)
    torch.testing.assert_close(conditional_gradient[..., 0], balls.new_tensor([[[0., 0., 0.], [0., -1., 0.]]]),
                               rtol=0, atol=0)

    # A hand state can be uncertain with no sphere translated by its joints.
    # Both partitions must then retain every pair, not discard all self costs.
    fixed = make_scene(moving=False)
    _, terms = evaluate(fixed)
    torch.testing.assert_close(terms['collision/robot/self'], hard.new_tensor([[.045, .045]]), rtol=0, atol=1e-15)
    assert torch.count_nonzero(terms['conditional/collision/robot/self']) == 0

    # Even if a caller removes the hand before rebinding, the dynamic false
    # uncertainty flag must select all pairs, despite a prepared known subset.
    detached = scene.clone_for_rollout()
    detached.gripper = None
    values, terms = evaluate(detached)
    torch.testing.assert_close(terms['collision/robot/self'], hard.new_tensor([[.045, .045]]), rtol=0, atol=1e-15)
    assert torch.count_nonzero(terms['conditional/collision/robot/self']) == 0
    gradient = torch.autograd.grad(terms['collision/robot/self'].sum(), values)[0]
    torch.testing.assert_close(gradient[..., 0], values.new_tensor([[[1., 0., -1.], [1., 0., -1.]]]),
                               rtol=0, atol=0)

    # A new fixed hand binding recompiles membership and replaces, rather than
    # clears, the old clone's cache; uncertain state is no longer moving volume.
    rebound = scene.clone_for_rollout()
    rebound.gripper = fixed.gripper
    old_cache = scene._self_collision_pair_device
    rebind(rebound, [[1, 2], [0, 1]])
    assert rebound._uncertain_sphere_indices == ()
    assert scene._uncertain_sphere_indices == (0,)
    assert rebound._self_collision_pair_device is not old_cache
    _, terms = evaluate(rebound)
    torch.testing.assert_close(terms['collision/robot/self'], hard.new_tensor([[.045, .045]]), rtol=0, atol=1e-15)


def test_cached_integer_indices_preserve_chunked_float_values_and_gradients():
    # Exercise more than one unchanged 512-pair chunk with deliberately
    # non-sorted pair order. Compare the old ndarray input contract to cached
    # device indices through the existing distance routine, not a new formula.
    rng = np.random.default_rng(3)
    pairs = np.array([(i, j) for i in range(50) for j in range(i+1, 50)], dtype=np.int64)
    pairs = pairs[rng.permutation(len(pairs))[:1025]]
    scene = make_scene()
    scene.robot_sphere_materials = (None,)*50
    scene.bind_online_robot(robot_sphere_geometry=[
        {'link_name': f'link{i}', 'sphere_indices': [i], 'centers_local': [[0., 0., 0.]], 'radii': [.2]}
        for i in range(50)], robot_self_collision_pairs=pairs)
    values = torch.tensor(rng.normal(0., .1, (1, 2, 50, 4)), dtype=torch.float64)
    values[..., 3] = .2
    values.requires_grad_()
    cached, known = scene._self_collision_pair_tensors(values)
    expected = _self_pair_sum(values, pairs)
    actual = _self_pair_sum(values, cached)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.sum(), values, retain_graph=True)[0],
                       torch.autograd.grad(expected.sum(), values)[0])
    assert torch.equal(cached, known)
    assert cached.tolist() == pairs.tolist()


def test_clone_retains_old_indices_when_original_rebinds_and_dtypes_share_cache():
    original = make_scene()
    values, before = evaluate(original)
    indices = original._self_collision_pair_tensors(values)
    assert original._self_collision_pair_tensors(values.float()) is indices
    assert all(index.dtype == torch.long for index in indices)
    assert all(not array.flags.writeable for array in original._self_collision_pair_arrays)
    clone = original.clone_for_rollout()
    assert clone._self_collision_pair_tensors(values) is indices
    old_cache = clone._self_collision_pair_device
    old_arrays = clone._self_collision_pair_arrays
    rebind(original, [[0, 2]])
    assert original._self_collision_pair_device == {}
    assert clone._self_collision_pair_device is old_cache
    assert clone._self_collision_pair_arrays is old_arrays
    assert original._self_collision_pair_device is not old_cache
    _, unchanged = evaluate(clone)
    _, changed = evaluate(original)
    assert torch.equal(unchanged['collision/robot/self'], before['collision/robot/self'])
    torch.testing.assert_close(changed['collision/robot/self'], values.new_tensor([[.005, .005]]), rtol=0, atol=1e-15)
    assert clone._self_collision_pair_tensors(values) is indices

    # Memory reserves one read-only host array and one CPU long tensor for
    # each partition, plus one GPU long tensor, independent of float dtype.
    empty = clone.clone_for_rollout()
    rebind(empty, [])
    occupied, baseline = clone.estimated_workspace_allocation(1, 2), empty.estimated_workspace_allocation(1, 2)
    pair_bytes = sum(array.nbytes for array in old_arrays)
    assert occupied['cpu_shared']-baseline['cpu_shared'] == 2*pair_bytes
    assert occupied['gpu_shared']-baseline['gpu_shared'] == pair_bytes


def test_unbound_native_admission_reserves_pair_tables_before_fk_binding():
    scene = make_scene(moving=True, bound=False)
    unbound = scene.estimated_workspace_allocation(1, 1)
    # Three total spheres have three native unordered pairs; two fixed spheres
    # have one known pair. Each row owns two int64 indices, independent of FK.
    rebind(scene, [])
    empty = scene.estimated_workspace_allocation(1, 1)
    assert unbound['cpu_shared']-empty['cpu_shared'] == 2*(3+1)*2*8
    assert unbound['gpu_shared']-empty['gpu_shared'] == (3+1)*2*8
# END fixed preparation cannot change runtime uncertainty or old rollout state.
