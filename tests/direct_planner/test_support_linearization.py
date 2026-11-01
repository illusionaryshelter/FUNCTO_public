"""Small native-API checks for support factors and shared frame Jacobians.

The maintained FUNCTO CPU fixtures supply robot descriptions, a calibrated
tool, cubic entrance and original condition producers. Tests use declared
point/halfspace data and evaluate existing curves; no optimizer or scene run
is involved. Mixed components and named joints exercise the actual assembly
boundary where analytic and AD rows meet.
"""

from dataclasses import replace
import gc
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import weakref

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation
import torch

EXPERIMENT = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                 str(Path(__file__).resolve().parents[2] / 'tests'),
                 str(Path(__file__).resolve().parents[2] / 'tests/planning')]

from manipulation.direct_planner.differentiable_kinematics import DifferentiablePinKinematics
from manipulation.direct_planner.material_hull import material_hull_indices
from manipulation.direct_planner.sparse_physical import SparsePhysicalRows
from manipulation.direct_planner.sparse_task import SparseTaskRows
from manipulation.direct_planner.terminal_binding import bind_terminal_hold
from manipulation.direct_planner.support_linearization import SupportLinearization, linearized_robot_state, task_linearization
from manipulation.goals import SupportSurface
from manipulation.gripper_trajectory import NominalFingerGeometry
from manipulation.physical_joint_domain import CubicJointDomain
from manipulation.pin_kinematics import PinKinematics
from manipulation.planning_problem import SupportRequirement
from manipulation.planning_scene import PrismaticGripper
from manipulation.preference_planner.joint_computation import JointComputation
from manipulation.preference_planner.terminal import TerminalHold
from manipulation.unified_solver import TrajectoryComputation, robot_tool_states, task_jacobian


def load_fixture(name, filename):
    """Import maintained numerical fixture definitions without rewriting them."""
    spec = importlib.util.spec_from_file_location(name, PROJECT / 'tests' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


curve_fixture = load_fixture('support_original_curve_fixture', 'test_unified_solver.py')
pin_fixture = load_fixture('support_original_pin_fixture', 'test_pin_kinematics.py')


class CountingKinematics(DifferentiablePinKinematics):
    """Count actual native batches while retaining the complete implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.native_calls = 0

    def _evaluate(self, *args, **kwargs):
        self.native_calls += 1
        return super()._evaluate(*args, **kwargs)


def support_case(*, analytic=True):
    """Two same-event declarations, mixed rows and a reordered controlled axis."""
    original, full = curve_fixture.computation()
    rotation = Rotation.from_rotvec([.21, -.13, .17]).as_matrix()
    surface = SupportSurface([.1, -.2, .03], rotation[:, :2], rotation[:, 2],
        [[-.3, -.2], [.4, -.2], [.4, .3], [-.3, .3]], 'declared point-support unit fixture')
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec([-.2, .11, .3]).as_matrix()
    transform[:3, 3] = [.04, -.01, .03]
    first = SupportRequirement(.5, transform,
        [[-.02, -.01, -.03], [.03, -.01, -.03], [.03, .02, -.03], [-.02, .02, .01]],
        [[-.02, -.01, -.03], [.03, -.01, -.03], [.03, .02, -.03]],
        [0., 0., -1.], surface, .01, .2, .004, 'first declared support',
        hard_components=('footprint', 'gap', 'body', 'normal') if analytic else ('gap', 'normal'))
    second_transform = transform.copy()
    second_transform[:3, 3] += [.02, .04, -.01]
    second = replace(first, tool_from_carrier=second_transform,
        body_points=first.body_points[:3], source='second declared support',
        hard_components=('normal', 'body') if analytic else ('normal',))
    permutation = [2, 0, 1]
    names = tuple(original.joint_names[i] for i in permutation)
    problem = replace(original.problem, supports=(first, second), joint_names=names,
        start_q=original.problem.start_q[permutation], start_dq=original.problem.start_dq[permutation])
    curve = original.domain._curve_map
    domain = CubicJointDomain(original.domain.bounds[..., permutation],
        torch.tensor(problem.start_q), torch.tensor(problem.start_dq), curve.knot_dt, curve.knots)
    kinematics = CountingKinematics(original.kinematics.model,
        frames=original.kinematics.frames, T_reference_base=problem.base_pose)
    base = TrajectoryComputation(problem=problem, scene=original.scene, kinematics=kinematics,
        domain=domain, joint_names=names, parameters=original.parameters,
        progress=original.progress.numpy()[0], ordinary_indices=original.ordinary_indices,
        motion_configuration=original.motion_configuration, native_dt=original.native_dt,
        constraint_weight=original.constraint_weight, max_host_rss_bytes=original.max_host_rss_bytes)
    terminal = bind_terminal_hold(curve_map=domain._curve_map,
        clock_u=[0., .8, 1.], clock_z=[0., 1., 1.], source='declared terminal unit-fixture clock')
    geometry = SimpleNamespace(rows=lambda value: value.reshape(-1)[:0])
    bound = JointComputation(base=base, terminal=terminal, geometry=geometry)
    bound.support_material_indices = tuple(material_hull_indices(item.body_points) for item in problem.supports)
    reduced = terminal.reduce(full[:, permutation] + np.random.default_rng(7).normal(0., .02, full.shape))
    return bound, reduced


def assemble(values, blocks, controls):
    """Mirror the explicit protocol: only remaining rows enter ordinary AD."""
    sizes = [value.numel() for value in values]
    remaining = np.ones(sum(sizes), dtype=bool)
    result = np.empty((sum(sizes), controls.numel()))
    offset = 0
    for size, group in zip(sizes, blocks):
        for row_slice, factor in group.values():
            target = slice(offset + row_slice.start, offset + row_slice.stop)
            result[target] = np.stack([factor.row_values(row, np.arange(controls.numel()))
                                       for row in range(len(factor.joint_jacobian))])
            remaining[target] = False
        offset += size
    vector = torch.cat(values)
    result[remaining] = task_jacobian(vector[remaining], controls,
        check_request=lambda: None, admit_work=lambda: None)
    return result, remaining


# BEGIN independent event-unit conversion for the original task-value oracle.
def original_values_in_solver_units(bound, controls, *, masks):
    """Transform native signed errors using their declared geometric tolerances.

    The original task producer returns distance/angle minus tolerance. The
    direct planner expresses the same event set by squared normalized distance
    or half-angle sine. This independent transform retains original support
    values and their reverse graph, including finite-difference queries.
    """
    values = bound.task(controls, solver_rows_out=masks)
    result = []
    for kind, (vector, group) in enumerate(zip(values, masks)):
        offset, parts = 0, []
        for name, mask in group.items():
            count = mask.numel()
            value = vector[offset:offset+count]
            identity = name.split(':')
            if kind == 1 and identity[0] == 'function_pose':
                tolerance = bound.problem.pose_events[int(identity[1])].component_tolerances[identity[2]]
                error = value+tolerance
                value = ((error/tolerance).square()-1. if identity[2] == 'position'
                    else torch.sin(error/2).square()/np.sin(tolerance/2)**2-1.)
            parts.append(value)
            offset += count
        result.append(torch.cat(parts) if parts else vector)
    return tuple(result)
# END the production event-normalization function is not used by this oracle.


@pytest.mark.parametrize('analytic', [True, False])
def test_original_named_rows_mixed_components_and_reduced_columns(analytic, monkeypatch):
    """Mixed body/footprint and gap/normal retain all original values and derivatives."""
    bound, initial = support_case(analytic=analytic)
    original_controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    expected_masks = ({}, {})
    # BEGIN compare the mixed rows in the planner's current declared units.
    original = original_values_in_solver_units(bound, original_controls, masks=expected_masks)
    # END native support values and masks retain their original identities.
    expected = task_jacobian(torch.cat(original), original_controls,
        check_request=lambda: None, admit_work=lambda: None)
    controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    masks = ({}, {})
    before = bound.kinematics.native_calls
    with monkeypatch.context() as patch:
        def derivative_not_requested(*args):
            raise AssertionError('The value producer assembled a dense support Jacobian')
        patch.setattr(SupportLinearization, 'row_values', derivative_not_requested)
        values, blocks = task_linearization(bound, controls, solver_rows_out=masks,
            support_material_indices=bound.support_material_indices)
    assert bound.kinematics.native_calls == before + 1
    for actual, wanted in zip(values, original):
        # BEGIN independent trigonometric formulations have round-off error.
        torch.testing.assert_close(actual.detach(), wanted.detach(), rtol=2e-12, atol=2e-12)
        # END this tolerance applies only to the independent test oracle.
    for actual, wanted in zip(masks, expected_masks):
        # Native JointComputation supplies a separate empty geometry fixture.
        wanted = {key: value for key, value in wanted.items() if key != 'registered_geometry_witnesses'}
        assert list(actual) == list(wanted)
        for key in wanted:
            torch.testing.assert_close(actual[key], wanted[key], rtol=0., atol=0.)
    expected_names = ['support:0:footprint', 'support:0:body', 'support:1:body'] if analytic else []
    assert list(blocks[1]) == expected_names and blocks[0] == {}
    actual, remaining = assemble(values, blocks, controls)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)
    direction = np.random.default_rng(11).normal(size=initial.shape)
    direction /= np.linalg.norm(direction)
    epsilon = 1e-6
    with torch.no_grad():
        # BEGIN directional differences use the same independent event units.
        plus = torch.cat(original_values_in_solver_units(bound,
            torch.tensor(initial + epsilon*direction), masks=({}, {}))).numpy()
        minus = torch.cat(original_values_in_solver_units(bound,
            torch.tensor(initial - epsilon*direction), masks=({}, {}))).numpy()
        # END the raw task producer still supplies all values for this oracle.
    np.testing.assert_allclose(actual @ direction.ravel(), (plus-minus)/(2*epsilon), rtol=2e-6, atol=3e-8)
    if analytic:
        assert (~remaining).sum() == 4*4 + 4 + 3
        assert np.max(np.abs(actual[~remaining])) > .01


def test_analytic_values_have_no_reverse_graph_and_discarded_graphs_release():
    """Detached support blocks do not survive as zero-covector reverse work."""
    bound, initial = support_case()

    def forward(consume):
        controls = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
        reference = weakref.ref(controls)
        values, blocks = task_linearization(bound, controls,
            support_material_indices=bound.support_material_indices)
        if consume:
            autodiff = task_jacobian(torch.cat(values), controls,
                check_request=lambda: None, admit_work=lambda: None)
            offset = values[0].numel()
            for row_slice, _ in blocks[1].values():
                np.testing.assert_array_equal(autodiff[offset+row_slice.start:offset+row_slice.stop], 0.)
        # Keeping the small factors cannot keep controls or native autograd nodes.
        return reference, blocks

    for consume in (False, True):
        reference, factors = forward(consume)
        gc.collect()
        assert reference() is None
        assert all(factor.row_values(0, np.arange(initial.size)).shape == (initial.size,)
                   for _, factor in factors[1].values())


def finger_case():
    """Use the existing independent-finger URDF and maintained nominal model API."""
    model = pin.buildModelFromXML(pin_fixture.URDF)
    base = np.eye(4)
    base[:3, :3] = Rotation.from_rotvec([.4, -.2, .3]).as_matrix()
    base[:3, 3] = [1.2, -.5, .3]
    tool = np.eye(4)
    tool[:3, :3] = Rotation.from_rotvec([.1, .2, -.1]).as_matrix()
    tool[:3, 3] = [.07, -.04, .08]
    kinematics = CountingKinematics(model, frames=('tool', 'left', 'right'), T_reference_base=base)
    nominal = NominalFingerGeometry(('right_slide', 'left_slide'), np.array([0., 1.]),
        np.array([[-.02, .03], [-.04, .06]]), np.array([[-.01, .02], [0., 0.]]),
        1., .4, 'declared nominal unit-fixture prefix')
    hand = PrismaticGripper(controller='binary', joint_names=nominal.joint_names,
        current_positions=nominal.position[0], joint_limits=[[-.1, 0.], [0., .1]],
        joint_bounds=None, coupling_matrix=np.empty((0, 2)), coupling_offset=[],
        commands=(), source='existing independent-finger model interface', control_timestep_s=.002)
    hand.nominal_finger_trajectory = nominal
    times = torch.tensor([[0., .25, .75]], dtype=torch.float64)
    bound = SimpleNamespace(problem=SimpleNamespace(tool_link='left', tool_transform=tool),
        scene=SimpleNamespace(gripper=hand), kinematics=kinematics,
        joint_names=('elbow', 'shoulder'), progress=times, times=times)
    q = torch.tensor([[[.3, -.4], [-.2, .5], [.1, .2]]], dtype=torch.float64, requires_grad=True)
    dq = torch.tensor([[[-.5, .3], [.4, -.6], [.2, -.1]]], dtype=torch.float64, requires_grad=True)
    return bound, q, dq


def test_named_fingers_nominal_prefix_and_nonidentity_tcp_share_one_native_call():
    """The original hand completion and same-call J retain caller column order."""
    bound, q, dq = finger_case()
    expected = robot_tool_states(problem=bound.problem, scene=bound.scene, kinematics=bound.kinematics,
        q=q, dq=dq, joint_names=bound.joint_names, progress=bound.progress, times=bound.times)
    before = bound.kinematics.native_calls
    actual, jacobians = linearized_robot_state(bound, q, dq)
    assert bound.kinematics.native_calls == before + 1
    assert actual[2] == expected[2] == ('elbow', 'shoulder', 'right_slide', 'left_slide')
    for index in (0, 1):
        torch.testing.assert_close(actual[index], expected[index], rtol=0., atol=0.)
    for frame in actual[3]:
        for got, wanted in zip(actual[3][frame], expected[3][frame]):
            torch.testing.assert_close(got, wanted, rtol=0., atol=0.)
    for index in (4, 5):
        for got, wanted in zip(actual[index], expected[index]):
            torch.testing.assert_close(got, wanted, rtol=0., atol=0.)
    complete_q, complete_dq, names = actual[:3]
    poses, twists, ordered = bound.kinematics.states_with_jacobians(complete_q, complete_dq, names)
    for frame, jacobian in ordered.items():
        assert not jacobian.requires_grad
        expected_twist = torch.cat(twists[frame], -1)
        torch.testing.assert_close((jacobian @ complete_dq[..., None]).squeeze(-1), expected_twist,
                                   rtol=2e-13, atol=2e-13)
    assert torch.all(jacobians['left'][..., :, names.index('right_slide')] == 0.)
    assert float(jacobians['left'][..., :3, names.index('left_slide')].abs().max()) > 0.


def test_state_pose_and_twist_gradients_and_explicit_jacobian_lifetime():
    """The native output extension retains original q/dq VJPs and graph release."""
    bound, _, _ = finger_case()
    names, position, velocity = pin_fixture.sample()
    position, velocity = position.requires_grad_(), velocity.requires_grad_()
    original = PinKinematics(bound.kinematics.model, frames=bound.kinematics.frames,
        T_reference_base=np.block([[bound.kinematics._rotation, bound.kinematics._translation[:, None]],
                                  [np.zeros((1, 3)), np.ones((1, 1))]]))
    expected = pin_fixture.packed(*original.states(position, velocity, names))
    poses, twists, jacobians = bound.kinematics.states_with_jacobians(position, velocity, names)
    actual = pin_fixture.packed(poses, twists)
    torch.testing.assert_close(actual, expected, rtol=0., atol=0.)
    expected_grad = torch.autograd.grad((expected.square()+torch.sin(expected)).sum(), (position, velocity))
    actual_grad = torch.autograd.grad((actual.square()+torch.sin(actual)).sum(), (position, velocity))
    for got, wanted in zip(actual_grad, expected_grad):
        torch.testing.assert_close(got, wanted, rtol=2e-13, atol=2e-13)

    def native_query():
        q = position.detach().clone().requires_grad_()
        reference = weakref.ref(q)
        _, _, geometric = bound.kinematics.states_with_jacobians(q, velocity.detach(), names)
        return reference, geometric

    reference, geometric = native_query()
    gc.collect()
    assert reference() is None
    assert all(not value.requires_grad for value in geometric.values())
