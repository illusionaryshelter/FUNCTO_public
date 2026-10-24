"""Complete-cover bounds on the production Pinocchio model interface."""

# BEGIN analytic checks use native model/frame/joint types from production.
from pathlib import Path
import sys

import numpy as np
import pinocchio as pin
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.geometry_motion_bound import (
    SphereBody, build_geometry_motion_bound, relative_speed_bound,
)


def model(parents=(), joints=(), *, names=None, limits=None, translations=None,
          rotations=None):
    """Construct only test kinematics; no task or robot collision asset is loaded."""
    native = pin.Model()
    native.addFrame(pin.Frame("link0", 0, 0, pin.SE3.Identity(), pin.BODY))
    count = len(joints)
    names = tuple(f"q{i}" for i in range(count)) if names is None else names
    translations = np.zeros((count, 3)) if translations is None else translations
    rotations = np.repeat(np.eye(3)[None], count, axis=0) if rotations is None else rotations
    for i, (parent, joint, name, translation, rotation) in enumerate(
            zip(parents, joints, names, translations, rotations), start=1):
        index = native.addJoint(parent, joint, pin.SE3(rotation, np.asarray(translation, float)), name)
        assert index == i
        native.addFrame(pin.Frame(f"link{i}", index, 0, pin.SE3.Identity(), pin.BODY))
    limits = np.array([[-1.]*count, [1.]*count]) if limits is None else np.asarray(limits, float)
    native.lowerPositionLimit = limits[0]
    native.upperPositionLimit = limits[1]
    return native


def body(link="link1", *, spheres=((1., 0., 0., .2),), name="body"):
    return SphereBody(name, link, np.asarray(spheres))


def build(native, bodies=None, calibration=None):
    return build_geometry_motion_bound(model=native,
        bodies=[body()] if bodies is None else bodies,
        T_reference_base=np.eye(4) if calibration is None else calibration)


def test_revolute_bound_covers_sphere_surface_not_just_its_centre():
    result = build(model([0], [pin.JointModelRZ()], names=("q",)))
    # The outer sphere point x=1.2 has speed 1.2*|dq| at q=0.
    assert 1.2 <= result.coefficients[0, 0] < 1.2+1e-12
    assert 3.6 <= result.speed_bounds({"q": 3.})[0] < 3.6+1e-11
    assert not result.coefficients.flags.writeable


@pytest.mark.parametrize("joint, expected", [
    (pin.JointModelRX(), 1.2), (pin.JointModelRY(), 1.2), (pin.JointModelRZ(), 1.2),
    (pin.JointModelRevoluteUnaligned(np.array([1., 2., -3.])/np.sqrt(14.)), 1.2),
    (pin.JointModelPX(), 1.), (pin.JointModelPY(), 1.), (pin.JointModelPZ(), 1.),
    (pin.JointModelPrismaticUnaligned(np.array([-1., 2., 3.])/np.sqrt(14.)), 1.),
])
def test_native_aligned_and_unaligned_axis_subspaces(joint, expected):
    native = model([0], [joint], names=("q",))
    result = build(native)
    assert expected <= result.coefficients[0, 0] < expected+1e-12
    data = native.createData()
    for q in (-1., 0., 1.):
        pin.forwardKinematics(native, data, np.array([q]), np.array([1.]))
        pin.updateFramePlacements(native, data)
        frame = native.getFrameId("link1")
        velocity = pin.getFrameVelocity(native, data, frame, pin.LOCAL_WORLD_ALIGNED)
        for direction in np.eye(3):
            local_point = np.array([1., 0., 0.])+.2*direction
            arm = data.oMf[frame].rotation @ local_point
            tangent = velocity.linear + np.cross(velocity.angular, arm)
            assert np.linalg.norm(tangent) <= result.coefficients[0, 0]


def test_fixed_frame_offset_precedes_ancestor_lever_and_root_translation_does_not():
    native = model([0], [pin.JointModelRZ()], translations=[[10., 0., 0.]])
    native.addFrame(pin.Frame("fixed", 1, 0,
        pin.SE3(np.eye(3), np.array([2., 0., 0.])), pin.BODY))
    result = build(native, [body("fixed", spheres=((.5, 0., 0., .1),))])
    assert 2.6 <= result.coefficients[0, 0] < 2.6+1e-12


def test_prismatic_full_domain_and_negative_axis_enlarge_ancestor_radius():
    native = model([0, 1], [pin.JointModelRZ(),
        pin.JointModelPrismaticUnaligned(np.array([-1., 0., 0.]))],
        names=("turn", "slide"), limits=[[-1., -.2], [1., .3]],
        translations=[[0., 0., 0.], [.2, 0., 0.]])
    result = build(native, [body("link2", spheres=((.1, 0., 0., .05),))])
    assert .65 <= result.coefficients[0, 0] < .65+1e-12
    assert 1. <= result.coefficients[0, 1] < 1.+1e-12
    assert result.speed_bounds({"turn": 0., "slide": 0.})[0] == 0.


def test_fixed_root_entity_is_static_and_calibration_rotation_scales_velocity():
    result = build(model(), [body("link0")])
    assert result.coefficients.shape == (1, 0)
    assert result.speed_bounds({})[0] == 0.
    calibration = np.eye(4)
    calibration[0, 0] = 1.0000001
    calibration[:3, 3] = [100., 200., 300.]
    moving = build(model([0], [pin.JointModelPX()]), calibration=calibration)
    assert 1.0000001 <= moving.coefficients[0, 0] < 1.0000001+1e-12


def test_stored_frame_and_joint_rotation_norms_are_not_assumed_exactly_one():
    matrix = Rotation.from_euler("xyz", [.27, -.43, .81]).as_matrix().astype(np.float32)
    matrix *= np.float32(1.000001)
    native = model([0], [pin.JointModelRZ()], rotations=[matrix.astype(float)])
    native.addFrame(pin.Frame("fixed", 1, 0,
        pin.SE3(matrix.astype(float), np.zeros(3)), pin.BODY))
    result = build(native, [body("fixed")])
    for angle in np.linspace(-1., 1., 31):
        rz = Rotation.from_rotvec([0., 0., angle]).as_matrix()
        tangent = matrix @ np.cross([0., 0., 1.], rz @ matrix @ np.array([1.2, 0., 0.]))
        assert np.linalg.norm(tangent) <= result.coefficients[0, 0]
    assert result.coefficients[0, 0] < 1.20002


def test_complete_finger_coordinate_is_required_and_increases_arm_lever():
    native = model([0, 1], [pin.JointModelRZ(), pin.JointModelPX()],
        names=("arm", "finger"), limits=[[-1., 0.], [1., .08]])
    result = build(native, [body("link2", spheres=((0., 0., 0., .01),))])
    assert result.joint_names == ("arm", "finger")
    assert .09 <= result.coefficients[0, 0] < .09+1e-12
    assert 1. <= result.coefficients[0, 1] < 1.+1e-12
    assert .28 <= result.speed_bounds({"arm": 2., "finger": .1})[0] < .28+1e-12
    with pytest.raises(ValueError, match="every registered robot"):
        result.speed_bounds({"arm": 2.})


def test_branched_tree_preserves_zero_nonancestor_coefficients_and_complete_covers():
    native = model([0, 0], [pin.JointModelRZ(), pin.JointModelPX()], names=("turn", "slide"))
    first = body(spheres=((0., 0., 0., .1), (0., 3., 0., .4)), name="large")
    second = body("link2", spheres=((5., 0., 0., .2),), name="slide")
    result = build(native, [first, second])
    assert 3.4 <= result.coefficients[0, 0] < 3.4+1e-12
    assert result.coefficients[0, 1] == result.coefficients[1, 0] == 0.
    assert 1. <= result.coefficients[1, 1] < 1.+1e-12


def test_mixed_chain_bound_dominates_native_full_surface_point_velocities():
    rotations = Rotation.from_euler("xyz", [[.1, .2, -.3], [-.4, .1, .2], [.3, -.2, .1]]).as_matrix()
    native = model([0, 1, 2], [pin.JointModelRZ(),
        pin.JointModelPrismaticUnaligned(np.array([-1., 2., 1.])/np.sqrt(6.)),
        pin.JointModelRevoluteUnaligned(np.array([2., -1., 1.])/np.sqrt(6.))],
        names=("turn", "slide", "wrist"), limits=[[-1., -.2, -.9], [1., .4, .9]],
        translations=[[.3, -.1, .2], [.4, .2, .1], [.2, 0., 0.]], rotations=rotations)
    native.addFrame(pin.Frame("tool", 3, 0,
        pin.SE3(Rotation.from_euler("xyz", [.3, .2, .1]).as_matrix(), np.array([.07, .02, -.01])), pin.BODY))
    calibration = np.eye(4)
    calibration[:3, :3] = Rotation.from_euler("xyz", [.2, .3, -.1]).as_matrix()
    calibration[:3, 3] = [1., 2., 3.]
    result = build(native, [body("tool", spheres=((.1, .03, -.02, .08),))], calibration)
    rng = np.random.default_rng(193)
    data = native.createData()
    frame = native.getFrameId("tool")
    for _ in range(100):
        q = rng.uniform(native.lowerPositionLimit, native.upperPositionLimit)
        dq = rng.uniform(-2., 2., 3)
        direction = rng.normal(size=3)
        point = np.array([.1, .03, -.02])+.08*direction/np.linalg.norm(direction)
        pin.forwardKinematics(native, data, q, dq)
        pin.updateFramePlacements(native, data)
        velocity = pin.getFrameVelocity(native, data, frame, pin.LOCAL_WORLD_ALIGNED)
        tangent = calibration[:3, :3] @ (velocity.linear
            +np.cross(velocity.angular, data.oMf[frame].rotation @ point))
        speed = result.speed_bounds(dict(zip(result.joint_names, abs(dq))))[0]
        assert np.linalg.norm(tangent) <= speed


def test_relative_speed_requires_explicit_moving_target_bound():
    assert .7 <= relative_speed_bound(.3, target_speed_m_s=.4) < .7+1e-15
    assert relative_speed_bound(.3, target_speed_m_s=0.) == .3
    with pytest.raises(TypeError):
        relative_speed_bound(.3)
    for unknown in (None, np.nan, np.inf, -.1):
        with pytest.raises(ValueError):
            relative_speed_bound(.3, target_speed_m_s=unknown)


@pytest.mark.parametrize("invalid", [
    SphereBody("body", "absent", [[0., 0., 0., .1]]),
    SphereBody("body", "link1", [[0., 0., 0., -1.]]),
    SphereBody("body", "link1", []),
])
def test_complete_cover_needs_existing_frame_and_nonnegative_spheres(invalid):
    with pytest.raises(ValueError):
        build(model([0], [pin.JointModelRZ()]), [invalid])
# END tests verify model bounds and frame contracts, not execution outcomes.
