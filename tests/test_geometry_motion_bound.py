"""CPU properties of complete-cover point-speed bounds, without GPU or FK setup."""

# BEGIN analytic and adversarial checks of the independently derived bounds.
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.geometry_motion_bound import (
    SphereBody, build_geometry_motion_bound, relative_speed_bound,
)


def model(parents, kinds, joints, *, names=("q",), limits=None, translations=None, offsets=None):
    count = len(parents)
    transforms = np.zeros((count, 3, 4))
    transforms[:, :3, :3] = np.eye(3)
    if translations is not None:
        transforms[:, :, 3] = translations
    return dict(fixed_transforms=transforms, link_map=np.asarray(parents),
                joint_map_type=np.asarray(kinds), joint_map=np.asarray(joints),
                joint_offset_map=np.tile([1., 0.], count) if offsets is None else np.asarray(offsets).reshape(-1),
                joint_names=names, joint_position_limits=np.array([[-1.]*len(names), [1.]*len(names)]) if limits is None else np.asarray(limits),
                link_name_to_idx_map={f"link{i}": i for i in range(count)})


def body(link=1, *, spheres=((1., 0., 0., .2),), affine=None, name="body"):
    return SphereBody(name, f"link{link}", np.asarray(spheres), {} if affine is None else affine)


def build(metadata, bodies=None, extra=None):
    return build_geometry_motion_bound(**metadata, bodies=[body()] if bodies is None else bodies,
                                       extra_joint_limits={} if extra is None else extra)


def test_revolute_bound_covers_sphere_surface_not_just_its_centre():
    result = build(model([0, 0], [-1, 5], [-1, 0]))
    # At q=0 the point x=1.2 moves at exactly 1.2*|dq|, while the centre
    # moves at |dq|. A centre-only bound would miss this material point.
    assert 1.2 <= result.coefficients[0, 0] < 1.2+1e-12
    assert 3.6 <= result.speed_bounds({"q": 3.})[0] < 3.6+1e-11
    assert not result.coefficients.flags.writeable


def test_fixed_child_offset_is_part_of_ancestor_lever_arm():
    metadata = model([0, 0, 1], [-1, 5, -1], [-1, 0, -1],
                     translations=[[0., 0., 0.], [10., 0., 0.], [2., 0., 0.]])
    result = build(metadata, [body(2, spheres=((.5, 0., 0., .1),))])
    # The root-to-joint translation does not rotate, whereas the child's 2 m
    # translation does. This catches both missing and incorrectly added terms.
    assert 2.6 <= result.coefficients[0, 0] < 2.6+1e-12


def test_downstream_prismatic_full_range_and_negative_mimic_scale():
    metadata = model([0, 0, 1], [-1, 5, 0], [-1, 0, 1], names=("turn", "slide"),
                     limits=[[-1., .1], [1., .3]], offsets=[[1., 0.], [1., 0.], [-2., .4]],
                     translations=[[0., 0., 0.], [0., 0., 0.], [.2, 0., 0.]])
    result = build(metadata, [body(2, spheres=((.1, 0., 0., .05),))])
    assert .55 <= result.coefficients[0, 0] < .55+1e-12
    assert 2. <= result.coefficients[0, 1] < 2.+1e-12
    assert result.speed_bounds({"turn": 0., "slide": 0.})[0] == 0.


def test_repeated_mimic_coordinate_adds_both_physical_joint_contributions():
    metadata = model([0, 0, 1], [-1, 5, 5], [-1, 0, 0],
                     translations=[[0., 0., 0.], [0., 0., 0.], [1., 0., 0.]],
                     offsets=[[1., 0.], [1., 0.], [2., .0]])
    result = build(metadata, [body(2)])
    # p(q)=R(q)*(e_x + R(2q)*1.2e_x): at q=0, |dp/dq|=1+3*1.2.
    assert 4.6 <= result.coefficients[0, 0] < 4.6+1e-12


def test_fixed_loaded_joint_has_no_invented_velocity_term_and_base_is_included():
    metadata = model([0, 0], [-1, -1], [-1, -1], names=(), limits=np.empty((2, 0)))
    result = build(metadata)
    assert result.coefficients.shape == (1, 0)
    assert result.speed_bounds({})[0] == 0.
    moving = model([0, 0], [-1, 0], [-1, 0])
    moving["fixed_transforms"][0, 0, 0] = 1.0000001
    stretched = build(moving)
    assert 1.0000001 <= stretched.coefficients[0, 0] < 1.0000001+1e-12


def test_float32_fixed_rotation_norm_is_never_assumed_exactly_one():
    metadata = model([0, 0], [-1, 5], [-1, 0])
    matrix = Rotation.from_euler("xyz", [.27, -.43, .81]).as_matrix().astype(np.float32)
    matrix *= np.float32(1.000001)
    metadata["fixed_transforms"][1, :3, :3] = matrix
    result = build(metadata)
    # The continuous model represented by the stored numbers can stretch.
    # Compare its independent exact tangent at many angles and cover points.
    for angle in np.linspace(-1., 1., 31):
        tangent = matrix @ np.array([-1.2*np.sin(angle), 1.2*np.cos(angle), 0.])
        assert np.linalg.norm(tangent) <= result.coefficients[0, 0]
    assert result.coefficients[0, 0] < 1.20001


def test_affine_hand_adds_local_velocity_and_full_range_to_rotating_lever():
    metadata = model([0, 0], [-1, 5], [-1, 0], names=("arm",))
    hand = body(spheres=((0., 0., 0., .01),), affine={"finger": [[1., 0., 0.]]})
    result = build(metadata, [hand], {"finger": (0., .08)})
    assert result.joint_names == ("arm", "finger")
    assert .09 <= result.coefficients[0, 0] < .09+1e-12
    assert 1. <= result.coefficients[0, 1] < 1.+1e-12
    assert .28 <= result.speed_bounds({"arm": 2., "finger": .1})[0] < .28+1e-12
    with pytest.raises(ValueError, match="every native and affine"):
        result.speed_bounds({"arm": 2.})
    with pytest.raises(ValueError, match="lacks joint position bounds"):
        build(metadata, [hand])


def test_affine_motion_can_share_existing_active_coordinate_without_double_limits():
    metadata = model([0, 0], [-1, 5], [-1, 0], limits=[[0.], [.1]])
    moving = body(spheres=((0., 0., 0., .1),), affine={"q": [[2., 0., 0.]]})
    result = build(metadata, [moving])
    # Local translation rate <=2, rotational material-point lever <=.3.
    assert 2.3 <= result.coefficients[0, 0] < 2.3+1e-12


def test_different_cover_spheres_and_branched_tree_do_not_drop_extrema():
    metadata = model([0, 0, 0], [-1, 5, 0], [-1, 0, 1], names=("turn", "slide"))
    first = body(spheres=((0., 0., 0., .1), (0., 3., 0., .4)), name="large")
    second = body(2, spheres=((5., 0., 0., .2),), name="slide")
    result = build(metadata, [first, second])
    assert 3.4 <= result.coefficients[0, 0] < 3.4+1e-12
    assert result.coefficients[0, 1] == 0.
    assert result.coefficients[1, 0] == 0.
    assert 1. <= result.coefficients[1, 1] < 1.+1e-12


def test_mixed_chain_bound_dominates_independent_surface_point_velocities():
    # An independent analytic two-axis expression exercises the whole joint
    # box, including rotated fixed offsets and both signs of prismatic speed.
    metadata = model([0, 0, 1, 2], [-1, 5, 0, 4], [-1, 0, 1, 2],
                     names=("z", "slide", "y"), limits=[[-1., -.2, -.9], [1., .4, .9]],
                     translations=[[0., 0., 0.], [.3, -.1, .2], [.4, .2, .1], [.2, 0., 0.]],
                     offsets=[[1., 0.], [1., .1], [-1.5, .03], [2., -.2]])
    result = build(metadata, [body(3, spheres=((.1, .03, -.02, .08),))])
    rng = np.random.default_rng(193)
    for _ in range(100):
        q = rng.uniform(metadata["joint_position_limits"][0], metadata["joint_position_limits"][1])
        dq = rng.uniform(-2., 2., 3)
        direction = rng.normal(size=3)
        point = np.array([.1, .03, -.02])+.08*direction/np.linalg.norm(direction)
        rz = Rotation.from_rotvec([0., 0., q[0]+.1]).as_matrix()
        ry = Rotation.from_rotvec([0., 2*q[2]-.2, 0.]).as_matrix()
        child = np.array([.2, 0., 0.])+ry@point
        local = np.array([.4, .2, .1])+np.array([-1.5*q[1]+.03, 0., 0.])+child
        tangent = (dq[0]*np.cross([0., 0., 1.], rz@local)
                   +rz@(np.array([-1.5*dq[1], 0., 0.])+2*dq[2]*np.cross([0., 1., 0.], ry@point)))
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


@pytest.mark.parametrize("change", [
    {"link_map": [0, 1]},
    {"link_map": [0., 0.]},
    {"joint_map_type": [-1, 11]},
    {"joint_map": [-1, -1]},
    {"joint_position_limits": [[-np.inf], [1.]]},
    {"joint_position_limits": [[2.], [1.]]},
    {"link_name_to_idx_map": {"link0": 0, "link1": 0}},
])
def test_invalid_or_unsupported_native_metadata_is_rejected(change):
    metadata = model([0, 0], [-1, 5], [-1, 0])
    metadata.update(change)
    with pytest.raises(ValueError):
        build(metadata)


@pytest.mark.parametrize("invalid", [
    SphereBody("body", "absent", [[0., 0., 0., .1]], {}),
    SphereBody("body", "link1", [[0., 0., 0., -1.]], {}),
    SphereBody("body", "link1", [[0., 0., 0., .1]], None),
    SphereBody("body", "link1", [[0., 0., 0., .1]], {"q": [[0., 1.]]}),
])
def test_incomplete_body_motion_is_not_silently_static(invalid):
    with pytest.raises(ValueError):
        build(model([0, 0], [-1, 5], [-1, 0]), [invalid])
# END tests certify algebraic properties and input gaps, not robot execution.
