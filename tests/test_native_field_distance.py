"""Mixed-geometry decisions preserve native shape and raw TSDF semantics."""

# BEGIN exercise geometric claims, finite permissions and unresolved boundaries.
import coal
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation.implicit_distance import ObservedField, PairPermission
from manipulation.native_field_distance import query_native_field
from manipulation.robot_geometry_check import _convex_solid


def solid_cell(lower, size=.02):
    return ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                         origin=lower, voxel_size=size)


def query(shape, field, *, pose=None, permission=None):
    return query_native_field(shape, np.eye(4) if pose is None else pose,
                              field, np.eye(4), permission=permission)


# BEGIN all positive gaps share the same nonpenetration rule.
@pytest.mark.parametrize("gap,status", [(.003, "clear"), (.001, "clear"), (1e-7, "clear"),
                                      (-1e-7, "intersection"), (-.01, "intersection")])
def test_known_box_against_original_supported_cell(gap, status):
    result = query(coal.Box(.02, .02, .02), solid_cell([.01 + gap, -.005, -.005]))
    assert result["status"] == status
    if status == "intersection":
        assert result["upper_m"] == 0
        assert result["witness_first_bounds"] == result["witness_second_bounds"]
        assert result["witness_coordinates"] == "interval_midpoints_for_display_and_derivative"


def test_overlapping_axis_aligned_proxies_are_not_shape_intersection():
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    # The thin diagonal bar's enclosing box contains this cell, but the bar
    # itself is far away. Native support projections establish separation.
    result = query(coal.Box(.2, .004, .02), solid_cell([.055, -.06, -.005], .005), pose=pose)
    assert result["status"] == "clear"
    assert result["intersection"] is False


def test_cylinder_retains_curved_shape_outside_its_enclosing_box_corner():
    result = query(coal.Cylinder(.05, .1), solid_cell([.041, .041, -.005], .004))
    assert result["status"] == "clear"


def test_whole_pair_permission_and_finite_partial_permission_differ():
    shape, field = coal.Box(.02, .02, .02), solid_cell([-.005, -.005, -.005])
    normals = np.vstack((np.eye(3), -np.eye(3)))
    whole = (normals, np.ones(6))
    assert query(shape, field, permission=PairPermission(((whole, whole),)))["status"] == "clear"
    left = (np.array([[1., 0., 0.]]), np.array([-.002]))
    result = query(shape, field, permission=PairPermission(((left, whole),)))
    assert result["status"] == "intersection"
    assert result["witness_first_bounds"][0][0] > -.002


def test_empty_observation_does_not_invent_task_material():
    field = ObservedField(-np.ones((2, 2, 2)), np.zeros((2, 2, 2), bool),
                          origin=[0, 0, 0], voxel_size=.01)
    result = query(coal.Box(.1, .1, .1), field)
    assert result["status"] == "clear" and result["empty_set"]


def test_touching_support_face_remains_unresolved_instead_of_becoming_collision():
    result = query(coal.Box(.02, .02, .02), solid_cell([.01, -.005, -.005]))
    assert result["status"] == "unresolved"
    assert result["intersection"] is None


def test_constant_zero_cell_has_occupied_volume():
    field = ObservedField(np.zeros((2, 2, 2)), np.ones((2, 2, 2), bool),
                          origin=[-.005] * 3, voxel_size=.01)
    assert query(coal.Box(.02, .02, .02), field)["status"] == "intersection"


def test_negative_corner_inside_native_shape_is_found_after_refinement():
    values = np.ones((2, 2, 2))
    values[0, 0, 0] = -1.
    field = ObservedField(values, np.ones_like(values, bool), origin=[0.] * 3, voxel_size=1.)
    assert query(coal.Box(4., 4., 4.), field)["status"] == "intersection"


def test_a_touching_cell_does_not_hide_penetration_in_another_cell():
    # Two disconnected supported cells: the first touches the box's upper
    # face; the second lies inside it. Neither the unknown gap nor the first
    # cell may terminate the query over this entity pair.
    values = -np.ones((5, 4, 2))
    observed = np.zeros_like(values, bool)
    observed[:2, 2:4] = True
    observed[3:5, :2] = True
    field = ObservedField(values, observed, origin=[-.02, -.01, -.005], voxel_size=.01)
    pose = np.eye(4)
    result = query(coal.Box(.08, .02, .02), field, pose=pose)
    assert result["status"] == "intersection"
    assert result["witness_second"][0] > .01
# END unresolved local contact never becomes a pair-wide clear verdict.


@pytest.mark.parametrize("center", [[0., 0., 0.], [.25, .1, 0.], [-.35, .3, .2]])
def test_native_convex_contains_negative_field_at_arbitrary_interior_position(center):
    # The nearest four vertices can all lie on one face. The constructor must
    # still find an actual convex-combination point near this interior target.
    vertices = np.array([(x, y, z) for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])
    field = solid_cell(np.array(center) - .005, .01)
    native = query(_convex_solid(vertices), field)
    analytic = query(coal.Box(1., 1., 1.), field)
    assert native["status"] == analytic["status"] == "intersection"
    assert native["upper_m"] == 0.
# END all tests concern declared geometry, not dataset or skill identities.
