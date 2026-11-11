"""Independent volume and support expectations for observed trilinear material."""

# BEGIN observed volume and continuous support regressions use analytic fields.
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from manipulation.goals import SupportSurface
from manipulation.implicit_distance import ObservedField
from manipulation.observed_support import (
    estimate_occupied_volume_center, place_observed_on, support_extrema,
)


def occupied_box(shape=(3, 3, 3), *, origin=(0., 0., 0.), size=.01, observed=None):
    """A negative constant over the declared supported rectangular grid."""
    values = -np.ones(shape)
    return ObservedField(values, np.ones(shape, bool) if observed is None else observed,
                         origin=origin, voxel_size=size)


def support(*, half_width=.1, origin=(0., 0., 0.), angle=0.):
    """Express one world-aligned square in an arbitrary tangent chart."""
    basis = Rotation.from_rotvec([0., 0., angle]).as_matrix()[:, :2]
    vertices = np.array([[-half_width, -half_width], [half_width, -half_width],
                         [half_width, half_width], [-half_width, half_width]])
    world_vertices = vertices @ np.eye(3)[:, :2].T
    boundary = world_vertices @ basis
    return SupportSurface(origin, basis, [0., 0., 1.], boundary,
                          "Explicit observed support fixture")


def assert_interval(interval, expected, precision=1e-6):
    """Check the analytic value and the declared numerical interval width."""
    lower, upper = interval
    assert lower <= expected + 1e-14
    assert upper >= expected - 1e-14
    assert upper - lower <= precision


def test_uniform_occupied_box_has_analytic_volume_center():
    field = occupied_box((3, 4, 5), origin=(.3, -.2, .7), size=.02)
    result = estimate_occupied_volume_center(field)
    np.testing.assert_allclose(result["center_local"], [.32, -.17, .74], atol=1e-14)
    assert result["volume_m3"] == pytest.approx(.04 * .06 * .08)
    assert result["quadrature_spacing_m"] == .01
    assert result["occupied_sample_count"] == 2 * 3 * 4 * 8


def test_partial_trilinear_cell_is_integrated_as_material_not_candidate_volume():
    # f=x-1/2 splits the one fully observed unit cell into equal halves.
    values = np.broadcast_to(np.array([-.5, .5])[:, None, None], (2, 2, 2))
    field = ObservedField(values, np.ones((2, 2, 2), bool),
                          origin=[0., 0., 0.], voxel_size=1.)
    result = estimate_occupied_volume_center(field)
    np.testing.assert_allclose(result["center_local"], [.25, .5, .5])
    assert result["volume_m3"] == .5
    assert result["occupied_sample_count"] == 4


def test_unknown_cell_cannot_contribute_occupied_volume():
    observed = np.ones((3, 2, 2), bool)
    observed[2] = False
    field = occupied_box((3, 2, 2), observed=observed, size=1.)
    result = estimate_occupied_volume_center(field)
    np.testing.assert_allclose(result["center_local"], [.5, .5, .5])
    assert result["volume_m3"] == 1.
    assert result["occupied_sample_count"] == 8


def test_zero_quadrature_volume_has_no_center():
    field = ObservedField(np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
                          origin=[0., 0., 0.], voxel_size=1.)
    with pytest.raises(ValueError, match="zero volume"):
        estimate_occupied_volume_center(field)


def test_rotating_rectangular_material_changes_its_minimum_height():
    field = occupied_box((7, 3, 2), origin=(-.03, -.01, -.005))
    surface = support()
    pose = np.eye(4)
    upright = support_extrema(field, pose, surface, precision_m=1e-7)
    pose[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    turned = support_extrema(field, pose, surface, precision_m=1e-7)
    assert_interval((upright["normal"]["lower_m"], upright["normal"]["upper_m"]), -.005, 1e-7)
    assert_interval((turned["normal"]["lower_m"], turned["normal"]["upper_m"]), -.03, 1e-7)
    for result in (upright, turned):
        assert result["normal"]["status"] == "bounded"
        assert all(status == "bounded" for status in result["footprint"]["statuses"])


def test_each_finite_edge_bound_has_a_feasible_original_field_witness():
    field = occupied_box((5, 3, 2), origin=(-.02, -.01, -.005))
    pose = np.eye(4)
    pose[:3, 3] = [.03, -.02, .1]
    surface = support(origin=(.01, -.01, .2))
    result = support_extrema(field, pose, surface, precision_m=1e-7)
    box_corners = np.array([[x, y, z] for x in (-.02, .02)
                            for y in (-.01, .01) for z in (-.005, .005)])
    world = box_corners + pose[:3, 3] - surface.origin
    for direction, lower, upper, witness in zip(
            result["footprint"]["directions_world"], result["footprint"]["lower_m"],
            result["footprint"]["upper_m"], result["footprint"]["witnesses_local"]):
        assert_interval((lower, upper), np.max(world @ direction), 1e-7)
        value, valid, _, _ = field.sample(np.asarray(witness))
        assert valid and value <= 0


def test_taller_current_material_raises_center_and_preserves_rotation():
    surface = support(origin=(.1, -.2, .4))
    preferred = np.eye(4)
    preferred[:3, :3] = Rotation.from_euler("z", .4).as_matrix()
    preferred[:3, 3] = [.11, -.22, 9.]
    results = []
    for layers, height in ((3, .02), (9, .08)):
        field = occupied_box((3, 3, layers), origin=(-.01, -.01, -height / 2))
        center = estimate_occupied_volume_center(field)["center_local"]
        pose, evidence = place_observed_on(preferred, field, surface, center_local=center,
                                           margin=.001, precision_m=1e-7)
        np.testing.assert_array_equal(pose[:3, :3], preferred[:3, :3])
        np.testing.assert_allclose(evidence["center_world"][:2], preferred[:2, 3], atol=1e-12)
        assert evidence["center_world"][2] == pytest.approx(.4 + height / 2, abs=1e-7)
        assert_interval(evidence["minimum_gap_interval_m"], 0., 1e-7)
        results.append(evidence)
    assert results[1]["center_world"][2] - results[0]["center_world"][2] == pytest.approx(.03, abs=1e-7)


def test_finite_patch_projects_center_and_rejects_a_shape_that_cannot_fit():
    field = occupied_box((5, 3, 2), origin=(-.02, -.01, -.005))
    center = estimate_occupied_volume_center(field)["center_local"]
    preferred = np.eye(4)
    preferred[:3, 3] = [.2, -.2, .1]
    pose, evidence = place_observed_on(preferred, field, support(half_width=.05),
        center_local=center, margin=.003, precision_m=1e-7)
    np.testing.assert_allclose(evidence["center_world"][:2], [.027, -.037], atol=1e-7)
    assert max(evidence["footprint"]["inequality_upper_m"]) <= 1e-7
    assert pose[2, 3] == pytest.approx(.005, abs=1e-7)
    with pytest.raises(ValueError, match="does not fit"):
        place_observed_on(preferred, field, support(half_width=.015),
            center_local=center, margin=0., precision_m=1e-7)


def test_changing_only_the_support_chart_preserves_the_world_placement():
    field = occupied_box((5, 3, 2), origin=(-.02, -.01, -.005))
    center = estimate_occupied_volume_center(field)["center_local"]
    preferred = np.eye(4)
    preferred[:3, :3] = Rotation.from_euler("z", .3).as_matrix()
    preferred[:3, 3] = [.2, -.2, .1]
    poses = [place_observed_on(preferred, field, support(half_width=.05, angle=angle),
                center_local=center, margin=.002, precision_m=1e-7)[0]
             for angle in (0., .73)]
    np.testing.assert_allclose(poses[0], poses[1], atol=1e-10)


def test_local_field_translation_preserves_the_physical_target():
    first = occupied_box((5, 3, 2), origin=(-.02, -.01, -.005))
    shift = np.array([.31, -.27, .43])
    second = occupied_box((5, 3, 2), origin=np.asarray(first.origin) - shift)
    preferred = np.eye(4)
    preferred[:3, :3] = Rotation.from_euler("xyz", [.1, .4, -.2]).as_matrix()
    preferred[:3, 3] = [.03, -.04, .1]
    outputs = [place_observed_on(preferred, field, support(),
        center_local=estimate_occupied_volume_center(field)["center_local"],
        margin=.002, precision_m=1e-6)[0] for field in (first, second)]
    np.testing.assert_allclose(outputs[0][:3, :3], outputs[1][:3, :3], atol=1e-14)
    np.testing.assert_allclose(outputs[0][:3, 3],
        outputs[1][:3, 3] - preferred[:3, :3] @ shift, atol=1e-6)


def test_missing_observation_retains_query_states_and_has_no_placement():
    field = occupied_box(observed=np.zeros((3, 3, 3), bool))
    surface = support()
    result = support_extrema(field, np.eye(4), surface, precision_m=1e-7)
    assert result["normal"]["status"] == "missing_observation"
    assert result["normal"]["lower_m"] is None
    assert all(value is None for value in result["footprint"]["upper_m"])
    assert set(result["footprint"]["statuses"]) == {"missing_observation"}
    with pytest.raises(ValueError, match="requires represented observed occupancy"):
        place_observed_on(np.eye(4), field, surface,
            center_local=[0., 0., 0.], margin=0., precision_m=1e-7)


def test_finite_unresolved_queries_remain_evidence_without_an_added_placement_gate(monkeypatch):
    """The initializer consumes valid bounds without redefining query status."""
    import manipulation.observed_support as component
    original = component.query_minimum_height

    def unresolved(*args, **kwargs):
        result = original(*args, **kwargs)
        return {**result, "status": "unresolved"}

    monkeypatch.setattr(component, "query_minimum_height", unresolved)
    field = occupied_box(origin=(-.01, -.01, -.01))
    _, evidence = place_observed_on(np.eye(4), field, support(),
        center_local=[0., 0., 0.], margin=0., precision_m=1e-7)
    assert evidence["normal"]["status"] == "unresolved"
    assert set(evidence["footprint"]["statuses"]) == {"unresolved"}
    assert_interval(evidence["minimum_gap_interval_m"], 0., 1e-7)
# END analytic expectations cover volume, unknown support, rotation and finite support.
