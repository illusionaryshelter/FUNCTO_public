"""Small CPU counterexamples for native new-volume coverage, without physics."""

# BEGIN full-volume coverage tests; no CUDA, rendering, or recorded environment.
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.robot_coverage import CoverageBudgetExceeded, RobotCoverage, _free_rectangles
from manipulation.robot_geometry_check import NativeSolid, RobotGeometryChecker
from manipulation.planning_scene import ObservedSpace


URDF = """<robot name="coverage_slides">
  <link name="base"/><link name="first"/><link name="second"/>
  <joint name="first_slide" type="prismatic"><parent link="base"/><child link="first"/>
    <axis xyz="1 0 0"/><limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
  <joint name="second_slide" type="prismatic"><parent link="base"/><child link="second"/>
    <origin xyz="0 0.5 0"/><axis xyz="1 0 0"/>
    <limit lower="-1" upper="1" effort="10" velocity="1"/>
  </joint>
</robot>"""


def corners(half):
    return half * np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])


@pytest.fixture
def urdf(tmp_path):
    path = tmp_path / "robot.urdf"
    path.write_text(URDF)
    return path


def build(path, *, free=None, q0=(0., 0.), half=.05, base=None, extra=(), **kwargs):
    shape = NativeSolid("body", "first", corners(half), "declared exact native box")
    shapes = (shape, *extra)
    checker = RobotGeometryChecker(path, base_frame="base", geometries=shapes)
    return RobotCoverage(checker=checker, solids=shapes,
        joint_names=("first_slide", "second_slide"), baseline_joint_positions=q0,
        T_reference_base=np.eye(4) if base is None else base,
        free_grid=np.zeros((7, 7, 7), bool) if free is None else free,
        origin=(-.3, -.3, -.3), voxel_size=.1, reference_frame="world",
        initialization_source="explicit valid model-state assumption for this synthetic test", **kwargs)


def query(coverage, first, second=0., *, entrance=False):
    return coverage.check_state(joint_names=("first_slide", "second_slide"),
                                joint_positions=(first, second), entrance=entrance)


def test_exact_cell_merging_preserves_holes_and_disconnected_evidence():
    mask = np.zeros((4, 5, 7), bool)
    mask[:, 1:4, 1:6] = True
    mask[1:3, 2, 3] = False
    mask[0, 0, 0] = True
    boxes = _free_rectangles(mask)
    reconstructed = np.zeros_like(mask, dtype=np.int64)
    for (x0, y0, z0), (x1, y1, z1) in boxes:
        reconstructed[z0:z1, y0:y1, x0:x1] += 1
    np.testing.assert_array_equal(reconstructed, mask.astype(int))
    assert len(_free_rectangles(np.ones((9, 8, 7), bool))) == 1
    assert len(_free_rectangles(np.zeros((3, 3, 3), bool))) == 0


def test_q0_passes_with_no_observed_free_volume_but_new_volume_fails(urdf):
    coverage = build(urdf)
    current = query(coverage, 0., entrance=True)
    assert current["passed"] and current["free_evidence_deficit_volume_m3"] == 0
    assert current["residual_classification"] == "empty"
    future = query(coverage, .08)
    assert not future["passed"] and future["failure"] == "new_volume_without_free_evidence"
    assert future["free_evidence_deficit_volume_m3"] > 0
    # BEGIN F/V0 alone cannot identify occupied material versus unknown space.
    assert future["residual_classification"] == "unclassified"
    assert "unknown_volume_m3" not in future
    assert "unknown_boundary_faces" not in future
    # END the failed predicate keeps its original, strictly positive residual.
    assert future["witness"] is not None
    witness = np.asarray(future["witness"]["point_reference"])
    assert witness[0] > .05
    np.testing.assert_allclose(witness - [.08, 0., 0.], future["witness"]["point_frame"])
    description = coverage.describe()
    assert description["new_free_cells_added"] == 0
    assert description["initial_validity_is_assumption"]
    assert not description["hidden_environment_validated"]
    assert not future["continuous_path_certified"]


def test_new_volume_in_actual_free_cells_passes_without_baseline_mutation(urdf):
    free = np.ones((7, 7, 7), bool)
    initial = free.copy()
    coverage = build(urdf, free=free)
    baseline = coverage.baseline_joint_positions.copy()
    assert query(coverage, .2)["passed"]
    assert query(coverage, -.2)["passed"]
    np.testing.assert_array_equal(free, initial)
    np.testing.assert_array_equal(coverage.baseline_joint_positions, baseline)
    assert not query(coverage, .5)["passed"]  # Grid exterior remains unknown.


# BEGIN scene/oracle evidence identity and cumulative request-budget regressions.
# END cloning a candidate neither widens F nor restarts proof work.


def test_unknown_voxel_inside_robot_volume_is_not_missed_by_surface_vertices(urdf):
    free = np.ones((7, 7, 7), bool)
    free[3, 3, 3] = False
    coverage = build(urdf, free=free, q0=(-.3, 0.), half=.15)
    # All corners of this candidate lie in observed-free cells, but a complete
    # unknown cell is enclosed in its interior. Surface-point checks cannot
    # establish this full-volume predicate.
    result = query(coverage, 0.)
    assert not result["passed"] and result["failure"] == "new_volume_without_free_evidence"
    assert result["free_evidence_deficit_volume_m3"] > 0
    assert result["witness"] is not None
    assert np.max(np.abs(result["witness"]["point_reference"])) < .05


def test_global_initial_union_can_cover_another_component_without_collision_permission(urdf):
    # Both solids belong to the same moving link but are distinct native
    # components. The first component can occupy the second's original volume;
    # this ONLY checks coverage, leaving actual pair collision policy separate.
    extra = NativeSolid("offset_body", "first", corners(.05) + [.2, 0., 0.],
                        "second exact native box at a fixed local offset")
    free = np.zeros((7, 7, 11), bool)
    free[:, :, 5:] = True
    coverage = build(urdf, free=free, extra=(extra,))
    assert query(coverage, .2)["passed"]
    assert coverage.describe()["native_solid_names"] == ["body", "offset_body"]


def test_single_declared_convex_tetrahedron_uses_its_volume_not_aabb(urdf):
    points = np.array([[0., 0., 0.], [.1, 0., 0.], [0., .1, 0.], [0., 0., .1]])
    shape = NativeSolid("tetra", "first", points, "native convex tetrahedron")
    checker = RobotGeometryChecker(urdf, base_frame="base", geometries=(shape,))
    coverage = RobotCoverage(checker=checker, solids=(shape,),
        joint_names=("first_slide", "second_slide"), baseline_joint_positions=(0., 0.),
        T_reference_base=np.eye(4), free_grid=np.zeros((3, 3, 3), bool),
        origin=(-.1, -.1, -.1), voxel_size=.1, reference_frame="world",
        initialization_source="synthetic exact initial tetrahedron")
    assert query(coverage, 0., entrance=True)["passed"]
    result = query(coverage, .02)
    assert not result["passed"]
    guidance = coverage.guidance_geometry()["solids"][0]
    assert len(guidance["faces"]) == 4
    points = guidance["vertices_frame"]
    planes = guidance["baseline_halfspaces_world"]
    assert np.max(points @ planes[:, :3].T + planes[:, 3]) < 1e-12


def test_initial_float32_binding_is_exact_and_never_used_for_future_samples(urdf):
    measured = .03123456789
    represented = float(np.float32(measured))
    assert represented != measured
    coverage = build(urdf, q0=(measured, 0.))
    initial = query(coverage, represented, entrance=True)
    assert initial["passed"] and initial["canonicalized_initial_float32"]
    assert initial["joint_positions"][coverage.baseline_joint_names.index("first_slide")] == measured
    assert not query(coverage, represented)["passed"]
    with pytest.raises(ValueError, match="exact float32"):
        query(coverage, np.nextafter(represented, np.inf), entrance=True)
    with pytest.raises(ValueError, match="exact float32"):
        query(coverage, measured + 1e-6, entrance=True)


def test_small_nonzero_sliver_has_no_user_acceptance_threshold(urdf):
    coverage = build(urdf)
    result = query(coverage, 1e-7)
    assert not result["passed"] and 0 < result["free_evidence_deficit_volume_m3"] < 1e-8
    assert not result["numerics"]["volume_threshold_used"]
    assert not coverage.describe()["acceptance_uses_volume_threshold"]


def test_reference_base_applied_once_and_guidance_does_not_alias_baseline(urdf):
    base = np.eye(4)
    base[:3, :3] = [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]]
    base[:3, 3] = [.7, -.4, .2]
    coverage = build(urdf, base=base, q0=(.03, 0.))
    assert query(coverage, .03, entrance=True)["passed"]
    data = coverage.guidance_geometry()
    row = data["solids"][0]
    np.testing.assert_allclose(row["baseline_pose"][:3, 3], [.7, -.37, .2])
    vertices = row["vertices_frame"] @ row["baseline_pose"][:3, :3].T + row["baseline_pose"][:3, 3]
    planes = row["baseline_halfspaces_world"]
    assert np.max(vertices @ planes[:, :3].T + planes[:, 3]) < 1e-12
    row["baseline_pose"][:] = 0
    data["baseline_joint_positions"][:] = .9
    base[:] = 0
    assert query(coverage, .03, entrance=True)["passed"]
    assert not query(coverage, .06)["passed"]


@pytest.mark.parametrize("names,positions", [
    (("first_slide",), (0.,)),
    (("first_slide", "first_slide"), (0., 0.)),
    (("first_slide", "absent"), (0., 0.)),
    (("first_slide", "second_slide"), (np.nan, 0.)),
])
def test_complete_finite_named_state_is_mandatory(urdf, names, positions):
    coverage = build(urdf)
    with pytest.raises(ValueError, match="every actual robot joint"):
        coverage.check_state(joint_names=names, joint_positions=positions)


def test_query_and_cpu_work_exhaustion_are_rejections(urdf):
    coverage = build(urdf, max_queries=1)
    assert query(coverage, 0.)["passed"]
    assert query(coverage, 0.)["failure"] == "state_query_budget"
    with pytest.raises(CoverageBudgetExceeded, match="CPU budget"):
        build(urdf, max_working_bytes=1024)
    with pytest.raises(ValueError, match="2-GiB"):
        build(urdf, max_working_bytes=3 * 1024**3)
    declared = build(urdf, max_queries=131_073)
    assert declared.describe()["max_queries"] == 131_073
    with pytest.raises(ValueError, match="state-query"):
        build(urdf, max_queries=1_000_001)


def test_input_free_evidence_is_copied_and_not_expanded(urdf):
    free = np.zeros((7, 7, 7), bool)
    coverage = build(urdf, free=free)
    free[:] = True
    assert coverage.describe()["free_cells"] == 0
    assert not query(coverage, .1)["passed"]
# END candidate points or positive volume thresholds never substitute for solids.
