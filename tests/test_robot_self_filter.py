"""CPU geometry and evidence-contract regressions for three-state self filtering."""

# BEGIN finite native solid filtering, without rendering or a planner.
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manipulation.robot_self_filter import (
    AMBIGUOUS_SHADOW, ENVIRONMENT, SELF, RobotCollisionShape,
    classify_depth, validate_self_filter,
)


def shape(kind="box", centre=(0., 0., 2.), dimensions=(2., 2., 2.), name="body"):
    pose = np.eye(4)
    pose[:3, 3] = centre
    return RobotCollisionShape(name, kind, pose, "declared test collision component", tuple(dimensions))


def classify(point, *, camera=(0., 0., 0.), geometries=None, scale=1., offset=.1):
    point, camera = np.asarray(point, float), np.asarray(camera, float)
    delta = point - camera
    pose = np.eye(4)
    pose[:3, 3] = camera
    K = np.eye(3)
    K[0, 2], K[1, 2] = -delta[:2] / delta[2]
    return classify_depth(observed_depth=np.array([[delta[2]]]), K=K, T_world_camera=pose,
        geometries=[shape()] if geometries is None else geometries,
        padding_scale=scale, padding_offset_m=offset, source="explicit CPU test uncertainty")


@pytest.mark.parametrize("z,expected", [(.8, ENVIRONMENT), (.95, AMBIGUOUS_SHADOW),
    (1., AMBIGUOUS_SHADOW), (2., SELF), (4., AMBIGUOUS_SHADOW)])
def test_front_surface_shell_boundary_interior_and_shadow(z, expected):
    result = classify((0., 0., z))["self_filter"]
    assert result["labels"][0, 0] == expected
    assert result["free_ray_depth_m"][0, 0] == (z if expected == ENVIRONMENT else 0.)


def test_euclidean_padding_keeps_diagonal_foreground_beyond_rounded_corner():
    # A face-offset enlarged box would falsely hide this ray. Its clearance
    # from the native corner is sqrt(.09^2+.09^2), greater than padding .1.
    result = classify((1.09, 1.09, 4.), camera=(1.09, 1.09, 0.))
    assert result["self_filter"]["labels"][0, 0] == ENVIRONMENT


@pytest.mark.parametrize("offset,x,expected", [(0., 1., AMBIGUOUS_SHADOW),
    (.1, 1.1, AMBIGUOUS_SHADOW), (.1, 1.1001, ENVIRONMENT)])
def test_zero_padding_and_padded_tangent_lines_remain_unknown(offset, x, expected):
    assert classify((x, 0., 4.), camera=(x, 0., 0.), offset=offset)["self_filter"]["labels"][0, 0] == expected


@pytest.mark.parametrize("camera_z", [2., .95])
def test_camera_inside_nominal_or_padding_cannot_certify_environment(camera_z):
    result = classify((0., 0., 4.), camera=(0., 0., camera_z))
    assert result["self_filter"]["labels"][0, 0] == AMBIGUOUS_SHADOW


def test_scaling_is_about_each_local_geometry_origin():
    shifted = shape(centre=(2., 0., 2.), dimensions=(.2, .2, .2))
    result = classify((2.19, 0., 2.), camera=(2.19, 0., 0.), geometries=[shifted], scale=2., offset=0.)
    assert result["self_filter"]["labels"][0, 0] == AMBIGUOUS_SHADOW
    assert result["provenance"]["geometries"][0]["T_world_geometry"][0][3] == 2.


def test_native_components_keep_empty_gap_instead_of_assembly_convex_hull():
    pieces = [shape(centre=(x, 0., 2.), dimensions=(.2, .2, .2), name=f"piece{i}")
              for i, x in enumerate((-.3, .3))]
    assert classify((0., 0., 4.), geometries=pieces, offset=0.)["self_filter"]["labels"][0, 0] == ENVIRONMENT


def test_rotated_analytic_cylinder_and_convex_collision_volume():
    cylinder = shape("cylinder", dimensions=(.1, 2.))
    cylinder.T_world_geometry[:3, :3] = Rotation.from_euler("y", 90, degrees=True).as_matrix()
    assert classify((.8, 0., 2.), geometries=[cylinder])["self_filter"]["labels"][0, 0] == SELF
    vertices = np.array([[x, y, z] for x in (-1., 1.) for y in (-1., 1.) for z in (-1., 1.)])
    convex = replace(shape(), kind="convex", dimensions=(), vertices=vertices)
    assert classify((0., 0., 2.), geometries=[convex])["self_filter"]["labels"][0, 0] == SELF


def test_inside_back_component_keeps_explicit_model_relative_self_priority():
    pieces = [shape(centre=(0., 0., z), dimensions=(.2, .2, .2), name=f"piece{i}")
              for i, z in enumerate((1., 2.))]
    result = classify((0., 0., 2.), geometries=pieces)
    assert result["self_filter"]["labels"][0, 0] == SELF
    assert "not certified physical" in result["provenance"]["self_scope"]


def test_invalid_depth_and_nonfinite_native_queries_never_create_evidence():
    depth = np.array([[0., -1., np.nan, np.inf]])
    result = classify_depth(observed_depth=depth, K=np.eye(3), T_world_camera=np.eye(4),
        geometries=[shape()], padding_scale=1., padding_offset_m=.1, source="test")["self_filter"]
    assert np.all(result["labels"] == AMBIGUOUS_SHADOW)
    assert np.all(result["free_ray_depth_m"] == 0.)
    with patch("manipulation.robot_self_filter.coal.distance", return_value=np.nan):
        result = classify((0., 0., 4.))
    assert result["self_filter"]["labels"][0, 0] == AMBIGUOUS_SHADOW
    assert result["provenance"]["counts"]["nonfinite_queries"] == 1


@pytest.mark.parametrize("mutation", ["unknown_free", "extended_free", "invalid_env", "old_mask", "wrong_dtype"])
def test_evidence_contract_rejects_invented_free_rays_and_legacy_masks(mutation):
    depth = np.array([[1., 2.]])
    value = {"labels": np.array([[ENVIRONMENT, SELF]], dtype=np.uint8),
             "free_ray_depth_m": np.array([[1., 0.]]), "source": "test"}
    if mutation == "unknown_free":
        value["free_ray_depth_m"][0, 1] = 2.
    elif mutation == "extended_free":
        value["free_ray_depth_m"][0, 0] = 1.1
    elif mutation == "invalid_env":
        depth[0, 0] = np.nan
    elif mutation == "old_mask":
        value = {"robot_mask": np.array([[False, True]])}
    else:
        value["labels"] = value["labels"].astype(int)
    with pytest.raises(ValueError):
        validate_self_filter(value, depth)
# END no scene ownership or free-space evidence is manufactured by masking.
