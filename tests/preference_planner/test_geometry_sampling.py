"""Real native/observed queries under the original bounded sampling contract.

Analytic solids and fields are numerical unit fixtures.  The checks retain
the distinction between an actual queried intersection and an unresolved
interval; finite sampling is not a certificate for unsampled times.
"""

from pathlib import Path
import sys

import coal
import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "tests"), str(PROJECT / "tests/planning")]


from manipulation.implicit_distance import ObservedField, PairPermission
from manipulation.interval_geometry import check_interval_geometry
from manipulation.native_field_distance import query_native_field
from test_interval_geometry import ConstantMotion, callback, pair, trajectory


# BEGIN regress real permitted overlap through the original query and sampler.
def test_moving_finite_permission_keeps_inconclusive_separation_diagnostic():
    """Zero enclosure separation does not reject an analytically allowed pair.

    The native box translates linearly by 25 mm inside a 100 mm observed
    occupied cube. Both full solids remain inside the finite [-1, 1]^3
    permission throughout this specific analytic construction. The native
    query consequently reports clear at each queried time, while its full
    entity separation lower bound is zero. The sampler must preserve that
    unavailable separation proof without converting it to an intersection.
    """
    curve = trajectory()
    shape = coal.Box(.02, .02, .02)
    field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[-.05] * 3, voxel_size=.1)
    region = (np.r_[np.eye(3), -np.eye(3)], np.ones(6))
    permission = PairPermission(((region, region),))
    point_results = []

    def evidence(time, side):
        pose = np.eye(4)
        pose[0, 3] = curve._curve(time)[0]
        result = query_native_field(shape, pose, field, np.eye(4), permission=permission)
        point_results.append(result)
        return [pair(**result, permission_kind="finite_region")]

    result = check_interval_geometry(curve, ConstantMotion(), callback(evidence),
        max_query_points=8, spatial_step_m=.1)
    assert result["query_points"] == 2
    assert all(row["status"] == "clear" and row["full_entity_lower_m"] == 0.
               for row in point_results)
    assert not result["confirmed_intersections"] and not result["proved_intervals"]
    unresolved, = result["unresolved_intervals"]
    assert unresolved["reason"] == "spatial_resolution"
    assert unresolved["pairs"][0]["reason"] == "dynamic_permission"


def test_requested_adaptive_scale_finds_real_intersection_between_endpoints():
    """The retained adaptive query still exposes an actual midpoint collision."""
    curve = trajectory()
    field = ObservedField(-np.ones((2, 2, 2)), np.ones((2, 2, 2), bool),
        origin=[.0115, -.001, -.001], voxel_size=.002)
    shape = coal.Box(.002, .002, .002)

    def evidence(time, side):
        pose = np.eye(4)
        pose[0, 3] = curve._curve(time)[0]
        return [pair(**query_native_field(shape, pose, field, np.eye(4)))]

    result = check_interval_geometry(curve, ConstantMotion(), callback(evidence),
        max_query_points=20, spatial_step_m=.001)
    assert result["confirmed_intersections"][0]["time_s"] == .0125
    assert result["query_points"] == 3
    assert result["unresolved_intervals"][0]["reason"] == "not_checked_after_intersection"
# END actual queries and the declared sampling scale retain separate meanings.
