"""Bind native robot new-volume coverage to a verified recorded fixture.

Only this adapter knows fixture/feedback schemas. The production oracle gets
explicit native convex components, the same complete named FK, original free
cells, and a caller-declared valid initial-state assumption. Environment state
and object identities are never read to fill observation gaps.
"""

# BEGIN source-bound native coverage geometry, separate from visual occupancy.
from pathlib import Path
import sys

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.robot_coverage import RobotCoverage
from manipulation.robot_geometry_check import NativeSolid, RobotGeometryChecker
from check_recorded_robot_geometry import _named, _native_shapes
from robot_fixture import _pose, validate_fixture_collision_contract


def build_robot_coverage(fixture, feedback, coverage, *, initialization_source,
                         max_queries=2048, max_working_bytes=2 * 1024**3):
    """Use every articulated native solid and the actual complete measured q0.

    Fixed installation primitives remain ordinary collision owners; they do
    not enlarge V0. Native convex collision components define the declared
    MODEL volume, not an assertion of observed or physical occupied material.
    """
    if not isinstance(initialization_source, str) or not initialization_source.strip():
        raise ValueError("The caller must explicitly identify its valid initial-model assumption")
    config = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
    validate_fixture_collision_contract(fixture, config)
    # BEGIN validate current robot evidence instead of a recording-file label.
    # Both recorded and continuous callers supply the same named q/dq schema.
    # The native query below still validates all q0 values, joint limits and FK;
    # velocities must independently contain one finite value per same joint.
    if (tuple(feedback["joint_names"]) != tuple(fixture["joint_names"])
            or feedback["reference_frame"] != coverage.reference_frame):
        raise ValueError("Coverage needs complete named feedback in the current reference frame")
    _named(feedback["joint_names"], feedback["start_dq"], fixture["joint_names"])
    source = feedback.get("source")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Coverage needs an explicit current robot feedback source")
    base = _pose(fixture["base_pose"]["position"], fixture["base_pose"]["quaternion_wxyz"])
    if not np.array_equal(base, np.asarray(feedback["base_pose"])):
        raise ValueError("Coverage feedback base must retain the verified native model placement")
    tool = _pose(fixture["tool_transform"]["position"], fixture["tool_transform"]["quaternion_wxyz"])
    if (feedback["tool_link"] != fixture["tool_link"]
            or feedback["tool_frame"] != fixture["tcp_site"]
            or not np.array_equal(tool, np.asarray(feedback["tool_transform"]))):
        raise ValueError("Coverage feedback must retain the verified native tool identity and transform")
    # END no episode identity, q0, FK tolerance or initial-volume assumption is substituted.
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    if model.nq > 64 or model.nbody > 128 or model.ngeom > 64:
        raise ValueError("Coverage fixture exceeds the established CPU robot-model budget")
    shapes, _ = _native_shapes(fixture, model)
    expected = tuple(row["geom"] for row in fixture["collision_geometries"])
    selected = tuple(shape for shape in shapes if shape.name in expected)
    if (tuple(shape.name for shape in selected) != expected
            or any(not isinstance(shape, NativeSolid) for shape in selected)
            or set(expected) & {row["geom"] for row in fixture["fixed_collision_geometries"]}):
        raise ValueError("Every articulated native convex component must appear once, without mount primitives")
    checker = RobotGeometryChecker(fixture["urdf_path"], base_frame=fixture["base_link"], geometries=shapes)
    state = checker.query(joint_names=feedback["joint_names"], joint_positions=feedback["start_q"],
                          T_reference_base=feedback["base_pose"], pairs=(), frames=[fixture["tcp_site"]])
    if (state["joint_limit_violations"]
            or not np.allclose(state["frame_poses"][fixture["tcp_site"]], feedback["eef_pose"],
                               atol=1e-8, rtol=0)):
        raise ValueError("Coverage native q0/FK must agree with the current measured entrance")
    return RobotCoverage(checker=checker, solids=selected, joint_names=feedback["joint_names"],
        baseline_joint_positions=feedback["start_q"], T_reference_base=feedback["base_pose"],
        free_grid=coverage.known_free, origin=coverage.origin, voxel_size=coverage.voxel_size,
        reference_frame=coverage.reference_frame,
        initialization_source=f"{initialization_source}; measured snapshot: {source}",
        max_queries=max_queries, max_working_bytes=max_working_bytes)
# END no free cells, contacts, scene obstacles, or physics parameters are changed.
