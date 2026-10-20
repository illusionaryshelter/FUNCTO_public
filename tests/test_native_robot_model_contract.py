"""Focused regressions for the restored known robot model API."""

# BEGIN model records and structural clearance are independent of observations.
import numpy as np
import pytest

from manipulation.robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker
from test_robot_geometry_check import URDF, corners
from test_robot_validation import feedback, model, validator


def test_record_keeps_source_geometry_and_returns_owned_data(tmp_path):
    xml = URDF.replace('<link name="base"/>', '''<link name="base">
      <visual><geometry><box size=".1 .2 .3"/></geometry></visual>
      <collision><geometry><box size=".1 .2 .3"/></geometry></collision>
    </link>''')
    path = tmp_path / "known_robot.urdf"
    path.write_text(xml)
    local = np.eye(4)
    local[:3, 3] = [.1, .2, .3]
    vertices = corners()
    model = RobotGeometryChecker(path, base_frame="base", geometries=(
        NativeSolid("link_shape", "left", vertices, "reusable robot component"),
        NativePrimitive("pad", "right", "box", (.01, .02, .03), local, "known pad geometry"),
    ))
    record = model.model_record()
    assert record["urdf_xml"] == model.urdf_xml
    assert "<collision>" in record["urdf_xml"]
    np.testing.assert_array_equal(record["geometries"][0]["vertices_frame"], vertices)
    np.testing.assert_array_equal(record["geometries"][1]["T_frame_geometry"], local)
    assert record["geometries"][1]["dimensions"] == [.01, .02, .03]
    record["geometries"][0]["vertices_frame"][0][0] = 100.
    record["geometries"][1]["T_frame_geometry"][0][3] = 100.
    again = model.model_record()
    np.testing.assert_array_equal(again["geometries"][0]["vertices_frame"], vertices)
    np.testing.assert_array_equal(again["geometries"][1]["T_frame_geometry"], local)


def test_one_millimetre_structure_gap_is_separated_and_request_can_interrupt(tmp_path):
    path = tmp_path / "known_robot.urdf"
    path.write_text(URDF)
    local = np.eye(4)
    local[0, 3] = 2.001
    model = RobotGeometryChecker(path, base_frame="base", geometries=(
        NativePrimitive("first", "base", "box", (2., 2., 2.), np.eye(4), "known component"),
        NativePrimitive("second", "base", "box", (2., 2., 2.), local, "known component"),
    ))
    args = dict(joint_names=("left_slide", "right_slide"), joint_positions=(.04, -.02),
                T_reference_base=np.eye(4),
                pairs=(PairQuery("first", "second", "required", "declared model pair"),))
    row, = model.query(**args)["pairs"]
    assert row["relation"] == "separated"
    assert row["signed_distance_m"] == pytest.approx(.001, abs=2e-6)

    def stop():
        raise RuntimeError("request interrupted")

    with pytest.raises(RuntimeError, match="request interrupted"):
        model.query(**args, check_request=stop)


def test_executor_consumes_restored_distance_vector_in_declared_pair_order(model):
    from manipulation.execution import TrajectoryExecutor
    gate = validator(model)
    # Exercise the real consumer and producer directly. This test needs no
    # actuator backend because no motion is requested or sent.
    executor = TrajectoryExecutor.__new__(TrajectoryExecutor)
    executor.geometry_checker = gate
    executor._geometry_pair_count = len(gate.describe()["pair_order"])
    result = executor._geometry_result(
        "check_state", feedback(), stage="before_step", trajectory_time=0.)
    assert result["passed"] and result["distance_query_completed"]
    assert len(result["distances_m"]) == executor._geometry_pair_count == 1
    assert result["distances_m"][0] == result["min_required_clearance_m"]
# END no task margin or image coverage changes native structural separation.
