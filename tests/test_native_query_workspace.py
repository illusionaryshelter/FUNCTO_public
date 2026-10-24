"""CPU resource admission checks for fixed tables and transient native chunks."""

# BEGIN actual scene allocations remain distinct before and after table creation.
import numpy as np

from manipulation.robot_geometry_check import PairQuery
from test_native_scene_contract import make_scene


def test_warm_table_keeps_the_same_transient_query_reservation(tmp_path):
    scene = make_scene(tmp_path)
    checker = scene.robot_model
    cold = scene.estimated_workspace_allocation(1, 32)
    assert checker.native_query_resident_nbytes == 0
    checker.query(joint_names=checker.joint_names, joint_positions=[.04, -.02],
                  T_reference_base=np.eye(4), pairs=scene.robot_validator.required_pairs)
    warm = scene.estimated_workspace_allocation(1, 32)
    assert 0 < checker.native_query_resident_nbytes <= checker.native_query_storage_nbytes
    assert warm["cpu_resident"] - cold["cpu_resident"] == checker.native_query_resident_nbytes
    assert cold["cpu_pending_shared"] - warm["cpu_pending_shared"] == checker.native_query_storage_nbytes
    assert warm["cpu_live"] == cold["cpu_live"]
    assert warm["cpu_live"] >= checker.native_query_temporary_nbytes


def test_changed_table_reserves_replacement_without_losing_resident_inventory(tmp_path):
    scene = make_scene(tmp_path)
    checker = scene.robot_model
    required = scene.robot_validator.required_pairs
    reverse = tuple(PairQuery(pair.second, pair.first, pair.policy, pair.source) for pair in required)
    checker.query(joint_names=checker.joint_names, joint_positions=[.04, -.02],
                  T_reference_base=np.eye(4), pairs=reverse)
    replacement = scene.estimated_workspace_allocation(1, 32)
    assert replacement["cpu_resident"] >= checker.native_query_resident_nbytes > 0
    assert replacement["cpu_pending_shared"] >= checker.native_query_storage_nbytes
    assert replacement["cpu_live"] >= checker.native_query_temporary_nbytes
# END warm admission covers the arrays required by the next native query.
