"""CPU-only checks of independent host and device admission boundaries."""

# BEGIN test byte boundaries using measured-resource stand-ins, not allocations.
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest
import manipulation.resource_admission as resource_admission

from manipulation.planning import (
    PlanningBudget, _PlanningWorkspace, _planning_workspace_bytes,
    _require_planning_host_memory, _require_planning_memory,
)
from manipulation.resource_admission import (
    ResourceLimitError, configure_host_process_root, require_host_memory, require_device_memory,
)


GIB = 1024**3


@pytest.fixture(autouse=True)
def isolated_process_scope(monkeypatch):
    # A production worker binds one scope for its lifetime. Unit tests each
    # represent an independent worker without adding a public reset/bypass.
    monkeypatch.setattr(resource_admission, "_HOST_PROCESS_ROOT_IDENTITY", None)


def host_snapshot(monkeypatch, *, rss=2*GIB, child_rss=GIB, available=8*GIB):
    child = SimpleNamespace(memory_info=Mock(return_value=SimpleNamespace(rss=child_rss)))
    process = SimpleNamespace(memory_info=Mock(return_value=SimpleNamespace(rss=rss)),
                              children=Mock(return_value=[child]))
    monkeypatch.setattr(psutil, "Process", Mock(return_value=process))
    monkeypatch.setattr(psutil, "virtual_memory", Mock(return_value=SimpleNamespace(available=available)))
    return process


def test_host_admission_includes_live_children_and_unmaterialized_geometry(monkeypatch):
    process = host_snapshot(monkeypatch)
    require_host_memory(GIB, pending_existing_bytes=2*GIB, max_rss_bytes=6*GIB)
    process.children.assert_called_once_with(recursive=True)
    # The old/new geometry may still be lazy. An extra promised byte must
    # refuse before allocation even though the current RSS has not grown.
    with pytest.raises(ResourceLimitError) as captured:
        require_host_memory(GIB+1, pending_existing_bytes=2*GIB, max_rss_bytes=6*GIB)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "host_process_rss", 6*GIB+1, 6*GIB)


def test_host_admission_retains_two_gib_system_headroom(monkeypatch):
    host_snapshot(monkeypatch, rss=GIB, child_rss=0, available=4*GIB-1)
    with pytest.raises(ResourceLimitError) as captured:
        require_host_memory(GIB, pending_existing_bytes=GIB, max_rss_bytes=10*GIB)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "host_available", 4*GIB, 4*GIB-1)


def test_planner_host_check_adds_live_work_to_already_resident_scene(monkeypatch):
    admit = Mock()
    monkeypatch.setattr("manipulation.planning.require_host_memory", admit)
    workspace = _PlanningWorkspace(4*GIB, 127, GIB, 61)
    _require_planning_host_memory(workspace.cpu_live_bytes,
        encoded_record_bytes_upper=11, max_host_rss_bytes=10*GIB)
    # The frozen observed fields are resident and represented by measured RSS.
    # Only new live query work and encoded records require another allocation.
    admit.assert_called_once_with(127+3*11, pending_existing_bytes=0, max_rss_bytes=10*GIB)


def test_new_validation_records_include_previous_candidate_rss_at_one_byte_boundary(monkeypatch):
    from manipulation.tsdf_geometry import ObservedTsdfGeometry

    monkeypatch.setattr(ObservedTsdfGeometry, "pending_storage_total_nbytes", classmethod(lambda cls: 0))
    process = host_snapshot(monkeypatch, rss=10*GIB, child_rss=GIB, available=4*GIB)
    encoded_bytes = (GIB-1)//3
    _require_planning_host_memory(1, encoded_record_bytes_upper=encoded_bytes, max_host_rss_bytes=12*GIB)
    # One additional resident byte, for example from an earlier candidate,
    # cannot be spent again by the new candidate's records/serialization.
    process.memory_info.return_value.rss += 1
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_host_memory(1, encoded_record_bytes_upper=encoded_bytes, max_host_rss_bytes=12*GIB)
    assert (captured.value.resource, captured.value.required_bytes) == ("host_process_rss", 12*GIB+1)


def test_cpu_residency_does_not_change_gpu_budget():
    allocation = {"gpu_retained": 5, "gpu_live": 150, "gpu_shared": 23,
                  "gpu_graph_live": 60, "cpu_shared": 17, "cpu_live": 31}
    scene = SimpleNamespace(required_progress_anchors=(), contacts=(),
        estimated_workspace_allocation=lambda *args, **kwargs: dict(allocation))
    metadata = {"active_joint_names": ["joint"], "links": 1, "spheres": 1}
    budget = PlanningBudget(max_scene_gpu_workspace_bytes=13*5+150+23)
    small = _planning_workspace_bytes(scene, metadata, budget, 1)
    allocation["cpu_shared"] = 9*GIB
    large = _planning_workspace_bytes(scene, metadata, budget, 1)
    assert small.gpu_bytes == large.gpu_bytes == budget.max_scene_gpu_workspace_bytes
    assert large.cpu_bytes-small.cpu_bytes == 9*GIB-17


def device_snapshot(*, free=12*GIB, total=12*GIB, reserved=0, allocated=0):
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=Mock(return_value=(free, total)),
        memory_reserved=Mock(return_value=reserved), memory_allocated=Mock(return_value=allocated)))


def test_device_admission_counts_gpu_storage_independently_of_host_inventory():
    runtime = device_snapshot()
    workspace = _PlanningWorkspace(9*GIB, 0, GIB, 0)
    admitted = _require_planning_memory(runtime, "cuda:0", workspace, max_device_used_bytes=6*GIB)
    assert admitted is workspace


def test_eager_device_overflow_is_a_typed_refusal():
    runtime = device_snapshot(free=10*GIB)
    workspace = _PlanningWorkspace(0, 0, 2*GIB+1, 0)
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(runtime, "cuda:0", workspace, max_device_used_bytes=6*GIB)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_device_used", 6*GIB+1, 6*GIB)


def test_standalone_device_check_cannot_bypass_the_physical_maximum():
    runtime = device_snapshot()
    with pytest.raises(ValueError, match="ten GiB"):
        _require_planning_memory(runtime, "cuda:0", _PlanningWorkspace(0, 0, 0, 0),
                                 max_device_used_bytes=10*GIB+1)
    runtime.cuda.mem_get_info.assert_not_called()


def test_physical_limit_defaults_and_explicit_maxima_are_distinct():
    default = PlanningBudget()
    assert (default.max_scene_gpu_workspace_bytes, default.max_host_rss_bytes,
            default.max_device_used_bytes) == (GIB, 8*GIB, 6*GIB)
    requested = replace(default, max_host_rss_bytes=12*GIB, max_device_used_bytes=8*GIB)
    assert requested.max_host_rss_bytes == 12*GIB
    for values in ({"max_host_rss_bytes": 12*GIB+1}, {"max_device_used_bytes": 10*GIB+1}):
        with pytest.raises(ValueError):
            replace(default, **values)
    with pytest.raises(TypeError):
        PlanningBudget(max_scene_workspace_bytes=GIB)


# BEGIN allocator identity and whole-device usage remain distinct byte limits.
def test_device_admission_reuses_torch_cache_without_charging_existing_maps_twice():
    runtime = device_snapshot(free=6*GIB, reserved=3*GIB, allocated=GIB)
    # Six GiB is already resident across all applications. Two GiB of future
    # tensors reuse owned Torch cache; only the native GiB increases usage.
    assert require_device_memory(runtime, "cuda:0", native_bytes=GIB,
        torch_bytes=2*GIB, max_device_used_bytes=8*GIB) == GIB


def test_native_mapping_cannot_spend_unused_torch_cache():
    runtime = device_snapshot(free=4*GIB, reserved=3*GIB, allocated=GIB)
    with pytest.raises(ResourceLimitError) as captured:
        require_device_memory(runtime, "cuda:0", native_bytes=1,
            max_device_used_bytes=8*GIB)
    assert (captured.value.resource, captured.value.required_bytes) == ("gpu_device_used", 8*GIB+1)
    runtime.cuda.memory_reserved.assert_not_called()
    runtime.cuda.memory_allocated.assert_not_called()


def test_native_mapping_keeps_free_headroom_on_a_smaller_device():
    runtime = device_snapshot(free=3*GIB, total=8*GIB)
    with pytest.raises(ResourceLimitError) as captured:
        require_device_memory(runtime, "cuda:0", native_bytes=GIB+1,
            max_device_used_bytes=8*GIB)
    assert (captured.value.resource, captured.value.required_bytes) == ("gpu_free_reserve", 3*GIB+1)


def test_host_accepts_explicit_twelve_gib_but_never_more(monkeypatch):
    process = host_snapshot(monkeypatch, rss=10*GIB, child_rss=GIB, available=4*GIB)
    require_host_memory(GIB, pending_existing_bytes=0, max_rss_bytes=12*GIB)
    process.memory_info.reset_mock()
    with pytest.raises(ValueError, match="twelve GiB"):
        require_host_memory(0, pending_existing_bytes=0, max_rss_bytes=12*GIB+1)
    process.memory_info.assert_not_called()


def process_scope_snapshot(monkeypatch):
    root = SimpleNamespace(pid=10, create_time=Mock(return_value=123.),
        memory_info=Mock(return_value=SimpleNamespace(rss=9*GIB)), children=Mock())
    current = SimpleNamespace(pid=11, create_time=Mock(return_value=124.),
        memory_info=Mock(return_value=SimpleNamespace(rss=GIB)), parents=Mock(return_value=[root]))
    sibling = SimpleNamespace(memory_info=Mock(return_value=SimpleNamespace(rss=GIB)))
    root.children.return_value = [current, sibling]
    monkeypatch.setattr(psutil, "Process", Mock(side_effect=lambda pid=None: current if pid is None else root))
    monkeypatch.setattr(psutil, "virtual_memory", Mock(return_value=SimpleNamespace(available=4*GIB)))
    return root, current


def test_explicit_parent_scope_counts_retained_simulator_and_other_descendants(monkeypatch):
    root, current = process_scope_snapshot(monkeypatch)
    assert configure_host_process_root(root.pid) == {"pid": 10, "create_time": 123.}
    require_host_memory(GIB, pending_existing_bytes=0, max_rss_bytes=12*GIB)
    root.children.assert_called_once_with(recursive=True)
    current.memory_info.assert_called_once()
    with pytest.raises(ResourceLimitError) as captured:
        require_host_memory(GIB+1, pending_existing_bytes=0, max_rss_bytes=12*GIB)
    assert captured.value.required_bytes == 12*GIB+1


def test_process_scope_rejects_unrelated_pid_and_cannot_be_narrowed(monkeypatch):
    root, current = process_scope_snapshot(monkeypatch)
    with pytest.raises(ValueError, match="ancestors"):
        configure_host_process_root(999)
    configure_host_process_root(root.pid)
    configure_host_process_root(root.pid)
    with pytest.raises(ValueError, match="cannot change"):
        configure_host_process_root(current.pid)


def test_reused_root_pid_refuses_before_reading_replacement_process_memory(monkeypatch):
    root, _ = process_scope_snapshot(monkeypatch)
    configure_host_process_root(root.pid)
    root.create_time.return_value = 125.
    with pytest.raises(RuntimeError, match="PID was reused"):
        require_host_memory(0, pending_existing_bytes=0, max_rss_bytes=12*GIB)
    root.memory_info.assert_not_called()


def test_reparented_worker_cannot_admit_an_empty_old_simulator_tree(monkeypatch):
    root, current = process_scope_snapshot(monkeypatch)
    configure_host_process_root(root.pid)
    current.parents.return_value = []
    with pytest.raises(RuntimeError, match="no longer contains"):
        require_host_memory(0, pending_existing_bytes=0, max_rss_bytes=12*GIB)
    root.memory_info.assert_not_called()
# END resource tests allocate only small Python stand-ins.
# END no test constructs a real planner, initializes CUDA or allocates a large array.
