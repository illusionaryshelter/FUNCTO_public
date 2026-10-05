"""CPU-only checks of independent host and device admission boundaries."""

# BEGIN test byte boundaries using measured-resource stand-ins, not allocations.
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from manipulation.planning import (
    PlanningBudget, _PlanningWorkspace, _planning_workspace_bytes,
    _require_planning_host_memory, _require_planning_memory,
)
from manipulation.resource_admission import ResourceLimitError, require_host_memory


GIB = 1024**3


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


def test_planner_host_check_adds_only_live_work_and_global_pending(monkeypatch):
    from manipulation.tsdf_geometry import ObservedTsdfGeometry

    monkeypatch.setattr(ObservedTsdfGeometry, "pending_storage_total_nbytes", classmethod(lambda cls: 73))
    admit = Mock()
    monkeypatch.setattr("manipulation.planning.require_host_memory", admit)
    workspace = _PlanningWorkspace(4*GIB, 127, GIB, 0, 61, False)
    _require_planning_host_memory(workspace, max_host_rss_bytes=10*GIB)
    # The four GiB shared capacity is already represented by actual RSS and
    # the geometry's pending count. Adding it again would reject valid work.
    admit.assert_called_once_with(127, pending_existing_bytes=73, max_rss_bytes=10*GIB)


def test_cpu_residency_does_not_change_gpu_budget_or_graph_choice():
    allocation = {"gpu_retained": 5, "gpu_live": 150, "gpu_shared": 23,
                  "gpu_graph_live": 60, "cpu_shared": 17, "cpu_live": 31}
    scene = SimpleNamespace(required_progress_anchors=(), contacts=(),
        estimated_workspace_allocation=lambda *args, **kwargs: dict(allocation))
    metadata = {"active_joint_names": ["joint"], "links": 1, "spheres": 1}
    budget = PlanningBudget(max_scene_gpu_workspace_bytes=13*5+150+23+120)
    small = _planning_workspace_bytes(scene, metadata, budget, 1)
    allocation["cpu_shared"] = 9*GIB
    large = _planning_workspace_bytes(scene, metadata, budget, 1)
    assert small.trajectory_graph is large.trajectory_graph is True
    assert small.gpu_bytes == large.gpu_bytes == budget.max_scene_gpu_workspace_bytes
    assert large.cpu_bytes-small.cpu_bytes == 9*GIB-17


def device_snapshot(*, free=12*GIB, total=12*GIB, reserved=0, allocated=0):
    return SimpleNamespace(cuda=SimpleNamespace(mem_get_info=Mock(return_value=(free, total)),
        memory_reserved=Mock(return_value=reserved), memory_allocated=Mock(return_value=allocated)))


def test_device_limit_disables_only_extra_graph_pools_when_eager_fits():
    runtime = device_snapshot()
    workspace = _PlanningWorkspace(9*GIB, 0, GIB, 4*GIB, 0, True)
    admitted = _require_planning_memory(runtime, "cuda:0", workspace, max_device_used_bytes=6*GIB)
    assert admitted == replace(workspace, trajectory_graph=False)
    assert workspace.trajectory_graph is True
    # An explicit eight-GiB limit admits these graph pools without charging
    # the same nine-GiB host inventory to CUDA.
    assert _require_planning_memory(runtime, "cuda:0", workspace,
                                     max_device_used_bytes=8*GIB) is workspace


def test_eager_device_overflow_is_a_typed_refusal():
    runtime = device_snapshot(free=10*GIB)
    workspace = _PlanningWorkspace(0, 0, 2*GIB+1, GIB, 0, True)
    with pytest.raises(ResourceLimitError) as captured:
        _require_planning_memory(runtime, "cuda:0", workspace, max_device_used_bytes=6*GIB)
    assert (captured.value.resource, captured.value.required_bytes, captured.value.limit_bytes) == (
        "gpu_device_used", 6*GIB+1, 6*GIB)


def test_standalone_device_check_cannot_bypass_the_physical_maximum():
    runtime = device_snapshot()
    with pytest.raises(ValueError, match="eight GiB"):
        _require_planning_memory(runtime, "cuda:0", _PlanningWorkspace(0, 0, 0, 0, 0, False),
                                 max_device_used_bytes=8*GIB+1)
    runtime.cuda.mem_get_info.assert_not_called()


def test_physical_limit_defaults_and_explicit_maxima_are_distinct():
    default = PlanningBudget()
    assert (default.max_scene_gpu_workspace_bytes, default.max_host_rss_bytes,
            default.max_device_used_bytes) == (GIB, 8*GIB, 6*GIB)
    requested = replace(default, max_host_rss_bytes=10*GIB, max_device_used_bytes=8*GIB)
    assert requested.max_host_rss_bytes == 10*GIB
    for values in ({"max_host_rss_bytes": 10*GIB+1}, {"max_device_used_bytes": 8*GIB+1}):
        with pytest.raises(ValueError):
            replace(default, **values)
    with pytest.raises(TypeError):
        PlanningBudget(max_scene_workspace_bytes=GIB)
# END no test constructs a real planner, initializes CUDA or allocates a large array.
