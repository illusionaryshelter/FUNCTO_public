"""Check opt-in profiling without CUDA, simulator construction, or sleeps."""

import cProfile
import json
import pstats

import pytest

from manipulation.runtime_profiling import run_with_optional_profile


def test_default_disabled_preserves_call_and_creates_no_artifacts(tmp_path, monkeypatch):
    def forbidden():
        raise AssertionError("Disabled profiling must not construct a profiler")
    monkeypatch.setattr(cProfile, "Profile", forbidden)
    returned = object()
    calls = []
    def operation(value, *, marker):
        calls.append((value, marker))
        return returned
    assert run_with_optional_profile(operation, 7, marker="checked", output=tmp_path/"profile") is returned
    assert calls == [(7, "checked")]
    assert not (tmp_path/"profile").exists()


def test_enabled_profile_retains_nested_call_counts_and_full_statistics(tmp_path):
    output = tmp_path/"profile"
    def child():
        return sum(range(100))
    def operation():
        return [child() for _ in range(3)]
    assert run_with_optional_profile(operation, enabled=True, output=output) == [4950]*3
    report = json.loads((output/"runtime.json").read_text())
    child_row = next(row for row in report["functions_by_self_seconds"] if row["function"] == "child")
    assert child_row["calls"] == 3
    assert child_row["cumulative_seconds"] >= child_row["self_seconds"] >= 0
    assert report["call_succeeded"] is True
    assert report["elapsed_wall_seconds"] > 0
    assert report["profiled_self_seconds"] > 0
    assert pstats.Stats(str(output/"runtime.pstats")).total_calls == report["total_calls"]


def test_original_exception_is_retained_and_partial_profile_is_written(tmp_path):
    output = tmp_path/"profile"
    original = RuntimeError("physical operation failed")
    def operation():
        raise original
    with pytest.raises(RuntimeError) as caught:
        run_with_optional_profile(operation, enabled=True, output=output)
    assert caught.value is original
    assert json.loads((output/"runtime.json").read_text())["call_succeeded"] is False


def test_profile_export_failure_does_not_replace_operation_failure(tmp_path, monkeypatch, capsys):
    import manipulation.runtime_profiling as profiling
    def fail_export(*args):
        raise OSError("diagnostic storage unavailable")
    monkeypatch.setattr(profiling, "_write_profile", fail_export)
    original = ValueError("original failure")
    def operation():
        raise original
    with pytest.raises(ValueError) as caught:
        run_with_optional_profile(operation, enabled=True, output=tmp_path)
    assert caught.value is original
    assert "Runtime profile export failed" in capsys.readouterr().err


@pytest.mark.parametrize("operation_fails", [False, True])
def test_broken_diagnostic_stream_preserves_return_or_primary_exception(tmp_path, monkeypatch, operation_fails):
    import manipulation.runtime_profiling as profiling
    class ClosedStream:
        def write(self, value):
            raise BrokenPipeError("diagnostic stream is closed")
    def fail_export(*args):
        raise OSError("diagnostic storage unavailable")
    monkeypatch.setattr(profiling, "_write_profile", fail_export)
    monkeypatch.setattr(profiling.sys, "stderr", ClosedStream())
    original, returned = RuntimeError("primary operation failure"), object()
    def operation():
        if operation_fails:
            raise original
        return returned
    if operation_fails:
        with pytest.raises(RuntimeError) as caught:
            run_with_optional_profile(operation, enabled=True, output=tmp_path)
        assert caught.value is original
    else:
        assert run_with_optional_profile(operation, enabled=True, output=tmp_path) is returned


def test_failed_reused_output_preserves_previous_profile(tmp_path, capsys):
    output = tmp_path/"profile"
    output.mkdir()
    paths = [output/"runtime.json", output/"runtime.pstats"]
    for path in paths:
        path.write_bytes(b"previous completed profile")
    original = FileExistsError("trial output already exists")
    def operation():
        raise original
    with pytest.raises(FileExistsError) as caught:
        run_with_optional_profile(operation, enabled=True, output=output)
    assert caught.value is original
    assert all(path.read_bytes() == b"previous completed profile" for path in paths)
    assert "Runtime profile export failed" in capsys.readouterr().err
