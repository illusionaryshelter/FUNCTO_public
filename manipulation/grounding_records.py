"""Archive completed grounding evidence and complete reference hypotheses.

The writer consumes existing results. Numerical evidence stays in typed NPZ
arrays; continuous paths and physical-time bindings retain their original
operands. HTML is a read-only consumer of source/current RGB-D and results.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .execution_binding import plain
from transfer.features import write_json


def _save_reference(directory, hypothesis, record):
    """Save the same nominal expression and bound source-clock samples."""
    reference = hypothesis.get("reference")
    if reference is None:
        return
    np.savez_compressed(directory / "nominal-reference.npz",
        **{key: value for key, value in reference.items() if isinstance(value, np.ndarray)})
    write_json(directory / "nominal-reference-paths.json", plain(dict(
        frame_indices=reference["frame_indices"], timestamps=reference["timestamps"],
        events=reference["events"], paths={name: dict(path=reference[name + "_path"],
            timestamps=reference[name + "_timestamps"], boundaries=reference[name + "_boundaries"])
            for name in ("object", "eef") if name + "_path" in reference})))
    bound = hypothesis.get("bound_reference")
    if bound is None:
        return
    source_clock = np.asarray(reference["eef_timestamps"])
    elapsed = source_clock - source_clock[0]
    np.savez_compressed(directory / "reference.npz", eef_poses=bound.sample_numpy(elapsed),
        eef_timestamps=source_clock, elapsed_s=elapsed,
        initial_pose=bound.initial_pose, initial_twist=bound.initial_twist,
        source_frame_indices=reference["frame_indices"], source_timestamps=reference["timestamps"])
    write_json(directory / "reference-paths.json", plain(dict(
        source_frame_indices=reference["frame_indices"], source_timestamps=reference["timestamps"],
        source_events=reference["events"], source_time_origin_s=float(source_clock[0]),
        tool_reference=bound.to_record())))
    np.save(directory / "reference-samples.npy", bound.sample_numpy(np.linspace(0., elapsed[-1], 21)))
    record["reference_samples"] = dict(count=21, file="reference-samples.npy")


def save(result, request, output_directory):
    """Save common evidence, every hypothesis, and an immediately openable HTML.

    The manifest references ``hypotheses/<id>`` for both one and many results.
    No hypothesis is selected by this writer. The planning portfolio later
    archives and chooses its own completed curves. An absent physical trace
    remains explicitly missing in every reference view.
    """
    from .reference_visualization import write_reference_html

    directory = Path(output_directory)
    directory.mkdir(parents=True, exist_ok=True)
    excluded = {"prepared_observations", "region_result", "grasp_result", "reference_hypotheses",
                "views", "traceback", "selection"}
    record = plain({key: value for key, value in result.items() if key not in excluded})
    if "selection" in result:
        selection = result["selection"]
        record["selection"] = plain({**{key: value for key, value in selection.items() if key != "views"},
            "views": {identity: {key: value for key, value in view.items() if key != "regions"}
                      for identity, view in selection["views"].items()}})
        masks = {f"{identity}__{item['candidate_id']}": item["mask"]
                 for identity, view in selection["views"].items() for item in view["regions"]}
        if masks:
            np.savez_compressed(directory / "selected-regions.npz", **masks)
            record["selection"]["mask_evidence"] = "selected-regions.npz"
    if "traceback" in result:
        (directory / "error.txt").write_text(result["traceback"])
    binding_id = request["reference_binding_id"]
    views = result.get("views", request["bindings"][binding_id]["views"])
    region = result.get("region_result")
    if region is not None:
        if "arrays" in region:
            numerical = result.get("grasp_result", region)
            np.savez(directory / "grounding-arrays.npz", **numerical["arrays"])
            record["numerical_evidence"] = "grounding-arrays.npz"
            # Native response support and input measurement IDs are preserved
            # separately from both region summaries and the final tool field.
            for view in region["views"]:
                np.savez(directory / (view["id"] + "-native-evidence.npz"),
                    **{key: value for key, value in view.items() if isinstance(value, np.ndarray)})
            report_fields = ("status", "stage", "input_context", "input_evidence", "source_frame",
                "coarse", "coarse_frame", "refinement", "region_frame", "context",
                "mapped_observation_support", "grounding_stage_seconds")
            write_json(directory / "region.json", plain({key: region[key] for key in report_fields if key in region}))
        elif region.get("outputs") is not None:
            from transfer.experiment import store_prediction
            from transfer.multiview_experiment import save_surface
            save_surface(directory, region["outputs"]["surface"])
            store_prediction(directory / "region", region["outputs"]["result"], {})
    if "grasp_result" in result:
        write_json(directory / "grasp.json", plain(result["grasp_result"]["report"]))
    links = []
    for hypothesis in result["reference_hypotheses"]:
        folder = directory / "hypotheses" / hypothesis["id"]
        folder.mkdir(parents=True, exist_ok=True)
        row = plain({key: value for key, value in hypothesis.items()
                     if key not in {"reference", "bound_reference"}})
        diagnostics = hypothesis.get("reference_diagnostics", {})
        row.update(status=hypothesis.get("status", result["status"]), region_status=result["region_status"],
            reference_status=diagnostics.get("reference_status", hypothesis.get("reference_status", "unbound")),
            region_warnings=result.get("region_warnings", []))
        if "failed_stage" in diagnostics:
            row.update(failed_stage=diagnostics["failed_stage"], error=diagnostics["error"])
        reference = hypothesis.get("reference")
        if reference is not None:
            row.update(reference_boundary_checks=plain(reference["boundary_checks"]),
                reference_boundaries={name: plain(reference[name + "_boundaries"])
                    for name in ("object", "eef") if name + "_boundaries" in reference})
        _save_reference(folder, hypothesis, row)
        write_json(folder / "report.json", row)
        write_reference_html(folder, request["skill"], views, region, reference, row,
                             tool_reference=hypothesis.get("bound_reference"))
        links.append(dict(id=hypothesis["id"], directory=str(folder.relative_to(directory)),
                          reference_status=row["reference_status"], source=row["source"]))
    record["reference_hypotheses"] = links
    record["reference_links"] = links
    write_json(directory / "report.json", record)
    single = result["reference_hypotheses"][0] if len(result["reference_hypotheses"]) == 1 else None
    write_reference_html(directory, request["skill"], views, region,
        None if single is None else single.get("reference"), record,
        tool_reference=None if single is None else single.get("bound_reference"))
    return record
