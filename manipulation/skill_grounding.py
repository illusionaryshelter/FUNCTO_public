"""Public skill grounding from semantic bindings to complete references.

Instance relation chooses a region algorithm per explicit source/current
binding. The loaded representation's movement relation chooses its existing
reference generator. Task frame F and tool-motion anchor B remain separate.
The output retains every reference hypothesis; planning and execution consume
those results through their own public interfaces.
"""
from __future__ import annotations

from time import perf_counter
import traceback
from types import SimpleNamespace

import numpy as np

from .grounding_observations import prepare
from .representation import generate_reference, trajectory_summary
from .region_transfer import run_region_transfer


def _validate_relations(request):
    """Validate declared coordinate identities before model computation."""
    feedback, relations = request["feedback"], request.get("relations", {})
    for field in ("J", "A"):
        if field in relations and not relations.get(field + "_source"):
            raise ValueError(f"Current {field} requires its explicit source")
    if "A" in relations:
        for field, value in relations.get("A_context", {}).items():
            if field in {"tool_id", "tool_frame", "reference_frame"} and value != feedback.get(field):
                raise ValueError(f"Current A_context.{field} conflicts with robot feedback")


def _reference(skill, F, feedback, relations, anchor, support):
    """Use the existing complete relation and its declared task geometry."""
    if support is not None:
        from .reference_transfer import generate_observed_reference
        return generate_observed_reference(skill["representation"], functional_frame=F,
            J=relations["J"], A=relations["A"], T_world_eef=feedback["eef_pose"], **support)
    parameters = {key: relations[key] for key in ("J", "A", "T_world_body", "motion_model",
        "source_coordinates", "current_coordinates", "contact_motion", "held_until_time") if key in relations}
    return generate_reference(skill["representation"], T_world_function=F,
        tool_reference_anchor=anchor, T_world_eef=feedback.get("eef_pose"), **parameters)


def _publish_reference(record, hypothesis, skill, feedback, relations, support):
    """Publish the nominal operands before binding the measured entrance."""
    from .planning_problem import reference_phase
    from .timed_reference import BoundToolReference

    reference = _reference(skill, hypothesis["T_world_function"], feedback, relations,
                           hypothesis["tool_reference_anchor_world"], support)
    hypothesis["reference"] = reference
    record.update(nominal_reference_status=reference["status"],
        reference_status="unbound" if "eef_path" in reference else "missing",
        reference_missing=dict(reference["missing"]),
        nominal_reference_summaries=reference["summaries"],
        reference_boundary_checks=reference["boundary_checks"],
        reference_boundaries={name: reference[name + "_boundaries"]
                              for name in ("object", "eef") if name + "_boundaries" in reference},
        nominal_reference_scope=reference["scope"], nominal_reference_method=reference.get("method"))
    if "rotation_transfer" in reference:
        record["reference_rotation_transfer"] = reference["rotation_transfer"]
    if "eef_path" not in reference:
        hypothesis["reference_status"] = "missing"
        return
    if hypothesis["T_world_tool"] is None:
        hypothesis["T_world_tool"] = reference["eef_boundaries"]["goals"][0]["pose"].copy()
    if skill["representation"]["kind"] == "grasp" and hypothesis["tool_reference_anchor_world"] is None:
        hypothesis["tool_reference_anchor_world"] = hypothesis["T_world_function"]
    record["stage"] = "bind_reference"
    phase = reference_phase(reference, skill_id=skill["metadata"]["skill_id"],
        source=dict(hypothesis_id=hypothesis["id"], source=hypothesis["source"]))
    source_clock = np.asarray(reference["eef_timestamps"])
    elapsed = source_clock - source_clock[0]
    bound = BoundToolReference(SimpleNamespace(phases=(phase,)), [0., 1.], [0., 1.],
        float(elapsed[-1]), feedback["eef_pose"], feedback["eef_twist"])
    hypothesis.update(bound_reference=bound,
        reference_status="reference" if bound.diagnostics["entrance_twist_bound"] else "unbound")
    poses = bound.sample_numpy(elapsed)
    record.update(reference_binding_status=bound.diagnostics["status"],
        reference_status=hypothesis["reference_status"], reference_binding=bound.diagnostics,
        reference_clock=dict(source_time_origin_s=float(source_clock[0]), duration_s=float(elapsed[-1]),
            progress_u=[0., 1.], progress_s=[0., 1.],
            meaning="source elapsed seconds; planner rebinds complete nominal operands to its request clock"),
        reference_summaries={"eef": trajectory_summary(poses, source_clock,
            approach_time=reference["summaries"]["eef"]["approach_time"],
            interaction_time=reference["summaries"]["eef"]["interaction_time"])},
        reference_scope="current tool pose and twist bound to nominal interaction on source elapsed seconds",
        reference_method="BoundToolReference")


def _ground(request, prepared, record):
    skill, feedback = request["skill"], request["feedback"]
    binding_id = request["reference_binding_id"]
    binding = prepared["bindings"][binding_id]
    record.update(prepared_observations=prepared, views=binding["views"],
                  selection=binding["selection"], strategy=binding["strategy"])
    record["stages"].append("select_current_entity")
    record["stage"] = "region_transfer"
    selection = binding["selection"]
    if binding["strategy"] is None:
        record.update(status="completed", region_reasons=["source_target_relation:" +
                      selection["source_target_relation"]])
        return record
    if binding["strategy"] == "same_instance":
        if any(view["observation"] is not None for view in binding["views"]):
            from transfer.dual_view.pipeline import prepare as prepare_runtime
            record["runtime"] = prepare_runtime(skill["source"]["region_geometry"])
        region = run_region_transfer(skill["source"], binding["views"],
            skill_kind=skill["representation"]["kind"], same_state_source=request["same_state_source"],
            current_region_owner=request.get("current_region_owner"))
        frame = region["reference_frame"]
        F = None if frame is None else frame["T_world_region"]
    else:
        from transfer.cross_instance.pipeline import run_region_transfer as run_cross_instance_region
        # BEGIN the cross-instance algorithm consumes exactly two observations.
        # A selected entity in one image is valid semantic evidence, but it
        # does not supply the second measurement required by this region fit.
        # Preserve the actual view identities and report the missing contract
        # before numerical region preparation or grasp generation is invoked.
        view_ids = [view["id"] for view in binding["views"]]
        if len(view_ids) != 2 or len(set(view_ids)) != 2:
            record.update(status="completed", region_status="missing",
                region_reasons=["cross_instance_requires_two_distinct_observations"],
                observation_contract=dict(required_count=2,
                    provided_view_ids=view_ids, distinct_count=len(set(view_ids))))
            return record
        # END observation identity is never manufactured by copying a view.
        # BEGIN selected reference and declared owner keep separate depth support.
        missing_views = {view["id"]: view["missing_reasons"] for view in binding["views"]
                         if view["observation"] is None or view["feature"] is None
                         or view["owner_observation"] is None}
        if missing_views:
            record.update(status="completed", region_reasons=["missing_cross_instance_observations"],
                          missing_views=missing_views)
            return record
        # END valid reference pixels do not manufacture owner observations.
        region = run_cross_instance_region(skill, binding["views"],
            context=request["context"], **request["region_parameters"])
        frame = region.get("region_frame")
        F = None if frame is None else frame["T_world_region"]
    record.update(region_result=region, region_status=region["status"],
                  region_reasons=region.get("reasons", [region["reason"]] if "reason" in region else []))
    record["stages"].append("region_transfer")
    if F is not None:
        record["functional_frame"] = frame
    if region.get("outputs") is not None:
        record.update(region_geometry=skill["source"]["region_geometry"],
            region_warnings=list(region["outputs"]["result"]["diagnostics"].get("quality_warnings", [])))
    if F is None:
        record.update(status="completed", reference_missing={"functional_frame": "No computed task frame"})
        return record
    record["stage"] = "generate_reference"
    if skill["representation"]["kind"] == "grasp":
        if any(feedback[field] != skill["metadata"][field] for field in ("tool_id", "tool_frame")):
            raise ValueError("Grasp reference needs an explicit source/current tool-frame calibration")
    if binding["strategy"] == "cross_instance" and skill["representation"]["kind"] == "grasp":
        from .cross_instance_grasp import NoGraspCandidates, run_cross_instance_grasp
        record["stage"] = "cross_instance_grasp"
        try:
            grasp = run_cross_instance_grasp(skill, region, binding["views"],
                feedback=request["capture_feedback"], **request["grasp_parameters"])
        except NoGraspCandidates as error:
            # The real model-consumed cloud, steering and seed identities are
            # completed failure evidence and remain available to the writer.
            record["grasp_failure"] = error.diagnostics
            raise
        record["grasp_result"] = grasp
        hypotheses = grasp["hypotheses"]
        record["stages"].append("cross_instance_grasp")
    else:
        # Existing movement relations consume the computed task frame directly.
        # An explicit B is a grasp operand; manipulation keeps H_m/J/A/model.
        hypotheses = [dict(id=binding_id, T_world_function=F, T_world_tool=None,
            tool_reference_anchor_world=None,
            source=dict(binding_id=binding_id, strategy=binding["strategy"]))]
    record["reference_hypotheses"].extend(hypotheses)
    failures = []
    for hypothesis in hypotheses:
        diagnostics = dict(stage="generate_reference")
        try:
            _publish_reference(diagnostics, hypothesis, skill, feedback,
                request.get("relations", {}), request.get("support_geometry"))
            hypothesis["status"] = "completed"
        except Exception as error:
            diagnostics.update(status="failed", failed_stage=diagnostics["stage"],
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
            hypothesis.update(status="failed", reference_status="failed")
            failures.append(diagnostics)
        hypothesis["reference_diagnostics"] = diagnostics
    if len(hypotheses) == 1:
        record.update(hypotheses[0]["reference_diagnostics"])
    else:
        record.update(reference_status="reference_hypotheses",
            reference_binding_status="per_hypothesis",
            nominal_reference_status="per_hypothesis",
            reference_missing={row["id"]: row["reference_diagnostics"].get("reference_missing", {})
                               for row in hypotheses})
    record["stages"].extend(("generate_reference", "bind_reference"))
    if failures:
        record.update(status="failed", stage=failures[0]["failed_stage"],
            failed_stage=failures[0]["failed_stage"], error_type=failures[0]["error_type"],
            error=failures[0]["error"], traceback=failures[0]["traceback"])
    else:
        record.update(status="completed", stage="complete")
    return record


def _run(request, prepared, *, detector=None, client=None, feature_extractor=None):
    skill = request["skill"]
    result = dict(status="running", stage="validate_current_relations", stages=[],
        region_status="missing", reference_status="missing", nominal_reference_status="missing",
        reference_binding_status="missing", reference_hypotheses=[],
        skill_id=skill["metadata"]["skill_id"], context=request["context"],
        current_robot_feedback=request["feedback"], executed=False,
        target_region_truth_used=False, relations=request.get("relations", {}))
    started = perf_counter()
    try:
        _validate_relations(request)
        result["stages"].append(result["stage"])
        if prepared is None:
            result["stage"] = "select_current_entity"
            prepared = prepare(request, detector=detector, client=client,
                feature_extractor=feature_extractor,
                on_stage=lambda name: result.update(stage=name))
        return _ground(request, prepared, result)
    except Exception as error:
        result.update(status="failed", failed_stage=result["stage"],
            error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        return result
    finally:
        result["seconds"] = perf_counter() - started


def ground_skill(request, *, detector=None, client=None, feature_extractor=None):
    """Ground a complete loaded skill using one shared observation stage.

    The request declares bindings, primary ``reference_binding_id``, sensor
    inputs, capture/entrance feedback, source/current relations, prerequisite
    and start context, and the selected algorithms' original parameters.
    Grounding returns all complete references and their actual failure/missing
    evidence. It starts neither planning nor physical execution.
    """
    return _run(request, None, detector=detector, client=client,
                feature_extractor=feature_extractor)


def ground_prepared_skill(request, prepared_observations):
    """Consume an explicit completed observation stage without model reruns.

    This stage API supports saved sensor evidence and deterministic numerical
    comparisons. The supplied object is the same ``prepare`` output, including
    actual selection, instance relation and source provenance for each binding.
    It is not an inferred relation or a substitute target-region label.
    """
    return _run(request, prepared_observations)
