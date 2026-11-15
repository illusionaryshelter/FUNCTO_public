"""Prepare calibrated observations and joint semantic entity bindings.

The source is an already loaded complete skill. Current images, optical Z
depth in metres and CV camera calibration are explicit sensor inputs. SAM
uses the source's stored reference phrase. A single VLM request selects the
current candidates and reports both identity relations. MARCO is evaluated
only for uniquely selected, associated observations used by a region strategy.
"""
from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from .instance_relation import region_strategy, select_bindings
from .skill import prepare_interaction_candidates
from transfer.features import write_json
from transfer.segmentation import prepare_mask_observation, prepare_selected_observation, store_detections


def load_scene(scene_directory, view_ids):
    """Read a saved sensor capture without opening native simulation state.

    Each explicit view folder supplies RGB, optical depth and calibration.
    Layout is acquisition metadata; only its declared capture time is used.
    Robot feedback retains its own named joint/pose/time fields unchanged.
    """
    directory = Path(scene_directory)
    feedback = json.loads((directory / "robot_feedback.json").read_text())
    layout = json.loads((directory / "layout.json").read_text())
    views = []
    for identity in view_ids:
        folder = directory / identity
        calibration = json.loads((folder / "calibration.json").read_text())
        with Image.open(folder / "rgb.png") as opened:
            image = opened.convert("RGB")
        views.append(dict(id=identity, image=image,
            depth_m=np.load(folder / "depth_m.npy", allow_pickle=False),
            K=np.asarray(calibration["K"]),
            T_world_camera_cv=np.asarray(calibration["T_world_camera_cv"]),
            timestamp_s=calibration["timestamp_s"]))
    return dict(views=views, feedback=feedback, acquisition=layout,
                scene=str(directory.resolve()), simulation_time_s=layout["sim_time_s"])


def prepare(request, *, detector=None, client=None, feature_extractor=None, on_stage=None):
    """Prepare explicit bindings once and retain every missing-input state.

    ``request['bindings']`` maps program-given binding IDs to ``skill``,
    ``views``, ``source_owner_id``, ``current_entity_id`` and ``owner_role``.
    The views are ordered raw sensor records. An optional output directory
    records real detections before the VLM call and stores exact FP32 feature
    values in uncompressed NPZ files. It does not build any region surface;
    the selected numerical strategy owns that representation.
    """
    if detector is None:
        from local_models.sam31 import detect
        detector = detect
    if client is None:
        from local_models.gemini import Gemini
        client = Gemini()
    if feature_extractor is None:
        from local_models.marco import extract_image
        feature_extractor = extract_image
    output = request.get("output_directory")
    output = None if output is None else Path(output)
    binding_inputs, prepared_bindings = {}, {}
    started = perf_counter()
    if on_stage is not None:
        on_stage("select_current_entity")
    for binding_id, binding in request["bindings"].items():
        views = binding["views"]
        ordered = iter(views)
        # One real detector call per explicit current view. The writer records
        # the exact returned masks before a semantic transport can fail.
        def recorded_detection(image, labels):
            view = next(ordered)
            detected = detector(image, labels)
            if output is not None:
                folder = output if len(request["bindings"]) == 1 else output / binding_id
                folder.mkdir(parents=True, exist_ok=True)
                store_detections(folder / (view["id"] + "-detections"),
                                 np.asarray(image), labels[0], detected)
            return detected
        prepared = prepare_interaction_candidates(binding["skill"], request["instruction"],
            {view["id"]: {key: view[key] for key in ("image", "selected_owner") if key in view}
             for view in views}, detector=recorded_detection)
        binding_inputs[binding_id] = dict(source=prepared["source_evidence"],
            views=prepared["selection_views"], selection_scope=prepared["selection_scope"],
            **{key: binding[key] for key in ("source_owner_id", "current_entity_id", "owner_role")})
        prepared_bindings[binding_id] = dict(
            sam_queries=prepared["sam_queries"], selection_scope=prepared["selection_scope"],
            detections=prepared["detections"])
    semantic = select_bindings(binding_inputs, request["instruction"], client=client)
    selection_seconds = perf_counter() - started
    if output is not None:
        # Masks remain in their detector archive rather than in JSON.
        saved = {"bindings": {identity: {**{key: value for key, value in row.items() if key != "views"},
            "views": {view_id: {key: value for key, value in view.items() if key != "regions"}
                      for view_id, view in row["views"].items()}}
            for identity, row in semantic["bindings"].items()},
            "vlm_used": semantic["vlm_used"], "raw_response": semantic["raw_response"]}
        write_json(output / "selection.json", saved)
    if on_stage is not None:
        on_stage("prepare_calibrated_observations_and_features")
    for binding_id, binding in request["bindings"].items():
        owner_id, owner_source = binding["owner_id"], binding["owner_source"]
        row = semantic["bindings"][binding_id]
        views = []
        associated = row["same_entity"] is True
        strategy = region_strategy(row["source_target_relation"])
        for raw in binding["views"]:
            view = dict(raw, entity_id=binding["current_entity_id"], observation=None,
                        feature=None, crop_xyxy=None, missing_reasons=[])
            selected = row["views"][view["id"]]
            if strategy is None:
                view["missing_reasons"] = ["source_target_relation:" + row["source_target_relation"]]
            elif not associated:
                view["missing_reasons"] = ["cross_view_entity_identity_unconfirmed"]
            elif selected["status"] != "selected" or len(selected["regions"]) != 1:
                view["missing_reasons"] = ["entity_selection:" + selected["status"]]
            else:
                observation = prepare_selected_observation(np.asarray(view["image"]), view["depth_m"],
                    view["K"], view["T_world_camera_cv"], selected["regions"][0])
                view.update(observation=observation["observation"],
                    crop_xyxy=observation["crop_xyxy"], missing_reasons=observation["reasons"])
                if observation["prepared"]:
                    if "selected_owner" in raw:
                        owner = raw["selected_owner"]
                        if owner["owner_id"] != owner_id:
                            raise ValueError("selected_owner and declared geometry owner IDs differ")
                        owner_prepared = prepare_mask_observation(np.asarray(view["image"]), view["depth_m"],
                            view["K"], view["T_world_camera_cv"], owner["mask"], source=owner["source"])
                        view.update(owner_observation=owner_prepared["observation"],
                            owner_mask=np.asarray(owner["mask"]),
                            owner_evidence=dict(owner_id=owner_id, scope="selected_owner_observation",
                                                source=owner["source"]))
                        if not owner_prepared["prepared"]:
                            view["missing_reasons"].extend(owner_prepared["reasons"])
                    else:
                        scope = ("selected_entity_observation" if row["selection_scope"] == "entity"
                                 else "local_material")
                        # A component supplies local material observations of
                        # its explicitly declared owner. The scope records that
                        # extent; it does not upgrade the mask to a whole owner.
                        view.update(owner_observation=view["observation"],
                            owner_mask=view["observation"]["object_mask"],
                            owner_evidence=dict(owner_id=owner_id, scope=scope, source=owner_source))
                    # The caller declares the reference material's body owner;
                    # its observed local surface does not imply a complete mask.
                    view["owner_evidence"].update(reference_material_owner_id=owner_id,
                                                  reference_material_source=owner_source)
                    view["background_mask"] = (
                        np.asarray(raw["background_mask"]) if "background_mask" in raw
                        else np.zeros_like(view["owner_mask"]) if view["owner_evidence"]["scope"] == "local_material"
                        else ~view["owner_mask"])
                    view["semantic_selection"] = selected
                    view["feature"] = feature_extractor(Image.fromarray(observation["crop_rgb"]))
                    if output is not None:
                        folder = output if len(request["bindings"]) == 1 else output / binding_id
                        np.savez_compressed(folder / (view["id"] + "-observation.npz"), **view["observation"])
                        if view["owner_observation"] is not None:
                            np.savez_compressed(folder / (view["id"] + "-owner-observation.npz"), **view["owner_observation"])
                        write_json(folder / (view["id"] + "-owner-evidence.json"), view["owner_evidence"])
                        np.savez(folder / (view["id"] + "-feature.npz"), feature=view["feature"])
                        Image.fromarray(observation["crop_rgb"]).save(folder / (view["id"] + "-crop.png"))
            views.append(view)
        prepared_bindings[binding_id].update(selection=row, strategy=strategy, views=views)
    return dict(bindings=prepared_bindings, semantic_result=semantic,
                stage_seconds=dict(select_current_entity=selection_seconds,
                                   prepare_observations=perf_counter()-started-selection_seconds))
