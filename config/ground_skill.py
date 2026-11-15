"""File adapter for the public complete-skill grounding pipeline.

The library owns perception, instance routing, numerical transfer and complete
reference hypotheses. This command loads explicit saved sensor inputs and
writes their returned evidence. Model settings and relation declarations are
ordinary request data; no skill name selects an algorithm or model allowance.
"""
# BEGIN thin file boundary delegates all computation to production components.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from manipulation.grounding_observations import load_scene
from manipulation.grounding_records import save
from manipulation.skill import load_grounding_skill
from manipulation.skill_grounding import ground_skill as ground_loaded_skill


def ground_skill(skill_directory, scene_directory, output, *, instruction,
                 relations=None, view_ids=None, parameters=None):
    """Read one capture and call the same public API used by programmatic callers.

    parameters supplies explicit region/grasp model settings, current
    relation/model declarations and prerequisite context. view_ids names
    the actual sensor folders; when omitted the capture's calibrated folders
    define their deterministic order. Native simulation files are not opened.
    """
    skill = load_grounding_skill(skill_directory)
    scene_directory, output = Path(scene_directory), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    parameters = dict(parameters or {})
    if view_ids is None:
        view_ids = sorted(folder.name for folder in scene_directory.iterdir()
                          if folder.is_dir() and (folder / "calibration.json").is_file())
    capture = load_scene(scene_directory, view_ids)
    binding_id = "interaction_reference"
    context = {"skill": str(Path(skill_directory).resolve()),
        "observation_source": capture["scene"], "capture_time_s": capture["simulation_time_s"],
        "predecessor": parameters.get("predecessor", {"status": "unspecified"}),
        **parameters.get("context", {})}
    entity_id = parameters.get("current_entity_id", "current_instruction_entity")
    request = dict(parameters, skill=skill, instruction=instruction,
        output_directory=str(output), feedback=capture["feedback"],
        capture_feedback=capture["feedback"], context=context,
        relations={} if relations is None else relations,
        reference_binding_id=binding_id,
        same_state_source=parameters.get("same_state_source",
            "The explicit view records and robot feedback belong to this saved joint sensor capture."),
        bindings={binding_id: dict(skill=skill, views=capture["views"],
            source_owner_id=skill["source"]["owner_id"],
            current_entity_id=entity_id,
            owner_id=parameters.get("owner_id", entity_id),
            owner_source=parameters.get("owner_source", "Current task declaration for the selected reference owner"),
            owner_role=skill["metadata"]["owner_role"])})
    result = ground_loaded_skill(request)
    return save(result, request, output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--view", action="append", dest="view_ids")
    parser.add_argument("--relations", type=Path)
    parser.add_argument("--parameters", type=Path,
                        help="Explicit algorithm settings, relation and prerequisite request data")
    arguments = parser.parse_args()
    relations = None if arguments.relations is None else json.loads(arguments.relations.read_text())
    parameters = None if arguments.parameters is None else json.loads(arguments.parameters.read_text())
    record = ground_skill(arguments.skill, arguments.scene, arguments.output,
        instruction=arguments.instruction, relations=relations, view_ids=arguments.view_ids,
        parameters=parameters)
    print(json.dumps({key: record[key] for key in ("status", "region_status", "reference_status")}), flush=True)
    raise SystemExit(0 if record["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
# END the command retains only input loading, result writing and exit status.
