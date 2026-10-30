"""Build evaluation identities for the actual compiled simulation model.

The source recording supplies explicit entity membership by element name.
MuJoCo resolves those names in the current model; the collector's existing
catalog API then records current bodies, joints, geometry and sites. These
records describe the simulation being evaluated and belong only to offline
evaluation, never to the planner's observed scene or robot feedback.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import mujoco

from collect.gt import model_catalog


def save_current_catalog(simulation, *, source_catalog_path, scene_path, output_path):
    """Save and return the collector catalog with current native identities.

    ``simulation`` is the already restored MjSim used by the executor. Its
    native model supplies IDs and model metadata; this function neither reads
    a recorded state nor advances the simulation. Source membership survives
    only for named geoms/sites present in that model. Removed elements are
    absent from the current entity, rather than retaining their former IDs.

    This is the same name-resolution procedure already used by
    ``2026-10-24-continuous-skills-resource-test/execute_next.py`` before calling
    ``collect.gt.model_catalog``. No task or skill identity selects a branch.
    """
    source_catalog_path, scene_path = Path(source_catalog_path), Path(scene_path)
    source = json.loads(source_catalog_path.read_text())
    model = simulation.model._model
    kinds = (("geom", "geoms", mujoco.mjtObj.mjOBJ_GEOM),
             ("site", "sites", mujoco.mjtObj.mjOBJ_SITE))
    names = {kind: {row["id"]: row["name"] for row in source[table]}
             for kind, table, _ in kinds}
    instances = {}
    for entity in source["entities"]:
        groups = {}
        for kind, _, native_kind in kinds:
            current_ids = [mujoco.mj_name2id(model, native_kind, names[kind][index])
                           for index in entity[kind + "_ids"] if names[kind][index] is not None]
            groups[kind] = [index for index in current_ids if index >= 0]
        if groups["geom"] or groups["site"]:
            instances[entity["name"]] = groups

    # The collector owns the catalog schema and derives all body relationships,
    # coordinate addresses and element IDs from this current native model.
    catalog = model_catalog(SimpleNamespace(sim=simulation, model=SimpleNamespace(
        instances_to_ids=instances, get_xml=lambda: scene_path.read_text())))
    catalog["source"] = dict(
        current_scene=str(scene_path.resolve()),
        entity_membership_catalog=str(source_catalog_path.resolve()),
        identity_method="Recorded entity geom/site names resolved in the current native model; collector model_catalog supplies current IDs",
        scope="Current simulation model metadata for independent offline evaluation only",
        native_steps_by_catalog_generator=0,
    )
    Path(output_path).write_text(json.dumps(catalog, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    return catalog
