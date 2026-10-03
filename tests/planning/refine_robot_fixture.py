"""Refine explicitly selected native robot covers and revalidate the full model.

Only the existing convex-cell partition and enclosing-sphere algorithm is used.
This CPU model tool consumes known robot geometry, never current object truth,
and cannot create a planned trajectory or authorize physical execution.
"""

# BEGIN explicit complete-volume refinement of an existing robot fixture.
import argparse
from copy import deepcopy
import json
from pathlib import Path
import resource
import time

import mujoco
import yaml

from robot_fixture import (
    _collision_query_contract, _effective_ignored_pairs, _geom_cover,
    validate_fixture_collision_contract,
)


def refine(fixture_path, specification_path, output):
    """Change declared covers only; validate original and generated bindings.

    Unchanged geometry certificates and URDF assets remain explicit references.
    Material bindings depend on sphere ordering, so this export accepts only
    unbound per-sphere materials; callers bind named materials after reload.
    """
    started = time.monotonic()
    fixture = json.loads(fixture_path.read_text())
    original = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())
    specification = json.loads(specification_path.read_text())
    records = {row["geom"]: row for row in fixture["collision_geometries"]}
    if not specification or not set(specification) <= records.keys():
        raise ValueError("Refinement must name existing native collision geometries")
    if any(value is not None for value in fixture["sphere_materials"]):
        raise ValueError("Refinement requires unbound sphere materials; rebind named materials afterward")
    validate_fixture_collision_contract(fixture, original)
    output.mkdir(parents=True, exist_ok=False)
    trial, config = deepcopy(fixture), deepcopy(original)
    rows = {row["geom"]: row for row in trial["collision_geometries"]}
    kinematics = config["robot_cfg"]["kinematics"]
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    if model.nq > 64 or model.ngeom > 64 or model.nbody > 128:
        raise ValueError("Refinement exceeds the bounded CPU robot model capacity")
    changes = []
    for name, spec in specification.items():
        row = rows[name]
        # Replacing a link list is valid only when this geom owns that list.
        # Multiple geom ownership needs an explicit index splice, not truncation.
        if sum(other["link"] == row["link"] for other in rows.values()) != 1:
            raise ValueError("Selected refinement requires one native geometry per link")
        previous = deepcopy(row)
        spheres, generated, _ = _geom_cover(
            model, model.geom(name).id, spec, (output/(name+".npz")).resolve())
        row.update(generated)
        kinematics["collision_spheres"][row["link"]] = spheres
        changes.append({"geometry": name, "before_spheres": previous["sphere_count"],
                        "after_spheres": row["sphere_count"],
                        "before_support_excess_m": previous["maximum_support_plane_excess_m"],
                        "after_support_excess_m": row["maximum_support_plane_excess_m"]})
    contract = _collision_query_contract(
        model, kinematics, mechanical_contact_pairs=fixture["mechanical_contact_pairs"])
    if contract["effective_query_pairs"] != fixture["effective_query_pairs"]:
        raise ValueError("Refinement changed the native collision-pair policy")
    kinematics["self_collision_ignore"] = _effective_ignored_pairs(contract)
    original_kinematics = original["robot_cfg"]["kinematics"]
    if any(kinematics[key] != value for key, value in original_kinematics.items()
           if key != "collision_spheres"):
        raise ValueError("Refinement changed robot kinematics or collision exclusions")
    changed_links = {rows[name]["link"] for name in specification}
    if any(balls != original_kinematics["collision_spheres"][link]
           for link, balls in kinematics["collision_spheres"].items() if link not in changed_links):
        raise ValueError("An unselected sphere cover changed")
    trial.update(contract)
    trial["sphere_count"] = sum(map(len, kinematics["collision_spheres"].values()))
    trial["sphere_materials"] = [None]*trial["sphere_count"]
    trial["robot_config_path"] = str((output/"robot.yml").resolve())
    trial.pop("collision_contract_validation", None)
    trial.pop("diagnostic_trial", None)
    trial["sphere_refinement"] = {"source_fixture": str(fixture_path.resolve()),
                                  "specification": specification, "changes": changes,
                                  "scope": "Known robot model only; no scene or trajectory acceptance"}
    Path(trial["robot_config_path"]).write_text(yaml.safe_dump(config, sort_keys=False))
    # Consumer validation rebuilds every cell from both original and exported
    # native geometry and checks the actual serialized float32 sphere lists.
    trial["collision_contract_validation"] = validate_fixture_collision_contract(
        trial, yaml.safe_load(Path(trial["robot_config_path"]).read_text()))
    (output/"manifest.json").write_text(json.dumps(trial, indent=2, allow_nan=False)+"\n")
    report = {"source_fixture": str(fixture_path.resolve()), "sphere_count": trial["sphere_count"],
              "changes": changes, "validation": trial["collision_contract_validation"],
              "elapsed_seconds": time.monotonic()-started,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024}
    (output/"refinement-report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "specification", "output"):
        parser.add_argument("--"+name, type=Path, required=True)
    args = parser.parse_args()
    refine(args.fixture, args.specification, args.output)
# END refinement preserves complete native volume and collision policy.
