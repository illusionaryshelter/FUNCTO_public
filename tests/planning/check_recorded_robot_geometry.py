"""CPU diagnostic of explicit native robot solids and already saved q states.

This adapter reads only the selected robot fixture, complete named feedback and
optional frozen state reports. It never solves IK, reads live scene state, steps
physics, or imports a GPU planner. Pinocchio is selected normally by Python;
the launcher must expose the installed robotics distribution named ``pin``.
"""

# BEGIN bounded replay of named robot states through Pinocchio and Coal.
import argparse
from itertools import combinations
import json
from pathlib import Path
import resource
import sys
import time

import mujoco
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from manipulation.robot_geometry_check import NativePrimitive, NativeSolid, PairQuery, RobotGeometryChecker
from manipulation.tool_geometry import _collision_geometry
from robot_fixture import _native_geom_pose, _validate_fixture_kinematics


def _read(path):
    return json.loads(Path(path).read_text())


def _pose(position, rotation):
    value = np.eye(4)
    value[:3, :3], value[:3, 3] = np.asarray(rotation).reshape(3, 3), position
    return value


def _named(names, values, expected):
    names, values = tuple(names), np.asarray(values)
    if (len(names) != len(set(names)) or set(names) != set(expected)
            or values.shape != (len(names),) or values.dtype.kind not in "iuf"
            or not np.isfinite(values).all()):
        raise ValueError("Saved named coordinates must match the declared model joints exactly")
    return dict(zip(names, values.astype(float)))


def _native_shapes(fixture, model):
    """Tie source vertices AND local transforms to the same compiled model.

    Compiled mesh vertices already contain scale and recentering; the local
    pose is baked once before making a Coal Convex. Fixed boxes/cylinders keep
    their native analytic shape, not a tessellated or assembled convex hull.
    """
    shapes, evidence = [], {}
    for row in fixture["collision_geometries"]:
        name, link = row["geom"], row["link"]
        index = model.geom(name).id
        native = _collision_geometry(model, index, "robot_geometry_diagnostic")
        local = _native_geom_pose(model, index)
        if (model.body(model.geom_bodyid[index]).name != link
                or native["collision_representation"] != row["representation"]
                or not np.array_equal(local, row["T_body_geom"])):
            raise ValueError(f"Native geometry identity or compiled transform changed: {name}")
        with np.load(row["certificate_path"], allow_pickle=False) as artifact:
            for key, value in (("native_vertices", native["vertices"]),
                               ("native_faces", native["faces"]), ("T_body_geom", local)):
                if not np.array_equal(artifact[key], value):
                    raise ValueError(f"Certificate and compiled source geometry disagree: {name}/{key}")
        vertices = native["vertices"] @ local[:3, :3].T + local[:3, 3]
        source = f"{fixture['robot_mjcf_path']}::geom[{name}]; verified {row['certificate_path']}"
        shapes.append(NativeSolid(name, link, vertices, source))
        evidence[name] = {"id": index, "link": link, "vertices_frame": vertices,
                          "vertices_geometry": native["vertices"], "local": local,
                          "kind": native["collision_representation"]}
    for row in fixture.get("fixed_collision_geometries", []):
        name, link = row["geom"], row["link"]
        index, kind = model.geom(name).id, row["shape_kind"]
        expected = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
        if kind not in expected or model.geom_type[index] != expected[kind]:
            raise ValueError(f"Fixed native primitive type changed: {name}")
        size = model.geom_size[index]
        dimensions = 2*size if kind == "box" else np.array([size[0], 2*size[1]])
        local = _native_geom_pose(model, index)
        if (model.body(model.geom_bodyid[index]).name != link
                or not np.array_equal(dimensions, row["dimensions"])
                or not np.array_equal(local, row["T_body_geom"])):
            raise ValueError(f"Fixed native dimensions, frame or pose changed: {name}")
        shapes.append(NativePrimitive(name, link, kind, tuple(dimensions), local,
                                      f"{fixture['robot_mjcf_path']}::geom[{name}]"))
        evidence[name] = {"id": index, "link": link, "local": local, "kind": kind}
    if len(shapes) != len(evidence) or len(shapes) > 32:
        raise ValueError("Diagnostic requires unique native geometry names and at most 32 solids")
    return shapes, evidence


def _states(args, fixture, feedback, config):
    names = fixture["joint_names"]
    measured = _named(feedback["joint_names"], feedback["start_q"], names)
    states = [{"kind": "measured_feedback", "source": str(args.feedback.resolve()),
               "named_q": measured, "reference_tool_pose": feedback["eef_pose"]}]
    for path in args.saved_state_report:
        saved = _read(path)
        if saved["status"] != "exported" or Path(saved["feedback_path"]).resolve() != args.feedback.resolve():
            raise ValueError("Frozen state report must have completed with the explicitly supplied feedback")
        probe = _read(saved["probe_report"])
        active = tuple(probe["joint_names"])
        source_fixture = _read(saved["fixture_path"])
        source_config = yaml.safe_load(Path(source_fixture["robot_config_path"]).read_text())
        locked = source_config["robot_cfg"]["kinematics"]["lock_joints"]
        if (set(active) & set(locked) or set(active) | set(locked) != set(names)
                or locked != config["lock_joints"] or any(measured[name] != value for name, value in locked.items())):
            raise ValueError("Saved active q and explicitly measured locked joints must cover the current model")
        for index, row in enumerate(saved["states"]):
            # The recorded producer explicitly locked these native joints.
            # Completing q therefore uses recorded coordinates, never a guessed
            # symmetric aperture or an interpolation from a saved TCP pose.
            values = _named(active, row["q"], active)
            values.update(locked)
            states.append({"kind": "saved_state", "source": str(path.resolve()),
                           "saved_index": index, "mode": row["mode"], "source_frame": row["source_frame"],
                           "rank": row["rank"], "named_q": values,
                           "locked_joint_source": str(Path(source_fixture["robot_config_path"]).resolve()),
                           "reference_tool_pose": row["tool_pose"]})
    if len(states) > 129:
        raise ValueError("CPU diagnostic admits one measured state and at most 128 explicitly saved states")
    return states


def run(args):
    start = time.monotonic()
    fixture, feedback = _read(args.fixture), _read(args.feedback)
    config = yaml.safe_load(Path(fixture["robot_config_path"]).read_text())["robot_cfg"]["kinematics"]
    model = mujoco.MjModel.from_xml_path(fixture["robot_mjcf_path"])
    if model.nq > 64 or model.nbody > 128 or model.ngeom > 64:
        raise ValueError("Selected robot exceeds this bounded CPU diagnostic's model size")
    _validate_fixture_kinematics(fixture, config, (model,))
    shapes, geometries = _native_shapes(fixture, model)
    checker = RobotGeometryChecker(fixture["urdf_path"], base_frame=fixture["base_link"], geometries=shapes)
    names = tuple(fixture["joint_names"])
    if set(names) != set(checker.joint_names) or len(names) != model.nq:
        raise ValueError("Pinocchio, native model and explicit fixture must cover identical scalar joints")
    addresses = [int(model.joint(name).qposadr[0]) for name in names]
    for name in names:
        if not np.array_equal(checker.joint_limits[name], model.jnt_range[model.joint(name).id]):
            raise ValueError(f"Pinocchio and native joint limits differ: {name}")
    if args.pair_policy:
        pairs = [PairQuery(**row) for row in _read(args.pair_policy)]
        policy_source = str(args.pair_policy.resolve())
    else:
        policy_source = "all distinct supplied native solids; geometric diagnostic, not an acceptance matrix"
        pairs = [PairQuery(first, second, "diagnostic", policy_source)
                 for first, second in combinations(geometries, 2)]
    states = _states(args, fixture, feedback, config)
    data = mujoco.MjData(model)
    maximum_fk = maximum_vertices = maximum_distance = maximum_witness = 0.
    maximum_saved_tcp = 0.
    counts = {"separated": 0, "penetrating": 0, "numerical_boundary": 0}
    results = []
    for state in states:
        positions = [state["named_q"][name] for name in names]
        data.qpos[addresses] = positions
        mujoco.mj_kinematics(model, data)
        base = _pose(data.xpos[model.body(fixture["base_link"]).id], data.xmat[model.body(fixture["base_link"]).id])
        if not np.allclose(base, feedback["base_pose"], atol=1e-12, rtol=0):
            raise ValueError("Explicit feedback base and native fixed-base placement differ")
        result = checker.query(joint_names=names, joint_positions=positions, T_reference_base=feedback["base_pose"],
                               pairs=pairs, frames=[fixture["tcp_site"]])
        fk_errors = {}
        for frame, pose in result["frame_poses"].items():
            if frame == fixture["tcp_site"]:
                index = model.site(frame).id
                native = _pose(data.site_xpos[index], data.site_xmat[index])
            else:
                index = model.body(frame).id
                native = _pose(data.xpos[index], data.xmat[index])
            error = float(np.max(np.abs(np.asarray(pose)-native)))
            fk_errors[frame] = error
            maximum_fk = max(maximum_fk, error)
        vertices_error = 0.
        for name, geometry in geometries.items():
            index = geometry["id"]
            native_pose = _pose(data.geom_xpos[index], data.geom_xmat[index])
            if "vertices_frame" in geometry:
                link_pose = np.asarray(result["frame_poses"][geometry["link"]])
                actual = geometry["vertices_frame"] @ link_pose[:3, :3].T + link_pose[:3, 3]
                expected = geometry["vertices_geometry"] @ native_pose[:3, :3].T + native_pose[:3, 3]
                vertices_error = max(vertices_error, float(np.max(np.linalg.norm(actual-expected, axis=1))))
            else:
                maximum_fk = max(maximum_fk, float(np.max(np.abs(np.asarray(result["geometry_poses"][name])-native_pose))))
        maximum_vertices = max(maximum_vertices, vertices_error)
        for row in result["pairs"]:
            witness = np.empty(6)
            native_distance = float(mujoco.mj_geomDistance(model, data, geometries[row["first"]]["id"],
                                                           geometries[row["second"]]["id"], 100., witness))
            if not np.isfinite(native_distance) or native_distance == 100.:
                raise RuntimeError("Native distance query failed or exceeded its explicit search radius")
            error = abs(row["signed_distance_m"]-native_distance)
            row.update(native_signed_distance_m=native_distance, native_distance_difference_m=error,
                       native_witness_first=witness[:3].tolist(), native_witness_second=witness[3:].tolist())
            maximum_distance = max(maximum_distance, error)
            maximum_witness = max(maximum_witness, row["witness_identity_error_m"])
            counts[row["relation"]] += 1
        saved_tcp_error = float(np.max(np.abs(np.asarray(result["frame_poses"][fixture["tcp_site"]])
                                                - state["reference_tool_pose"])))
        maximum_saved_tcp = max(maximum_saved_tcp, saved_tcp_error)
        results.append({**{key: value for key, value in state.items() if key not in ("named_q", "reference_tool_pose")},
                        "fk_matrix_errors": fk_errors, "maximum_world_vertex_error_m": vertices_error,
                        "saved_tcp_matrix_error": saved_tcp_error, **result})
    if maximum_fk > 1e-10 or maximum_vertices > 1e-10:
        raise ValueError(f"Independent FK or transformed native vertices disagree: {maximum_fk}, {maximum_vertices}")
    report = {"status": "completed", "fixture_path": str(args.fixture.resolve()),
              "feedback_path": str(args.feedback.resolve()), "pair_policy_source": policy_source,
              "geometry_count": len(geometries), "pair_count": len(pairs),
              "saved_state_count": len(states)-1, "state_count": len(states),
              "joint_names": list(names), "numerics": checker.numerics,
              "maximum_fk_matrix_error": maximum_fk, "maximum_world_vertex_error_m": maximum_vertices,
              "maximum_coal_native_distance_difference_m": maximum_distance,
              "maximum_witness_identity_error_m": maximum_witness,
              "maximum_saved_tcp_matrix_error": maximum_saved_tcp, "raw_pair_relation_counts": counts,
              "scope": "native-solid numerical diagnostics; raw overlapping excluded/adjacent pairs are not an acceptance verdict",
              "source_tree_and_geom_arrays_checked": True, "physics_stepped": False,
              "gpu_imported": False, "ik_solver_called": False, "trajectory_planning_called": False,
              "elapsed_seconds": time.monotonic()-start,
              "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "states": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "states"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("fixture", "feedback", "output"):
        parser.add_argument("--"+flag, type=Path, required=True)
    parser.add_argument("--saved-state-report", action="append", type=Path, default=[])
    parser.add_argument("--pair-policy", type=Path)
    run(parser.parse_args())
# END bounded replay of named robot states through Pinocchio and Coal.
