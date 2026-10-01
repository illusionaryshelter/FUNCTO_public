# Planning checks and visualization

These scripts exercise FUNCTO's production `manipulation.planning*` modules
against the installed cuRobo V2 and OMPL backends. They include synthetic
robot scenes and independent numerical diagnostics. The scene-specific
fixtures belong here; production modules consume the existing compiled
`program` and grounded `scene` interfaces.

Run scripts directly from this checkout, with the existing `recovery`
environment active. Every script accepts `--help` without starting a GPU
solve. Reports require an explicit `--output`; use a new directory so an
earlier result is not overwritten. No script depends on a dated directory
or a checkout named `FUNCTO_public`.

```bash
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m unittest discover -s tests -p 'test_manipulation_*.py'
python tests/planning/probe_robot_collision.py --help
```

The ordinary CPU suite does not discover these explicit scripts. GPU checks
must run one at a time. Robot scripts cap the Torch allocator at 2 GiB;
standalone scene queries use their smaller declared limits. This cap does
not include every allocation made by native libraries.

| Scripts | Scope |
|---|---|
| `check_recorded_task_inputs.py` | CPU replay of manifest-declared task references and input completeness; preserves candidate ambiguity and independent snapshots |
| `check_recorded_robot_binding.py` | CPU comparison of an explicitly mapped recorded robot and planning URDF, including collision-sphere coverage counterexamples |
| `visualize_recorded_task.py` | Local HTML and PNG presentation of those input audits and task-space references; does not run joint planning or simulation |
| `robot_fixture.py`, `prepare_recorded_task.py` | Explicit native robot geometry, named proprioception, FK checks and current planning-problem preparation |
| `recorded_observation.py`, `recorded_owner_geometry.py`, `detect_recorded_owners.py` | Current robot/owner masks and explicitly bound offline carrier geometry; no current object truth enters planning |
| `run_recorded_skill.py`, `recorded_task_planning.py` | Current bounded joint planning, unchanged skill constraints and classified diagnostic curve export |
| `probe_ompl_constraints.py` | CPU OMPL projection and connection contracts |
| `probe_scalar_progress_clock.py`, `diagnose_moving_gripper_capacity.py`, `probe_collision_suffix_capacity.py` | CPU representation and interpolation diagnostics |
| `probe_fk_gradient.py`, `probe_process_gradient.py` | Native FK and full-process derivative checks |
| `probe_scene_query_gpu.py`, `probe_observation_scene_gpu.py`, `probe_known_solid_fk_gpu.py` | Native collision queries, observed-space mapping and current robot evidence |
| `probe_planning_gpu.py`, `probe_process_planner.py`, `probe_robot_collision.py`, `probe_required_excursion.py`, `probe_robot_projected_seed.py` | Explicit trajectory generation and projected-seed checks |
| `probe_moving_gripper_gpu.py`, `probe_native_contact_permission.py`, `probe_on_endpoint_family.py` | Gripper commands, local contact and permitted support freedom |
| `probe_future_branch_coupling.py`, `probe_ur10e_branch_ik.py`, `probe_ur10e_future_branch.py` | Diagnostic branch fixtures; a completed run may still fail its stated acceptance criterion |

For example, the existing full-robot collision check is run explicitly with:

```bash
python tests/planning/probe_robot_collision.py \
  --knots 12 --output runs/planning/robot-collision
```

All script-to-script imports stay local to this directory. Run the scripts by
file path; they are not a second production planning package. Historical JSON,
trajectory and log artifacts remain in the workspace's stage records and are
not runtime dependencies. In particular, `probe_ur10e_future_branch.py` takes
its independently generated IK audit through `--audit`.

Recorded-task checks take explicit input paths through their CLI or manifest;
the scripts contain no fixed episode or task sequence. Run input audits with
`CUDA_VISIBLE_DEVICES=''`. Their `reference_not_planned` curves are grounding
references, and successful recomputation does not establish candidate selection,
complete collision inputs, or robot execution. The robot audit reads only the
declared robot subtree and proprioception columns; a rigid TCP calibration does
not repair mismatched collision geometry. Use each script's `--help` for its
required paths and mappings.
