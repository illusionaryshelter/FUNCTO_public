# Native planning checks

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
