# Planning checks and visualization

<!-- BEGIN resident motion backend and shared online execution geometry. -->
The production entry points are `run_recorded_skill.py` and
`run_recorded_simulation.py`. Scene-specific input preparation belongs in these
adapters. `manipulation` accepts named robot state, compiled task constraints,
explicit geometry and event evidence; it contains no dataset dispatch.

Run with the existing `recovery` environment and `PYTHONNOUSERSITE=1`. GPU and
simulation scripts are explicit commands, excluded from ordinary CPU test
discovery. Run them serially within the caller's host/GPU memory limits. No
installation or environment modification is required.

## Production planning and execution

`run_recorded_skill.py` owns a `MotionBackend` for the request lifetime. Library
callers can retain that backend across requests; native IK, trajectory models
and capacity remain resident. Each request binds its measured q/dq, fixed scene
and task. A feasible supplied joint reference can return directly; otherwise
there is one primary solve and at most one selected recovery (OMPL or numerical
recovery). There is no candidate quality stage or scene-reconstruction loop.

`planning_observation.build_scene_from_rgbd` constructs `OnlinePlanningScene`:
fixed background and owner distance fields, native source-validity bits, observed
free-space evidence, named robot spheres and finite contact rules. Missing
observations remain unknown. The required `--unknown-policy` explicitly names
bounded regions in which unknown space may be attempted and forbids unknown
space elsewhere. This permission cannot erase known occupied geometry.

`--contact-bindings` supplies finite material faces, target patches, their source
and grasp/holding activation mappings. Only the identified contact rules can
activate. The nominal planning interval does not establish actual contact;
execution requires the corresponding arrival/close evidence or a freshly
verified current holding condition. A finite patch cannot disable a complete
hand/object collision pair or erase geometry outside the patch.

`recorded_task_planning.py` saves the backend result once. Accepted curves carry
an execution binding: exact polynomial, original event clock, consumed evidence,
executable scope, registered robot data and the same frozen online scene.
Rejected diagnostic curves cannot be executed. Recording performs no additional
solve, candidate ranking or source-solid validation.

`TrajectoryExecutor` loads that binding and queries the same scene using spheres
placed by FK at actual named joint feedback. Evidence boundaries terminate the
current authorization. A successful boundary returns `needs_rebind`; the next
request binds fresh feedback and scene evidence while preserving original event
identities and consumed dependencies. A reference or planning success is never
a Holding observation.

The runner requires explicit `--finger-targets` and `--execution-limits` shared
with execution. These specify calibrated arm tracking, finger position/speed and
stop requirements without creating another event clock. Robot fixture identity,
base/TCP calibration and native physical limits remain checked.

Use each entry point's `--help` for the complete current interface. In particular,
old source-coverage assumptions, adaptive-refinement flags and whole-hand
permission inputs are no longer production interfaces.

## Diagnostic tools

| Files | Purpose |
|---|---|
| `check_recorded_task_inputs.py`, `prepare_recorded_task.py` | Explicit task references and current problem preparation |
| `robot_fixture.py`, `check_recorded_robot_binding.py` | Named robot, calibration and collision-sphere registration |
| `recorded_observation.py`, `recorded_owner_geometry.py`, `detect_recorded_owners.py` | Calibrated current observations, masks and declared carrier geometry |
| `recorded_simulation.py`, `recorded_skill_evaluation.py` | Physical recording adapter and independent task outcome evaluation |
| `visualize_recorded_task.py`, `visualize_current_recorded_task.py`, `visualize_recorded_failure.py` | Input references, saved trajectories, actual feedback and failures |
| `check_recorded_robot_geometry.py`, `audit_recorded_collision.py`, `robot_coverage_fixture.py`, `refine_robot_fixture.py` | Explicit independent source-geometry audits; never a production fallback |
| `probe_finger_control.py`, `probe_controller_takeover.py`, `probe_mounted_execution.py` | Explicit control and tracking diagnostics |

Independent source audits use Pinocchio/Coal and explicitly supplied native
models. Their reports neither authorize execution nor add source-world checks to
normal planning. `planning_observation.build_audit_scene_from_rgbd` retains the
separate reconstruction API for these diagnostics.

All script-to-script imports remain local to this directory. Historical run
artifacts are evidence, not runtime defaults. Legacy archives lacking the current
complete binding must not be treated as accepted executable trajectories.

Both main runners accept optional `--profile-runtime`. Its calling-thread
`cProfile` output includes profiler overhead, which consumes the same time
budget; it does not independently time asynchronous GPU kernels. Simulation
also accepts explicit `--result-only-goal` / `--diagnostic-goal` declarations.
These change only the declared offline evaluation scope; they do not supply
runtime feedback or waive arrival/event requirements.
<!-- END resident motion backend and shared online execution geometry. -->
