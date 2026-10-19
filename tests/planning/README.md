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
and task. Every request performs one native optimization using its nominal
task/reference costs. A model-collision failure may use one bounded OMPL
connection and one further native solve. Task error is reported as quality;
it is not proof that an attempt must be rejected or that a skill will succeed.

`planning_observation.build_scene_from_rgbd` constructs `OnlinePlanningScene`:
fixed background and owner ESDF fields from the installed Mapper, observed
free-space metadata, named robot spheres and local interaction permissions.
Missing observations remain unknown diagnostic evidence. They do not become
known obstacles or evidence of free space. Known model collisions remain checked.

`--contact-bindings` selects named robot/owner sphere groups, target owners and
interaction intervals, with current grasp/holding identities. Only those pairs
receive contact permission. A nominal interval does not establish actual contact
or successful Holding. The executor retains fresh attachment/effect evidence and
original command identity; target-arrival error cannot veto close or continued
execution.

`recorded_task_planning.py` saves the backend result once. Accepted curves carry
an execution binding: exact polynomial, original event clock, consumed evidence,
executable scope, registered robot data and the same frozen online scene.
Rejected diagnostic curves cannot be executed. Recording performs no additional
solve, candidate ranking or source-solid validation.

`TrajectoryExecutor` loads that binding and checks measured robot limits and
its bound robot model. It does not query the frozen observed-world fields.
Evidence boundaries terminate the
current authorization. A successful boundary returns `needs_rebind`; the next
request binds fresh feedback and scene evidence while preserving original event
identities and consumed dependencies. A reference or planning success is never
a Holding observation.

The runner requires explicit `--finger-targets` and `--execution-limits` shared
with execution. These specify finger position/speed and explicit stop requirements
without creating another event clock. Ordinary arm/TCP tracking error is quality
diagnostic data. Robot fixture identity,
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
normal planning. Production mapping does not construct a parallel source-solid
world or confirm every optimization step against one.

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
