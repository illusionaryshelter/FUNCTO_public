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
must run one at a time. Robot probes cap the Torch allocator at 2 GiB;
standalone scene queries use their smaller declared limits. This cap does
not include every allocation made by native libraries.
The recorded skill runner permits an explicit 4 GiB Torch limit and 2 GiB
scene-workspace budget, with native/FK allocations and actual free memory
checked independently. It preserves all required source anchors.

<!-- BEGIN optional runtime measurement and explicit independent scoring roles. -->
`run_recorded_skill.py` and `run_recorded_simulation.py` accept
`--profile-runtime`, disabled by default. It records the calling thread with
Python's `cProfile` and writes `runtime-profile/runtime.pstats` plus
`runtime-profile/runtime.json` under the fresh run directory. The JSON ranks
function self time, cumulative time, call counts, and file self time. Self time
excludes callees and is additive; cumulative rows overlap and must not be
summed. Measurements include profiler overhead, which still consumes existing
wall-time budgets. Asynchronous GPU kernels, child processes and other threads
are not independently timed. The same opt-in wrapper is available as
`manipulation.runtime_profiling.run_with_optional_profile` for other entry points.

The simulation runner also accepts repeated `--diagnostic-goal COLLECTION:INDEX`
declarations. An identity must already be explicitly `--result-only-goal` and
must not carry an execution arrival or subsequent-event obligation. Its original
T/F/U and pose errors remain reported, but it does not veto the original skill
effects. Unselected goals, effects, process requirements, final stability, and
complete evidence remain required. Omitting this option preserves strict goal
grading; result-only execution scope alone does not imply diagnostic grading.
<!-- END neither profiling nor offline scoring supplies runtime observations. -->

| Scripts | Scope |
|---|---|
| `check_recorded_task_inputs.py` | CPU replay of manifest-declared task references and input completeness; preserves candidate ambiguity and independent snapshots |
| `check_recorded_robot_binding.py` | CPU comparison of an explicitly mapped recorded robot and planning URDF, including collision-sphere coverage counterexamples |
| `visualize_recorded_task.py` | Local HTML and PNG presentation of those input audits and task-space references; does not run joint planning or simulation |
| `robot_fixture.py`, `prepare_recorded_task.py` | Explicit native robot geometry, named proprioception, FK checks and current planning-problem preparation |
| `refine_robot_fixture.py` | CPU refinement of explicitly selected complete native convex-cell covers, preserving source geometry and pair policy and revalidating every serialized sphere |
| `check_recorded_robot_geometry.py` | Independent Pinocchio FK and Coal convex/analytic-solid pair distances for complete named recorded states; CPU only |
| `probe_finger_control.py` | Bounded native asset control diagnostic with actual q/dq, actuator/generalized forces, contact identities and explicit convergence thresholds |
| `probe_mounted_execution.py` | Complete mounted-scene control measurement or explicitly supervised diagnostic reference through the production backend/executor; preserves measured entrance velocity |
| `probe_controller_takeover.py` | Short measured-position hold versus unchanged-curve comparison, retaining native torque position/damping/compensation terms |
| `robot_coverage_fixture.py` | Bind the complete native robot solid model and original free-cell evidence to an explicit initial-state assumption |
| `probe_mounted_planning.py` | Bounded native free-reach planning with complete fixed mounts and explicitly synthetic known space; retains accepted curve and effective solver configuration |
| `probe_recorded_robot_mask.py` | One calibrated recorded RGB-D view with the complete known robot assembly, fixed-mount coverage checks and actual OpenGL device metadata |
| `recorded_observation.py`, `recorded_owner_geometry.py`, `detect_recorded_owners.py` | Current robot/owner masks and explicitly bound offline carrier geometry; no current object truth enters planning |
| `run_recorded_skill.py`, `recorded_task_planning.py` | Current bounded joint planning, unchanged skill constraints and classified diagnostic curve export |
| `audit_recorded_collision.py` | CPU replay of a candidate's saved scene-query inputs against independent robot and scene solids, retaining rule sides and uncertain nominal hand geometry without execution authorization |
| `recorded_simulation.py`, `robot_validation_fixture.py`, `recorded_skill_evaluation.py`, `run_recorded_simulation.py` | Recording/model adapters to the production executor and mandatory independent validator, with separate task evaluation and calibrated videos |
| `visualize_current_recorded_task.py` | Current references, classified joint curves, actual robot feedback and physical outcomes; retains historical incomplete attempts |
| `visualize_recorded_failure.py` | Frozen reference, commanded FK and actual TCP, plus joint position/velocity/acceleration and native limits on their recorded clocks |
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

<!-- BEGIN explicit pre-grasp selection for recorded grasp requests. -->
Grasp requests to `run_recorded_skill.py` require `--pregrasp-frame SOURCE_FRAME`.
The preparation script takes repeated `--pregrasp-frame SKILL_ID SOURCE_FRAME`
selections. These must identify evidenced original approach entrances before
contact/closing; neither script infers a dataset-specific index or silently uses
the grasp pose. Missing selections remain reported input failures.
<!-- END explicit pre-grasp selection for recorded grasp requests. -->

<!-- BEGIN shared actual-hand execution contract and independent geometry diagnostics. -->
Recorded planning and execution require the same `--execution-limits` JSON,
validated by `ExecutionLimits` in `manipulation/execution.py`. It supplies complete
ordered joint names, a source, arm/TCP tracking limits, actual finger position
and speed intervals for the initial state and every original command event,
and the velocity threshold for original stops. It contains no duplicate event
clock and has no physical defaults. Position targets remain the separate
explicit `--finger-targets` input to both planning and execution. Planning
optimizes only the arm; a saved named `FingerTrajectory` supplies the nominal
hand at the same formal output time. Actual position intervals remain feedback
guards and no longer create mechanical-travel geometry envelopes.

Each command's position interval must contain its preceding interval until new
feedback is bound. Commands can take effect at the next native step, so a
command event alone cannot justify narrowing the actual-state guard before it is applied.
Narrower bounds require new feedback at the existing evidence barrier. Saved
curves retain the exact limits, and every physical trial requires the same input.
Historical curves without the complete current arm/finger contract remain
diagnostic evidence, not executable inputs. The same finger polynomial is
verified during planning, archived, checked against its named arrays on reload,
and sampled during execution. Original event times and native-tick activation
times are both retained. A position target or reference is never Holding evidence.
These limits must be calibrated; the CPU tests' synthetic numbers and a failed
asset probe are not execution calibration.

<!-- BEGIN explicit new-volume contract for recorded planning. -->
`run_recorded_skill.py` also requires `--initial-state-assumption-source`.
This records the caller's basis for using the native robot solid model at
measured q0 as V0. Its CPU oracle checks `V(q) \ (V0 ∪ F)` at the original
validation states, using the unchanged depth-free grid F. Mount geometry stays
in collision owners and does not enlarge V0. Native-model occupancy is not
claimed as observed free space. The original grid and complete measured
entrance are archived for independent diagnosis.
<!-- END unknown-space acceptance remains separate from collision proxies. -->

Fixture export uses `collision_root` for the articulated sphere scope. The
visual self-filter covers the complete calibrated assembly. Fixed installation
primitives remain in the native model and as static collision owners, with
native pair exclusions scoped only to the corresponding robot query. The
manifest preserves native double-precision `position_limits` as well as velocity
limits. Regenerate older fixtures; they are not silently upgraded.

The independent checker needs the robotics distribution `pin` (Python import
`pinocchio`) and Coal. If an unrelated same-name package shadows this installed
distribution, select the existing distribution only for the launched process:

```bash
python -c 'from importlib.metadata import distribution; from pathlib import Path; import runpy,sys; d=distribution("pin"); roots=[d.locate_file(f).resolve().parent.parent for f in d.files if str(f).endswith("/pinocchio/__init__.py")]; assert len(roots)==1; script=Path(sys.argv[1]).resolve(); sys.path[:0]=[str(roots[0]),str(script.parent)]; sys.argv=sys.argv[1:]; runpy.run_path(str(script),run_name="__main__")' \
  tests/planning/check_recorded_robot_geometry.py --help
```

The checker consumes an explicit `--fixture`, complete `--feedback`, optional
repeated `--saved-state-report`, explicit `--pair-policy` when acceptance-related
pair identities are needed, and a new `--output` JSON path. Raw intersections of
excluded pairs remain in the report. Default distance tolerances are numerical
settings, not proven error bounds or collision exemptions.

Native planner records keep seed controls, rollout q/dq, formal interpolation,
valid sample counts and their distinct dt/knot_dt values. Constructed configs
are saved as JSON plus NPZ tensors; observer errors remain visible. Controls
are not a sampled joint path. Execution adds actual q/dq/TCP, applied controls,
event gates and the first feedback violation. Stopping simulation and preparing
a hold command do not establish physical braking or Holding.

The finger asset probe preserves its input model's contact topology. A hand
welded directly to world can have different parent collision filtering from
the same hand attached to a movable arm. Use its contact/force records to inspect
that distinction; calibrate whole-robot control in its original mounted context.

The production `TrajectoryExecutor` combines `RobosuiteExecutionBackend` with a
mandatory named independent validator. The recording adapter only restores the
source integration state and loads classified curves. Independent preflight
checks actual full robot geometry/FK plus polynomial extrema of the original
arm cubic; every actual step receives the same robot-state checks. This gate
does not certify observed obstacles, unknown space, future finger geometry or
continuous collision between steps. Source-native excluded pairs retain raw
distances; sphere separation certificates do not exclude native entity pairs.

Robot fixtures require an explicit `mechanical_contact_pairs` list (`[]` means
none). Each entry names exactly two native geoms and the robot-model source of
their permitted contact. Native physics remains enabled; independent execution
queries retain the raw distances under the distinct `mechanical_contact` policy.
The same declaration enters planning only when the native link-ignore interface
can represent it exactly: every enabled geom pair between those links must be
explicitly permitted. Partial coverage is rejected. Re-export older fixtures
with an explicit declaration instead of inferring permission from contact flags.

`probe_mounted_execution.py` constructs a new, explicitly identified control
reference from the measured q/dq, supplied arm displacement and move/hold times.
Without `--execution-limits` it measures the backend without inventing a pass
band. With that input it exercises the production executor in
`diagnostic_control_reference` mode, with the same mandatory checks as other
modes. Neither mode is a planned skill or Holding result. This mode cannot load
historical curve archives through `load_saved_trajectory`. Reports retain
native parameters, requested/applied torques, actual robot state, complete
bounded contacts and independent robot geometry results. All paths, source
controller identity, step/contact budgets and reference parameters are explicit.
The `arm_control_saturated_coordinates` counter covers arm torque clipping only;
the probe separately counts actual finger force samples at enabled native force
limits. Touching a force boundary is recorded as an observation, not converted
into a tracking failure or proof of excess unclipped demand.

The same mounted execution probe also accepts `--trajectory` with the original
`--execution-limits`. This mode loads an already accepted archive and executes
its unchanged curve, event clock and evidence barriers; it cannot add arm
displacements, move/hold timing or finger events. The separate mounted planning
probe can produce a small free-reach archive with measured finger coordinates
as rigid planner parameters. Its known box and empty background are explicitly
synthetic. Physical fingers remain actuated and independently supervised;
successful execution of that experiment does not validate the recorded RGB-D
environment or a task grasp. The mask probe performs only depth consistency
against the complete known robot and preserves all unmatched projections as
unknown boundaries; its output alone cannot establish free-space coverage.
<!-- END shared actual-hand execution contract and independent geometry diagnostics. -->

<!-- BEGIN formal archive authorization is required before native simulation. -->
The native simulator requires a formally accepted saved curve and its complete
execution binding. Rejected candidates remain available to the independent CPU
diagnostic; a diagnostic pass does not authorize execution. The restored full scene is used for
simulator setup and independent evaluation. Planning consumes named robot
feedback and RGB-D observations. After setup, execution applies the recorded
controller configuration through actuator commands and native physics steps.
It stops at the first evidence barrier, clamps the final command to that barrier,
and reports any sub-step physical clock overrun separately. It never infers that
issuing a close command establishes a grasp.
<!-- END input restoration and independent evaluation do not bypass acceptance. -->

Current visualization takes explicit run and trial directories. It keeps solver
acceptance, validation completion and measured physics outcome separate; an
input reference or an unexecuted candidate cannot become a successful trial.
MP4 playback uses a constant frame rate, while saved capture times preserve
the exact native simulation times, including the final barrier state.

<!-- BEGIN complete candidate diagnostics remain separate from execution permission. -->
Each retained planning candidate also has a `candidate-XX-scene.json` record.
It owns that candidate's full observed scene, target, curve, nominal hand and
actual verification query inputs, including adjacent rule modes. Failed or
unfinished exports retain their diagnostic classification; absent evidence is
reported explicitly. `audit_recorded_collision.py` accepts this snapshot and
the same robot fixture. Its separate finite query budget and streamed solid
evidence do not change planner acceptance or the executor's preflight budget.
Raw negative distances can belong to permitted contacts or native excluded
pairs, so read the pair policy and violations as well as its distance. Native
query states are retained, while independent solid/rule arithmetic is binary64;
this is not a bit-identical reproduction of the cuRobo ESDF collision cost.
<!-- END rejected candidates never acquire an execution binding from diagnosis. -->
