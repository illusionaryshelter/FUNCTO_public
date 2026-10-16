# Independent observed-surface neighborhood transfer

This package transfers the location preference represented by demonstrated
interaction pixels between RGB-D observations, then samples nearby observed
surfaces with an explicit metric budget. It does not register or call the production perception,
skill-compilation, planning, or robot-execution interfaces. It reuses the existing
MARCO model and native coordinate operations as standalone feature utilities.

The numerical core accepts source/target visible points, source evidence pixels,
target pixel coordinates, camera origins, and explicit correspondence indices.
It does not accept target region labels,
container names, joint models, or hidden geometry.

| File | Responsibility |
|---|---|
| `core.py` | Registration-to-anchor preparation, location-preference scores, signed surface orientation, and fixed metric observed-neighborhood sampling |
| `sim3.py` | Shared finite-point validation, rank checks, proper positive-scale Umeyama fitting, and transform application |
| `registration.py` | Explicit RANSAC, fixed-budget PROSAC with simple local optimization, and optional native TEASER++ backends with common residual/refit rules |
| `features.py` | Offline MARCO extraction, native matchers, and shared percentile/interpolation primitives |
| `scenes.py` | Independent MuJoCo fixtures derived from the permanent `scene/original` layout; observation and evaluation records are separate |
| `neighborhood_scenes.py` | Repeated internal shelf levels and real `put_mug_on_shelf` evidence, with independent evaluation annotations |
| `evaluation.py` | Held-out surface identity, localization and sampling metrics, approximate oracle correspondences, and image overlays |
| `experiment.py` | Manifest-driven extraction, five-method comparison, and shared frozen prediction reading/writing |
| `test_core.py` | Mathematical, coordinate-convention, data-contract, and target-label isolation regressions |
| `test_registration.py` | Scalar-baseline equivalence, sampler boundaries, degeneracy, and backend contracts |

The supported source evidence is visible and approximately planar; its mask may
contain holes. Its outline is never copied to the target. Sampling radius and
minimum sample spacing are physical lengths and are never multiplied by the
container's estimated scale. Camera-facing normals and observed pixel adjacency
keep nearby parallel surfaces separate; missing pixels are not filled.

The distance signature is invariant under a shared positive uniform scale,
rotation, and translation. Nonuniform shape changes are empirical tests, not a
theoretical guarantee. Output neighborhoods are proposals on visible surfaces.
Placed-object fit, collision checks, stability, and reachability have separate
input requirements and are outside this module.

## Numerical interface

```python
from transfer.core import prepare_anchors, infer_region

# Arrays use metres. Indices must be integral, in range, and one-to-one.
prepared = prepare_anchors(src_xyz, tgt_xyz, src_idx, tgt_idx, mode="automatic")
observation = dict(
    target_pixels=tgt_pixels,  # integer [N,2], original observed (x,y) coordinates
    source_camera_origin=src_camera_xyz,
    target_camera_origin=tgt_camera_xyz,
    sampling_radius_m=0.03,
    sampling_spacing_m=0.005,
    max_modes=3,
)
global_result = infer_region(src_region_mask, prepared, method="global_sim3", **observation)
relation_result = infer_region(src_region_mask, prepared, method="relational", **observation)
```

<!-- BEGIN explicit interchangeable registration, with no silent fallback. -->
`prepare_anchors` and `transfer_region` accept `registration_algorithm="ransac"`,
`"prosac_lo"`, or `"teaserpp"`. For PROSAC, pass
`correspondence_scores=scores`, a finite vector aligned with `src_idx/tgt_idx`;
larger scores are sampled earlier, and ties retain the original pair order.
The scores are matching priorities, not calibrated inlier probabilities.
`segmented_experiment predict --registration ...` exposes the same choice and
uses the existing MARCO scores.

The default RANSAC keeps its original random triangles, scalar arithmetic,
count/median ordering, thresholds, and final bounded consensus refits. It skips
repeated validation/centering and median calculations for hypotheses whose
inlier count has already lost. PROSAC uses exact integer growth counts and
simple all-inlier local optimization with the same hypothesis budget; it does
not implement the paper's adaptive non-randomness stopping criterion.

TEASER++ requires the native `teaserpp_python` package in the active environment.
Missing software raises `RegistrationBackendUnavailable`; selecting it never
runs a replacement algorithm. Scale estimation is enabled. Native solver
solutions are recorded before the common final refit; every backend's final
inliers use the same target-frame Euclidean threshold. Alternate estimators can
change the result, and are not covered by the default RANSAC equivalence claim.
No solver receives support labels, a known transform, or an object identity.
<!-- END explicit interchangeable registration. -->

`accepted` and `reasons` are separate from `modes`. Each mode contains an observed
seed, an oriented normal, a point-aligned neighborhood mask, and sampled point
indices. Scores are ranking evidence, not calibrated probabilities. Uncertain
surface alternatives remain visible even when the result is rejected. Invalid
array contracts raise an error; insufficient observed geometry produces an
explicit rejection. Oracle and source-region-only direct matching use separate
preparation modes so they cannot silently reuse whole-object automatic anchors.
Acceptance describes localization evidence, not verified object placement.
<!-- BEGIN advisory correspondence ratio semantics. -->
The final correspondence inlier ratio below `min_inlier_ratio` (default 0.2)
adds `low_correspondence_inlier_ratio` to `diagnostics.quality_warnings` rather
than rejecting a candidate. The list is present on every prepared/result
diagnostic record, including failures. `accepted=True` with this warning means
the candidate passed the remaining checks with weak proportional matching
support; it does not establish accurate localization. Too few absolute inliers
(default 6), invalid registration, and every region/surface check still reject.
Warnings can also coexist with a rejected result. The Euclidean distance
threshold, solver, refitting, selected anchors, and candidate geometry are
unaffected. In particular, a wrong position on the same visible plane can pass
all surface checks; this policy intentionally exposes more uncertain candidates.
<!-- END advisory correspondence ratio semantics. -->
Secondary modes can include non-support edges; inspect their explicit geometry
and evidence instead of treating every returned mode as a valid support patch.

The five methods are experimental alternatives. The current controlled results
favor the simpler `direct_local` route (native source-region queries, local
geometric fitting, and the shared sampler); they show no unique layer-selection
benefit from the whole-object relation score. This does not establish general
robustness to occlusion, incorrect segmentation, or unseen assets.

## Commands

Run from `FUNCTO_public` using the existing recovery environment. No installation
is required. `--out` directories for observations and evaluation must be fresh.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer.neighborhood_scenes --out /tmp/region-data

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /home/shelter/miniforge3/envs/recovery/bin/python \
  ../sim-version-plans/2026-10-05-witness-feasible-revision/run_recorded_bounded.py \
  --record-prefix /tmp/region-marco --max-process-rss-mib 8192 --max-device-used-mib 6144 -- \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer extract \
  --data /tmp/region-data --out /tmp/region-features

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer evaluate \
  --data /tmp/region-data --features /tmp/region-features --out /tmp/region-results

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m unittest transfer.test_core -v
```

The scene-generation command above builds the 15-pair experiment from existing
frozen original fixtures and the saved task demonstration; these assets must
already be present. See the
[neighborhood protocol](../../sim-version-plans/2026-10-13-local-region-transfer/neighborhood-plan.md)
and [recorded commands](../../sim-version-plans/2026-10-13-local-region-transfer/neighborhood-commands.md).
Evaluation expects the new unified physical-surface
annotations; old frozen result files remain historical evidence.

The model worker uses cached offline weights and caps the Torch allocator at
6 GiB. Actual model experiments must additionally supervise the full process
tree and total device use; the [recorded experiment commands](../../sim-version-plans/2026-10-13-local-region-transfer/neighborhood-commands.md) provide that
supervision, including memory outside Torch. Do not run multiple model stages
concurrently.

## Experiment evidence

The current [15-pair report](../../sim-version-plans/2026-10-13-local-region-transfer/neighborhood-results.md)
includes repeated internal shelf levels, recorded `put_mug_on_shelf` evidence,
the original six cases, fixed parameters, independent reviews and failure
analysis. The [interactive viewer](../../sim-version-plans/2026-10-13-local-region-transfer/results-neighborhood-001/viewer.html)
shows frozen images, proposals, acceptance states and held-out measurements.

The supplied [initial proposal](transfer_plan.md) is the research starting point.
Its claims and prototype rules have been audited and corrected in the
[experiment plan](../../sim-version-plans/2026-10-13-local-region-transfer/plan.md),
[mathematical review](../../sim-version-plans/2026-10-13-local-region-transfer/math-review.md),
and [scene review](../../sim-version-plans/2026-10-13-local-region-transfer/scene-review.md).
The experiment uses supplied whole-entity segmentation, real rendered RGB-D and
pretrained MARCO features. Target local labels are evaluation-only. Scene names
occur in fixtures and reporting, never in transfer inference.

The [first experiment report](../../sim-version-plans/2026-10-13-local-region-transfer/results.md)
preserves the six original comparisons and their limitations. Its source-outline
reconstruction contract has been removed following the revised task definition:
demonstration evidence, a placed object's footprint, and admissible placement
centers are distinct quantities. The current experiment uses one shared physical
sampling budget across all fixtures and real demonstration inputs.

<!-- BEGIN independent multi-point joint region transfer interface. -->
## Independent joint region path

`joint_features.source_queries` selects up to 32 demonstration observations and
32 additional observations strictly inside `entity_member & ~region_mask` from
one source RGB-D. `appearance_evidence` preserves four separated MARCO response
peaks plus the native decoder's observed pixel. Response percentiles use the
fixed allowed entity feature-cell domain before interpolation to depth points.
They are ordinal evidence, not calibrated correctness probabilities.

`joint_region.infer_joint_region` receives the source/target observation records,
these appearance arrays and the query provenance mask. New proposals use
six- and twelve-point distance-consistent sets. An existing native-region
consensus transform may enter the same candidate pool; it is never a fallback.
Each proposal has one partial injective linear assignment, one fixed-support
Sim(3) refit, and the existing minimum-six, non-collinearity and scale checks.
<!-- BEGIN exact hard-gate score evaluation and workload accounting. -->
Every admitted hypothesis considers every target entity point. Points passing
the unchanged hard geometric gates receive the complete perimeter-normalized
triangle score and fixed missing-node costs; other entries are infinity.
Original center/edge blocks and reduction layout preserve floating-point ties.
Proposal distance compatibility is computed only for rows actually accessed.
Search remains exact over the same finite set of valid candidates.
Diagnostics separate `full_domain_candidate_pairs` from the actual
`graph_score_rows_evaluated`, including duplicate rows needed to preserve the
original singleton reduction layout. The
[equivalence and timing report](../../sim-version-plans/2026-10-15-joint-lossless-speedup/report.md)
records the full prediction comparisons and repeated CPU measurements.
<!-- END no proposal budget, score, gate or sampling rule changes. -->

The old oriented-plane, center-projection-distance and observed pixel topology
requirements govern eligibility. Sampling remains on the frozen seed plane
with an unscaled physical radius and spacing. The first valid candidate in
global score order is the unique primary. The best valid competing surface is
retained as a diagnostic; ambiguity adds a warning. `select_observation` chooses
the smallest common score across independently solved target views sharing
exactly the same source query set. No view name grants a score bonus.

```python
from transfer.joint_features import source_queries, appearance_evidence
from transfer.joint_region import infer_joint_region, select_observation

query_ids, query_region = source_queries(source)
evidence = appearance_evidence(
    source, target, source_feature, target_feature, source_crop, target_crop,
    query_indices=query_ids,
)
result = infer_joint_region(source, target, evidence, query_region=query_region,
                           sampling_radius_m=0.03, sampling_spacing_m=0.005)
```

`python -m transfer.joint_experiment --index INDEX.json --output RESULTS`
freezes the unchanged old RANSAC/PROSAC/TEASER++ comparisons and the new path.
Its process-local read guard rejects evaluation-only artifacts. The index
contains `cases` with `id`, `task`, `view`, and `input_dir`; each input directory
uses the existing segmented experiment's observation/crop/feature contract.
Evaluation is a separate process after all declared predictions finish.
The [experiment plan](../../sim-version-plans/2026-10-15-independent-joint-transfer/plan.md)
defines the fixed comparison matrix and resource limits.
The [image viewer](../../sim-version-plans/2026-10-15-independent-joint-transfer/index.html)
retains every outcome, including missing observations and regressions. The
[frozen failure analysis](../../sim-version-plans/2026-10-15-independent-joint-transfer/failure-analysis.md)
documents why the current score is not a guarantee of better true localization
or correct cross-view ordering; this independent path does not replace the
existing default method.
<!-- END independent multi-point joint region transfer interface. -->

<!-- BEGIN reconstructed two-view observed-surface interface. -->
## Joint surface from two target views

This route reconstructs the complete valid depth of two calibrated target
images before forming correspondences. It is separate from the single-view
`joint_region` method above. No production robot interface imports this path.

| Module | Input and output |
|---|---|
| `multiview_geometry.py` | Two full depth images and K/T camera calibration → actual surface elements, original observation provenance, oriented local planes, and original pixel adjacency |
| `multiview_features.py` | Frozen per-view multi-peak appearance records → allowed source-query/element edges and separate per-camera costs |
| `multiview_grounding.py` | Joint surface and candidate edges → finite transform hypotheses, partial injective assignments, observed regions, or explicit ambiguity |
| `multiview_experiment.py` | Explicit paired case index and existing caches → frozen predictions; separate evaluation command |
| `multiview_visualization.py`, `multiview_web.py` | Frozen predictions and original RGB → local HTML with observed regions, actual samples, states and historical comparisons |

An element contains at most one actual pixel measurement from each camera.
Cross-view pairing requires mutual nearest measurements, agreeing oriented
planes and bidirectional full-depth footprint support. The representative is
one actual measurement; it is never an average. `member_views` records the
cameras supplying element members. `observed_views` records geometric surface
support from each camera. Unpaired points can still describe the same physical
surface; their discrete redundancy is retained. Entity segmentation labels
existing geometry rather than determining which scene surfaces exist.

Each source query retains its own allowed edges from both cameras. Assignment
columns are surface-element identities, while appearance has two independent
coordinates. Missing appearance is NaN with an explicit unavailable mask;
unsupported cached placeholders cannot supply edges or costs. A camera cost is
read from that element's unique original member, or from one deterministic,
geometrically supported projected original pixel. There is no cross-view cost
mean or favorable-neighbor search.

Growth and new-fit budgets are each 256, split 85/85/86 among the two individual
edge pools and their union. At most 32 hypotheses enter assignment, including
at most one supplied native transform per camera. For a fixed proposal, the
assignment first maximizes allowed support inside its fixed residual tolerance,
then minimizes normalized residual. It fits that support once and removes
correspondences that exceed the same tolerance afterward. The tolerance is
0.075 times source-entity RMS radius times the initial proposal's scale; scale
must lie in [0.2, 5]. These are frozen engineering assumptions, not uncertainty
estimates from the target image.

Each admitted hypothesis produces one sampleable observed region. Coverage,
fit, triangle geometry, fixed non-fitting source-point checks and separate
appearance costs form a vector of evidence. Appearance uses a common query
domain fixed across the admitted hypothesis pool for each camera. Empty camera
domains are omitted with a warning. A candidate dominates another only when it
is no worse on every retained coordinate and better on at least one. All
non-dominated regions must agree pairwise at the physical sampling resolution
before a primary can be returned; otherwise the result is rejected as
ambiguous and its regions remain diagnostic. This does not alter the existing
single-view primary-selection policy.

Sampling uses original-pixel graph connectivity from both images, restricted
to one fixed seed plane, signed normal, entity and metric radius. It adds no
hidden points. Full-depth checks separately report support, front-of-plane
model conflicts and unknown states, including occlusion. Support and conflict
may coexist when the cameras disagree. They are not a calibrated free-space
or placement certificate.

The inherited slab thickness is twice the local observed point spacing.
Distinct same-facing parallel layers closer than that model resolution can
still be associated or connected. Signed normals separate opposite visible
sides, but do not resolve every arbitrarily thin same-facing layer. The finite
proposal pool likewise has no exhaustive-recall guarantee.
The one-region-per-hypothesis policy keeps the first sampleable center in
geometric-score order, using stable element order for exact score ties. It can
omit a second, inequivalent region tied within the same hypothesis. Therefore
the final ambiguity decision concerns the retained finite hypothesis/region
pool, not every physically possible region.

```python
from transfer.multiview_geometry import build_target_surface
from transfer.multiview_features import build_correspondence_candidates
from transfer.multiview_grounding import infer_multiview_region

surface = build_target_surface(views, sampling_radius_m=0.03)
candidates = build_correspondence_candidates(
    source, views, surface, query_indices=query_ids, query_region=query_region,
)
result = infer_multiview_region(
    source, surface, candidates, initial_transforms=native_proposals,
    sampling_radius_m=0.03, sampling_spacing_m=0.005,
)
```

Each view supplies `id`, full `depth_m`, `K`, `T_world_camera_cv`, and optional
`observation`/cached `evidence`. Missing entity or appearance records do not
remove depth. Surface masks, seeds and samples index `surface['xyz']`;
assignment columns index `candidates['candidate_indices']`. Both index domains
are saved explicitly. Commands are `python -m transfer.multiview_experiment
predict --index INDEX --cache-root CACHE --output RESULTS`, then the separate
`evaluate --output RESULTS`, followed by `python -m transfer.multiview_web
--results RESULTS --output VIEWER.html`. Native initial proposals reuse the
explicit `--native-backend teaserpp` (default) or `ransac` frozen result; absence
does not switch algorithms.

The previous fixed-mean/single-peak fusion API and its old experiment schema
have been removed. Historical RGB, predictions and HTML remain immutable.
`multiview_registration_cases.py` is a historical three-case fixture preparer;
`multiview_grounding_validation/` is an independent synthetic study. Neither
is the new general input interface. Historical scripts that import the retired
API require their corresponding earlier code revision for exact replay.
<!-- END reconstructed two-view observed-surface interface. -->
