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
| `features.py` | Offline MARCO extraction, reciprocal raw-dot anchor matching, and native Gaussian-decoded source-region matches |
| `scenes.py` | Independent MuJoCo fixtures derived from the permanent `scene/original` layout; observation and evaluation records are separate |
| `neighborhood_scenes.py` | Repeated internal shelf levels and real `put_mug_on_shelf` evidence, with independent evaluation annotations |
| `evaluation.py` | Held-out surface identity, localization and sampling metrics, approximate oracle correspondences, and image overlays |
| `experiment.py` | Manifest-driven extraction and five-method comparison |
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
