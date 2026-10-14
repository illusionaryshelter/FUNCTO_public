# Independent local-region transfer

This package studies how a known visible local surface region transfers between
RGB-D observations. It does not register or call the production perception,
skill-compilation, planning, or robot-execution interfaces. It reuses the existing
MARCO model and native coordinate operations as standalone feature utilities.

The numerical core accepts source/target visible points, a source region mask,
and explicit correspondence indices. It does not accept target region labels,
container names, joint models, or hidden geometry.

| File | Responsibility |
|---|---|
| `core.py` | Validated Sim(3) registration, normalized anchor-distance scores, full-domain ambiguity checks, and visible planar region recovery |
| `features.py` | Offline MARCO extraction, reciprocal raw-dot anchor matching, and native Gaussian-decoded source-region matches |
| `scenes.py` | Independent MuJoCo fixtures derived from the permanent `scene/original` layout; observation and evaluation records are separate |
| `evaluation.py` | Held-out metrics, approximate oracle correspondences, and image overlays |
| `experiment.py` | Manifest-driven extraction and five-method comparison |
| `test_core.py` | Mathematical, coordinate-convention, data-contract, and target-label isolation regressions |

The supported source region is a visible, approximately planar, convex patch.
The distance signature is invariant under a shared positive uniform scale,
rotation, and translation. Nonuniform shape changes are empirical tests, not a
theoretical guarantee. An accepted output is a region candidate, not a statement
about collision-free placement, physical support, or robot reachability.

## Numerical interface

```python
from transfer.core import prepare_anchors, infer_region

# Arrays use metres. Indices must be integral, in range, and one-to-one.
prepared = prepare_anchors(src_xyz, tgt_xyz, src_idx, tgt_idx, mode="automatic")
global_result = infer_region(src_region_mask, prepared, method="global_sim3")
relation_result = infer_region(src_region_mask, prepared, method="relational")
```

`accepted` and `reasons` are separate from `candidate_center` and
`candidate_mask`. Rejected candidates are retained only for inspection. Invalid
array contracts raise an error; insufficient observed geometry produces an
explicit rejection. Oracle and source-region-only direct matching use separate
preparation modes so they cannot silently reuse whole-object automatic anchors.

## Commands

Run from `FUNCTO_public` using the existing recovery environment. No installation
is required. `--out` directories for observations and evaluation must be fresh.

```bash
PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer.scenes --out /tmp/region-data

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer extract \
  --data /tmp/region-data --out /tmp/region-features

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m transfer evaluate \
  --data /tmp/region-data --features /tmp/region-features --out /tmp/region-results

PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/shelter/miniforge3/envs/recovery/bin/python -m unittest transfer.test_core -v
```

The model worker uses cached offline weights and caps the Torch allocator at
6 GiB. Actual model experiments must additionally supervise the full process
tree and total device use; the [recorded experiment commands](../../sim-version-plans/2026-10-13-local-region-transfer/commands.md) provide that
supervision, including memory outside Torch. Do not run multiple model stages
concurrently.

## Experiment evidence

The supplied [initial proposal](transfer_plan.md) is the research starting point.
Its claims and prototype rules have been audited and corrected in the
[experiment plan](../../sim-version-plans/2026-10-13-local-region-transfer/plan.md),
[mathematical review](../../sim-version-plans/2026-10-13-local-region-transfer/math-review.md),
and [scene review](../../sim-version-plans/2026-10-13-local-region-transfer/scene-review.md).
The experiment uses supplied whole-entity segmentation, real rendered RGB-D and
pretrained MARCO features. Target local labels are evaluation-only. Scene names
occur in fixtures and reporting, never in transfer inference.

The [completed experiment report](../../sim-version-plans/2026-10-13-local-region-transfer/results.md)
contains all six comparisons, candidate acceptance decisions, oracle diagnostics,
resource measurements, and limitations. The three pose variants were accepted;
the three nonuniform shape variants were rejected by automatic whole-entity
methods. This experiment does not establish transfer across different designs.
