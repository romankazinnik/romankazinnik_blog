# Structural Element Extraction (Phase 1 geometry + Phase 2 PTv3, additive)

Identify structural elements (slabs, walls, columns, beams) in an architectural point cloud.
Two additive phases share one object schema and one evaluator. Coverage only grows.

## Files
- `schema.py`      object record + frozen feature keys + type colors
- `geometry.py`    load/canonicalize, the ONE feature extractor, OBB utils
- `scorer.py`      pluggable confidence: `RuleScorer` now, `MLScorer` later (same feature vector)
- `phases.py`      `phase1_geometry` (rule-based) + `phase2_classify` (PTv3, falls back to geometric)
- `run_pipeline.py`        driver: load -> phase1 -> phase2 -> write plys/jsons -> bucket remainder
- `extraction_evaluation.py`  single source of truth: re-assign, RE-EXTRACT features, RE-SCORE, report

## Install
    uv pip install numpy open3d opencv-python scipy scikit-learn pye57
    # Phase 2 real PTv3 only: torch + Pointcept on the repo PYTHONPATH (see your working setup)

## Run

Self-test (no E57, no GPU; uses a synthetic scene):
    python run_pipeline.py --out out

Your scan (Phase 2 falls back to a relaxed geometric pass if no PTv3 weights given):
    python run_pipeline.py --e57 /path/808_Brannan_LB_E57wP-002-001.e57 --out out

With real PTv3 in Phase 2 (on the L4 box, Pointcept importable):
    PYTHONPATH=/path/Pointcept python run_pipeline.py \
        --e57 /path/scan.e57 --out out \
        --config /path/Pointcept/configs/s3dis/semseg-pt-v3m1-1-rpe.py \
        --weights /path/model_best.pth --device cuda:0

Evaluate (original cloud + BOTH phase jsons -> additive coverage):
    python extraction_evaluation.py out/working_cloud.ply \
        out/boxes_phase1.json out/boxes_phase2.json --out out/report --tau 0.4

## Outputs (in --out)
- `working_cloud.ply`            downsampled cloud used for detection (= evaluator input)
- `boxes_phase{1,2}.json`        object records: id,type,obb,FEATURES,confidence,source
- `boxes_phase{1,2}.ply`         colored OBB meshes (per-type color)
- `classified_phase{1,2}.ply`    claimed points, colored by type
- `remaining_phase1.ply` / `remaining_final.ply`   unclaimed points (shrinks each phase)
- `remaining_summary.json`       remainder count bucketed by height (floor/mid/ceiling)
- `report.json` / `report.txt`   coverage %, per-type counts, volumes, point counts, per-box confidence

## Generalizing to other clouds
Resolution and slab/soffit band widths are now derived from the data (point spacing + Z-histogram
FWHM), so the pipeline adapts to scan density and ceiling height automatically. Remaining constants
are documented in `ASSUMPTIONS.md` (Tier A derived / Tier B physical / Tier C hand-tuned). Assumes
the cloud is in METRES and one storey per file.

## Key design points
- `confidence` is a GEOMETRIC PLAUSIBILITY score in [0,1], NOT a ground-truth probability.
- Swap the scorer: pass any object with the same `feature_keys`; features are stored in the
  JSON so you can re-score offline or train an `MLScorer` from the exact same vectors.
- Phase 2 infers PTv3 on the FULL cloud (context) but CLAIMS only remaining points, and accepts
  a detection only if it also passes the geometric scorer gate (domain-gap firewall).
- The remainder is expected (furniture/MEP/clutter). Characterize it (buckets), don't classify to zero.

## Tuning dials (physical, stable - not eps-fragile)
- `--tau` accept/gate threshold (0.4 default)
- `scorer.py` per-type bands (extent thinness, vertical_consistency, planarity)
- `geometry.vertical_consistency_map` nbands / vote fraction (column vs clutter)
