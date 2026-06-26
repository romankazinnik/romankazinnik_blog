# Structural Element Extraction — Project Report

**Objective:** From a terrestrial LiDAR scan (E57) of 808 Brannan, automatically extract
structural elements (slabs, walls, columns, beams) as oriented boxes, and **measure quality
without ground truth**. Core design: two **additive** phases over a shrinking "remaining" cloud
+ one **objective evaluator** that is the single source of truth.

## Pipeline (additive components + objective evaluation)
```
  .e57 ─► [ Ingest + Canonicalize ] ─► working cloud (+ floor/ceiling, band widths)
                     │  (full cloud = CONTEXT for both phases)
        ┌────────────┴───────────────────────────────────────┐
        ▼                                                     │ context
  ┌──────────────┐  boxes1, remaining1   ┌────────────────┐   │
  │ PHASE 1      │ ────────────────────► │ PHASE 2        │◄──┘
  │ geometry     │   remaining1 (claim   │ PTv3 + geom    │
  │ (reliable)   │   only unclaimed) ──► │ corroborated   │
  └──────┬───────┘                       └───────┬────────┘
         │ boxes1                                │ boxes2, remaining2 (smaller)
         ▼                                       ▼
  ┌──────────────────────────────────────────────────────────┐
  │ OBJECTIVE EVALUATOR  — re-assign pts, RE-score, gate <τ   │ ◄─ boxes1 + boxes2
  │   → coverage %, per-type counts, volumes, confidence      │
  └──────────────────────────────────────────────────────────┘
         remaining2 ─► bucket by height ─► characterized gap (not forced to a class)
```

## Components
| Component | What it does · in → out | How used · why needed · pros | Cons |
|---|---|---|---|
| **Ingest + Canonicalize** | Merge E57 scans; derive resolution from point spacing; RANSAC floor→Z-up; floor/ceiling + band widths from Z-histogram peaks (FWHM). · in: `.e57` → out: working cloud, levels | Shared front-end for both phases. Data-driven → adapts to scan density, datum, ceiling height. Pros: no hard-coded voxel/levels; generalizes. | Assumes **metres**, **one storey/file**; RANSAC can mis-pick floor if sparse. |
| **Phase 1 — Geometry** | Slabs (Z-peaks), walls (vertical-consistency→Hough→collinear-merge), columns (CC minus walls), beams (soffit Hough). · in: working → out: `boxes_phase1`, classified/remaining `.ply` | Reliable backbone — geometry has **no domain gap**. Additive: claims pts, shrinks remainder. Pros: high-confidence, deterministic, GPU-free. | Hough fragments long walls; misses dense timber **joists**. |
| **Phase 2 — PTv3 corroborated** | PTv3 seg (tiled, **full-cloud context**); claim only remaining pts; instance per structural class (DBSCAN); accept **iff** passes geometric gate. · in: working + remaining1 + boxes1 → out: `boxes_phase2`, smaller remaining | Adds structure geometry missed via learned semantics; gate = **domain-gap firewall**. Same interface → swappable; falls back to relaxed-geometry w/o GPU. Pros: additive, coverage↑. | S3DIS gap on timber loft; DBSCAN fragments walls → over-count; checkpoint/config version skew. |
| **Pluggable Scorer** | Geometric features → confidence [0,1]; fragment/ghost gate. · in: features → out: confidence | One interface: `RuleScorer` now, `MLScorer` later (same vector; features persisted in JSON). Pros: scoring decoupled from detection; retrainable. | Rule bands hand-tuned (Tier-C); absolute-metre thresholds don't fully generalize. |
| **Objective Evaluator** | Union both box sets; assign pts (highest-conf wins); **RE-extract features + RE-score**; gate <τ; report coverage/counts/volumes/conf; flag tilted boxes red. · in: cloud + `[json…]` → out: `report.json/.txt` | Single source of truth → phase metrics reconcile; **needs no ground truth**. Pros: additive coverage, catches ghosts/fragments, honest confidence. | Confidence = geometric **plausibility**, not calibrated P(correct); point-in-OBB overlap heuristic. |
| **Remainder characterization** | Unclaimed pts bucketed by height. · in: remaining → out: `remaining_summary.json` + `.ply` | Reports what is **not** classified (furniture/MEP/joists) vs forcing labels. Pros: honest, bounds the gap. | Descriptive only; joist soffit (largest bucket) stays unclaimed. |

## Results & honest limitations
- **LB scan (primary):** ~83% coverage after fragment-gating; clean slabs/walls/columns.
- **Mezzanine (generalization, no re-tune):** ran unmodified; ~71% coverage — structural backbone held; wall fragmentation + joist gap more severe; meter-based gate under-performed (confirms Tier-C weakness).
- **Known gaps:** (1) PTv3+DBSCAN wall **fragmentation** (fix: apply collinear-merge to Phase 2); (2) **joist soffit** unclaimed (fix: periodicity/ceiling-band detector); (3) scorer self-calibration / `MLScorer`.
