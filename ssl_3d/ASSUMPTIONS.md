# Assumptions & Constants

Every number in the pipeline falls in one of three tiers. Tier A is derived from the data at
runtime (generalizes automatically). Tier B is a true physical constant in metres (generalizes,
stated). Tier C is a hand-tuned guess (the honest weak point; flagged, with how to remove it).

## Tier A — derived from the data (no manual value)
| quantity | how it's derived | where |
|---|---|---|
| working / raster resolution | median nearest-neighbour spacing (sample, density-corrected) x2, clamped [0.02,0.10] m | `geometry.estimate_resolution` |
| floor & ceiling elevation | lowest / highest dominant Z-histogram peaks | `geometry.detect_levels` |
| slab & soffit band thickness | full-width-half-max of each Z peak | `geometry.detect_levels._fwhm_hw` |
| vertical zone (between slabs) | floor_z+floor_hw .. ceil_z-ceil_hw | `geometry.vertical_consistency_map` |
| object features / confidence inputs | computed per object from its points | `geometry.extract_features` |

## Tier B — physical constants (metres; generalize across buildings; stated)
| constant | value | justification |
|---|---|---|
| min wall length | 1.5 m | structural walls are long; rejects fragments |
| min beam length | 2.0 m | beams/girders span bays |
| collinear-merge angle tol | 8 deg | segments within 8 deg are the same member |
| collinear-merge offset tol | 0.4 m | parallel segments <0.4 m apart are one member |
| vertical-consistency bands / vote | 6 bands, >=50% | a real vertical occupies most of the height |
| RANSAC floor acceptance | abs(n_z) >= 0.85 | only a near-horizontal plane is the floor (not a wall) |
| Z-peak prominence / spacing | 0.2*max / 0.3 m | floor & ceiling are dominant, well-separated bands |
| resolution clamp | [0.02, 0.10] m | architectural scan in METRES; guards absurd auto values |

ASSUMPTION (units): the cloud is in **metres**. If it is in mm/feet the clamp and the metre
constants are wrong — convert first, or scale these by the unit factor.
ASSUMPTION (storey): one storey per file (one floor peak, one ceiling peak). Multi-storey needs
detect_levels to return >2 peaks and a per-storey loop (not implemented).

## Tier C — hand-tuned guesses (weak point; candidates to remove)
| guess | value | why it's a guess | how to remove |
|---|---|---|---|
| column footprint gate | area 0.04-1.0 m2, aspect <=3 | encodes column size in metres | score footprint *relative to other verticals in this scene* (percentile) |
| wall thickness band | <=0.5 m | encodes wall thickness | derive from the perpendicular spread of wall inliers |
| scorer per-type bands | see `scorer.py` | hand-set extent/planarity/consistency cutoffs | replace `RuleScorer` with `MLScorer` (same feature vector) once labels exist |
| accept threshold tau | 0.4 | global gate | per-type, or learn from a labelled validation set |
| phase-2 DBSCAN eps | 0.2 m | instance spacing | derive from resolution (e.g. 4*res) |

**Direction of travel:** Tier C is exactly where the self-calibrating scorer and the trained
`MLScorer` plug in — the features are persisted in the box JSON precisely so these can be
re-derived or learned without touching detection. With more time, the column/wall size gates
move to scene-relative percentiles and the scorer becomes distributional, leaving Tier C empty.
