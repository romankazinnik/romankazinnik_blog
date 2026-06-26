"""Pluggable confidence scorer. RuleScorer now; MLScorer later consumes the SAME feature vector.

NOTE: confidence here is a GEOMETRIC PLAUSIBILITY score in [0,1], not a ground-truth probability.
"""
from schema import FEATURE_KEYS


def _ge(x, thr, soft):                       # 1 if x>=thr, ramp below
    return 1.0 if x >= thr else max(0.0, 1 - (thr - x) / (soft + 1e-9))


def _le(x, thr, soft):                       # 1 if x<=thr, ramp above
    return 1.0 if x <= thr else max(0.0, 1 - (x - thr) / (soft + 1e-9))


def _near(x, t, w):                          # 1 at x==t, ramp over w
    return max(0.0, 1 - abs(x - t) / (w + 1e-9))


def _gmean(s):                               # geometric mean: any gross violation tanks score
    s = [max(v, 1e-3) for v in s]
    import numpy as np
    return float(np.exp(np.mean(np.log(s))))


class RuleScorer:
    feature_keys = FEATURE_KEYS

    def __init__(self, strict=False):
        self.strict = strict

    def score(self, type, f):
        min_pts = 300 if self.strict else 20            # strict: drop small spurious boxes
        if f["n_points"] < min_pts or f["fill_ratio"] < 0.005:      # ghost / fragment gate
            return 0.0
        if type == "slab":
            s = [_le(f["planarity"], 0.04, 0.06),                       # flat
                 _le(f["extent_h"], 0.4, 0.3),                          # thin
                 _ge(f["extent_l"] * f["extent_w"], 10, 10),           # large area
                 max(_near(f["z_center_rel"], 0, 0.25), _near(f["z_center_rel"], 1, 0.25))]
        elif type == "wall":
            s = [_le(f["extent_h"], 0.5, 0.4),                          # thin (thickness)
                 _ge(f["extent_l"], 1.5, 1.5),                          # long
                 _ge(f["vertical_consistency"], 0.55, 0.45),
                 _le(f["planarity"], 0.06, 0.08)]                       # planar sheet
        elif type == "column":
            s = [_le(f["extent_w"], 0.9, 0.6),                          # small footprint (mid dim)
                 _le(f["extent_h"], 0.9, 0.6),                          # small footprint (thin dim)
                 _ge(f["vertical_consistency"], 0.55, 0.45)]
        elif type == "beam":
            s = [_le(f["extent_h"], 0.4, 0.4),                          # thin
                 _ge(f["extent_l"], 1.5, 1.5),                          # long
                 _ge(f["z_center_rel"], 0.7, 0.3)]                      # near ceiling
        elif type == "joist_ceiling":
            s = [_ge(f["z_center_rel"], 0.75, 0.25),                    # high up
                 _ge(f["extent_l"] * f["extent_w"], 10, 10),           # large region
                 _le(f["extent_h"], 0.8, 0.5)]                          # a band, not a volume
        elif type == "mep":
            s = [_ge(f["linearity"], 4, 4),                            # tubular run = linear
                 _le(max(f["extent_w"], f["extent_h"]), 0.8, 0.5),     # compact cross-section
                 _ge(f["z_center_rel"], 0.55, 0.35)]                   # in the ceiling zone
        else:
            return 0.0
        return round(_gmean(s), 4)
