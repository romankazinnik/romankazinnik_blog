"""Shared contracts: object record schema, frozen feature keys, type colors."""
TYPES = ("slab", "wall", "column", "beam", "joist_ceiling", "mep")
COLORS = {"slab": [0.45, 0.45, 0.45], "wall": [0.1, 0.8, 0.1],
          "column": [0.9, 0.1, 0.1], "beam": [0.1, 0.3, 0.95],
          "joist_ceiling": [0.8, 0.55, 0.15], "mep": [0.15, 0.8, 0.8],
          "unclassified": [0.75, 0.62, 0.1]}

# Frozen, geometric, model-neutral. Both RuleScorer and a future MLScorer consume THIS, in order.
FEATURE_KEYS = ["plane_mse", "density", "fill_ratio", "n_points",
                "extent_l", "extent_w", "extent_h",
                "planarity", "linearity", "vertical_consistency", "z_center_rel"]


def make_record(id, type, phase, source, obb, features, confidence):
    return {"id": id, "type": type, "phase": phase, "source": source,
            "obb": {"center": list(map(float, obb["center"])),
                    "extent": list(map(float, obb["extent"])),
                    "R": [list(map(float, r)) for r in obb["R"]]},
            "features": {k: float(features[k]) for k in FEATURE_KEYS},
            "confidence": float(confidence)}
