"""Single source of truth. Re-assigns points, RE-EXTRACTS features, RE-SCORES, reports.

Usage: python extraction_evaluation.py cloud.ply boxes_phase1.json [boxes_phase2.json ...]
       [--tau 0.4] [--out report]
"""
import argparse, json
import numpy as np
import open3d as o3d
import geometry as G
from scorer import RuleScorer
from schema import TYPES


def evaluate(cloud_ply, box_jsons, scorer=None, tau=0.4):
    scorer = scorer or RuleScorer()
    pts = np.asarray(o3d.io.read_point_cloud(cloud_ply).points)
    z = pts[:, 2]; fz, cz = float(np.percentile(z, 2)), float(np.percentile(z, 98))

    recs = []
    for jf in box_jsons:
        recs.extend(json.load(open(jf)))

    # assign points; overlap -> highest reported confidence wins (deterministic)
    claimed_by = np.full(len(pts), -1, np.int64)
    order = sorted(range(len(recs)), key=lambda i: -recs[i]["confidence"])
    for ri in order:
        idx = G.points_in_obb(pts, recs[ri]["obb"])
        free = idx[claimed_by[idx] < 0]
        claimed_by[free] = ri

    objects = []
    for ri, r in enumerate(recs):
        idx = np.where(claimed_by == ri)[0]
        feats = G.extract_features(pts[idx], r["obb"], fz, cz)   # RE-EXTRACT from assigned pts
        conf = scorer.score(r["type"], feats)                   # RE-SCORE
        ext = np.maximum(np.asarray(r["obb"]["extent"]), 1e-3)
        objects.append({"id": r["id"], "type": r["type"], "phase": r["phase"],
                        "n_points": int(len(idx)), "volume": float(np.prod(ext)),
                        "density": feats["density"], "plane_mse": feats["plane_mse"],
                        "confidence": conf, "gated_out": conf < tau})

    kept = [o for o in objects if not o["gated_out"]]
    claimed_pts = int((claimed_by >= 0).sum())
    # coverage counts only points claimed by NON-gated objects
    keep_ids = {o["id"] for o in kept}
    keep_ri = {ri for ri, r in enumerate(recs) if r["id"] in keep_ids}
    cov_pts = int(np.isin(claimed_by, list(keep_ri)).sum()) if keep_ri else 0

    report = {
        "total_points": len(pts),
        "classified_points": cov_pts,
        "coverage_percent": round(100 * cov_pts / len(pts), 2),
        "remaining_points": len(pts) - cov_pts,
        "objects_total": len(objects),
        "objects_kept": len(kept),
        "objects_gated_out": len(objects) - len(kept),
        "counts_by_type": {t: sum(o["type"] == t and not o["gated_out"] for o in objects) for t in TYPES},
        "ghost_objects": sum(o["n_points"] < 20 for o in objects),
        "mean_confidence": round(float(np.mean([o["confidence"] for o in kept])), 3) if kept else 0.0,
        "objects": sorted(objects, key=lambda x: -x["n_points"]),
    }
    return report


def write_txt(rep, path):
    with open(path, "w") as f:
        f.write("=== EXTRACTION EVALUATION ===\n")
        for k in ["total_points", "classified_points", "coverage_percent", "remaining_points",
                  "objects_kept", "objects_gated_out", "ghost_objects", "mean_confidence"]:
            f.write(f"{k:22s}: {rep[k]}\n")
        f.write(f"counts_by_type        : {rep['counts_by_type']}\n\n")
        f.write(f"{'id':16s} {'type':8s} {'ph':3s} {'pts':>8s} {'vol':>8s} {'mse':>8s} {'conf':>6s} gated\n")
        for o in rep["objects"]:
            f.write(f"{o['id']:16s} {o['type']:8s} {o['phase']:<3d} {o['n_points']:>8d} "
                    f"{o['volume']:>8.2f} {o['plane_mse']:>8.4f} {o['confidence']:>6.3f} "
                    f"{'Y' if o['gated_out'] else ''}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cloud"); ap.add_argument("boxes", nargs="+")
    ap.add_argument("--tau", type=float, default=0.4); ap.add_argument("--out", default="report")
    a = ap.parse_args()
    rep = evaluate(a.cloud, a.boxes, tau=a.tau)
    json.dump(rep, open(a.out + ".json", "w"), indent=2)
    write_txt(rep, a.out + ".txt")
    print(json.dumps({k: rep[k] for k in ["coverage_percent", "classified_points",
          "remaining_points", "objects_kept", "objects_gated_out", "counts_by_type"]}, indent=2))
    print(f"\nwrote {a.out}.json / {a.out}.txt")


if __name__ == "__main__":
    main()
