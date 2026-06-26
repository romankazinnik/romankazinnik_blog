"""Driver: load+canonicalize -> phase1 -> phase2 -> write all .ply/.json -> bucket remainder.

Usage:
  python run_pipeline.py --e57 /path/scan.e57 --out out
  python run_pipeline.py                       # synthetic self-test (no E57, no GPU)
  add --config ... --weights ... to enable real PTv3 in phase 2 (else relaxed geometric)
"""
import argparse, json, os
import numpy as np
import open3d as o3d
import geometry as G
from scorer import RuleScorer
from schema import COLORS
from phases import phase1_geometry, phase2_classify


def write_ply_cloud(pts, path, colors=None):
    if len(pts) == 0:
        return
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if colors is not None:
        pc.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(path, pc)


def _is_suspect(r, tol_deg=15.0):
    """A clean structural box is axis-aligned in plan and plumb. Flag as suspect (junk)
    if no OBB axis is near-vertical OR none lies near the horizontal plane -> a tilted fragment.
    Slabs are exempt (they are horizontal by construction)."""
    if r["type"] == "slab":
        return False
    R = np.asarray(r["obb"]["R"])                       # columns are the box axes
    c = np.cos(np.radians(90 - tol_deg))                # |axis . z| threshold for "vertical"
    zc = np.abs(R[2, :])                                # vertical component of each axis
    has_vertical = (zc > np.cos(np.radians(tol_deg))).any()      # one axis ~plumb
    has_horizontal = (zc < np.sin(np.radians(tol_deg))).sum() >= 2  # two axes ~in-plane
    return not (has_vertical and has_horizontal)


def write_boxes_ply(recs, path, mark_suspect=True):
    if not recs:
        return
    mesh = o3d.geometry.TriangleMesh()
    for r in recs:
        m = o3d.geometry.TriangleMesh.create_from_oriented_bounding_box(G.obb_to_o3d(r["obb"]))
        color = [1, 0, 0] if (mark_suspect and _is_suspect(r)) else COLORS.get(r["type"], [1, 1, 1])
        m.paint_uniform_color(color); mesh += m
    if len(mesh.vertices):
        mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(path, mesh)


SEMANTIC_ID = {"unclassified": 0, "wall": 1, "floor": 2, "ceiling": 3, "column": 4, "beam": 5,
               "slab": 6, "joist_ceiling": 7, "mep": 8}
ID_NAME = {v: k for k, v in SEMANTIC_ID.items()}
# RGB per semantic id (0..255). Unclassified = black so it recedes in MeshLab.
SEM_RGB = {0: (0, 0, 0), 1: (60, 200, 60), 2: (120, 120, 120),
           3: (190, 190, 190), 4: (220, 40, 40), 5: (40, 80, 240), 6: (150, 110, 60),
           7: (205, 140, 38), 8: (38, 205, 205)}


def _instance_palette(n):
    """n maximally-distinct RGB colors via golden-ratio HSV hue stepping (neighbors differ)."""
    import colorsys
    out = []
    h = 0.0
    for _ in range(max(n, 1)):
        r, g, b = colorsys.hsv_to_rgb(h % 1.0, 0.75, 0.95)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
        h += 0.61803398875
    return out


def _assign(pts, recs):
    """Deduplicated point->object assignment (highest-confidence box wins). Returns
    semantic_id[N] and instance_id[N] (instance 1..len(recs), 0 = unclassified)."""
    n = len(pts)
    sem = np.zeros(n, np.int32)
    inst = np.zeros(n, np.int32)
    order = sorted(range(len(recs)), key=lambda i: -recs[i].get("confidence", 0.0))
    for oid in order:
        idx = G.points_in_obb(pts, recs[oid]["obb"])
        free = idx[inst[idx] == 0]
        inst[free] = oid + 1                              # 1..N
        sem[free] = SEMANTIC_ID.get(recs[oid]["type"], 0)
    return sem, inst


def _write_ply_scalars(path, pts, rgb, sem, inst):
    """Binary PLY: xyz + RGB (uchar) + semantic_id (int) + instance_id (int)."""
    import struct
    n = len(pts)
    xyz = pts.astype(np.float32)
    col = np.asarray(rgb, np.uint8)
    with open(path, "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\n"
                 f"element vertex {n}\n"
                 "property float x\nproperty float y\nproperty float z\n"
                 "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                 "property int semantic_id\nproperty int instance_id\n"
                 "end_header\n").encode("ascii"))
        rec = struct.Struct("<fffBBBii")
        buf = bytearray()
        for i in range(n):
            buf += rec.pack(xyz[i, 0], xyz[i, 1], xyz[i, 2],
                            col[i, 0], col[i, 1], col[i, 2], int(sem[i]), int(inst[i]))
        f.write(buf)


def write_semantic_instance(pts, recs, out_dir, tag):
    """Write the semantic + instance PLYs and the labels JSON for the given object set.
       tag = 'phase1' or 'phase1_phase2'. Both PLYs carry semantic_id AND instance_id;
       they differ only in what the RGB encodes (type vs per-object)."""
    sem, inst = _assign(pts, recs)
    # semantic-colored
    sem_rgb = np.array([SEM_RGB[s] for s in sem], np.uint8)
    _write_ply_scalars(f"{out_dir}/working-cloud-semantic-{tag}.ply", pts, sem_rgb, sem, inst)
    # instance-colored (per-object palette; 0 -> black)
    pal = _instance_palette(len(recs))
    inst_rgb = np.zeros((len(pts), 3), np.uint8)
    nz = inst > 0
    inst_rgb[nz] = np.array([pal[i - 1] for i in inst[nz]], np.uint8)
    _write_ply_scalars(f"{out_dir}/working-cloud-instance-{tag}.ply", pts, inst_rgb, sem, inst)
    # labels json: semantic names + per-instance {semantic_id, type}
    labels = {"semantic_names": {str(k): ID_NAME[k] for k in sorted(ID_NAME)},
              "instances": {str(i + 1): {"semantic_id": SEMANTIC_ID.get(r["type"], 0),
                                         "type": r["type"], "object_id": r["id"]}
                            for i, r in enumerate(recs)}}
    json.dump(labels, open(f"{out_dir}/labels_{tag}.json", "w"), indent=2)


def colored_by_type(pts, recs):
    rgb = np.tile(COLORS["unclassified"], (len(pts), 1)).astype(float)
    for r in recs:
        rgb[G.points_in_obb(pts, r["obb"])] = COLORS.get(r["type"], [1, 1, 1])
    return rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e57"); ap.add_argument("--out", default="out")
    ap.add_argument("--voxel", type=float, default=None, help="override auto resolution (m)")
    ap.add_argument("--tau", type=float, default=0.4)
    ap.add_argument("--phase", choices=["both", "1", "2", "2only"], default="both",
                    help="both=full; 1=geometry only; 2=PTv3 on cached phase-1 remainder; "
                         "2only=PTv3 standalone on the full E57 (no phase 1), writes *_phase2 outputs")
    ap.add_argument("--config"); ap.add_argument("--weights"); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--strict", action="store_true",
                    help="precision mode: higher accept bar + tighter column/wall/mep gates, "
                         "lower coverage, fewer false positives (slab/joist_ceiling unchanged)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scorer = RuleScorer(strict=a.strict)

    if a.phase == "2only":
        # ---- Phase 2 STANDALONE on the full E57: no phase 1, every point claimable ----
        if a.e57 and os.path.exists(a.e57):
            full = G.load_e57(a.e57)
        else:
            print("no E57 -> synthetic scene"); _, full = G.synthetic()
        res = a.voxel if a.voxel else G.estimate_resolution(np.asarray(full.points))
        working = np.asarray(full.voxel_down_sample(res).points)
        working, T, fz, cz, fhw, chw = G.canonicalize(working)
        remaining = np.ones(len(working), bool)         # nothing pre-claimed
        print(f"phase2only: working={len(working)} pts | res={res:.3f} | "
              f"floor_z={fz:.2f} ceil_z={cz:.2f}")
        write_ply_cloud(working, f"{a.out}/working_cloud.ply")

        r2, claim2 = phase2_classify(working, remaining, [], scorer, fz, cz, fhw, chw, res,
                                     max(a.tau, 0.45), a.config, a.weights, a.device)
        json.dump(r2, open(f"{a.out}/boxes_phase2only.json", "w"), indent=2)
        write_boxes_ply(r2, f"{a.out}/boxes_phase2only.ply")
        write_ply_cloud(working[claim2], f"{a.out}/classified_phase2only.ply",
                        colored_by_type(working[claim2], r2))
        write_semantic_instance(working, r2, a.out, "phase2")
        rem = working[~claim2]
        write_ply_cloud(rem, f"{a.out}/remaining_phase2only.ply")
        H = cz - fz
        buckets = {"near_floor": int(((rem[:, 2] - fz) < 0.25 * H).sum()),
                   "mid": int((((rem[:, 2] - fz) >= 0.25 * H) & ((rem[:, 2] - fz) <= 0.75 * H)).sum()),
                   "near_ceiling": int(((rem[:, 2] - fz) > 0.75 * H).sum())}
        json.dump({"remaining_total": int((~claim2).sum()), "buckets": buckets},
                  open(f"{a.out}/remaining_phase2only_summary.json", "w"), indent=2)
        print(f"phase2only: {len(r2)} objects | claimed {claim2.sum()} | remaining {(~claim2).sum()}")
        print(f"\nDONE. Evaluate with:\n  python extraction_evaluation.py {a.out}/working_cloud.ply "
              f"{a.out}/boxes_phase2only.json")
        return

    if a.phase == "2":
        # ---- Phase 2 only: reuse Phase 1 outputs from --out, skip geometry ----
        working = np.asarray(o3d.io.read_point_cloud(f"{a.out}/working_cloud.ply").points)
        r1 = json.load(open(f"{a.out}/boxes_phase1.json"))
        rem_pts = np.asarray(o3d.io.read_point_cloud(f"{a.out}/remaining_phase1.ply").points)
        from scipy.spatial import cKDTree                       # reconstruct remaining mask
        d, _ = cKDTree(rem_pts).query(working, k=1)
        remaining = d < 1e-6
        z = working[:, 2]; fz, cz, fhw, chw = G.detect_levels(z)
        res = a.voxel if a.voxel else G.estimate_resolution(working)
        print(f"phase2-only: working={len(working)} | loaded {len(r1)} phase-1 boxes | "
              f"remaining={int(remaining.sum())}")
    else:
        if a.e57 and os.path.exists(a.e57):
            full = G.load_e57(a.e57)
        else:
            print("no E57 -> synthetic scene"); _, full = G.synthetic()
        xyz_full = np.asarray(full.points)
        res = a.voxel if a.voxel else G.estimate_resolution(xyz_full)
        full_ds = full.voxel_down_sample(res)            # builtin voxel-AVERAGE color
        working = np.asarray(full_ds.points)
        working, T, fz, cz, fhw, chw = G.canonicalize(working)
        print(f"working={len(working)} pts | res={res:.3f} | "
              f"floor_z={fz:.2f}(+/-{fhw:.2f}) ceil_z={cz:.2f}(+/-{chw:.2f})")
        write_ply_cloud(working, f"{a.out}/working_cloud.ply")
        # NEW: original-color cloud, in the SAME canonicalized frame as `working`
        if full_ds.has_colors():
            orig_rgb = (np.clip(np.asarray(full_ds.colors), 0, 1) * 255).astype(np.uint8)
            _write_ply_scalars(f"{a.out}/working-cloud-orig.ply", working, orig_rgb,
                               np.zeros(len(working), np.int32), np.zeros(len(working), np.int32))
            print(f"wrote working-cloud-orig.ply (color OK)")
        else:
            print("WARNING: cloud has no color -> working-cloud-orig.ply skipped")

        remaining = np.ones(len(working), bool)
        r1, claim1 = phase1_geometry(working, remaining, [], scorer, fz, cz, fhw, chw, res, a.tau,
                                     strict=a.strict)
        remaining &= ~claim1
        json.dump(r1, open(f"{a.out}/boxes_phase1.json", "w"), indent=2)
        write_boxes_ply(r1, f"{a.out}/boxes_phase1.ply")
        write_ply_cloud(working[claim1], f"{a.out}/classified_phase1.ply", colored_by_type(working[claim1], r1))
        write_ply_cloud(working[remaining], f"{a.out}/remaining_phase1.ply")
        write_semantic_instance(working, r1, a.out, "phase1")
        print(f"phase1: {len(r1)} objects | claimed {claim1.sum()} | remaining {remaining.sum()}")

    if a.phase == "1":
        print("phase 1 only -> done."); return

    r2, claim2 = phase2_classify(working, remaining, r1, scorer, fz, cz, fhw, chw, res,
                                 max(a.tau, 0.45), a.config, a.weights, a.device)
    remaining &= ~claim2
    json.dump(r2, open(f"{a.out}/boxes_phase2.json", "w"), indent=2)
    write_boxes_ply(r2, f"{a.out}/boxes_phase2.ply")
    write_ply_cloud(working[claim2], f"{a.out}/classified_phase2.ply", colored_by_type(working[claim2], r2))
    # NEW: semantic + instance PLYs + labels JSON over ALL phase-1 + phase-2 objects
    write_semantic_instance(working, r1 + r2, a.out, "phase1_phase2")
    print(f"phase2: {len(r2)} objects | claimed {claim2.sum()} | remaining {remaining.sum()}")

    # remainder, bucketed by height + colored, RGB-preserving from full-res cloud
    rem = working[remaining]
    write_ply_cloud(rem, f"{a.out}/remaining_final.ply")
    H = cz - fz
    buckets = {"near_floor": int(((rem[:, 2] - fz) < 0.25 * H).sum()),
               "mid": int((((rem[:, 2] - fz) >= 0.25 * H) & ((rem[:, 2] - fz) <= 0.75 * H)).sum()),
               "near_ceiling": int(((rem[:, 2] - fz) > 0.75 * H).sum())}
    json.dump({"remaining_total": int(remaining.sum()), "buckets": buckets},
              open(f"{a.out}/remaining_summary.json", "w"), indent=2)
    print(f"remainder buckets: {buckets}")
    print(f"\nDONE. Evaluate with:\n  python extraction_evaluation.py {a.out}/working_cloud.ply "
          f"{a.out}/boxes_phase1.json {a.out}/boxes_phase2.json")


if __name__ == "__main__":
    main()
