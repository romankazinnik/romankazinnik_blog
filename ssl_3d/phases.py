"""Additive phases. Both obey: phase_i(working, remaining_mask, boxes, scorer, fz, cz)
   -> (records, newly_claimed_mask).  Coverage only grows; context is the FULL cloud."""
import numpy as np
import cv2
import geometry as G
from schema import make_record


def _accept(working, remaining, obbs, type, phase, source, scorer, fz, cz, tau, min_pts=30):
    """Common path: gate each OBB by geometric confidence, claim only remaining points."""
    records, claimed = [], np.zeros(len(working), bool)
    for k, obb in enumerate(obbs):
        idx = np.asarray(G.points_in_obb(working, obb), dtype=np.int64)
        if idx.size == 0:
            continue
        idx = idx[remaining[idx]]                       # claim only unclaimed points
        if len(idx) < min_pts:
            continue
        feats = G.extract_features(working[idx], obb, fz, cz)
        conf = scorer.score(type, feats)
        if conf < tau:
            continue
        records.append(make_record(f"p{phase}_{type}_{k:03d}", type, phase, source, obb, feats, conf))
        claimed[idx] = True
    return records, claimed



def _merge_segments(segs, res, ang_tol_deg=8.0, perp_tol_m=0.4):
    """Fuse collinear/near Hough segments into single runs. segs: list of (x1,y1,x2,y2) in px."""
    if not segs:
        return []
    items = []
    for x1, y1, x2, y2 in segs:
        ang = np.arctan2(y2 - y1, x2 - x1) % np.pi
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        perp = (-np.sin(ang) * mx + np.cos(ang) * my)        # signed dist of midpoint to line dir
        items.append([ang, perp, x1, y1, x2, y2])
    used = [False] * len(items)
    out = []
    perp_tol_px = perp_tol_m / res
    for i in range(len(items)):
        if used[i]:
            continue
        ai, pi = items[i][0], items[i][1]
        grp = [items[i]]; used[i] = True
        for j in range(i + 1, len(items)):
            if used[j]:
                continue
            da = abs(items[j][0] - ai); da = min(da, np.pi - da)
            if da < np.radians(ang_tol_deg) and abs(items[j][1] - pi) < perp_tol_px:
                grp.append(items[j]); used[j] = True
        # project all endpoints onto the group's mean direction, take extremes
        am = np.mean([g[0] for g in grp]); dx, dy = np.cos(am), np.sin(am)
        pts = []
        for g in grp:
            pts += [(g[2], g[3]), (g[4], g[5])]
        t = [px * dx + py * dy for px, py in pts]
        lo, hi = int(np.argmin(t)), int(np.argmax(t))
        out.append((pts[lo][0], pts[lo][1], pts[hi][0], pts[hi][1]))
    return out

def _hough(mask, res, min_len_m, gap_m=0.4):
    e = cv2.Canny(mask, 40, 120)
    L = cv2.HoughLinesP(e, 1, np.pi/180, 40, minLineLength=int(min_len_m/res), maxLineGap=int(gap_m/res))
    return [] if L is None else [l[0] for l in L]


def _seg_to_obb(x1, y1, x2, y2, mn, res, zc, thick, h):
    wx1, wy1 = x1*res+mn[0], y1*res+mn[1]; wx2, wy2 = x2*res+mn[0], y2*res+mn[1]
    length = np.hypot(wx2-wx1, wy2-wy1)
    ang = np.arctan2(wy2-wy1, wx2-wx1)
    R = np.array([[np.cos(ang), -np.sin(ang), 0], [np.sin(ang), np.cos(ang), 0], [0, 0, 1.]])
    return length, {"center": np.array([(wx1+wx2)/2, (wy1+wy2)/2, zc]),
                    "extent": np.array([length, thick, h]), "R": R}


def phase1_geometry(working, remaining, boxes, scorer, fz, cz, fhw, chw, res, tau=0.4, strict=False):
    if strict:
        tau = max(tau, 0.55)                            # higher accept bar across all classes
    H = cz - fz; recs, claimed = [], np.zeros(len(working), bool)

    def add(obbs, t, src="geometry", mp=30):
        r, c = _accept(working, remaining & ~claimed, obbs, t, 1, src, scorer, fz, cz, tau, mp)
        recs.extend(r); claimed[:] |= c

    # floor slab only (concrete slab floor); ceiling handled as joist_ceiling below
    fp = working[np.abs(working[:, 2] - fz) < fhw]
    if len(fp) > 50:
        o = G.fit_obb(fp); o["extent"][2] = max(2 * fhw, 0.1); o["center"][2] = fz
        add([o], "slab", mp=200)

    vmask, mn, res = G.vertical_consistency_map(working, fz, cz, fhw, chw, res)
    vmask = cv2.morphologyEx(vmask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    # walls (drywall partitions): Hough on consistency map + build wall mask for columns
    wall_obbs, wallmask = [], np.zeros_like(vmask)
    wall_min = 1.5 if strict else 1.0
    for x1, y1, x2, y2 in _merge_segments(_hough(vmask, res, 1.5), res):
        length, o = _seg_to_obb(x1, y1, x2, y2, mn, res, fz + H/2, 0.2, H)
        if length >= wall_min:
            wall_obbs.append(o); cv2.line(wallmask, (x1, y1), (x2, y2), 255, int(0.4/res))
    add(wall_obbs, "wall")

    # columns: detect directly as full-height vertical clusters. Footprint measured at
    # MID-HEIGHT only (floor+25%..ceiling-25%) so a flared base/capital doesn't inflate it;
    # then confirm the cluster spans floor-to-ceiling. Robust for free-standing shafts.
    add(_detect_columns(working, remaining & ~claimed, wallmask, mn, res, fz, cz,
                        foot_max=0.8 if strict else 1.0,
                        span_frac=0.78 if strict else 0.65), "column")

    # beams (supporting girders): strongest long linear ridges in the soffit band
    soff = working[np.abs(working[:, 2] - cz) < chw]
    if len(soff) > 50:
        g = np.zeros(vmask.shape, np.uint8)
        ij = ((soff[:, :2] - mn)/res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < vmask.shape[1]) & (ij[:, 1] >= 0) & (ij[:, 1] < vmask.shape[0])
        ij = ij[ok]; g[ij[:, 1], ij[:, 0]] = 255
        g = cv2.morphologyEx(g, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        beam_obbs = []
        for x1, y1, x2, y2 in _merge_segments(_hough(g, res, 2.0, 0.5), res):
            length, o = _seg_to_obb(x1, y1, x2, y2, mn, res, cz - 0.3, 0.15, 0.3)
            if length >= 2.0:                          # girders are long; precision-first
                beam_obbs.append(o)
        add(beam_obbs, "beam")

    # MEP (round HVAC ducts / sprinkler+conduit pipe): crude horizontal tubular clusters that
    # hang below the joist soffit. Not Hough/plane - cluster + PCA: keep linear, horizontal,
    # compact-cross-section runs. (True cylinder RANSAC is future work.)
    add(_detect_mep(working, remaining & ~claimed, fz, cz, chw, strict=strict), "mep")

    # joist_ceiling (closely-spaced wood joist ceiling): the dense near-ceiling band as ONE
    # region object. Claims whatever near-ceiling points beams/mep did not, so it is additive.
    band = working[np.abs(working[:, 2] - cz) < 1.5 * chw]
    if len(band) > 200:
        o = G.fit_obb(band); o["extent"][2] = max(3 * chw, 0.15); o["center"][2] = cz
        add([o], "joist_ceiling", mp=200)

    return recs, claimed


def _detect_columns(working, free, wallmask, mn, res, fz, cz, max_pts=200000,
                    foot_max=1.0, span_frac=0.65):
    """Columns as full-height vertical shafts. Cluster the MID-HEIGHT band in XY (so the
    flared base/capital is excluded), keep compact clusters, then confirm each spans
    floor->ceiling using its full-height points. No consistency map, no Hough."""
    from sklearn.cluster import DBSCAN
    H = cz - fz
    lo, hi = fz + 0.25 * H, cz - 0.25 * H              # clean shaft window (no base/capital)
    mid = free & (working[:, 2] > lo) & (working[:, 2] < hi)
    pts = working[mid]
    if len(pts) < 100:
        return []
    samp = pts
    if len(samp) > max_pts:
        samp = samp[np.random.default_rng(0).choice(len(samp), max_pts, replace=False)]
    labels = DBSCAN(eps=4 * res, min_samples=10).fit(samp[:, :2]).labels_   # cluster footprints
    # dilate the wall mask so a candidate ON or adjacent to a wall line is excluded (a wall
    # pilaster is full-height + locally compact -> would otherwise be a false column)
    wall_d = cv2.dilate(wallmask, np.ones((int(0.5 / res) | 1,) * 2, np.uint8))
    Wh, Ww = wall_d.shape
    # full-height points (for the span test), restricted to unclaimed
    allpts = working[free]
    obbs = []
    for c in set(labels) - {-1}:
        fp = samp[labels == c]
        ex, ey = np.ptp(fp[:, 0]), np.ptp(fp[:, 1])
        if max(ex, ey) > foot_max or max(ex, ey) / max(min(ex, ey), 1e-3) > 2.5:
            continue                                   # not a compact square shaft
        cx, cy = fp[:, 0].mean(), fp[:, 1].mean()
        gi, gj = int((cx - mn[0]) / res), int((cy - mn[1]) / res)   # footprint cell on the raster
        if 0 <= gj < Wh and 0 <= gi < Ww and wall_d[gj, gi] > 0:
            continue                                   # sits on a wall -> not a free column
        r = max(max(ex, ey) / 2 + 0.1, 0.2)
        near = allpts[(np.abs(allpts[:, 0] - cx) < r) & (np.abs(allpts[:, 1] - cy) < r)]
        if len(near) < 100:
            continue
        if np.ptp(near[:, 2]) < span_frac * H:         # must reach floor->ceiling
            continue
        obbs.append({"center": np.array([cx, cy, fz + H / 2]),
                     "extent": np.array([max(ex, .2), max(ey, .2), H]), "R": np.eye(3)})
    return obbs


def _detect_mep(working, free, fz, cz, chw, max_pts=150000, strict=False):
    from sklearn.cluster import DBSCAN
    min_cluster = 250 if strict else 80                  # strict: only long unambiguous runs
    lin_min = 6 if strict else 4                          # strict: stronger linearity required
    H = cz - fz
    lo, hi = cz - 0.30 * H, cz - 1.5 * chw            # band between mid-height and the soffit
    m = free & (working[:, 2] > lo) & (working[:, 2] < hi)
    pts = working[m]
    if len(pts) < 200:
        return []
    if len(pts) > max_pts:
        pts = pts[np.random.default_rng(0).choice(len(pts), max_pts, replace=False)]
    labels = DBSCAN(eps=0.25, min_samples=12).fit(pts).labels_
    obbs = []
    for c in set(labels) - {-1}:
        cp = pts[labels == c]
        if len(cp) < min_cluster:
            continue
        d = cp - cp.mean(0); w, V = np.linalg.eigh(d.T @ d / len(cp))
        axis = V[:, -1]                                # eigenvector of largest eigenvalue
        w = np.clip(np.sort(w)[::-1], 1e-9, None)
        linearity = w[0] / w[1]
        horiz = abs(axis[2]) < 0.35                    # main axis roughly horizontal
        if linearity >= lin_min and horiz:
            obbs.append(G.fit_obb(cp))
    return obbs





def _ptv3_labels(working, config, weights, device, tile_m=10.0, overlap_m=1.0, budget=400000):
    """Real PTv3, TILED to fit GPU memory. Infers per spatial XY tile and stitches labels back.
    Overlap regions: a later tile only fills points not yet labelled (cheap, avoids seams)."""
    import torch
    from pointcept.models import build_model
    from pointcept.utils.config import Config
    cfg = Config.fromfile(config)
    model = build_model(cfg.model).to(device).eval()
    ck = torch.load(weights, map_location="cpu")
    model.load_state_dict({k.replace("module.", "", 1): v for k, v in ck.get("state_dict", ck).items()},
                          strict=False)

    labels = np.full(len(working), -1, np.int64)
    lo, hi = working[:, :2].min(0), working[:, :2].max(0)
    step = tile_m - overlap_m
    zmin = working[:, 2].min()
    for x0 in np.arange(lo[0], hi[0] + step, step):
        for y0 in np.arange(lo[1], hi[1] + step, step):
            m = ((working[:, 0] >= x0) & (working[:, 0] < x0 + tile_m) &
                 (working[:, 1] >= y0) & (working[:, 1] < y0 + tile_m))
            idx = np.where(m)[0]
            if len(idx) < 100:
                continue
            if len(idx) > budget:                       # cap a dense tile to fit VRAM
                idx = np.random.default_rng(0).choice(idx, budget, replace=False)
            tp = working[idx]
            c = tp - tp.mean(0); c[:, 2] = tp[:, 2] - zmin
            gc = np.floor(tp / 0.04).astype(np.int64); gc -= gc.min(0)
            data = {"coord": torch.tensor(c, dtype=torch.float32, device=device),
                    "grid_coord": torch.tensor(gc, device=device),
                    "feat": torch.tensor(np.concatenate([c, np.full_like(c, 0.5)], 1),
                                         dtype=torch.float32, device=device),
                    "offset": torch.tensor([len(tp)], device=device)}
            with torch.no_grad():
                out = model(data)
            seg = out["seg_logits"] if isinstance(out, dict) else out
            pred = seg.argmax(1).cpu().numpy()
            fill = labels[idx] < 0                       # don't overwrite already-labelled overlap
            labels[idx[fill]] = pred[fill]
            del data, out, seg
            torch.cuda.empty_cache()
    return labels


# S3DIS ids -> our types
_S3DIS = {2: "wall", 4: "column", 3: "beam"}


def phase2_classify(working, remaining, boxes, scorer, fz, cz, fhw, chw, res, tau=0.45,
                    config=None, weights=None, device="cuda:0"):
    """PTv3-corroborated if available; else relaxed geometric second pass. Same interface."""
    try:
        if not (config and weights):
            raise RuntimeError("no PTv3 config/weights given")
        labels = _ptv3_labels(working, config, weights, device)
        recs, claimed = [], np.zeros(len(working), bool)
        for sid, t in _S3DIS.items():
            m = (labels == sid) & remaining
            if m.sum() < 50:
                continue
            pts = working[m]; idxall = np.where(m)[0]
            from sklearn.cluster import DBSCAN
            lab = DBSCAN(eps=0.2, min_samples=10).fit(pts).labels_
            obbs = [G.fit_obb(pts[lab == c]) for c in set(lab) - {-1} if (lab == c).sum() >= 30]
            r, cl = _accept(working, remaining & ~claimed, obbs, t, 2, "ptv3+geometry",
                            scorer, fz, cz, tau)
            recs.extend(r); claimed[:] |= cl
        return recs, claimed
    except Exception as e:
        print(f"[phase2] PTv3 unavailable ({e}); relaxed geometric pass")
        # relaxed second geometric pass on the remainder only, lower thresholds
        sub = working.copy()
        r, claimed = phase1_geometry(sub, remaining, boxes, scorer, fz, cz, fhw, chw, res, tau=tau * 0.7)
        for rec in r:
            rec["phase"] = 2; rec["source"] = "geometry-relaxed"
            rec["id"] = "p2_" + rec["id"][3:]
        return r, claimed
