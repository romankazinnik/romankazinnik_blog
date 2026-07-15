"""
ssl_scene_probe.py
==================
Minimal, self-contained test of one question:

    "Does a self-supervised pretext (mask points, predict their color/geometry
     from spatial context) produce per-point features that separate structural
     elements BETTER than the raw xyz+rgb inputs -- on held-out data?"

That comparison is the point. Training loss going down is NOT proof (point-SSL
is famous for collapsing to a 'geometric shortcut' where loss drops but features
are trivial). So the success criterion here is:

    k-means(SSL features)  vs  k-means(raw xyz+rgb),  scored on a HELD-OUT scene.

On synthetic data we have ground-truth structure labels, so the score is a real
number (Adjusted Rand Index). On your real scene, swap in --data / --eval_data
and read the same pipeline qualitatively (PCA-RGB + cluster .ply written out).

Train scene 1, validate on scene 2 -- matching the two-dataset setup.

Deps: torch, numpy, scipy, scikit-learn  (matplotlib + open3d/laspy optional).
Runs in well under 30 min on one L4 at default config; seconds on CPU at --smoke.
"""

import argparse, time, os
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    data: str = ""          # path to train scene; "" -> synthetic
    run_classifier: str = ""  # label a NEW cloud: dir of a previous run (uses its cache/model.pt + classifier.pkl)
    extra_data: str = ""    # comma-separated UNLABELLED clouds that join SSL pretraining (no labels needed)
    block_scale: float = 2.0  # fixed metric scale (m) for block coords -> density/scale-stable embeddings
    color_ply: str = ""        # .ply (same points as --data) with the REAL scan colours for SSL features
    labels_from_colors: bool = False  # force labels from --data vertex colours even if semantic_id exists
    color_min_count: int = 50  # when decoding colour labels, drop colours with fewer points than this
    max_color_classes: int = 50  # if colour-decoding finds more classes than this, error (looks like a scan)
    names_json: str = ""    # {"semantic_names":{"1":"wall"}} for ids, or {"color_names":{"0,180,0":"wall"}} for colours
    sem_field: str = "semantic_id"   # per-vertex label field in the .ply
    palette_ply: str = ""   # optional .ply with semantic_id + type-RGB -> match output colors to it
    voxel: float = 0.0      # voxel downsample size (m); 0 = keep all points (e57 still uses 0.04)
    max_points: int = 4_000_000  # random cap after voxel (keeps KDTree/RAM sane)
    dbscan_eps: float = 0.3      # DBSCAN neighbourhood radius (m) for instance splitting
    dbscan_min: int = 30         # DBSCAN min samples per object
    dbscan_cap: int = 150_000    # per-class point cap for DBSCAN (subsample+propagate above this)
    probe_max_train: int = 50_000   # cap on points used to FIT the classifier (sklearn is CPU-only)
    eval_frac: float = 0.2          # fraction of each class held out for eval (rest -> fit, capped by probe_max_train)
    probe_pca: int = 64             # PCA-reduce SSL features before the classifier (0 = off)
    head: str = "mlp"              # downstream classifier: "mlp" (2-layer NN) or "linear"
    conf_thresh: float = 0.5        # below this, a predicted point is treated as low-confidence (black)
    smooth_k: int = 12              # KNN majority smoothing of predictions (0 = off) -> reduces speckle
    force_retrain: bool = False     # ignore cached model.pt / features.npz and recompute
    K: int = 2048           # points per block
    n_blocks: int = 512     # training blocks sampled from the scene
    n_eval_blocks: int = 96 # disjoint blocks used for evaluation
    max_eval_centers: int = 1500  # cap on tiles for dense visualization inference
    dim: int = 256          # embedding width
    depth: int = 4          # transformer layers
    heads: int = 4
    mask_ratio: float = 0.5
    xyz_noise: float = 0.05  # std of coord noise for the geometry (denoising) task
    w_xyz: float = 1.0       # weight on the xyz loss; lower -> more semantic, less positional
    epochs: int = 40
    batch: int = 16
    lr: float = 1e-3
    max_minutes: float = 25.0
    n_clusters: int = 6     # k for k-means (set to your expected #structures)
    seed: int = 0
    out: str = "ssl_out"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Data: loader + synthetic fallback (with GT labels for a real score)
# --------------------------------------------------------------------------- #
def validate_inputs(cfg):
    """Fail fast, BEFORE any loading or long processing, if any cmdline input path is
    missing. Collects all problems and reports them together."""
    missing = []
    for flag, p in (("--data", cfg.data), ("--color_ply", cfg.color_ply),
                    ("--names_json", cfg.names_json), ("--palette_ply", cfg.palette_ply)):
        if p and not os.path.exists(p):
            missing.append(f"{flag}: {p} (file not found)")
    for p in [q.strip() for q in cfg.extra_data.split(",") if q.strip()]:
        if not os.path.exists(p):
            missing.append(f"--extra_data: {p} (file not found)")
    if cfg.run_classifier:
        d = cfg.run_classifier
        if not os.path.isdir(d):
            missing.append(f"--run_classifier: {d} (directory not found)")
        else:
            mp = os.path.join(d, "cache", "model.pt")
            bp = os.path.join(d, "cache", "classifier.pkl")
            if not os.path.exists(mp):
                missing.append(f"--run_classifier: {mp} (no trained SSL model in this run dir)")
            if not os.path.exists(bp):
                msg = f"--run_classifier: {bp} (classifier bundle not found"
                if os.path.exists(mp):
                    msg += ("; model.pt EXISTS but the bundle is missing -- the training run predates "
                            "bundle saving or never reached classification. Rerun the training command "
                            "once: cached model.pt/features.npz make it fast")
                missing.append(msg + ")")
    if missing:
        raise SystemExit("**** INPUT VALIDATION FAILED ****\n  " + "\n  ".join(missing))


def load_points(path, voxel=0.0, rng=None, sem_field="semantic_id"):
    """Return (xyz[N,3], rgb[N,3] in [0,1] or None, labels[N] or None)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".e57":
        return load_e57(path, voxel)
    if ext == ".npy":
        a = np.load(path)
        xyz, rgb = a[:, :3], a[:, 3:6]
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
        return xyz.astype(np.float32), rgb.astype(np.float32), None
    if ext == ".ply":
        from plyfile import PlyData
        v = PlyData.read(path)["vertex"].data
        names = v.dtype.names
        xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
        if all(c in names for c in ("red", "green", "blue")):
            rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
        else:
            rgb = None
        sem = v[sem_field].astype(np.int64) if sem_field in names else None
        return xyz, rgb, sem
    if ext == ".pcd":
        import open3d as o3d
        pc = o3d.io.read_point_cloud(path)
        xyz = np.asarray(pc.points, np.float32)
        rgb = np.asarray(pc.colors, np.float32)
        return xyz, (rgb if rgb.size else None), None
    if ext in (".las", ".laz"):
        import laspy
        f = laspy.read(path)
        xyz = np.stack([f.x, f.y, f.z], 1).astype(np.float32)
        rgb = np.stack([f.red, f.green, f.blue], 1).astype(np.float32)
        rgb = rgb / rgb.max() if rgb.max() > 0 else rgb
        return xyz, rgb, None
    raise ValueError(f"Unsupported file type: {ext}")


def load_e57(path, voxel):
    """Read an E57 scan-by-scan, voxel-downsampling each scan immediately so a
    multi-GB / 100M-point file never sits in RAM at full resolution. Handles
    multiple scans, missing color (falls back to intensity, then grey)."""
    import pye57
    e = pye57.E57(path)
    vox = voxel if voxel > 0 else 0.04   # always downsample E57 during read
    xs, cs = [], []
    for i in range(e.scan_count):
        d = e.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True)
        xyz = np.stack([d["cartesianX"], d["cartesianY"], d["cartesianZ"]], 1).astype(np.float32)
        if d.get("colorRed") is not None and len(d["colorRed"]) == len(xyz):
            rgb = np.stack([d["colorRed"], d["colorGreen"], d["colorBlue"]], 1).astype(np.float32)
            if rgb.max() > 1.5:
                rgb /= 255.0
        elif d.get("intensity") is not None and len(d["intensity"]) == len(xyz):
            it = np.asarray(d["intensity"], np.float32)
            it = (it - it.min()) / (np.ptp(it) + 1e-6)
            rgb = np.repeat(it[:, None], 3, 1)
        else:
            rgb = np.full_like(xyz, 0.5)
        xyz, rgb, _ = voxel_downsample(xyz, rgb, None, vox)   # bound memory per scan
        xs.append(xyz); cs.append(rgb)
        print(f"  scan {i}: {len(xyz):,} pts after {vox} m voxel")
    e.close()
    xyz = np.concatenate(xs); rgb = np.concatenate(cs)
    xyz, rgb, _ = voxel_downsample(xyz, rgb, None, vox)        # dedup scan overlap
    return xyz, rgb, None


def synthetic_scene(seed, n=120_000):
    """A toy room: floor, two walls, ceiling, box columns, cylinder pipes.
    Colors overlap and are noisy on purpose, so raw rgb alone is NOT enough to
    recover structure -- the model must use spatial context to win."""
    rng = np.random.default_rng(seed)
    pts, labs, cols = [], [], []

    def add(p, lab, base):
        c = np.clip(base + rng.normal(0, 0.18, p.shape), 0, 1)  # heavy color noise
        pts.append(p); labs.append(np.full(len(p), lab)); cols.append(c)

    # 0 floor, 1 wall, 2 ceiling, 3 column, 4 pipe, 5 wall-2
    add(np.c_[rng.uniform(0, 8, n//4), rng.uniform(0, 8, n//4), np.zeros(n//4)], 0, [.25, .25, .25])
    add(np.c_[np.zeros(n//8), rng.uniform(0, 8, n//8), rng.uniform(0, 3, n//8)], 1, [.85, .85, .8])
    add(np.c_[rng.uniform(0, 8, n//8), np.zeros(n//8), rng.uniform(0, 3, n//8)], 5, [.8, .82, .85])
    add(np.c_[rng.uniform(0, 8, n//4), rng.uniform(0, 8, n//4), np.full(n//4, 3.0)], 2, [.55, .4, .3])
    for cx, cy in [(2, 2), (6, 2), (2, 6), (6, 6)]:
        m = n//20
        t = rng.uniform(0, 3, m)
        add(np.c_[cx + rng.uniform(-.2, .2, m), cy + rng.uniform(-.2, .2, m), t], 3, [.5, .5, .55])
    for cy in [4.0]:
        m = n//20
        x = rng.uniform(0, 8, m); ang = rng.uniform(0, 2*np.pi, m)
        add(np.c_[x, cy + .1*np.cos(ang), 2.7 + .1*np.sin(ang)], 4, [.6, .55, .4])

    xyz = np.concatenate(pts).astype(np.float32)
    rgb = np.concatenate(cols).astype(np.float32)
    lab = np.concatenate(labs).astype(np.int64)
    idx = rng.permutation(len(xyz))
    return xyz[idx], rgb[idx], lab[idx]


def voxel_downsample(xyz, rgb, lab, vox):
    """Keep one point per voxel. Hashes 3D voxel index to a single int64 and
    uniques in 1D -- far lighter than np.unique(axis=0) at 100M points."""
    if vox <= 0:
        return xyz, rgb, lab
    keys = np.floor(xyz / vox).astype(np.int64)
    keys -= keys.min(0)
    dims = keys.max(0) + 1
    if dims[1] * dims[2] > 4e17 or dims[0] * dims[1] * dims[2] > 9e18:
        _, keep = np.unique(keys, axis=0, return_index=True)   # overflow-safe fallback
    else:
        flat = keys[:, 0] * (dims[1] * dims[2]) + keys[:, 1] * dims[2] + keys[:, 2]
        _, keep = np.unique(flat, return_index=True)
    keep.sort()
    return xyz[keep], (rgb[keep] if rgb is not None else None), (lab[keep] if lab is not None else None)


def cap_points(xyz, rgb, lab, max_points, rng):
    if max_points and len(xyz) > max_points:
        sel = rng.choice(len(xyz), max_points, replace=False); sel.sort()
        return xyz[sel], (rgb[sel] if rgb is not None else None), (lab[sel] if lab is not None else None)
    return xyz, rgb, lab


GEOM_KS = (16, 64)          # neighbourhood sizes (multi-scale -> robust to scan density)
GEOM_D = 5 * len(GEOM_KS)   # descriptor width


def geom_descriptor(xyz, tree, ks=GEOM_KS, chunk=200_000):
    """Rotation- and translation-INVARIANT local features (transfer across sites/frames).
    Per scale k: [normal verticality, direction verticality, planarity, linearity, curvature].
    Eigen*values* + vertical projections only -- no frame-dependent eigenvector directions.
    direction verticality (|z| of dominant eigenvector) separates columns (vertical linear)
    from beams (horizontal linear); curvature = surface variation l3/(l1+l2+l3)."""
    N = len(xyz); out = np.zeros((N, 5 * len(ks)), np.float32)
    for si, k in enumerate(ks):
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            _, idx = tree.query(xyz[s:e], k=min(k, N))       # [c,k]
            nb = xyz[idx]                                    # [c,k,3]
            nb = nb - nb.mean(1, keepdims=True)
            cov = np.einsum("ckm,ckn->cmn", nb, nb) / k      # [c,3,3]
            w, vec = np.linalg.eigh(cov)                     # ascending: w0<=w1<=w2
            l3, l2, l1 = w[:, 0], w[:, 1], w[:, 2]
            l1c = l1 + 1e-9
            o = 5 * si
            out[s:e, o + 0] = 1.0 - np.abs(vec[:, 2, 0])    # normal verticality (normal = smallest evec)
            out[s:e, o + 1] = np.abs(vec[:, 2, 2])          # direction verticality (dominant evec)
            out[s:e, o + 2] = (l2 - l3) / l1c               # planarity
            out[s:e, o + 3] = (l1 - l2) / l1c               # linearity
            out[s:e, o + 4] = l3 / (l1 + l2 + l3 + 1e-9)    # curvature (surface variation)
    return out


def median_spacing(xyz, tree, n_sample=20_000):
    """Median 1-NN distance (m): the density fingerprint that must roughly match between
    training and inference clouds for the descriptor/SSL features to mean the same thing."""
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xyz), min(len(xyz), n_sample), replace=False)
    d, _ = tree.query(xyz[sub], k=2)
    return float(np.median(d[:, 1]))


def estimate_up(xyz, tree, n_sample=200_000):
    """Estimate the gravity (up) axis FROM THE DATA: local normals on a sample,
    hemisphere-canonicalised, dominant eigenvector of their structure tensor = up
    (slab normals dominate a building's normal distribution). Returns (u, R, tilt_deg)
    with R mapping u -> +z. Assumes the scan is not upside-down (tilt < 90 deg)."""
    rng = np.random.default_rng(0)
    sub = rng.choice(len(xyz), min(len(xyz), n_sample), replace=False)
    _, idx = tree.query(xyz[sub], k=16)
    nb = xyz[idx]; nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("ckm,ckn->cmn", nb, nb) / 16
    w, vec = np.linalg.eigh(cov)
    keep = w[:, 0] < 0.1 * (w[:, 2] + 1e-9)             # planar points only
    n = vec[keep][:, :, 0]                               # local normals (smallest evec)
    n[n[:, 2] < 0] *= -1                                 # hemisphere canonicalisation
    T = n.T @ n / max(1, len(n))
    _, tv = np.linalg.eigh(T)
    u = tv[:, 2]                                         # dominant normal direction = up
    if u[2] < 0:
        u = -u
    tilt = float(np.degrees(np.arccos(np.clip(abs(u[2]), -1, 1))))
    if tilt < 2.0:                                       # already gravity-aligned
        return np.array([0., 0., 1.], np.float32), np.eye(3, dtype=np.float32), tilt
    v = np.cross(u, [0., 0., 1.]); s = np.linalg.norm(v); c = float(u[2])
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + K + K @ K * ((1 - c) / (s * s + 1e-12))   # Rodrigues: u -> z
    return u.astype(np.float32), R.astype(np.float32), tilt


def height_features(xyz, tree=None, cell=2.0, verbose=True):
    """Per-point [height above story floor, height below story ceiling] in METERS.
    Story elevations are detected globally as MODES of the z-histogram of near-horizontal
    points (slabs dominate the mass), which is robust to below-floor reflection ghosts
    (extrema-based percentiles are not), occluded cells, and multi-story clouds.
    Falls back to smoothed per-cell percentiles if no clear slab modes exist (e.g. terrain)."""
    N = len(xyz)
    # -- horizontal points via cheap local normals on a subsample --
    rng = np.random.default_rng(0)
    sub = rng.choice(N, min(N, 300_000), replace=False)
    t = tree if tree is not None else cKDTree(xyz)
    _, idx = t.query(xyz[sub], k=16)
    nb = xyz[idx]; nb = nb - nb.mean(1, keepdims=True)
    cov = np.einsum("ckm,ckn->cmn", nb, nb) / 16
    _, vec = np.linalg.eigh(cov)
    horiz = np.abs(vec[:, 2, 0]) > 0.9                      # normal ~vertical -> horizontal surface
    zh = xyz[sub][horiz, 2]
    levels = []
    if len(zh) > 1000:
        lo, hi = zh.min(), zh.max()
        nbins = max(8, int((hi - lo) / 0.05))
        hist, edges = np.histogram(zh, bins=nbins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        order = np.argsort(-hist)
        for oi in order:                                    # greedy peaks, >=1.5m apart, >=2% mass
            if hist[oi] < 0.02 * len(zh):
                break
            z0 = centers[oi]
            if all(abs(z0 - L) > 1.5 for L in levels):
                m = np.abs(zh - z0) < 0.15                  # refine peak by local median
                levels.append(float(np.median(zh[m])))
        levels.sort()
    if levels:
        if verbose:
            print(f"  height ref: {len(levels)} slab level(s) at z = "
                  + ", ".join(f"{L:.2f}" for L in levels) + " m (z-histogram modes)")
        L = np.array(levels, np.float32); tol = 0.1
        z = xyz[:, 2]
        # story floor = highest level STRICTLY below (a ceiling slab belongs to the story under it);
        # bottom-slab points clamp to level 0. Symmetric for the ceiling side.
        below = np.clip(np.searchsorted(L, z - tol) - 1, 0, len(L) - 1)
        hf = np.maximum(z - L[below], 0.0)
        above = np.searchsorted(L, z + tol)
        ztop = float(np.percentile(z, 99.9))                 # roof for above-top-slab points
        top = np.where(above < len(L), L[np.clip(above, 0, len(L) - 1)], ztop)
        hc = np.maximum(top - z, 0.0)
        return np.stack([hf, hc], 1).astype(np.float32)
    # -- fallback: per-cell percentiles with neighbour-median smoothing --
    if verbose:
        print("  height ref: no slab modes found -> per-cell percentile fallback")
    g = np.floor((xyz[:, :2] - xyz[:, :2].min(0)) / cell).astype(np.int64)
    gw = g[:, 1].max() + 2
    key = g[:, 0] * gw + g[:, 1]
    order = np.argsort(key)
    ks_, zs = key[order], xyz[order, 2]
    starts = np.r_[0, np.flatnonzero(np.diff(ks_)) + 1, len(ks_)]
    cell_lo, cell_hi, cell_key = [], [], []
    for a, b in zip(starts[:-1], starts[1:]):
        z = zs[a:b]
        cell_lo.append(np.percentile(z, 2)); cell_hi.append(np.percentile(z, 98))
        cell_key.append(ks_[a])
    cell_lo = np.array(cell_lo); cell_hi = np.array(cell_hi); cell_key = np.array(cell_key)
    med_lo = np.median(cell_lo)                              # smooth outlier cells to global median
    bad = np.abs(cell_lo - med_lo) > 1.0
    cell_lo[bad] = med_lo
    lut_lo = dict(zip(cell_key, cell_lo)); lut_hi = dict(zip(cell_key, cell_hi))
    hf = xyz[:, 2] - np.array([lut_lo[k] for k in key], np.float32)
    hc = np.array([lut_hi[k] for k in key], np.float32) - xyz[:, 2]
    return np.stack([np.maximum(hf, 0), np.maximum(hc, 0)], 1).astype(np.float32)


def load_names(path):
    if not path or not os.path.exists(path):
        return {}
    import json
    d = json.load(open(path)).get("semantic_names", {})
    return {int(k): v for k, v in d.items()}


def names_for_colors(path, palette):
    """Map decoded class ids -> names using {"color_names": {"r,g,b": name}} in the JSON.
    palette[id] is that class's RGB (0..1); unmatched classes fall back to their id string."""
    cn = {}
    if path and os.path.exists(path):
        import json
        cn = json.load(open(path)).get("color_names", {})
    cn = {tuple(int(x) for x in k.replace(" ", "").split(",")): v for k, v in cn.items()}
    names = {}
    for cid in range(1, len(palette)):
        rgb = tuple(int(round(v * 255)) for v in palette[cid])
        names[cid] = cn.get(rgb, str(cid))
    return names


def transfer_rgb(target_xyz, color_ply, rng):
    """Pull per-vertex RGB from a separate cloud onto target_xyz by nearest neighbour.
    Robust to point reordering/subsampling; prints max match distance (~0 = same points)."""
    sx, srgb, _ = load_points(color_ply, 0.0, rng)
    if srgb is None:
        raise ValueError(f"{color_ply} has no red/green/blue vertex properties")
    d, nn = cKDTree(sx).query(target_xyz, k=1)
    print(f"  rgb transfer from {os.path.basename(color_ply)}: "
          f"max NN dist {float(d.max()):.4f} m (≈0 means same points)")
    return srgb[nn].astype(np.float32)


SEM_PALETTE = np.array([
    [.45, .45, .45], [.20, .55, .85], [.95, .75, .15], [.25, .75, .35],
    [.85, .25, .25], [.65, .35, .80], [.30, .80, .80], [.90, .50, .20],
    [.55, .40, .25], [.90, .30, .60]], dtype=np.float32)  # idx 0 = unclassified (grey)


def labels_from_color(rgb, min_count=50):
    """Turn class-coloured points into semantic ids. Black (0,0,0) = unlabelled (id 0);
    each distinct non-black colour becomes a class (ids by frequency, most common first).
    Returns (sem[N], palette[K+1,3], n_distinct_nonblack_colours)."""
    q = np.round(np.clip(rgb, 0, 1) * 255).astype(np.int64)
    key = (q[:, 0] << 16) | (q[:, 1] << 8) | q[:, 2]           # pack rgb -> single int
    black = key == 0
    sem = np.zeros(len(rgb), np.int64)
    if (~black).sum() == 0:
        return sem, SEM_PALETTE, 0
    ukey, inv, cnt = np.unique(key[~black], return_inverse=True, return_counts=True)
    lut = np.zeros(len(ukey), np.int64)                        # ukey index -> class id (0 = dropped)
    colors = [SEM_PALETTE[0]]                                  # id 0 -> grey placeholder
    cid = 1
    for oi in np.argsort(-cnt):                                # frequent colours first
        if cnt[oi] < min_count:
            continue
        lut[oi] = cid
        k = int(ukey[oi])
        colors.append(np.array([(k >> 16) & 255, (k >> 8) & 255, k & 255], np.float32) / 255.0)
        cid += 1
    sem[~black] = lut[inv]
    return sem, np.array(colors, np.float32), len(ukey)


def derive_palette(path, sem_field, rng):
    """Build a semantic_id -> RGB palette from a type-coloured .ply, so outputs match
    the user's own colour scheme (mean colour per semantic_id). Falls back to SEM_PALETTE."""
    _, rgb, sem = load_points(path, 0.0, rng, sem_field)
    if rgb is None or sem is None:
        return SEM_PALETTE
    maxid = int(sem.max())
    pal = SEM_PALETTE.copy()
    if maxid + 1 > len(pal):
        pal = np.vstack([pal, np.tile(pal[-1], (maxid + 1 - len(pal), 1))])
    for c in np.unique(sem):
        pal[c] = rgb[sem == c].mean(0)
    return pal.astype(np.float32)


def classify_and_report(model, xyz, feat_chan, sem, names, cfg, tree, rng, palette, feat=None,
                        xyz_out=None):
    """Train a classifier model on labelled points (sem>0), score on a held-out labelled
    split, then predict the unclassified (sem==0) points. Compares SSL features vs raw
    geometry; reports per-class IoU, how many points were classified, and confidence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    xo = xyz if xyz_out is None else xyz_out            # outputs stay in the INPUT frame
    if feat is None:
        print("\nembedding full cloud (dense inference) ...", flush=True)
        feat, _ = dense_embed(model, xyz, feat_chan, cfg, tree, rng)   # SSL features, full cloud
    feature_health(feat)
    pca = None
    if cfg.probe_pca and feat.shape[1] > cfg.probe_pca:
        pca = PCA(cfg.probe_pca).fit(feat)                            # kept: needed to label new clouds
        feat = pca.transform(feat).astype(np.float32)
    feat = ((feat - feat.mean(0)) / (feat.std(0) + 1e-6)).astype(np.float32)  # per-cloud moment alignment
    # SSL blocks are mean-centred (height stripped) -> give the classifier position back, but
    # only as INVARIANT height-above-floor / below-ceiling (meters). No absolute xy: that would
    # memorise this building's coordinates and destroy transfer to other clouds/sites.
    hpos = height_features(xyz, tree=tree)
    hdbg = np.clip(hpos[:, :1] / (np.percentile(hpos[:, 0], 99) + 1e-6), 0, 1)
    save_ply(os.path.join(cfg.out, "height_debug.ply"), xo,
             np.concatenate([hdbg, 1 - hdbg, np.zeros_like(hdbg)], 1))   # red=high, green=floor
    print(f"  wrote {cfg.out}/height_debug.ply (audit: colour = height above story floor)")
    feat = np.concatenate([feat, hpos], 1)                            # SSL features + invariant height
    raw = np.concatenate([hpos, feat_chan], 1)                       # geometry baseline (same height)

    lab = sem > 0; unl = ~lab
    nm = lambda c: names.get(int(c), str(int(c)))
    print("\n=== CLASSIFICATION (semantic) ===")
    if int(sem.max()) >= len(palette):
        print(f"**** WARNING: semantic ids reach {int(sem.max())} but palette has "
              f"{len(palette)} colours; colours will repeat (id % {len(palette)}) ****")
    print("colour legend (semantic_id -> name -> output RGB):")
    for c in np.unique(sem[lab]):
        r, g, b = (palette[int(c) % len(palette)] * 255).astype(int)
        print(f"    {int(c)} {nm(c):<10} rgb({r},{g},{b})")
    print(f"labelled points: {lab.sum():,} | unclassified (sem==0): {unl.sum():,}")
    for c in np.unique(sem[lab]):
        print(f"    {nm(c):<12} id={c}  {int((sem == c).sum()):,}")

    # per-class split: hold out eval_frac for eval (>=1 pt if class has >=2), rest is fit,
    # then balance-cap the fit at probe_max_train/n_classes. Eval = every non-fit labelled point.
    lab_idx = np.where(lab)[0]
    classes = np.unique(sem[lab_idx])
    per_cls = max(1, cfg.probe_max_train // len(classes))
    fit = []
    for c in classes:
        ci = rng.permutation(lab_idx[sem[lab_idx] == c])
        n_ev = min(len(ci) - 1, int(round(len(ci) * cfg.eval_frac))) if len(ci) > 1 else 0
        fit_c = ci[n_ev:]                                # rest of class is eligible for fit
        fit += list(fit_c[:per_cls])                     # balance-cap
    fit = np.array(fit)
    in_fit = np.zeros(len(sem), bool); in_fit[fit] = True
    ev = lab_idx[~in_fit[lab_idx]]                        # all labelled points not used to fit
    print(f"  (classifier={cfg.head}, eval_frac={cfg.eval_frac}, fit on {len(fit):,} balanced pts, "
          f"eval on {len(ev):,} = all other labelled)")

    def make_clf():
        if cfg.head == "linear":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, class_weight="balanced"))
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                           early_stopping=True, n_iter_no_change=8, random_state=cfg.seed))

    def fit_eval(name, X):
        print(f"  fitting model[{name}] on {len(fit):,}x{X.shape[1]} ...", flush=True)
        clf = make_clf().fit(X[fit], sem[fit])
        pred = clf.predict(X[ev])
        acc = float((pred == sem[ev]).mean())
        per = {}
        for c in np.unique(sem[ev]):
            inter = np.sum((pred == c) & (sem[ev] == c)); union = np.sum((pred == c) | (sem[ev] == c))
            if union: per[int(c)] = inter / union
        miou = float(np.mean(list(per.values())))
        cls = "  ".join(f"{nm(c)}:{v:.2f}" for c, v in per.items())
        print(f"  model[{name:4}]  acc {acc:.3f}  mIoU {miou:.3f}  | IoU  {cls}")
        return clf, miou

    print("\nclassification quality on held-out labelled points:")
    _,        miou_raw = fit_eval("raw", raw)
    clf_ssl,  miou_ssl = fit_eval("SSL", feat)
    print(f"  => SSL mIoU {miou_ssl:.3f} vs raw {miou_raw:.3f} :: "
          f"{'SSL helps' if miou_ssl > miou_raw + 0.02 else 'no gain over raw geometry'}")

    # persist everything needed to label a NEW cloud with this classifier (see --run_classifier)
    import pickle
    bundle_path = os.path.join(cfg.out, "cache", "classifier.pkl")
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    spacing = median_spacing(xyz, tree)
    print(f"  median point spacing: {spacing:.3f} m (stored in bundle for density parity checks)")
    with open(bundle_path, "wb") as f:
        pickle.dump({"clf": clf_ssl, "pca": pca, "palette": palette, "names": names,
                     "in_feat": feat_chan.shape[1], "spacing": spacing}, f)
    print(f"[cache] saved classifier bundle -> {bundle_path}")

    # ---- train/eval labelled-set outputs: predicted class + mistake maps ----
    OK, BAD = np.array([.20, .80, .20], np.float32), np.array([.90, .15, .15], np.float32)
    for tag, sel in (("train", fit), ("eval", ev)):
        p = clf_ssl.predict(feat[sel])
        save_ply(os.path.join(cfg.out, f"pred_{tag}_points.ply"), xo[sel], palette[p % len(palette)])
        wrong = p != sem[sel]
        save_ply(os.path.join(cfg.out, f"mistake_{tag}_points.ply"), xo[sel],
                 np.where(wrong[:, None], BAD, OK))
        print(f"  wrote pred_{tag}_points.ply + mistake_{tag}_points.ply "
              f"({len(sel):,} pts, {int(wrong.sum()):,} wrong)")

    # predict the unclassified points with the SSL model
    if unl.sum():
        proba = clf_ssl.predict_proba(feat[unl])
        pred = clf_ssl.classes_[proba.argmax(1)]
        conf = proba.max(1)
        print(f"\npredicted {unl.sum():,} unclassified points:")
        for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]:
            mm = (conf >= lo) & (conf < hi)
            print(f"    confidence [{lo:.1f},{hi if hi<=1 else 1.0:.1f}): {int(mm.sum()):,}")
        print(f"    mean conf {conf.mean():.3f}  median {np.median(conf):.3f}  "
              f"(low-conf < {cfg.conf_thresh}: {int((conf < cfg.conf_thresh).sum()):,})")
        for c in np.unique(pred):
            print(f"    -> {nm(c):<12} {int((pred == c).sum()):,}")
    else:
        pred = conf = None

    # assemble final labels: known kept; unclassified filled with prediction (low-conf -> 0)
    final = sem.copy()
    low = None
    if unl.sum():
        filled = pred.copy()
        low = conf < cfg.conf_thresh
        filled[low] = 0                       # low-confidence -> leave unclassified
        final[unl] = filled

    # KNN majority smoothing of the *predicted* points only (known labels stay fixed) -> kills speckle
    if cfg.smooth_k and unl.sum():
        upd = unl & (final > 0)               # only confidently-predicted points get smoothed
        if upd.any():
            _, nb = tree.query(xyz[upd], k=cfg.smooth_k)
            uniq, dense = np.unique(final, return_inverse=True)      # compact ids -> no blowup for sparse ids
            counts = (dense[nb][:, :, None] == np.arange(len(uniq))).sum(1)
            final[upd] = uniq[counts.argmax(1)]
            print(f"  smoothed {int(upd.sum()):,} predicted points (KNN k={cfg.smooth_k})")

    save_ply(os.path.join(cfg.out, "pred_types.ply"), xo, palette[final % len(palette)])
    if unl.sum():
        cvec = np.full(len(xyz), 1.0); cvec[unl] = conf
        cmap = np.stack([1 - cvec, cvec, np.zeros_like(cvec)], 1)   # red=low, green=high
        save_ply(os.path.join(cfg.out, "pred_confidence.ply"), xo, cmap)
        # NEW: only the newly-classified points; low-confidence rendered black
        new_col = palette[final[unl] % len(palette)].copy()
        new_col[final[unl] == 0] = 0.0        # low-conf / abstained -> black
        save_ply(os.path.join(cfg.out, "pred_new_points.ply"), xo[unl], new_col)
    print(f"\nwrote {cfg.out}/pred_types.ply, pred_confidence.ply, pred_new_points.ply")

    # ---- instances: DBSCAN within each predicted class (objects are discovered, not from labels) ----
    print("\n=== INSTANCES (DBSCAN per class) ===")
    inst_global = np.full(len(xyz), -1, np.int64)   # -1 = noise/unassigned
    next_id = 0; total_noise = 0
    for c in np.unique(final[final > 0]):
        ci = np.where(final == c)[0]
        loc = dbscan_instances(xyz[ci], cfg.dbscan_eps, cfg.dbscan_min, cfg.dbscan_cap, rng)
        n_obj = int(loc.max()) + 1 if loc.max() >= 0 else 0
        noise = int((loc < 0).sum()); total_noise += noise
        valid = loc >= 0
        inst_global[ci[valid]] = next_id + loc[valid]
        next_id += n_obj
        print(f"    {nm(c):<12} objects found: {n_obj:>4}   (noise pts {noise:,})")
    print(f"  total objects found: {next_id:,}   total noise pts: {total_noise:,}")

    # colour each object distinctly; noise -> dark grey
    rngc = np.random.default_rng(7)
    colors = np.full((len(xyz), 3), 0.15, np.float32)
    if next_id > 0:
        pal = rngc.random((next_id, 3))
        assigned = inst_global >= 0
        colors[assigned] = pal[inst_global[assigned]]
    save_ply(os.path.join(cfg.out, "pred_instances.ply"), xo, colors)
    print(f"  wrote {cfg.out}/pred_instances.ply")


def label_new_cloud(src_dir, xyz, F, cfg, tree, rng, xyz_out=None):
    """Label a NEW cloud using a previous run's frozen SSL encoder + trained classifier.
    No training happens here: the classifier is only valid on embeddings from the SAME
    encoder (and PCA) it was trained with, so both are loaded from src_dir/cache/."""
    import pickle
    mp = os.path.join(src_dir, "cache", "model.pt")
    bp = os.path.join(src_dir, "cache", "classifier.pkl")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"{mp} not found -- no trained SSL model in {src_dir}")
    if not os.path.exists(bp):
        raise FileNotFoundError(
            f"{bp} not found (model.pt exists) -- the training run predates classifier-bundle "
            f"saving or never reached classification; rerun the training command once "
            f"(cached model.pt/features.npz make it fast)")
    bundle = pickle.load(open(bp, "rb"))
    if bundle["in_feat"] != F.shape[1]:
        trained_d = bundle["in_feat"]
        raise ValueError(f"feature width mismatch: classifier was trained on {trained_d}d "
                         f"features but this cloud produced {F.shape[1]}d "
                         f"(colour availability must match the training run)")
    ck = torch.load(mp, map_location=cfg.device)
    model = MaskedPointAE(ck["in_feat"], ck["dim"], ck["depth"], ck["heads"])
    model.load_state_dict(ck["state"]); model.to(cfg.device)
    print(f"**** INFO: labelling new cloud with frozen SSL + classifier from {src_dir} ****")

    # embed with the SOURCE encoder (cached under this run\'s out dir)
    feat_path = os.path.join(cfg.out, "cache", "features.npz")
    os.makedirs(os.path.dirname(feat_path), exist_ok=True)
    feat = None
    if os.path.exists(feat_path) and not cfg.force_retrain:
        try:
            z = np.load(feat_path); feat = z["feat"]
            assert feat.shape[0] == len(xyz)
            print(f"[cache] loaded features {feat_path} feat{feat.shape}")
        except Exception:
            feat = None
    if feat is None:
        print("embedding new cloud (dense inference) ...", flush=True)
        feat, _ = dense_embed(model, xyz, F, cfg, tree, rng)
        np.savez(feat_path, feat=feat, F=F)
        print(f"[cache] saved features -> {feat_path}")

    if bundle["pca"] is not None:
        feat = bundle["pca"].transform(feat).astype(np.float32)     # SOURCE-fitted PCA
    feat = ((feat - feat.mean(0)) / (feat.std(0) + 1e-6)).astype(np.float32)  # THIS cloud's moments
    feat = np.concatenate([feat, height_features(xyz, tree=tree)], 1)  # same invariant height features

    xo = xyz if xyz_out is None else xyz_out
    sp_tr = bundle.get("spacing")
    if sp_tr:
        sp = median_spacing(xyz, tree)
        r = sp / sp_tr
        if r > 1.4 or r < 1 / 1.4:
            print(f"**** WARNING: point spacing {sp:.3f} m vs {sp_tr:.3f} m at training "
                  f"({r:.2f}x) -- densities differ; features shift. Re-voxel both clouds to the "
                  f"same --voxel for reliable transfer ****")
    clf, palette, names = bundle["clf"], bundle["palette"], bundle["names"]
    nm = lambda c: names.get(int(c), str(int(c)))
    proba = clf.predict_proba(feat)
    pred = clf.classes_[proba.argmax(1)].astype(np.int64)
    conf = proba.max(1)
    final = pred.copy(); final[conf < cfg.conf_thresh] = 0

    if cfg.smooth_k:
        keep = final > 0
        if keep.any():
            _, nb = tree.query(xyz[keep], k=cfg.smooth_k)
            uniq, dense = np.unique(final, return_inverse=True)
            counts = (dense[nb][:, :, None] == np.arange(len(uniq))).sum(1)
            final[keep] = uniq[counts.argmax(1)]

    print(f"\n=== NEW-CLOUD CLASSIFICATION ({len(xyz):,} pts) ===")
    for lo, hi in [(0.9, 1.01), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5)]:
        mm = (conf >= lo) & (conf < hi)
        print(f"    confidence [{lo:.1f},{hi if hi <= 1 else 1.0:.1f}): {int(mm.sum()):,}")
    print(f"    mean conf {conf.mean():.3f}  median {np.median(conf):.3f}  "
          f"abstained (<{cfg.conf_thresh}): {int((final == 0).sum()):,}")
    for c in np.unique(final[final > 0]):
        print(f"    -> {nm(c):<12} {int((final == c).sum()):,}")

    save_ply(os.path.join(cfg.out, "pred_types.ply"), xo, palette[final % len(palette)])
    cmap = np.stack([1 - conf, conf, np.zeros_like(conf)], 1)
    save_ply(os.path.join(cfg.out, "pred_confidence.ply"), xo, cmap)
    print(f"\nwrote {cfg.out}/pred_types.ply and pred_confidence.ply")


def dbscan_instances(xyzc, eps, min_samples, cap, rng):
    """Cluster one class's points into objects. Above `cap` points, fit DBSCAN on a
    subsample then propagate to all points by nearest neighbour (keeps it tractable)."""
    from sklearn.cluster import DBSCAN
    n = len(xyzc)
    if n == 0:
        return np.empty(0, np.int64)
    if n > cap:
        sub = rng.choice(n, cap, replace=False)
        lab_sub = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyzc[sub])
        _, nn = cKDTree(xyzc[sub]).query(xyzc, k=1)
        return lab_sub[nn]
    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(xyzc)


# --------------------------------------------------------------------------- #
# Blocks: KNN crops (training) and disjoint voxel-cell crops (evaluation)
# --------------------------------------------------------------------------- #
def knn_blocks(xyz, K, n_blocks, rng, tree):
    centers = rng.choice(len(xyz), size=n_blocks, replace=False)
    _, idx = tree.query(xyz[centers], k=min(K, len(xyz)))
    return idx  # [n_blocks, K] global indices


def disjoint_blocks(xyz, K, n_blocks, rng, tree):
    """Centers far apart so evaluated points barely overlap -> clean held-out score."""
    span = xyz.max(0) - xyz.min(0)
    step = span / max(1, int(round(n_blocks ** (1/3))))
    grid = np.floor((xyz - xyz.min(0)) / np.maximum(step, 1e-6)).astype(np.int64)
    _, cell_rep = np.unique(grid, axis=0, return_index=True)
    rng.shuffle(cell_rep)
    centers = cell_rep[:n_blocks]
    _, idx = tree.query(xyz[centers], k=min(K, len(xyz)))
    return idx




# --------------------------------------------------------------------------- #
# Model: tiny masked-point autoencoder
# --------------------------------------------------------------------------- #
class MaskedPointAE(nn.Module):
    def __init__(self, in_feat, dim, depth, heads):
        super().__init__()
        self.tok = nn.Sequential(nn.Linear(in_feat, dim), nn.GELU(), nn.Linear(dim, dim))  # feat -> token
        self.pos = nn.Linear(3, dim)                                                 # xyz -> pos emb
        self.mask_token = nn.Parameter(torch.zeros(dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 2, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, depth)
        self.head_feat = nn.Linear(dim, in_feat)  # reconstruct masked feature (rgb and/or geom)
        self.head_xyz = nn.Linear(dim, 3)         # denoise coordinates (geometry)
        nn.init.normal_(self.mask_token, std=0.02)

    def encode(self, xyz, feat):
        return self.enc(self.tok(feat) + self.pos(xyz))  # clean features for probing/clustering

    def forward(self, xyz_noisy, feat, mask):  # mask: [B,K] bool, True = feature hidden
        tokens = self.tok(feat)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        h = self.enc(tokens + self.pos(xyz_noisy))
        return self.head_feat(h), self.head_xyz(h)


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def run_epoch(model, blocks, xyz_t, rgb_t, cfg, groups, opt=None, gen=None):
    """One pass over `blocks`. Train if opt given, else eval.
    Returns dict of per-group feature MSE (e.g. {'rgb':.., 'geom':..}) plus 'xyz'."""
    dev = cfg.device
    is_train = opt is not None
    model.train(is_train)
    order = gen.permutation(len(blocks)) if is_train else np.arange(len(blocks))
    sums = {g[0]: 0.0 for g in groups}; sums["xyz"] = 0.0; nb = 0
    with torch.set_grad_enabled(is_train):
        for i in range(0, len(blocks), cfg.batch):
            sel = order[i:i + cfg.batch]
            bi = torch.from_numpy(blocks[sel])
            xb = xyz_t[bi].to(dev); cb = rgb_t[bi].to(dev)            # [b,K,F]
            xb = (xb - xb.mean(1, keepdim=True)) / cfg.block_scale    # FIXED metric scale: meters

            if is_train:                                             # random corruption
                th = torch.rand(xb.shape[0], device=dev) * 2 * torch.pi   # random z-rotation per block
                c_, s_ = torch.cos(th), torch.sin(th)                     # -> orientation-invariant SSL
                R = torch.stack([torch.stack([c_, -s_], -1), torch.stack([s_, c_], -1)], -2)
                xb = torch.cat([torch.einsum("bij,bkj->bki", R, xb[:, :, :2]), xb[:, :, 2:]], -1)
                mask = torch.rand(xb.shape[:2], device=dev) < cfg.mask_ratio
                noise = torch.randn_like(xb) * cfg.xyz_noise
            else:                                                    # fixed -> comparable val curve
                tg = torch.Generator(device=dev).manual_seed(cfg.seed + i)
                mask = torch.rand(xb.shape[:2], device=dev, generator=tg) < cfg.mask_ratio
                noise = torch.randn(xb.shape, device=dev, generator=tg) * cfg.xyz_noise
            mask[:, 0] = False
            pred_feat, pred_xyz = model(xb + noise, cb, mask)
            feat_err = (pred_feat - cb) ** 2                          # [b,K,F]
            loss_feat = feat_err[mask].mean()                        # all feature dims (backprop)
            loss_xyz = ((pred_xyz - xb) ** 2).mean()                 # denoise all coords
            if is_train:
                opt.zero_grad(); (loss_feat + cfg.w_xyz * loss_xyz).backward(); opt.step()
            for name, s, e in groups:                                # per-group logging
                sums[name] += feat_err[:, :, s:e][mask].mean().item()
            sums["xyz"] += loss_xyz.item(); nb += 1
    return {k: v / max(1, nb) for k, v in sums.items()}


def train(model, train_blocks, xa, ca, eval_blocks, xe, ce, cfg, groups):
    model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    xa_t, ca_t = torch.from_numpy(xa), torch.from_numpy(ca)
    xe_t, ce_t = torch.from_numpy(xe), torch.from_numpy(ce)
    gen = np.random.default_rng(cfg.seed)
    gnames = [g[0] for g in groups] + ["xyz"]
    t0 = time.time()
    hdr_tr = " ".join(f"{'tr_'+n:>8}" for n in gnames)
    hdr_va = " ".join(f"{'val_'+n:>8}" for n in gnames)
    print(f"  {'ep':>3} | {hdr_tr} | {hdr_va}")
    for ep in range(cfg.epochs):
        tr = run_epoch(model, train_blocks, xa_t, ca_t, cfg, groups, opt=opt, gen=gen)
        va = run_epoch(model, eval_blocks,  xe_t, ce_t, cfg, groups)        # opt=None -> val
        row_tr = " ".join(f"{tr[n]:8.4f}" for n in gnames)
        row_va = " ".join(f"{va[n]:8.4f}" for n in gnames)
        print(f"  {ep:3d} | {row_tr} | {row_va}")
        if (time.time() - t0) / 60 > cfg.max_minutes:
            print(f"  [time budget {cfg.max_minutes} min reached]"); break
    print(f"  trained {time.time()-t0:.1f}s")


# --------------------------------------------------------------------------- #
# Evaluate representation quality on held-out scene
# --------------------------------------------------------------------------- #


@torch.no_grad()
def dense_embed(model, xyz, rgb, cfg, tree, rng):
    """Embed (almost) every point by tiling space with overlapping KNN blocks, so
    the saved cloud covers the whole region instead of scattered sample balls."""
    model.eval(); dev = cfg.device
    N = len(xyz)
    s = rng.choice(N, min(N, 500), replace=False)
    dist, _ = tree.query(xyz[s], k=min(cfg.K, N))
    radius = float(np.median(dist[:, -1])) + 1e-6
    step = max(radius * 0.7, 1e-6)                            # spacing < radius -> overlap
    cells = np.floor((xyz - xyz.min(0)) / step).astype(np.int64)
    _, rep = np.unique(cells, axis=0, return_index=True)      # one center per occupied cell
    if len(rep) > cfg.max_eval_centers:
        rep = rng.choice(rep, cfg.max_eval_centers, replace=False)
    _, idx = tree.query(xyz[rep], k=min(cfg.K, N))
    if idx.ndim == 1:
        idx = idx[None, :]
    xyz_t = torch.from_numpy(xyz); rgb_t = torch.from_numpy(rgb)
    feat = np.zeros((N, cfg.dim), np.float32); seen = np.zeros(N, bool)
    for i in range(0, len(idx), cfg.batch):
        bidx = idx[i:i + cfg.batch]; bi = torch.from_numpy(bidx)
        xb = xyz_t[bi].to(dev); cb = rgb_t[bi].to(dev)
        xb = (xb - xb.mean(1, keepdim=True)) / cfg.block_scale        # FIXED metric scale: meters
        f = model.encode(xb, cb).cpu().numpy()
        for b in range(len(bidx)):
            g = bidx[b]; new = ~seen[g]
            feat[g[new]] = f[b][new]; seen[g[new]] = True
    if (~seen).any():                                         # nearest-fill leftovers
        miss = np.where(~seen)[0]; idxs = np.where(seen)[0]
        _, nn = cKDTree(xyz[idxs]).query(xyz[miss], k=1)
        feat[miss] = feat[idxs[nn]]; seen[miss] = True
    return feat, seen


def feature_health(feat):
    """Quick collapse check: top-PCA explained variance and per-dim std.
    If PC1 explains ~all variance, features are ~1D -> collapsed/uninformative."""
    f = feat - feat.mean(0)
    ev = PCA(min(10, feat.shape[1])).fit(f).explained_variance_ratio_
    print(f"  feature health: mean|std|={np.abs(feat.std(0)).mean():.3f}  "
          f"PC1={ev[0]:.2f} PC1-3={ev[:3].sum():.2f}  "
          f"({'likely collapsed' if ev[0] > 0.9 else 'looks distributed'})")


def score(name, X, labels, k, seed):
    km = KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    if labels is None:
        return km, None
    ari = adjusted_rand_score(labels, km)
    print(f"  {name:22s} k-means ARI: {ari:.3f}")
    return km, ari




def save_ply(path, xyz, rgb01):
    rgb = (np.clip(rgb01, 0, 1) * 255).astype(np.uint8)
    header = ("ply\nformat ascii 1.0\nelement vertex %d\n"
              "property float x\nproperty float y\nproperty float z\n"
              "property uchar red\nproperty uchar green\nproperty uchar blue\n"
              "end_header\n" % len(xyz))
    rows = np.column_stack([xyz.astype(np.float32), rgb.astype(np.int32)])
    with open(path, "w") as f:
        f.write(header)
        np.savetxt(f, rows, fmt="%.4f %.4f %.4f %d %d %d")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    for k, v in vars(cfg).items():
        if isinstance(v, bool):
            ap.add_argument(f"--{k}", action="store_true", default=v)
        else:
            ap.add_argument(f"--{k}", type=type(v), default=v)
    ap.add_argument("--smoke", action="store_true", help="tiny/fast config for a CPU sanity run")
    a = ap.parse_args()
    cfg = Config(**{k: getattr(a, k) for k in vars(cfg)})
    validate_inputs(cfg)                                 # fail fast on missing input files
    if a.smoke:
        cfg.n_blocks, cfg.n_eval_blocks, cfg.K = 48, 24, 512
        cfg.dim, cfg.depth, cfg.epochs, cfg.batch = 96, 2, 6, 8
    os.makedirs(cfg.out, exist_ok=True)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ---- load the cloud: xyz, rgb (from --data or --color_ply), semantic labels ----
    if cfg.data:
        X, C, S = load_points(cfg.data, cfg.voxel, rng, cfg.sem_field)
    else:
        X, C, S = synthetic_scene(cfg.seed)
    X, C, S = voxel_downsample(X, C, S, cfg.voxel)
    X, C, S = cap_points(X, C, S, cfg.max_points, rng)
    def info(msg):  print(f"**** INFO: {msg} ****")
    def warn(msg):  print(f"**** WARNING: {msg} ****")

    # ---- gravity alignment: estimate up-axis from slab normals; work in the aligned frame
    # internally (descriptor verticality, slab z-modes, SSL coords) but write ALL outputs in
    # the original input frame so they overlay the source clouds ----
    u, Rg, tilt = estimate_up(X, cKDTree(X))
    if tilt >= 2.0:
        info(f"cloud tilted {tilt:.1f} deg from gravity (up = {np.round(u, 3)}); "
             f"aligning internally, outputs stay in the input frame")
    Xa = (X @ Rg.T).astype(np.float32)
    tree_full = cKDTree(Xa)

    # ---- LABELS: semantic_id field, else (or if forced) decode from --data vertex colours ----
    # ---- INFERENCE MODE: label a new cloud with a previous run's SSL + classifier ----
    if cfg.run_classifier:
        rgb = transfer_rgb(X, cfg.color_ply, rng) if cfg.color_ply else C
        parts = []
        if rgb is not None:  # identical standardisation to training-time build_F
            parts.append(((rgb - rgb.mean(0)) / (rgb.std(0) + 1e-6) * 0.25 + 0.5).astype(np.float32))
        parts.append(geom_descriptor(Xa, tree_full))
        F = np.concatenate(parts, 1).astype(np.float32)
        label_new_cloud(cfg.run_classifier, Xa, F, cfg, tree_full, rng, xyz_out=X)
        return

    use_color_labels = cfg.labels_from_colors or (S is None)
    class_palette, names = None, {}
    if use_color_labels:
        if C is None:
            raise ValueError("no labels: --data has no semantic_id and no vertex colours to decode")
        S, class_palette, n_raw = labels_from_color(C, cfg.color_min_count)
        if n_raw > cfg.max_color_classes:
            raise ValueError(f"--data has {n_raw} distinct colours (> {cfg.max_color_classes}): "
                             f"looks like a scan, not class labels. Provide labels via semantic_id, "
                             f"a class-coloured file, or raise --max_color_classes.")
        n_cls = len(class_palette) - 1
        if n_cls == 0:
            raise ValueError(f"no colour had >= {cfg.color_min_count} points (--color_min_count); "
                             f"nothing to label.")
        names = names_for_colors(cfg.names_json, class_palette)
        C = None                                     # --data colours are labels, not features
        (info if cfg.labels_from_colors else warn)(
            f"labels decoded from --data vertex colours: {n_cls} classes, "
            f"{int((S > 0).sum()):,} labelled / {int((S == 0).sum()):,} unlabelled (0,0,0)")
    else:
        names = load_names(cfg.names_json)
        info("labels taken from --data semantic_id field")

    # ---- COLOUR for SSL: --color_ply, else --data colours if they're not the labels, else none ----
    if cfg.color_ply:
        rgb_src, has_rgb = "color_ply", True; info("SSL learns geometry + scan colour from --color_ply")
    elif C is not None:
        rgb_src, has_rgb = "data", True;      info("SSL learns geometry + scan colour from --data")
    else:
        rgb_src, has_rgb = None, False;       warn("no scan colour available: SSL learns geometry only")

    # ---- feature layout (geom always; rgb if available) ----
    groups, off, kinds = [], 0, []
    if has_rgb:
        groups.append(("rgb", off, off + 3)); off += 3; kinds.append("rgb")
    groups.append(("geom", off, off + GEOM_D)); off += GEOM_D
    kinds.append(f"geom[{GEOM_D}d](nvert,dvert,planar,linear,curv x{len(GEOM_KS)} scales)")
    in_feat = off
    feat_kind = "+".join(kinds)

    def build_F():                                   # rgb + geometry descriptor (the slow part)
        rgb = transfer_rgb(X, cfg.color_ply, rng) if rgb_src == "color_ply" else (C if rgb_src == "data" else None)
        parts = []
        if has_rgb:  # per-cloud channel standardisation -> robust to exposure/white-balance shifts
            parts.append(((rgb - rgb.mean(0)) / (rgb.std(0) + 1e-6) * 0.25 + 0.5).astype(np.float32))
        parts.append(geom_descriptor(Xa, tree_full))
        Fb = np.concatenate(parts, 1).astype(np.float32)
        viz = rgb if rgb is not None else Fb[:, :3]
        save_ply(os.path.join(cfg.out, "subsampled.ply"), X, viz)
        print(f"wrote {cfg.out}/subsampled.ply ({len(X):,} pts) | feature[{Fb.shape[1]}d]={feat_kind}")
        return Fb

    # ---- cache: model.pt gates SSL training; features.npz gates dense embedding ----
    cache = os.path.join(cfg.out, "cache"); os.makedirs(cache, exist_ok=True)
    model_path = os.path.join(cache, "model.pt")
    feat_path  = os.path.join(cache, "features.npz")
    force = cfg.force_retrain

    model = MaskedPointAE(in_feat, cfg.dim, cfg.depth, cfg.heads)
    F = None; retrained = False; have_model = False
    if os.path.exists(model_path) and not force:
        try:
            ck = torch.load(model_path, map_location=cfg.device)
            assert ck["in_feat"] == in_feat, f"feature width changed ({ck['in_feat']}!={in_feat})"
            assert ck.get("block_scale") == cfg.block_scale, \
                f"block_scale changed ({ck.get('block_scale')}!={cfg.block_scale}) -- pre-fixed-scale cache"
            model.load_state_dict(ck["state"]); have_model = True
            print(f"[cache] loaded SSL model {model_path} (in_feat={in_feat}) -- skipping training")
        except Exception as e:
            print(f"[cache] cannot use {model_path} ({e}); retraining")
    model.to(cfg.device)

    def prep_extra(path):                            # unlabelled cloud -> (aligned coords, features)
        Xe_, Ce_, _ = load_points(path, cfg.voxel, rng)
        Xe_, Ce_, _ = voxel_downsample(Xe_, Ce_, None, cfg.voxel)
        Xe_, Ce_, _ = cap_points(Xe_, Ce_, None, cfg.max_points, rng)
        _, Re_, _ = estimate_up(Xe_, cKDTree(Xe_))
        Xea = (Xe_ @ Re_.T).astype(np.float32)
        p_ = []
        if has_rgb:
            if Ce_ is None:
                warn(f"extra_data {path} has no RGB but this run uses rgb features -- skipping it")
                return None, None
            p_.append(((Ce_ - Ce_.mean(0)) / (Ce_.std(0) + 1e-6) * 0.25 + 0.5).astype(np.float32))
        p_.append(geom_descriptor(Xea, cKDTree(Xea)))
        return Xea, np.concatenate(p_, 1).astype(np.float32)

    if not have_model:                               # ---- SSL TRAINING ----
        F = build_F()
        ax = int(np.argmax(Xa.max(0) - Xa.min(0))); thr = np.median(Xa[:, ax]); m = Xa[:, ax] <= thr
        xs, fs = [Xa[m]], [F[m]]                     # main cloud train half...
        for p in [q.strip() for q in cfg.extra_data.split(",") if q.strip()]:
            Xe_, Fe_ = prep_extra(p)                 # ...plus UNLABELLED clouds: SSL needs no labels,
            if Xe_ is not None:                      # so the target cloud can join pretraining
                info(f"extra_data joins SSL pretraining: {os.path.basename(p)} ({len(Xe_):,} pts, unlabelled)")
                xs.append(Xe_); fs.append(Fe_)
        # main cloud keeps its FULL block budget (no dilution); extras add their own on top
        nb_extra = max(8, cfg.n_blocks // max(1, len(xs) - 1)) if len(xs) > 1 else 0
        blocks, off = [], 0
        for i, xp in enumerate(xs):                  # KNN blocks per cloud (no cross-cloud mixing)
            nb = cfg.n_blocks if i == 0 else nb_extra
            blocks.append(knn_blocks(xp, cfg.K, min(nb, len(xp)), rng, cKDTree(xp)) + off)
            off += len(xp)
        train_blocks = np.vstack(blocks)
        Xtr = np.vstack(xs).astype(np.float32); Ftr = np.vstack(fs).astype(np.float32)
        eval_blocks = disjoint_blocks(Xa[~m], cfg.K, cfg.n_eval_blocks, rng, cKDTree(Xa[~m]))
        print(f"\n=== SSL TRAINING === | feat: reconstruct masked {feat_kind} | xyz: denoise coordinates")
        train(model, train_blocks, Xtr, Ftr, eval_blocks, Xa[~m], F[~m], cfg, groups)
        torch.save({"state": model.state_dict(), "in_feat": in_feat, "block_scale": cfg.block_scale,
                    "dim": cfg.dim, "depth": cfg.depth, "heads": cfg.heads}, model_path)
        print(f"[cache] saved SSL model -> {model_path}")
        retrained = True

    feat = None                                      # ---- dense embedding (features.npz) ----
    if os.path.exists(feat_path) and not force and not retrained:
        try:
            z = np.load(feat_path); feat, F = z["feat"], z["F"]
            assert feat.shape[0] == len(X) and F.shape[1] == in_feat
            print(f"[cache] loaded features {feat_path} feat{feat.shape} -- skipping embedding")
        except Exception as e:
            print(f"[cache] cannot use {feat_path} ({e}); re-embedding"); feat = None
    if feat is None:
        if F is None: F = build_F()
        print("\nembedding full cloud (dense inference) ...", flush=True)
        feat, _ = dense_embed(model, Xa, F, cfg, tree_full, rng)
        np.savez(feat_path, feat=feat, F=F)
        print(f"[cache] saved features -> {feat_path} feat{feat.shape}")

    # ---- downstream (always runs: cheap and what you iterate on) ----
    if S is not None:                                # labelled cloud -> supervised classification
        if class_palette is not None:
            palette = class_palette                  # colours decoded from --data
        elif cfg.palette_ply:
            palette = derive_palette(cfg.palette_ply, cfg.sem_field, rng)
        else:
            palette = SEM_PALETTE
        classify_and_report(model, Xa, F, S, names, cfg, tree_full, rng, palette, feat=feat, xyz_out=X)
        return

    # ---- no labels: unsupervised representation check on a held-out half ----
    print("\nrepresentation quality on HELD-OUT scene:")
    ax = int(np.argmax(Xa.max(0) - Xa.min(0))); thr = np.median(Xa[:, ax]); m = Xa[:, ax] <= thr
    xe, ce = Xa[~m], F[~m]; xe_out = X[~m]; tree_e = cKDTree(xe)
    fe, seen = dense_embed(model, xe, ce, cfg, tree_e, rng)
    m2 = seen
    feature_health(fe[m2])
    raw = np.concatenate([(xe[m2]-xe[m2].mean(0))/(xe[m2].std(0)+1e-6), ce[m2]], 1)
    score("raw geom+feat", raw,    None, cfg.n_clusters, cfg.seed)
    km, _ = score("SSL features", fe[m2], None, cfg.n_clusters, cfg.seed)
    pca = PCA(3).fit_transform(fe[m2]); pca = (pca - pca.min(0)) / (np.ptp(pca, axis=0) + 1e-6)
    save_ply(os.path.join(cfg.out, "eval_pca_rgb.ply"), xe_out[m2], pca)
    save_ply(os.path.join(cfg.out, "eval_clusters.ply"), xe_out[m2], SEM_PALETTE[km % len(SEM_PALETTE)])
    print(f"\nwrote {cfg.out}/eval_pca_rgb.ply and eval_clusters.ply")


if __name__ == "__main__":
    main()
