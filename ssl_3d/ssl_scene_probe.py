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
    eval_data: str = ""     # path to held-out scene; "" -> synthetic (new seed)
    color_free: bool = False   # ignore RGB; use a geometry descriptor as the SSL feature
    color_ply: str = ""        # optional .ply (same points as --data) supplying per-vertex RGB
    features: str = "auto"     # SSL feature channel: auto | rgb | geom | rgb_geom
    names_json: str = ""    # JSON with {"semantic_names": {"1":"wall", ...}}
    sem_field: str = "semantic_id"   # per-vertex label field in the .ply
    palette_ply: str = ""   # optional .ply with semantic_id + type-RGB -> match output colors to it
    inst_field: str = "instance_id"  # per-vertex instance field (read but unused)
    voxel: float = 0.0      # voxel downsample size (m); 0 = keep all points (e57 still uses 0.04)
    max_points: int = 4_000_000  # random cap after voxel (keeps KDTree/RAM sane)
    dbscan_eps: float = 0.3      # DBSCAN neighbourhood radius (m) for instance splitting
    dbscan_min: int = 30         # DBSCAN min samples per object
    dbscan_cap: int = 150_000    # per-class point cap for DBSCAN (subsample+propagate above this)
    probe_max_train: int = 50_000   # cap on points used to FIT the classifier (sklearn is CPU-only)
    probe_pca: int = 64             # PCA-reduce SSL features before the classifier (0 = off)
    head: str = "mlp"              # downstream classifier: "mlp" (2-layer NN) or "linear"
    conf_thresh: float = 0.5        # below this, a predicted point is treated as low-confidence (black)
    smooth_k: int = 12              # KNN majority smoothing of predictions (0 = off) -> reduces speckle
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


def geom_descriptor(xyz, tree, k=24, chunk=200_000):
    """Per-point local-PCA descriptor in [0,1]: [verticality, planarity, linearity].
    Self-supervised (from raw xyz, no labels), sign-invariant, and directly
    discriminative: walls vertical+planar, floor/ceiling/slab horizontal+planar, beams linear."""
    N = len(xyz); out = np.zeros((N, 3), np.float32)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        _, idx = tree.query(xyz[s:e], k=k)                  # [c,k]
        nb = xyz[idx]                                        # [c,k,3]
        nb = nb - nb.mean(1, keepdims=True)
        cov = np.einsum("ckm,ckn->cmn", nb, nb) / k         # [c,3,3]
        w, vec = np.linalg.eigh(cov)                        # ascending: w0<=w1<=w2
        l2, l1, l0 = w[:, 0], w[:, 1], w[:, 2]
        l0c = l0 + 1e-9
        out[s:e, 0] = 1.0 - np.abs(vec[:, 2, 0])           # verticality (normal = eigvec col 0)
        out[s:e, 1] = (l1 - l2) / l0c                       # planarity
        out[s:e, 2] = (l0 - l1) / l0c                       # linearity
    return out


def load_names(path):
    if not path or not os.path.exists(path):
        return {}
    import json
    d = json.load(open(path)).get("semantic_names", {})
    return {int(k): v for k, v in d.items()}


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


def classify_and_report(model, xyz, feat_chan, sem, names, cfg, tree, rng, palette):
    """Train a classifier model on labelled points (sem>0), score on a held-out labelled
    split, then predict the unclassified (sem==0) points. Compares SSL features vs raw
    geometry; reports per-class IoU, how many points were classified, and confidence."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import train_test_split

    print("\nembedding full cloud (dense inference) ...", flush=True)
    feat, seen = dense_embed(model, xyz, feat_chan, cfg, tree, rng)   # SSL features, full cloud
    feature_health(feat)
    if cfg.probe_pca and feat.shape[1] > cfg.probe_pca:
        feat = PCA(cfg.probe_pca).fit_transform(feat).astype(np.float32)  # speed up classifier
    # SSL blocks are mean-centred (height stripped) -> give the classifier global position back.
    # Without this, floor/ceiling/slab (all horizontal planes) are indistinguishable.
    gxyz = ((xyz - xyz.mean(0)) / (xyz.std(0) + 1e-6)).astype(np.float32)
    feat = np.concatenate([feat, gxyz], 1)                            # SSL features + global xyz/height
    raw = np.concatenate([gxyz, feat_chan], 1)                       # geometry baseline (same global xyz)

    lab = sem > 0; unl = ~lab
    nm = lambda c: names.get(int(c), str(int(c)))
    print("\n=== CLASSIFICATION (semantic) ===")
    print("colour legend (semantic_id -> name -> output RGB):")
    for c in np.unique(sem[lab]):
        r, g, b = (palette[int(c)] * 255).astype(int)
        print(f"    {int(c)} {nm(c):<10} rgb({r},{g},{b})")
    print(f"labelled points: {lab.sum():,} | unclassified (sem==0): {unl.sum():,}")
    for c in np.unique(sem[lab]):
        print(f"    {nm(c):<12} id={c}  {int((sem == c).sum()):,}")

    # held-out split of labelled points to gauge trustworthiness
    idx = np.where(lab)[0]
    try:
        tr, va = train_test_split(idx, test_size=0.2, random_state=cfg.seed, stratify=sem[idx])
    except ValueError:
        tr, va = train_test_split(idx, test_size=0.2, random_state=cfg.seed)

    # BALANCED fit set: equal points/class up to the cap (MLP has no class_weight; this fixes imbalance)
    per_cls = max(1, cfg.probe_max_train // len(np.unique(sem[tr])))
    keep = []
    for c in np.unique(sem[tr]):
        ci = tr[sem[tr] == c]
        keep += list(rng.choice(ci, min(per_cls, len(ci)), replace=False))
    tr = np.array(keep)
    print(f"  (classifier={cfg.head}, fit on {len(tr):,} balanced pts, eval on {len(va):,})")

    def make_clf():
        if cfg.head == "linear":
            return make_pipeline(StandardScaler(), LogisticRegression(max_iter=200, class_weight="balanced"))
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                                           early_stopping=True, n_iter_no_change=8, random_state=cfg.seed))

    def fit_eval(name, X):
        print(f"  fitting model[{name}] on {len(tr):,}x{X.shape[1]} ...", flush=True)
        clf = make_clf().fit(X[tr], sem[tr])
        pred = clf.predict(X[va])
        acc = float((pred == sem[va]).mean())
        per = {}
        for c in np.unique(sem[va]):
            inter = np.sum((pred == c) & (sem[va] == c)); union = np.sum((pred == c) | (sem[va] == c))
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
            C = int(final.max()) + 1
            counts = (final[nb][:, :, None] == np.arange(C)).sum(1)
            final[upd] = counts.argmax(1)
            print(f"  smoothed {int(upd.sum()):,} predicted points (KNN k={cfg.smooth_k})")

    save_ply(os.path.join(cfg.out, "pred_types.ply"), xyz, palette[final % len(palette)])
    if unl.sum():
        cvec = np.full(len(xyz), 1.0); cvec[unl] = conf
        cmap = np.stack([1 - cvec, cvec, np.zeros_like(cvec)], 1)   # red=low, green=high
        save_ply(os.path.join(cfg.out, "pred_confidence.ply"), xyz, cmap)
        # NEW: only the newly-classified points; low-confidence rendered black
        new_col = palette[final[unl] % len(palette)].copy()
        new_col[final[unl] == 0] = 0.0        # low-conf / abstained -> black
        save_ply(os.path.join(cfg.out, "pred_new_points.ply"), xyz[unl], new_col)
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
    save_ply(os.path.join(cfg.out, "pred_instances.ply"), xyz, colors)
    print(f"  wrote {cfg.out}/pred_instances.ply")


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


def normalize_block(xyz_b):
    c = xyz_b.mean(0, keepdims=True)
    out = xyz_b - c
    scale = np.abs(out).max() + 1e-6
    return out / scale


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
            xb = xb - xb.mean(1, keepdim=True)
            xb = xb / (xb.abs().amax(dim=(1, 2), keepdim=True) + 1e-6)
            if is_train:                                             # random corruption
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
def embed(model, blocks, xyz, rgb, cfg):
    """Per-point SSL embeddings over eval blocks, deduped by global index."""
    model.eval()
    dev = cfg.device
    xyz_t = torch.from_numpy(xyz); rgb_t = torch.from_numpy(rgb)
    feat = np.zeros((len(xyz), cfg.dim), np.float32)
    seen = np.zeros(len(xyz), bool)
    for i in range(0, len(blocks), cfg.batch):
        bidx = blocks[i:i + cfg.batch]
        bi = torch.from_numpy(bidx)
        xb = xyz_t[bi].to(dev); cb = rgb_t[bi].to(dev)
        xb = xb - xb.mean(1, keepdim=True)
        xb = xb / (xb.abs().amax(dim=(1, 2), keepdim=True) + 1e-6)
        f = model.encode(xb, cb).cpu().numpy()                   # [b,K,D]
        for b in range(len(bidx)):
            g = bidx[b]; new = ~seen[g]
            feat[g[new]] = f[b][new]; seen[g[new]] = True
    return feat, seen


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
        xb = xb - xb.mean(1, keepdim=True)
        xb = xb / (xb.abs().amax(dim=(1, 2), keepdim=True) + 1e-6)
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


def few_shot_probe(name, X, labels, seed, n_per_class=20):
    """Train a linear classifier on n_per_class labelled points/class, report mIoU
    on the rest. This is the 'small labeling effort' claim, measured directly."""
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(seed)
    train_idx = []
    for c in np.unique(labels):
        ci = np.where(labels == c)[0]
        train_idx += list(rng.choice(ci, min(n_per_class, len(ci)), replace=False))
    train_idx = np.array(train_idx)
    test_mask = np.ones(len(labels), bool); test_mask[train_idx] = False
    clf = LogisticRegression(max_iter=300)
    clf.fit(X[train_idx], labels[train_idx])
    pred = clf.predict(X[test_mask]); gt = labels[test_mask]
    ious = []
    for c in np.unique(labels):
        inter = np.sum((pred == c) & (gt == c)); union = np.sum((pred == c) | (gt == c))
        if union: ious.append(inter / union)
    miou = float(np.mean(ious))
    print(f"  {name:22s} {n_per_class}-shot linear-probe mIoU: {miou:.3f}")
    return miou


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
    tree_full = cKDTree(X)

    rgb = transfer_rgb(X, cfg.color_ply, rng) if cfg.color_ply else C   # color_ply overrides --data rgb

    # resolve which feature channel to build
    feats = "geom" if cfg.color_free else cfg.features
    if feats == "auto":
        feats = "rgb_geom" if cfg.color_ply else ("rgb" if rgb is not None else "geom")
    parts, names_dbg, groups = [], [], []
    off = 0
    if "rgb" in feats:
        if rgb is None:
            raise ValueError("features need RGB but none available -- pass --color_ply or an RGB .ply")
        parts.append(rgb); names_dbg.append("rgb"); groups.append(("rgb", off, off + 3)); off += 3
    if "geom" in feats:
        parts.append(geom_descriptor(X, tree_full)); names_dbg.append("geom(vert,planar,linear)")
        groups.append(("geom", off, off + 3)); off += 3
    F = np.concatenate(parts, 1).astype(np.float32)
    feat_kind = "+".join(names_dbg)

    viz = rgb if rgb is not None else F[:, :3]
    save_ply(os.path.join(cfg.out, "subsampled.ply"), X, viz)
    print(f"wrote {cfg.out}/subsampled.ply ({len(X):,} pts) | feature[{F.shape[1]}d]={feat_kind} | "
          f"labels={'yes' if S is not None else 'no'} | device {cfg.device}")

    # ---- spatial split (train half / val half) for SSL pretext monitoring ----
    ax = int(np.argmax(X.max(0) - X.min(0)))
    thr = np.median(X[:, ax]); m = X[:, ax] <= thr
    xa, ca = X[m], F[m]; xe, ce = X[~m], F[~m]
    tree_a = cKDTree(xa); tree_e = cKDTree(xe)
    train_blocks = knn_blocks(xa, cfg.K, cfg.n_blocks, rng, tree_a)
    eval_blocks  = disjoint_blocks(xe, cfg.K, cfg.n_eval_blocks, rng, tree_e)

    # ---- self-supervised pretext training ----
    print(f"\n=== SSL TRAINING === | feat: reconstruct masked {feat_kind} | xyz: denoise coordinates")
    model = MaskedPointAE(F.shape[1], cfg.dim, cfg.depth, cfg.heads)
    train(model, train_blocks, xa, ca, eval_blocks, xe, ce, cfg, groups)

    # ---- downstream ----
    if S is not None:                          # labelled cloud -> supervised classification
        palette = derive_palette(cfg.palette_ply, cfg.sem_field, rng) if cfg.palette_ply else SEM_PALETTE
        classify_and_report(model, X, F, S, load_names(cfg.names_json), cfg, tree_full, rng, palette)
        return

    # ---- no labels: unsupervised representation check on the held-out half ----
    print("\nrepresentation quality on HELD-OUT scene:")
    feat, seen = dense_embed(model, xe, ce, cfg, tree_e, rng)
    m2 = seen
    feature_health(feat[m2])
    raw = np.concatenate([(xe[m2]-xe[m2].mean(0))/(xe[m2].std(0)+1e-6), ce[m2]], 1)
    km_raw, _ = score("raw geom+feat", raw,     None, cfg.n_clusters, cfg.seed)
    km,     _ = score("SSL features",  feat[m2], None, cfg.n_clusters, cfg.seed)
    print("  no labels -> open eval_pca_rgb.ply / eval_clusters.ply to eyeball structure.")
    pca = PCA(3).fit_transform(feat[m2]); pca = (pca - pca.min(0)) / (np.ptp(pca, axis=0) + 1e-6)
    save_ply(os.path.join(cfg.out, "eval_pca_rgb.ply"), xe[m2], pca)
    save_ply(os.path.join(cfg.out, "eval_clusters.ply"), xe[m2], SEM_PALETTE[km % len(SEM_PALETTE)])
    print(f"\nwrote {cfg.out}/eval_pca_rgb.ply and eval_clusters.ply")


if __name__ == "__main__":
    main()
