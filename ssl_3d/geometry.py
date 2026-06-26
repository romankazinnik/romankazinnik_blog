"""Shared geometry: load, canonicalize, OBB fit, the ONE feature extractor, OBB utils."""
import numpy as np
import open3d as o3d


def load_e57(path):
    """-> full cloud (o3d, hi-res with color). Driver derives working-res from point spacing.
    Color is normalized by its ACTUAL range (E57 exporters vary: 0-255, 0-65535, or 0-1)."""
    import pye57
    e = pye57.E57(path); clouds = []
    for i in range(e.scan_count):
        # colors=True,intensity=True is the call that actually returns color on these files;
        # fall back to raw vectors, then to a plain read, if a pye57 version rejects the flags.
        try:
            d = e.read_scan(i, colors=True, intensity=True, ignore_missing_fields=True)
        except TypeError:
            try:
                d = e.read_scan_raw(i)
            except Exception:
                d = e.read_scan(i, ignore_missing_fields=True)
        p = np.column_stack([np.asarray(d["cartesianX"]), np.asarray(d["cartesianY"]),
                             np.asarray(d["cartesianZ"])])
        m = np.any(p != 0, 1) & ~np.isnan(p).any(1)
        p = p[m]
        pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(p))
        if all(k in d for k in ("colorRed", "colorGreen", "colorBlue")):
            c = np.column_stack([np.asarray(d["colorRed"], np.float64),
                                 np.asarray(d["colorGreen"], np.float64),
                                 np.asarray(d["colorBlue"], np.float64)])[m]
            if c.size and c.max() > 0:           # only attach if color actually present
                cmax = c.max()
                if cmax > 1.5:                   # 0-255 or 0-65535 -> scale to [0,1]
                    c = c / (65535.0 if cmax > 255 else 255.0)
                pc.colors = o3d.utility.Vector3dVector(np.clip(c, 0.0, 1.0))
        clouds.append(pc)
    full = clouds[0]
    for c in clouds[1:]:
        full += c
    print(f"load_e57: {len(full.points)} pts | has_colors={full.has_colors()}")
    return full


def estimate_resolution(xyz, sample=200000):
    """Working/raster resolution from the data: median nearest-neighbour spacing (density-
    corrected sample->full), set to ~2x spacing. CLAMPED to [0.02,0.10] m (architectural,
    metres) so an odd cloud can't produce an absurd value. The clamp is a stated assumption."""
    from scipy.spatial import cKDTree
    n = len(xyz); m = min(sample, n)
    s = xyz[np.random.default_rng(0).choice(n, m, replace=False)]
    d, _ = cKDTree(s).query(s, k=2)
    med = float(np.median(d[:, 1])) * (m / n) ** (1.0 / 3.0)   # sample->full density correction
    return float(np.clip(round(med * 2, 3), 0.02, 0.10))


def synthetic():
    """Test scene: slabs, perimeter+interior walls, 3x4 columns, joists."""
    rng = np.random.default_rng(0); W, L, H = 18., 24., 4.; P = []
    for z in (0., H):
        P.append(np.column_stack([rng.uniform(0, W, 25000), rng.uniform(0, L, 25000),
                                  np.full(25000, z) + rng.normal(0, .01, 25000)]))
    for x0, y0, x1, y1 in [(0, 0, W, 0), (0, L, W, L), (0, 0, 0, L), (W, 0, W, L), (9, 0, 9, 12)]:
        t = rng.uniform(0, 1, 12000)
        P.append(np.column_stack([x0 + t*(x1-x0) + rng.normal(0, .02, 12000),
                                  y0 + t*(y1-y0) + rng.normal(0, .02, 12000),
                                  rng.uniform(0, H, 12000)]))
    for cx in np.linspace(3, W-3, 3):
        for cy in np.linspace(3, L-3, 4):
            P.append(np.column_stack([cx + rng.uniform(-.2, .2, 3000),
                                      cy + rng.uniform(-.2, .2, 3000), rng.uniform(0, H, 3000)]))
    for by in np.linspace(2, L-2, 8):
        P.append(np.column_stack([rng.uniform(0, W, 4000), by + rng.normal(0, .08, 4000),
                                  np.full(4000, H-0.3) + rng.normal(0, .03, 4000)]))
    xyz = np.concatenate(P)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pc.colors = o3d.utility.Vector3dVector(np.tile([0.6, 0.6, 0.65], (len(xyz), 1)))
    return xyz, pc



def _fwhm_hw(hist, idx, bin_h):
    """Half-width at half-max around a histogram peak -> data-driven slab/soffit band width."""
    half = 0.5 * hist[idx]
    li = ri = idx
    while li > 0 and hist[li] >= half:
        li -= 1
    while ri < len(hist) - 1 and hist[ri] >= half:
        ri += 1
    return float(np.clip(0.5 * (ri - li) * bin_h, 2 * bin_h, 1.0))


def detect_levels(z, bin_h=0.05):
    """Floor/ceiling = lowest/highest dominant Z-histogram peaks. Band half-widths from FWHM,
    so slab and (thick joist) soffit thickness come from the DATA, not a fixed +/-0.4."""
    from scipy.signal import find_peaks
    hist, edges = np.histogram(z, bins=max(int((z.max() - z.min()) / bin_h), 5))
    centers = 0.5 * (edges[:-1] + edges[1:]); bw = edges[1] - edges[0]
    pk, _ = find_peaks(hist, height=0.2 * hist.max(), distance=max(int(0.3 / bw), 1))
    if len(pk) >= 2:
        fi, ci = int(pk[0]), int(pk[-1])
        return (float(centers[fi]), float(centers[ci]),
                _fwhm_hw(hist, fi, bw), _fwhm_hw(hist, ci, bw))
    return float(np.percentile(z, 2)), float(np.percentile(z, 98)), 0.3, 0.3

def canonicalize(xyz):
    """RANSAC floor -> Z up. Returns xyz, T, floor_z, ceil_z, floor_hw, ceil_hw (data-driven)."""
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    T = np.eye(4)
    try:
        model, inl = pc.segment_plane(0.03, 3, 1000)
        n = np.array(model[:3]); n = n / np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        if abs(n[2]) < 0.85:                 # not horizontal -> RANSAC hit a wall, skip rotation
            raise RuntimeError("dominant plane is not the floor")
        v = np.cross(n, [0, 0, 1]); s = np.linalg.norm(v)
        if s > 1e-6:
            c = n[2]; vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
            R = np.eye(3) + vx + vx @ vx * ((1 - c) / s**2)
            xyz = xyz @ R.T; T[:3, :3] = R
    except Exception:
        pass
    z = xyz[:, 2]
    fz, cz, fhw, chw = detect_levels(z)
    return xyz, T, fz, cz, fhw, chw


def fit_obb(pts):
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    try:
        b = pc.get_minimal_oriented_bounding_box(robust=True)
    except Exception:
        b = pc.get_axis_aligned_bounding_box().get_oriented_bounding_box()
    return {"center": np.array(b.center, float), "extent": np.array(b.extent, float), "R": np.array(b.R, float)}


def obb_to_o3d(obb):
    return o3d.geometry.OrientedBoundingBox(
        center=np.asarray(obb["center"]), R=np.asarray(obb["R"]),
        extent=np.maximum(np.asarray(obb["extent"]), 0.02))


def points_in_obb(pts, obb):
    v = o3d.utility.Vector3dVector(pts)
    return np.asarray(obb_to_o3d(obb).get_point_indices_within_bounding_box(v), dtype=np.int64)


def extract_features(pts, obb, floor_z, ceil_z):
    """THE feature extractor. Geometric only. Computed from the points assigned to `obb`."""
    n = len(pts)
    ext = np.maximum(np.asarray(obb["extent"], float), 1e-3)
    vol = float(np.prod(ext))
    R = np.asarray(obb["R"], float); ctr = np.asarray(obb["center"], float)
    local = (pts - ctr) @ R if n else np.zeros((0, 3))
    thin = int(np.argmin(ext))
    plane_mse = float(np.mean(local[:, thin] ** 2)) if n else 0.0
    res = 0.05
    if n:
        vox = np.floor((pts - pts.min(0)) / res).astype(np.int64)
        occ = len(np.unique(vox, axis=0))
        fill = min(occ * res ** 3 / vol, 1.0)
    else:
        fill = 0.0
    if n >= 3:
        d = pts - pts.mean(0); w = np.sort(np.linalg.eigvalsh(d.T @ d / n))[::-1]
        w = np.clip(w, 1e-12, None)
        linearity = float(w[0] / w[1]); planarity = float(w[2] / w[0])
    else:
        linearity = planarity = 1.0
    if n:
        zlo, zhi = pts[:, 2].min(), pts[:, 2].max()
        if zhi - zlo > 1e-3:
            nb = 8; b = np.clip(((pts[:, 2] - zlo) / ((zhi - zlo) / nb)).astype(int), 0, nb - 1)
            vc = len(np.unique(b)) / nb
        else:
            vc = 0.0
    else:
        vc = 0.0
    ext_sorted = np.sort(ext)[::-1]
    return {"plane_mse": plane_mse, "density": n / vol, "fill_ratio": fill, "n_points": n,
            "extent_l": ext_sorted[0], "extent_w": ext_sorted[1], "extent_h": ext_sorted[2],
            "planarity": planarity, "linearity": linearity,
            "vertical_consistency": vc, "z_center_rel": float((ctr[2] - floor_z) / max(ceil_z - floor_z, 1e-3))}


def vertical_consistency_map(pts, floor_z, ceil_z, floor_hw, ceil_hw, res=0.05, nbands=6):
    """Multi-band occupancy vote -> mask of consistent verticals. Band spans the clear zone
    BETWEEN the slabs (floor_z+floor_hw .. ceil_z-ceil_hw), derived from the data."""
    lo, hi = floor_z + floor_hw, ceil_z - ceil_hw
    if hi - lo < 0.5:                       # degenerate thin storey -> small symmetric margin
        lo, hi = floor_z + 0.2, ceil_z - 0.2
    vert = pts[(pts[:, 2] > lo) & (pts[:, 2] < hi)]
    mn = vert[:, :2].min(0); dims = (np.ceil((vert[:, :2].max(0) - mn) / res).astype(int) + 1)
    votes = np.zeros(dims[::-1], np.float32)
    for z0 in np.linspace(lo, hi, nbands):
        band = vert[(vert[:, 2] >= z0) & (vert[:, 2] < z0 + (hi - lo) / nbands)]
        ij = ((band[:, :2] - mn) / res).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < dims[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < dims[1])
        ij = ij[ok]
        g = np.zeros(dims[::-1], bool); g[ij[:, 1], ij[:, 0]] = True; votes += g
    return ((votes >= 0.5 * nbands) * 255).astype(np.uint8), mn, res
