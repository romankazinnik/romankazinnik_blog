"""Plots for tracker.csv. Usage: python plot.py [tracker.csv]  -> writes plot_h*.png

H1 calibration roofline  T_batch(B) = max(a, B*k)          from calib_table, per server config
H2/H3 greedy collapse    p99 / a_eff = 2/(1-u)              all greedy rows, colored by S
H3b latency vs S         the hyperbola, not a line          greedy rows grouped by n_params
H4 fixed window          B_mean vs lambda*T_w, p99 vs T_w + T_batch
H5 bursts                p99 with and without peak at equal base lambda
"""
import json, sys
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from client import GREEDY_P50_CYCLES, GREEDY_P99_CYCLES, TIMEOUT_SLO_MULT, slope_fit   # shared constants and the slope estimator
LAT_AXIS_MIN_MS = 1
def clipped(ax, x, y, ymax, **kw):
    """Scatter with the y axis capped at ymax; points beyond it are drawn as ▲ on the top edge."""
    x, y = np.asarray(x, float), np.asarray(y, float); over = y > ymax
    pts = ax.scatter(x[~over], y[~over], **kw)
    if over.any(): ax.scatter(x[over], np.full(over.sum(), ymax), marker="^", s=70, color=pts.get_facecolor()[0] if len(pts.get_facecolor()) else None)
    return pts
U_PLOT_MAX = 0.98                       # x-range of the 1/(1-u) curve

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "tracker.csv")
from server import derive_d, N_LAYERS, FFN_MULT    # same d(P) as the server, for the attention term
from client import FLOPS_PER_MAC, ATTN_MATMULS_PER_LAYER
df["P"] = df["n_params"].apply(lambda n: 10.0 ** round(np.log10(n)))  # size class = nearest decade (pos-embedding adds S*d)
df["label"] = df.apply(lambda r: f"P={r.P/1e6:g}M S={r.seq_len} {r.device}", axis=1)
window = df[df["mode"] == "window"]
greedy = df[(df["mode"] == "greedy") & (df["qps_peak"] == df["qps"])]      # no bursts
gpu = greedy[greedy["device"] == "cuda"]
burst = df[(df["mode"] == "greedy") & (df["qps_peak"] > df["qps"])]
def flops_req(P, S):
    d = derive_d(P); return FLOPS_PER_MAC * P * S + ATTN_MATMULS_PER_LAYER * FLOPS_PER_MAC * S * S * d * N_LAYERS
df["k_ms"] = df["calib_table"].apply(lambda s: slope_fit({int(b): v for b, v in json.loads(s).items()}))
df["F_eff_tflops"] = [flops_req(P, S) / (k * 1e-3) / 1e12 for P, S, k in zip(df["n_params"], df["seq_len"], df["k_ms"])]
df["u_pred"] = df["qps"] * df["k_ms"] / 1e3
SLO = df["slo_ms"].iloc[0]
LAT_AXIS_MAX_MS = TIMEOUT_SLO_MULT * SLO      # the client caps every latency here; nothing above it is a measurement

# H1: calibration roofline. Panel 0: S = 1 across P (floor flat in P). Panels 1..: one P each across S (knee moves with S).
H1_SEQ_LENS = [1, 10, 100, 1000]  # the roofline is shown at decade S only; intermediate S belong to H3
H1_PANELS = [1e8, "S=1", 1e7]       # left to right: a P value -> sequence-length sweep at that P; "S=1" -> model-size sweep
def median_table(g):
    """Per-bucket median over every calibration of this configuration: robust to one disturbed server start."""
    tables = [{int(k): v for k, v in json.loads(s).items()} for s in g["calib_table"]]
    return {B: float(np.median([t[B] for t in tables if B in t])) for B in sorted({B for t in tables for B in t})}
def roofline(ax, g, label, style="o-"):
    t = median_table(g); B = np.array(sorted(t)); a, k = t[1], slope_fit(t)
    line, = ax.loglog(B, [t[b] for b in B], style, label=f"{label}  a={a:.2g} ms, k={k * 1e3:.2g} µs/req")
    ax.loglog(B, a + B * k, ":", color=line.get_color(), alpha=0.6, lw=1)
fig, axes = plt.subplots(1, len(H1_PANELS), figsize=(6.5 * len(H1_PANELS), 5), sharey=True)
for ax, panel in zip(axes, H1_PANELS):
    if panel == "S=1":
        for (P, dev), g in df[df["seq_len"] == 1].groupby(["P", "device"]):
            roofline(ax, g, f"P={P/1e6:g}M {dev}", "o-" if dev == "cuda" else "s--")
        ax.set(title="S = 1: model size sweep", xlabel="batch size B")
    else:
        for (S, dev), g in df[(df["P"] == panel) & df["seq_len"].isin(H1_SEQ_LENS)].groupby(["seq_len", "device"]):
            roofline(ax, g, f"S={S} {dev}", "o-" if dev == "cuda" else "s--")
        ax.set(title=f"P = {panel/1e6:g}M: sequence length sweep", xlabel="batch size B")
for ax in axes: ax.set_ylabel("T_batch (ms)"); ax.tick_params(labelleft=True); ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("H1  T_batch(B) measured (solid/dashed) vs a + B·k fit (dotted)")
fig.tight_layout(); fig.savefig("plot_h1_roofline.png", dpi=120)

# H2/H3: collapse onto 2/(1-u)
fig, ax = plt.subplots(figsize=(7, 5))
u = np.linspace(0, U_PLOT_MAX, 200)
ax.plot(u, GREEDY_P99_CYCLES / (1 - u), "k--", label=f"p99 pred: {GREEDY_P99_CYCLES:g}/(1−u)")
ax.plot(u, GREEDY_P50_CYCLES / (1 - u), "--", color="gray", label=f"p50 pred: {GREEDY_P50_CYCLES:g}/(1−u)")
ymax = LAT_AXIS_MAX_MS / gpu["a_eff_ms"].median()
for S, g in gpu.groupby("seq_len"):
    clipped(ax, g["u_pred"], g["lat_p99_ms"] / g["a_eff_ms"], ymax, label=f"S={S}", s=40)
    ax.scatter(g["u_pred"], g["lat_p50_ms"] / g["a_eff_ms"], marker="x", color="gray", s=25)
ax.set(xlabel="predicted utilization u = λ·FLOPs_req/F", ylabel="latency / a_eff", yscale="log", ylim=(LAT_AXIS_MIN_MS, ymax * 1.3), xlim=(0, 1.3),
       title="H2/H3  greedy drain: p99 (dots) and p50 (x) in units of the floor a_eff")
ax.grid(True, alpha=0.3); ax.legend(fontsize=8); fig.savefig("plot_h3_collapse.png", dpi=120)

# H3b: p99 vs S, one panel per model size; dashed = 2*a_eff/(1-u(S)) with a_eff, F_eff taken from that panel's rows
sizes = sorted(gpu["P"].unique())
fig, axes = plt.subplots(1, len(sizes), figsize=(4.5 * len(sizes), 4.5), sharey=True, squeeze=False)
for ax, P in zip(axes[0], sizes):
    g = gpu[gpu["P"] == P]
    for qi, (q, gq) in enumerate(g.groupby("qps")):
        gq = gq.sort_values("seq_len")
        pts = [clipped(ax, gq["seq_len"], gq["lat_p99_ms"], LAT_AXIS_MAX_MS, label=f"λ={q:g}/s measured")]
        a_eff, F = gq["a_eff_ms"].median(), gq["F_eff_tflops"].median() * 1e12
        S = np.logspace(0, 3, 200); u = q * flops_req(P, S) / F
        pred = np.where(u < 1, GREEDY_P99_CYCLES * a_eff / np.maximum(1 - u, 1e-9), np.nan)
        col = pts[0].get_facecolor()[0]
        ax.plot(S, pred, "--", color=col, alpha=0.7, label=f"λ={q:g}/s  {GREEDY_P99_CYCLES:g}·a_eff/(1−u)")
        cross = S[np.nan_to_num(pred, nan=np.inf) > SLO]               # predicted largest servable S: 2·a_eff/(1−u(S)) = SLO
        if len(cross) and cross[0] > S[0]:
            ax.axvline(cross[0], color=col, ls=":", alpha=0.8)
            ax.text(cross[0], LAT_AXIS_MIN_MS * 1.3 ** (1 + 2 * qi), f" S*≈{cross[0]:.0f}", color=col, fontsize=8)
    ax.axhline(SLO, color="r", ls=":", label="SLO")
    ax.set(xscale="log", yscale="log", ylim=(LAT_AXIS_MIN_MS, LAT_AXIS_MAX_MS * 1.3), xlabel="sequence length S", title=f"P = {P/1e6:g}M")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=7)
for ax in axes[0]: ax.set_ylabel("p99 latency (ms)"); ax.tick_params(labelleft=True)
fig.suptitle("H3b  greedy drain: p99 vs S per model size — the largest S under the SLO is the servable model")
fig.tight_layout(); fig.savefig("plot_h3b_vs_S.png", dpi=120)

# H4: fixed window. Rows with u >= 1 are overloaded and shown hollow: they do not test F15.
if len(window):
    ok = window["u_pred"] < 1
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for sub, mk in ((window[ok], "o"), (window[~ok], "o")):
        ax[0].scatter(sub["B_pred"], sub["B_mean"], marker=mk, facecolors="none" if mk == "o" and sub is window[~ok] else None)
    B_inactive = np.maximum(window["B_pred"], window["qps"] * window["a_eff_ms"] / 1e3 / np.maximum(1 - window["u_pred"], 1e-9))
    ax[0].scatter(window["B_pred"], B_inactive, facecolors="none", edgecolors="k", label="pred: max(λ·T_w, λ·a_eff/(1−u))")
    lim = [0, window[["B_pred", "B_mean"]].max().max() * 1.1]; ax[0].plot(lim, lim, "k--", label="B = λ·T_w")
    ax[0].set(xlabel="λ·T_w", ylabel="B_mean measured", title="H4  B = λ·T_w (window active) or λ·a_eff/(1−u) (inactive)")
    ax[0].legend(fontsize=8)
    greedy_p99 = GREEDY_P99_CYCLES * window["a_eff_ms"] / np.maximum(1 - window["u_pred"], 1e-9)
    x = np.maximum(window["max_wait_ms"] + window["T_batch_pred_ms"], greedy_p99)   # inactive window -> greedy latency
    clipped(ax[1], x[ok], window["lat_p99_ms"][ok], LAT_AXIS_MAX_MS, label="u < 1")
    clipped(ax[1], x[~ok], window["lat_p99_ms"][~ok], LAT_AXIS_MAX_MS, facecolors="none", edgecolors="C1", label="u ≥ 1 (overloaded)")
    lim = [LAT_AXIS_MIN_MS, LAT_AXIS_MAX_MS]; ax[1].plot(lim, lim, "k--")
    ax[1].set(xscale="log", yscale="log", xlim=lim, ylim=(LAT_AXIS_MIN_MS, LAT_AXIS_MAX_MS * 1.3),
              xlabel="max(T_w + T_batch, 2·a_eff/(1−u)) (ms)", ylabel="p99 measured (ms)", title="H4  p99 ≈ T_w + T_batch, or greedy when the window is inactive")
    ax[1].legend(fontsize=8)
    for a_ in ax: a_.grid(True, which="both", alpha=0.3)
    fig.tight_layout(); fig.savefig("plot_h4_window.png", dpi=120)

# H5: each burst run next to its own no-burst baseline (same P, S, qps, device). Nothing else belongs on this plot.
KEY = ["P", "seq_len", "qps", "device"]
pairs = burst.merge(greedy[KEY + ["lat_p99_ms", "B_mean", "u_pred"]], on=KEY, suffixes=("", "_base"))
if len(pairs):
    fig, ax = plt.subplots(figsize=(max(9, 3 + 3 * len(pairs)), 5))
    x = np.arange(len(pairs)); w = 0.38
    ax.bar(x - w / 2, pairs["lat_p99_ms_base"], w, label="p99, no bursts")
    ax.bar(x + w / 2, pairs["lat_p99_ms"], w, label="p99, with bursts")
    ratio = pairs["qps_peak"] / pairs["qps"]; a_eff = pairs["a_eff_ms"]; u = pairs["u_pred_base"]
    pred_base = GREEDY_P99_CYCLES * a_eff / np.maximum(1 - u, 1e-9)
    pred_burst = np.where(ratio * u < 1, GREEDY_P99_CYCLES * a_eff / np.maximum(1 - ratio * u, 1e-9), LAT_AXIS_MAX_MS)
    ax.scatter(x - w / 2, pred_base, marker="_", s=600, color="k", zorder=3, label="pred: 2·a_eff/(1−u)")
    ax.scatter(x + w / 2, pred_burst, marker="_", s=600, color="k", zorder=3, label="pred: 2·a_eff/(1−k·u), k = peak ratio")
    for i, r in pairs.reset_index().iterrows():
        kr, ub = r.qps_peak / r.qps, r.u_pred_base
        B_ratio_pred = kr * (1 - ub) / max(1 - kr * ub, 1e-9)         # F13 at the peak rate / F13 at the base rate
        ax.text(i, max(r.lat_p99_ms, r.lat_p99_ms_base) * 1.15,
                f"u_peak = {kr * ub:.2f}\nB_max/B_mean = {r.B_max / r.B_mean_base:.0f} (F13 at peak: {B_ratio_pred:.1f})",
                ha="center", fontsize=8)
    ax.axhline(SLO, color="r", ls=":", label="SLO")
    ax.set_xticks(x); ax.set_xticklabels([f"P={r.P/1e6:g}M S={r.seq_len}\nλ={r.qps:g}/s, peak {r.qps_peak/r.qps:g}×" for _, r in pairs.iterrows()], fontsize=8)
    ax.set(yscale="log", ylabel="p99 latency (ms)", ylim=(1, max(SLO, pairs["lat_p99_ms"].max()) * 4),
           title="H5  p99 with bursts ≈ greedy fixed point at the peak rate: 2·a_eff/(1−k·u)")
    ax.grid(True, axis="y", which="both", alpha=0.3); ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig("plot_h5_bursts.png", dpi=120)

cols = ["tag", "seed", "mode", "qps", "n_params", "seq_len", "u_pred", "F_eff_tflops", "B_mean", "B_pred", "cycle_ms",
        "lat_p50_ms", "lat_p99_ms", "lat_p99_pred_ms", "send_lag_p99_ms", "n_failed", "n_mismatch"]
print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:.3g}"))
