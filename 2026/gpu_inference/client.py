"""Open-loop Poisson load generator + latency/correctness reporter for server.py.

Sends Poisson traffic at --qps for --duration-s (with --qps-peak bursts of BURST_S every PERIOD_S), measures
latency from the *scheduled* send time, verifies every response id and checksum, pulls server stats, and appends
one row of measured + predicted values to OUT_CSV. All hardware constants are measured: F comes from the slope
of the server's calibration table, a from its first entry.
"""
import argparse, asyncio, csv, json, math, os, struct, time
from datetime import datetime
import numpy as np

# ---- protocol (must match server.py) ----
REQ, RSP, CTRL = struct.Struct("<I"), struct.Struct("<Ifi"), struct.Struct("<II")
CTRL_RESET, CTRL_STATS = 0xFFFFFFFF, 0xFFFFFFFE
HOST, PORT = "127.0.0.1", 9000
TOKEN_STRIDE = 7919
# ---- experiment shape ----
WARMUP_S = 10                  # excluded from statistics; server counters reset at its end
BURST_S, PERIOD_S = 5, 30      # qps_peak for BURST_S seconds at the start of every PERIOD_S
TIMEOUT_SLO_MULT = 10          # unanswered after TIMEOUT_SLO_MULT * SLO counts as failed, at that latency
OUT_CSV = "tracker.csv"
# ---- client mechanics ----
SLEEP_RESOLUTION_S = 1e-3      # asyncio.sleep granularity; the last SLEEP_RESOLUTION_S before each send is spun with yields
SPIN_THRESHOLD_S = 1.5 * SLEEP_RESOLUTION_S   # gaps shorter than this are spun entirely
DRAIN_EVERY = 256              # flush the socket buffer every N sends
COMPLETION_POLL_S = 0.01       # how often to check for the last responses after the schedule ends
SEED_MAX = 100_000             # random seeds are drawn below this so they print short
MS_PER_S, P50, P99 = 1e3, 50, 99
# ---- formulas from benchmark_design.md ----
FLOPS_PER_MAC = 2              # F8: one multiply + one add per weight entry
ATTN_MATMULS_PER_LAYER = 2     # F8: Q·K^T and A·V, each S x S x d
GREEDY_P50_CYCLES = 1.5        # F14: mean/median = half a cycle of residual wait + own cycle (plot reference only)
GREEDY_P99_CYCLES = 2.0        # F14: p99 = residual of current cycle (<= 1 cycle) + own cycle
WINDOW_P99_WAIT_FRACTION = 1.0 # F15: the first request of a window waits the full T_w


def schedule(a, rng):
    """Arrival times (s): Poisson at qps, switching to qps_peak during [0, BURST_S) of every PERIOD_S."""
    t, ts = 0.0, []
    while t < a.duration_s:
        rate = a.qps_peak if (t % PERIOD_S) < BURST_S else a.qps
        t += rng.exponential(1.0 / rate); ts.append(t)
    ts = np.array(ts[:-1])
    assert np.all(np.diff(ts) > 0), "schedule must be strictly increasing"
    return ts


SLOPE_POINTS = 4               # k = least-squares slope of T_batch(B) over the largest SLOPE_POINTS buckets (F11)


def slope_fit(calib):
    """k in ms/request from a calibration table {B: ms}. Robust to one non-monotonic bucket (kernel selection)."""
    B = np.array(sorted(calib))[-SLOPE_POINTS:]
    return float(np.polyfit(B, [calib[b] for b in B], 1)[0]) if len(B) >= 2 else math.nan


def predictions(a, st):
    """Formulas F11-F15 with measured constants: a = calib[1], k = slope_fit(calib)."""
    calib = {int(k): v for k, v in st["calib"].items()}
    P, S, d, L, T_w = st["n_params"], st["seq_len"], st["d_model"], st["n_layers"], st["max_wait_ms"]
    a_ms = calib[1]
    k_ms = slope_fit(calib)
    flops_req = FLOPS_PER_MAC * P * S + ATTN_MATMULS_PER_LAYER * FLOPS_PER_MAC * S * S * d * L  # F8: 2PS + 4S^2dL
    B_obs = max(1, int(round(st["B_mean"])))
    overhead = max(0.0, st["cycle_ms_mean"] - calib[1 << (B_obs - 1).bit_length()])           # dispatch per batch
    a_eff = a_ms + overhead
    u = a.qps * k_ms / MS_PER_S                                                                 # F12: u = lambda * k
    p = dict(a_ms=a_ms, a_eff_ms=a_eff, F_eff_tflops=flops_req / (k_ms / MS_PER_S) / 1e12, u_pred=u,
             implied_cpu_nodes=a.qps * a_ms / MS_PER_S if st["device"] == "cpu" else math.nan)   # F7
    if T_w > 0:                                               # F15, fixed window
        B = a.qps * T_w / MS_PER_S; T = a_ms + B * k_ms       # F1, F11
        p.update(B_pred=B, T_batch_pred_ms=T, lat_p99_textbook_ms=WINDOW_P99_WAIT_FRACTION * T_w + T,
                 lat_p99_pred_ms=WINDOW_P99_WAIT_FRACTION * T_w + T + overhead)
    elif u < 1:                                               # F13/F14, greedy fixed point
        T, T_eff = a_ms / (1 - u), a_eff / (1 - u)
        p.update(B_pred=a.qps * T / MS_PER_S, T_batch_pred_ms=T, lat_p99_textbook_ms=GREEDY_P99_CYCLES * T,
                 lat_p99_pred_ms=GREEDY_P99_CYCLES * T_eff)
    else:
        p.update(B_pred=math.inf, T_batch_pred_ms=math.inf, lat_p99_textbook_ms=math.inf, lat_p99_pred_ms=math.inf)
    return p


async def run(a):
    rng = np.random.default_rng(a.seed)
    reader, writer = await asyncio.open_connection(HOST, PORT)
    ctrl = {}

    async def control(cid):
        ctrl[cid] = asyncio.get_running_loop().create_future()
        writer.write(REQ.pack(cid)); await writer.drain()
        return await ctrl[cid]

    sched = schedule(a, rng); n = len(sched)
    sent, recv = np.full(n, np.nan), np.full(n, np.nan)
    state = dict(received=0, mismatch=0, dup=0, chk_table=None)

    async def read_loop():
        while True:
            (rid,) = REQ.unpack(await reader.readexactly(REQ.size))
            if rid >= CTRL_STATS:
                (_, ln) = CTRL.unpack(REQ.pack(rid) + await reader.readexactly(CTRL.size - REQ.size))
                ctrl.pop(rid).set_result(json.loads(await reader.readexactly(ln))); continue
            _, _score, chk = RSP.unpack(REQ.pack(rid) + await reader.readexactly(RSP.size - REQ.size))
            now = time.perf_counter() - t0
            assert rid < n, f"unknown id {rid}"
            if not np.isnan(recv[rid]): state["dup"] += 1; continue
            recv[rid] = now; state["received"] += 1
            if chk != state["chk_table"][rid % len(state["chk_table"])]: state["mismatch"] += 1

    t0 = time.perf_counter()
    reader_task = asyncio.create_task(read_loop())
    st0 = await control(CTRL_STATS)
    S, V = st0["seq_len"], st0["vocab"]
    state["chk_table"] = np.array([sum((r + i * TOKEN_STRIDE) % V for i in range(S)) for r in range(V)])

    print(f"sending {n} requests over {a.duration_s}s (qps={a.qps}, peak={a.qps_peak} for {BURST_S}s/{PERIOD_S}s, seed={a.seed})")
    t0 = time.perf_counter(); reset_done = False
    for i, t in enumerate(sched):
        if not reset_done and t >= WARMUP_S:
            await control(CTRL_RESET); reset_done = True
        delay = t0 + t - time.perf_counter()
        if delay > SPIN_THRESHOLD_S: await asyncio.sleep(delay - SLEEP_RESOLUTION_S)
        while t0 + t > time.perf_counter(): await asyncio.sleep(0)   # yield each spin so the reader keeps running
        writer.write(REQ.pack(i)); sent[i] = time.perf_counter() - t0
        if i % DRAIN_EVERY == 0: await writer.drain()
    await writer.drain()
    timeout_ms = TIMEOUT_SLO_MULT * a.slo_ms
    deadline = t0 + sched[-1] + timeout_ms / MS_PER_S
    while state["received"] < n and time.perf_counter() < deadline: await asyncio.sleep(COMPLETION_POLL_S)
    st = await control(CTRL_STATS)
    reader_task.cancel(); writer.close()

    # ---- statistics over the measured window ----
    m = sched >= WARMUP_S
    lat = (recv - sched)[m] * MS_PER_S
    late = np.isnan(lat) | (lat > timeout_ms)                     # unanswered, or answered after the timeout
    failed = int(late.sum()); lat = np.where(late, timeout_ms, lat)
    lag = (sent - sched)[m] * MS_PER_S
    pct = lambda q: float(np.percentile(lat, q))
    pr = predictions(a, st)
    row = dict(datetime=datetime.now().isoformat(timespec="seconds"), tag=a.tag, seed=a.seed,
               mode="window" if st["max_wait_ms"] > 0 else "greedy", qps=a.qps, qps_peak=a.qps_peak,
               n_params=st["n_params"], seq_len=S, max_wait_ms=st["max_wait_ms"],
               duration_s=a.duration_s, slo_ms=a.slo_ms, device=st["device"], gpu=st["gpu"], max_batch=st["max_batch"],
               lat_p50_ms=pct(P50), lat_p99_ms=pct(P99), lat_max_ms=float(lat.max()),
               B_mean=st["B_mean"], B_max=st["B_max"], cycle_ms=st["cycle_ms_mean"],
               n_failed=failed + state["dup"], n_mismatch=state["mismatch"], send_lag_p99_ms=float(np.percentile(lag, P99)),
               **pr, calib_table=json.dumps(st["calib"]))

    out, i = OUT_CSV, 1                                       # never append under a different header
    while os.path.exists(out) and open(out).readline().strip() != ",".join(row):
        out = OUT_CSV.replace(".csv", f"_{i}.csv"); i += 1
    if out != OUT_CSV: print(f"header mismatch in {OUT_CSV}; writing to {out}")
    new = not os.path.exists(out)
    with open(out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        if new: w.writeheader()
        w.writerow(row)

    ok = row["n_failed"] == 0 and row["n_mismatch"] == 0 and row["lat_p99_ms"] <= a.slo_ms
    print(f"{'PASS' if ok else 'FAIL'}  sent={int(m.sum())} failed={row['n_failed']} mismatch={row['n_mismatch']} "
          f"lag_p99={row['send_lag_p99_ms']:.2f}  p50={row['lat_p50_ms']:.2f} p99={row['lat_p99_ms']:.2f} "
          f"max={row['lat_max_ms']:.2f} ms (SLO {a.slo_ms})  B_mean={row['B_mean']:.1f} (pred {row['B_pred']:.1f})  "
          f"cycle={row['cycle_ms']:.2f}ms p99/cycle={row['lat_p99_ms'] / row['cycle_ms']:.1f}  "
          f"u={row['u_pred']:.3f} F_eff={row['F_eff_tflops']:.1f}T  p99_pred={row['lat_p99_pred_ms']:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--qps", type=float, default=1000, help="lambda: steady Poisson arrival rate")
    p.add_argument("--qps-peak", type=float, default=None, help=f"rate during {BURST_S}s bursts every {PERIOD_S}s (default: no bursts)")
    p.add_argument("--duration-s", type=float, default=120)
    p.add_argument("--slo-ms", type=float, default=100, help="p99 target: PASS/FAIL and the timeout")
    p.add_argument("--tag", default="", help="free-text label written to the CSV")
    p.add_argument("--seed", type=int, default=None, help="arrival schedule seed (default: random, recorded in the CSV)")
    a = p.parse_args()
    a.qps_peak = a.qps_peak or a.qps
    a.seed = a.seed if a.seed is not None else int(time.time()) % SEED_MAX
    assert WARMUP_S < a.duration_s and a.qps_peak >= a.qps
    asyncio.run(run(a))
