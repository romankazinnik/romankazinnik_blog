#!/usr/bin/env bash
# Sweep for benchmark_design.md, hypothesis by hypothesis.
#   ./sweep.sh            full run: archives previous tracker*.csv/plots into runs/<timestamp>/, then H1..H6 + h3b + h3c
#                         (~2 h on a T4, ~1.5 h on a 3070), then python plot.py
#   ./sweep.sh h3 h5      run a subset, appending to the current tracker.csv
#   ./sweep.sh smoke      20 s CPU run to check the plumbing
# Override for quick dry runs:  DURATION=20 ./sweep.sh h2
set -euo pipefail
cd "$(dirname "$0")"

DURATION=${DURATION:-120}          # client --duration-s (must exceed WARMUP_S = 10)
CALIB_ONLY_DURATION=20             # H1 only needs the calibration table; a short run captures it into the CSV
SERVER_START_TIMEOUT_S=600         # 1B model: init + calibration up to B = 4096 takes a few minutes
LOG=sweep.log

# ---------- helpers ----------
SERVER_PID=""
server() {                         # server <server args...>   starts server.py, waits for "listening"
  stop_server
  echo "### server $*" | tee -a "$LOG"
  python server.py "$@" > server.log 2>&1 &
  SERVER_PID=$!
  for ((i = 0; i < SERVER_START_TIMEOUT_S; i++)); do
    grep -q "listening" server.log && { grep -E "model:|calibrating" server.log | tee -a "$LOG"; return; }
    kill -0 "$SERVER_PID" 2>/dev/null || { cat server.log; echo "server died"; exit 1; }
    sleep 1
  done
  echo "server did not start within ${SERVER_START_TIMEOUT_S}s"; cat server.log; exit 1
}
client() {                         # client <tag> <client args...>
  local tag=$1; shift
  echo "--- client $tag $*" | tee -a "$LOG"
  python client.py --tag "$tag" --duration-s "$DURATION" "$@" | tee -a "$LOG"
}
stop_server() { [[ -n "$SERVER_PID" ]] && kill "$SERVER_PID" 2>/dev/null && wait "$SERVER_PID" 2>/dev/null || true; SERVER_PID=""; }
trap stop_server EXIT

# ---------- hypotheses ----------
smoke() {   # plumbing check on CPU, no GPU needed
  server --device cpu --n-params 1e6 --max-batch 16
  DURATION=$CALIB_ONLY_DURATION client smoke --qps 200
}

h1() {   # Roofline: T_batch(B) = max(a, a + B k), slope ∝ P·S.  Only the calibration table matters -> short runs.
  for n in 1e6 1e7 1e9; do
    server --n-params "$n" --seq-len 1
    DURATION=$CALIB_ONLY_DURATION client "h1_n${n}_s1" --qps 1000
  done
  server --n-params 1e8 --seq-len 10 --max-batch 1024
  DURATION=$CALIB_ONLY_DURATION client h1_n1e8_s10 --qps 1000
  server --n-params 1e8 --seq-len 100 --max-batch 256
  DURATION=$CALIB_ONLY_DURATION client h1_n1e8_s100 --qps 1000
  # 100M / S=1 is captured by h2 below
}

h2() {   # Greedy drain: B = λ·cycle, p99 ≤ ~3 cycles, flat in λ while u ≪ 1.
  server --n-params 1e8 --seq-len 1
  for q in 100 1000 5000; do client "h2_q${q}" --qps "$q"; done
  # 5000 QPS: check send_lag_p99_ms in the row; if > 1 ms the client, not the GPU, set the tail
}

h3() {   # Utilization is the only variable: p99/a_eff collapses onto 2/(1−u); u→1 near S ≈ 65 for 100M @ 1000/s.
  for s in 10 30 50 60 65; do
    server --n-params 1e8 --seq-len "$s" --max-batch 512
    client "h3_s${s}" --qps 1000
  done
  # S=60/65 rows are expected to FAIL the 100 ms SLO: that crossing is the result, not a bug
}

h3b() {  # S sweep for the other model sizes, so plot_h3b_vs_S has measured points in every panel (~25 min).
  # Predicted SLO crossings at 1000/s from the S=1 rows: 1M and 10M near S ≈ 300 (narrow matmuls, low F_eff), 1B near S ≈ 7.
  for n in 1e6 1e7; do
    for s in 10 100 200 300 400; do
      server --n-params "$n" --seq-len "$s" --max-batch 256
      client "h3b_n${n}_s${s}" --qps 1000
    done
  done
  for s in 3 5 7 10; do
    server --n-params 1e9 --seq-len "$s" --max-batch 512
    client "h3b_n1e9_s${s}" --qps 1000
  done
  for n in 1e7 1e8; do                                                 # S = 1000 tables for the H1 sequence-length panels
    server --n-params "$n" --seq-len 1000 --max-batch 64
    DURATION=$CALIB_ONLY_DURATION client "h1_n${n}_s1000" --qps 50     # 100M: u ≈ 1 near 70/s, 50/s stays below
  done
}

h3c() {  # Densify the 100M panel of plot_h3b_vs_S on ANY card: calibrate, predict S*(λ) = F/(2·P·λ), sample 6 points per λ.
  local P=1e8 FRACS="0.1 0.2 0.35 0.5 0.65 0.8" RATES="50 100 1000 5000"
  local DURATION=${H3C_DURATION:-60}                                 # 60 s is enough here (>= 3000 samples at 50/s)
  server --n-params $P --seq-len 1
  DURATION=$CALIB_ONLY_DURATION client h3c_calib --qps 100          # one row whose F_eff_tflops sizes the grid
  local plan; plan=$(python - "$FRACS" "$RATES" <<'PY'
import sys, glob, pandas as pd
df = pd.concat(pd.read_csv(f) for f in glob.glob("tracker*.csv"))
row = df[df["tag"] == "h3c_calib"].iloc[-1]
F, P = row["F_eff_tflops"] * 1e12, row["n_params"]
want = {}
for q in map(float, sys.argv[2].split()):
    s_star = F / (2 * P * q)
    for f in map(float, sys.argv[1].split()):
        s = max(1, int(round(f * s_star)))
        want.setdefault(s, set()).add(int(q))
for s in sorted(want): print(s, *sorted(want[s]))
PY
)
  echo "h3c plan (S: rates):"; echo "$plan" | tee -a "$LOG"
  while read -r s rates; do
    local mb=512; (( s > 100 )) && mb=128; (( s > 500 )) && mb=64; (( s > 1500 )) && mb=16
    server --n-params $P --seq-len "$s" --max-batch "$mb"
    for q in $rates; do client "h3c_s${s}_q${q}" --qps "$q"; done
  done <<< "$plan"
}

h4() {   # Fixed window: B = λ·T_w, p99 ≈ T_w + T_batch — until T_batch > T_w.
  for w in 20 50; do
    server --n-params 1e8 --seq-len 1 --max-wait-ms "$w"
    client "h4_w${w}_s1" --qps 1000
  done
  server --n-params 1e8 --seq-len 50 --max-wait-ms 2 --max-batch 512    # u ≈ 0.6, T_w < a/(1-u) ≈ 4 ms: window inactive, expect B ≈ λ·T_cycle not λ·T_w
  client h4_w2_s50 --qps 1000
}

h5() {   # Bursts below saturation leave p99 unchanged (u_peak = k·u < 1); B_max ≈ k·B_mean.
  server --n-params 1e8 --seq-len 1
  client h5_s1_peak5x --qps 1000 --qps-peak 5000        # u_peak ≈ 0.08
  server --n-params 1e8 --seq-len 30 --max-batch 512
  client h5_s30_peak2x --qps 1000 --qps-peak 2000       # u ≈ 0.45 -> u_peak ≈ 0.9: still < 1, but close
}

h6() {   # CPU node (4 threads, fp32, 10M): saturates at λ ≈ 1/a; batching helps only up to the CPU roofline.
  server --device cpu --n-params 1e7 --seq-len 1 --max-batch 1
  for q in 100 300 1000; do client "h6_cpu_b1_q${q}" --qps "$q"; done   # expect FAIL once q > 1000/a_ms
  server --device cpu --n-params 1e7 --seq-len 1 --max-batch 4096
  client h6_cpu_batched_q1000 --qps 1000
  for s in 10 100 1000; do                                   # CPU calibration tables for the H1 10M sequence-length panel
    server --device cpu --n-params 1e7 --seq-len "$s" --max-batch $((1024 / s < 8 ? 8 : 1024 / s))
    DURATION=$CALIB_ONLY_DURATION client "h6_cpu_calib_s${s}" --qps 10
  done
}

# ---------- main ----------
targets=("$@")
if [[ ${#targets[@]} -eq 0 ]]; then                                   # full run: archive previous results, run every phase
  targets=(h1 h2 h3 h3b h4 h5 h6 h3c)
  if ls tracker*.csv >/dev/null 2>&1; then
    d="runs/$(date +%Y%m%d_%H%M%S)"; mkdir -p "$d"; mv tracker*.csv plot_*.png "$LOG" "$d"/ 2>/dev/null || true
    echo "previous results moved to $d"
  fi
fi
echo "===== sweep $(date -Is) targets: ${targets[*]} duration: ${DURATION}s" | tee -a "$LOG"
for t in "${targets[@]}"; do "$t"; done
stop_server
echo "===== done $(date -Is)" | tee -a "$LOG"
python plot.py tracker.csv
