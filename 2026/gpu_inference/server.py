"""Batched Transformer scoring server over a fixed-size binary TCP protocol.

Wire protocol (little-endian):
  request : <I   id                   (4 bytes). Tokens are derived on the server from id.
  response: <Ifi id, score, checksum  (12 bytes). checksum = sum of the token row the model scored.
  control : id in {CTRL_RESET, CTRL_STATS} -> <II (id, n) followed by n bytes of JSON stats.

Batching modes:  --max-wait-ms 0  -> greedy drain (run everything queued, immediately)
                 --max-wait-ms W  -> fixed window (close W ms after the batch's first arrival)
"""
import argparse, asyncio, gc, json, math, struct, sys, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, torch, torch.nn as nn

# ---- protocol ----
REQ, RSP, CTRL = struct.Struct("<I"), struct.Struct("<Ifi"), struct.Struct("<II")
CTRL_RESET, CTRL_STATS = 0xFFFFFFFF, 0xFFFFFFFE
PORT = 9000
TOKEN_STRIDE = 7919            # tokens[i] = (id + i*TOKEN_STRIDE) % VOCAB; the client uses the same formula
# ---- model shape ----
VOCAB = 256                    # tiny embedding; parameters live in the transformer layers
N_LAYERS = 4                   # L
N_HEADS = 8
FFN_MULT = 4                   # feed-forward width = FFN_MULT * d
D_ROUND = 64                   # d_model is rounded to a multiple of this
ATTN_PARAMS_PER_D2 = 4         # Q, K, V, O projections: 4 d^2 per layer
PARAMS_PER_LAYER_PER_D2 = ATTN_PARAMS_PER_D2 + 2 * FFN_MULT   # + two FFN matrices of d x FFN_MULT*d  -> 12 d^2 (doc sec. 4)
# ---- runtime ----
CPU_THREADS = 4                # emulates a pod-sized CPU node
DTYPE = {"cuda": torch.float16, "cpu": torch.float32}
CALIB_WARMUP, CALIB_TIMED = 2, 9   # forward passes per bucket during calibration: discarded, then median of timed
GIL_SWITCH_S = 1e-4            # CPython default 5 ms: the event loop could wait that long while the model thread runs
FLOAT32_MAX_EXACT_INT = 2 ** 24   # checksums travel as float32 on the GPU and must stay exact
MS_PER_S = 1e3
BIND_ADDR = "0.0.0.0"


class Scorer(nn.Module):
    def __init__(self, d, seq_len):
        super().__init__()
        assert d % N_HEADS == 0, f"d_model={d} must be divisible by {N_HEADS} heads"
        self.emb, self.pos = nn.Embedding(VOCAB, d), nn.Parameter(torch.zeros(seq_len, d))
        layer = nn.TransformerEncoderLayer(d, N_HEADS, FFN_MULT * d, dropout=0.0, batch_first=True)
        self.enc, self.head = nn.TransformerEncoder(layer, N_LAYERS), nn.Linear(d, 1)

    def forward(self, x):  # x: [B, S] long -> [B] float in (0, 1)
        return self.head(self.enc(self.emb(x) + self.pos).mean(1)).squeeze(-1).sigmoid()


def derive_d(n_params):
    """P ~ PARAMS_PER_LAYER_PER_D2 * d^2 * N_LAYERS  ->  d, rounded to a multiple of D_ROUND."""
    return max(D_ROUND, D_ROUND * round(math.sqrt(n_params / (PARAMS_PER_LAYER_PER_D2 * N_LAYERS)) / D_ROUND))


def bucket(b):
    """Pad a batch to the next power of two so the GPU sees static shapes."""
    return 1 << (b - 1).bit_length()


class Engine:
    def __init__(self, a):
        self.a, self.dev, self.dtype = a, torch.device(a.device), DTYPE[a.device]
        torch.manual_seed(a.seed)
        self.d = derive_d(a.n_params)
        self.model = Scorer(self.d, a.seq_len).to(self.dev, self.dtype).eval()
        self.n_params = sum(p.numel() for p in self.model.parameters())
        self.offsets = torch.arange(a.seq_len, device=self.dev) * TOKEN_STRIDE
        assert a.seq_len * VOCAB < FLOAT32_MAX_EXACT_INT, "checksum must stay exact in float32"
        self.reset()

    def reset(self):
        self.batch_sizes, self.gpu_ms, self.cycle_ms = [], [], []

    @torch.inference_mode()
    def infer(self, ids):
        """ids: np.uint32 [B] -> (scores[B], checksums[B]). Runs in the worker thread."""
        B = len(ids); padded = bucket(B)
        ids_t = torch.zeros(padded, dtype=torch.int64, device=self.dev)
        ids_t[:B] = torch.from_numpy(ids.astype(np.int64)).to(self.dev)
        x = (ids_t[:, None] + self.offsets[None, :]) % VOCAB               # [padded, S]
        assert x.shape == (padded, self.a.seq_len)
        t = time.perf_counter()
        out = torch.stack([self.model(x).float(), x.sum(1).float()]).cpu().numpy()  # one sync, one copy
        self.gpu_ms.append((time.perf_counter() - t) * MS_PER_S); self.batch_sizes.append(B)
        return out[0, :B], out[1, :B].astype(np.int64)

    def calibrate(self):
        """T_batch(B) for every bucket up to max_batch: the empirical roofline."""
        self.calib, B = {}, 1
        while B <= self.a.max_batch:
            ids = np.arange(B, dtype=np.uint32)
            for _ in range(CALIB_WARMUP + CALIB_TIMED): self.infer(ids)
            self.calib[B] = round(float(np.median(self.gpu_ms[-CALIB_TIMED:])), 3)
            B *= 2
        self.reset()
        return self.calib

    def stats(self):
        bs, a = np.array(self.batch_sizes or [0]), self.a
        return dict(n_params=self.n_params, d_model=self.d, n_layers=N_LAYERS, seq_len=a.seq_len, vocab=VOCAB,
                    device=a.device, gpu=torch.cuda.get_device_name() if a.device == "cuda" else "cpu", max_batch=a.max_batch, max_wait_ms=a.max_wait_ms, calib=self.calib,
                    B_mean=float(bs.mean()), B_max=int(bs.max()),
                    gpu_ms_mean=float(np.mean(self.gpu_ms or [0])), cycle_ms_mean=float(np.mean(self.cycle_ms or [0])))


async def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-params", type=float, default=1e8, help="target parameter count P; d_model is derived")
    p.add_argument("--seq-len", type=int, default=1, help="S: tokens per request (built server-side from the id)")
    p.add_argument("--max-batch", type=int, default=4096, help="cap on B: GPU memory (attention is B*heads*S^2) and admission control")
    p.add_argument("--max-wait-ms", type=float, default=0, help="0 = greedy drain; W > 0 = fixed window T_w")
    p.add_argument("--device", default="cuda", choices=list(DTYPE), help="cuda runs fp16, cpu runs fp32")
    p.add_argument("--seed", type=int, default=0, help="weight initialization")
    a = p.parse_args(); a.n_params = int(a.n_params)
    torch.set_num_threads(CPU_THREADS)

    eng = Engine(a)
    print(f"model: P={eng.n_params:,} (target {a.n_params:,}) d={eng.d} L={N_LAYERS} S={a.seq_len} {a.device} "
          f"mode={'greedy' if a.max_wait_ms == 0 else f'window {a.max_wait_ms}ms'}")
    print("calibrating T_batch(B) ms:", eng.calibrate(), flush=True)
    sys.setswitchinterval(GIL_SWITCH_S)
    gc.disable()  # per-request garbage is refcounted; avoid gen-2 pauses in the serving path

    q, loop, pool = asyncio.Queue(), asyncio.get_running_loop(), ThreadPoolExecutor(1)

    async def handle(reader, writer):
        try:
            while True:
                (rid,) = REQ.unpack(await reader.readexactly(REQ.size))
                if rid >= CTRL_STATS:
                    if rid == CTRL_RESET: eng.reset()
                    body = json.dumps(eng.stats()).encode()
                    writer.write(CTRL.pack(rid, len(body)) + body)
                else:
                    q.put_nowait((rid, writer, loop.time()))
        except (asyncio.IncompleteReadError, ConnectionResetError):
            pass
        finally:
            writer.close()

    async def batcher():
        while True:
            items = [await q.get()]
            while len(items) < a.max_batch and not q.empty():      # take everything already queued (both modes)
                items.append(q.get_nowait())
            if a.max_wait_ms > 0:  # fixed window: wait for more until T_w after the arrival of the first request
                deadline = items[0][2] + a.max_wait_ms / MS_PER_S
                while len(items) < a.max_batch and (left := deadline - loop.time()) > 0:
                    try: items.append(await asyncio.wait_for(q.get(), left))
                    except asyncio.TimeoutError: break
            t_cycle = loop.time()
            ids = np.fromiter((rid for rid, _, _ in items), dtype=np.uint32, count=len(items))
            scores, chk = await loop.run_in_executor(pool, eng.infer, ids)  # the loop keeps reading meanwhile
            assert len(scores) == len(chk) == len(items)
            for (rid, w, _), s, c in zip(items, scores, chk):
                if not w.is_closing(): w.write(RSP.pack(rid, float(s), int(c)))
            eng.cycle_ms.append((loop.time() - t_cycle) * MS_PER_S)  # dispatch + model + responses

    asyncio.create_task(batcher())
    server = await asyncio.start_server(handle, BIND_ADDR, PORT)
    print(f"listening on :{PORT}", flush=True)
    async with server: await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())
