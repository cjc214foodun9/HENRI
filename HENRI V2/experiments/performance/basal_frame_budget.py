#!/usr/bin/env python3
"""OBSERVED_GPU pass 3: the PER-FRAME question, and Triton launch separation.

WHY THIS PASS EXISTS
  Pass 2 measured graphed per-step cost only at steps=32 and 1024. The
  architecture question is per Zone A FRAME (50 us aperture), i.e. steps=1,2,4.
  Extrapolating the linear fit there is exactly the sin this project keeps
  catching (a derived number quoted as measured). So measure it directly.

  Also separates the Triton path's launch cost from its kernel cost, which
  decides whether the mandate's BLOCK_SIZE=256 remedy is viable.
"""
import json, math, os, platform, sys, time

sys.path.insert(0, os.getcwd())
import torch
import basal_triton_kernel as tk
import basal_boundary_engine as bbe

OUT = os.path.join(os.getcwd(), "experiments", "verification",
                   "basal_frame_budget_observed.json")
N = int(bbe.SPEC_NUM_TILES)
out = {"schema": "henri.basal.frame-budget.v1", "evidence_class": "OBSERVED_GPU"}

if not torch.cuda.is_available():
    out["verdict"] = "BLOCKED_NO_CUDA"; json.dump(out, open(OUT, "w"), indent=2); raise SystemExit(2)
DEV = torch.device("cuda:0")
p = torch.cuda.get_device_properties(0)
SMS = p.multi_processor_count
print(f"device: {p.name} sm_{p.major}{p.minor} SMs={SMS}")
out["device"] = {"name": p.name, "sm_count": SMS, "cc": [p.major, p.minor],
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "python": platform.python_version()}

DECAY = bbe.recommended_leakage_length(N)
W_TAPS = tk.default_kernel(N, DECAY)
K_RING = tk.ring_kernel(N, DECAY)
theta0 = torch.linspace(-math.pi, math.pi, N, device=DEV, dtype=torch.float32)
w_t = W_TAPS.to(DEV); k_r = K_RING.to(DEV)
out["operator"] = {"num_channels": N, "decay_length": DECAY,
                   "taps": int(W_TAPS.numel()),
                   "sealed_span": int(bbe.SPEC_NON_LOCAL_SPAN),
                   "sealed_horizon": int(bbe.SPEC_LOCK_HORIZON_STEPS)}
print(f"N={N} decay={DECAY:.3f} taps={int(W_TAPS.numel())} "
      f"sealed span={bbe.SPEC_NON_LOCAL_SPAN} horizon={bbe.SPEC_LOCK_HORIZON_STEPS}")


def ev(fn, iters=15, warmup=3, probe=None):
    if probe is not None and not probe.is_cuda:
        raise AssertionError(f"probe on {probe.device}; refusing to label it GPU")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a = torch.cuda.Event(enable_timing=True); b = torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000.0)
    ts.sort()
    return {"iters": iters, "mean_us": round(sum(ts)/len(ts), 4),
            "median_us": round(ts[len(ts)//2], 4), "min_us": round(ts[0], 4)}


SHUTTER = float(tk.SPEC_SHUTTER_US)

# ---------------------------------------------- A. eager per-frame
print("\n=== A. fft_relax EAGER at frame-scale step counts ===")
eager = {}
for s in (1, 2, 4, 8, 16, 32):
    r = ev(lambda s=s: tk.fft_relax(theta0, k_r, steps=s), iters=25, warmup=5, probe=theta0)
    eager[str(s)] = {**r, "fits_shutter": bool(r["mean_us"] <= SHUTTER),
                     "steps_per_frame_at_this_cost": round(SHUTTER/r["mean_us"], 3)}
    print(f"  steps={s:3d} eager={r['mean_us']:9.3f} us  "
          f"{'FITS' if r['mean_us']<=SHUTTER else 'over'} 50us  ({SHUTTER/r['mean_us']:.2f} calls/frame)")
out["eager_frame_scale"] = eager

# ---------------------------------------------- B. graphed per-frame
print("\n=== B. fft_relax CUDA-GRAPHED at frame-scale step counts ===")
def cap(steps):
    pool = torch.cuda.graphs.graph_pool_handle()
    g = torch.cuda.CUDAGraph()
    th = theta0.clone()
    with torch.cuda.graph(g, pool=pool):
        tk.fft_relax(th, k_r, steps=steps)
    return g

graphed = {}
for s in (1, 2, 4, 8, 16, 32, 64):
    try:
        g = cap(s)
        r = ev(lambda g=g: g.replay(), iters=25, warmup=5)
        graphed[str(s)] = {**r, "steps": s, "fits_shutter": bool(r["mean_us"] <= SHUTTER),
                           "steps_per_frame_at_this_cost": round(SHUTTER/r["mean_us"], 3),
                           "eager_us": eager[str(s)]["mean_us"] if str(s) in eager else None}
        print(f"  steps={s:3d} graph={r['mean_us']:9.3f} us  per-step={r['mean_us']/s:7.3f}  "
              f"{'FITS' if r['mean_us']<=SHUTTER else 'over'} 50us  "
              f"eager={eager[str(s)]['mean_us'] if str(s) in eager else float('nan'):8.2f}  "
              f"speedup={(eager[str(s)]['mean_us']/r['mean_us']) if str(s) in eager else float('nan'):.2f}x")
    except Exception as e:
        graphed[str(s)] = {"status": "CAPTURE_FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"  steps={s}: capture FAILED {type(e).__name__}")
out["graphed_frame_scale"] = graphed

# fit graphed over the points we captured
gp = [(float(k), v["mean_us"]) for k, v in graphed.items() if "mean_us" in v]
if len(gp) >= 2:
    n = len(gp); sx = sum(q[0] for q in gp); sy = sum(q[1] for q in gp)
    sxx = sum(q[0]**2 for q in gp); sxy = sum(q[0]*q[1] for q in gp)
    sl = (n*sxy - sx*sy)/(n*sxx - sx*sx); ic = (sy - sl*sx)/n
    out["graphed_fit"] = {"intercept_us": round(ic, 4), "marginal_us_per_step": round(sl, 4)}
    print(f"\n  graphed fit: total = {ic:.3f} + {sl:.3f}*steps")
    out["graphed_fit"]["max_steps_per_frame"] = round((SHUTTER - ic)/sl, 3)
    print(f"  => max steps whose GRAPHED total fits {SHUTTER:.0f} us: "
          f"{(SHUTTER-ic)/sl:.2f}")

# ---------------------------------------------- C. batch + graph compose
print("\n=== C. batched rings AND graphed (do the two gains compose?) ===")
def relax_batched(ph, kern, steps):
    th = ph.clone(); nn = th.shape[-1]
    kf = torch.fft.fft(kern)
    om = torch.zeros(nn, dtype=torch.float32, device=th.device)
    for _ in range(steps):
        z = torch.fft.ifft(torch.fft.fft(torch.exp(1j*th), dim=-1)*kf, dim=-1)
        force = torch.cos(th)*z.imag - torch.sin(th)*z.real
        th = th + (om + float(tk.SPEC_KURAMOTO_COUPLING_K)*force)*0.01
    return th

comp = {}
for B in (1, 8, 32):
    ph = theta0.unsqueeze(0).repeat(B, 1).contiguous()
    try:
        pool = torch.cuda.graphs.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=pool):
            relax_batched(ph, k_r, 32)
        r = ev(lambda g=g: g.replay(), iters=15, warmup=3)
        per = r["mean_us"]/(B*32)
        comp[str(B)] = {**r, "batch": B, "steps": 32, "per_ring_step_us": round(per, 4),
                        "relative_throughput_vs_B1": None}
        print(f"  B={B:3d} graph total={r['mean_us']:10.3f} us  per ring*step={per:8.4f} us")
    except Exception as e:
        comp[str(B)] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        print(f"  B={B}: FAILED {type(e).__name__}")
if comp.get("1") and comp.get("32"):
    base = comp["1"]["per_ring_step_us"]
    for k in comp:
        if "per_ring_step_us" in comp[k]:
            comp[k]["relative_throughput_vs_B1"] = round(base/comp[k]["per_ring_step_us"], 3)
    print(f"  => B=32 graph throughput vs B=1 graph: "
          f"{comp['1']['per_ring_step_us']/comp['32']['per_ring_step_us']:.2f}x")
out["batch_graph_compose"] = comp

# ---------------------------------------------- D. Triton launch separation
print("\n=== D. Triton fused_relax: launch vs kernel (BLOCK 16 vs 256) ===")
tri = {}
for bs in (16, 256):
    rec = {"block_size": bs, "programs": (N+bs-1)//bs}
    for s in (1, 8, 32):
        try:
            r = ev(lambda bs=bs, s=s: tk.fused_relax(theta0, w_t, steps=s, block_size=bs),
                   iters=10, warmup=3, probe=theta0)
            rec[f"steps_{s}"] = r
        except Exception as e:
            rec[f"steps_{s}"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:140]}"}
    t1 = rec["steps_1"].get("mean_us"); t32 = rec["steps_32"].get("mean_us")
    if t1 and t32:
        marg = (t32 - t1)/31.0
        rec["launch_us"] = round(t1 - marg, 4)
        rec["marginal_us_per_step"] = round(marg, 4)
        print(f"  BLOCK={bs:4d} 1-step={t1:9.3f}  32-step={t32:11.3f}  "
              f"launch~{t1-marg:7.3f} us  marginal~{marg:8.3f} us/step")
    else:
        print(f"  BLOCK={bs:4d} 1-step={t1}  32-step={t32}")
    tri[str(bs)] = rec
out["triton_launch_separation"] = tri

# ---------------------------------------------- E. span anchor
print("\n=== E. direct tap-sum anchor (relax_span, steps=1) ===")
try:
    r = ev(lambda: tk.relax_span(theta0, w_t, steps=1), iters=5, warmup=2, probe=theta0)
    out["span_anchor"] = r
    print(f"  relax_span 1-step = {r['mean_us']:.1f} us  "
          f"({r['mean_us']/float(tk.SPEC_SHUTTER_US):.1f}x the shutter)")
except Exception as e:
    out["span_anchor"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
    print(f"  FAILED {type(e).__name__}")

# ---------------------------------------------- F2. stream overlap
print("\n=== F2. two independent streams: is a second ring nearly free? ===")
streams_res = {}
try:
    s1 = torch.cuda.Stream(); s2 = torch.cuda.Stream()
    for _ in range(3):
        tk.fft_relax(theta0, k_r, steps=32)
        with torch.cuda.stream(s1):
            tk.fft_relax(theta0, k_r, steps=32)
        with torch.cuda.stream(s2):
            tk.fft_relax(theta0, k_r, steps=32)
    torch.cuda.synchronize()
    seq, par = [], []
    for _ in range(10):
        torch.cuda.synchronize(); a = time.perf_counter()
        tk.fft_relax(theta0, k_r, steps=32)
        tk.fft_relax(theta0, k_r, steps=32)
        torch.cuda.synchronize(); seq.append((time.perf_counter() - a) * 1e6)
    for _ in range(10):
        torch.cuda.synchronize(); a = time.perf_counter()
        with torch.cuda.stream(s1):
            tk.fft_relax(theta0, k_r, steps=32)
        with torch.cuda.stream(s2):
            tk.fft_relax(theta0, k_r, steps=32)
        torch.cuda.synchronize(); par.append((time.perf_counter() - a) * 1e6)
    seq.sort(); par.sort()
    sm_ = sum(seq)/len(seq); pm_ = sum(par)/len(par)
    streams_res = {"sequential_2x32_us": round(sm_, 3), "two_stream_2x32_us": round(pm_, 3),
                   "overlap_speedup": round(sm_/pm_, 3),
                   "second_ring_nearly_free": bool(sm_/pm_ > 1.7)}
    print(f"  sequential 2x32-step = {sm_:9.3f} us")
    print(f"  two-stream 2x32-step = {pm_:9.3f} us   overlap={sm_/pm_:.2f}x")
except Exception as e:
    streams_res = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
    print(f"  FAILED {type(e).__name__}: {str(e)[:120]}")
out["stream_overlap"] = streams_res

# ---------------------------------------------- F. verdict
print("\n=== F. verdict ===")
g1 = graphed.get("1", {}); g2 = graphed.get("2", {}); g32 = graphed.get("32", {})
e1 = eager["1"]; e32 = eager["32"]
v = {
    "shutter_us": SHUTTER,
    "steps_per_slot_default": 32,
}
v["eager_1step_us"] = e1["mean_us"]
v["graphed_1step_us"] = g1.get("mean_us")
v["graphed_2step_us"] = g2.get("mean_us")
v["graphed_32step_us"] = g32.get("mean_us")
v["eager_32step_within_shutter"] = bool(e32["mean_us"] <= SHUTTER)
v["graphed_1step_within_shutter"] = bool((g1.get("mean_us") or 1e9) <= SHUTTER)
v["graphed_2step_within_shutter"] = bool((g2.get("mean_us") or 1e9) <= SHUTTER)
v["graphed_32step_within_shutter"] = bool((g32.get("mean_us") or 1e9) <= SHUTTER)
for k, val in v.items():
    if val is not None:
        print(f"  {k} = {val}")
out["verdict"] = {k: val for k, val in v.items() if val is not None}

json.dump(out, open(OUT, "w"), indent=2)
print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")
print("VERDICT: OBSERVED_GPU")
