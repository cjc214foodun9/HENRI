#!/usr/bin/env python3
"""OBSERVED_GPU pass 2: corrected metrics + extended Triton/occupancy sweep.

CORRECTED FROM PASS 1 (two mislabeled derived fields -- raw data was sound):
  * pass-1 `verdict.graph_captured_per_step_us` divided the 32-step GRAPH time
    by 1024. Each graph figure must be divided by ITS OWN step count.
  * pass-1 `verdict.launch_bound` compared the one-time launch (53 us) to ONE
    marginal step (43 us) and reported True. That is a category error: a
    one-time cost cannot dominate a per-step cost. The question the directive
    actually asks is whether PER-STEP cost FALLS as steps grow (amortisation).
    Measured here as the ratio per_step(1024)/per_step(1).

NEW IN PASS 2
  * Triton sweep extended DOWN to 16/32/64: pass 1 showed latency falling 20.5x
    from BLOCK 1024 -> 128, so the minimum may lie below 128.
  * BLOCK_SPAN sweep at the best block size (the tap walk is `span/BLOCK_SPAN`
    iterations of a register-resident accumulator).
  * Batched-ring FFT: B independent rings relaxed CONCURRENTLY in one launch.
    This is the real test of the "pre-planned batch" hypothesis. The relaxation
    is sequential WITHIN a ring, so steps cannot be batched -- only independent
    rings can. If per-ring-step cost falls with B, launch amortises across
    instances; if it is flat, it does not.
"""
import json, math, os, platform, sys

sys.path.insert(0, os.getcwd())
import torch

import basal_triton_kernel as tk
import basal_boundary_engine as bbe

OUT = os.path.join(os.getcwd(), "experiments", "verification",
                   "basal_tau_decomposition_observed_v2.json")

N = int(bbe.SPEC_NUM_TILES)
BLOCK_SWEEP = [16, 32, 64, 128, 256, 512, 1024]
SPAN_SWEEP = [64, 256, 1024]
BATCH = [1, 2, 4, 8, 16, 32]
STEPS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
out = {"schema": "henri.basal.tau-decomposition.v2", "evidence_class": "OBSERVED_GPU",
       "corrections_to_v1": [
           "verdict.graph_captured_per_step_us divided the 32-step graph by 1024",
           "verdict.launch_bound compared a one-time cost to a per-step cost"]}

print("=== device ===")
if not torch.cuda.is_available():
    out["verdict"] = "BLOCKED_NO_CUDA"
    json.dump(out, open(OUT, "w"), indent=2)
    raise SystemExit(2)
DEV = torch.device("cuda:0")
p = torch.cuda.get_device_properties(0)
print(f"  {p.name}  sm_{p.major}{p.minor}  SMs={p.multi_processor_count}  "
      f"VRAM={p.total_memory/2**20:.0f} MiB  torch {torch.__version__} cuda {torch.version.cuda}")
out["device"] = {"name": p.name, "sm_count": p.multi_processor_count,
                 "cc": [p.major, p.minor], "torch": torch.__version__,
                 "cuda": torch.version.cuda, "python": platform.python_version(),
                 "kernel_sm_constant": int(tk.BLACKWELL_SM_COUNT)}
SMS = p.multi_processor_count

DECAY = bbe.recommended_leakage_length(N)
HALF = tk.taps_for_reach(tk.default_half_width(N, DECAY)) // 2
W_TAPS = tk.default_kernel(N, DECAY)
K_RING = tk.ring_kernel(N, DECAY)
theta0 = torch.linspace(-math.pi, math.pi, N, device=DEV, dtype=torch.float32)
w_t = W_TAPS.to(DEV); k_r = K_RING.to(DEV)
assert theta0.is_cuda and w_t.is_cuda and k_r.is_cuda
print(f"  N={N} decay={DECAY:.3f} reach=+/-{HALF} taps={int(W_TAPS.numel())}")
out["operator"] = {"num_channels": N, "decay_length": DECAY, "half_width": HALF,
                   "taps": int(W_TAPS.numel())}


def ev(fn, iters=10, warmup=3, probe=None):
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


# ---------------------------------------------------- A. launch reference
print("\n=== A. launch reference (how much of a call is not work) ===")
tiny = torch.zeros(64, device=DEV)
out["launch_reference"] = {"null_op_sync": ev(lambda: torch.cuda.synchronize(),
                                              iters=30, warmup=5)}
print(f"  cudaSynchronize : {out['launch_reference']['null_op_sync']['mean_us']:8.3f} us")


# ---------------------------------------------------- B. steps sweep
print("\n=== B. cuFFT steps sweep (production fft_relax) ===")
sweep = {}
for s in STEPS_SWEEP:
    r = ev(lambda s=s: tk.fft_relax(theta0, k_r, steps=s),
           iters=(3 if s >= 512 else 20), warmup=3, probe=theta0)
    sweep[str(s)] = r
    print(f"  steps={s:5d} mean={r['mean_us']:12.4f} us  per-step={r['mean_us']/s:8.4f} us")
out["fft_steps_sweep"] = sweep
pts = [(float(s), sweep[str(s)]["mean_us"]) for s in STEPS_SWEEP]
n = len(pts); sx = sum(q[0] for q in pts); sy = sum(q[1] for q in pts)
sxx = sum(q[0]**2 for q in pts); sxy = sum(q[0]*q[1] for q in pts)
slope = (n*sxy - sx*sy)/(n*sxx - sx*sx); icpt = (sy - slope*sx)/n
ps1 = sweep["1"]["mean_us"]; ps32 = sweep["32"]["mean_us"]/32; ps1024 = sweep["1024"]["mean_us"]/1024
print(f"  fit: total_us = {icpt:.2f} + {slope:.3f}*steps")
print(f"  per-step: 1={ps1:.2f}  32={ps32:.2f}  1024={ps1024:.2f} us")
amort = 1.0 - (ps1024/ps1)
print(f"  per-step improvement 1 -> 1024 steps: {100*amort:.1f}%")
out["fft_decomposition"] = {
    "fit_intercept_us": round(icpt, 4), "fit_marginal_us_per_step": round(slope, 4),
    "per_step_us_at_1": round(ps1, 4), "per_step_us_at_32": round(ps32, 4),
    "per_step_us_at_1024": round(ps1024, 4),
    "per_step_improvement_1_to_1024_pct": round(100*amort, 4),
    "amortisation_is_material": bool(amort > 0.5)}
print(f"  => per-step amortisation is {'MATERIAL' if amort > 0.5 else 'IMMATERIAL'}")


# ---------------------------------------------------- C. CUDA graph
print("\n=== C. CUDA-graph replay (removes per-step HOST launch) ===")
graphs = {}
for s in (32, 1024):
    try:
        pool = torch.cuda.graphs.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        th = theta0.clone()
        with torch.cuda.graph(g, pool=pool):
            tk.fft_relax(th, k_r, steps=s)
        r = ev(lambda: g.replay(), iters=10, warmup=3)
        eager = sweep[str(s)]["mean_us"]
        graphs[str(s)] = {**r, "steps": s, "per_step_us": round(r["mean_us"]/s, 4),
                          "eager_us": eager, "speedup_vs_eager": round(eager/r["mean_us"], 4),
                          "host_launch_us_per_step": round((eager - r["mean_us"])/s, 4)}
        print(f"  steps={s:5d} graph={r['mean_us']:11.4f} us  per-step={r['mean_us']/s:8.4f}  "
              f"eager={eager:11.2f}  speedup={eager/r['mean_us']:.2f}x  "
              f"host-launch/step={(eager-r['mean_us'])/s:.2f} us")
    except Exception as e:
        graphs[str(s)] = {"status": "CAPTURE_FAILED", "error": f"{type(e).__name__}: {str(e)[:200]}"}
        print(f"  steps={s}: capture FAILED ({type(e).__name__})")
out["cuda_graph"] = graphs


# ---------------------------------------------------- D. batched rings
print("\n=== D. batched independent rings in ONE launch (pre-planned batch) ===")
def relax_batched(ph, kern, steps):
    """Same arithmetic as fft_relax._step, on [B, N]. Probe only, not production."""
    th = ph.clone(); n = th.shape[-1]
    kf = torch.fft.fft(kern)
    om = torch.zeros(n, dtype=torch.float32, device=th.device)
    for _ in range(steps):
        z = torch.fft.ifft(torch.fft.fft(torch.exp(1j * th), dim=-1) * kf, dim=-1)
        force = torch.cos(th) * z.imag - torch.sin(th) * z.real
        th = th + (om + float(tk.SPEC_KURAMOTO_COUPLING_K) * force) * 0.01
    return th

bat = {}
for B in BATCH:
    ph = theta0.unsqueeze(0).repeat(B, 1).contiguous()
    r = ev(lambda ph=ph: relax_batched(ph, k_r, 32), iters=10, warmup=3, probe=ph)
    per = r["mean_us"] / (B * 32)
    bat[str(B)] = {**r, "batch": B, "steps": 32, "per_ring_step_us": round(per, 4)}
    print(f"  B={B:3d} rings  total={r['mean_us']:11.4f} us  per-(ring*step)={per:8.4f} us")
out["batched_rings"] = bat
b1 = bat["1"]["per_ring_step_us"]; b32 = bat["32"]["per_ring_step_us"]
gain = 1.0 - (b32/b1)
print(f"  batching gain B=1 -> 32: {100*gain:.1f}%  ({b1:.2f} -> {b32:.2f} us per ring*step)")
out["batch_gain_pct"] = round(100*gain, 4)


# ---------------------------------------------------- E. Triton sweep
print("\n=== E. Triton block sweep (directive 2) ===")
print(f"  grid = cdiv({N}, block_size);  device has {SMS} SMs")
blocks = {}
if not tk.TRITON_AVAILABLE:
    blocks["status"] = "TRITON_UNAVAILABLE"
else:
    for bs in BLOCK_SWEEP:
        prog = (N + bs - 1)//bs
        rec = {"block_size": bs, "programs": prog, "programs_per_sm": round(prog/SMS, 4)}
        try:
            rec["one_step"] = ev(lambda bs=bs: tk.fused_relax(theta0, w_t, steps=1, block_size=bs),
                                 iters=10, warmup=3, probe=theta0)
        except Exception as e:
            rec["one_step"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        try:
            wc = theta0.clone(); oc = torch.empty_like(theta0)
            ck = tk._fused_autopoietic_kuramoto_kernel.warmup(
                wc, torch.zeros(N, device=DEV), w_t, oc, N, int(w_t.numel()),
                float(tk.SPEC_KURAMOTO_COUPLING_K), 0.01, BLOCK_SIZE=bs, BLOCK_SPAN=64, grid=(1,))
            rec["n_regs"] = int(getattr(ck, "n_regs", -1))
            rec["n_spills"] = int(getattr(ck, "n_spills", -1))
        except Exception as e:
            rec["n_regs"] = rec["n_spills"] = -1
            rec["register_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        blocks[str(bs)] = rec
        t = rec["one_step"].get("mean_us")
        print(f"  BLOCK={bs:5d} prog={prog:4d} ({prog/SMS:5.2f}/SM) n_regs={rec['n_regs']:4d} "
              f"n_spills={rec['n_spills']:5d}  1-step={t if t is None else round(t,1)} us")
out["triton_block_sweep"] = blocks

ok = {k: v for k, v in blocks.items() if isinstance(v, dict) and "mean_us" in v.get("one_step", {})}
if ok:
    best = min(ok, key=lambda k: ok[k]["one_step"]["mean_us"])
    print(f"  => fastest BLOCK_SIZE = {best} ({ok[best]['one_step']['mean_us']:.1f} us/step, "
          f"{ok[best]['n_spills']} spills)")
    out["triton_best"] = {"block_size": int(best),
                          "one_step_us": ok[best]["one_step"]["mean_us"],
                          "n_spills": ok[best]["n_spills"],
                          "n_regs": ok[best]["n_regs"]}
    zero = [k for k, v in ok.items() if v.get("n_spills") == 0]
    out["triton_zero_spill_configs"] = [int(k) for k in zero]
    print(f"     n_spills==0 at: {sorted(int(k) for k in zero) or 'NONE of the swept sizes'}")

print("\n=== F. BLOCK_SPAN sweep at the best block size ===")
spans = {}
if ok:
    bbs = int(min(ok, key=lambda k: ok[k]["one_step"]["mean_us"]))
    for sp in SPAN_SWEEP:
        rec = {}
        try:
            rec["one_step"] = ev(lambda sp=sp: tk.fused_relax(theta0, w_t, steps=1,
                                                              block_size=bbs, block_span=sp),
                                 iters=10, warmup=3, probe=theta0)
        except Exception as e:
            rec["one_step"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        try:
            wc = theta0.clone(); oc = torch.empty_like(theta0)
            ck = tk._fused_autopoietic_kuramoto_kernel.warmup(
                wc, torch.zeros(N, device=DEV), w_t, oc, N, int(w_t.numel()),
                float(tk.SPEC_KURAMOTO_COUPLING_K), 0.01, BLOCK_SIZE=bbs, BLOCK_SPAN=sp, grid=(1,))
            rec["n_spills"] = int(getattr(ck, "n_spills", -1))
            rec["n_regs"] = int(getattr(ck, "n_regs", -1))
        except Exception:
            rec["n_spills"] = rec["n_regs"] = -1
        spans[str(sp)] = rec
        t = rec["one_step"].get("mean_us")
        print(f"  BLOCK={bbs} SPAN={sp:5d}  n_regs={rec['n_regs']:4d} n_spills={rec['n_spills']:5d}  "
              f"1-step={t if t is None else round(t,1)} us")
    out["block_span_sweep"] = {"block_size": bbs, "results": spans}


# ---------------------------------------------------- G. verdict
print("\n=== G. verdict ===")
g32 = graphs.get("32", {}); g1024 = graphs.get("1024", {})
steps_per_slot = 32
v = {
    "shutter_us": float(tk.SPEC_SHUTTER_US),
    "sub_budget_us": float(tk.SPEC_TAU_BUDGET_US),
    "per_step_eager_us": round(ps1024, 4),
    "per_step_graphed_us": g1024.get("per_step_us"),
    "host_launch_us_per_step": g1024.get("host_launch_us_per_step"),
    "host_share_of_per_step_pct": (round(100.0*g1024["host_launch_us_per_step"]/ps1024, 2)
                                   if g1024.get("host_launch_us_per_step") else None),
    "one_step_fits_shutter": bool(g1024.get("per_step_us", 1e9) <= tk.SPEC_SHUTTER_US),
    "graph_32step_slot_us": g32.get("mean_us"),
    "graph_32step_slot_fits_shutter": bool(g32.get("mean_us", 1e9) <= tk.SPEC_SHUTTER_US),
    "graph_1024_horizon_us": g1024.get("mean_us"),
    "graph_1024_horizon_fits_shutter": bool(g1024.get("mean_us", 1e9) <= tk.SPEC_SHUTTER_US),
    "per_step_drops_materially_with_steps": bool(amort > 0.5),
}
v["shutter_is_execution_model_question"] = bool(v["per_step_drops_materially_with_steps"]
                                               and v["graph_32step_slot_fits_shutter"])
for k, val in v.items():
    print(f"  {k} = {val}")
out["verdict"] = v

json.dump(out, open(OUT, "w"), indent=2)
print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")
print("VERDICT: OBSERVED_GPU")
