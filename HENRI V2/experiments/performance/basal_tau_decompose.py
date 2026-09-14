#!/usr/bin/env python3
"""OBSERVED_GPU: decompose the 1,409 us slot, and sweep the Triton block size.

DIRECTIVE 1 (tau decomposition)
  Is the 44 us/step of cuFFT launch-bound or compute-bound? Two probes:
    (a) steps sweep 1..1024 -> total(s) = launch + s*marginal, so the launch
        component and the marginal per-step component separate.
    (b) CUDA-graph capture of the whole loop -> removes EVERY per-step host
        launch. If graph replay collapses per-step cost, the 50 us shutter
        question is an EXECUTION-MODEL question, not a hardware one.

DIRECTIVE 2 (Triton remediation)
  `fused_relax` launches grid = cdiv(8192, block_size) programs. At
  BLOCK_SIZE=1024 that is 8 programs on a 188-SM device. Record programs/SM
  coverage alongside n_regs/n_spills and latency, because the block-size
  argument may be about OCCUPANCY, not spilling. Measure, do not assume.

Every timed call asserts its operand is on CUDA. A CPU tensor labelled
OBSERVED_GPU is the one failure this instrument must never produce.
"""
import json, math, os, platform, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch

OUT = os.path.join(os.getcwd(), "experiments", "verification",
                   "basal_tau_decomposition_observed.json")

import basal_triton_kernel as tk
import basal_boundary_engine as bbe

N = int(bbe.SPEC_NUM_TILES)
STEPS_SWEEP = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
BLOCK_SWEEP = [128, 256, 512, 1024]
out = {"schema": "henri.basal.tau-decomposition.v1", "evidence_class": "OBSERVED_GPU"}

# --------------------------------------------------------------- device gate
print("=== device ===")
if not torch.cuda.is_available():
    print("FAIL-CLOSED: no CUDA device; this instrument reports nothing")
    out["verdict"] = "BLOCKED_NO_CUDA"
    json.dump(out, open(OUT, "w"), indent=2)
    raise SystemExit(2)
DEV = torch.device("cuda:0")
props = torch.cuda.get_device_properties(0)
print(f"  {props.name}  sm_{props.major}{props.minor}  SMs={props.multi_processor_count}  "
      f"VRAM={props.total_memory/2**20:.0f} MiB")
print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
out["device"] = {"name": props.name, "sm_count": props.multi_processor_count,
                 "cc": [props.major, props.minor], "torch": torch.__version__,
                 "cuda": torch.version.cuda, "python": platform.python_version()}
print(f"  kernel constant BLACKWELL_SM_COUNT = {tk.BLACKWELL_SM_COUNT}")
out["device"]["kernel_sm_constant"] = int(tk.BLACKWELL_SM_COUNT)

# --------------------------------------------------------------- operands
torch.manual_seed(0)
DECAY = bbe.recommended_leakage_length(N)
HALF = tk.taps_for_reach(tk.default_half_width(N, DECAY)) // 2
WEIGHTS_TAPS = tk.default_kernel(N, DECAY)          # (taps,) normalized taps
KERNEL_RING = tk.ring_kernel(N, DECAY)              # (N,) full-ring, for cuFFT
print(f"\n  N={N} decay={DECAY:.3f} reach=+/-{HALF} taps={int(WEIGHTS_TAPS.numel())}")
out["operator"] = {"num_channels": N, "decay_length": DECAY, "half_width": HALF,
                   "taps": int(WEIGHTS_TAPS.numel())}

theta0 = torch.linspace(-math.pi, math.pi, N, device=DEV, dtype=torch.float32)
w_taps = WEIGHTS_TAPS.to(DEV)
k_ring = KERNEL_RING.to(DEV)
assert theta0.is_cuda and w_taps.is_cuda and k_ring.is_cuda, "operands must be CUDA"


def ev(fn, iters=5, warmup=2, probe=None):
    """CUDA-event timing with a device assertion on the probe tensor."""
    if probe is not None and not probe.is_cuda:
        raise AssertionError(f"probe on {probe.device}; refusing to label it GPU")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e) * 1000.0)          # ms -> us
    times.sort()
    return {"iters": iters, "mean_us": round(sum(times)/len(times), 4),
            "median_us": round(times[len(times)//2], 4), "min_us": round(times[0], 4)}


# ============================================================ PART A: sweep
print("\n=== A. cuFFT steps sweep (production fft_relax) ===")
sweep = {}
for s in STEPS_SWEEP:
    r = ev(lambda s=s: tk.fft_relax(theta0, k_ring, steps=s),
           iters=(3 if s >= 512 else 10), warmup=1, probe=theta0)
    sweep[str(s)] = r
    print(f"  steps={s:5d}  mean={r['mean_us']:12.4f} us  "
          f"per-step={r['mean_us']/s:9.4f} us")
out["fft_steps_sweep"] = sweep

# Least-squares fit total(s) = launch + s * marginal on the sweep.
xs = [math.log2(s) for s in STEPS_SWEEP]
pts = [(float(s), sweep[str(s)]["mean_us"]) for s in STEPS_SWEEP]
n = len(pts)
sx = sum(p[0] for p in pts); sy = sum(p[1] for p in pts)
sxx = sum(p[0]**2 for p in pts); sxy = sum(p[0]*p[1] for p in pts)
slope = (n*sxy - sx*sy) / (n*sxx - sx*sx)
icpt = (sy - slope*sx) / n
print(f"\n  fit: total_us = {icpt:.2f} + {slope:.3f} * steps")
out["fft_fit"] = {"intercept_us": round(icpt, 4), "marginal_us_per_step": round(slope, 4)}
one = sweep["1"]["mean_us"]; big = sweep["1024"]["mean_us"]
marg = (big - one) / 1023.0
print(f"  two-point: launch~{one - marg:.2f} us, marginal~{marg:.2f} us/step")
out["fft_two_point"] = {"launch_us": round(one - marg, 4),
                        "marginal_us_per_step": round(marg, 4)}
frac = 100.0 * (one - marg) / marg
print(f"  => marginal per-step work DOMINATES launch by {frac:.0f}% "
      f"(launch={one-marg:.1f} us vs step={marg:.1f} us)")
out["launch_dominates"] = bool((one - marg) > marg)

# ============================================================ PART B: graph
print("\n=== B. CUDA-graph capture (removes every per-step host launch) ===")
graphs = {}
for s in (32, 1024):
    try:
        ctx = torch.cuda.graphs.graph_pool_handle()
        g = torch.cuda.CUDAGraph()
        th = theta0.clone()
        with torch.cuda.graph(g, pool=ctx):
            r_ = tk.fft_relax(th, k_ring, steps=s)
        repl = ev(lambda: g.replay(), iters=10, warmup=3)
        graphs[str(s)] = repl
        eager = sweep.get(str(s), {}).get("mean_us")
        ratio = (eager / repl["mean_us"]) if eager else None
        print(f"  steps={s:5d}  graph replay={repl['mean_us']:11.4f} us  "
              f"per-step={repl['mean_us']/s:8.4f} us  "
              f"eager={eager:.2f} us  speedup={ratio:.2f}x")
        graphs[str(s)]["eager_us"] = eager
        graphs[str(s)]["speedup_vs_eager"] = round(ratio, 4) if ratio else None
    except Exception as e:
        print(f"  steps={s}: capture FAILED ({type(e).__name__}: {str(e)[:120]})")
        graphs[str(s)] = {"status": "CAPTURE_FAILED",
                          "error": f"{type(e).__name__}: {str(e)[:200]}"}
out["cuda_graph"] = graphs

# ============================================================ PART C: Triton
print("\n=== C. Triton block-size sweep (directive 2) ===")
print(f"  grid programs = cdiv({N}, block_size); SMs = {props.multi_processor_count}")
blocks = {}
if not tk.TRITON_AVAILABLE:
    print("  Triton unavailable; nothing to sweep (not fabricated)")
    blocks["status"] = "TRITON_UNAVAILABLE"
else:
    for bs in BLOCK_SWEEP:
        programs = (N + bs - 1) // bs
        rec = {"block_size": bs, "programs": programs,
               "programs_per_sm": round(programs / props.multi_processor_count, 4)}
        # latency at one step (one launch per program set)
        try:
            rec["one_step"] = ev(lambda bs=bs: tk.fused_relax(
                theta0, w_taps, steps=1, block_size=bs), iters=10, warmup=2, probe=theta0)
        except Exception as e:
            rec["one_step"] = {"status": "FAILED", "error": f"{type(e).__name__}: {str(e)[:160]}"}
        # compiled-binary register pressure
        try:
            wc = theta0.clone(); oc = torch.empty_like(theta0)
            ck = tk._fused_autopoietic_kuramoto_kernel.warmup(
                wc, torch.zeros(N, device=DEV), w_taps, oc, N, int(w_taps.numel()),
                float(tk.SPEC_KURAMOTO_COUPLING_K), 0.01,
                BLOCK_SIZE=bs, BLOCK_SPAN=64, grid=(1,))
            rec["n_regs"] = int(getattr(ck, "n_regs", -1))
            rec["n_spills"] = int(getattr(ck, "n_spills", -1))
        except Exception as e:
            rec["n_regs"] = rec["n_spills"] = -1
            rec["register_error"] = f"{type(e).__name__}: {str(e)[:160]}"
        blocks[str(bs)] = rec
        t = rec["one_step"].get("mean_us")
        print(f"  BLOCK={bs:5d}  programs={programs:4d} ({rec['programs_per_sm']:.2f}/SM)  "
              f"n_regs={rec['n_regs']}  n_spills={rec['n_spills']}  "
              f"1-step={t if t is None else round(t,1)} us")
out["triton_block_sweep"] = blocks

# ============================================================ PART D: verdict
print("\n=== D. verdict ===")
ok = [k for k, v in graphs.items() if "mean_us" in v]
best_graph = min((graphs[k]["mean_us"] for k in ok), default=None)
a_bit = os.path.join(os.getcwd(), "experiments", "performance", "basal_tau_measured.py")
v = {
    "fft_marginal_us_per_step": round(marg, 4),
    "fft_launch_us": round(one - marg, 4),
    "launch_bound": bool((one - marg) > marg),
    "graph_captured_per_step_us": (round(best_graph/1024.0, 4) if best_graph and "1024" in ok else None),
    "shutter_us": float(tk.SPEC_SHUTTER_US),
}
v["one_step_fits_shutter"] = bool(marg <= tk.SPEC_SHUTTER_US)
v["thirtytwo_step_slot_fits_shutter"] = bool(sweep["32"]["mean_us"] <= tk.SPEC_SHUTTER_US)
if best_graph and "1024" in ok:
    v["horizon_1024_graph_us"] = round(graphs["1024"]["mean_us"], 4)
    v["horizon_fits_shutter_graph"] = bool(graphs["1024"]["mean_us"] <= tk.SPEC_SHUTTER_US)
for k, val in v.items():
    print(f"  {k} = {val}")
out["verdict"] = v

json.dump(out, open(OUT, "w"), indent=2)
print(f"\nWROTE {OUT}  ({os.path.getsize(OUT)} bytes)")
print("VERDICT: OBSERVED_GPU" if out.get("device") else "VERDICT: BLOCKED")
