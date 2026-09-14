#!/usr/bin/env python3
"""OBSERVED_GPU pass 4: numerics parity at small BLOCK_SIZE, then batch latency.

A faster kernel that computes something else is worthless. `fused_relax`'s own
documented CPU parity reference is `relax_span` (the same tap sum, evaluated
directly). Pass 3 showed BLOCK_SIZE=16 is 219x faster than 1024 -- so the only
question that matters is whether it still computes the SAME thing.

Latency is reported only for configs that pass parity. Latency without parity
is not a result.
"""
import json, math, os, platform, sys, time

sys.path.insert(0, os.getcwd())
import torch
import basal_triton_kernel as tk
import basal_boundary_engine as bbe

OUT = os.path.join(os.getcwd(), "experiments", "verification",
                   "basal_triton_parity_observed.json")
N = int(bbe.SPEC_NUM_TILES)
out = {"schema": "henri.basal.triton-parity.v1", "evidence_class": "OBSERVED_GPU"}

if not torch.cuda.is_available():
    out["verdict"] = "BLOCKED_NO_CUDA"; json.dump(out, open(OUT, "w"), indent=2); raise SystemExit(2)
DEV = torch.device("cuda:0")
p = torch.cuda.get_device_properties(0)
print(f"device: {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count}")
out["device"] = {"name": p.name, "sm_count": p.multi_processor_count, "cc": [p.major, p.minor],
                 "torch": torch.__version__, "cuda": torch.version.cuda,
                 "python": platform.python_version()}

DECAY = bbe.recommended_leakage_length(N)
W = tk.default_kernel(N, DECAY).to(DEV)          # TAP vector, 2H+1 = 3025
K_RING = tk.ring_kernel(N, DECAY).to(DEV)        # RING kernel, N = 8192
theta0 = torch.linspace(-math.pi, math.pi, N, device=DEV, dtype=torch.float32)
assert W.is_cuda and theta0.is_cuda
out["operator"] = {"num_channels": N, "decay_length": DECAY, "taps": int(W.numel()),
                   "half_width": tk.taps_for_reach(tk.default_half_width(N, DECAY)) // 2}
print(f"N={N} decay={DECAY:.3f} taps={int(W.numel())}")


def ev(fn, iters=10, warmup=3):
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


def circ(a, b):
    d = torch.atan2(torch.sin(a - b), torch.cos(a - b))
    return float(d.abs().max().item())


STEPS = 8 if not tk.TRITON_AVAILABLE else 8
print("\n=== A. parity vs relax_span (the kernel's own reference), steps=%d ===" % STEPS)
ref = tk.relax_span(theta0, W, steps=STEPS)
r_ref = tk.order_parameter(ref)
print(f"  reference relax_span r = {r_ref:.6f}")
out["reference"] = {"steps": STEPS, "path": "relax_span", "r": r_ref}

par = {}
if not tk.TRITON_AVAILABLE:
    par["status"] = "TRITON_UNAVAILABLE"
    print("  Triton unavailable")
else:
    for bs in (16, 32, 64, 128, 256, 512, 1024):
        rec = {"block_size": bs, "programs": (N + bs - 1)//bs}
        try:
            got = tk.fused_relax(theta0, W, steps=STEPS, block_size=bs)
            rec["max_circular_diff_rad"] = circ(got, ref)
            rec["r"] = tk.order_parameter(got)
            rec["parity_ok"] = bool(rec["max_circular_diff_rad"] < 1e-3)
            rec["latency_8step"] = ev(lambda bs=bs: tk.fused_relax(theta0, W, steps=STEPS, block_size=bs))
            rec["per_step_us"] = round(rec["latency_8step"]["mean_us"]/STEPS, 4)
        except Exception as e:
            rec["status"] = "FAILED"; rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        par[str(bs)] = rec
        cd = rec.get("max_circular_diff_rad"); t = rec.get("per_step_us")
        print(f"  BLOCK={bs:5d} prog={(N+bs-1)//bs:4d}  circular_diff="
              f"{'--' if cd is None else f'{cd:.3e}'}  r={rec.get('r') if rec.get('r') is None else round(rec['r'],6)}"
              f"  parity={'OK  ' if rec.get('parity_ok') else 'FAIL'}  per-step={'' if t is None else f'{t:8.2f}'} us")
out["triton_parity_by_block_size"] = par

ok = {k: v for k, v in par.items() if v.get("parity_ok")}
out["parity_passing_block_sizes"] = sorted(int(k) for k in ok)
if ok:
    best = min(ok, key=lambda k: ok[k]["per_step_us"])
    out["best_parity_config"] = {"block_size": int(best),
                                 "per_step_us": ok[best]["per_step_us"],
                                 "max_circular_diff_rad": ok[best]["max_circular_diff_rad"]}
    print(f"  => fastest PARITY-CORRECT BLOCK_SIZE = {best} ({ok[best]['per_step_us']:.2f} us/step)")

print("\n=== B. batch latency: per-ENVIRONMENT step cost (multi-env) ===")


def relax_batched(ph, kern, steps):
    """FFT coupling: `kern` MUST be the full-ring kernel (length N).

    TRAP (hit 4x this session): the tap vector from `default_kernel()` has
    length 2H+1 (3025); the ring kernel from `ring_kernel()` has length N (8192).
    An FFT-based path needs the ring; a direct tap sum needs the taps. Passing
    the wrong one raises a size mismatch, and passing both "successfully" would
    silently change the operator.
    """
    th = ph.clone(); nn = th.shape[-1]
    assert kern.shape[0] == nn, (
        f"relax_batched needs the RING kernel (len {nn}); got len {kern.shape[0]}"
        " -- that looks like the tap vector"
    )
    kf = torch.fft.fft(kern)
    om = torch.zeros(nn, dtype=torch.float32, device=th.device)
    for _ in range(steps):
        z = torch.fft.ifft(torch.fft.fft(torch.exp(1j*th), dim=-1)*kf, dim=-1)
        force = torch.cos(th)*z.imag - torch.sin(th)*z.real
        th = th + (om + float(tk.SPEC_KURAMOTO_COUPLING_K)*force)*0.01
    return th

BAT = {}
for B in (1, 2, 4, 8, 16, 32, 64):
    ph = theta0.unsqueeze(0).repeat(B, 1).contiguous()
    eager = ev(lambda ph=ph: relax_batched(ph, K_RING, 32), iters=8, warmup=2)
    lat_per_ring_step = eager["mean_us"] / 32.0          # all rings concurrent -> LATENCY per step
    thr_per_ring_step = eager["mean_us"] / (B*32.0)      # throughput per ring-step
    BAT[str(B)] = {**eager, "batch": B, "latency_us_per_step_per_ring": round(lat_per_ring_step, 4),
                   "throughput_us_per_ring_step": round(thr_per_ring_step, 4),
                   "epoch_fits_shutter": bool(lat_per_ring_step <= float(tk.SPEC_SHUTTER_US))}
    print(f"  B={B:3d}  32-step epoch={eager['mean_us']:10.2f} us  "
          f"latency/step/ring={lat_per_ring_step:8.3f} us  "
          f"{'FITS' if lat_per_ring_step <= tk.SPEC_SHUTTER_US else 'over'} 50us")
out["batch_latency"] = BAT

print("\n=== C. verdict ===")
g1 = float(tk.SPEC_SHUTTER_US)
v = {
    "shutter_us": g1,
    "sealed_horizon_steps": int(bbe.SPEC_LOCK_HORIZON_STEPS),
    "steps_per_slot": 32,
    "parity_passing_block_sizes": out["parity_passing_block_sizes"],
    "triton_fastest_parity_config": out.get("best_parity_config"),
    "triton_still_slower_than_cufft": None,
    "multi_env_epoch_within_shutter": bool(BAT[str(max(int(k) for k in BAT))]["epoch_fits_shutter"]),
    "max_batch_tested": max(int(k) for k in BAT),
}
# direct comparison: triton best vs cuFFT eager marginal
try:
    fft32 = ev(lambda: tk.fft_relax(theta0, K_RING, steps=32), iters=8, warmup=2)
    v["cufft_32step_us"] = fft32["mean_us"]
    v["cufft_per_step_us"] = round(fft32["mean_us"]/32, 4)
    if out.get("best_parity_config"):
        v["triton_still_slower_than_cufft"] = bool(
            out["best_parity_config"]["per_step_us"] > v["cufft_per_step_us"])
except Exception as e:
    v["cufft_probe_error"] = f"{type(e).__name__}: {str(e)[:120]}"
for k, val in v.items():
    if val is not None:
        print(f"  {k} = {val}")
out["verdict"] = v

json.dump(out, open(OUT, "w"), indent=2)
print(f"\nWROTE {OUT} ({os.path.getsize(OUT)} bytes)")
print("VERDICT: OBSERVED_GPU")
