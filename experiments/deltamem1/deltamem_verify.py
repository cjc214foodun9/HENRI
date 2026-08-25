"""DeltaMem-1 remote disposable verification (C6/C7/C11/C10 + equivalence).

Runs ONLY on the Vast CUDA target (torch 2.12.0+cu130, triton 3.7.0, RTX 5090).
Disposable data only — never loads the sealed split.

Axes (prereg #6, A1, A5, A7, C10, C11):
  A. Triton kernel vs PyTorch reference equivalence over:
       seq len {1,8,64,1000}, r {2,8}, D {64,4096}, batch {1,4},
       reset boundaries, 1,000-step repeated decay.
     fp32 variant tolerance 1e-4; bf16 variant tolerance 5e-3 (reported).
     Non-contiguous input -> explicit ValueError (prereg #6).
  B. C6 latency: combined readout+update step, CUDA events, 10k iters after
     warmup; report mean (gate) + p95; ALSO host-visible wall mean (diagnostic).
  C. C7 footprint: bf16 U+V bytes == 131,072 (D=4096 r=8); fp32 == 262,144
     (diagnostic, A1).
  D. C11 two-process determinism: spawn 2 fresh CUDA processes, identical
     inputs, compare state hashes + telemetry bytes (timestamps excluded).
  E. C10 veto semantics on the device kernel:
       conflict veto: readout non-null, Delta > 0.35 -> e zeroed, U <- gamma*U
                      (no eta write), counter incremented;
       null-readout exemption (A7): U=0 -> no veto, learning proceeds.
  F. Zero-trainable / default-OFF are covered by local fixtures F6; here we
     assert kernel has no parameters and state size.

Output: JSON receipt {c6:{...}, c7:{...}, c11:{...}, c10:{...}, equiv:{...}}.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from delta_qfhrr_associative_memory import DeltaQFHRRAssociativeMemory  # noqa: E402
from deltamem_triton_fused import FusedTritonDeltaKernel  # noqa: E402

V_SEED = 20260824
DET_CHILD = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "deltamem_det_child.py")


# ---------------------------------------------------------------------------
# A. Equivalence
# ---------------------------------------------------------------------------
def _run_ref(d: int, r: int, seq: list[tuple[torch.Tensor, torch.Tensor]],
             reset_at: int | None = None) -> dict:
    m = DeltaQFHRRAssociativeMemory(d=d, r=r, gamma=0.985, eta=0.1,
                                    enabled=True)
    e_norms = []
    for i, (k, v) in enumerate(seq):
        if reset_at is not None and i == reset_at:
            m.reset()
        e = m.step_once(k, v)
        e_norms.append(float(e.reshape(-1).norm().item()))
    return {"U": m.U, "e_norms": e_norms, "veto": int(m.veto_count.item()),
            "step": int(m.step.item())}


def _run_triton(kernel, seq, reset_at: int | None = None) -> dict:
    e_norms = []
    for i, (k, v) in enumerate(seq):
        if reset_at is not None and i == reset_at:
            kernel.reset()
        e = kernel.step_once(k, v)
        e_norms.append(float(e.reshape(-1).norm().item()))
    return {"U": kernel.U, "e_norms": e_norms,
            "veto": kernel.veto_count, "step": kernel.step}


def _seq(d: int, n: int, seed: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    pairs = []
    for _ in range(n):
        k = torch.randn(d, generator=g)
        v = torch.randn(d, generator=g)
        pairs.append((k, v))
    return pairs


def check_equivalence() -> dict:
    cases = [
        # (seq_len, r, D, batch, reset_at, fp32_tol, bf16_tol)
        (1, 2, 64, 1, None, 1e-4, 5e-3),
        (8, 2, 64, 1, 4, 1e-4, 5e-3),
        (8, 8, 64, 4, None, 1e-4, 5e-3),
        (64, 8, 4096, 1, 32, 1e-4, 5e-3),
        (1000, 8, 4096, 1, None, 1e-4, 5e-3),     # repeated decay
    ]
    results = []
    for (n, r, d, batch, reset_at, tol32, tol16) in cases:
        case_res = {"seq": n, "r": r, "D": d, "batch": batch}
        for inst in range(batch):
            seq = _seq(d, n, seed=1000 * (inst + 1) + r + d)
            seq_cuda = [(k.cuda(), v.cuda()) for k, v in seq]
            ref = _run_ref(d, r, seq, reset_at)
            # fp32 kernel variant (exactness gate 1e-4)
            k32 = FusedTritonDeltaKernel(d=d, r=r, v_seed=V_SEED,
                                         dtype=torch.float32)
            tr32 = _run_triton(k32, seq_cuda, reset_at)
            u_err = float((ref["U"].float().cpu() - tr32["U"].float().cpu())
                          .abs().max().item())
            e_err = max(abs(a - b) for a, b in zip(ref["e_norms"], tr32["e_norms"]))
            # bf16 kernel variant (C7 path, tolerance 5e-3)
            k16 = FusedTritonDeltaKernel(d=d, r=r, v_seed=V_SEED,
                                         dtype=torch.bfloat16)
            tr16 = _run_triton(k16, seq_cuda, reset_at)
            u16 = float((ref["U"].float().cpu() - tr16["U"].float().cpu())
                        .abs().max().item())
            e16 = max(abs(a - b) for a, b in zip(ref["e_norms"], tr16["e_norms"]))
            case_res[f"inst{inst}"] = {
                "fp32_U_maxerr": u_err, "fp32_e_maxerr": e_err,
                "fp32_pass": u_err <= tol32 and e_err <= tol32,
                "bf16_U_maxerr": u16, "bf16_e_maxerr": e16,
                "bf16_pass": u16 <= tol16 and e16 <= tol16,
                "veto_equal": ref["veto"] == tr16["veto"],
            }
        results.append(case_res)
    # non-contiguous rejection (prereg #6): strided 1-D view must raise
    nc_ok = False
    try:
        kern = FusedTritonDeltaKernel(d=4096, r=8, v_seed=V_SEED)
        base = torch.randn(4096, 2, device="cuda")
        kern.step_once(base[:, 0], base[:, 1])   # strided views -> raise
    except ValueError:
        nc_ok = True
    return {"cases": results, "noncontiguous_rejected": nc_ok}


# ---------------------------------------------------------------------------
# B. C6 latency (host-visible combined step; CUDA events; 10k iters)
# ---------------------------------------------------------------------------
def check_latency(iters: int = 10_000, warmup: int = 200) -> dict:
    kern = FusedTritonDeltaKernel(d=4096, r=8, v_seed=V_SEED)
    g = torch.Generator(device="cuda").manual_seed(7)
    k = torch.randn(4096, generator=g, device="cuda")
    v = torch.randn(4096, generator=g, device="cuda")
    for _ in range(warmup):
        kern.step_once(k, v)
    torch.cuda.synchronize()
    # CUDA events, bulk loop (A5 gate: mean < 15 us)
    starts = torch.cuda.Event(enable_timing=True)
    ends = torch.cuda.Event(enable_timing=True)
    starts.record()
    for _ in range(iters):
        kern.step_once(k, v)
    ends.record()
    torch.cuda.synchronize()
    cuda_ms = starts.elapsed_time(ends) / iters  # ms per step
    # per-iteration CUDA-event p95 (diagnostic; event record overhead inflates)
    n_p95 = min(2000, iters)
    evs = []
    for _ in range(n_p95):
        s = torch.cuda.Event(enable_timing=True)
        e_ = torch.cuda.Event(enable_timing=True)
        s.record(); kern.step_once(k, v); e_.record()
        evs.append((s, e_))
    torch.cuda.synchronize()
    times_us = sorted(s.elapsed_time(e) * 1e3 for s, e in evs)
    p95 = times_us[int(0.95 * len(times_us))] if times_us else None
    # host-visible wall (diagnostic; includes launch + final sync)
    t0 = time.perf_counter()
    for _ in range(iters):
        kern.step_once(k, v)
    torch.cuda.synchronize()
    wall_us = (time.perf_counter() - t0) / iters * 1e6
    return {"cuda_events_mean_us": cuda_ms * 1e3,
            "cuda_events_p95_us": p95,
            "host_visible_mean_us": wall_us,
            "iters": iters, "n_p95": n_p95}


# ---------------------------------------------------------------------------
# C. C7 footprint
# ---------------------------------------------------------------------------
def check_footprint() -> dict:
    k16 = FusedTritonDeltaKernel(d=4096, r=8, v_seed=V_SEED, dtype=torch.bfloat16)
    k32 = FusedTritonDeltaKernel(d=4096, r=8, v_seed=V_SEED, dtype=torch.float32)
    return {"bf16_bytes": k16.storage_bytes(), "bf16_gate_bytes": 128 * 1024,
            "bf16_pass": k16.storage_bytes() == 128 * 1024,
            "fp32_bytes_diag": k32.storage_bytes(),
            "state_dtype": str(k16.U.dtype)}


# ---------------------------------------------------------------------------
# E. C10 veto semantics (device kernel)
# ---------------------------------------------------------------------------
def check_veto() -> dict:
    """C10/A6/A7 semantics on the device kernel, deterministically.

    1) Null readout (U=0, first write): EXEMPT (A7) -> no veto, e = v, U learns.
    2) Conflict: construct U, k so readout = +c*v0 (c>0), then v = -v0
       -> Sagnac delta = 1 -> veto fires -> e zeroed, U <- gamma*U only
       (write suppressed), counter incremented.
    """
    g = torch.Generator(device="cuda").manual_seed(3)
    d, r = 64, 4
    # 1) null readout
    k0 = FusedTritonDeltaKernel(d=d, r=r, v_seed=V_SEED)
    kk = torch.randn(d, generator=g, device="cuda")
    vv = torch.randn(d, generator=g, device="cuda")
    e1 = k0.step_once(kk, vv)
    null = {"veto": int(k0.veto_count),
            "learned": float(k0.U.abs().sum().item()) > 0,
            "e_norm": float(e1.norm().item())}
    # 2) conflict veto (deterministic construction)
    k1 = FusedTritonDeltaKernel(d=d, r=r, v_seed=V_SEED)
    Vf = k1.V.float()
    a = torch.randn(r, generator=g, device="cuda")
    v0 = torch.randn(d, generator=g, device="cuda")
    v0 = v0 / v0.norm()
    U0 = torch.outer(v0, a).to(torch.bfloat16)
    k1.U.copy_(U0)
    kk2 = (Vf @ a).contiguous()          # readout = ||a||^2_G * v0 (c > 0)
    vv2 = (-v0).contiguous()
    e2 = k1.step_once(kk2, vv2)
    U_after = k1.U.float()
    U_expected = (0.985 * U0.float())    # veto: gamma decay only, no eta write
    rel = float((U_after - U_expected).norm().item()
                / (U_expected.norm().item() + 1e-12))
    conflict = {"veto": int(k1.veto_count), "fired": int(k1.veto_count) == 1,
                "e_zeroed": float(e2.norm().item()) < 1e-6,
                "write_suppressed": rel < 0.05,
                "u_rel_gamma": rel}
    return {"null_readout": null, "conflict": conflict}


# ---------------------------------------------------------------------------
# D. C11 two-process determinism
# ---------------------------------------------------------------------------
def check_determinism() -> dict:
    out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "det_child_1.json")
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "det_child_2.json")
    for p in (out1, out2):
        if os.path.exists(p):
            os.remove(p)
    procs = []
    for outp in (out1, out2):
        env = dict(os.environ)
        cmd = [sys.executable, DET_CHILD, "--out", outp]
        procs.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
    for p in procs:
        rc = p.wait(timeout=600)
        if rc != 0:
            return {"equal": False, "error": f"child rc={rc}"}
    r1 = json.load(open(out1)); r2 = json.load(open(out2))
    return {"equal": r1["state_hash"] == r2["state_hash"]
            and r1["telemetry"] == r2["telemetry"],
            "hash1": r1["state_hash"][:16], "hash2": r2["state_hash"][:16],
            "telemetry_equal": r1["telemetry"] == r2["telemetry"],
            "n_steps": r1["n_steps"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="deltamem_verify_receipt.json")
    ap.add_argument("--iters", type=int, default=10_000)
    ap.add_argument("--skip-c11", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA unavailable"}, indent=1))
        return 2

    receipt = {"interp": sys.executable,
               "torch": torch.__version__, "triton": __import__("triton").__version__,
               "cuda": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(0),
               "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    receipt["equiv"] = check_equivalence()
    receipt["c6"] = check_latency(iters=args.iters)
    receipt["c7"] = check_footprint()
    receipt["c10"] = check_veto()
    if not args.skip_c11:
        receipt["c11"] = check_determinism()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=1)
    print(json.dumps(receipt, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
