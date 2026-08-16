"""Phase 8.16.1 — Shared-Projection Calibration & Gauge Realignment CUDA matrix.

Spec: HENRI-SPEC-2026-08-PHASE8.16.1-REFORM (SHA bdf4602b...).
Pre-registered gates (phase8161_shared_projection_design.md):
  G1-8.16.1: L_valid < 1e-4 (shared evaluator, codebook round-trip pairs)
  G2-8.16.1: ratio L_mism / max(L_valid, 1e-12) >= 10.0
  G3-8.16.1: Triton phase-ring LUT sustained latency <= 50.0 us (CUDA-event, 20 launches)
  G4-8.16.1: top-1 codebook recall >= 0.9900 (noisy queries, Triton path)
DONE marker: rc=0 + failures=[] only when all gates pass.
"""
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from henri_functor_flow import DiagrammaticSharedEgressEvaluator  # noqa: E402
from qfhrr_kernels import (  # noqa: E402
    build_cos_lut,
    phase_codes_to_wave,
    qfhrr_similarity,
    qfhrr_similarity_triton,
    wave_to_phase_codes,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
D, NB, LAT = 65536, 8192, 2048
S = 5.0e-6
M_SYMBOLS = 256
OUT = os.environ.get("JEPA_DM_OUT", "p8161_matrix_d65536.json")


def _bc2w(q):
    n, nb, _ = q.shape
    return phase_codes_to_wave(q.reshape(n * nb, 4)).reshape(n, nb, 8)


def _bw2c(w):
    n, nb, _ = w.shape
    return wave_to_phase_codes(w.reshape(n * nb, 8)).reshape(n, nb, 4)


def main():
    t0 = time.time()
    torch.manual_seed(20260816)
    gates = {}

    # ---- G1/G2: shared evaluator at production D ----
    gen = torch.Generator().manual_seed(7)
    q1 = torch.randint(0, 256, (M_SYMBOLS, NB, 4), generator=gen, device="cpu").to(torch.uint8).to(DEVICE)
    w1 = _bc2w(q1)
    q1_rt = _bw2c(w1)
    w1_rt = _bc2w(q1_rt)
    q2 = torch.randint(0, 256, (M_SYMBOLS, NB, 4), generator=torch.Generator().manual_seed(9),
                       device="cpu").to(torch.uint8).to(DEVICE)
    w2 = _bc2w(q2)

    ev = DiagrammaticSharedEgressEvaluator(dim=D, latent_dim=LAT, scale=S).to(DEVICE)
    with torch.no_grad():
        l_valid = ev(w1.reshape(M_SYMBOLS, -1), w1_rt.reshape(M_SYMBOLS, -1)).item()
        l_mism = ev(w1.reshape(M_SYMBOLS, -1), w2.reshape(M_SYMBOLS, -1)).item()
    ratio = l_mism / max(l_valid, 1e-12)
    gates["G1_8161"] = {
        "pass": l_valid < 1e-4,
        "detail": {"l_valid": l_valid, "l_mism": l_mism, "ratio": ratio, "s": S, "k": LAT},
    }
    gates["G2_8161"] = {
        "pass": ratio >= 10.0,
        "detail": {"l_valid": l_valid, "l_mism": l_mism, "ratio": ratio},
    }

    # ---- G4: codebook recall (Triton path) ----
    lut = build_cos_lut(256, device=DEVICE)
    q_flat_cb = wave_to_phase_codes(w1.reshape(M_SYMBOLS * NB, 8)).reshape(M_SYMBOLS, -1).to(torch.uint8)
    recalls = {}
    for sigma in (0.01, 0.05, 0.1):
        noise = (torch.randn(M_SYMBOLS, NB, 8, generator=torch.Generator().manual_seed(3)) * sigma).to(DEVICE)
        q_noisy = _bw2c((w1 + noise).reshape(M_SYMBOLS, NB, 8)).reshape(M_SYMBOLS, -1).to(torch.uint8)
        top1 = 0
        for i in range(M_SYMBOLS):
            sims = qfhrr_similarity_triton(q_noisy[i], q_flat_cb, lut)
            top1 += int(sims.argmax().item() == i)
        recalls[str(sigma)] = top1 / M_SYMBOLS
    r_min = min(recalls.values())
    gates["G4_8161"] = {
        "pass": r_min >= 0.99,
        "detail": {"recalls": recalls, "n": M_SYMBOLS},
    }

    # ---- G3: Triton LUT sustained latency (CUDA-event interval, 20 launches) ----
    q_a = torch.randint(0, 256, (NB * 4,), device=DEVICE).to(torch.uint8)
    q_b = torch.randint(0, 256, (NB * 4,), device=DEVICE).to(torch.uint8)
    q_b2 = torch.randint(0, 256, (NB * 4,), device=DEVICE).to(torch.uint8)
    q_b3 = torch.randint(0, 256, (NB * 4,), device=DEVICE).to(torch.uint8)
    qb = torch.stack([q_b, q_b2, q_b3])  # 3 engrams, [M=3, D/2]
    # warmup
    for _ in range(5):
        _ = qfhrr_similarity_triton(q_a, qb, lut)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    N_ITER = 20
    start.record()
    for _ in range(N_ITER):
        _ = qfhrr_similarity_triton(q_a, qb, lut)
    end.record()
    torch.cuda.synchronize()
    t_us = start.elapsed_time(end) * 1e3 / N_ITER
    gates["G3_8161"] = {
        "pass": t_us <= 50.0,
        "detail": {"latency_us": t_us, "iters": N_ITER, "m": qb.shape[0], "d": D},
    }

    failures = [g for g, v in gates.items() if not v["pass"]]
    rc = 0 if not failures else 1
    result = {
        "phase": "8.16.1",
        "commit": os.environ.get("HENRI_COMMIT", "local"),
        "device": str(DEVICE),
        "rc": rc,
        "failures": failures,
        "gates": gates,
        "elapsed_s": round(time.time() - t0, 2),
        "done_marker": True,
    }
    with open(OUT, "w") as f:
        json.dump(result, f, indent=2)
    print(f"DONE_MARKER rc={rc} failures={failures}")
    for g, v in gates.items():
        print(f"  {g}: pass={v['pass']} {json.dumps(v['detail'])[:160]}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
