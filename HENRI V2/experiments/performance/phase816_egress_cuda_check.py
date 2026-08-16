"""Phase 8.16 — Diagrammatic FunctorFlow Egress CUDA verification matrix.

Spec: HENRI-SPEC-2026-08-PHASE8.16-EGRESS (SHA 2ec60178...).
Gates (pre-registered in phase816_egress_design.md):
  G1-EGRESS  Diagrammatic obstruction loss: as-shipped spec init must FAIL
             (valid >= 1e-4) at D=65,536 — full-scale confirmation of the
             pre-registered falsification (noise floor is D-INDEPENDENT).
  G2-EGRESS  Top-1 phase-ring codebook recall >= 0.99 (256 symbols,
             sigma 0.01/0.05/0.1) at D=65,536 on CUDA.
  G3-EGRESS  Triton phase-ring LUT unbinding latency <= 50.0 us @D=65,536
             (CUDA-event sustained interval over 20 launches).
  G4-EGRESS  Live ARC task grounding — standing BLOCKED_NO_DEMONSTRATIONS.
DONE_MARKER: rc=0 only when all evaluated gates pass; any failure -> rc=1.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from henri_functor_flow import DiagrammaticEgressEvaluator
from qfhrr_kernels import (
    phase_codes_to_wave,
    wave_to_phase_codes,
    qfhrr_similarity_triton,
    qfhrr_similarity_torch,
    build_cos_lut,
    K_PHASE,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 65536
NB = D // 8
LAT = 2048
N_TR, N_HO = 128, 64
M_SYMBOLS = 256
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p816_matrix_d65536.json")
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"


def _bc2w(q, nb):
    n, nb_, _ = q.shape
    return phase_codes_to_wave(q.reshape(n * nb_, 4)).reshape(n, nb_, 8)


def _bw2c(w, nb):
    n, nb_, _ = w.shape
    return wave_to_phase_codes(w.reshape(n * nb_, 8)).reshape(n, nb_, 4)


def main():
    results = {}
    failures = []
    t0 = time.time()
    assert torch.cuda.is_available(), "CUDA required"

    # --------------------------------------------------------------- G1
    # as-shipped spec init (two independent default Linear projections)
    q_tr = torch.randint(0, K_PHASE, (N_TR, NB, 4), device=DEVICE).to(torch.uint8)
    w_tr = _bc2w(q_tr, NB).reshape(N_TR, D)
    a_tr = _bc2w(_bw2c(w_tr.reshape(N_TR, NB, 8), NB), NB).reshape(N_TR, D)
    q_ho = torch.randint(0, K_PHASE, (N_HO, NB, 4), device=DEVICE).to(torch.uint8)
    w_ho = _bc2w(q_ho, NB).reshape(N_HO, D)
    a_ho = _bc2w(_bw2c(w_ho.reshape(N_HO, NB, 8), NB), NB).reshape(N_HO, D)
    q_mis = torch.randint(0, K_PHASE, (N_HO, NB, 4), device=DEVICE).to(torch.uint8)
    w_mis = _bc2w(q_mis, NB).reshape(N_HO, D)

    ev = DiagrammaticEgressEvaluator(dim=D, latent_dim=LAT).to(DEVICE)
    with torch.no_grad():
        lv = ev(w_ho, a_ho).item()
        lm = ev(w_ho, w_mis).item()
    g1_valid = lv >= 1e-4          # spec gate FALSIFIED if valid >= 1e-4
    g1_ordered = lv < lm           # spec ordering must be INVERTED on real data
    results["gate_g1"] = {
        "pass": g1_valid,  # pass = falsification confirmed (valid >= 1e-4)
        "detail": {
            "valid_loss": lv,
            "mism_loss": lm,
            "ordering_inverted": g1_ordered,
            "note": "as-shipped spec init; gate 1e-4 unreachable (noise floor 2/3)",
        },
    }
    if not g1_valid:
        failures.append("G1")

    # --------------------------------------------------------------- G2
    # 256-symbol phase-ring codebook recall @D=65,536 on CUDA (Triton path)
    lut = build_cos_lut(DEVICE)
    q_cb = torch.randint(0, K_PHASE, (M_SYMBOLS, NB, 4), device=DEVICE).to(torch.uint8)
    w_cb = _bc2w(q_cb, NB).reshape(M_SYMBOLS, D)
    q_flat = q_cb.reshape(M_SYMBOLS, -1)
    recalls = {}
    for sigma in (0.01, 0.05, 0.1):
        gen = torch.Generator(device=DEVICE).manual_seed(31)
        noise = torch.randn(M_SYMBOLS, NB, 8, device=DEVICE, generator=gen) * sigma
        q_noisy = _bw2c((w_cb.reshape(M_SYMBOLS, NB, 8) + noise), NB).reshape(M_SYMBOLS, -1)
        hits = 0
        for i in range(M_SYMBOLS):
            sims = qfhrr_similarity_triton(q_noisy[i], q_flat, lut)
            hits += int(sims.argmax().item() == i)
        recalls[str(sigma)] = hits / M_SYMBOLS
    g2_ok = all(v >= 0.99 for v in recalls.values())
    results["gate_g2"] = {"pass": g2_ok, "detail": {"recalls": recalls}}
    if not g2_ok:
        failures.append("G2")

    # --------------------------------------------------------------- G3
    # Triton LUT latency: CUDA-event sustained interval over 20 launches @D=65,536
    q_a = torch.randint(0, K_PHASE, (NB * 4,), device=DEVICE).to(torch.uint8)  # flat [nb,4] codes
    q_b = torch.randint(0, K_PHASE, (1, NB * 4), device=DEVICE).to(torch.uint8)
    _ = qfhrr_similarity_triton(q_a, q_b, lut)  # warmup
    torch.cuda.synchronize()
    n_reps = 20
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n_reps):
        _ = qfhrr_similarity_triton(q_a, q_b, lut)
    end.record()
    torch.cuda.synchronize()
    sustained_us = start.elapsed_time(end) * 1000.0 / n_reps
    g3_ok = sustained_us <= 50.0
    results["gate_g3"] = {"pass": g3_ok, "detail": {"sustained_us": sustained_us, "reps": n_reps}}
    if not g3_ok:
        failures.append("G3")

    # --------------------------------------------------------------- G4
    results["gate_g4"] = {
        "pass": False,
        "detail": {"status": "BLOCKED_NO_DEMONSTRATIONS", "envs": "20/20", "note": "standing; not attempted"},
    }
    failures.append("G4")  # always failed: blocked

    results["done"] = "DONE_MARKER"
    results["rc"] = 0 if not failures else 1
    results["failures"] = failures
    results["elapsed_s"] = time.time() - t0
    results["device"] = torch.cuda.get_device_name(0)
    results["d"] = D
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "gate_g1"}, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
