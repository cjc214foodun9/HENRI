"""Phase 8.18 — Field-to-Wave Isomorphic Transducer CUDA matrix.

Spec: HENRI-SPEC-2026-08-PHASE8.18-TRANSDUCER (SHA 158c02c7...).
Pre-registered gates (experiments/sweeps/phase818_transducer_design.md):
G1 round-trip < 1e-5; G2 non-commutativity > 0.5 (+ commuting control);
G3 Triton su3 log kernel sustained latency <= 50 us (CUDA-event interval,
20 back-to-back post-warmup launches);
G4 live ARC > 0 solved — pre-registered BLOCKED (BLOCKED_NO_DEMONSTRATIONS,
no pseudo-demos). DONE_MARKER rc=1 failures=["G4"] by design.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "HENRI V2"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NB = 8192

results = {}
failures = []


def gell_mann_basis(dtype=torch.complex64):
    import math
    b = []
    e = lambda i, j: (lambda m: (m.__setitem__((i, j), torch.tensor(1.0, dtype=dtype)), m)[1])(
        torch.zeros(3, 3, dtype=dtype)
    )
    b.append(e(0, 1) + e(1, 0))
    b.append(-1j * e(0, 1) + 1j * e(1, 0))
    b.append(torch.diag(torch.tensor([1.0, -1.0, 0.0], dtype=dtype)))
    b.append(e(0, 2) + e(2, 0))
    b.append(-1j * e(0, 2) + 1j * e(2, 0))
    b.append(e(1, 2) + e(2, 1))
    b.append(-1j * e(1, 2) + 1j * e(2, 1))
    b.append(torch.diag(torch.tensor([1.0, 1.0, -2.0], dtype=dtype)) / math.sqrt(3.0))
    return torch.stack(b)


def rand_su3(n, basis, theta_scale=1.0, seed=0):
    g = torch.Generator().manual_seed(seed)
    theta = (torch.rand(n, 8, generator=g) * 2 - 1) * theta_scale
    theta = theta.to(basis.device).to(basis.dtype)  # CPU gen -> device (CUDA trap)
    alg = 1j * torch.einsum("na,abc->nbc", theta, basis)
    return torch.matrix_exp(alg)


from universal_data_transducer import SU3FieldWaveTransducer  # noqa: E402
import qfhrr_kernels  # noqa: E402

print(f"[p818] device={DEVICE} torch={torch.__version__} "
      f"triton={getattr(qfhrr_kernels, '_HAS_TRITON', False)}", flush=True)

trans = SU3FieldWaveTransducer(gell_mann_basis()).to(DEVICE)

# ---- G1: round-trip @ NB=8192, both dtypes ----
for dt, key in ((torch.complex128, "g1_c128"), (torch.complex64, "g1_c64")):
    basis = trans.basis.to(dt)
    U = rand_su3(NB, basis, seed=1).to(DEVICE)
    w = trans.field_to_wave(U.unsqueeze(0))
    rec = trans.wave_to_field(w)[0]
    err = float((U - rec).norm(dim=(-2, -1)).mean().item())
    results[key] = err
    ok = err < 1e-5
    print(f"[p818] G1 {key} round-trip={err:.3e} {'PASS' if ok else 'FAIL'}", flush=True)
    if not ok:
        failures.append(key)

# ---- G2: non-commutativity + commuting control ----
UA = rand_su3(NB, trans.basis, seed=2).to(DEVICE)
UB = rand_su3(NB, trans.basis, seed=3).to(DEVICE)
d_nc = float((trans.field_to_wave((UA @ UB).unsqueeze(0))
              - trans.field_to_wave((UB @ UA).unsqueeze(0))).norm().item())
g = torch.Generator().manual_seed(4)
theta = (torch.rand(NB, 8, generator=g) * 2 - 1) * 0.5
theta = theta.to(trans.basis.device).to(trans.basis.dtype)  # CPU gen -> device
alg1 = 1j * torch.einsum("na,abc->nbc", theta, trans.basis)
U_same = torch.matrix_exp(alg1)
U_com = U_same @ U_same
d_com = float((trans.field_to_wave((U_same @ U_com).unsqueeze(0))
               - trans.field_to_wave((U_com @ U_same).unsqueeze(0))).norm().item())
results["g2_noncomm"] = d_nc
results["g2_commuting_control"] = d_com
ok2 = d_nc > 0.5 and d_com < 0.05
print(f"[p818] G2 non-comm={d_nc:.3f} (gate>0.5) commuting={d_com:.3e} (control<0.05) "
      f"{'PASS' if ok2 else 'FAIL'}", flush=True)
if not ok2:
    failures.append("g2")

# ---- G3: Triton 3x3 matrix log latency (CUDA-event sustained interval) ----
if DEVICE.type == "cuda" and getattr(qfhrr_kernels, "_HAS_TRITON", False):
    U = rand_su3(NB, trans.basis, seed=5).to(DEVICE)
    # torch reference for correctness
    evals, evecs = torch.linalg.eig(U)
    ref = evecs @ torch.diag_embed(torch.log(evals)) @ evecs.conj().transpose(-2, -1)
    lg = qfhrr_kernels.su3_matrix_log_triton(U)
    err_log = float((lg - ref).abs().norm(dim=(-2, -1)).mean().item())
    results["g3_log_err_vs_eig"] = err_log
    ok3a = err_log < 1e-3
    print(f"[p818] G3 triton log mean err vs eig={err_log:.3e} "
          f"{'PASS' if ok3a else 'FAIL'}", flush=True)
    if not ok3a:
        failures.append("g3_correctness")
    # latency: CUDA events over 20 back-to-back post-warmup launches
    for _ in range(5):
        qfhrr_kernels.su3_matrix_log_triton(U)
    torch.cuda.synchronize()
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    n_rep = 20
    start_ev.record()
    for _ in range(n_rep):
        qfhrr_kernels.su3_matrix_log_triton(U)
    end_ev.record()
    torch.cuda.synchronize()
    t_us = start_ev.elapsed_time(end_ev) * 1e3 / n_rep
    results["g3_latency_us"] = t_us
    ok3 = t_us <= 50.0
    print(f"[p818] G3 triton sustained latency={t_us:.2f} us (gate<=50) "
          f"{'PASS' if ok3 else 'FAIL'}", flush=True)
    if not ok3:
        failures.append("g3_latency")
else:
    results["g3_latency_us"] = None
    results["g3_log_err_vs_eig"] = None
    print("[p818] G3 SKIPPED (no CUDA or no triton)", flush=True)

# ---- G4: live ARC demo preflight (pre-registered BLOCKED) ----
g4 = {"status": "BLOCKED_NO_DEMONSTRATIONS", "solved": 0, "attempted": 0,
      "reason": "arc_agi environments expose examples: None (OBSERVED 8.17 preflight)"}
try:
    import arc_agi
    arcade = arc_agi.Arcade()
    envs = [e.game_id if hasattr(e, "game_id") else e
            for e in arcade.available_environments][:3]
    demo_status = []
    for eid in envs:
        try:
            game = arcade.make(eid)
            demo_status.append({"env": eid, "examples": None if not game.examples
                                else len(game.examples)})
        except Exception as ex:
            demo_status.append({"env": eid, "error": str(ex)})
    g4["env_demo_probe"] = demo_status
    g4["any_demos"] = any(d.get("examples") for d in demo_status)
    print(f"[p818] G4 preflight: {demo_status}", flush=True)
except Exception as exc:
    g4["preflight_error"] = str(exc)
    print(f"[p818] G4 preflight failed: {exc}", flush=True)
results["g4"] = g4
failures.append("G4")  # pre-registered standing block

print(f"[p818] DONE_MARKER rc={1 if failures else 0} failures={failures}", flush=True)

out_path = os.environ.get("JEPA_DM_OUT", "/tmp/p818_matrix.json")
with open(out_path, "w") as f:
    json.dump({"phase": "8.18", "device": str(DEVICE), "nb": NB,
               "results": results, "failures": failures,
               "done_marker_rc": 1 if failures else 0}, f, indent=2)
print(f"[p818] wrote {out_path}", flush=True)
sys.exit(1 if failures else 0)
