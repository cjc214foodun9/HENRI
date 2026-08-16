"""Phase 8.17 — In-Context Task Alignment & Viscoelastic Creep CUDA matrix.

Spec: HENRI-SPEC-2026-08-PHASE8.17-ALIGNMENT (SHA 1342944c...).
Pre-registered gates (phase817_in_context_alignment_design.md):
  G1: unitarity ||W^dag W - I||_F — c128 < 1e-6 (math), c64 < 1e-3 (live fidelity).
  G2: recovery < 0.05 on det-1 SU(3) consistent pairs; inconsistent >= 0.05.
  G3: thermal ratio >= 100x with real noise; SU(3) preserved; failing vs stable
      channel displacement discriminates.
  G4: live ARC 20-env gauntlet — STANDING BLOCKED. Demo-availability preflight
      (arcade envs expose examples: None -> BLOCKED_NO_DEMONSTRATIONS).
DONE_MARKER rc=1 failures=["G4"] is the designed honest outcome.
"""
import json
import math
import os
import sys
import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NB = 8192
M_PAIRS = 3
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p817_matrix_d65536.json")
DONE_MARKER = "DONE_MARKER"


def _su3_exp(nb, dtype, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    h = (torch.randn(nb, 3, 3, dtype=dtype, generator=g, device="cpu")
         + 1j * torch.randn(nb, 3, 3, dtype=dtype, generator=g, device="cpu"))
    h = (h + h.conj().transpose(-1, -2)) / 2.0
    tr = h.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
    h = h - tr * torch.eye(3, dtype=dtype) / 3.0
    return torch.matrix_exp(1j * h).to(device)


def compile_w(demo_inputs, demo_outputs):
    M = demo_inputs.shape[0]
    K = torch.einsum('mbij,mbkj->bik', demo_outputs, demo_inputs.conj()) / float(M)
    U, S, Vh = torch.linalg.svd(K)
    W = torch.einsum('bij,bjk->bik', U, Vh)
    det = torch.linalg.det(W)
    c = torch.pow(det.conj(), 1.0 / 3.0).unsqueeze(-1).unsqueeze(-1)
    return W * c


def unitarity_err(w):
    return float(
        (w.conj().transpose(-1, -2) @ w
         - torch.eye(3, dtype=w.dtype, device=w.device)).norm().item()
    )


def main():
    print(f"Substrate: {DEVICE} | torch {torch.__version__} | cuda {torch.version.cuda}")
    results = {"device": DEVICE, "nb": NB, "m_pairs": M_PAIRS}
    failures = []

    # ---- G1: unitarity (dual dtype) ----
    g1 = {}
    for dtype, name in [(torch.complex64, "c64"), (torch.complex128, "c128")]:
        w = _su3_exp(NB, dtype, seed=1, device=DEVICE)
        ux = torch.stack([_su3_exp(NB, dtype, seed=2 + i, device=DEVICE)
                          for i in range(M_PAIRS)])
        uy = torch.stack([w @ u for u in ux])
        t0 = time.time()
        wc = compile_w(ux, uy)
        g1[name] = {"unitary_err": unitarity_err(wc),
                    "det_min": float(torch.linalg.det(wc).abs().min().item()),
                    "compile_s": float(time.time() - t0)}
    g1_pass = g1["c128"]["unitary_err"] < 1e-6 and g1["c64"]["unitary_err"] < 1e-3
    results["G1_817"] = {"pass": g1_pass, **g1}
    print(f"G1_817: pass={g1_pass} c128={g1['c128']['unitary_err']:.3e} "
          f"c64={g1['c64']['unitary_err']:.3e}")
    if not g1_pass:
        failures.append("G1")

    # ---- G2: recovery (det-1 consistent) + discrimination ----
    w = _su3_exp(NB, torch.complex128, seed=11, device=DEVICE)
    ux = torch.stack([_su3_exp(NB, torch.complex128, seed=12 + i, device=DEVICE)
                      for i in range(M_PAIRS)])
    uy = torch.stack([w @ u for u in ux])
    wc = compile_w(ux, uy)
    g2_cons = max(float((wc @ u - y).norm().item()) for u, y in zip(ux, uy))
    ux2 = torch.stack([_su3_exp(NB, torch.complex128, seed=21 + i, device=DEVICE)
                       for i in range(M_PAIRS)])
    uy2 = torch.stack([_su3_exp(NB, torch.complex128, seed=31 + i, device=DEVICE)
                       for i in range(M_PAIRS)])
    wc2 = compile_w(ux2, uy2)
    g2_incons = max(float((wc2 @ u - y).norm().item()) for u, y in zip(ux2, uy2))
    g2_pass = g2_cons < 0.05 and g2_incons >= 0.05
    results["G2_817"] = {"pass": g2_pass, "recovery_err": g2_cons,
                         "inconsistent_err": g2_incons}
    print(f"G2_817: pass={g2_pass} recovery={g2_cons:.3e} "
          f"inconsistent={g2_incons:.3f}")
    if not g2_pass:
        failures.append("G2")

    # ---- G3: anisotropic creep (real noise, channel discrimination) ----
    from adaptive_viscoelastic_thermostat import AdaptiveViscoelasticThermostat
    therm = AdaptiveViscoelasticThermostat(d_model=65536)
    delta = torch.zeros(NB, device=DEVICE)
    delta[:10] = 1.0
    delta[10:] = 0.01
    w3 = _su3_exp(NB, torch.complex64, seed=41, device=DEVICE)
    grad = torch.zeros_like(w3)
    noise = (torch.randn(w3.shape, dtype=torch.complex64, generator=torch.Generator().manual_seed(5))
             .to(DEVICE))
    out, tele = therm.apply_anisotropic_langevin_creep(
        w3, grad, delta, t_base=1e-4, alpha=5.0, noise=noise)
    disp_fail = float((out[:10] - w3[:10]).abs().norm().item())
    disp_stable = float((out[10:] - w3[10:]).abs().norm().item())
    g3_pass = (tele["thermal_ratio"] >= 100.0 and tele["n_failing"] == 10
               and tele["su3_det_min"] > 0.999
               and disp_fail > 10.0 * disp_stable + 1e-6)
    results["G3_817"] = {"pass": g3_pass, **tele,
                         "disp_failing": disp_fail, "disp_stable": disp_stable}
    print(f"G3_817: pass={g3_pass} ratio={tele['thermal_ratio']:.1f}x "
          f"det_min={tele['su3_det_min']:.6f} disp_fail={disp_fail:.3e} "
          f"disp_stable={disp_stable:.3e}")
    if not g3_pass:
        failures.append("G3")

    # ---- G4: live ARC gauntlet — standing blocked; demo-availability preflight ----
    g4 = {"status": "BLOCKED_NO_DEMONSTRATIONS", "solved": 0, "envs": 20}
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
        import arc_agi
        arc = arc_agi.Arcade()
        envs = [e.game_id if hasattr(e, "game_id") else e
                for e in arc.available_environments][:3]
        demos_seen = []
        for eid in envs:
            try:
                g = arc.make(eid)
                ex = getattr(g, "examples", None)
                demos_seen.append({"env": eid, "examples": None if ex is None
                                   else len(ex)})
            except Exception as exc:
                demos_seen.append({"env": eid, "examples": f"ERR:{exc}"})
        g4["demo_preflight"] = demos_seen
    except Exception as exc:
        g4["demo_preflight"] = {"error": str(exc)}
    g4_pass = False
    results["G4_817"] = {"pass": g4_pass, **g4}
    failures.append("G4")
    print(f"G4_817: pass=False BLOCKED_NO_DEMONSTRATIONS (standing)")

    results["failures"] = failures
    results["rc"] = 1 if failures else 0
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"{DONE_MARKER} rc={results['rc']} failures={failures}")


if __name__ == "__main__":
    main()
