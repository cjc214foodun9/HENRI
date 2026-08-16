"""Phase 8.13 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase813_amplitude_preserving_ingress_design.md
Blueprint: docs-HENRI_V2_PHASE_8_12_POSTMORTEM_AND_PHASE_8_13....pdf.pdf
(SHA 5e435cd95ccd8f25a8e95146efa3c68d25626348bdacd56932ef0cdef265afbf)

Local contract (d=512, OBSERVED): pre-registered G1 KILL established by
exact math, D-independent:
- COLOR-BLIND: amplitude-weighted color is cosine-scale-invariant:
  cos(3z,6z) = 1.0 exactly for same-position grids (any carriers).
- SHARED-SUPPORT: position-carrier superposition adds shared cells
  coherently: B-ring vs C-line share 3 cells -> cos = 6/sqrt(96) =
  0.6113 (D cancels; O(1/sqrt(D)) finite-D correction).
- LEGACY CONTROL: 7.3/7.4 encoder (incommensurate+bg_mask) phase-encodes
  color -> 0.0000 (color pair), 0.0033 (shared pair). Discriminative
  channel is COLOR PHASE, not amplitude.

Gates at D=65,536 (this runner):
- G1 KILL confirm: color pair cos >= 0.0100 AND shared pair cos >= 0.0100
  (both must FAIL the blueprint gate < 0.0100 = kill reproduced at scale).
- G3 LEGACY CONTROL: legacy incommensurate+bg_mask < 0.0100 on both pairs
  (dominance at scale).
- G2 FIT (surviving property, NOT promotion): 8.11 transition fits the
  translation operator on amplitude waves, held-out loss <= 0.05.
- G3 LATENCY: ingress forward + forward_complex + egress <= 1.0 ms at
  D=65,536 (30x30 grid, 50 iters).
- G4 DEMO BLOCK (pre-registered EXPECTED): probe production ARC demo
  ingress; record BLOCKED_NO_DEMONSTRATIONS (arcade examples: None,
  observed lf52/tn36/sc25; no provenance API). Never fabricate demos.
- G5 DEFAULT-OFF: production EFEPlanner default = LowRankCoupledTransition.

Multi-arm DONE marker: rc=1 if ANY gate FAILS (aggregated, learned lesson).
Env: JEPA_DM_OUT=/tmp/p813_matrix_d65536.json; HENRI_SMOKE=1 -> bounded.
"""

import json
import math
import os
import time

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p813_matrix_d65536.json")

NB = 8192
BD = 8
D = NB * BD

results = {}
failures = []


def record(name, ok, **kv):
    results[name] = {"ok": bool(ok), **kv}
    print(f"[{name}] ok={ok} " + " ".join(
        f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
        for k, v in kv.items()), flush=True)
    if not ok:
        failures.append(name)


def main():
    from amplitude_preserving_ingress import AmplitudePreservingComplexIngress
    from complex_phase_transition import NativeComplexWaveTransition
    from henri_vision_encoder import HENRIVisionEncoder
    from efe_planner import EFEPlanner

    ing = AmplitudePreservingComplexIngress(
        dimension=D, num_blocks=NB, block_size=BD, device=DEVICE
    )
    tr = NativeComplexWaveTransition(
        dimension=D, num_blocks=NB, block_dim=BD, device=DEVICE
    )
    leg = HENRIVisionEncoder(
        d_model=D, k_blocks=NB, block_dim=BD, device=DEVICE,
        spatial_basis_kind="incommensurate", bg_mask=True,
    )
    assert DEVICE == "cuda", "runner requires CUDA target"

    GA = torch.tensor([[0, 0, 0], [0, 3, 0], [0, 0, 0]], dtype=torch.long)
    GA6 = torch.tensor([[0, 0, 0], [0, 6, 0], [0, 0, 0]], dtype=torch.long)
    GB = torch.tensor([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=torch.long)
    GC = torch.tensor([[0, 0, 2], [0, 0, 2], [0, 0, 2]], dtype=torch.long)

    def cos(fn, a, b):
        za = fn(a).reshape(-1)
        zb = fn(b).reshape(-1)
        return float(torch.abs(torch.vdot(za, zb)) / (torch.norm(za) * torch.norm(zb) + 1e-12))

    def leg_cos(a, b):
        za = leg.encode_spatial_grid(a).reshape(-1)
        zb = leg.encode_spatial_grid(b).reshape(-1)
        return float(torch.abs(torch.dot(za, zb)) / (torch.norm(za) * torch.norm(zb) + 1e-12))

    # ---- G1 KILL confirm at scale (both pairs must violate < 0.0100) ----
    c_color = cos(ing.forward, GA, GA6)
    c_shared = cos(ing.forward, GB, GC)
    g1_ok = (c_color >= 0.0100) and (c_shared >= 0.0100)
    record("G1_KILL_COLOR", g1_ok, cos=c_color, threshold=0.0100,
           note="amplitude ingress color pair must FAIL blueprint gate")
    record("G1_KILL_SHARED", g1_ok, cos=c_shared, threshold=0.0100,
           note="shared-support pair must FAIL blueprint gate")

    # ---- G3 legacy control dominance at scale ----
    l_color = leg_cos(GA, GA6)
    l_shared = leg_cos(GB, GC)
    g3_ok = (l_color < 0.0100) and (l_shared < 0.0100)
    record("G3_LEGACY_COLOR", g3_ok, cos=l_color, threshold=0.0100)
    record("G3_LEGACY_SHARED", g3_ok, cos=l_shared, threshold=0.0100)
    record("G3_DOMINANCE", (l_color < c_color) and (l_shared < c_shared),
           legacy_color=l_color, amp_color=c_color,
           legacy_shared=l_shared, amp_shared=c_shared)

    # ---- G2 FIT on amplitude waves (surviving property) ----
    z0 = ing.forward(GA).reshape(-1)
    z1 = ing.forward(torch.tensor([[0, 0, 0], [0, 0, 3], [0, 0, 0]], dtype=torch.long)).reshape(-1)
    pre = tr.update_phase_complex(z0, z1, action_idx=0, lr=1.0)
    g_ho = torch.tensor([[0, 4, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    g_ho_s = torch.tensor([[0, 0, 4], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    z_ho = ing.forward(g_ho).reshape(-1)
    z_ho_s = ing.forward(g_ho_s).reshape(-1)
    z_pred = tr.forward_complex(z_ho, action_idx=0)
    fit_loss = float(torch.angle(z_ho_s * torch.conj(z_pred)).detach().abs().mean())
    record("G2_FIT", fit_loss <= 0.05, loss=fit_loss, threshold=0.05, pre=pre)

    # ---- G3 LATENCY at D=65,536 ----
    big = torch.randint(0, 10, (30, 30), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        z_big = ing.forward(big)
        zr = tr.forward_complex(z_big.reshape(-1), action_idx=0)
        out = tr.project_to_real_egress(zr)
    iters = 50 if not SMOKE else 10
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(iters):
            z_big = ing.forward(big)
            zr = tr.forward_complex(z_big.reshape(-1), action_idx=0)
            out = tr.project_to_real_egress(zr)
    torch.cuda.synchronize()
    dt_ms = (time.perf_counter() - t0) / iters * 1e3
    record("G3_LATENCY", dt_ms <= 1.0, ms=dt_ms, threshold=1.0,
           shape=list(out.shape), finite=bool(torch.isfinite(out).all()))

    # ---- G4 demo block (pre-registered EXPECTED BLOCKED) ----
    demo_status = "BLOCKED_NO_DEMONSTRATIONS"
    try:
        import arc_agi  # noqa: F401 — probe availability only
        demo_note = "arc_agi importable; examples provenance API absent (standing)"
    except Exception as e:
        demo_note = f"arc_agi unavailable on runner host ({type(e).__name__})"
    record("G4_DEMO_BLOCK", True, status=demo_status, note=demo_note,
           expected="BLOCKED_NO_DEMONSTRATIONS (never fabricated)")

    # ---- G5 default-OFF ----
    p = EFEPlanner(num_blocks=NB, d_model=D)
    record("G5_DEFAULT_OFF", type(p.transition).__name__ == "LowRankCoupledTransition",
           transition_type=type(p.transition).__name__)

    ok = len(failures) == 0
    print(f"DONE_MARKER rc={0 if ok else 1} failures={failures}", flush=True)
    with open(OUT, "w") as f:
        json.dump({
            "phase": "8.13", "device": DEVICE, "D": D, "smoke": SMOKE,
            "results": results, "failures": failures, "rc": 0 if ok else 1,
        }, f, indent=2)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
