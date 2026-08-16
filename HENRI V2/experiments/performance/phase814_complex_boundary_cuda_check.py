"""Phase 8.14 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase814_complex_boundary_wiring_design.md
Roadmap: Project HENRI V2 Strategic R&D Roadmap.pdf (SHA 0ca9f7a1...).

Arms:
- G1 scale: complex_cos on re-paired waves (color / shared / disjoint
  hard pairs) <= 0.02 (roadmap G1 < 0.05 satisfied as corollary).
- G2 paired held-out transfer: 32 deterministic translation pairs
  (smoke 8); fit 24 (smoke 6), eval 8 (smoke 2) held-out. Real arm =
  8.12 G4 recipe (forward + update_wirtinger lr=0.05). Complex arm =
  un_realify -> forward_complex + update_phase_complex lr=0.05 ->
  re_realify. ACCEPT iff post_complex <= 0.90 AND
  post_complex - post_real >= +0.02.
- G3 demo block (BLOCKED_NO_DEMONSTRATIONS, never fabricated).
- G4 latency: un_realify + forward_complex + re_realify cycle
  <= 2.0 ms @ D=65,536.
- G5 default-OFF: toy-scale EFEPlanner.transition == LowRankCoupledTransition.
- G6 round-trip: re_realify(un_realify(w)) == w, max err < 1e-6 @ scale.

DONE marker aggregates gate failures (rc=1 if any pre-registered ACCEPT
gate fails; kill gates firing = rc=1 with KILL verdict).
"""

import json
import os
import time

import torch

from complex_boundary import (
    complex_cosine,
    complex_cycle,
    re_realify,
    un_realify,
)
from complex_phase_transition import NativeComplexWaveTransition
from henri_vision_encoder import HENRIVisionEncoder

D = int(os.environ.get("JEPA_D", "65536"))
NB, BD = 8192, 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT_PATH = os.environ.get("JEPA_DM_OUT", "/tmp/p814_matrix_d65536.json")

GRID_A = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
GRID_A6 = [[0, 6, 0], [6, 2, 6], [0, 6, 0]]  # color 3 -> 6, same shape
GRID_B = [[0, 0, 1], [0, 2, 1], [0, 1, 1]]  # shares 3 cells with C
GRID_C = [[0, 2, 0], [2, 1, 2], [0, 2, 0]]
HARD = [("color", GRID_A, GRID_A6), ("shared", GRID_B, GRID_C),
        ("disjoint", GRID_A, GRID_C)]


def real_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def shift_right(grid, cols=3):
    return [[row[(x - 1) % cols] for x in range(cols)] for row in grid]


def main():
    out = {"phase": "8.14", "device": DEVICE, "D": D, "smoke": SMOKE,
           "arms": {}, "verdicts": {}}
    failures = []

    enc = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD,
                             device=DEVICE, spatial_basis_kind="incommensurate",
                             bg_mask=True)
    # transition phase space = native complex state space (D/2)
    tr_c = NativeComplexWaveTransition(dimension=D // 2, num_actions=16,
                                       device=DEVICE, num_blocks=D // 2,
                                       block_dim=1)
    tr_r = NativeComplexWaveTransition(dimension=D // 2, num_actions=16,
                                       device=DEVICE, num_blocks=D // 2,
                                       block_dim=1)

    # ---------------- G1 scale discrimination ------------------------------
    g1 = {}
    for name, ga, gb in HARD:
        wa = enc.encode_spatial_grid(ga).squeeze(0)
        wb = enc.encode_spatial_grid(gb).squeeze(0)
        cc = complex_cosine(un_realify(wa), un_realify(wb))
        rc = real_cos(wa, wb)
        g1[name] = {"complex_cos": round(cc, 7), "real_cos": round(rc, 7),
                    "ok": bool(cc <= 0.02)}
    out["arms"]["G1_scale"] = g1
    g1_ok = all(v["ok"] for v in g1.values())
    out["verdicts"]["G1"] = "SCALE_PASS" if g1_ok else "KILL"
    if not g1_ok:
        failures.append("G1")

    # ---------------- G2 paired held-out transfer ---------------------------
    n = 8 if SMOKE else 32
    n_fit = 3 * n // 4
    g = torch.Generator(device="cpu").manual_seed(814)
    pairs = []
    for _ in range(n):
        rows = []
        for _ in range(3):
            rows.append([int(torch.randint(1, 6, (1,), generator=g).item())
                         for _ in range(3)])
        rows[1][1] = 0
        pairs.append((rows, shift_right(rows)))
    fit_pairs, hold_pairs = pairs[:n_fit], pairs[n_fit:]

    def real_loss(pairs_, trans, action_wave):
        losses = []
        for ga, gb in pairs_:
            wa = enc.encode_spatial_grid(ga).squeeze(0)
            wb = enc.encode_spatial_grid(gb).squeeze(0)
            pred = trans.forward(wa, action_wave)
            p, o = pred.reshape(-1), wb.reshape(-1)
            losses.append(1.0 - torch.dot(p, o) /
                          (p.norm() * o.norm()).clamp(min=1e-12))
        return float(sum(losses) / len(losses))

    def complex_loss(pairs_, trans):
        losses = []
        for ga, gb in pairs_:
            wa = enc.encode_spatial_grid(ga).squeeze(0)
            wb = enc.encode_spatial_grid(gb).squeeze(0)
            za, zb = un_realify(wa), un_realify(wb)
            zp = trans.forward_complex(za, 0)
            losses.append(1.0 - complex_cosine(zp, zb))
        return float(sum(losses) / len(losses))

    action_wave = torch.randn(D // 2, device=DEVICE)
    action_wave = action_wave / action_wave.norm()

    pre_r = real_loss(hold_pairs, tr_r, action_wave)
    pre_c = complex_loss(hold_pairs, tr_c)
    for ga, gb in fit_pairs:
        wa = enc.encode_spatial_grid(ga).squeeze(0)
        wb = enc.encode_spatial_grid(gb).squeeze(0)
        tr_r.update_wirtinger(wa, action_wave, wb, lr=0.05)
        tr_c.update_phase_complex(un_realify(wa), un_realify(wb), 0, lr=0.05)
    post_r = real_loss(hold_pairs, tr_r, action_wave)
    post_c = complex_loss(hold_pairs, tr_c)

    g2 = {"n": n, "n_fit": n_fit, "n_held": n - n_fit,
          "pre_real": round(pre_r, 5), "post_real": round(post_r, 5),
          "pre_complex": round(pre_c, 5), "post_complex": round(post_c, 5),
          "delta": round(post_c - post_r, 5)}
    out["arms"]["G2_paired_transfer"] = g2
    g2_ok = bool(post_c <= 0.90 and (post_c - post_r) >= 0.02)
    out["verdicts"]["G2"] = "ACCEPT" if g2_ok else "KILL"
    if not g2_ok:
        failures.append("G2")

    # ---------------- G3 demo block ------------------------------------------
    out["arms"]["G3_demo_block"] = {"status": "BLOCKED_NO_DEMONSTRATIONS",
                                    "ok": True,
                                    "expected": "never fabricated"}
    out["verdicts"]["G3"] = "BLOCKED_NO_DEMONSTRATIONS"

    # ---------------- G4 latency ----------------------------------------------
    wa = enc.encode_spatial_grid(GRID_A).squeeze(0)
    t0 = time.perf_counter()
    z = un_realify(wa)
    z2 = tr_c.forward_complex(z, 0)
    w_cycle = re_realify(z2)
    t_cyc = (time.perf_counter() - t0) * 1e3
    g4 = {"cycle_ms": round(t_cyc, 4), "ok": bool(t_cyc <= 2.0),
          "shape": list(w_cycle.shape), "finite": bool(torch.isfinite(w_cycle).all())}
    out["arms"]["G4_latency"] = g4
    out["verdicts"]["G4"] = "CYCLE_PASS" if t_cyc <= 2.0 else "CYCLE_SLOW"
    if t_cyc > 2.0:
        failures.append("G4")

    # ---------------- G5 default-OFF (toy scale) ------------------------------
    from efe_planner import EFEPlanner
    p = EFEPlanner(num_blocks=64, d_model=512)
    out["arms"]["G5_default_off"] = {
        "transition_type": type(p.transition).__name__,
        "ok": bool(type(p.transition).__name__ == "LowRankCoupledTransition")}
    out["verdicts"]["G5"] = "DEFAULT_OFF_PASS" if out["arms"]["G5_default_off"]["ok"] \
        else "DEFAULT_OFF_FAIL"
    if not out["arms"]["G5_default_off"]["ok"]:
        failures.append("G5")

    # ---------------- G6 round-trip @ scale -----------------------------------
    rt = re_realify(un_realify(wa))
    err = float((rt - wa).abs().max())
    out["arms"]["G6_roundtrip"] = {"max_err": err, "ok": bool(err < 1e-6)}
    out["verdicts"]["G6"] = "ROUNDTRIP_PASS" if err < 1e-6 else "ROUNDTRIP_FAIL"
    if err >= 1e-6:
        failures.append("G6")

    out["failures"] = failures
    out["rc"] = 1 if failures else 0
    out["verdict"] = ("KILL" if failures else "ACCEPT")
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print("DONE_MARKER rc=%d failures=%s verdict=%s" % (out["rc"], failures, out["verdict"]))
    return out["rc"]


if __name__ == "__main__":
    raise SystemExit(main())
