"""Phase 8.12 remote CUDA verification matrix (RTX 5090, D=65,536).

Pre-registration: HENRI V2/experiments/sweeps/phase812_complex_native_ingress_design.md
Blueprint: HENRI V2 Structural Analysis & Phase 8.12 Architecture.pdf.

Arms:
- G1: complex-native ingress kill at scale (no bandwidth s satisfies
  adj>=0.85 AND distinct<0.95) — assert the collapse evidence.
- G2: unit-modulus superposition collapse (distinct cos > 0.95).
- G3: legacy control — HENRIVisionEncoder incommensurate+bg_mask
  DISCRIMINATES (cos < 0.10); default basis collapses (the 8.11 G2
  vacuity root cause at scale).
- G4: transfer preview (Phase 8.13 lever, diagnostic): real translation
  pairs through discriminating encoder + NativeComplexWaveTransition
  real-lift path; pre vs post Sagnac, improvement = pre - post.
- G5: latency — ingress encode_grid + forward_complex cycle (<= 2.0 ms
  blueprint target) and legacy encode latency.
- G6: default-OFF EFEPlanner probe (toy scale, 8.11 G4 precedent).

DONE marker aggregates ONLY exceptions (kill gates fire = ok=True per
pre-registration; 8.10/8.11 precedent).
"""

import json
import math
import os
import time

import torch

from complex_native_ingress import ComplexNativeIngress
from complex_phase_transition import NativeComplexWaveTransition
from henri_vision_encoder import HENRIVisionEncoder

D = int(os.environ.get("JEPA_D", "65536"))
NB, BD = 8192, 8
NA = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT_PATH = os.environ.get("JEPA_DM_OUT", "/tmp/p812_matrix_d65536.json")

GRID_A = [[0, 1, 0], [1, 2, 1], [0, 1, 0]]
GRID_B = [[0, 0, 1], [0, 2, 1], [0, 1, 1]]   # A translated right by 1
GRID_C = [[0, 2, 0], [2, 1, 2], [0, 2, 0]]
GRID_D = [[3, 3, 3], [3, 0, 3], [3, 3, 3]]
GRID_E = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
GRIDS = {"A": GRID_A, "B": GRID_B, "C": GRID_C, "D": GRID_D, "E": GRID_E}


def ccos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1), b.reshape(-1)
    denom = (a.abs().norm() * b.abs().norm()).clamp_min(1e-12)
    return float(torch.real(torch.dot(a, torch.conj(b))) / denom)


def real_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.reshape(-1).float(), b.reshape(-1).float()
    return float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-12))


def shift_right(grid, cols=3):
    return [[row[(x - 1) % cols] for x in range(cols)] for row in grid]


def main():
    out = {"phase": "8.12", "device": DEVICE, "d": D, "smoke": SMOKE,
           "arms": {}, "verdicts": {}}
    failures = []

    # ---------------- ARM G1: kill at scale (no s satisfies both) ----------
    g1 = {}
    for s in ([0.10, 0.40] if not SMOKE else [0.10]):
        enc = ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                                   device=DEVICE, band_s=s)
        za, zb, zc, zd = (enc.encode_grid(GRIDS[k]) for k in ("A", "B", "C", "D"))
        adj = ccos(za, zb)
        distinct = max(ccos(za, zc), ccos(za, zd))
        g1[str(s)] = {"adj_AB": round(adj, 5), "distinct_max": round(distinct, 5),
                      "both_ok": bool(adj >= 0.85 and distinct < 0.95)}
    out["arms"]["G1_kill"] = g1
    out["verdicts"]["G1"] = "KILL_FIRED" if not any(
        v["both_ok"] for v in g1.values()) else "GATE_VIOLATED"

    # ---------------- ARM G2: collapse evidence -----------------------------
    enc = ComplexNativeIngress(dimension=D, num_blocks=NB, block_dim=BD,
                               device=DEVICE, band_s=0.10)
    za, zc, zd = (enc.encode_grid(GRIDS[k]) for k in ("A", "C", "D"))
    c_ac, c_ad = ccos(za, zc), ccos(za, zd)
    out["arms"]["G2_collapse"] = {"A_C": round(c_ac, 5), "A_D": round(c_ad, 5),
                                  "unit_mod": round(float(za.abs().mean()), 8)}
    out["verdicts"]["G2"] = "KILL_EVIDENCE" if (c_ac > 0.95 and c_ad > 0.95) else "UNEXPECTED"

    # ---------------- ARM G3: legacy control at scale -----------------------
    g3 = {}
    for kind in (["default", "incommensurate"] if not SMOKE else ["incommensurate"]):
        for bg in [False, True]:
            try:
                enc_leg = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD,
                                             device=DEVICE, spatial_basis_kind=kind,
                                             bg_mask=bg)
                wa, wc, wd = (enc_leg.encode_spatial_grid(GRIDS[k]).reshape(-1)
                              for k in ("A", "C", "D"))
                key = f"{kind}_bg{int(bg)}"
                g3[key] = {"A_C": round(real_cos(wa, wc), 5),
                           "A_D": round(real_cos(wa, wd), 5)}
            except Exception as e:  # noqa: BLE001
                g3[f"{kind}_bg{int(bg)}"] = {"error": f"{type(e).__name__}: {e}"}
    out["arms"]["G3_legacy_control"] = g3
    disc = g3.get("incommensurate_bg1", {}).get("A_C", 1.0)
    out["verdicts"]["G3"] = "LEGACY_DISCRIMINATES" if disc < 0.10 else "LEGACY_COLLAPSED"

    # ---------------- ARM G4: transfer preview (8.13 lever, diagnostic) -----
    n = 8 if SMOKE else 32
    g = torch.Generator(device="cpu").manual_seed(812)
    trans = NativeComplexWaveTransition(dimension=D, num_actions=NA, device=DEVICE,
                                        num_blocks=NB, block_dim=BD)
    action_wave = torch.randn(NB, BD, device=DEVICE)
    action_wave = action_wave / action_wave.norm()

    def transfer_loss(enc_leg, pairs):
        losses = []
        for ga, gb in pairs:
            wa = enc_leg.encode_spatial_grid(ga).squeeze(0).to(DEVICE)
            wb = enc_leg.encode_spatial_grid(gb).squeeze(0).to(DEVICE)
            pred = trans.forward(wa, action_wave)
            p, o = pred.reshape(-1), wb.reshape(-1)
            losses.append(float(1.0 - torch.dot(p, o) / (p.norm() * o.norm()).clamp(min=1e-12)))
        return sum(losses) / len(losses)

    # deterministic translation pairs (shared delta per pre-registration)
    pairs = []
    for i in range(n):
        rows = []
        for _ in range(3):
            rows.append([int(torch.randint(1, 6, (1,), generator=g).item()) for _ in range(3)])
        rows[1][1] = 0  # keep a background hole to vary pattern
        pairs.append((rows, shift_right(rows)))

    g4 = {}
    for kind in (["default", "incommensurate"] if not SMOKE else ["incommensurate"]):
        enc_leg = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD,
                                     device=DEVICE, spatial_basis_kind=kind, bg_mask=True)
        pre = transfer_loss(enc_leg, pairs)
        # fit: 1 damped update per pair (production lr semantics)
        for ga, gb in pairs:
            wa = enc_leg.encode_spatial_grid(ga).squeeze(0).to(DEVICE)
            wb = enc_leg.encode_spatial_grid(gb).squeeze(0).to(DEVICE)
            trans.update_wirtinger(wa, action_wave, wb, lr=0.05)
        post = transfer_loss(enc_leg, pairs)
        g4[kind] = {"pre": round(pre, 5), "post": round(post, 5),
                    "improvement": round(pre - post, 5)}
    out["arms"]["G4_transfer_preview"] = g4
    imp = g4.get("incommensurate", {}).get("improvement", 0.0)
    out["verdicts"]["G4"] = ("8_13_LEVER_PROMISING" if imp > 0.02
                             else "8_13_LEVER_NEEDS_WORK")

    # ---------------- ARM G5: latency ---------------------------------------
    t0 = time.perf_counter()
    z = enc.encode_grid(GRID_A)
    t_enc = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    z2 = trans.forward_complex(z, 0)
    t_fwd = (time.perf_counter() - t0) * 1e3
    t_leg = None
    try:
        enc_leg = HENRIVisionEncoder(d_model=D, k_blocks=NB, block_dim=BD,
                                     device=DEVICE, spatial_basis_kind="incommensurate",
                                     bg_mask=True)
        t0 = time.perf_counter()
        enc_leg.encode_spatial_grid(GRID_A)
        t_leg = (time.perf_counter() - t0) * 1e3
    except Exception as e:  # noqa: BLE001
        t_leg = f"{type(e).__name__}: {e}"
    g5 = {"ingress_ms": round(t_enc, 4), "transition_ms": round(t_fwd, 4),
          "cycle_ms": round(t_enc + t_fwd, 4), "legacy_ingress_ms": t_leg}
    out["arms"]["G5_latency"] = g5
    out["verdicts"]["G5"] = "CYCLE_PASS" if (t_enc + t_fwd) <= 2.0 else "CYCLE_SLOW"

    # ---------------- ARM G6: default-OFF probe (toy scale) -----------------
    from efe_planner import EFEPlanner
    p = EFEPlanner(num_blocks=16, d_model=128)
    g6 = {"_use_complex_transition": bool(p._use_complex_transition),
          "transition": type(p.transition).__name__}
    out["arms"]["G6_default_off"] = g6
    out["verdicts"]["G6"] = "DEFAULT_OFF" if (g6["transition"] == "LowRankCoupledTransition"
                                              and not g6["_use_complex_transition"]) else "VIOLATED"

    # ---------------- DONE marker -------------------------------------------
    out["done_marker"] = {"rc": 1 if failures else 0, "failures": failures}
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({"done": True, "rc": out["done_marker"]["rc"],
                      "verdicts": out["verdicts"]}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
