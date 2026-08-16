"""Phase 8.15 — SU(3) chromodynamic grounding CUDA verification matrix.

Gates (pre-registered in phase815_chromodynamic_grounding_design.md):
  G1-QCD  min ||U_A U_B - U_B U_A||_F > 0.5000 across all 45 distinct color pairs
  G2-QCD  singlet veto rate == 0.0 AND nonsinglet veto rate == 1.0
  G3-QCD  fitted-gauge held-out SU(3) transport loss < 0.1500 (steps 10..50)
  G4-QCD  Triton 3x3 complex matmul <= 50.0 us at N=65,536, max err < 1e-4 vs torch

HENRI_SMOKE=1 -> N=8192, 20-step trajectory; else N=65,536, 50-step trajectory.
Writes JSON to $JEPA_DM_OUT. DONE_MARKER with failures; rc=1 iff failures.
"""

import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import chromodynamic_grounding as cg  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SMOKE = os.environ.get("HENRI_SMOKE", "0") == "1"
OUT = os.environ.get("JEPA_DM_OUT", "/tmp/p815_matrix_d65536.json")
torch.manual_seed(20260816)

N = 8192 if SMOKE else 65536
STEPS = 20 if SMOKE else 50
FAILURES: list[str] = []
M: dict[str, object] = {}


def check(gate: str, ok: bool, detail: object) -> None:
    M[f"gate_{gate}"] = {"pass": bool(ok), "detail": detail}
    if not ok:
        FAILURES.append(gate)


def main() -> int:
    print(f"[phase815] device={DEVICE} smoke={SMOKE} N={N} steps={STEPS} triton={cg.TRITON_AVAILABLE}")

    # ---- G1-QCD: non-commutative color binding (PDF 2.1) ----
    grid = torch.arange(10, device=DEVICE).reshape(1, 1, 10)
    u = cg.encode_su3_color_field(grid)[0, 0]  # [10,3,3]
    dists = []
    for a in range(10):
        for b in range(a + 1, 10):
            d = torch.linalg.matrix_norm(u[a] @ u[b] - u[b] @ u[a]).item()
            dists.append({"pair": f"{a}{b}", "dist": round(d, 4)})
    min_d = min(d["dist"] for d in dists)
    check("G1-QCD", min_d > 0.5000, {"min_dist": min_d, "pairs": dists})

    # ---- G2-QCD: confinement veto rates ----
    a = torch.randn(N, dtype=torch.complex64, device=DEVICE)
    singlet = torch.einsum("n,jk->njk", a, torch.eye(3, dtype=torch.complex64, device=DEVICE))
    veto_s, pen_s = cg.confinement_veto(singlet)
    nonsinglet = torch.randn(N, 3, 3, dtype=torch.complex64, device=DEVICE)
    veto_n, pen_n = cg.confinement_veto(nonsinglet)
    rate_s = veto_s.float().mean().item()
    rate_n = veto_n.float().mean().item()
    check(
        "G2-QCD",
        rate_s == 0.0 and rate_n == 1.0,
        {"singlet_rate": rate_s, "nonsinglet_rate": rate_n,
         "max_pen_singlet": pen_s.max().item(), "min_pen_nonsinglet": pen_n.min().item()},
    )

    # ---- G3-QCD: gauge transport fit, held-out loss ----
    theta0 = (torch.rand(N, 8, device=DEVICE) * 2 - 1) * 1.0
    psi0 = torch.matrix_exp(1j * torch.einsum("na,arc->nrc", theta0.to(torch.complex64), cg.GELL_MANN_BASIS.to(DEVICE)))
    u_true = torch.matrix_exp(1j * torch.einsum("a,arc->rc", (torch.randn(8, device=DEVICE) * 0.5).to(torch.complex64), cg.GELL_MANN_BASIS.to(DEVICE)))
    traj = [psi0]
    for _ in range(STEPS - 1):
        traj.append(cg.su3_transport(traj[-1], u_true))
    traj = torch.stack(traj)  # [T,N,3,3]
    fit_n = min(10, STEPS - 1)
    u_hat = cg.fit_su3_gauge(traj[:fit_n])
    x = traj[fit_n:-1].reshape(-1, 3, 3)
    y = traj[fit_n + 1 :].reshape(-1, 3, 3)
    loss = (torch.matmul(x, u_hat) - y).abs().pow(2).mean().item()
    check("G3-QCD", loss < 0.1500, {"heldout_loss": loss, "fit_steps": fit_n, "eval_steps": STEPS - fit_n - 1})

    # ---- G4-QCD: Triton kernel latency + correctness ----
    if cg.TRITON_AVAILABLE:
        ba = torch.randn(N, 3, 3, dtype=torch.complex64, device=DEVICE)
        bb = torch.randn(N, 3, 3, dtype=torch.complex64, device=DEVICE)
        ref = cg.su3_matmul_torch(ba, bb)
        # warmup
        cg.su3_matmul_triton(ba, bb)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        got = cg.su3_matmul_triton(ba, bb)
        torch.cuda.synchronize()
        lat_us = (time.perf_counter() - t0) * 1e6
        err = (got - ref).abs().max().item()
        check("G4-QCD", lat_us <= 50.0 and err < 1e-4, {"latency_us": round(lat_us, 3), "max_err": err})
    else:
        check("G4-QCD", False, "triton_unavailable")

    ok = not FAILURES
    M["done"] = {
        "marker": "DONE",
        "rc": 0 if ok else 1,
        "failures": FAILURES,
        "verdict": "ACCEPT" if ok else "KILL",
        "device": DEVICE,
        "smoke": SMOKE,
        "N": N,
        "steps": STEPS,
        "seed": 20260816,
    }
    Path(OUT).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(M, f, indent=2)
    print(f"[phase815] DONE_MARKER rc={0 if ok else 1} failures={FAILURES}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
