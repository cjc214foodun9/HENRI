#!/usr/bin/env python3
"""Decisive check: is the analytic thermodynamic gradient CORRECT?

A falsified gate is only meaningful if the instrument is right. The first version
of this check computed the finite difference in float32 at a loss magnitude of
~24, where the true difference (~5e-6) is below representable precision -- it
returned exactly 0.0 for every coupling and one incoherent value (0.381 vs 0.240)
from cancellation. That was an instrument failure, not a result.

FIX: evaluate -ln Ptilde in float64 and use a step size well above the precision
floor. The gradient is a property of the model, not of the dtype we probe it in.

  fd      = d(-ln Ptilde)/d(theta)      by central difference (float64)
  analytic= -grad, where grad = d ln Ptilde/d(theta) from the module's rule

The module's rule is Eq. (10-11) of arXiv:2506.15121v3.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from henri_thermo_langevin import LangevinComputer, LangevinConfig, make_stripe_patterns


def neg_logprob_f64(model: LangevinComputer, traj: torch.Tensor,
                    J: torch.Tensor | None = None,
                    b: torch.Tensor | None = None) -> float:
    """-ln Ptilde (Eq. 7, summed over steps) in float64.

    Equation (7):  -ln Ptilde = SUM_i (-dx_i + mu d_i V(x') dt)^2 / (4 mu kT dt)
    with x' = x + dx the endpoint of the FORWARD (noising) step.
    """
    cfg = model.cfg
    Jd = model.effective_J().detach().double() if J is None else J.double()
    bd = model.b.detach().double() if b is None else b.double()

    dx = (traj[1:] - traj[:-1]).double()
    xp = traj[1:].double()

    force = 2.0 * cfg.j2 * xp + 4.0 * cfg.j4 * xp ** 3 + bd + xp @ Jd.t()
    r = -dx + cfg.mu * force * cfg.dt
    return float((r ** 2).sum() / (4.0 * cfg.mu * cfg.kT * cfg.dt))


def analytic_grad_f64(model: LangevinComputer, traj: torch.Tensor
                      ) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. (10-11) in float64: d ln Ptilde / dJ_ij and / db_i (ascent direction)."""
    cfg = model.cfg
    Jd = model.effective_J().detach().double()
    bd = model.b.detach().double()

    x = traj[:-1].double()
    dx = (traj[1:] - traj[:-1]).double()
    xp = traj[1:].double()

    force = 2.0 * cfg.j2 * xp + 4.0 * cfg.j4 * xp ** 3 + bd + xp @ Jd.t()
    # Eq. (10-11) numerator: (-dx_i + mu d_i V(x') dt) / (2 kT)
    r = (-dx + cfg.mu * force * cfg.dt) / (2.0 * cfg.kT)

    # The couple term of the force is evaluated at x', so its derivative w.r.t.
    # J_ij is x'_j (not x_j). Using the step start is the D1 defect.
    gx = torch.einsum("ki,kj->ij", r, xp)         # sum_k r_i x'_j
    grad_J = -(gx + gx.t())                       # d ln Ptilde / dJ_ij
    off = 1.0 - torch.eye(grad_J.shape[0], dtype=torch.float64)
    grad_J = grad_J * off
    grad_b = -r.sum(dim=0)
    return grad_J, grad_b


def gradcheck(seed: int = 0, n_units: int = 16, eps: float = 1e-3,
              n_pairs: int = 6) -> dict:
    cfg = LangevinConfig(n_units=n_units, n_visible=n_units, t_final=0.05)
    patterns = make_stripe_patterns(n_units)

    torch.manual_seed(seed)
    model = LangevinComputer(cfg, seed=seed)
    g = torch.Generator().manual_seed(11)
    with torch.no_grad():
        # Non-zero couplings so the gradient is not trivially zero.
        J0 = 0.5 * torch.randn(n_units, n_units, generator=g)
        model.J_raw.copy_(0.5 * (J0 + J0.t()))
        model.b.copy_(0.1 * torch.randn(n_units, generator=g))

    traj, _ = model.noise_trajectory(patterns[0],
                                     generator=torch.Generator().manual_seed(seed))

    grad_J, grad_b = analytic_grad_f64(model, traj)

    Jbase = model.effective_J().detach().double().clone()
    bbase = model.b.detach().double().clone()

    gg = torch.Generator().manual_seed(1234)
    jrows, brows = [], []
    for _ in range(n_pairs):
        i = int(torch.randint(0, n_units, (1,), generator=gg))
        j = int(torch.randint(0, n_units, (1,), generator=gg))
        if i == j:
            continue
        Jp = Jbase.clone(); Jp[i, j] += eps; Jp[j, i] += eps   # symmetric perturbation
        Jm = Jbase.clone(); Jm[i, j] -= eps; Jm[j, i] -= eps
        lp = neg_logprob_f64(model, traj, J=Jp)
        lm = neg_logprob_f64(model, traj, J=Jm)
        fd = (lp - lm) / (2.0 * eps)
        an = -float(grad_J[i, j])
        jrows.append({"i": i, "j": j, "fd": fd, "analytic": an,
                      "abs_err": abs(fd - an)})

    for i in range(min(4, n_units)):
        bp = bbase.clone(); bp[i] += eps
        bm = bbase.clone(); bm[i] -= eps
        lp = neg_logprob_f64(model, traj, b=bp)
        lm = neg_logprob_f64(model, traj, b=bm)
        fd = (lp - lm) / (2.0 * eps)
        an = -float(grad_b[i])
        brows.append({"i": i, "fd": fd, "analytic": an, "abs_err": abs(fd - an)})

    jerr = max((r["abs_err"] for r in jrows), default=0.0)
    berr = max((r["abs_err"] for r in brows), default=0.0)
    scale = max([abs(r["fd"]) for r in jrows] + [1e-12])
    rel = jerr / scale
    return {"J_rows": jrows, "b_rows": brows, "eps": eps,
            "max_abs_err_J": jerr, "max_abs_err_b": berr,
            "fd_scale_J": scale, "relative_err_J": rel,
            "PASS": bool(rel < 1e-4 and berr < 1e-4)}


if __name__ == "__main__":
    res = gradcheck()
    print("=== FLOAT64 FINITE-DIFFERENCE GRADIENT CHECK (Eq. 10-11) ===")
    for r in res["J_rows"]:
        print(f"  J[{r['i']:2d},{r['j']:2d}]  fd={r['fd']:+.8e}  analytic={r['analytic']:+.8e}"
              f"  err={r['abs_err']:.3e}")
    for r in res["b_rows"]:
        print(f"  b[{r['i']:2d}]     fd={r['fd']:+.8e}  analytic={r['analytic']:+.8e}"
              f"  err={r['abs_err']:.3e}")
    print(f"  eps              = {res['eps']}")
    print(f"  fd_scale_J       = {res['fd_scale_J']:.6e}")
    print(f"  max_abs_err_J    = {res['max_abs_err_J']:.3e}  (relative {res['relative_err_J']:.3e})")
    print(f"  max_abs_err_b    = {res['max_abs_err_b']:.3e}")
    print(f"  GRADIENT_CHECK_PASS = {res['PASS']}")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "thermo_gradcheck_observed.json")
    res["schema"] = "henri.thermo-gradcheck.v1"
    res["source"] = "arXiv:2506.15121v3 Eq. 10-11"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2)
    print(f"wrote {out}")
