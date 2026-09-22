#!/usr/bin/env python3
"""Diagnostic: is the thermodynamic gradient correct, or is my implementation wrong?

Two independent checks, run locally (numeric, CPU-only by nature):

  1. FINITE-DIFFERENCE GRADIENT CHECK.  The analytic rule (Eq. 10-11) must match
     a numeric derivative of -ln Ptilde (Eq. 7) w.r.t. the couplings. If it does
     not, the module does not implement the source and any verdict is void.

  2. LOSS DESCENT vs STEP SIZE.  The source updates once per trajectory using a
     sum over ALL steps (Eq. 8-9), so the effective step size scales with the
     trajectory length K. A learning rate calibrated for a short trajectory will
     overshoot on a long one. Sweep alpha on the real loss.

Both are diagnostics, not capability claims.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import torch

from henri_thermo_langevin import LangevinComputer, LangevinConfig, make_stripe_patterns


def true_neg_logprob(model: LangevinComputer, traj: torch.Tensor) -> float:
    """-ln Ptilde_theta(reverse trajectory), the source's Eq. (7) without the
    constant term:  SUM_i (-dx_i + mu d_iV(x') dt)^2 / (4 mu kT dt), summed over steps."""
    cfg = model.cfg
    x = traj[:-1]
    dx = traj[1:] - traj[:-1]
    xp = traj[1:]
    r = (-dx + cfg.mu * model.force(xp) * cfg.dt)
    return float((r ** 2).sum() / (4.0 * cfg.mu * cfg.kT * cfg.dt))


def fd_check(seed: int = 0, n_units: int = 16, eps: float = 1e-5,
             n_pairs: int = 6) -> dict:
    cfg = LangevinConfig(n_units=n_units, n_visible=n_units, t_final=0.05)
    patterns = make_stripe_patterns(n_units)
    torch.manual_seed(seed)
    model = LangevinComputer(cfg, seed=seed)
    # Non-zero couplings so the gradient is not trivially zero.
    with torch.no_grad():
        g = torch.Generator().manual_seed(11)
        model.J_raw.copy_(0.5 * torch.randn(n_units, n_units, generator=g))
        model.b.copy_(0.1 * torch.randn(n_units, generator=g))

    g2 = torch.Generator().manual_seed(seed)
    traj, _ = model.noise_trajectory(patterns[0], generator=g2)

    grad_J, grad_b = model.reverse_logprob_grad(traj)

    g3 = torch.Generator().manual_seed(1234)
    rows = []
    for t in range(n_pairs):
        i = int(torch.randint(0, n_units, (1,), generator=g3))
        j = int(torch.randint(0, n_units, (1,), generator=g3))
        if i == j:
            continue
        base = model.J_raw[i, j].item()
        with torch.no_grad():
            model.J_raw[i, j] = base + eps
        lp_plus = true_neg_logprob(model, traj)
        with torch.no_grad():
            model.J_raw[i, j] = base - eps
        lp_minus = true_neg_logprob(model, traj)
        with torch.no_grad():
            model.J_raw[i, j] = base
        # d(-lnPtilde)/dJ_ij by finite difference
        fd = (lp_plus - lp_minus) / (2.0 * eps)
        # analytic: d(-lnPtilde)/dJ = -grad_J (grad_J is ascent on ln Ptilde)
        an = -float(grad_J[i, j])
        rows.append({"i": i, "j": j, "fd": fd, "analytic": an,
                     "abs_err": abs(fd - an)})

    # bias check
    brows = []
    for i in range(min(4, n_units)):
        base = model.b[i].item()
        with torch.no_grad():
            model.b[i] = base + eps
        lp_plus = true_neg_logprob(model, traj)
        with torch.no_grad():
            model.b[i] = base - eps
        lp_minus = true_neg_logprob(model, traj)
        with torch.no_grad():
            model.b[i] = base
        fd = (lp_plus - lp_minus) / (2.0 * eps)
        an = -float(grad_b[i])
        brows.append({"i": i, "fd": fd, "analytic": an, "abs_err": abs(fd - an)})

    jerr = max((r["abs_err"] for r in rows), default=0.0)
    berr = max((r["abs_err"] for r in brows), default=0.0)
    scale = max((abs(r["fd"]) for r in rows), default=1.0)
    return {"J_rows": rows, "b_rows": brows,
            "max_abs_err_J": jerr, "max_abs_err_b": berr,
            "fd_scale_J": scale,
            "relative_err_J": jerr / scale if scale > 0 else float("inf"),
            "PASS": bool(jerr < 1e-3 * max(scale, 1.0) and berr < 1e-3)}


def alpha_sweep(seed: int = 0, steps: int = 60, n_units: int = 48) -> dict:
    patterns = make_stripe_patterns(16)
    out = {}
    for alpha in (0.001, 0.005, 0.02):
        cfg = LangevinConfig(n_units=n_units, n_visible=16)
        torch.manual_seed(seed)
        m = LangevinComputer(cfg, seed=seed)
        g = torch.Generator().manual_seed(seed)
        tr, _ = m.noise_trajectory(patterns[0], generator=g)
        loss0 = true_neg_logprob(m, tr)
        res = m.train_on_patterns(patterns, steps=steps, n_traj_per_step=2,
                                  alpha=alpha, seed=seed)
        tr1, _ = m.noise_trajectory(patterns[0], generator=torch.Generator().manual_seed(seed))
        loss1 = true_neg_logprob(m, tr1)
        out[f"alpha={alpha}"] = {
            "loss_before": loss0, "loss_after": loss1,
            "relative_drop": (loss0 - loss1) / abs(loss0),
            "train_loss_first": res["loss_first"], "train_loss_last": res["loss_last"],
        }
    return out


if __name__ == "__main__":
    print("=== 1. FINITE-DIFFERENCE GRADIENT CHECK ===")
    fd = fd_check()
    for r in fd["J_rows"]:
        print(f"  J[{r['i']:2d},{r['j']:2d}]  fd={r['fd']:+.6e}  analytic={r['analytic']:+.6e}"
              f"  err={r['abs_err']:.3e}")
    for r in fd["b_rows"]:
        print(f"  b[{r['i']:2d}]     fd={r['fd']:+.6e}  analytic={r['analytic']:+.6e}"
              f"  err={r['abs_err']:.3e}")
    print(f"  max_abs_err_J={fd['max_abs_err_J']:.3e}  relative={fd['relative_err_J']:.3e}")
    print(f"  max_abs_err_b={fd['max_abs_err_b']:.3e}")
    print(f"  GRADIENT_CHECK_PASS = {fd['PASS']}")

    print("\n=== 2. ALPHA SWEEP (real loss) ===")
    sw = alpha_sweep()
    print(json.dumps(sw, indent=2))

    payload = {"schema": "henri.thermo-diagnostic.v1",
               "gradient_check": {k: v for k, v in fd.items() if not k.endswith("rows")},
               "alpha_sweep": sw}
    with open("thermo_diagnostic_observed.json", "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("\nwrote thermo_diagnostic_observed.json")
