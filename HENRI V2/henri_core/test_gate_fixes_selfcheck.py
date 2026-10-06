"""Inline verification of the two approved fixes, plus run_all_v2.

Checks (all deterministic, CPU, no model):
  T1  C2: run_all_v2 routes BLOCKED -> NOT_ACCEPTED (the gates.py:360 defect).
      Negative control: the OLD rule must mis-classify a BLOCKED set as ACCEPTED.
  T2  C2: run_all_v2 accepts only when every gate is PASS.
  T3  C2: unknown statuses are coerced to BLOCKED, never to PASS.
  T4  C3: dust_zo carries the CUDA guard, and the CPU path still runs.
  T5  Regression: the frozen estimator still reproduces (pos_ctl ~1,
      shuffled ~0, and real K=16 equal to the frozen receipt value).
"""
from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, ".")
from henri_core import gates_v2 as g2
from henri_core.dust_zo import DustZOConfig, matcher_loss_fn
from henri_core.exp_dust_g1 import cosine, run_arm

FROZEN_K16 = 0.20452914191982666


def main() -> int:
    ok = True
    lines = []

    # ---- T1/T2/T3: aggregate driver behaviour -------------------------------
    sets = {
        "all_pass": [{"status": g2.PASS, "id": "a"}] * 3,
        "one_fail": [{"status": g2.PASS, "id": "a"}, {"status": g2.FAIL, "id": "b"}],
        "one_vacuous": [{"status": g2.PASS, "id": "a"}, {"status": g2.VACUOUS, "id": "b"}],
        "one_blocked": [{"status": g2.PASS, "id": "a"}, {"status": g2.BLOCKED, "id": "b"}],
        "blocked_only": [{"status": g2.BLOCKED, "id": "b"}],
        "measured_only": [{"status": g2.MEASURED, "id": "c"}],
        "unknown_status": [{"status": "SOMETHING_NEW", "id": "d"}],
        "empty": [],
    }
    expect = {
        "all_pass": "ACCEPTED",
        "one_fail": "NOT_ACCEPTED",
        "one_vacuous": "NOT_ACCEPTED",
        "one_blocked": "NOT_ACCEPTED",
        "blocked_only": "NOT_ACCEPTED",
        "measured_only": "NOT_ACCEPTED",
        "unknown_status": "NOT_ACCEPTED",
        "empty": "NOT_ACCEPTED",
    }

    def old_rule(gates):
        """Reproduce gates.py:360 exactly, as the negative control."""
        counts = {g2.PASS: 0, g2.FAIL: 0, g2.VACUOUS: 0, g2.BLOCKED: 0}
        for g in gates:
            counts[g["status"]] = counts.get(g["status"], 0) + 1
        return "NOT_ACCEPTED" if (counts[g2.FAIL] or counts[g2.VACUOUS]) else "ACCEPTED"

    for name, gates in sets.items():
        got = g2.run_all_v2(gates)["overall"]
        good = got == expect[name]
        ok &= good
        lines.append(f"  {name:15s} -> {got:13s} expected {expect[name]:13s} "
                     f"{'OK' if good else 'DEFECT'}")

    # negative control: the old rule MUST mis-classify one_blocked as ACCEPTED
    old_got = old_rule(sets["one_blocked"])
    ctl_ok = old_got == "ACCEPTED"
    ok &= ctl_ok
    lines.append(f"  NEG CONTROL old rule on one_blocked -> {old_got}  "
                 f"{'OK (defect reproduced)' if ctl_ok else 'CONTROL DID NOT REPRODUCE'}")

    r = g2.run_all_v2(sets["one_blocked"])
    lines.append(f"  detail: blocked={r['counts'][g2.BLOCKED]} score={r['score']} "
                 f"reasons={len(r['blocking_reasons'])}")

    # ---- T4: dust_zo CUDA guard --------------------------------------------
    here = os.path.dirname(os.path.abspath(__file__))
    src = open(os.path.join(here, "dust_zo.py"), encoding="utf-8").read()
    has_guard = ("is_cuda" in src) and ("CPU-only" in src)
    ok &= has_guard
    lines.append(f"  dust_zo CUDA guard present: {has_guard}")

    # CPU path still runs after the guard was added
    cfg = DustZOConfig(K=8, sigma=0.05, gamma=0.98, seed=1)
    y = torch.randn(4, 2, 16)
    x = torch.randn(4, 2, 8)
    lab = torch.randint(0, 3, (4, 2))
    readout = torch.randn(3, 16) / 4.0
    g = __import__("henri_core.dust_zo", fromlist=["x"]).node_perturbation_descent(
        y, x, matcher_loss_fn(readout), lab, cfg)
    cpu_ok = tuple(g.shape) == (16, 8)
    ok &= cpu_ok
    lines.append(f"  dust_zo CPU happy path: shape={tuple(g.shape)} {'OK' if cpu_ok else 'BROKEN'}")

    # ---- T5: regression on the frozen estimator ----------------------------
    n, t, d_in, d_out, classes, seed = 64, 8, 32, 256, 16, 20261005
    gg = torch.Generator().manual_seed(seed)
    X = torch.randn(n, t, d_in, generator=gg)
    labels = torch.randint(0, classes, (n, t), generator=gg)
    W = torch.randn(d_out, d_in, generator=gg) * 0.3
    readout = torch.randn(classes, d_out, generator=gg) / (d_out ** 0.5)
    lf = matcher_loss_fn(readout)

    Wg = W.detach().clone().requires_grad_(True)
    lf(X @ Wg.t(), labels).sum().backward()
    ref = (-Wg.grad).detach()

    base = DustZOConfig(K=64, sigma=0.05, gamma=0.98, seed=seed)
    pos = run_arm(W[:, :d_in], X, labels, readout, base, "pos_ctl")[0]
    shuf = run_arm(W[:, :d_in], X, labels, readout,
                   DustZOConfig(K=256, sigma=0.05, gamma=0.98, seed=seed), "shuffled")[0]
    real = run_arm(W[:, :d_in], X, labels, readout,
                   DustZOConfig(K=16, sigma=0.05, gamma=0.98, seed=seed), "real")[0]
    cp, cs, cr = cosine(pos, ref), cosine(shuf, ref), cosine(real, ref)
    t5 = (cp >= 0.99) and (abs(cs) <= 0.10) and (abs(cr - FROZEN_K16) < 1e-6)
    ok &= t5
    lines.append(f"  estimator regression: pos_ctl={cp:.16f} shuffled={cs:.6f} "
                 f"realK16={cr:.12f} frozen={FROZEN_K16:.12f} "
                 f"{'OK' if t5 else 'REGRESSION'}")

    lines.append(f"  outcome vocabulary: {list(g2._OUTCOMES)}")

    print("\n".join(lines))
    print(f"\nSELF-CHECK {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
