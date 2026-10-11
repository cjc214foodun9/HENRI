"""C1 measured where E_p is NONZERO, plus the D_d = I arm.

PROBE DEFECT IN MY PREDECESSOR (self-caught): probe_ep_ode_wbar.py measured the
one-step identity at t = 0, where V = Vbar = 0 => F = B = 0 => E_p = 0 exactly.
The residual was then divided by max(E_p, 1e-30), giving 3.5e28. That number is
an artifact of the denominator, NOT evidence about the ODE.

FIX: step the network WARMUP steps first, then measure the one-step identity where
E_p is O(1). Also report the residual against |up - dn| so a small E_p does not
inflate the relative error.

IDENTITY UNDER TEST (derived from Eq. 13/14 + the module's feedback recursion):
    dot{E}_p = -gamma E_p + eta*s_l*outer(abar^l, h^{l+1})
                          - eta_bar*s_{l+1}*relu(hbar^l).outer(Wbar^l a^{l+1}, abar^{l+1})
with dWbar/dt = 0. Residual small => ODE validated for the module's configuration.

D_d FIDELITY: the paper's benchmarks use D_d = I ("take D = I in the backward pass of
Algorithm 1"); this module uses D_d = relu. Both arms reported. A configuration that
validates only with D_d = relu does NOT validate the paper's benchmark configuration.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, "HENRI V2/henri_core")
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

DIMS = (32, 64, 64, 32)
GAMMA, ETA_W, WARMUP = 0.02, 0.002, 40


def s(net, l):
    if net.drive_scale == "sqrt_d":
        return 1.0 / math.sqrt(net.dims[l])
    if net.drive_scale == "d":
        return 1.0 / net.dims[l]
    return 1.0


def blocks(net, h, ab, l, dd_I):
    Du1 = (h[l + 1] > 0).to(net.dtype)
    F = net.V[l] @ ((h[l] > 0).to(net.dtype)[:, None] * net.W[l].t())
    hbar = net.Wbar[l] @ ab[l + 1]
    Braw = net.Wbar[l] @ net.Vbar[l].t()
    B = Braw if dd_I else ((hbar > 0).to(net.dtype)[:, None] * Braw)
    a_1 = net._prep(h[l + 1])
    up = net.eta * s(net, l) * torch.outer(ab[l], h[l + 1])
    dn = (net.eta_bar * s(net, l + 1) * ((hbar > 0).to(net.dtype)[:, None] if not dd_I else 1.0)
          * torch.outer(net.Wbar[l] @ a_1, ab[l + 1]))
    return float((F - B).norm()), up, dn


g = torch.Generator().manual_seed(0)
xs = [torch.randn(DIMS[0], generator=g) for _ in range(4)]
tg = [torch.randn(DIMS[-1], generator=g) for _ in range(4)]

print("=" * 84)
print(f"C1 at NONZERO E_p (after {WARMUP} warmup steps), both D_d configurations")
print("=" * 84)
for dd_I in (False, True):
    tag = "D_d = I   (PAPER)" if dd_I else "D_d = relu (MODULE)"
    net = DualStreamNetwork(DIMS, gamma=GAMMA, eta_w=ETA_W, seed=0)
    for t in range(WARMUP):
        i = t % 4
        with torch.no_grad():
            net.sal_step(xs[i], 2.0 * (net.activations(xs[i])[-1] - tg[i]))
    print(f"\n--- {tag} ---")
    print(f"{'l':>2} {'|Ep|':>9} {'|up|':>9} {'|dn|':>9} {'|up-dn|':>9} "
          f"{'resid/|Ep|':>11} {'resid/|up-dn|':>14}")
    rows = {}
    for l in range(net.n):
        with torch.no_grad():
            h0 = net.activations(xs[0])
            ab0 = net.feedback(h0, 2.0 * (h0[-1] - tg[0]))
            e0, up, dn = blocks(net, h0, ab0, l, dd_I)
            pred = (1.0 - GAMMA) * e0 + float(up.norm()) - float(dn.norm())
            net.sal_step(xs[0], 2.0 * (h0[-1] - tg[0]), retract=False)
            h1 = net.activations(xs[0])
            ab1 = net.feedback(h1, 2.0 * (h1[-1] - tg[0]))
            e1 = blocks(net, h1, ab1, l, dd_I)[0]
        res = abs(e1 - pred)
        r_e = res / max(e0, 1e-30)
        r_d = res / max(abs(float(up.norm()) - float(dn.norm())), 1e-30)
        rows[l] = {"ep": e0, "ep_next": e1, "pred": pred, "up": float(up.norm()),
                   "dn": float(dn.norm()), "resid": res, "resid_rel_ep": r_e,
                   "resid_rel_forcing": r_d}
        print(f"{l:>2} {e0:>9.4f} {float(up.norm()):>9.4f} {float(dn.norm()):>9.4f} "
              f"{abs(float(up.norm()) - float(dn.norm())):>9.4f} {r_e:>11.4e} {r_d:>14.4e}")
    verdict = all(v["resid_rel_ep"] < 0.10 for v in rows.values())
    print(f"  C1 HOLDS for {tag} (resid/|Ep| < 0.10 every layer): {verdict}")
    if dd_I:
        root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                            "..", "..", ".."))
        p = os.path.join(root, "design", "zone_a", "evidence",
                         "henri_ep_ode_c1_nonzero_observed.json")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({"warmup": WARMUP, "note": "D_d=I arm shown; module arm printed above",
                       "layers": rows, "c1_holds_paper_config": verdict}, fh, indent=2)
        print(f"\n[receipt] {p}")
