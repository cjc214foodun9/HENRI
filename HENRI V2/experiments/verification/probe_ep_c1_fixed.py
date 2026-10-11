"""C1 with the CORRECT downstream forcing term (third self-caught defect in this class).

MY THIRD DEFECT, found by re-deriving against the module source rather than trusting my
own earlier algebra:
    correct   dB/dt_forcing = eta_bar * s_{l+1} * D_d ⊙ outer(hbar^l, a^{l+1})
    my code   outer(Wbar^l @ a^{l+1}, abar^{l+1})     <-- ARGUMENTS SWAPPED
Derivation: dVbar^l = eta_bar * s_{l+1} * outer(a^{l+1}, abar^{l+1});  B = D_d ⊙ (Wbar Vbar^T)
so  Wbar @ d(Vbar^T) = outer(Wbar @ abar^{l+1}, a^{l+1}) = outer(hbar^l, a^{l+1}).
The swap predicts exactly the residual growth I measured (1.2% at l=0, O(1) at l=1,2).

IDENTITY UNDER TEST:
    dot{E}_p = -gamma E_p + eta*s_l*outer(abar^l, h^{l+1})
                          - eta_bar*s_{l+1}*D_d ⊙ outer(hbar^l, a^{l+1})
with dWbar/dt = 0. Measured AFTER WARMUP so E_p is O(1) (my second defect was dividing
by E_p = 0 at initialization).

D_d FIDELITY: paper benchmarks use D_d = I; this module uses D_d = relu. Both arms shown.
A configuration valid only with D_d = relu does NOT validate the paper's configuration.
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
    """Exact RHS blocks. Returns (|E_p|, up, dn) with dn FIXED."""
    Du = (h[l] > 0).to(net.dtype)
    F = net.V[l] @ (Du[:, None] * net.W[l].t())
    hbar = net.Wbar[l] @ ab[l + 1]
    Dd = torch.ones_like(hbar) if dd_I else (hbar > 0).to(net.dtype)
    B = Dd[:, None] * (net.Wbar[l] @ net.Vbar[l].t())
    a_prev = net._prep(h[l])          # a^l
    a_next = net._prep(h[l + 1])      # a^{l+1}
    up = net.eta * s(net, l) * torch.outer(ab[l], a_next * 0.0 + h[l + 1])
    dn = net.eta_bar * s(net, l + 1) * (Dd[:, None] * torch.outer(hbar, a_next))
    return float((F - B).norm()), up, dn


g = torch.Generator().manual_seed(0)
xs = [torch.randn(DIMS[0], generator=g) for _ in range(4)]
tg = [torch.randn(DIMS[-1], generator=g) for _ in range(4)]

print("=" * 86)
print(f"C1 with FIXED downstream term, measured after {WARMUP} warmup steps")
print("=" * 86)
receipt = {}
for dd_I in (False, True):
    tag = "D_d = I   (PAPER)" if dd_I else "D_d = relu (MODULE)"
    net = DualStreamNetwork(DIMS, gamma=GAMMA, eta_w=ETA_W, seed=0)
    for t in range(WARMUP):
        i = t % 4
        with torch.no_grad():
            net.sal_step(xs[i], 2.0 * (net.activations(xs[i])[-1] - tg[i]))
    print(f"\n--- {tag} ---")
    print(f"{'l':>2} {'|Ep|':>9} {'|up|':>9} {'|dn|':>9} {'|up-dn|':>9} "
          f"{'resid/|Ep|':>11} {'resid/|force|':>13}")
    rows = {}
    for l in range(net.n):
        with torch.no_grad():
            h0 = net.activations(xs[0])
            ab0 = net.feedback(h0, 2.0 * (h0[-1] - tg[0]))
            e0, up, dn = blocks(net, h0, ab0, l, dd_I)
            net.sal_step(xs[0], 2.0 * (h0[-1] - tg[0]), retract=False)
            h1 = net.activations(xs[0])
            ab1 = net.feedback(h1, 2.0 * (h1[-1] - tg[0]))
            e1 = blocks(net, h1, ab1, l, dd_I)[0]
        force = abs(float(up.norm()) - float(dn.norm()))
        pred = (1.0 - GAMMA) * e0 + float(up.norm()) - float(dn.norm())
        res = abs(e1 - pred)
        r_e = res / max(e0, 1e-30)
        r_f = res / max(force, 1e-30)
        rows[l] = {"ep": e0, "ep_next": e1, "pred": pred, "up": float(up.norm()),
                   "dn": float(dn.norm()), "resid": res, "resid_rel_ep": r_e,
                   "resid_rel_forcing": r_f}
        print(f"{l:>2} {e0:>9.4f} {float(up.norm()):>9.4f} {float(dn.norm()):>9.4f} "
              f"{force:>9.4f} {r_e:>11.4e} {r_f:>13.4e}")
    holds = all(v["resid_rel_ep"] < 0.10 for v in rows.values())
    tagk = "D_d_I" if dd_I else "D_d_relu"
    receipt[tagk] = {"layers": rows, "c1_holds": holds, "warmup": WARMUP}
    print(f"  C1 HOLDS ({tag}, resid/|Ep| < 0.10 every layer): {holds}")

print()
print("=" * 86)
print("SUMMARY")
print("=" * 86)
for k, v in receipt.items():
    worst = max(r["resid_rel_ep"] for r in v["layers"].values())
    print(f"  {k:9}  holds={v['c1_holds']}  worst resid/|Ep| = {worst:.4e}")
print("\n  If the module arm now holds at every layer, the SWAPPED-ARGUMENT diagnosis")
print("  is confirmed and the ODE is VALIDATED for the module's configuration.")

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
p = os.path.join(root, "design", "zone_a", "evidence",
                 "henri_ep_ode_c1_fixed_observed.json")
os.makedirs(os.path.dirname(p), exist_ok=True)
with open(p, "w", encoding="utf-8") as fh:
    json.dump(receipt, fh, indent=2)
print(f"[receipt] {p}")
