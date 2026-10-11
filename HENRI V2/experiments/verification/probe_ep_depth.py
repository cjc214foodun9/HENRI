"""probe_ep_depth.py -- settle E_p at depth by EXTENDING the identity with the omitted terms.

CLOSING THE THREE-CLASS DEFECT (measured last turn):
  the one-step identity closed at layer 0 (resid/|E_p| = 1.20e-02) and failed at depth
  (4.84 at l=1, 20.50 at l=2). diag_ep_vdt.py localized it: at layer 2 the omitted term
  |V.dT| = 2341.5 against the kept term |dV.T| = 227.6.

EXACT ACCOUNTING (every term retained):
  F^l = V^l @ M^l,   M^l = diag(D_u^l) @ (W^l)^T
  Delta F = (dV)@M0 + V0@(dM) + (dV)@(dM)              <- algebraically exact
  Delta B = Dd1 . (Wbar0 @ Vbar1^T) - Dd0 . (Wbar0 @ Vbar0^T)
  Delta E_p = Delta F - Delta B                         <- exact to machine precision

ODE PREDICTION (the one that failed at depth):
  pred = -gamma*F0 + eta*s_l*outer(abar0, h0_next)
                    - ( eta_bar*s_next*Dd0 . outer(hbar0, a0_next) - gamma*B0 )
  resid = Delta E_p - pred

THREE ARMS:
  A  EXACT identity (all terms)      -> MUST close to machine precision at EVERY layer
  B  ODE prediction                  -> closes at l=0, degrades at depth
  C  ODE prediction at eta/4         -> if relative resid FALLS, the depth failure is a
                                        LARGE-STEP regime, not a derivation error
Arm A is the discriminator: if A closes, the derivation is right and only the omitted-term
approximation fails. If A does NOT close, the accounting itself is wrong.
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "henri_core"))
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

DIMS = (32, 64, 64, 32)
WARMUP = 40


def s(net, l):
    if net.drive_scale == "sqrt_d":
        return 1.0 / math.sqrt(net.dims[l])
    if net.drive_scale == "d":
        return 1.0 / net.dims[l]
    return 1.0


def snap(net, x, tgt):
    h = net.activations(x)
    ab = net.feedback(h, 2.0 * (h[-1] - tgt))
    return h, ab


def Mmat(h, W):
    return (h > 0).to(W.dtype)[:, None] * W.t()


def Fmat(V, M):
    return V @ M


def Bmat(hbar_pre, Wbar, Vbar):
    return (hbar_pre > 0).to(Wbar.dtype)[:, None] * (Wbar @ Vbar.t())


g = torch.Generator().manual_seed(0)
xs = [torch.randn(DIMS[0], generator=g) for _ in range(4)]
tg = [torch.randn(DIMS[-1], generator=g) for _ in range(4)]

print("=" * 96)
print("ARM A: EXACT accounting must close at EVERY layer;  B/C: ODE prediction vs step size")
print("=" * 96)
print(f"{'arm':>10} {'l':>2} {'resid/|Ep|':>12} {'resid/|force|':>14} "
      f"{'|dV.M0|':>10} {'|V0.dM|':>10} {'|dV.dM|':>9} {'|dB|':>9}")
rec = {}
for scale in (1.0, 0.25):
    tag = "B_eta" if scale == 1.0 else "C_eta4"
    net = DualStreamNetwork(DIMS, gamma=0.02, eta_w=0.002 * scale,
                            eta=0.05 * scale, eta_bar=0.05 * scale, seed=0)
    for t in range(WARMUP):
        i = t % 4
        with torch.no_grad():
            net.sal_step(xs[i], 2.0 * (net.activations(xs[i])[-1] - tg[i]))
    x, t0 = xs[0], tg[0]
    with torch.no_grad():
        h0, ab0 = snap(net, x, t0)
        V0 = [v.clone() for v in net.V]
        Vb0 = [v.clone() for v in net.Vbar]
        W0 = [w.clone() for w in net.W]
        Wb0 = [w.clone() for w in net.Wbar]
        M0 = [Mmat(h0[l], W0[l]) for l in range(net.n)]
        F0 = [Fmat(V0[l], M0[l]) for l in range(net.n)]
        hb0 = [Wb0[l] @ ab0[l + 1] for l in range(net.n)]
        B0 = [Bmat(hb0[l], Wb0[l], Vb0[l]) for l in range(net.n)]
        Ep0 = [F0[l] - B0[l] for l in range(net.n)]
        net.sal_step(x, 2.0 * (h0[-1] - t0), retract=False)
        h1, ab1 = snap(net, x, t0)
        M1 = [Mmat(h1[l], net.W[l]) for l in range(net.n)]
        F1 = [net.V[l] @ M1[l] for l in range(net.n)]
        hb1 = [net.Wbar[l] @ ab1[l + 1] for l in range(net.n)]
        B1 = [Bmat(hb1[l], net.Wbar[l], net.Vbar[l]) for l in range(net.n)]
        Ep1 = [F1[l] - B1[l] for l in range(net.n)]
    for l in range(net.n):
        dV, dM = net.V[l] - V0[l], M1[l] - M0[l]
        t1, t2, t3 = dV @ M0[l], V0[l] @ dM, dV @ dM
        dF = F1[l] - F0[l]
        dB = B1[l] - B0[l]
        dEp = Ep1[l] - Ep0[l]
        exact = t1 + t2 + t3 - dB
        residual = dEp - exact                     # ARM A
        a_next = net._prep(h0[l + 1])
        pred = (-0.02 * F0[l]
                + net.eta * s(net, l) * torch.outer(ab0[l], h0[l + 1])
                - (net.eta_bar * s(net, l + 1)
                   * ((hb0[l] > 0).to(net.dtype)[:, None] * torch.outer(hb0[l], a_next))
                   - 0.02 * B0[l]))
        r_ode = dEp - pred                          # ARM B/C
        force = abs(float((net.eta * s(net, l) * torch.outer(ab0[l], h0[l + 1])).norm())
                    - float((net.eta_bar * s(net, l + 1) * torch.outer(hb0[l], a_next)).norm()))
        e0 = float(Ep0[l].norm())
        rec[(tag, l)] = {
            "armA_resid_rel_ep": float(residual.norm()) / max(e0, 1e-30),
            "armB_resid_rel_ep": float(r_ode.norm()) / max(e0, 1e-30),
            "resid_rel_force": float(r_ode.norm()) / max(force, 1e-30),
            "dV_M0": float(t1.norm()), "V0_dM": float(t2.norm()),
            "dV_dM": float(t3.norm()), "dB": float(dB.norm()), "ep0": e0,
        }
        r = rec[(tag, l)]
        print(f"{tag:>10} {l:>2} {r['armB_resid_rel_ep']:>12.4e} {r['resid_rel_force']:>14.4e} "
              f"{r['dV_M0']:>10.4f} {r['V0_dM']:>10.4f} {r['dV_dM']:>9.4f} {r['dB']:>9.4f}")
    print(f"           -> ARM A exact-accounting residual (max over layers): "
          f"{max(rec[(tag,l)]['armA_resid_rel_ep'] for l in range(net.n)):.3e}")

print()
print("=" * 96)
print("VERDICT")
print("=" * 96)
maxA = max(v["armA_resid_rel_ep"] for v in rec.values())
b0 = rec[("B_eta", 0)]["armB_resid_rel_ep"]
b2 = rec[("B_eta", 2)]["armB_resid_rel_ep"]
c2 = rec[("C_eta4", 2)]["armB_resid_rel_ep"]
print(f"  ARM A (exact accounting) max residual : {maxA:.3e}")
print(f"    {'CLOSES -> derivation validated; only the omitted-term approximation fails' if maxA < 1e-4 else 'DOES NOT CLOSE -> the accounting itself is wrong'}")
print(f"  ARM B ODE  l=0 {b0:.3e}  ->  l=2 {b2:.3e}   ({b2/max(b0,1e-30):.1f}x worse at depth)")
print(f"  ARM C ODE at eta/4, l=2 : {c2:.3e}   "
      f"({'FALLS -> LARGE-STEP REGIME confirmed' if c2 < b2 else 'does not fall -> derivation error'})")
print(f"  ratio C/B at l=2 : {c2/max(b2,1e-30):.4f}")
