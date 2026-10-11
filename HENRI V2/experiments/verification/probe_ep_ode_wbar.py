"""G-H2b: corrected E_p ODE, and the paper-faithful Wbar FLOW vs three falsified solves.

PART 1 -- CORRECTED C1 IDENTITY. My first probe used the WRONG DOWNSTREAM TERM: it
assumed dB/dt = eta_bar * outer(abar^l, a^{l+1}). Correctly:
    B^l   = D_d^l Wbar^l (Vbar^l)^T ,   D_d^l = relu(hbar^l)  [module dd_mask=True]
    Vbar^l += eta_bar * s_{l+1} * outer(a^{l+1}, abar^{l+1}) - gamma Vbar^l
    => dB/dt = eta_bar*s_{l+1}*relu(hbar^l) . outer(Wbar^l a^{l+1}, abar^{l+1}) - gamma B^l
And dF/dt = eta*s_l*outer(abar^l, h^{l+1}) - gamma F^l  because T^T a^l = W^l a^l = h^{l+1}.
Hence, with dWbar/dt = 0:
    dot{E}_p = -gamma E_p + eta*s_l*outer(abar^l, h^{l+1})
                          - eta_bar*s_{l+1}*relu(hbar^l) . outer(Wbar^l a^{l+1}, abar^{l+1})
That is the identity this part tests. Residual small => the ODE is validated.

PART 2 -- THE PAPER Wbar FLOW. Hypothesis under test:
  H1  The paper rule is a FLOW (matched-mirror local outer product at 400x rate with a
      SEPARATE 1e-3 decay), not a SOLVE. Every solve pinned Wbar to E_p=0 at one time
      point from a rank-deficient Vbar and collapsed the stream to |abar| = 0.0000.
  H2  If a fast Wbar flow makes B track the backward stream's own target, E_p's ratio
      should fall BELOW the no-rule plateau.

PRE-REGISTERED KILL TEST (fixed before running; no retuning of eta_wbar to pass):
  FALSIFIED if any of: |abar|_end < 0.05 * |abar|_start (collapse), OR non-finite, OR
  E_p_end >= E_p_start (no reduction).
  PASS-if-alive requires: stream alive AND E_p_end < E_p_start AND no NaN.

NON-COLLAPSE COLUMNS ARE PRINTED FOR EVERY ARM. Three of four earlier Wbar repairs "hit
their target" by killing the stream; only |abar|/|Vbar| revealed it.
"""
import json
import math
import os
import sys

import torch

sys.path.insert(0, "HENRI V2/henri_core")
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

DIMS = (32, 64, 64, 32)
GAMMA, ETA_W, ETA = 0.02, 0.002, 0.05
STEPS, NSAMP, SEED = 150, 4, 0


def scale_of(net, l):
    if net.drive_scale == "sqrt_d":
        return 1.0 / math.sqrt(net.dims[l])
    if net.drive_scale == "d":
        return 1.0 / net.dims[l]
    return 1.0


def parts(net, h, ab, l):
    """Return the exact RHS building blocks at layer l. All floats."""
    Du1 = (h[l + 1] > 0).to(net.dtype)
    Du = (h[l] > 0).to(net.dtype)
    F = net.V[l] @ (Du[:, None] * net.W[l].t())
    hbar = net.Wbar[l] @ ab[l + 1]
    Braw = net.Wbar[l] @ net.Vbar[l].t()
    B = (hbar > 0).to(net.dtype)[:, None] * Braw
    Ep = F - B
    a_l = net._prep(h[l])
    a_1 = net._prep(h[l + 1]) if l + 1 < net.n else ab[l + 1]
    # upstream contribution: eta * s_l * outer(abar^l, h^{l+1})
    up = net.eta * scale_of(net, l) * torch.outer(ab[l], h[l + 1])
    # downstream contribution: eta_bar * s_{l+1} * relu(hbar) . outer(Wbar a^{l+1}, abar^{l+1})
    Wb_a = net.Wbar[l] @ a_1
    dn = (net.eta_bar * scale_of(net, l + 1) * (hbar > 0).to(net.dtype)[:, None]
          * torch.outer(Wb_a, ab[l + 1]))
    return (float(Ep.norm()), float(F.norm()), float(B.norm()),
            up, dn, float(1.0 - Du1.mean()))


def batch(seed, n=NSAMP):
    g = torch.Generator().manual_seed(seed)
    return ([torch.randn(DIMS[0], generator=g) for _ in range(n)],
            [torch.randn(DIMS[-1], generator=g) for _ in range(n)])


print("=" * 84)
print("PART 1  CORRECTED C1: dot{E_p} = -gamma E_p + up - dn")
print("=" * 84)
net = DualStreamNetwork(DIMS, gamma=GAMMA, eta_w=ETA_W, seed=SEED)
xs, tg = batch(SEED)
x, tt = xs[0], tg[0]
with torch.no_grad():
    h0 = net.activations(x)
    ab0 = net.feedback(h0, 2.0 * (h0[-1] - tt))
    pre = {l: parts(net, h0, ab0, l) for l in range(net.n)}
    net.sal_step(x, 2.0 * (h0[-1] - tt), retract=False)
    h1 = net.activations(x)
    ab1 = net.feedback(h1, 2.0 * (h1[-1] - tt))
print(f"{'l':>2} {'|Ep|':>9} {'|up|':>9} {'|dn|':>9} {'|a+dn|':>9} "
      f"{'resid/|Ep|':>11} {'dead':>7}")
c1 = {}
for l in range(net.n):
    Ep0, F0, B0, up, dn, dead = pre[l]
    Ep1 = parts(net, h1, ab1, l)[0]
    pred = (1.0 - GAMMA) * Ep0 + float(up.norm()) - float(dn.norm())
    resid = abs(Ep1 - pred)
    rel = resid / max(Ep0, 1e-30)
    c1[l] = {"ep0": Ep0, "ep1": Ep1, "up": float(up.norm()), "dn": float(dn.norm()),
             "pred": pred, "resid_abs": resid, "resid_rel": rel, "dead": dead}
    print(f"{l:>2} {Ep0:>9.4f} {float(up.norm()):>9.4f} {float(dn.norm()):>9.4f} "
          f"{float((up - dn).norm()):>9.4f} {rel:>11.4e} {dead:>7.4f}")
c1_ok = all(v["resid_rel"] < 0.10 for v in c1.values())
print(f"\nC1 CORRECTED IDENTITY HOLDS (resid/|Ep| < 0.10 on every layer): {c1_ok}")
if not c1_ok:
    print("  -> the ODE still misses a term; report the residual, do NOT claim validation.")

print()
print("=" * 84)
print("PART 2  Wbar FLOW (paper) vs three falsified SOLVES -- with non-collapse columns")
print("=" * 84)
ARMS = [("A_no_rule", dict()),
        ("B_paper_flow", dict(wbar_rule="paper")),
        ("C_falsified_closure", dict(ep_closure=True))]


def alive(t):
    try:
        return bool(torch.isfinite(t).all().item())
    except Exception:
        return False


results = {}
for name, kw in ARMS:
    net = DualStreamNetwork(DIMS, gamma=GAMMA, eta_w=ETA_W, seed=SEED, **kw)
    xs, tg = batch(SEED)
    ab_start = None
    rec = []
    ok = True
    for t in range(STEPS + 1):
        e_sum, a_sum, v_sum = 0.0, 0.0, 0.0
        for xx, tgt in zip(xs, tg):
            with torch.no_grad():
                h = net.activations(xx)
                ab = net.feedback(h, 2.0 * (h[-1] - tgt))
                e_sum += parts(net, h, ab, 0)[0]
                a_sum += float(ab[0].norm())
                v_sum += float(net.Vbar[0].norm())
        rec.append((e_sum / NSAMP, a_sum / NSAMP, v_sum / NSAMP))
        if t == 0:
            ab_start = rec[0][1]
        if not (alive(net.V[0]) and alive(net.Vbar[0])):
            ok = False
            break
        if t < STEPS:
            i = t % NSAMP
            with torch.no_grad():
                net.sal_step(xs[i], 2.0 * (net.activations(xs[i])[-1] - tg[i]))
    e0, a0, v0 = rec[0]
    eN, aN, vN = rec[-1]
    collapse = (not ok) or (aN < 0.05 * max(a0, 1e-30))
    nan = not (alive(net.Vbar[0]) and alive(net.Wbar[0]))
    results[name] = {"ep_start": e0, "ep_end": eN, "abar_start": a0, "abar_end": aN,
                     "vbar_start": v0, "vbar_end": vN, "collapse": collapse,
                     "nonfinite": nan, "steps_done": len(rec) - 1,
                     "ep_reduced": eN < e0}
    print(f"\n--- {name} ---")
    print(f"  |Ep|_l0      {e0:>10.4f} -> {eN:>10.4f}   "
          f"({'REDUCED' if eN < e0 else 'NOT REDUCED'})")
    print(f"  |abar|_l0    {a0:>10.4f} -> {aN:>10.4f}   "
          f"{'*** COLLAPSE ***' if aN < 0.05 * max(a0, 1e-30) else 'stream alive'}")
    print(f"  |Vbar|_l0    {v0:>10.4f} -> {vN:>10.4f}")
    print(f"  non-finite   {nan}   steps {len(rec) - 1}/{STEPS}")

paper = results["B_paper_flow"]
base = results["A_no_rule"]
print()
print("=" * 84)
if paper["nonfinite"] or paper["collapse"]:
    print("VERDICT: PAPER Wbar FLOW **FALSIFIED** (stream collapse or non-finite).")
    print("         Keep wbar_rule='paper' default-OFF. Record the instability; do NOT")
    print("         retune eta_wbar to hide it (that is bar-moving in disguise).")
elif paper["ep_reduced"] and paper["ep_end"] < base["ep_end"]:
    print("VERDICT: PAPER Wbar FLOW **PARTIAL PASS** -- reduces E_p below the no-rule")
    print("         plateau with the stream alive. Promote to a pre-registered gate test.")
else:
    print("VERDICT: PAPER Wbar FLOW **NO GAIN** -- stream alive but E_p does not fall")
    print("         below the no-rule plateau. FALSIFIED as a G-H2b remedy.")
print("=" * 84)

root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "..", "..", ".."))
path = os.path.join(root, "design", "zone_a", "evidence",
                    "henri_ep_ode_corrected_and_wbar_flow_observed.json")
os.makedirs(os.path.dirname(path), exist_ok=True)
with open(path, "w", encoding="utf-8") as fh:
    json.dump({"C1_identity": c1, "C1_holds": c1_ok, "wbar_arms": results,
               "paper_rate": {"eta_wbar": 0.8, "gamma_wbar": 1e-3,
                              "source": "PNAS SI: eta_W=1.0, eta_Wbar=0.8, "
                                        "eta_V=eta_Vbar=0.8, decay 1e-3 W/Wbar, 0.01 V/Vbar"},
               "kill_test": {"collapse_factor": 0.05, "bar": "ep_end < ep_start"}}, fh, indent=2)
print(f"\n[receipt] {path}")
