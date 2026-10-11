"""probe_ep_ode.py -- run the DERIVED E_p ODE against the module's real dynamics.

Supersedes the unrun Temp copy, which crashed on a 2-D/1-D shape defect of mine.
This is a TRACKED artifact: the Temp copy carried no evidence.

DERIVATION (Eq. 13/14, matched eta_V = eta_Vbar = eta, and dWbar/dt = 0):

    E_p^l = F^l - B^l,   F^l = V^l D_u^l (W^l)^T,   B^l = D_d^l Wbar^l (Vbar^l)^T

    dF/dt = (dV/dt) D_u (W)^T = eta * abar^l (h^{l+1})^T - gamma F^l
    dB/dt = D_d Wbar (dVbar^T/dt) = eta * abar^l (a^{l+1})^T - gamma B^l
            using D_d Wbar abar^{l+1} = abar^l (the module's own feedback recursion).

    =>  dot{E}_p = -gamma E_p + eta * abar^l (h^{l+1} - a^{l+1})^T
                 = -gamma E_p + eta * outer(abar^l, (1 - D_u^{l+1}) . h^{l+1})

THE SIMPLIFICATION IS EXACT, not an approximation. With a^l = _prep(h^l) = c * relu(h^l)
for c > 0, we have D_u^l . a^l = relu(a^l) = a^l, so a^T T = W a = h^{l+1}.

WHAT IT TESTS
  C1  One-step identity: E_p_new ~ (1-gamma) E_p_old + L_old. Residual small => the ODE
      holds under the module's dynamics. Residual O(1) => a rate mismatch or a dT/dt term.
  C2  The paper's reported QUANTITY is the RATIO ||E_p||/||F|| (Fig. S3 D), NOT the
      absolute norm. A rising absolute norm is compatible with a decreasing ratio.
  C3  The forcing term is carried ONLY by dead units (1 - D_u^{l+1}). If the dead-unit
      fraction -> 0 the forcing vanishes and E_p decays with NO Wbar change at all.

FIDELITY CAVEAT. The paper's benchmark runs use D_d = I ("take D = I in the backward
pass of Algorithm 1"); this module uses D_d = relu. Both arms are reported for C2, so the
cross-configuration comparison is explicit rather than hidden.

NO BAR IS MOVED HERE. This probe measures only. It writes one receipt.
"""
import json
import os
import statistics as st
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "henri_core"))
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

GAMMA = 0.02
DEFAULT_DIMS = (32, 64, 64, 32)
STEPS = 150
NSAMP = 4


def terms(net, hs, ab, l, dd_I):
    """Return (E_p, F, B, L, dead_frac) at layer l, shape (d_l, d_{l+1})."""
    Du = (hs[l] > 0).to(net.dtype)                       # D_u^l
    Du1 = (hs[l + 1] > 0).to(net.dtype)                  # D_u^{l+1}
    F = net.V[l] @ (Du[:, None] * net.W[l].t())          # V^l D_u^l (W^l)^T
    hbar = net.Wbar[l] @ ab[l + 1]
    Braw = net.Wbar[l] @ net.Vbar[l].t()
    B = Braw if dd_I else ((hbar > 0).to(net.dtype)[:, None] * Braw)
    Ep = F - B
    row = (1.0 - Du1) * hs[l + 1]                        # exact forcing row
    L = net.eta * torch.outer(ab[l], row)
    return (float(Ep.norm()), float(F.norm()), float(B.norm()),
            float(L.norm()), float(1.0 - Du1.mean()))


def make_batch(dims, seed, n=NSAMP):
    g = torch.Generator().manual_seed(seed)
    xs = [torch.randn(dims[0], generator=g) for _ in range(n)]
    tg = [torch.randn(dims[-1], generator=g) for _ in range(n)]
    return xs, tg


def run(dims, seed, steps=STEPS):
    net = DualStreamNetwork(dims, gamma=GAMMA, seed=seed)
    xs, tg = make_batch(dims, seed)
    rows = []
    for t in range(steps + 1):
        acc = {l: [] for l in range(net.n)}
        accI = {l: [] for l in range(net.n)}      # D_d = I arm (the paper's benchmark)

        for xx, tt in zip(xs, tg):
            with torch.no_grad():
                h = net.activations(xx)
                a2 = net.feedback(h, 2.0 * (h[-1] - tt))
                for l in range(net.n):
                    acc[l].append(terms(net, h, a2, l, False))
                    accI[l].append(terms(net, h, a2, l, True))
        rows.append({l: [st.mean(r[i] for r in acc[l]) for i in range(5)]
                        + [st.mean(r[0] for r in accI[l])] for l in acc})
        if t < steps:
            i = t % NSAMP
            with torch.no_grad():
                x0 = xs[i]
                net.sal_step(x0, 2.0 * (net.activations(x0)[-1] - tg[i]))
    return net, xs, tg, rows


def main():
    out = {}
    print("=" * 78)
    print("C1  ONE-STEP IDENTITY:  E_p_new = (1-gamma) E_p_old + L_old")
    print("=" * 78)
    net, xs, tg, rows = run(DEFAULT_DIMS, 0)
    x, tt = xs[0], tg[0]
    with torch.no_grad():
        h = net.activations(x)
        ab = net.feedback(h, 2.0 * (h[-1] - tt))
        pre = {l: terms(net, h, ab, l, False) for l in range(net.n)}
        net.sal_step(x, 2.0 * (h[-1] - tt))
        h2 = net.activations(x)
        ab2 = net.feedback(h2, 2.0 * (h2[-1] - tt))
    print(f"{'l':>2} {'|Ep|':>10} {'|L|':>10} {'resid/|Ep|':>12} {'resid/|L|':>12} "
          f"{'dead':>8}")
    c1 = {}
    for l in range(net.n):
        Ep0, F0, B0, L0, dead = pre[l]
        Ep1 = terms(net, h2, ab2, l, False)[0]
        pred = (1.0 - GAMMA) * Ep0 + L0
        res = abs(Ep1 - pred)
        r_e = res / max(Ep0, 1e-30)
        r_l = res / max(L0, 1e-30)
        c1[l] = {"ep_norm": Ep0, "l_norm": L0, "ep_new": Ep1, "pred": pred,
                 "resid_abs": res, "resid_rel_ep": r_e, "resid_rel_l": r_l,
                 "dead_frac": dead}
        print(f"{l:>2} {Ep0:>10.4f} {L0:>10.4f} "
              f"{r_e:>12.4e} {r_l:>12.4e} {dead:>8.4f}")
    out["C1"] = c1

    print()
    print("=" * 78)
    print("C2  THE PAPER'S QUANTITY:  ratio ||E_p|| / ||F||   (Fig. S3 D)")
    print("=" * 78)
    c2 = {}
    for l in (0, 1):
        print(f"\n--- layer {l} ---")
        print(f"{'t':>4} {'|Ep|':>9} {'|F|':>9} {'RATIO':>9} {'dead':>8} "
              f"{'RATIO_I':>9}")
        for t in (10, 30, 60, 90, 120, 150):
            Ep, F, B, L, dead, EpI = rows[t][l]
            print(f"{t:>4} {Ep:>9.4f} {F:>9.3f} {Ep / F if F > 0 else float('nan'):>9.4f} "
                  f"{dead:>8.4f} "
                  f"{(EpI / F if F > 0 else float('nan')):>9.4f}")
        e10, eN = rows[10][l][0], rows[-1][l][0]
        f10, fN = max(rows[10][l][1], 1e-30), max(rows[-1][l][1], 1e-30)
        i10, iN = rows[10][l][5], rows[-1][l][5]
        c2[l] = {"abs_rise": eN / max(e10, 1e-30), "ratio_10": e10 / f10,
                 "ratio_150": eN / fN, "dead_150": rows[-1][l][4],
                 "ratioI_10": i10 / f10, "ratioI_150": iN / fN,
                 "dd_mask": "relu (module)", "dd_I": "I (paper benchmarks)"}
        print(f"  ABSOLUTE |E_p| {e10:.4f} -> {eN:.4f}  "
              f"({'ROSE' if eN > e10 * 1.1 else 'FLAT/DOWN'}, x{eN / max(e10, 1e-30):.3f})")
        print(f"  RATIO ||Ep||/||F|| {e10 / f10:.4f} -> {eN / fN:.4f}  "
              f"({'DECREASING' if eN / fN < e10 / f10 * 0.9 else 'NOT DECREASING'})")
    out["C2"] = c2

    print()
    print("=" * 78)
    print("C3  DEAD-UNIT HYPOTHESIS:  does the E_p plateau track the dead fraction?")
    print("=" * 78)
    print(f"{'width':>6} {'seed':>5} {'dead_150':>9} {'|Ep|_150':>10} {'ratio_150':>10}")
    c3 = []
    for w in (16, 32, 64):
        for sd in (0, 1, 2):
            dims = (w, 2 * w, 2 * w, w)
            _, _, _, rr = run(dims, sd, steps=120)
            dead = rr[-1][0][4]
            ep = rr[-1][0][0]
            rt = ep / max(rr[-1][0][1], 1e-30)
            c3.append({"width": w, "seed": sd, "dead": dead, "ep": ep, "ratio": rt})
            print(f"{w:>6} {sd:>5} {dead:>9.4f} {ep:>10.4f} {rt:>10.4f}")
    out["C3"] = c3
    if len(c3) > 2:
        ds = [r["dead"] for r in c3]
        es = [r["ep"] for r in c3]
        md, me = st.mean(ds), st.mean(es)
        num = sum((a - md) * (b - me) for a, b in zip(ds, es))
        den = (sum((a - md) ** 2 for a in ds) ** 0.5) * (sum((b - me) ** 2 for b in es) ** 0.5)
        r = num / den if den > 0 else float("nan")
        out["C3_corr_dead_vs_ep"] = r
        print(f"\n  Pearson r(dead_frac, |Ep|_150) = {r:+.4f}  (n={len(c3)})")
        print("  Strong positive r => the disease lives in dead units, not in Wbar.")

    print()
    print("=" * 78)
    verdict = {
        "c1_holds": all(v["resid_rel_l"] < 0.10 for v in c1.values()),
        "c2_ratio_decreasing": all(
            c2[l]["ratio_150"] < c2[l]["ratio_10"] * 0.9 for l in c2),
        "c2_abs_rose": all(c2[l]["abs_rise"] > 1.1 for l in c2),
    }
    print(f"C1 ODE identity holds (resid/|L| < 0.10):   {verdict['c1_holds']}")
    print(f"C2 ratio ||Ep||/||F|| decreasing:           {verdict['c2_ratio_decreasing']}")
    print(f"C2 absolute ||Ep|| rose:                    {verdict['c2_abs_rose']}")
    if verdict["c2_abs_rose"] and verdict["c2_ratio_decreasing"]:
        print("VERDICT: PAPER RATIO REPRODUCED. The earlier 'E_p ROSE' reading was a")
        print("         reporting-unit error: the ratio falls while the norm rises.")
    elif verdict["c1_holds"]:
        print("VERDICT: ODE VALIDATED; E_p genuinely plateaus/rises in BOTH units.")
    else:
        print("VERDICT: ODE DOES NOT HOLD at this configuration. Report as FALSIFIED.")
    out["verdict"] = verdict

    root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", ".."))
    path = os.path.join(root, "design", "zone_a", "evidence", "henri_ep_ode_observed.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\n[receipt] {path}")


if __name__ == "__main__":
    main()
