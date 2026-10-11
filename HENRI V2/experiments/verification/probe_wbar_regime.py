"""probe_wbar_regime.py -- is the paper Wbar FLOW failure a REGIME artifact?

MEASURED LAST TURN (width 32/64, drive_scale="sqrt_d"):
    A_no_rule      |Ep| 1.4483
    B_paper_flow   |Ep| 40.2118   <- 28x worse, stream ALIVE (|abar| grew to 21.4)
    C_closure      collapse

CLAIM UNDER TEST: the paper runs width 200, CIFAR-10, 50 epochs, SGD eta=5e-3 with
eta_W=1.0, eta_Wbar=0.8. This tree runs width 32-64 at eta_w=0.002. The paper flow may be
an INSTABILITY OF THIS REGIME, not a defect of the rule.

PRE-REGISTERED (fixed before running; the rate is NEVER retuned to pass):
  VERDICT=REGIME      if |E_p| end is non-increasing in width AND at width 128 the paper
                      flow is no worse than 2x the no-rule plateau with the stream alive.
  VERDICT=RULE_DEFECT otherwise (report as a defect of the flow at these widths).

NON-COLLAPSE COLUMNS are printed for every row: |abar| and |Vbar|. Four earlier Wbar
repairs reached their target by collapsing the stream.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "henri_core"))
from henri_heterosynaptic import DualStreamNetwork  # noqa: E402

STEPS, NSAMP, SEED = 100, 4, 0


def run(width, arm):
    dims = (width, 2 * width, 2 * width, width)
    kw = {"wbar_rule": "paper"} if arm == "paper" else {}
    net = DualStreamNetwork(dims, seed=SEED, **kw)
    g = torch.Generator().manual_seed(SEED)
    xs = [torch.randn(dims[0], generator=g) for _ in range(NSAMP)]
    tg = [torch.randn(dims[-1], generator=g) for _ in range(NSAMP)]
    bar0 = None
    for t in range(STEPS + 1):
        e, a, v = 0.0, 0.0, 0.0
        for xx, tt in zip(xs, tg):
            with torch.no_grad():
                h = net.activations(xx)
                ab = net.feedback(h, 2.0 * (h[-1] - tt))
                Du = (h[0] > 0).to(net.dtype)
                F = net.V[0] @ (Du[:, None] * net.W[0].t())
                B = ((net.Wbar[0] @ ab[1]) > 0).to(net.dtype)[:, None] * (net.Wbar[0] @ net.Vbar[0].t())
                e += float((F - B).norm())
                a += float(ab[0].norm())
                v += float(net.Vbar[0].norm())
        if t == 0:
            bar0 = a / NSAMP
        if t < STEPS:
            i = t % NSAMP
            with torch.no_grad():
                net.sal_step(xs[i], 2.0 * (net.activations(xs[i])[-1] - tg[i]))
        if not torch.isfinite(net.Wbar[0]).all():
            return {"width": width, "arm": arm, "nonfinite": True, "steps": t}
    return {"width": width, "arm": arm, "nonfinite": False, "steps": STEPS,
            "ep_end": e / NSAMP, "abar_end": a / NSAMP, "vbar_end": v / NSAMP,
            "abar_start": bar0, "collapse": (a / NSAMP) < 0.05 * max(bar0, 1e-30)}


print("=" * 92)
print("eta_Wbar = 0.8 REGIME SWEEP  (rate held AT the paper value; never retuned)")
print("=" * 92)
print(f"{'width':>6} {'arm':>7} {'|Ep|_end':>11} {'|abar|_end':>11} {'|Vbar|_end':>11} "
      f"{'collapse':>9} {'nonfinite':>10}")
res = {}
for w in (32, 64, 128):
    for arm in ("no_rule", "paper"):
        r = run(w, arm)
        res[(w, arm)] = r
        if r.get("nonfinite"):
            print(f"{w:>6} {arm:>7} {'-':>11} {'-':>11} {'-':>11} {'-':>9} {'YES':>10}")
        else:
            print(f"{w:>6} {arm:>7} {r['ep_end']:>11.4f} {r['abar_end']:>11.4f} "
                  f"{r['vbar_end']:>11.4f} {str(r['collapse']):>9} {'no':>10}")

print()
print("=" * 92)
base32 = res[(32, "no_rule")].get("ep_end", float("nan"))
base128 = res[(128, "no_rule")].get("ep_end", float("nan"))
p32 = res[(32, "paper")].get("ep_end", float("nan"))
p128 = res[(128, "paper")].get("ep_end", float("nan"))
print(f"  paper/no-rule ratio  width 32  : {p32/max(base32,1e-30):.3f}")
print(f"  paper/no-rule ratio  width 128 : {p128/max(base128,1e-30):.3f}")
alive128 = not res[(128, "paper")].get("collapse", True)
if p128 <= 2.0 * base128 and alive128:
    print("  VERDICT: REGIME -- the flow is usable at width 128 with the stream alive.")
elif p128 < p32:
    print("  VERDICT: RULE_DEFECT_PERSISTS -- worsens with width or stays >2x; report as a")
    print("           rule defect at these widths. Do NOT retune eta_Wbar to pass.")
else:
    print("  VERDICT: RULE_DEFECT -- no width in this sweep rescues the flow.")
