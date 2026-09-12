"""Stage 3 functional check, dimension-AWARE noise.

WHY THIS REPLACES THE PREVIOUS CHECK
------------------------------------
The first functional check used a fixed perturbation `0.10 * randn(D)` in every
dimension. At D=8192 that noise has norm 0.1*sqrt(8192) = 9.05, i.e. NINE times
the unit norm of the stored memory. The "on-manifold" query was therefore not
near the manifold at all, and the measured H(Y) was governed by that blowup, not
by beta. It also contradicted an earlier standalone run at the same nominal
setting (H=1.5354 vs H=0.7371) purely through RNG-sensitive logit spread.

That is the dimension-blindness error: a raw threshold/noise scale reused across
dimensions without sqrt(D) normalisation.

The codebase already documents the correct treatment
(HENRI V2/henri_hopfield_egress.py:45):

    sigma_elem = eps / sqrt(D)      # eps=0.15, D=65536 -> 5.86e-4

so the TOTAL perturbation norm is ~eps, comparable to the unit memory norm.

WHAT THIS MEASURES
------------------
H(Y) over the softmax retrieval weights, using the live
ContinuousHopfieldCleanup through henri_egress.TextEgress, at the resolved beta
(beta=8.0 vs the sqrt(D) default), for dimension-aware noise eps/sqrt(D).

Sweeps eps so the reader can see the regime rather than one contested point.

CPU only. No GPU, no cost.
"""
import importlib.util
import math
import os
import sys

import torch

REPO = r"C:\Users\chan\Desktop\HENRI 7B SWARM"
VER = os.path.join(REPO, "HENRI V2")
sys.path.insert(0, VER)
LN2 = math.log(2.0)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def ent_weights(w):
    w = w.reshape(-1).double()
    w = w / w.sum()
    return float(-(w * (w + 1e-12).log()).sum() / LN2)


def build_bank(D, M, seed=0):
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(D, M, generator=g))
    return q.T.contiguous()          # [M, D], orthonormal rows


def main():
    print("=" * 78)
    print("STAGE 3 FUNCTIONAL CHECK - DIMENSION-AWARE NOISE (eps/sqrt(D))")
    print("=" * 78)
    print("documented noise floor: sigma_elem = eps / sqrt(D)")
    print("(HENRI V2/henri_hopfield_egress.py:45) -- total ||noise|| ~ eps")
    print()

    e = load("henri_egress", os.path.join(VER, "henri_egress.py"))
    rows = []
    for D, M in ((8192, 1000), (512, 100), (65536, 200)):
        bank = build_bank(D, M)
        sigma = None
        for eps in (0.05, 0.15, 0.30):
            s = eps / math.sqrt(D)
            for label, beta in (("beta=8.0", 8.0),
                                ("sqrt(D)", math.sqrt(D)),
                                ("2*sqrt(D)", 2 * math.sqrt(D))):
                for m in list(sys.modules):
                    if m in ("henri_egress", "hopfield_cleanup"):
                        del sys.modules[m]
                eng = load("henri_egress", os.path.join(VER, "henri_egress.py"))
                model = eng.TextEgress(d_model=D)
                model.cleanup.beta = beta
                model.cleanup.store_engrams(bank)
                g = torch.Generator().manual_seed(123 + int(eps * 1000))
                probe = bank[0] + s * torch.randn(D, generator=g)
                probe = torch.nn.functional.normalize(probe, p=2, dim=-1)
                cos = float(torch.dot(probe, bank[0]))
                _, w = model.cleanup.retrieve(probe, return_weights=True)
                h = ent_weights(w)
                rows.append((D, M, eps, label, h, cos))
                if sigma is None:
                    sigma = s
        print(f"  D={D:<6} M={M:<6} sigma_elem(eps=0.15)={0.15/math.sqrt(D):.3e}")
        for (dd, mm, ee, lab, h, cos) in rows:
            if dd != D or mm != M:
                continue
            print(f"     eps={ee:<5} {lab:<11} H(Y)={h:9.4f} "
                  f"{'PASS' if h <= 1.2 else 'FAIL'}   cos(probe,true)={cos:.4f}")
        print()

    # the headline: dimension-aware eps=0.15, does sqrt(D) snap?
    print("-" * 78)
    print("HEADLINE: eps = 0.15 (documented), dimension-aware")
    ok = True
    for (D, M, eps, lab, h, cos) in rows:
        if abs(eps - 0.15) < 1e-9 and lab == "sqrt(D)":
            ok &= (h <= 1.2)
            print(f"  D={D:<6} M={M:<6} sqrt(D) -> H(Y)={h:8.4f}  "
                  f"{'PASS (snapped)' if h <= 1.2 else 'FAIL (flat)'}")
    print()
    print(f"  sqrt(D) snaps on ALL d/M at eps=0.15: {bool(ok)}")
    print()
    print("  Comparison at the SAME points, beta=8.0 held fixed:")
    for (D, M, eps, lab, h, cos) in rows:
        if abs(eps - 0.15) < 1e-9 and lab == "beta=8.0":
            print(f"  D={D:<6} M={M:<6} beta=8.0  -> H(Y)={h:8.4f}  "
                  f"{'PASS' if h <= 1.2 else 'FAIL (too flat)'}")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
