"""Diagnostic: what IS the exact sample-side contract of the Gram whitening route?

Establishes the achievable identity instead of guessing at a tolerance.
"""
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_V2 = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_V2, os.path.join(_V2, "henri_core")):
    if p not in sys.path:
        sys.path.insert(0, p)

from henri_core.henri_splitjepa_encoder import CenteredZCA  # noqa: E402


def probe(N, D, max_dense):
    torch.manual_seed(0)
    X = torch.randn(N, D)
    z = CenteredZCA(max_dense=max_dense).fit(X)
    Z = z.transform(X)
    Zc = Z - Z.mean(0)
    G = (Zc @ Zc.T) / (N - 1)
    target = torch.eye(N) - torch.full((N, N), 1.0 / N)
    dev = (G - target).abs().max().item()
    print(f"N={N:>3} D={D:>6} route={z.route_:>17} k={len(z._S) if z._S is not None else '-'}"
          f"  diag_mean={G.diag().mean():.6f} (target {(N-1)/N:.6f})"
          f"  offdiag_absmax={float((G - torch.diag(G.diag())).abs().max()):.6f}"
          f"  (target {1/N:.6f})"
          f"  max|G-proj|={dev:.3e}")
    return z


if __name__ == "__main__":
    for (n, d, md) in ((48, 256, 64), (24, 64, 8), (24, 64, 10_000),
                       (6, 65_536, 8192), (6, 6, 4), (200, 12, 8192)):
        try:
            z = probe(n, d, md)
            if z._S is not None:
                print(f"      singular values (last 3): {z._S[-3:].tolist()}")
        except Exception as exc:                                  # noqa: BLE001
            print(f"N={n} D={d} max_dense={md} -> {type(exc).__name__}: {exc}")

    # Is the projector exact for the gram route at large D, or is there drift?
    print("\n-- rank / drift scan at D=65536 (gram route) --")
    for n in (4, 6, 10, 32, 64):
        try:
            probe(n, 65_536, 8192)
        except Exception as exc:                                  # noqa: BLE001
            print(f"N={n} -> {type(exc).__name__}: {exc}")
