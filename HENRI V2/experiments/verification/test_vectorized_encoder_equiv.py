"""Equivalence test: vectorized accumulator == legacy per-cell loop.

WHY THIS EXISTS
===============
`henri_vision_encoder.HENRIVisionEncoder(vectorized_accum=True)` replaces a
per-cell Python loop with a batched expression. Both compute the same linear
superposition, so they must agree to floating-point reassociation error. This
test pins that, and it is cited by the `_superpose_vectorized` docstring, so it
must exist and must be runnable.

Measured motivation (RTX 5090, stage1_latency_receipt.json):
    legacy 4x4: 402 kernel launches/call, 79.2% of wall in Python gaps

The test checks FOUR configurations, because the legacy path branches on
`bg_mask` and the basis kind changes the y-ramp:
    bg_mask False / True  x  spatial_basis_kind default / incommensurate

It also asserts the fail-closed behaviour is preserved: a fully-background grid
under `bg_mask=True` must still raise, in BOTH paths.
"""
import os
import sys

# DEFECT FIXED 2026-09-29: this file lives at HENRI V2/experiments/verification/,
# so the repo root is THREE levels up. Two dirname()s gave .../HENRI V2/experiments
# and the import raised ModuleNotFoundError.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))        # .../HENRI V2
sys.path.insert(0, _ROOT)

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

D = 65536
K = 8192
TOL = 1e-4          # absolute, on a unit-normalised vector

GRIDS = {
    "tiny_4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
    "mixed_5x5": [[1, 1, 0, 0, 0], [1, 2, 2, 0, 0], [0, 2, 3, 3, 0],
                  [0, 0, 3, 0, 0], [0, 0, 0, 0, 4]],
    "no_bg_3x6": [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12], [13, 14, 15, 1, 2, 3]],
    "wide_2x17": [[(i * 7 + j * 3) % 16 for j in range(17)] for i in range(2)],
    "tall_17x2": [[(i * 5 + j * 11) % 16 for j in range(2)] for i in range(17)],
}


def enc(bg: bool, kind: str, vectorized: bool):
    return HENRIVisionEncoder(d_model=D, k_blocks=K, device="cpu",
                              spatial_basis_kind=kind, bg_mask=bg,
                              vectorized_accum=vectorized)


def main() -> int:
    print("=" * 76)
    print("VECTORIZED ACCUMULATOR EQUIVALENCE  (D=%d, CPU)" % D)
    print("=" * 76)

    failures = []
    rows = []
    for kind in ("default", "incommensurate"):
        for bg in (False, True):
            legacy = enc(bg, kind, False)
            vect = enc(bg, kind, True)
            for gname, g in GRIDS.items():
                try:
                    a = legacy.encode_grid(g)
                    ea = None
                except Exception as exc:                      # noqa: BLE001
                    a, ea = None, exc
                try:
                    b = vect.encode_grid(g)
                    eb = None
                except Exception as exc:                      # noqa: BLE001
                    b, eb = None, exc

                # legacy fail-closed contract must be preserved exactly
                if ea is not None or eb is not None:
                    same = (ea is not None and eb is not None
                            and type(ea) is type(eb))
                    tag = "RAISE-MATCH" if same else "RAISE-MISMATCH"
                    if not same:
                        failures.append((kind, bg, gname, f"{ea} vs {eb}"))
                    rows.append((kind, bg, gname, tag, "", ""))
                    continue

                diff = float((a - b).abs().max())
                rel = diff / max(float(a.abs().max()), 1e-12)
                ok = diff < TOL
                if not ok:
                    failures.append((kind, bg, gname, diff))
                rows.append((kind, bg, gname, "PASS" if ok else "FAIL",
                             f"{diff:.3e}", f"{rel:.3e}"))

    print(f"\n{'kind':16s} {'bg':6s} {'grid':11s} {'verdict':14s} "
          f"{'max_abs_diff':>12s} {'rel':>10s}")
    for kind, bg, gname, tag, d, r in rows:
        print(f"{kind:16s} {str(bg):6s} {gname:11s} {tag:14s} {d:>12s} {r:>10s}")

    # ---- fail-closed preserved: all-background grid under bg_mask=True -----
    print("\n" + "=" * 76)
    print("FAIL-CLOSED CONTRACT (all-background grid, bg_mask=True)")
    print("=" * 76)
    allbg = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for vectorized in (False, True):
        name = "vectorized" if vectorized else "legacy"
        e = enc(True, "default", vectorized)
        try:
            e.encode_grid(allbg)
            print(f"  {name:11s} DID NOT RAISE  <-- regression")
            failures.append(("fail-closed", True, name, "no raise"))
        except ValueError as exc:
            print(f"  {name:11s} ValueError OK  ({str(exc)[:52]})")
        except Exception as exc:                          # noqa: BLE001
            print(f"  {name:11s} WRONG TYPE {type(exc).__name__}")
            failures.append(("fail-closed", True, name, type(exc).__name__))

    # ---- normalisation invariant on the new path ---------------------------
    print("\n" + "=" * 76)
    print("UNIT-NORM INVARIANT (vectorized path)")
    print("=" * 76)
    v = enc(True, "incommensurate", True)
    worst = 0.0
    for gname, g in GRIDS.items():
        w = v.encode_grid(g)
        n = float(w.norm())
        worst = max(worst, abs(n - 1.0))
        print(f"  {gname:11s} norm {n:.8f}  finite {bool(torch.isfinite(w).all())}")
    print(f"  max |norm - 1| = {worst:.3e}  (gate 1e-5: "
          f"{'PASS' if worst < 1e-5 else 'FAIL'})")
    if worst >= 1e-5:
        failures.append(("norm", True, "vectorized", worst))

    # ---- chunk-size invariance (row_chunk must not change the answer) -------
    print("\n" + "=" * 76)
    print("ROW-CHUNK INVARIANCE (memory bound must not change the result)")
    print("=" * 76)
    v2 = enc(True, "incommensurate", True)
    ref = None
    for chunk in (1, 2, 7, 32, 64):
        g = GRIDS["mixed_5x5"]
        gt = torch.tensor(g, dtype=torch.long)
        gc = torch.clamp(gt, 0, 15)
        pm = torch.ones(gt.shape, dtype=torch.float32)
        acc, contrib = v2._superpose_vectorized(gc, pm, gt.shape[0], gt.shape[1],
                                                row_chunk=chunk)
        w = torch.cat([acc.real, acc.imag], dim=-1)
        w = Fn.normalize(w, p=2, dim=-1)
        if ref is None:
            ref = w
            print(f"  chunk {chunk:3d}  reference  contributed={contrib}")
        else:
            d = float((w - ref).abs().max())
            ok = d < TOL
            print(f"  chunk {chunk:3d}  max_abs_diff {d:.3e}  "
                  f"contributed={contrib}  {'PASS' if ok else 'FAIL'}")
            if not ok:
                failures.append(("chunk", chunk, "invariance", d))

    print("\n" + "=" * 76)
    if failures:
        print(f"RESULT: FAIL  ({len(failures)} problem(s))")
        for f in failures[:10]:
            print("   ", f)
        print("EQUIVALENCE_HOLDS False")
        return 1
    print("RESULT: PASS")
    print("EQUIVALENCE_HOLDS True")
    return 0


if __name__ == "__main__":
    sys.exit(main())