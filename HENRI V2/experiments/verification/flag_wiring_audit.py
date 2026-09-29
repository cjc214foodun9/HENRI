"""Flag-wiring audit + clean 5-config measurement + identity proof.

WHY THIS EXISTS
===============
Three defects were found in already-committed code:
  1. `segment_grid(want_exterior=...)` accepted the flag and DROPPED it. The
     call to compute_parity_contour did not forward it, so the optimisation
     measured 1.00x -- a no-op that reads as a win.
  2. `_parity_contour_fast` was dead code. `grep -n 'fast=True'` matched only
     its own docstring; nothing ever called it.
  3. `parity_fast` was not a parameter at all, so the fast path was unreachable
     from the encoder.

This file is the regression guard for that class: it asserts each flag REACHES
A CONSUMER (by AST, not by reading the docs), then measures each flag
independently, then proves the wave output is unchanged.

The AST check is the important part. A flag that is accepted and ignored is
more dangerous than a missing flag, because it makes a slow path look tuned.
"""
import ast
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
sys.path.insert(0, _ROOT)

import numpy as np  # noqa: E402

ENCODER = os.path.join(_ROOT, "henri_vision_encoder.py")
SEGMENTER = os.path.join(_ROOT, "connected_component_segmenter.py")

GRIDS = {
    "4x4": [[0, 0, 0, 0], [0, 1, 1, 0], [0, 1, 2, 0], [0, 0, 0, 0]],
    "8x8": [[(i * 3 + j) % 4 for j in range(8)] for i in range(8)],
    "16x16": [[(i * 3 + j) % 10 for j in range(16)] for i in range(16)],
    "30x30": [[(i + j) % 10 for j in range(30)] for i in range(30)],
}


# --------------------------------------------------------------------------- #
# 1. AST wiring audit
# --------------------------------------------------------------------------- #

def kwargs_of_call(call: ast.Call) -> set:
    return {k.arg for k in call.keywords if k.arg}


def audit(path: str, fn_name: str, flags: list) -> dict:
    """For each flag in `flags`, is it USED (read) inside fn_name, and does it
    appear as a keyword in any call made inside fn_name?"""
    src = open(path, "r", encoding="utf-8", errors="ignore").read()
    tree = ast.parse(src, filename=path)
    fn = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == fn_name:
            fn = node
            break
    if fn is None:
        return {"error": f"{fn_name} not found in {path}"}
    reads, forwarded = {}, {}
    params = {a.arg for a in fn.args.args} | {a.arg for a in fn.args.kwonlyargs}
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and node.id in flags:
            reads.setdefault(node.id, 0)
            reads[node.id] += 1
        if isinstance(node, ast.Call):
            for k in kwargs_of_call(node):
                if k in flags:
                    forwarded.setdefault(k, 0)
                    forwarded[k] += 1
    out = {}
    for f in flags:
        out[f] = {"in_signature": f in params,
                  "read_count": reads.get(f, 0),
                  "forwarded_count": forwarded.get(f, 0)}
    return out


def main() -> int:
    print("=" * 80)
    print("FLAG WIRING AUDIT (AST -- does each flag reach a consumer?)")
    print("=" * 80)
    problems = []

    print("\n[henri_vision_encoder] HENRIVisionEncoder.__init__")
    enc_flags = ["vectorized_accum", "parity_dedup", "parity_fast", "bg_mask"]
    for f, d in audit(ENCODER, "__init__", enc_flags).items():
        ok = d["read_count"] > 0 or d["forwarded_count"] > 0
        print(f"  {'OK ' if ok else 'DEAD'} {f:18s} in_sig={d['in_signature']} "
              f"reads={d['read_count']} forwarded={d['forwarded_count']}")
        if d["in_signature"] and not ok:
            problems.append(f"encoder.__init__/{f} accepted but never used")

    print("\n[connected_component_segmenter] ConnectedComponentSegmenter.segment_grid")
    seg_flags = ["want_exterior", "fast"]
    seg = audit(SEGMENTER, "segment_grid", seg_flags)
    for f, d in seg.items():
        fwd = d["forwarded_count"]
        ok = fwd > 0
        print(f"  {'OK ' if ok else 'DEAD'} {f:18s} in_sig={d['in_signature']} "
              f"reads={d['read_count']} forwarded_to_call={fwd}")
        if d["in_signature"] and not ok:
            problems.append(f"segment_grid/{f} accepted but NOT forwarded to a call")

    print("\n[connected_component_segmenter] compute_parity_contour")
    for f, d in audit(SEGMENTER, "compute_parity_contour",
                      ["fast", "want_exterior"]).items():
        print(f"  {'OK ' if d['in_signature'] else 'MISSING'} {f:18s} "
              f"in_sig={d['in_signature']} reads={d['read_count']}")

    # is the fast path reachable at all from the encoder?
    esrc = open(ENCODER, encoding="utf-8", errors="ignore").read()
    reachable = "fast=self.parity_fast" in esrc
    print(f"\n  fast path reachable from encoder: {reachable}")
    if not reachable:
        problems.append("parity_fast never forwarded from encode_grid")

    print(f"\nWIRING_PROBLEMS {len(problems)}")
    for p in problems:
        print("   -", p)

    # ------------------------------------------------------------------ #
    # 2. identity: every config must give the SAME wave
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("IDENTITY: all configs vs baseline (max abs wave diff)")
    print("=" * 80)
    from henri_vision_encoder import HENRIVisionEncoder  # noqa: E402

    CONFIGS = {
        "A baseline": dict(),
        "B dedup": dict(parity_dedup=True),
        "C fast": dict(parity_fast=True),
        "D vect": dict(vectorized_accum=True),
        "E all": dict(parity_dedup=True, parity_fast=True, vectorized_accum=True),
    }

    def mk(**kw):
        return HENRIVisionEncoder(d_model=65536, k_blocks=8192, device="cpu",
                                  spatial_basis_kind="incommensurate",
                                  bg_mask=True, **kw)

    worst = 0.0
    for gname, g in GRIDS.items():
        ref = None
        line = f"  {gname:7s}"
        for label, kw in CONFIGS.items():
            w = mk(**kw).encode_grid(g)
            if ref is None:
                ref = w
            d = float((w - ref).abs().max())
            worst = max(worst, d)
            line += f" {label}={d:.1e}"
        print(line)
    print(f"\n  worst diff across all configs/grids: {worst:.3e}")
    print(f"  WAVE_IDENTICAL {worst < 1e-6}")
    if worst >= 1e-6:
        problems.append(f"wave differs: {worst}")

    # ------------------------------------------------------------------ #
    # 3. per-flag timing (each flag measured independently)
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 80)
    print("PER-FLAG TIMING (CPU; encode_grid segments on CPU)")
    print("=" * 80)
    hdr = f"  {'grid':7s}" + "".join(f"{k:>12s}" for k in CONFIGS)
    print(hdr)
    for gname, g in GRIDS.items():
        row = f"  {gname:7s}"
        base = None
        times = {}
        for label, kw in CONFIGS.items():
            e = mk(**kw)
            for _ in range(2):
                e.encode_grid(g)
            reps = 5
            t0 = time.perf_counter()
            for _ in range(reps):
                e.encode_grid(g)
            us = (time.perf_counter() - t0) / reps * 1e6
            times[label] = us
            if base is None:
                base = us
            row += f"{us:12.1f}"
        print(row)
        sp = "  speedup vs A: " + " ".join(
            f"{label.split()[0]}={base/times[label]:.2f}x" for label in CONFIGS)
        print(sp)

    print("\n" + "=" * 80)
    if problems:
        print(f"RESULT: FAIL ({len(problems)})")
        for p in problems:
            print("   -", p)
        return 1
    print("RESULT: PASS")
    print("WIRING_OK True")
    return 0


if __name__ == "__main__":
    sys.exit(main())
