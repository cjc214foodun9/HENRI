"""DIRECTIVE 2 GATE -- Hopfield beta on REAL waves via NOISE-TOLERANCE.

DIRECTIVE: HENRI-ARCH-2026-CRITICAL-DIRECTIVE-V1, item 2
  "Ensure continuous waves terminate in henri_calibrated_action_head.py using Modern
   Hopfield retrieval at beta* = 26.10 (T* = 0.038316)"

TWO INSTRUMENT FAILURES THIS FILE EXISTS TO AVOID (both real, both hit here)
  1. BETA-BLIND METRIC. The first revision scored P@1 through `lexical_snap`, which
     does `sim.argmax(dim=-1)` and never reads self.beta. The curve came back FLAT at
     0.5703 for every beta in [1, 256] -- a property of the metric, not a result.
     The defect is structural: softmax is monotonic, so
         argmax(softmax(beta*sim)) == argmax(sim)   for EVERY beta > 0
     Any P@1-by-argmax egress metric is therefore BETA-INVARIANT BY CONSTRUCTION.
     That also explains the PRIOR receipt's `argmax_claim:
     UNIDENTIFIED__TIES_DOMINATE_THE_TOP_OF_THE_CURVE` -- it measured a quantity that
     cannot depend on beta.
  2. SATURATION. The second revision used soft clean_cos at ONE noise level. At
     beta=8 it already read 0.9959, leaving 0.004 of headroom, so "beta=26.1 is not
     strictly better" was undecidable rather than false. Reading a saturated metric
     is exactly the failure the prior receipt documented.

THE INSTRUMENT (physical, unsaturated)
  beta controls ATTRACTOR SHARPNESS. The meaningful, unbounded quantity is the
  NOISE TOLERANCE: the largest eps at which cleanup still recovers the engram
  (clean_cos >= 0.90), with noise scaled so ||noise|| = eps against ||engram|| = 1.
  Tolerance is measured on a grid, so it does not saturate; two betas differ by a
  whole grid step or they are indistinguishable at this resolution.

PRE-REGISTERED DECISION RULE (fixed before running)
  tolerance(beta=26.1) >  tolerance(beta=8.0)  -> ADOPT_26.10_MEASURED
  tolerance(beta=26.1) == tolerance(beta=8.0)  -> DIRECTIVE_MANDATED_NOT_MEASURED
  tolerance(beta=26.1) <  tolerance(beta=8.0)  -> KEEP_SEALED_8.0
  All tolerances identical across every beta   -> INSUFFICIENT_RESOLUTION (no verdict)
  Invariance control moving, or soft channel flat -> INVALID (harness unsound)
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from henri_vision_encoder import HENRIVisionEncoder      # noqa: E402
from hopfield_cleanup import ContinuousHopfieldCleanup   # noqa: E402

SEALED_BETA = 8.0            # henri_hopfield_egress.py:60 (CanonicalCodebookEgress default)
DOC_BETA = 26.1              # directive constant
DOC_T = 0.038316             # stated reciprocal (1/26.1 = 0.0383142)
ACCEPT_COS = 0.90            # cleanup counts as recovered above this


def make_pattern(g: int, kind: int, k: int, ln: int = 1):
    """Deterministic structured grids; (kind, k, ln) spans a large unique family."""
    grid = [[0] * g for _ in range(g)]
    if kind == 0:
        grid[k % g][k % g] = 1
    elif kind == 1:
        grid[k % g][(k * 3) % g] = 1
    elif kind == 2:
        for i in range(ln):
            grid[k % g][(k + i) % g] = 1
            grid[(k + i) % g][k % g] = 1
    elif kind == 3:
        grid[k % g][k % g] = 1
        grid[(k + 5) % g][(k + 7) % g] = 1
    else:
        for i in range(ln):
            grid[k % g][(k + i) % g] = 1
    return grid


def build_waves(enc, m: int, g: int, device, seed: int):
    """Encode REAL structured grids to flat unit waves [d_model]. Guarantees UNIQUENESS.

    FIXTURE DEFECT THIS FIXES (caught by the separability control, not by review):
      An earlier revision used `k % g` with g=16, so the grid family had only
      16 residues x 5 kinds = 80 distinct grids. Requesting M=256 then produced
      DUPLICATE engrams, and the control reported `max off-diagonal cos = 1.0000
      -> DEGENERATE`. That flag was CORRECT: two identical engrams cannot be
      separated by any temperature, so a beta sweep over that fixture is void.
      The generator now walks a Cartesian product of (kind, row, col, length) and
      asserts uniqueness before returning.

    API NOTE: `encode_context` is a WaveJEPA method (wave_jepa.py:63), NOT an encoder
    method; the encoder primitive is `encode_grid(grid) -> [d_model]`.
    """
    seen, out = set(), []
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(m * 40, generator=gen).tolist()
    with torch.no_grad():
        for raw in order:
            kind = raw % 5
            row = (raw // 5) % g
            col = (raw // (5 * g)) % g
            ln = 1 + (raw // (5 * g * g)) % 5
            grid = tuple(tuple(r) for r in make_pattern(g, kind, row * g + col, ln))
            if grid in seen:
                continue
            seen.add(grid)
            w = enc.encode_grid([list(r) for r in grid]).view(-1)
            out.append(F.normalize(w, p=2, dim=0).to(device))
            if len(out) >= m:
                break
    return out


@torch.no_grad()
def measure(cleanup, engrams, eps: float, seed: int):
    """clean_cos against the true engram, plus the beta-INVARIANT argmax P@1.

    noise is scaled so ||noise|| = eps (engrams are unit norm), applied to the query.
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    argmax_hits, clean_cos = 0, []
    for i, e in enumerate(engrams):
        q = e.clone()
        if eps > 0:
            n = torch.randn(q.numel(), generator=gen)
            n = F.normalize(n, p=2, dim=0).view_as(q) * eps
            q = F.normalize(q + n.to(q.dtype), p=2, dim=0)
        clean = cleanup.retrieve(q)
        if int(cleanup.lexical_snap(q, top_k=1)[0]) == i:
            argmax_hits += 1
        clean_cos.append(float(F.cosine_similarity(
            clean.view(-1).float(), e.view(-1).float(), dim=0)))
    n_ = max(1, len(engrams))
    return dict(p1_argmax=argmax_hits / n_, clean_cos=sum(clean_cos) / n_)


def noise_tolerance(cleanup, engrams, grid, seed):
    """Largest eps where clean_cos >= ACCEPT_COS, plus the full curve."""
    curve = {}
    for j, e in enumerate(grid):
        curve[e] = measure(cleanup, engrams, e, seed + j + 1)["clean_cos"]
    ok = [e for e in grid if curve[e] >= ACCEPT_COS]
    return (max(ok) if ok else None), curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--nb", type=int, default=64)
    ap.add_argument("--grid", type=int, default=16)
    ap.add_argument("--m", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 90)
    print("DIRECTIVE 2 GATE -- HOPFIELD BETA, NOISE-TOLERANCE INSTRUMENT (real waves)")
    print("=" * 90)
    print(f"  device {device}  d={args.d}  nb={args.nb}  M={args.m} engrams")
    print(f"  constants: SEALED={SEALED_BETA}  DIRECTIVE={DOC_BETA} (T*={DOC_T})  "
          f"module_default=sqrt(dim)={math.sqrt(args.d):.2f}")
    print(f"  noise ||n|| = eps against ||engram|| = 1 ; recovered if clean_cos >= {ACCEPT_COS}")

    enc = HENRIVisionEncoder(d_model=args.d, k_blocks=args.nb, device=str(device),
                            spatial_basis_kind="incommensurate", bg_mask=True,
                            fused_superpose=True, parity_scipy=True)
    waves = build_waves(enc, args.m, args.grid, device, args.seed)
    # SEPARABILITY CONTROL. NOTE the dimension: build_waves may return FEWER than
    # `args.m` waves if the unique-grid family is exhausted (a request for 256 from a
    # 174-member family returns 174). An earlier revision built the identity with
    # args.m here and crashed with "size of tensor a (174) must match b (256)" --
    # my bug, caught by the run. Always size from the ACTUAL engram count.
    m_eff = len(waves)
    Wm = torch.stack(waves).float()
    S = (Wm @ Wm.T).clone()
    S.fill_diagonal_(-1e9)
    max_off = float(S.max())
    min_off = float((S + torch.eye(m_eff, device=S.device) * 2.0).min())
    sep_ok = max_off < 0.95
    if m_eff < args.m:
        print(f"  NOTE: requested M={args.m}, unique family yielded M={m_eff}.")
    gen = torch.Generator(device="cpu").manual_seed(999)
    rnd = [F.normalize(torch.randn(args.d, generator=gen), p=2, dim=0).to(device)
           for _ in range(m_eff)]
    print(f"  encoded {m_eff} real waves {tuple(waves[0].shape)}; "
          f"max off-diagonal cos {max_off:.4f} "
          f"-> {'SEPARABLE' if sep_ok else 'DEGENERATE'}   [M/d = {m_eff / args.d:.3f}]")
    print()

    grid = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    betas = sorted({1.0, 2.0, 4.0, SEALED_BETA, 12.0, 16.0, 20.0, DOC_BETA, 32.0,
                    64.0, float(round(math.sqrt(args.d), 2)), 128.0, 256.0})

    print("  REAL WAVES -- clean_cos by noise level")
    print(f"  {'beta':>8} " + " ".join(f"{('e'+format(e,'.1f')):>7}" for e in grid)
          + f" {'tol':>6}")
    rows = []
    for b in betas:
        c = ContinuousHopfieldCleanup(dim=args.d, beta=b).to(device)
        c.store_engrams(torch.stack(waves))
        tol, curve = noise_tolerance(c, waves, grid, args.seed)
        p1 = measure(c, waves, 0.0, args.seed + 99)["p1_argmax"]
        rows.append(dict(beta=b, tolerance=tol, curve=curve, p1_argmax=p1))
        print(f"  {b:>8.2f} " + " ".join(f"{curve[e]:>7.4f}" for e in grid)
              + f" {str(tol) if tol is not None else '--':>6}")

    print()
    print("  SYNTHETIC CONTROL -- random codebook (same instrument)")
    print(f"  {'beta':>8} " + " ".join(f"{('e'+format(e,'.1f')):>7}" for e in grid)
          + f" {'tol':>6}")
    ctrl = []
    for b in betas:
        c = ContinuousHopfieldCleanup(dim=args.d, beta=b).to(device)
        c.store_engrams(torch.stack(rnd))
        tol, curve = noise_tolerance(c, rnd, grid, args.seed)
        ctrl.append(dict(beta=b, tolerance=tol, curve=curve))
        print(f"  {b:>8.2f} " + " ".join(f"{curve[e]:>7.4f}" for e in grid)
              + f" {str(tol) if tol is not None else '--':>6}")

    def tol_of(table, b):
        for r in table:
            if abs(r["beta"] - b) < 1e-6:
                return r["tolerance"]
        return None

    t8, t26 = tol_of(rows, SEALED_BETA), tol_of(rows, DOC_BETA)
    # CONTROLS
    p1_vals = [r["p1_argmax"] for r in rows]
    p1_spread = max(p1_vals) - min(p1_vals)                 # MUST be ~0 (monotonicity)
    soft0 = [r["curve"][0.0] for r in rows]
    soft_spread = max(soft0) - min(soft0)                   # MUST move (beta-sensitive)
    tols = sorted({r["tolerance"] for r in rows if r["tolerance"] is not None})
    discriminates = len(tols) > 1

    print()
    print("=" * 90)
    print("VERDICT")
    print("=" * 90)
    print(f"  INVARIANCE CONTROL argmax P@1 spread over betas = {p1_spread:.6f}  "
          f"{'OK (beta-invariant as monotonicity requires)' if p1_spread <= 1e-9 else 'HARNESS SUSPECT'}")
    print(f"  SENSITIVITY CONTROL soft clean_cos(e=0) spread  = {soft_spread:.6f}  "
          f"{'OK' if soft_spread > 1e-6 else 'BETA-BLIND'}")
    print(f"  RESOLUTION CONTROL distinct tolerances observed = {tols}  "
          f"{'OK' if discriminates else 'SATURATED -- fixture cannot resolve beta'}")
    print(f"  REAL noise tolerance  beta={SEALED_BETA} -> {t8}    beta={DOC_BETA} -> {t26}")

    reasons, verdict = [], None
    if not sep_ok:
        reasons.append(f"real fixture DEGENERATE: max off-diagonal cos {max_off:.4f} >= 0.95")
    if p1_spread > 1e-9:
        reasons.append("argmax P@1 moved with beta -- violates monotonicity, unsound")
    if soft_spread <= 1e-6:
        reasons.append("no beta-sensitive channel moved -- beta-blind")
    if not discriminates:
        reasons.append("all tolerances identical -- fixture SATURATES, no verdict possible")

    if reasons:
        verdict = "INVALID"
        for r in reasons:
            print(f"    REASON: {r}")
    elif t26 > t8:
        verdict = "ADOPT_26.10_MEASURED"
        print(f"  -> beta=26.10 has strictly greater noise tolerance ({t26} > {t8}).")
        print("     Adopt with THIS receipt as provenance.")
    elif t26 == t8:
        verdict = "DIRECTIVE_MANDATED_NOT_MEASURED"
        print(f"  -> tolerance is EQUAL ({t26} == {t8}) at this resolution. beta=26.10 is")
        print("     NOT measurably better. Adopting it is a DIRECTIVE choice, not a result.")
    else:
        verdict = "KEEP_SEALED_8.0"
        print(f"  -> beta=26.10 is WORSE ({t26} < {t8}). Keep {SEALED_BETA}.")
    print()
    print(f"EGRESS_BETA_GATE: {verdict}")

    if args.out:
        rec = dict(measured_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
                   schema="henri.egress.beta-gate.v2.noise-tolerance", evidence_class="OBSERVED",
                   python=platform.python_version(), torch=torch.__version__,
                   device=str(device), d_model=args.d, num_blocks=args.nb, M=args.m,
                   accept_cos=ACCEPT_COS, noise_grid=grid,
                   fixture="HENRIVisionEncoder real structured grids (NOT synthetic)",
                   constants=dict(sealed=SEALED_BETA, directive=DOC_BETA, doc_T=DOC_T,
                                  module_default="sqrt(dim)"),
                   real=rows, synthetic_control=ctrl, verdict=verdict,
                   controls=dict(argmax_p1_spread=p1_spread, soft_clean_spread=soft_spread,
                                 distinct_tolerances=tols, discriminates=discriminates),
                   instrument_note="noise tolerance replaces saturated clean_cos; the prior "
                                   "receipt's argmax metric is beta-invariant by monotonicity")
        out = args.out if os.path.isabs(args.out) else os.path.join(_ROOT, args.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(rec, fh, indent=2)
        print(f"  receipt: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
