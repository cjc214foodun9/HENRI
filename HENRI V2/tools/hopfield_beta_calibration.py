"""GAP 3 BETA CALIBRATION - resolve the constant conflict by MEASUREMENT.

THE CONFLICT
============
The supplied blueprint demands a "Continuous Modern Hopfield Cleanup Layer with
temperature T* = 0.038316 (beta* = 26.10)". The repository's live, SEALED Hopfield
contract is a different number: henri_hopfield_egress.py constructs
ContinuousHopfieldCleanup(dim=dim, beta=8.0). And hopfield_cleanup.py's OWN default
is beta = sqrt(dim) -- a third value. Three claims, no measurement.

ARITHMETIC FIRST
================
  1 / 26.10   = 0.0383141762...   doc states T* = 0.038316  -> off by 1.8e-6
  1 / 0.038316= 26.09875...       doc states beta* = 26.10  -> internally consistent
So the doc's two constants agree with each other to ~5 sig figs but match neither the
sealed 8.0 nor the module default.

METHOD: measure P@1 of the live ContinuousHopfieldCleanup across a beta sweep, with
the SAME codebook and the SAME query noise for every beta, at three codebook sizes.

TWO GUARDS (both required, and the second was added after the first failed)
==========================================================================
  SATURATION GUARD : a size is DISCARDED if every beta gives the same mean, because an
                     argmax there is a tie artifact.
  UNIQUENESS GUARD : a size is DISCARDED unless the argmax is UNIQUE (best mean
                     strictly exceeds second-best by > 1e-9). The first sweep passed
                     the saturation guard while many betas tied at 1.0000, so its
                     printed argmax was still meaningless.
Both are reported, and a size is only used for selection if it passes BOTH.

CONTROLS: ceiling (noiseless exact query must retrieve P@1 = 1.0) and floor (a
collapsed readout shows up as P@1 -> 1/M).

HONEST LIMITS: synthetic random codebook, Gaussian query noise, D=1024. This measures
the CLEANUP OPERATOR's noise tolerance, NOT recall over a real embedding distribution.
No egress/action/benchmark claim. The result is REPORTED, never adopted as a new seal.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

C = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2"
sys.path.insert(0, C)

from hopfield_cleanup import ContinuousHopfieldCleanup  # noqa: E402

D = 1024
BETAS = [1.0, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0, 26.1, 32.0, 48.0, 64.0, 128.0, 256.0]
MS = [512, 1024, 2048]                      # sigma^2 = M/D = 0.5, 1.0, 2.0
NOISES = [0.0, 0.5, 1.0, 1.5, 2.0]  # widened so sizes leave the ceiling
SEED = 20260927
REC = r"C:/Users/chan/henri-worktrees/zone-a-selfplay/HENRI V2/experiments/verification/hopfield_beta_calibration.json"
UNIQUE_EPS = 1e-9
SEALED_BETA = 8.0
DOC_BETA = 26.1


def make_codebook(m: int, g: torch.Generator) -> torch.Tensor:
    v = torch.randn(m, D, generator=g)
    return v / (torch.linalg.vector_norm(v, dim=-1, keepdim=True) + 1e-12)


def p_at_1(cleanup, book: torch.Tensor, noise: float, n_queries: int,
           g: torch.Generator) -> float:
    idx = torch.randint(0, book.shape[0], (n_queries,), generator=g)
    q = book[idx].clone()
    if noise > 0:
        q = q + noise * torch.randn(q.shape, generator=g) / math.sqrt(D)
        q = q / (torch.linalg.vector_norm(q, dim=-1, keepdim=True) + 1e-12)
    with torch.no_grad():
        out = cleanup.retrieve(q)
    if isinstance(out, tuple):
        out = out[0]
    if out.dim() == 3:
        out = out[:, -1, :]
    got = (out @ book.t()).argmax(dim=-1)
    return float((got == idx).float().mean())


def resolve_receipt_path(out=None):
    """Resolve the receipt path with STRICT precedence.

    ORDER: --out  >  HENRI_RECEIPT_DIR  >  module default (the COMMITTED path).

    WHY (hazard proved live 2026-09-27): a backgrounded run of this tool targeted the
    committed, ledger-cited receipt path simply because the default IS that path. It
    missed clobbering committed evidence only because it crashed BEFORE its write --
    accidental protection, not a guard. An ad-hoc or backgrounded run must be able to
    redirect output WITHOUT editing the tool.

    A MALFORMED override RAISES. Falling back to the committed default on a bad
    override is the exact failure this function exists to prevent.
    """
    cand = out if out is not None else os.environ.get("HENRI_RECEIPT_DIR")
    if cand is None:
        return REC
    if not str(cand).strip():
        raise ValueError("empty receipt path override: refusing to fall back to the "
                         "committed default")
    p = os.path.abspath(str(cand))
    if p == os.path.abspath(REC):
        return p
    if os.path.isdir(p) or str(cand).endswith(("/", chr(92))):
        p = os.path.join(p, os.path.basename(REC))
    if not os.path.dirname(p):
        raise ValueError("receipt override has no parent directory: %r" % (cand,))
    return p


def main(out=None):
    rec_path = resolve_receipt_path(out)
    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("=" * 78)
    say("ARITHMETIC (before any model call)")
    say("=" * 78)
    say("  1/26.10     = %.9f   doc T* = 0.038316   (delta %.2e)"
        % (1.0 / 26.10, abs(1.0 / 26.10 - 0.038316)))
    say("  1/0.038316  = %.6f        doc beta* = 26.10  (internally consistent)"
        % (1.0 / 0.038316))
    say("  sqrt(D)     = %.1f          hopfield_cleanup.py DEFAULT beta" % math.sqrt(D))
    say("  sealed      = %.1f            henri_hopfield_egress.py constructor" % SEALED_BETA)

    # ---- ceiling control
    g = torch.Generator().manual_seed(SEED)
    book = make_codebook(MS[0], g)
    ceiling_ok = True
    say("")
    say("CEILING CONTROL: noiseless exact query must retrieve P@1 = 1.0")
    for b in (1.0, 4.0, SEALED_BETA, DOC_BETA, 256.0):
        c = ContinuousHopfieldCleanup(dim=D, beta=b)
        c.store_engrams(book)
        acc = p_at_1(c, book, 0.0, 256, torch.Generator().manual_seed(SEED + 1))
        if b >= SEALED_BETA:
            ceiling_ok &= abs(acc - 1.0) < 1e-9  # sharpness requires high beta
        say("  beta %-7.1f noiseless P@1 = %.4f" % (b, acc))
    say("  CEILING %s (asserted at beta >= %.1f; a low-beta noiseless miss is"
        " the measured SHARPNESS phenomenon, not a harness defect)"
        % ("PASS" if ceiling_ok else "FAIL -> harness broken", SEALED_BETA))

    # ---- sweep
    results = {}
    say("")
    say("=" * 78)
    say("BETA SWEEP (identical codebook/noise/seed for every beta)")
    say("=" * 78)
    for m in MS:
        g = torch.Generator().manual_seed(SEED)
        book = make_codebook(m, g)
        sigma2 = m / D
        say("")
        say("  codebook M=%-6d crosstalk sigma^2 = M/D = %.4f" % (m, sigma2))
        say("    %-8s" % "beta" + "".join("%11s" % ("n=%.1f" % n) for n in NOISES)
            + "%11s" % "mean")
        for b in BETAS:
            c = ContinuousHopfieldCleanup(dim=D, beta=b)
            c.store_engrams(book)
            row = [p_at_1(c, book, n, 256, torch.Generator().manual_seed(SEED + 7))
                   for n in NOISES]
            mean = sum(row) / len(row)
            results["M%d_beta%g" % (m, b)] = {
                "beta": b, "M": m, "sigma2": sigma2,
                "by_noise": {str(n): v for n, v in zip(NOISES, row)}, "mean": mean}
            say("    %-8g" % b + "".join("%11.4f" % v for v in row) + "%11.4f" % mean)

    # ---- guards
    say("")
    say("=" * 78)
    say("GUARDS (a size is usable only if it passes BOTH)")
    say("=" * 78)
    guard = {}
    usable = []
    for m in MS:
        rows = sorted(((v["mean"], v["beta"]) for v in results.values() if v["M"] == m),
                      reverse=True)
        means = [r[0] for r in rows]
        distinct = len({round(x, 6) for x in means})
        best, best_beta = rows[0]
        second = rows[1][0] if len(rows) > 1 else -1.0
        clean = distinct > 1
        unique = (best - second) > UNIQUE_EPS
        guard["M%d" % m] = {"distinct_means": distinct, "not_saturated": bool(clean),
                            "best_mean": best, "best_beta": best_beta,
                            "second_mean": second, "unique_argmax": bool(unique),
                            "usable": bool(clean and unique)}
        say("  M=%-6d distinct means %-3d %-14s | argmax beta %-6g mean %.4f | "
            "second %.4f | unique %-5s | usable %s"
            % (m, distinct, "DISCRIMINATING" if clean else "SATURATED",
               best_beta, best, second, unique, guard["M%d" % m]["usable"]))
        if guard["M%d" % m]["usable"]:
            usable.append(m)

    # ---- selection (usable sizes only)
    sel = {}
    for m in usable:
        rows = sorted(((v["mean"], v["beta"]) for v in results.values() if v["M"] == m),
                      reverse=True)
        best_mean, best_beta = rows[0]
        seal = next(v["mean"] for v in results.values()
                    if v["M"] == m and abs(v["beta"] - SEALED_BETA) < 1e-9)
        doc = next(v["mean"] for v in results.values()
                   if v["M"] == m and abs(v["beta"] - DOC_BETA) < 1e-9)
        sel["M%d" % m] = {"best_beta": best_beta, "best_mean": best_mean,
                          "beta8_mean": seal, "beta26.1_mean": doc,
                          "seal_minus_doc": seal - doc}

    # ---- verdict
    # ---- ROBUST PAIRWISE CLAIM (independent of whether an argmax is identifiable)
    # When several betas TIE at the top, the argmax is a tie artifact and must NOT be
    # published. What IS robust is the DOCUMENT'S OWN CONSTANT versus the SEAL:
    # does beta*=26.1 ever beat the sealed beta=8.0? That comparison needs no argmax.
    pair = {}
    for m in MS:
        b8 = next(v["mean"] for v in results.values()
                  if v["M"] == m and abs(v["beta"] - SEALED_BETA) < 1e-9)
        b26 = next(v["mean"] for v in results.values()
                   if v["M"] == m and abs(v["beta"] - DOC_BETA) < 1e-9)
        pair["M%d" % m] = {"beta8_mean": b8, "beta26.1_mean": b26, "delta": b26 - b8}
    n_better = sum(1 for v in pair.values() if v["delta"] > 1e-9)
    n_tie = sum(1 for v in pair.values() if abs(v["delta"]) <= 1e-9)
    n_worse = sum(1 for v in pair.values() if v["delta"] < -1e-9)

    if not sel:
        argmax_claim = "UNIDENTIFIED__TIES_DOMINATE_THE_TOP_OF_THE_CURVE"
    elif len({sel[k]["best_beta"] for k in sel}) == 1 and abs(
            next(iter(sel.values()))["best_beta"] - SEALED_BETA) < 1e-9:
        argmax_claim = "SEALED_BETA_IS_ARGMAX"
    else:
        argmax_claim = "ARGMAX_DIFFERS_FROM_SEALED_AND_DOC"

    if n_better == 0 and n_tie + n_worse == len(pair):
        verdict = "DOC_BETA_NOT_BETTER_THAN_SEALED__NEVER_STRICTLY_BETTER"
    else:
        verdict = "DOC_BETA_BETTER_AT_SOME_CODEBOOK_SIZE"

    payload = {
        "schema": "henri.egress.hopfield-beta-calibration.v1",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "evidence_class": "OBSERVED",
        "dim": D, "betas": BETAS, "codebook_sizes": MS, "noise_levels": NOISES,
        "seed": SEED, "unique_eps": UNIQUE_EPS,
        "ceiling_control_pass": bool(ceiling_ok),
        "guard": guard, "usable_codebook_sizes": usable,
        "selection": sel, "verdict": verdict,
        "argmax_claim": argmax_claim,
        "pairwise_doc_vs_sealed": pair,
        "pairwise_counts": {"doc_better": n_better, "tie": n_tie, "doc_worse": n_worse},
        "constants_in_conflict": {
            "doc_temperature_T_star": 0.038316,
            "doc_beta_star": DOC_BETA,
            "doc_internal_check_1_over_beta": 1.0 / 26.10,
            "sealed_egress_beta": SEALED_BETA,
            "module_default_beta": "sqrt(dim)",
            "all_three_differ": True},
        "results": results,
        "honest_limit": ("Synthetic random codebook, Gaussian query noise, D=%d. Measures "
                         "the CLEANUP OPERATOR noise tolerance; NOT recall over a real "
                         "embedding distribution. No egress/action/benchmark claim." % D),
        "next_step": ("Changing the sealed constant requires replicating the M=10,000 / "
                      "D=65,536 capacity contract and passing the receipt-pinned promotion "
                      "gate. This sweep deliberately does not."),
        "rejected_claims": {
            "phantom_argmax_4_4_8": ("not produced by this harness. An earlier "
                                  "reported curve (argmax 4/4/8, sealed "
                                  "0.9625/0.8875/0.8750) does not appear in any "
                                  "receipt this tool wrote; it also reported "
                                  "noiseless P@1=1.0000 at beta=1.0 where this "
                                  "tool measures ~0.57. Treat as FABRICATED."),
        },
        "writer_note": ("The receipt is written BEFORE the narration, so a print-time bug "
                        "can no longer leave a stale artifact on disk."),
    }

    # WRITE FIRST
    os.makedirs(os.path.dirname(rec_path), exist_ok=True)
    with open(rec_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    say("")
    say("=" * 78)
    say("VERDICT")
    say("=" * 78)
    say("  usable codebook sizes: %s" % usable)
    for k in sorted(sel):
        s = sel[k]
        say("  %-7s argmax beta %-6g (mean %.4f) | sealed 8.0 %.4f | doc 26.1 %.4f "
            "| seal-doc %+.4f"
            % (k, s["best_beta"], s["best_mean"], s["beta8_mean"],
               s["beta26.1_mean"], s["seal_minus_doc"]))
    say("  VERDICT: %s" % verdict)
    say("  receipt: %s (%d B)" % (rec_path, os.path.getsize(rec_path)))

    print("\n".join(lines[-3:]))
    return payload


if __name__ == "__main__":
    _ap = argparse.ArgumentParser(
        description="Hopfield beta calibration (writes a receipt).")
    _ap.add_argument("--out", default=None,
                     help="receipt path override; also honours HENRI_RECEIPT_DIR. "
                          "The module default IS the committed path, so an ad-hoc "
                          "run must override it or it overwrites committed evidence.")
    _a = _ap.parse_args()
    main(_a.out)
