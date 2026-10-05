"""D131 probe: is the Zone A positional phase SOUND?

CONTEXT
    D127 rotated token t's phasor by t * pos_omega, default pos_omega = pi/2.
    That fixed a real defect (order was discarded). But I derived a prediction
    from the code that I must test before trusting my own fix.

ANALYTIC PREDICTION
    acc[addr(tok)] += polar(1.0, angle[tok] + t*omega).
    Tokens at positions t and t+2 receive phases differing by 2*omega = pi, so
    their phasors OPPOSE and cancel EXACTLY. If that holds, a string with a
    repeated token at even position distance collapses toward zero:
        'IRIR'  -> addr(I) gets phases 0, pi      -> 1 - 1 = 0
                   addr(R) gets phases pi/2, 3pi/2 -> i - i = 0
    The wave becomes the ZERO vector, and structure is annihilated, not encoded.

WHAT THIS MEASURES
    1. tokenisation of each probe string (ByteBPE merges could hide structure)
    2. exact-id control: call _token_writes/_assemble directly, bypassing text
    3. wave NORM per string; a near-zero norm proves cancellation
    4. cos to the base string
    5. norm vs repeat count, to expose the period

No gate, no bound. Diagnostic only.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import build_corpus, build_system  # noqa: E402
from henri_core.tokenizer import ByteBPE                           # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a.norm().clamp_min(1e-12)
    nb = b.norm().clamp_min(1e-12)
    return float(torch.real(torch.vdot(a, b)) / (na * nb))


def main() -> int:
    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)
    system, _ = build_system(corpus)               # positional OFF (committed)
    system_on, _ = build_system(corpus, positional=True)
    ing_off = system.ingress
    ing_on = system_on.ingress

    print("D131 positional-phase soundness probe")
    print("=" * 78)

    # ---- 1. tokenisation of the probe strings
    probes = ["IR", "RI", "IRIR", "IRIRIR", "IIR", "IRR", "IXR"]
    print("tokenisation (ByteBPE):")
    for s in probes:
        print(f"   {s:<8} -> {tok.encode(s)}")
    print()

    # ---- exact-id control, bypassing the tokeniser
    # letters I and R map to stable byte ids in this tokeniser
    ids_I = tok.encode("I")
    ids_R = tok.encode("R")
    iI = ids_I[0] if ids_I else 1
    iR = ids_R[0] if ids_R else 2
    print(f"pinned ids: I={iI} R={iR}")
    if iI == iR:
        print("   WARNING: I and R tokenise to the same id; probe is degenerate")
    print()

    def wave_from_ids(ing, ids):
        with torch.no_grad():
            acc, _ = ing._token_writes(torch.tensor(ids, dtype=torch.long))
            return ing._assemble(acc)

    cases = {
        "IR":     [iI, iR],
        "RI":     [iR, iI],
        "IRIR":   [iI, iR, iI, iR],
        "IRIRIR": [iI, iR, iI, iR, iI, iR],
        "IIR":    [iI, iI, iR],
        "IIII":   [iI, iI, iI, iI],
    }

    for tag, ing, pos in (("OFF (pos_omega off)", ing_off, False),
                          ("ON  (pos_omega=pi/2)", ing_on, True)):
        print(f"ARM {tag}")
        base = wave_from_ids(ing, cases["IR"])
        for name, ids in cases.items():
            w = wave_from_ids(ing, ids)
            n = float(w.norm())
            c = cos(w, base) if n > 1e-9 else float("nan")
            flag = ""
            if n < 1e-6:
                flag = "   <-- ZERO WAVE (annihilated)"
            print(f"   {name:<8} norm={n:.6f}  cos_to_IR={c:+.6f}{flag}")
        print()

    # ---- 5. norm vs number of repeats, to expose the period
    print("norm vs repeat count for alternating I,R (ON arm):")
    for k in range(1, 13):
        ids = []
        for j in range(k):
            ids += [iI, iR]
        w = wave_from_ids(ing_on, ids)
        print(f"   repeats={k:<3} tokens={2*k:<3} norm={float(w.norm()):.6f}")

    print("=" * 78)
    print("READING")
    print("  A near-zero norm at even repeat counts confirms systematic")
    print("  destructive interference: the phase is FREQUENCY-LOCKED to position,")
    print("  so identical tokens at position distance 2 cancel exactly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
