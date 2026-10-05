"""Does the Zone A codec preserve ORDER? This is the causal test for M4.

WHY THIS TEST
    diag_codec showed the codec is slot-sparse and unit-norm: the cosine between
    two waves equals |A n B| / sqrt(|A||B|), i.e. it depends ONLY on which tokens
    appear, not on their positions. If that is true, then
        "apply IR to 1234"   and   "apply RI to 1234"
    produce the SAME wave, and the decoder cannot learn that IR and RI are
    different programs. That would explain M4's negative transfer (held-out
    accuracy 0.080 against a no-information control 0.259) as a CODEC limit,
    not a readout-capacity limit.

    It is better to know which of the two it is before building a fix.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from henri_core.m4_generative import build_corpus, build_system, run_program  # noqa: E402


def cos(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.to(torch.complex128)
    v = v.to(torch.complex128)
    return float((u.conj() @ v).real / (u.norm() * v.norm()).clamp_min(1e-30))


def main() -> int:
    corpus = build_corpus()
    system, tok = build_system(corpus)

    pairs = [
        ("apply IR to 1234", "apply RI to 1234", "order swap of IR vs RI"),
        ("apply IC to 1234", "apply CI to 1234", "order swap of IC vs CI"),
        ("apply I to 1234", "apply I to 5678", "different input only"),
        ("apply R to 1234", "apply C to 1234", "different op only"),
        ("apply I to 1234", "apply II to 1234", "length 1 vs 2"),
    ]

    print("=== 1. cosine for pairs that MUST differ for composition ===")
    for a, b, why in pairs:
        ca, cb = system.wave_of(a, tok), system.wave_of(b, tok)
        sa = set((ca.abs() > 1e-6).nonzero().flatten().tolist())
        sb = set((cb.abs() > 1e-6).nonzero().flatten().tolist())
        print("  %-34s cos=%.6f  |A|=%d |B|=%d |AnB|=%d"
              % (why, cos(ca, cb), len(sa), len(sb), len(sa & sb)))

    print("\n=== 2. the ACTUAL M4 task: do equal-output programs collide? ===")
    ops = ["I", "R", "C"]
    from itertools import product
    collisions = 0
    total = 0
    for inp in ["1234", "5678"]:
        seen: dict[tuple, list[str]] = {}
        for L in (1, 2, 3):
            for combo in product(ops, repeat=L):
                prog = "".join(combo)
                out = run_program(prog, inp)
                seen.setdefault(out, []).append(prog)
        for out, progs in seen.items():
            if len(progs) > 1:
                total += 1
                collisions += 1
        distinct_outputs = len(seen)
        print("  input %s: %d programs -> %d distinct outputs"
              % (inp, sum(len(v) for v in seen.values()), distinct_outputs))
        # Do distinct programs with the SAME output get the SAME wave?
        same_out_cos = []
        for out, progs in seen.items():
            if len(progs) > 1:
                ws = [system.wave_of(f"apply {p} to {inp}", tok) for p in progs]
                same_out_cos.append(cos(ws[0], ws[1]))
        if same_out_cos:
            print("    mean cos among same-output program pairs: %.6f"
                  % (sum(same_out_cos) / len(same_out_cos)))

    print("\n=== 3. does the wave separate DIFFERENT outputs? ===")
    inp = "1234"
    byout: dict[str, torch.Tensor] = {}
    for L in (1, 2, 3):
        for combo in product(ops, repeat=L):
            prog = "".join(combo)
            out = run_program(prog, inp)
            byout.setdefault(out, system.wave_of(f"apply {prog} to {inp}", tok))
    outs = list(byout)
    print("  distinct outputs for input %s: %s" % (inp, outs))
    if len(outs) > 1:
        cs = [cos(byout[outs[i]], byout[outs[j]])
              for i in range(len(outs)) for j in range(i + 1, len(outs))]
        print("  across-output cosines: min %.6f max %.6f mean %.6f"
              % (min(cs), max(cs), sum(cs) / len(cs)))

    print("\n=== 4. is the code position-sensitive at all? ===")
    w1 = system.wave_of("ab", tok)
    w2 = system.wave_of("ba", tok)
    print("  cos('ab','ba') = %.6f   (1.0 would mean order is discarded)"
          % cos(w1, w2))
    w3 = system.wave_of("apply I to 1234", tok)
    w4 = system.wave_of("1234 apply I to", tok)
    print("  cos(reordered spec) = %.6f" % cos(w3, w4))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
