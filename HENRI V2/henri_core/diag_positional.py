"""Does positional algebra restore token ORDER to the Zone A codec?

DEFECT D127
    zone_a.CliffordVLASlotEncoder._token_writes wrote
        acc[slot][tok % slot_dim] += polar(1.0, angle[tok])
    Position t never entered. So 'ab' and 'ba' wrote the SAME two phasors into
    the SAME two accumulators, and cos('ab','ba') was exactly 1.000000.
    The cosine of two waves was normalised set overlap, with no order term.

FIX UNDER TEST
    Rotate token t's phasor by t * pos_omega (relative position, as RoPE applies
    relative position to rotors). Default OFF, so committed receipts reproduce.

WHAT THIS SCRIPT MEASURES (diagnostic only, no gate, no bound)
    1. unit norm under both arms -- the Stiefel constraint must hold
    2. cos('ab','ab') must stay 1.0 in both arms (identity is preserved)
    3. cos('ab','ba') must DROP from 1.0 under ON if order is encoded
    4. mean cos over reversed pairs, real corpus specs
    5. set-overlap must SURVIVE: cos('ab','ac') stays near 0.5 in both arms

A fix that destroys same-order identity or set overlap is not a fix.
"""
from __future__ import annotations

import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                      # noqa: E402
from henri_core.m4_generative import build_corpus            # noqa: E402
from henri_core.system import TriModelSystem                 # noqa: E402
from henri_core.tokenizer import ByteBPE                     # noqa: E402


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """Hermitian cosine. conj on the first argument, real part only."""
    na = torch.linalg.vector_norm(a).clamp_min(1e-12)
    nb = torch.linalg.vector_norm(b).clamp_min(1e-12)
    return float(torch.real(torch.vdot(a, b)) / (na * nb))


def main() -> int:
    corpus = build_corpus()
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)
    arms = {
        "OFF (draft-1 defect)": TriModelSystem(vocab=tok.vocab_size, small=True,
                                               positional=False),
        "ON  (positional fix)": TriModelSystem(vocab=tok.vocab_size, small=True,
                                               positional=True),
    }

    pairs = [
        ("identity 'ab' vs 'ab'", "ab", "ab", 1.0),
        ("ORDER    'ab' vs 'ba'", "ab", "ba", None),
        ("ORDER    'abc' vs 'cba'", "abc", "cba", None),
        ("set-overl 'ab' vs 'ac'", "ab", "ac", None),
        ("corpus   IR vs RI", "apply IR to 1234", "apply RI to 1234", None),
        ("corpus   same spec", "apply IR to 1234", "apply IR to 1234", 1.0),
    ]

    print("D127 positional-encoder diagnostic")
    print("=" * 78)
    out = {}
    for name, sysm in arms.items():
        norms = []
        for s in ("ab", "ba", "abc", "apply IR to 1234"):
            norms.append(float(torch.linalg.vector_norm(sysm.wave_of(s, tok))))
        print(f"\nARM {name}")
        print(f"  unit norm  min={min(norms):.6f} max={max(norms):.6f} "
              f"(Stiefel |Psi|=1)")
        row = {}
        for label, x, y, want in pairs:
            c = cos(sysm.wave_of(x, tok), sysm.wave_of(y, tok))
            row[label] = c
            tag = ""
            if want is not None:
                tag = "  PASS" if abs(c - want) < 1e-4 else f"  FAIL (want {want})"
            print(f"  {label:<26} cos={c:+.6f}{tag}")
        # mean |cos| over reversed corpus specs
        specs = [corpus.specs[i] for i in corpus.train_idx[:24]
                 if len(corpus.specs[i]) > 3]
        revs = []
        for s in specs:
            r = " ".join(w[::-1] for w in s.split())
            revs.append(abs(cos(sysm.wave_of(s, tok), sysm.wave_of(r, tok))))
        row["mean_abs_cos_reversed"] = sum(revs) / max(1, len(revs))
        print(f"  mean |cos| reversed corpus  {row['mean_abs_cos_reversed']:.6f}")
        out[name] = row

    print("\n" + "=" * 78)
    off, on = out["OFF (draft-1 defect)"], out["ON  (positional fix)"]
    a = off["ORDER    'ab' vs 'ba'"]
    b = on["ORDER    'ab' vs 'ba'"]
    off_ident = off["identity 'ab' vs 'ab'"]
    on_ident = on["identity 'ab' vs 'ab'"]
    off_over = off["set-overl 'ab' vs 'ac'"]
    on_over = on["set-overl 'ab' vs 'ac'"]
    off_rev = off["mean_abs_cos_reversed"]
    on_rev = on["mean_abs_cos_reversed"]
    print("VERDICT")
    print(f"  order term  cos('ab','ba'):  OFF={a:+.6f}  ->  ON={b:+.6f}")
    print(f"  identity    cos('ab','ab'):  OFF={off_ident:+.6f}  ON={on_ident:+.6f}")
    print(f"  set overlap cos('ab','ac'):  OFF={off_over:+.6f}  ON={on_over:+.6f}")
    print(f"  reversed corpus mean |cos|:  OFF={off_rev:.6f}  ON={on_rev:.6f}")
    ok_order = abs(b) < 0.5
    ok_ident = abs(on_ident - 1.0) < 1e-4
    ok_overlap = abs(on_over - off_over) < 0.05
    print(f"\n  order encoded        : {'YES' if ok_order else 'NO'}")
    print(f"  identity preserved   : {'YES' if ok_ident else 'NO'}")
    print(f"  set overlap preserved: {'YES' if ok_overlap else 'NO'}")
    print(f"  root cause D127      : "
          f"{'CONFIRMED and FIXED' if (ok_order and ok_ident and ok_overlap) else 'NOT YET'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
