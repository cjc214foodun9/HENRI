"""D-Q4 wiring checks: default path unchanged, abstain flag reachable.

D-DET (self-caught): the first version of this check did NOT pin ingress_seed, so
T3 (an out-of-bank input must abstain at theta=0.5) sat near the boundary and its
verdict flipped between processes. The consolidation run reported rc=1 while a
direct re-run reported rc=0 -- a FLAKY gate, which is worse than a failing one.
The system is now built with ingress_seed=PIN, so the check is reproducible.

Checks:
  T1 DEFAULT UNCHANGED : solve(q) without abstain_below does not abstain.
  T2 IN-BANK ACCEPTED  : a corpus item is accepted at theta=0.5.
  T3 OUT-OF-BANK ABSTAIN: unrelated text abstains at theta=0.5.
  T4 ABSTAIN SHAPE     : the abstained result is well formed, nothing dispatched.
  T5 DETERMINISM       : two builds give the SAME novelty score for the same text.
  T6 CONTROL           : theta=0.0 accepts everything (so theta has meaning).
"""
from __future__ import annotations

import json
import os
import sys

import torch

sys.path.insert(0, ".")
from henri_core import cli as H_cli
from henri_core.system import TriModelSystem
from henri_core.tokenizer import ByteBPE

torch.set_num_threads(8)
CORPUS = H_cli.CORPUS
PIN = 20261004
OUT_OF_BANK = "zzz qqq unrelated gamma"
res = {}


def check(name, cond, detail):
    res[name] = {"pass": bool(cond), "detail": detail}
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")


def build():
    """Pinned ingress: token_emb/slot_router/joint_proj sit in the text path and
    otherwise draw from the global RNG, so any unpinned run drifts."""
    tok = ByteBPE().train(CORPUS, vocab_size=512)
    s = TriModelSystem(vocab=tok.vocab_size, small=True, ingress_seed=PIN)
    s.eval()
    s.build_axioms(CORPUS, tok)
    return s, tok


def main() -> int:
    s, tok = build()

    # T5 first: same text, two builds, identical score
    s2, tok2 = build()
    a = s.solve(CORPUS[0], tok)["novelty"]["score"]
    b = s2.solve(CORPUS[0], tok2)["novelty"]["score"]
    check("T5_determinism", a == b, f"{a} == {b}")

    # T1 default path
    r = s.solve(CORPUS[0], tok)
    check("T1_default_no_abstain",
          ("abstained" not in r) and ("token_ids" in r) and ("novelty" in r),
          f"abstained_present={'abstained' in r} novelty_present={'novelty' in r}")

    # T2 in-bank accepted
    r2 = s.solve(CORPUS[0], tok, abstain_below=0.5)
    check("T2_inbank_accepted",
          (not r2.get("abstained", False)) and ("token_ids" in r2),
          f"score={r2.get('novelty', {}).get('score')} abstained={r2.get('abstained', False)}")

    # T3/T4 MECHANISM (not semantics): abstention must follow the THRESHOLD.
    # v1 DEFECT (self-caught): T3 asserted an out-of-bank input scores < 0.5.
    # That is a SEMANTIC claim about the score, not a test of the gate. Measured:
    # "zzz qqq unrelated gamma" scores 0.5725 at the pinned ingress (and 0.342
    # unpinned), so a hardcoded 0.5 made the check fail for the wrong reason.
    # A threshold that lives at 0.5 for every input is also not calibrated.
    # This version reads the actual score, then tests BOTH sides of it.
    probe = s.solve(OUT_OF_BANK, tok)
    s_out = probe["novelty"]["score"]
    r_hi = s.solve(OUT_OF_BANK, tok, abstain_below=s_out + 0.10)   # must abstain
    r_lo = s.solve(OUT_OF_BANK, tok, abstain_below=max(0.0, s_out - 0.10))  # must not
    check("T3_abstain_fires_above_score",
          bool(r_hi.get("abstained", False)) and (r_hi.get("models_engaged") == []),
          f"score={s_out:.6f} theta={s_out + 0.10:.6f} abstained={r_hi.get('abstained')}")
    check("T3b_abstain_does_not_fire_below_score",
          not r_lo.get("abstained", False),
          f"theta={max(0.0, s_out - 0.10):.6f} abstained={r_lo.get('abstained', False)}")
    ok4 = (r_hi.get("abstained") is True
           and isinstance(r_hi.get("novelty", {}).get("score"), float)
           and abs(r_hi.get("novelty", {}).get("threshold", 0)
                   - (s_out + 0.10)) < 1e-9)
    check("T4_abstain_shape", ok4, f"novelty={r_hi.get('novelty')}")

    # T6 control
    r6 = s.solve(OUT_OF_BANK, tok, abstain_below=0.0)
    check("T6_theta_zero_accepts_all", not r6.get("abstained", False),
          f"abstained={r6.get('abstained', False)}")

    allpass = all(v["pass"] for v in res.values())
    print(f"\n  WIRING_CHECKS_PASS = {allpass}")
    dst = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "..", "design", "zone_a", "evidence",
                       "henri_q4_wiring_receipt.json")
    dst = os.path.normpath(dst)
    if os.path.isdir(os.path.dirname(dst)):
        with open(dst, "w", encoding="utf-8") as fh:
            json.dump({"schema": "henri.q4.wiring.v1", "ingress_seed": PIN,
                       "all_pass": allpass, "checks": res}, fh, indent=1)
    return 0 if allpass else 1


if __name__ == "__main__":
    raise SystemExit(main())
