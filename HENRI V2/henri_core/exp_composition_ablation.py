"""WHAT IS MISSING? Composition vs input generalization -- a 2x2 ablation.

MOTIVATION (measured at 4dee7c3)
    M4: train EM 1.000, held-out EM 0.083, no-information control 0.259.
    The readout MEMORIZES and does not compose. Two hypotheses remain and the
    existing corpus cannot separate them, because it varies programs but holds
    the inputs fixed:

        H1 DATA SCALE   -- too few training pairs; more would generalize.
        H2 ARCHITECTURE -- no compositional operator; more data will not help.

A THIRD possibility the old corpus could not see:
        H3 INPUT GENERALIZATION -- it fails even on SEEN program lengths with
        UNSEEN inputs, in which case the gap is not composition at all.

DESIGN: hold out on TWO axes independently.
    inputs   : 16 total, 12 train, 4 held out
    programs : length 1-2 train, length 3 held out (composition)

    train      len<=2, train inputs    144 rows   (real)
    train_small len<=2, first 4 inputs  48 rows   (the original M4 scale)
    B input    len<=2, held-out inputs  48 rows   -- INPUT generalization
    C compose  len 3,  train inputs    324 rows   -- COMPOSITION
    D both     len 3,  held-out inputs 108 rows   -- both held out

    control    identical model, SHUFFLED train pairing, same 144 rows.
               Built on a FRESH system with the same construction pin, so the
               only difference is the label pairing. (run_gates reuses one
               system for its arms, which shares the decoder -- avoided here.)

PRE-REGISTERED READING, written before the run
    If C stays near the control while A is ~1.0 -> H2 (architecture).
    If C rises materially with the 3x data arm          -> H1 (data scale).
    If B is near zero while A is high                   -> H3 (input, not order).
No bound moves. Diagnostic ablation, not a gate.
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import (Corpus, M4Config,    # noqa: E402
                                      WaveTextGenerator, build_system, programs,
                                      run_program, unigram_floor)

ALL_INPUTS = [f"{1000 + 100 * i:04d}" for i in range(16)]
TRAIN_IN, HOLD_IN = ALL_INPUTS[:12], ALL_INPUTS[12:]

P12 = programs(2)                                   # lengths 1 and 2
P3 = [p for p in programs(3) if len(p) == 3]        # length 3 only


def pairs(progs, ins):
    return [(f"apply {p} to {i}", run_program(p, i)) for p in progs for i in ins]


TRAIN = pairs(P12, TRAIN_IN)            # 144
TRAIN_SMALL = pairs(P12, TRAIN_IN[:4])  # 48, the original M4 scale
B_IN = pairs(P12, HOLD_IN)              # 48
C_COMP = pairs(P3, TRAIN_IN)            # 324
D_BOTH = pairs(P3, HOLD_IN)             # 108

SEED = 20261004


def make_corpus():
    specs = [s for s, _ in TRAIN] + [s for s, _ in C_COMP]
    traces = [t for _, t in TRAIN] + [t for _, t in C_COMP]
    return Corpus(P12 + P3, traces, specs,
                  list(range(len(TRAIN))), [],
                  specs + traces)


def fit_and_eval(train_rows, shuffle_targets: bool, cfg: M4Config,
                 corpus) -> dict:
    system, tok = build_system(corpus, pin_seed=SEED)
    if shuffle_targets:
        g = torch.Generator().manual_seed(7)
        idx = torch.randperm(len(train_rows), generator=g).tolist()
        rows = [(s, train_rows[j][1]) for (s, _), j in zip(train_rows, idx)]
    else:
        rows = train_rows
    gen = WaveTextGenerator(system, tok, train_body=True)
    spec_tr = [s for s, _ in rows]
    tgt_tr = [tok.encode(t) for _, t in rows]
    t0 = time.time()
    rep = gen.fit(spec_tr, tgt_tr, cfg)

    def ev(prs):
        sp = [s for s, _ in prs]
        tg = [tok.encode(t) for _, t in prs]
        return {"acc": round(gen.token_accuracy(sp, tg, cfg), 6),
                "em": round(gen.exact_match(sp, tg, cfg), 6)}

    return {"loss_first": round(rep.loss_first, 4),
            "loss_last": round(rep.loss_last, 4),
            "eval_s": round(time.time() - t0, 1),
            "ev": {"train": ev(TRAIN_SMALL), "A_train_full": ev(TRAIN),
                   "B_input_held": ev(B_IN), "C_compose": ev(C_COMP),
                   "D_both_held": ev(D_BOTH)}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    corpus = make_corpus()
    cfg = M4Config(steps=args.steps)
    _, tok = build_system(corpus, pin_seed=SEED)

    floor_c = unigram_floor(tok, [tok.encode(t) for _, t in TRAIN],
                            [tok.encode(t) for _, t in C_COMP])

    real_small = fit_and_eval(TRAIN_SMALL, False, cfg, corpus)
    real = fit_and_eval(TRAIN, False, cfg, corpus)
    ctrl = fit_and_eval(TRAIN, True, cfg, corpus)

    out = {
        "schema": "henri.composition.ablation.v1",
        "n_train_rows": len(TRAIN), "n_train_rows_small": len(TRAIN_SMALL),
        "unigram_floor_on_C": round(floor_c, 6),
        "real_small": real_small, "real": real, "control": ctrl,
        "pre_registered": ["H1 data scale", "H2 architecture",
                           "H3 input generalization"],
    }

    print("WHAT IS MISSING? composition vs input generalization (2x2)")
    print("=" * 78)
    print(f"  train rows: small={len(TRAIN_SMALL)}  full={len(TRAIN)}"
          f"   C_compose rows={len(C_COMP)}")
    print(f"  C control rows={len(ctrl['ev']['C_compose']) and len(C_COMP)}")
    print(f"  unigram floor on C: {floor_c:.6f}")
    print()
    hdr = f"  {'arm':<14}{'train':>9}{'B_input':>9}{'C_compose':>11}{'D_both':>9}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for tag, r in (("real_small", real_small), ("real_full", real),
                   ("control", ctrl)):
        e = r["ev"]
        print(f"  {tag:<14}{e['A_train_full']['em']:>9.4f}"
              f"{e['B_input_held']['em']:>9.4f}"
              f"{e['C_compose']['em']:>11.4f}{e['D_both_held']['em']:>9.4f}")
    print()
    print("  exact match (EM). A=train, B=unseen INPUT, C=unseen PROGRAM")
    print("  length, D=both.  C is the composition arm.")
    print("=" * 78)
    a = real["ev"]["A_train_full"]["em"]
    b = real["ev"]["B_input_held"]["em"]
    c = real["ev"]["C_compose"]["em"]
    cc = ctrl["ev"]["C_compose"]["em"]
    cs = real_small["ev"]["C_compose"]["em"]
    print(f"  memorization  A={a:.4f}")
    print(f"  input gen     B={b:.4f}   (unseen inputs, seen program lengths)")
    print(f"  composition   C={c:.4f}   control={cc:.4f}  floor={floor_c:.4f}")
    print(f"  data scale    C small={cs:.4f} -> full={c:.4f}"
          f"   (3x rows: {len(TRAIN_SMALL)} -> {len(TRAIN)})")
    print()
    if abs(c - cc) < 0.05 and a > 0.8:
        print("  VERDICT: H2 -- ARCHITECTURE. The readout memorizes but does not")
        print("  compose, and 3x training rows do not change it. More data of")
        print("  this kind will not produce the missing capability.")
    elif c > cs + 0.05:
        print("  VERDICT: H1 -- DATA SCALE. Held-out composition improves with")
        print("  more training rows, so the gap is volume, not structure.")
    elif b < 0.2:
        print("  VERDICT: H3 -- INPUT GENERALIZATION. It fails on unseen inputs")
        print("  with seen program lengths, so the gap is not composition.")
    else:
        print("  VERDICT: MIXED -- read the table. No single hypothesis explains")
        print("  the pattern; record the per-arm numbers and reframe.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
