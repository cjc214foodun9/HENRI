"""Is BUNDLING the missing algebra? Sum-codec vs FHRR binding-codec on M4.

HYPOTHESIS H-BIND (the remedy candidate)
    Zone A writes ONE phasor per token at an address derived from token IDENTITY
    (acc[tok % slot_dim] += polar(1, angle[tok])). That is BUNDLING: a sum.
    A sum is commutative, so order must be injected by a separate phase term,
    and that term aliases every 4 positions (D131). The cosine of two waves is
    normalized SET OVERLAP, not structure.

    Vector-symbolic theory separates BUNDLING (+) from BINDING (*). Binding must
    be NON-COMMUTATIVE and invertible. The standard choice is the element-wise
    complex product (FHRR):
        bind(token, position) = v_tok * p_pos        (full width, unit phasors)
    Position enters through the BOUND factor, so 'ab' and 'ba' differ at every
    distance, with no aliasing. The production Zone A has no binding at all.

DESIGN (diagnostic only; production zone_a.py is NOT modified)
    A  production sum codec, positional OFF   (as committed)
    B  production sum codec, positional ON    (the D127 fix)
    C  FHRR binding codec  (replaces system.ingress; exposes encode_text)
    D  shuffled-pairing control on arm C's own architecture
    Identical readout, construction pin, steps, split.

D132 FIX: the first version wrapped the system in a Shim that lacked wave_of, so
    the run died with AttributeError. The FHRR codec now exposes encode_text and
    is installed directly as system.ingress, which is the real consumer.

PRE-REGISTERED READING, before the run
    C clears B and the control on held-out programs -> binding is the missing
    algebra, and the remedy is a codec change.
    C ~ B ~ control -> the algebra is NOT the bottleneck; the limit is the
    readout's compositional mechanism or data scale, and this hypothesis dies.
    Either outcome is decisive. No bound moves.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import (M4Config, WaveTextGenerator,  # noqa: E402
                                      build_corpus, build_system, programs,
                                      run_program, unigram_floor)

PIN = 20261004
MAXLEN = 64

ALL_INPUTS = [f"{1000 + 100 * i:04d}" for i in range(16)]
TRAIN_IN, HOLD_IN = ALL_INPUTS[:12], ALL_INPUTS[12:]
P12 = programs(2)
P3 = [p for p in programs(3) if len(p) == 3]


def pairs(progs, ins):
    return [(f"apply {p} to {i}", run_program(p, i)) for p in progs for i in ins]


TRAIN = pairs(P12, TRAIN_IN)
B_IN = pairs(P12, HOLD_IN)
C_COMP = pairs(P3, TRAIN_IN)


class FHRRCodec(nn.Module):
    """VSA binding codec: bind(token, position) = v_tok * p_pos, element-wise.

    Exposes encode_text(text, tok) so it drops in as TriModelSystem.ingress.
    Buffers only: the codec is deterministic geometry, not a trained layer.
    """

    def __init__(self, dim: int, vocab: int, maxlen: int = MAXLEN,
                 seed: int = PIN):
        super().__init__()
        self.dim = int(dim)
        self.vocab = int(vocab)
        self.maxlen = int(maxlen)
        g = torch.Generator().manual_seed(int(seed))
        vph = torch.rand(self.vocab, self.dim, generator=g) * 2.0 * math.pi
        pph = torch.rand(self.maxlen, self.dim, generator=g) * 2.0 * math.pi
        self.register_buffer("v", torch.polar(torch.ones(self.vocab, self.dim), vph))
        self.register_buffer("p", torch.polar(torch.ones(self.maxlen, self.dim), pph))

    @torch.no_grad()
    def encode_text(self, text: str, tok) -> torch.Tensor:
        ids = tok.encode(text) or [0]
        acc = torch.zeros(self.dim, dtype=torch.complex64)
        for t, i in enumerate(ids[:self.maxlen]):
            i = int(i) % self.vocab
            acc = acc + self.v[i] * self.p[t]
        return acc / acc.norm().clamp_min(1e-12)


def make(kind: str, corpus):
    system, tok = build_system(corpus, pin_seed=PIN, positional=(kind == "pos"))
    if kind == "bind":
        system.ingress = FHRRCodec(dim=system.dim, vocab=tok.vocab_size, seed=PIN)
    return system, tok


def evaluate(gen, rows, tok, cfg) -> dict:
    sp = [s for s, _ in rows]
    tg = [tok.encode(t) for _, t in rows]
    return {"em": round(gen.exact_match(sp, tg, cfg), 6),
            "acc": round(gen.token_accuracy(sp, tg, cfg), 6)}


def run_arm(kind: str, corpus, tok, rows, cfg, shuffle: bool = False) -> dict:
    system, _ = make(kind, corpus)
    if shuffle:
        g = torch.Generator().manual_seed(7)
        idx = torch.randperm(len(rows), generator=g).tolist()
        rows = [(s, rows[j][1]) for (s, _), j in zip(rows, idx)]
    gen = WaveTextGenerator(system, tok, train_body=True)
    t0 = time.time()
    rep = gen.fit([s for s, _ in rows], [tok.encode(t) for _, t in rows], cfg)
    return {"loss_first": round(rep.loss_first, 4),
            "loss_last": round(rep.loss_last, 4),
            "s": round(time.time() - t0, 1),
            "ev": {"train": evaluate(gen, TRAIN, tok, cfg),
                   "B_input_held": evaluate(gen, B_IN, tok, cfg),
                   "C_compose": evaluate(gen, C_COMP, tok, cfg)}}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    corpus = build_corpus(max_len=3, holdout_len=3)
    _, tok = build_system(corpus, pin_seed=PIN)
    cfg = M4Config(steps=args.steps)
    floor = unigram_floor(tok, [tok.encode(t) for _, t in TRAIN],
                          [tok.encode(t) for _, t in C_COMP])

    arms = {
        "A_sum_off":   run_arm("sum", corpus, tok, TRAIN, cfg),
        "B_sum_pos":   run_arm("pos", corpus, tok, TRAIN, cfg),
        "C_fhrr_bind": run_arm("bind", corpus, tok, TRAIN, cfg),
        "D_control":   run_arm("bind", corpus, tok, TRAIN, cfg, shuffle=True),
    }
    out = {"schema": "henri.binding.codec.ablation.v1",
           "unigram_floor_C": round(floor, 6),
           "n_train_rows": len(TRAIN), "n_compose_rows": len(C_COMP),
           "arms": arms, "bounds_moved": False}

    print("IS BUNDLING THE MISSING ALGEBRA? sum vs FHRR binding codec")
    print("=" * 78)
    print(f"  train rows {len(TRAIN)}   compose rows {len(C_COMP)}"
          f"   unigram floor(C) {floor:.4f}")
    hdr = f"  {'arm':<14}{'train_EM':>10}{'B_input_EM':>12}{'C_compose_EM':>14}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for k, r in arms.items():
        e = r["ev"]
        print(f"  {k:<14}{e['train']['em']:>10.4f}"
              f"{e['B_input_held']['em']:>12.4f}"
              f"{e['C_compose']['em']:>14.4f}")
    print("=" * 78)
    a = arms["A_sum_off"]["ev"]["C_compose"]["em"]
    b = arms["B_sum_pos"]["ev"]["C_compose"]["em"]
    c = arms["C_fhrr_bind"]["ev"]["C_compose"]["em"]
    ctl = arms["D_control"]["ev"]["C_compose"]["em"]
    cb = arms["C_fhrr_bind"]["ev"]["B_input_held"]["em"]
    print(f"  compose  A(sum)={a:.4f}  B(sum+pos)={b:.4f}"
          f"  C(bind)={c:.4f}  control={ctl:.4f}  floor={floor:.4f}")
    print(f"  input    C(bind) B_input_EM = {cb:.4f}")
    if c > max(a, b) + 0.05 and c > ctl + 0.05:
        print("  VERDICT: H-BIND SUPPORTED. Binding lifts held-out composition")
        print("  above both the sum codec and the control. The ALGEBRA is the")
        print("  missing piece -> remediate the codec (SPEC_A item 3).")
    else:
        print("  VERDICT: H-BIND FALSIFIED at this scale. The binding codec does")
        print("  not lift composition over the sum codec or the control. The")
        print("  algebra is not the binding constraint; the limit is the readout")
        print("  mechanism or data scale. Recorded as a negative.")

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
