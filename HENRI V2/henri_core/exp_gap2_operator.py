"""GAP #2: does an operator-factorized readout COMPOSE where a flat readout memorizes?

PRE-REGISTERED BEFORE THE RUN (design/zone_a/GAP2_operator_factorization_v1.md).

Question
    M4's flat readout reaches train EM 1.000 and held-out composition EM 0.0556,
    below the unigram floor 0.2522. Is that a CAPACITY limit or an ALGEBRA limit?

Design
    Corpus: the M4 grammar. Ops I (digit+1 mod 10), R (reverse), C (copy first
    digit). Program p in {I,R,C}^L. Input x is 4 digits. Output y = apply(p, x).
    Train on L in {1,2}. Held out: L in {3,4,5}. Composition is tested, not recall.

    ARM A  flat      : wave features -> y directly (4 positions x 10 classes).
    ARM C  operator  : wave features -> (program, input)  then apply the EXACT
                       algebra of the three factors:
                         torus shift (Z_10 per digit)  <- I
                         position permutation (4x4)    <- R
                         copy / broadcast factor       <- C
                       The learnable part IDENTIFIES operators; the factorized
                       part APPLIES them exactly, so composition never has to be
                       learned from examples of length 3-5.
    ARM D  control   : identical to C, but the program->operator pairing is
                       SHUFFLED across rows, so identification carries no signal.
    POSITIVE control : C given the GROUND-TRUTH program and input. Exact algebra
                       must return EM = 1.0. If it does not, the algebra is wrong.

Criterion (SIGNED, D135)
    C must beat A by >= +0.05 held-out EM and degrade nothing. Movement alone,
    in either direction, is not support.

Isolation (D136/D138)
    Identical construction pin; identical corpus rows; identical feature
    extractor; decoder frozen. Arm must reproduce the baseline before deltas.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import build_corpus, build_system  # noqa: E402

PIN = 20261004
OPS = ("I", "R", "C")
MAXP = 5
ND = 10


# ------------------------------------------------------------------ algebra
def apply_exact(program: str, x: torch.Tensor) -> torch.Tensor:
    """Apply I/R/C to a [B,4] digit tensor. Exact, differentiable, no training."""
    out = x.clone()
    for op in program:
        if op == "I":
            out = (out + 1) % ND
        elif op == "R":
            out = out.flip(-1)
        elif op == "C":
            out = out[:, :1].expand(-1, out.shape[1]).contiguous()
        else:
            raise ValueError(op)
    return out


def encode_program(prog: str) -> torch.Tensor:
    """Program -> [MAXP] index in {0=I,1=R,2=C,3=pad}."""
    ids = [OPS.index(c) for c in prog]
    ids += [3] * (MAXP - len(ids))
    return torch.tensor(ids[:MAXP], dtype=torch.long)


def digits_of(s: str) -> torch.Tensor:
    return torch.tensor([int(c) for c in s], dtype=torch.long)


def make_rows(max_len: int):
    """(spec, program, input4, output4) over the M4 corpus."""
    from henri_core.m4_generative import INPUTS, programs
    rows = []
    for p in programs(max_len):
        for inp in INPUTS:
            out = inp
            for op in p:
                if op == "I":
                    out = "".join(str((int(c) + 1) % 10) for c in out)
                elif op == "R":
                    out = out[::-1]
                elif op == "C":
                    out = out[0] * len(out)
            rows.append((f"apply {p} to {inp}", p, inp, out))
    return rows


# -------------------------------------------------------------------- arms
class FlatArm(nn.Module):
    """wave feature -> 4 digits, 10 classes each. No algebra."""

    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, 4 * ND))

    def forward(self, f):
        return self.net(f).view(-1, 4, ND)


class FactorArm(nn.Module):
    """wave feature -> (program, input) ; then EXACT factorized application."""

    def __init__(self, d_in: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU())
        self.head_prog = nn.Linear(hidden, MAXP * 4)      # 4 classes w/ pad
        self.head_inp = nn.Linear(hidden, 4 * ND)

    def forward(self, f, program_gt=None, input_gt=None):
        h = self.trunk(f)
        prog_logits = self.head_prog(h).view(-1, MAXP, 4)
        inp_logits = self.head_inp(h).view(-1, 4, ND)
        if program_gt is not None and input_gt is not None:
            prog_hat, inp_hat = program_gt, input_gt
        else:
            prog_hat = prog_logits.argmax(-1)
            inp_hat = inp_logits.argmax(-1)
        return prog_logits, inp_logits, prog_hat, inp_hat


def apply_programs_batch(prog_idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """prog_idx [B,MAXP] in 0..3 ; x [B,4] -> y [B,4]. Vectorized exact algebra."""
    out = x.clone()
    b = x.shape[0]
    for t in range(prog_idx.shape[1]):
        op = prog_idx[:, t]
        inc = (out + 1) % ND
        rev = out.flip(-1)
        cpy = out[:, :1].expand(-1, 4)
        take = op.unsqueeze(-1)
        out = torch.where(take == 0, inc, torch.where(take == 1, rev,
              torch.where(take == 2, cpy, out)))
    return out


def feat_of(system, tok, specs, batch: int = 64) -> torch.Tensor:
    """Decoder pooled feature per spec: [N, d_model]. Frozen, no_grad."""
    feats = []
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            chunk = specs[i:i + batch]
            psi = torch.stack([system.wave_of(s, tok) for s in chunk])
            _bands, tokens = system.decoder.encode_wave(psi)
            feats.append(tokens.mean(dim=1).float())
    return torch.cat(feats, 0)


def em(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).all(dim=-1).float().mean())


def tokacc(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean())


# -------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, required=True)
    a = ap.parse_args()
    t0 = time.time()

    rows = make_rows(MAXP)
    tr = [r for r in rows if len(r[1]) <= 2]
    ho = [r for r in rows if len(r[1]) >= 3]
    print(f"[corpus] train={len(tr)} heldout={len(ho)} "
          f"(len3+={sum(1 for r in ho if len(r[1]) == 3)}, "
          f"len4={sum(1 for r in ho if len(r[1]) == 4)}, "
          f"len5={sum(1 for r in ho if len(r[1]) == 5)})", flush=True)

    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN)

    tr_spec = [r[0] for r in tr]
    ho_spec = [r[0] for r in ho]
    Ftr = feat_of(system, tok, tr_spec)
    Fho = feat_of(system, tok, ho_spec)
    print(f"[feat] d_model={Ftr.shape[1]}", flush=True)

    Xtr = torch.stack([digits_of(r[2]) for r in tr])
    Ytr = torch.stack([digits_of(r[3]) for r in tr])
    Ptr = torch.stack([encode_program(r[1]) for r in tr])
    Xho = torch.stack([digits_of(r[2]) for r in ho])
    Yho = torch.stack([digits_of(r[3]) for r in ho])
    Pho = torch.stack([encode_program(r[1]) for r in ho])

    # unigram floor on held-out: most frequent digit per position from TRAIN
    floor_pred = []
    for pos in range(4):
        vals = Ytr[:, pos]
        floor_pred.append(vals.bincount(minlength=ND).argmax())
    floor_pred = torch.stack(floor_pred).unsqueeze(0).expand(len(ho), -1)
    floor_em = em(floor_pred, Yho)
    print(f"[floor] unigram exact-match = {floor_em:.6f}", flush=True)

    results = {}

    # ---------- POSITIVE CONTROL: exact algebra with ground-truth operators
    y_oracle = apply_programs_batch(Pho, Xho)
    pc_em = em(y_oracle, Yho)
    print(f"[positive-control] exact algebra, GT operators: EM={pc_em:.6f} "
          f"(must be 1.0)", flush=True)

    def train_flat(steps, seed):
        torch.manual_seed(seed)
        m = FlatArm(Ftr.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        for _ in range(steps):
            opt.zero_grad()
            lg = m(Ftr)
            loss = F.cross_entropy(lg.reshape(-1, ND), Ytr.reshape(-1))
            loss.backward()
            opt.step()
        with torch.no_grad():
            pred_tr = m(Ftr).argmax(-1)
            pred_ho = m(Fho).argmax(-1)
        return m, em(pred_tr, Ytr), em(pred_ho, Yho), tokacc(pred_ho, Yho)

    def train_factor(steps, seed, shuffle: bool):
        torch.manual_seed(seed)
        m = FactorArm(Ftr.shape[1])
        opt = torch.optim.Adam(m.parameters(), lr=a.lr)
        g = torch.Generator().manual_seed(1234)
        tgt_ptr = Ptr.clone()
        tgt_xtr = Xtr.clone()
        if shuffle:
            perm = torch.randperm(len(tr), generator=g)
            tgt_ptr = Ptr[perm]
            tgt_xtr = Xtr[perm]
        for _ in range(steps):
            opt.zero_grad()
            prog_logits, inp_logits, _, _ = m(Ftr)
            loss = F.cross_entropy(prog_logits.reshape(-1, 4), tgt_ptr.reshape(-1))
            loss = loss + F.cross_entropy(inp_logits.reshape(-1, ND),
                                           tgt_xtr.reshape(-1))
            loss.backward()
            opt.step()
        with torch.no_grad():
            _, _, p_tr, x_tr = m(Ftr)
            _, _, p_ho, x_ho = m(Fho)
            y_tr = apply_programs_batch(p_tr, x_tr)
            y_ho = apply_programs_batch(p_ho, x_ho)
            prog_acc_ho = float((p_ho == Pho).float().mean())
        return (m, em(y_tr, Ytr), em(y_ho, Yho), tokacc(y_ho, Yho), prog_acc_ho)

    print(f"[arm A flat] training {a.steps} steps ...", flush=True)
    mA, a_tr, a_ho, a_tok = train_flat(a.steps, a.seed)
    print(f"[arm A] trainEM={a_tr:.4f} heldEM={a_ho:.4f} heldTok={a_tok:.4f}",
          flush=True)

    print(f"[arm C operator] training {a.steps} steps ...", flush=True)
    mC, c_tr, c_ho, c_tok, c_prog = train_factor(a.steps, a.seed, shuffle=False)
    print(f"[arm C] trainEM={c_tr:.4f} heldEM={c_ho:.4f} heldTok={c_tok:.4f} "
          f"prog_ident_acc={c_prog:.4f}", flush=True)

    print(f"[arm D control] training {a.steps} steps ...", flush=True)
    mD, d_tr, d_ho, d_tok, d_prog = train_factor(a.steps, a.seed, shuffle=True)
    print(f"[arm D] trainEM={d_tr:.4f} heldEM={d_ho:.4f} heldTok={d_tok:.4f}",
          flush=True)

    results = {
        "A_flat": {"train_em": a_tr, "held_em": a_ho, "held_tok": a_tok},
        "C_operator": {"train_em": c_tr, "held_em": c_ho, "held_tok": c_tok,
                       "program_identification_acc": c_prog},
        "D_shuffled": {"train_em": d_tr, "held_em": d_ho, "held_tok": d_tok},
        "positive_control_exact_algebra": pc_em,
        "unigram_floor_held": floor_em,
    }

    # ---- verdict: SIGNED (D135), with a vacuity guard on the positive control
    guards = {
        "positive_control_ok": pc_em >= 0.999,
        "armC_identifies_programs": c_prog > 0.50,
        "armD_below_armC": d_ho < c_ho,
        "armA_reproduces_memorization": a_tr > 0.90,
    }
    d_ca = c_ho - a_ho
    helps = d_ca >= 0.05
    hurts = d_ca < -0.05
    if not all((guards["positive_control_ok"], guards["armC_identifies_programs"],
                guards["armA_reproduces_memorization"])):
        verdict = "VACUOUS"
    elif helps:
        verdict = "FACTORIZED_COMPOSES"
    elif hurts:
        verdict = "FACTORIZED_HARMS"
    else:
        verdict = "NO_DIFFERENCE"
    why = (f"d(held EM, C-A)={d_ca:+.6f}; floor={floor_em:.6f}; "
           f"C={c_ho:.6f} A={a_ho:.6f} D={d_ho:.6f}")

    out = {
        "schema": "henri.gap2.operator-factorization.v1",
        "verdict": verdict,
        "why": why,
        "guards": guards,
        "delta_C_minus_A_held_em": d_ca,
        "arms": results,
        "corpus": {"train": len(tr), "heldout": len(ho),
                   "heldout_by_len": {
                       "3": sum(1 for r in ho if len(r[1]) == 3),
                       "4": sum(1 for r in ho if len(r[1]) == 4),
                       "5": sum(1 for r in ho if len(r[1]) == 5)}},
        "run": {"steps": a.steps, "lr": a.lr, "seed": a.seed, "pin": PIN},
        "seconds": round(time.time() - t0, 2),
        "defects": {
            "D145": "gap#2 harness: learnable part IDENTIFIES operators; the "
                    "tripartite algebra APPLIES them exactly (torus shift | "
                    "position permutation | copy broadcast). Composition is never "
                    "learned from length-3 examples.",
            "D146": "vacuity guard: the positive control runs the exact algebra "
                    "with GROUND-TRUTH operators and must return EM=1.0, else "
                    "the algebra itself is wrong and no verdict is valid.",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("verdict", "why", "guards", "seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
