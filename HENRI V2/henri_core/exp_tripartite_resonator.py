"""DIRECTIVE 2: Sequence-preserving Tripartite Resonator Network.

WHY THIS VERSION DIFFERS FROM exp_gap2_operator.py
    D148 (self-caught): the first attempt mean-pooled the macro-token sequence
    to ONE 128-d vector, so arm A could not fit even 48 training rows (train EM
    0.125 vs the M4 baseline 1.000). No verdict was taken from that run.
    D158: the corrected design keeps the FULL [B, M, d_model] sequence and
    aggregates it with a LEARNED attention, so token order survives to the head.

THE TRIPARTITE FACTORIZATION (document Dir 3), measured form
    The document writes
        T_task = Factorize(Torus Roll  (x)  Topological Mask  (x)  Spin(3) Rotor)
    On this corpus the three ops are EXACT and differentiable:
        I : per-digit increment on the Z_10 torus        (torus shift)
        R : the 4-position permutation                    (position reversal)
        C : broadcast of position 0                       (copy factor)
    The learnable part only IDENTIFIES which operators apply. The factorized
    part APPLIES them exactly, so composition is never learned from length-3
    examples. Positive control: ground-truth operators must give EM 1.0.

THE RESONATOR STEP
    Identification and application iterate. Pass 1 identifies (program, input),
    applies the exact algebra, and feeds the RESIDUAL back as an extra query
    channel for pass 2. Two passes are used; the second is the refinement the
    document calls crystallization.

ARMS
    F  flat_seq   : full sequence -> trace tokens directly (no algebra)
    T  tripartite : full sequence -> (program, input) -> EXACT tripartite algebra
    R  resonator  : tripartite + 2 refinement passes
    D  control    : tripartite, program->operator pairing SHUFFLED
    PC positive   : tripartite with GROUND-TRUTH operators (must be EM 1.0)

CRITERION (SIGNED, D135). T or R must beat F by >= +0.05 held-out EM and
degrade nothing. Movement alone is not support.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.m4_generative import build_corpus, build_system   # noqa: E402

PIN = 20261004
OPS = ("I", "R", "C")
MAXP = 5
ND = 10
NPOS = 4


# ------------------------------------------------------------------ corpus
def make_rows(max_len=MAXP):
    from henri_core.m4_generative import INPUTS, programs, run_program
    rows = []
    for p in programs(max_len):
        for inp in INPUTS:
            rows.append((f"apply {p} to {inp}", p, inp, run_program(p, inp)))
    return rows


def prog_ids(p: str):
    v = [OPS.index(c) for c in p]
    return v + [3] * (MAXP - len(v))


def digs(s: str):
    return [int(c) for c in s]


# ------------------------------------------------------- exact tripartite algebra
def apply_tripartite(prog_idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """prog_idx [B,MAXP] in 0..3 ; x [B,4] -> y [B,4]. EXACT and differentiable.

    Factor 1 torus shift  (I), factor 2 position permutation (R),
    factor 3 copy/broadcast (C). No training touches this map.
    """
    out = x.clone()
    for t in range(prog_idx.shape[1]):
        op = prog_idx[:, t].unsqueeze(-1)
        inc = (out + 1) % ND                       # torus roll on Z_10
        rev = out.flip(-1)                         # position reversal
        cpy = out[:, :1].expand(-1, NPOS)          # copy / broadcast
        out = torch.where(op == 0, inc,
              torch.where(op == 1, rev,
              torch.where(op == 2, cpy, out)))
    return out


# --------------------------------------------------------------- model
class SeqTripartite(nn.Module):
    """Sequence-preserving readout with the tripartite factorization."""

    def __init__(self, d_model: int, n_macro: int, hidden: int = 256,
                 passes: int = 1):
        super().__init__()
        self.passes = int(passes)
        # learned aggregation over the MACRO-TOKEN axis, not mean pooling.
        # One query vector per output slot attends over M tokens (D158).
        self.q_prog = nn.Parameter(torch.randn(1, d_model) * 0.02)
        self.q_inp = nn.Parameter(torch.randn(NPOS, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # per-position token embedding so order is explicit
        self.pos_emb = nn.Parameter(torch.randn(n_macro, d_model) * 0.02)
        self.trunk = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU())
        self.head_prog = nn.Linear(hidden, MAXP * 4)
        # D159 (same class, second site): head_inp is applied PER POSITION, so it
        # must emit ND per row, not NPOS*ND. The refine_inp below is different:
        # it consumes the whole [MAXP*4] program logits and DOES emit NPOS*ND.
        self.head_inp = nn.Linear(hidden, ND)
        # resonator refinement: residual -> operator correction
        self.refine_prog = nn.Linear(NPOS * ND, MAXP * 4)
        self.refine_inp = nn.Linear(MAXP * 4, NPOS * ND)

    def _attend(self, h: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """h [B,M,d], q [Q,d] -> [B,Q,d]. Learned softmax over the M axis."""
        k = self.k_proj(h)                                  # [B,M,d]
        v = self.v_proj(h)
        logits = torch.einsum("qd,bmd->bqm", q, k) / (h.shape[-1] ** 0.5)
        att = torch.softmax(logits, dim=-1)
        return torch.einsum("bqm,bmd->bqd", att, v)         # [B,Q,d]

    def forward(self, h: torch.Tensor, gt_prog=None, gt_inp=None,
                shuffled_prog=None):
        b, m, d = h.shape
        h = h + self.pos_emb[:m].unsqueeze(0)               # order explicit
        agg_p = self._attend(h, self.q_prog)                # [B,1,d]
        agg_i = self._attend(h, self.q_inp)                 # [B,4,d]
        hp = self.trunk(agg_p.reshape(b, d))
        hi = self.trunk(agg_i.reshape(b * NPOS, d)).reshape(b, NPOS, -1)
        prog_logits = self.head_prog(hp).view(b, MAXP, 4)
        inp_logits = self.head_inp(hi).view(b, NPOS, ND)

        for _ in range(self.passes - 1):                    # resonator passes
            prog_logits = prog_logits + self.refine_prog(
                inp_logits.reshape(b, -1)).view(b, MAXP, 4)
            inp_logits = inp_logits + self.refine_inp(
                prog_logits.reshape(b, -1)).view(b, NPOS, ND)

        if gt_prog is not None and gt_inp is not None:
            prog_hat, inp_hat = gt_prog, gt_inp
        elif shuffled_prog is not None:
            prog_hat, inp_hat = shuffled_prog, inp_logits.argmax(-1)
        else:
            prog_hat = prog_logits.argmax(-1)
            inp_hat = inp_logits.argmax(-1)
        return prog_logits, inp_logits, prog_hat, inp_hat


class FlatSeq(nn.Module):
    """Full sequence -> trace tokens. No algebra. Same sequence aggregation."""

    def __init__(self, d_model: int, n_macro: int, hidden: int = 256):
        super().__init__()
        self.q = nn.Parameter(torch.randn(NPOS, d_model) * 0.02)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.pos_emb = nn.Parameter(torch.randn(n_macro, d_model) * 0.02)
        self.trunk = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                   nn.Linear(hidden, hidden), nn.GELU())
        # D159 (self-caught): head emits ND per (batch, position) row. The first
        # version emitted NPOS*ND=40 and then reshaped to ND=10, so the view had
        # 7680 elements against 1920 -- 4x off. Emit ND and reshape [B,NPOS,ND].
        self.head = nn.Linear(hidden, ND)

    def forward(self, h):
        b, m, d = h.shape
        h = h + self.pos_emb[:m].unsqueeze(0)
        k, v = self.k_proj(h), self.v_proj(h)
        logits = torch.einsum("qd,bmd->bqm", self.q, k) / (d ** 0.5)
        att = torch.softmax(logits, dim=-1)
        agg = torch.einsum("bqm,bmd->bqd", att, v).reshape(b * NPOS, d)
        return self.head(self.trunk(agg)).view(b, NPOS, ND)


def feats_seq(system, tok, specs, batch=32):
    """FULL macro-token sequence per spec: [N, M, d_model]. No mean pooling."""
    out = []
    with torch.no_grad():
        for i in range(0, len(specs), batch):
            ch = specs[i:i + batch]
            psi = torch.stack([system.wave_of(s, tok) for s in ch])
            _b, tk = system.decoder.encode_wave(psi)
            out.append(tk.float())
    return torch.cat(out, 0)


def em(pred, y):
    return float((pred == y).all(-1).float().mean())


def fit_flat(Htr, Ytr, Hho, Yho, hidden, steps, lr, seed=0):
    torch.manual_seed(seed)
    m = FlatSeq(Htr.shape[-1], Htr.shape[1], hidden)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        lg = m(Htr)
        loss = F.cross_entropy(lg.reshape(-1, ND), Ytr.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (em(m(Htr).argmax(-1), Ytr), em(m(Hho).argmax(-1), Yho))


def fit_tri(Htr, Ptr, Xtr, Ytr, Hho, Pho, Xho, Yho, hidden, steps, lr,
            passes=1, shuffle=False, seed=0):
    torch.manual_seed(seed)
    m = SeqTripartite(Htr.shape[-1], Htr.shape[1], hidden, passes=passes)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    g = torch.Generator().manual_seed(1234)
    tp, ti = Ptr.clone(), Xtr.clone()
    if shuffle:
        perm = torch.randperm(len(Ptr), generator=g)
        tp, ti = Ptr[perm], Xtr[perm]
    for _ in range(steps):
        opt.zero_grad()
        pl, il, _, _ = m(Htr)
        loss = F.cross_entropy(pl.reshape(-1, 4), tp.reshape(-1))
        loss = loss + F.cross_entropy(il.reshape(-1, ND), ti.reshape(-1))
        loss.backward()
        opt.step()
    with torch.no_grad():
        _, _, p_tr, x_tr = m(Htr)
        _, _, p_ho, x_ho = m(Hho)
        y_tr = apply_tripartite(p_tr, x_tr)
        y_ho = apply_tripartite(p_ho, x_ho)
        prog_acc = float((p_ho == Pho).float().mean())
        inp_acc = float((x_ho == Xho).float().mean())
    return (em(y_tr, Ytr), em(y_ho, Yho), prog_acc, inp_acc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=3e-3)
    # D161 (self-caught): arm F read trainEM=0.1250 at hidden 256 AND at 1024,
    # identical to 4 decimals. The gap#2 capacity ladder already showed that at
    # dk_target=32 the pooled features need hidden=4096 to fit TRAIN (0.2083 at
    # 256, 1.0000 at 4096). My guard correctly refused a verdict on a baseline
    # that cannot fit its own training set. Default capacity raised to match.
    # D162 (self-caught): the first patch left the OLD --hidden 256 line in
    # place, so argparse declared --hidden twice and would have raised.
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--dk-target", type=int, default=32,
                    help="SPEC_B pooling width; 0 = old rule")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    rows = make_rows(MAXP)
    tr = [r for r in rows if len(r[1]) <= 2]
    ho = [r for r in rows if len(r[1]) >= 3]
    corpus = build_corpus(max_len=3, holdout_len=3)
    system, tok = build_system(corpus, ingress_seed=PIN, pin_seed=PIN,
                               dk_target=a.dk_target)
    pk = system.decoder.pooling
    print(f"[setup] n_mem={pk.n_mem} d_k={pk.d_k} n_macro={pk.n_macro} "
          f"train={len(tr)} heldout={len(ho)}", flush=True)

    Htr = feats_seq(system, tok, [r[0] for r in tr])
    Hho = feats_seq(system, tok, [r[0] for r in ho])
    print(f"[feat] seq shape {tuple(Htr.shape)} (M axis PRESERVED, D158)",
          flush=True)

    Ptr = torch.tensor([prog_ids(r[1]) for r in tr])
    Pho = torch.tensor([prog_ids(r[1]) for r in ho])
    Xtr = torch.tensor([digs(r[2]) for r in tr])
    Xho = torch.tensor([digs(r[2]) for r in ho])
    Ytr = torch.tensor([digs(r[3]) for r in tr])
    Yho = torch.tensor([digs(r[3]) for r in ho])

    # floor
    fp = torch.stack([Ytr[:, p].bincount(minlength=ND).argmax()
                      for p in range(NPOS)])
    floor = em(fp.unsqueeze(0).expand(len(ho), -1), Yho)
    print(f"[floor] unigram EM = {floor:.6f}", flush=True)

    # ---- positive control: exact algebra, GT operators
    pc = em(apply_tripartite(Pho, Xho), Yho)
    print(f"[positive control] exact algebra + GT operators EM = {pc:.6f}",
          flush=True)

    print(f"[F flat_seq] {a.steps} steps ...", flush=True)
    f_tr, f_ho = fit_flat(Htr, Ytr, Hho, Yho, a.hidden, a.steps, a.lr, seed=0)
    print(f"[F] trainEM={f_tr:.4f} heldEM={f_ho:.4f}", flush=True)

    print(f"[T tripartite] {a.steps} steps ...", flush=True)
    t_tr, t_ho, t_pa, t_ia = fit_tri(Htr, Ptr, Xtr, Ytr, Hho, Pho, Xho, Yho,
                                     a.hidden, a.steps, a.lr, passes=1, seed=0)
    print(f"[T] trainEM={t_tr:.4f} heldEM={t_ho:.4f} prog_acc={t_pa:.4f} "
          f"inp_acc={t_ia:.4f}", flush=True)

    print(f"[R resonator] {a.steps} steps, 2 passes ...", flush=True)
    r_tr, r_ho, r_pa, r_ia = fit_tri(Htr, Ptr, Xtr, Ytr, Hho, Pho, Xho, Yho,
                                     a.hidden, a.steps, a.lr, passes=2, seed=0)
    print(f"[R] trainEM={r_tr:.4f} heldEM={r_ho:.4f} prog_acc={r_pa:.4f} "
          f"inp_acc={r_ia:.4f}", flush=True)

    print(f"[D control] shuffled program pairing ...", flush=True)
    d_tr, d_ho, d_pa, d_ia = fit_tri(Htr, Ptr, Xtr, Ytr, Hho, Pho, Xho, Yho,
                                     a.hidden, a.steps, a.lr, passes=1,
                                     shuffle=True, seed=0)
    print(f"[D] trainEM={d_tr:.4f} heldEM={d_ho:.4f}", flush=True)

    best = max((t_ho, "T_tripartite"), (r_ho, "R_resonator"))
    d_ba = best[0] - f_ho
    guards = {
        "positive_control_ok": pc >= 0.999,
        "flat_fits_train": f_tr > 0.50,
        "control_below_real": d_ho < best[0],
    }
    if not all(guards.values()):
        verdict = "VACUOUS"
        why = f"guards {guards}"
    elif d_ba >= 0.05:
        verdict = f"FACTORIZED_COMPOSES ({best[1]})"
        why = f"best {best[1]} heldEM={best[0]:.6f} vs flat {f_ho:.6f} (d={d_ba:+.6f})"
    elif d_ba <= -0.05:
        verdict = "FACTORIZED_HARMS"
        why = f"d={d_ba:+.6f}"
    else:
        verdict = "NO_DIFFERENCE"
        why = f"d={d_ba:+.6f}; floor={floor:.6f}"

    out = {
        "schema": "henri.tripartite.resonator.v1",
        "verdict": verdict, "why": why, "guards": guards,
        "floor": floor, "positive_control": pc,
        "arms": {
            "F_flat_seq": {"train_em": f_tr, "held_em": f_ho},
            "T_tripartite": {"train_em": t_tr, "held_em": t_ho,
                             "program_acc": t_pa, "input_acc": t_ia},
            "R_resonator": {"train_em": r_tr, "held_em": r_ho,
                            "program_acc": r_pa, "input_acc": r_ia},
            "D_shuffled": {"train_em": d_tr, "held_em": d_ho},
        },
        "delta_best_minus_flat": d_ba,
        "pooling": {"n_mem": pk.n_mem, "d_k": pk.d_k, "n_macro": pk.n_macro},
        "run": {"steps": a.steps, "dk_target": a.dk_target, "pin": PIN,
                "hidden": a.hidden},
        "seconds": round(time.time() - t0, 2),
        "defects": {
            "D148": "the first gap#2 attempt mean-pooled the macro-token "
                    "sequence; arm A could not fit 48 rows. No verdict taken.",
            "D158": "this version PRESERVES the [B,M,d_model] sequence and "
                    "aggregates with a LEARNED attention plus explicit position "
                    "embeddings, so token order reaches the operator head.",
        },
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in ("verdict", "why", "guards", "arms")},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
