#!/usr/bin/env python
"""Multi-seed confirmation of the ONE positive learning result.

CLAIM UNDER TEST
    A full-rank DIAGONAL transition learns the exact roll+value-rotation
    task from 64 in-context support pairs; a rank-64 FACTORIZED transition
    does not. Single-seed evidence (seed 20261002):

        FACTORIZED r=64     tr 0.0583  shuf 0.0515  margin +0.0067
        DIAGONAL full-rank  tr 0.7521  shuf 0.1438  margin +0.6083

WHY THIS FILE EXISTS
    A single seed is an anecdote. The load-bearing claim of this MVP is that
    the DIAGONAL arm generalizes and the FACTORIZED arm does not. That claim
    must survive seed change or it is not a finding.

PARAMETER ARITHMETIC (why this is not memorization)
    FACTORIZED r=64:  2 * D * r * 2 + r = 524,352 params, N_TRAIN=64
    DIAGONAL:        2 * D            =   4,096 params, N_TRAIN=64
    The over-parameterized arm CAN memorize (train loss -> 0). The
    under-parameterized arm CANNOT (train loss floors). Only the arm that
    cannot memorize generalizes. That is the discriminating structure.

PRE-REGISTERED RULE
    DIAGONAL margin (trained - shuffled) > 0.30 in EVERY seed  -> CONFIRMED
    Otherwise                                                   -> NOT CONFIRMED

Run: python experiments/exploratory/phase1_multiseed_confirm.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri.ingress.spatial_tokenizer import SpatialCliffordTokenizer  # noqa: E402
from factorized_transition_kernel import FactorizedTransitionKernel   # noqa: E402
sys.path.insert(0, str(_HERE.parent))
from phase1_capability_gap_mvp import (  # noqa: E402
    DiagonalTransition, DIM, MODULUS, NUM_BLOCKS, BLOCK_SLOTS, N_TRAIN, N_TEST,
    STEPS, LR, corr, random_canvas, rotate_values,
)

SEEDS = (20261002, 7, 424242)
MARGIN_MIN = 0.30


def build_task_d(tok, gen, dw=1, dh=2, k=3):
    n = N_TRAIN + N_TEST
    grids = [random_canvas(gen) for _ in range(n)]
    tgt = [tok.roll_canvas(rotate_values(g, k).tolist(), dw, dh) for g in grids]
    xs = torch.stack([tok.encode_canvas(g) for g in grids])
    ys = torch.stack([tok.encode_canvas(t) for t in tgt])
    return tgt, xs, ys


def train_diag(seed, xs, ys, steps=STEPS, lr=LR):
    torch.manual_seed(seed)
    arm = DiagonalTransition(DIM, seed)
    opt = torch.optim.Adam([arm.m_re, arm.m_im], lr=lr)
    l0 = l1 = None
    for step in range(steps):
        opt.zero_grad()
        loss = 1.0 - corr(arm.apply_batch(xs), ys)
        loss.backward()
        opt.step()
        if step == 0:
            l0 = float(loss.item())
        l1 = float(loss.item())
    return arm, l0, l1


def train_fact(seed, xs, ys, steps=STEPS, lr=LR):
    torch.manual_seed(seed)
    k = FactorizedTransitionKernel(
        dim=DIM, rank=64, num_actions=1, block_slots=BLOCK_SLOTS, seed=seed
    )
    opt = torch.optim.Adam([k.U, k.V, k.sigma], lr=lr)
    l0 = l1 = None
    for step in range(steps):
        opt.zero_grad()
        pred = k.apply_batch(torch.zeros(xs.shape[0], dtype=torch.long), xs)
        loss = 1.0 - corr(pred, ys)
        loss.backward()
        opt.step()
        if step == 0:
            l0 = float(loss.item())
        l1 = float(loss.item())
    return k, l0, l1


def main() -> int:
    rows, ok_all = [], True
    for seed in SEEDS:
        tok = SpatialCliffordTokenizer(
            num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS, modulus=MODULUS, seed=seed
        )
        gen = torch.Generator().manual_seed(seed)
        _tgt, xs, ys = build_task_d(tok, gen)
        xtr, ytr, xte, yte = xs[:N_TRAIN], ys[:N_TRAIN], xs[N_TRAIN:], ys[N_TRAIN:]
        perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(seed + 1))

        d, dl0, dl1 = train_diag(seed, xtr, ytr)
        with torch.no_grad():
            d_tr = float(corr(d.apply_batch(xte), yte).item())
        ds, _, _ = train_diag(seed + 1, xtr, ytr[perm])
        with torch.no_grad():
            d_sh = float(corr(ds.apply_batch(xte), yte).item())

        f, fl0, fl1 = train_fact(seed, xtr, ytr)
        with torch.no_grad():
            f_tr = float(corr(f.apply_batch(torch.zeros(xte.shape[0], dtype=torch.long),
                                            xte), yte).item())

        margin = d_tr - d_sh
        ok = margin > MARGIN_MIN
        ok_all &= ok
        rows.append({"seed": seed, "diag_trained": round(d_tr, 4),
                     "diag_shuffled": round(d_sh, 4), "diag_margin": round(margin, 4),
                     "diag_loss_first": round(dl0, 4), "diag_loss_last": round(dl1, 4),
                     "fact_trained": round(f_tr, 4),
                     "fact_loss_first": round(fl0, 4), "fact_loss_last": round(fl1, 4),
                     "margin_ok": ok})
        print(f"seed {seed:>8}  DIAG tr {d_tr:.4f} shuf {d_sh:.4f} "
              f"margin {margin:+.4f} [{'OK' if ok else 'FAIL'}]   "
              f"loss {dl0:.3f}->{dl1:.3f} | FACT tr {f_tr:.4f} "
              f"loss {fl0:.3f}->{fl1:.3f}")

    verdict = "CONFIRMED" if ok_all else "NOT_CONFIRMED"
    print(f"\nVERDICT: {verdict}  (rule: DIAGONAL margin > {MARGIN_MIN} in every seed)")
    out = {"task": "SPEC-2026-10-01-PHASE1-TRANSDUCTION/multiseed-confirm",
           "rule": f"diag margin > {MARGIN_MIN} every seed",
           "seeds": list(SEEDS), "rows": rows, "verdict": verdict,
           "out": ""}
    dest = Path(__file__).with_name("phase1_multiseed_confirm_receipt.json")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[receipt] {dest}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
