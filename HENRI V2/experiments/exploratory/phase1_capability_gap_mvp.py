#!/usr/bin/env python
"""Phase 1 CAPABILITY-GAP MVP: which transition parameterization can work?

WHY THIS FILE EXISTS
--------------------
The wired MVP (phase1_learnable_mvp.py) closed the loop and LEARNED nothing
useful. Measured, D=2048, rank=64:

    N_TRAIN   trained  shuffled
         12    0.3079    0.3529
         32    0.3212    0.3363
        128    0.3131    0.3137     <- trained ~= shuffled
    delta +0.0052 over 10x data | trained - documented linear ceiling = 0.0331

Two readings compete:
  (a) MODEL CLASS: K = U Sigma V^H is LINEAR; the spec says that class
      ceilings at 0.280. Held-out 0.3131 is within 0.033 of that number.
  (b) PARAMETERIZATION: not the model class but the FACTORIZATION is wrong
      for the target's structure.

This file separates them. Three transition arms, two tasks, both controls.

THE STRUCTURAL ARGUMENT BEING TESTED
------------------------------------
Exact roll + value rotation is, in this tokenizer, an ELEMENTWISE operation:
    encode(roll(rotate(g, k), dw, dh)) = d (*) encode(g),   d in C^D fixed.
roll_multiplier is exp(i(dw*wx + dh*wy)) per (block, slot); value rotation is
exp(i*k*w) per (block, slot); both are diagonal in the (block, slot) basis.
So the target is a FULL-RANK DIAGONAL operator with a FLAT spectrum (|d_i| = 1).

A rank-r factorization cannot represent that. A flat-spectrum diagonal of
length D has rank D; truncating at r=64 keeps ~64/2048 = 3% of the energy.
A DIAGONAL parameterization represents it exactly with O(D) parameters.

That is a falsifiable prediction, not an opinion. Hence the gates below.

ARMS
  FACTORIZED  K = U Sigma V^H, rank 64 (the existing repo kernel). O(r D).
  DIAGONAL    y = m (*) x, m in C^D learned. Full rank, O(D) params.
  NONLINEAR   y = U (.) sigma (.) modReLU_b(V^H x). rank-64 bottleneck plus a
              magnitude non-linearity INSIDE the factorization.

TASKS
  Task D  exact discrete: cyclic roll (1,2) + value rotation (k=3). DIAGONAL.
  Task N  non-linear target: rank-8 factorizer wrapped in a GATED magnitude
          non-linearity (modReLU with a fixed negative bias). LINEAR arms
          cannot express the gate; NONLINEAR can.

PRE-REGISTERED GATES (declared BEFORE execution)
    D1  DIAGONAL  on Task D held-out corr  >= 0.90   can represent the target
    D2  FACTORIZED on Task D held-out corr <= 0.40   rank-64 cannot
    D3  D1 - D2                            >= 0.40   the gap is real
    N1  NONLINEAR on Task N held-out corr  >= 0.65   gate is expressible
    N2  FACTORIZED on Task N held-out corr <= 0.50   linear cannot gate
    N3  N1 - N2                            >= 0.20
    C1  each winning arm beats its own SHUFFLED control by >= 0.20

C1 is the control that makes "it learned" a measurement rather than a
tautology. The prior gate design compared each arm to an absolute bound and
therefore missed that trained == shuffled.

WHAT THIS DOES NOT ESTABLISH
    No ARC-AGI score. No latency. No GPU execution. No SOTA claim.
    Grids are SYNTHETIC and labelled synthetic. This is in-context task
    compilation over FROZEN tokenizer codes, not pretraining.

Run:  python experiments/exploratory/phase1_capability_gap_mvp.py --out <json>
Exit: 0 if every gate passes; 1 otherwise. A negative is a valid result.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri.ingress.spatial_tokenizer import SpatialCliffordTokenizer  # noqa: E402
from factorized_transition_kernel import FactorizedTransitionKernel   # noqa: E402
from hopfield_cleanup import ContinuousHopfieldCleanup                # noqa: E402

MODULUS = 8
NUM_BLOCKS = 256
BLOCK_SLOTS = 8
DIM = NUM_BLOCKS * BLOCK_SLOTS          # 2048
RANK = 64
RANK_G = 8
N_TRAIN = 64
N_TEST = 32
STEPS = 400
LR = 3e-2
RUN_SEED = 20261002
BETA = 26.10                            # 1/0.038316, unrounded
NONLIN_BIAS = -0.6                      # multiplier on mean|h| for Task N

GATES = {
    "D1_diagonal_taskD_min": 0.90,
    "D2_factorized_taskD_max": 0.40,
    "D3_taskD_gap_min": 0.40,
    "N1_nonlinear_taskN_min": 0.65,
    "N2_factorized_taskN_max": 0.50,
    "N3_taskN_gap_min": 0.20,
    "C1_winning_arm_minus_shuffled_min": 0.20,
}


# ------------------------------------------------------------------ helpers
def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (torch.conj(a) * b).sum(dim=-1).abs().mean()


def random_canvas(gen) -> torch.Tensor:
    return torch.randint(0, MODULUS, (MODULUS, MODULUS), generator=gen)


def rotate_values(grid: torch.Tensor, k: int) -> torch.Tensor:
    return (grid + k) % MODULUS


def _orth(dim: int, r: int, gen) -> torch.Tensor:
    z = torch.complex(torch.randn(dim, r, generator=gen),
                      torch.randn(dim, r, generator=gen))
    q, _ = torch.linalg.qr(z)
    return q[:, :r]


# --------------------------------------------------------------------- arms
class DiagonalTransition(nn.Module):
    """y = m (*) x. Full rank, O(D) parameters. Can represent roll+rotate."""

    def __init__(self, dim: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.m_re = nn.Parameter(torch.ones(dim) + 0.01 * torch.randn(dim, generator=g))
        self.m_im = nn.Parameter(0.01 * torch.randn(dim, generator=g))

    def matrix(self) -> torch.Tensor:
        return torch.complex(self.m_re, self.m_im)

    def apply_batch(self, x: torch.Tensor) -> torch.Tensor:
        return self.matrix().unsqueeze(0) * x


class NonlinearSlotTransition(nn.Module):
    """y = U (sigma (*) modReLU_b(V^H x)); the non-linearity sits INSIDE."""

    def __init__(self, dim: int, rank: int, seed: int = 0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        s = 1.0 / math.sqrt(dim * rank)
        self.U = nn.Parameter(torch.randn(dim, rank, 2, generator=g) * s)
        self.V = nn.Parameter(torch.randn(dim, rank, 2, generator=g) * s)
        self.sigma = nn.Parameter(torch.ones(rank))
        self.b = nn.Parameter(torch.zeros(rank))

    def apply_batch(self, x: torch.Tensor) -> torch.Tensor:
        U = torch.view_as_complex(self.U.contiguous())
        V = torch.view_as_complex(self.V.contiguous())
        h = x @ V                                    # [n, r]
        z = self.sigma.unsqueeze(0) * h              # [n, r]
        mag = z.abs() + 1e-12
        g = F.relu(mag + self.b.unsqueeze(0))        # magnitude gate
        h2 = g * z / mag                               # phase preserved
        return (U @ h2.transpose(0, 1)).transpose(0, 1)


def make_factorized(seed: int) -> FactorizedTransitionKernel:
    return FactorizedTransitionKernel(
        dim=DIM, rank=RANK, num_actions=1, block_slots=BLOCK_SLOTS, seed=seed
    )


# -------------------------------------------------------------------- tasks
def build_task_d(tok, gen, dw=1, dh=2, k=3):
    """Exact discrete: cyclic roll + value rotation. Target is diagonal in
    the (block, slot) basis -> full rank, flat spectrum."""
    n = N_TRAIN + N_TEST
    grids = [random_canvas(gen) for _ in range(n)]
    tgt = [tok.roll_canvas(rotate_values(g, k).tolist(), dw, dh) for g in grids]
    xs = torch.stack([tok.encode_canvas(g) for g in grids])
    ys = torch.stack([tok.encode_canvas(t) for t in tgt])
    return tgt, xs, ys


def build_task_n(dim, gen):
    """Non-linear target: rank-8 factorizer with a GATED magnitude non-linearity."""
    u = _orth(dim, RANK_G, gen)
    v = _orth(dim, RANK_G, gen)
    sig = torch.linspace(1.0, 0.5, RANK_G)
    n = N_TRAIN + N_TEST
    xs = torch.stack([
        torch.complex(torch.randn(dim, generator=gen), torch.randn(dim, generator=gen))
        for _ in range(n)
    ])
    xs = xs / (xs.norm(dim=-1, keepdim=True) + 1e-12)
    h = xs @ v                                   # [n, RANK_G]
    mag = h.abs() + 1e-12
    b = NONLIN_BIAS * float(mag.mean())
    g = F.relu(mag + b) * h / mag                # non-linear gate
    ys = (u @ (sig.unsqueeze(1) * g.transpose(0, 1))).transpose(0, 1)
    return xs, ys, b


# ----------------------------------------------------------------- training
def train_arm(arm, kind, xs, ys, seed, steps=STEPS, lr=LR):
    torch.manual_seed(seed)
    if kind == "factorized":
        params = [arm.U, arm.V, arm.sigma]
        forward = lambda x: arm.apply_batch(torch.zeros(x.shape[0], dtype=torch.long), x)
    elif kind == "diagonal":
        params = [arm.m_re, arm.m_im]
        forward = arm.apply_batch
    else:
        params = [arm.U, arm.V, arm.sigma, arm.b]
        forward = arm.apply_batch
    opt = torch.optim.Adam(params, lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        loss = 1.0 - corr(forward(xs), ys)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    return first, last


def forward_arm(arm, kind, x):
    if kind == "factorized":
        return arm.apply_batch(torch.zeros(x.shape[0], dtype=torch.long), x)
    return arm.apply_batch(x)


def eval_arm(arm, kind, xs, ys) -> float:
    with torch.no_grad():
        return float(corr(forward_arm(arm, kind, xs), ys).item())


def run_arm(kind, xs, ys, seed, name, rows):
    xtr, ytr, xte, yte = xs[:N_TRAIN], ys[:N_TRAIN], xs[N_TRAIN:], ys[N_TRAIN:]
    if kind == "factorized":
        arm = make_factorized(seed)
    elif kind == "diagonal":
        arm = DiagonalTransition(DIM, seed)
    else:
        arm = NonlinearSlotTransition(DIM, RANK, seed)
    untrained = eval_arm(arm, kind, xte, yte)
    l0, l1 = train_arm(arm, kind, xtr, ytr, seed)
    trained = eval_arm(arm, kind, xte, yte)
    perm = torch.randperm(N_TRAIN, generator=torch.Generator().manual_seed(99))
    if kind == "factorized":
        arm_s = make_factorized(seed + 1)
    elif kind == "diagonal":
        arm_s = DiagonalTransition(DIM, seed + 1)
    else:
        arm_s = NonlinearSlotTransition(DIM, RANK, seed + 1)
    train_arm(arm_s, kind, xtr, ytr[perm], seed + 1)
    shuffled = eval_arm(arm_s, kind, xte, yte)
    row = {"arm": name, "untrained": round(untrained, 4),
           "trained": round(trained, 4), "shuffled": round(shuffled, 4),
           "train_loss_first": round(l0, 4), "train_loss_last": round(l1, 4),
           "trained_minus_shuffled": round(trained - shuffled, 4)}
    print(f"  {name:<28} untr {untrained:>7.4f}  tr {trained:>7.4f}  "
          f"shuf {shuffled:>7.4f}  tr-shuf {trained - shuffled:>+7.4f}  "
          f"loss {l0:.3f}->{l1:.3f}")
    rows.append(row)
    return arm, row


def evaluate(tok, report: dict) -> int:
    gen = torch.Generator().manual_seed(RUN_SEED)
    print(f"[cfg] D={DIM} rank={RANK} rank_g={RANK_G} modulus={MODULUS} "
          f"train={N_TRAIN} test={N_TEST} steps={STEPS}")

    # ---------------------------------------------------------- Task D
    print("\n=== Task D: exact roll+rotate (diagonal in (block, slot)) ===")
    tgtD, xD, yD = build_task_d(tok, gen)
    idD = float(corr(xD[N_TRAIN:], yD[N_TRAIN:]).item())
    print(f"  identity baseline held-out {idD:.4f}")
    rowsD = []
    arm_fD, r_fD = run_arm("factorized", xD, yD, RUN_SEED, "FACTORIZED r=64", rowsD)
    arm_dD, r_dD = run_arm("diagonal", xD, yD, RUN_SEED, "DIAGONAL full-rank", rowsD)

    # end-to-end grid readout with the Task D winner
    hop = ContinuousHopfieldCleanup(dim=2 * DIM, beta=BETA)
    hits = tot = 0
    with torch.no_grad():
        for i in range(N_TRAIN, N_TRAIN + N_TEST):
            pred = arm_dD.apply_batch(xD[i:i + 1])[0]
            gp, _ = tok.decode_canvas(pred)
            tr = tgtD[i]
            for r in range(MODULUS):
                for c in range(MODULUS):
                    tot += 1
                    hits += int(gp[r][c] == tr[r][c])
    accD = hits / max(tot, 1)
    print(f"  DIAGONAL end-to-end grid accuracy {hits}/{tot} = {accD:.4f} "
          f"(chance {1.0/MODULUS:.4f})")

    # ---------------------------------------------------------- Task N
    print("\n=== Task N: gated non-linear target (modReLU inside) ===")
    xN, yN, nb = build_task_n(DIM, gen)
    idN = float(corr(xN[N_TRAIN:], yN[N_TRAIN:]).item())
    print(f"  identity baseline held-out {idN:.4f}   gate bias {nb:.4f}")
    rowsN = []
    arm_fN, r_fN = run_arm("factorized", xN, yN, RUN_SEED, "FACTORIZED r=64", rowsN)
    arm_nN, r_nN = run_arm("nonlinear", xN, yN, RUN_SEED, "NONLINEAR r=64+gate", rowsN)

    # ------------------------------------------------------------- gates
    gates = {
        "D1_diagonal_taskD_min": r_dD["trained"] >= GATES["D1_diagonal_taskD_min"],
        "D2_factorized_taskD_max": r_fD["trained"] <= GATES["D2_factorized_taskD_max"],
        "D3_taskD_gap_min": (r_dD["trained"] - r_fD["trained"])
        >= GATES["D3_taskD_gap_min"],
        "N1_nonlinear_taskN_min": r_nN["trained"] >= GATES["N1_nonlinear_taskN_min"],
        "N2_factorized_taskN_max": r_fN["trained"] <= GATES["N2_factorized_taskN_max"],
        "N3_taskN_gap_min": (r_nN["trained"] - r_fN["trained"])
        >= GATES["N3_taskN_gap_min"],
        "C1_winning_arm_minus_shuffled_min": (
            min(r_dD["trained_minus_shuffled"], r_nN["trained_minus_shuffled"])
            >= GATES["C1_winning_arm_minus_shuffled_min"]
        ),
    }
    print("\n=== pre-registered gates ===")
    for g, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {g} (bound {GATES[g]})")

    report.update({
        "taskD": {"identity_heldout": round(idD, 4), "arms": rowsD,
                  "diagonal_end_to_end_grid_accuracy": round(accD, 4),
                  "chance_grid_accuracy": round(1.0 / MODULUS, 4)},
        "taskN": {"identity_heldout": round(idN, 4), "gate_bias": round(nb, 4),
                  "arms": rowsN},
        "gates": gates,
        "hopfield_engrams": int(hop.num_engrams()),
        "all_gates_pass": all(gates.values()),
    })
    return 0 if all(gates.values()) else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    torch.manual_seed(RUN_SEED)
    tok = SpatialCliffordTokenizer(
        num_blocks=NUM_BLOCKS, block_slots=BLOCK_SLOTS, modulus=MODULUS, seed=RUN_SEED
    )
    report = {
        "task": "SPEC-2026-10-01-PHASE1-TRANSDUCTION/capability-gap-mvp",
        "config": {"modulus": MODULUS, "D": DIM, "rank": RANK, "rank_g": RANK_G,
                   "n_train": N_TRAIN, "n_test": N_TEST, "steps": STEPS,
                   "lr": LR, "beta": BETA, "seed": RUN_SEED,
                   "nonlin_bias_mult": NONLIN_BIAS},
        "data": "SYNTHETIC (random grids/vectors); no ARC-AGI data touched",
        "gates_declared": GATES,
    }
    t0 = time.time()
    rc = evaluate(tok, report)
    report["wall_s"] = round(time.time() - t0, 3)
    print(f"\n[rc] {rc}  wall {report['wall_s']}s")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"[receipt] {args.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
