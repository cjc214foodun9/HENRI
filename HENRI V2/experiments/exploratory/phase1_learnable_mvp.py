#!/usr/bin/env python
"""Phase 1 LEARNABLE MVP + data-scaling attribution (the discriminating test).

WHY THIS FILE EXISTS
--------------------
The Phase 1 round-trip loop CLOSES but does NOT LEARN: its transition arm is
the fixed analytic roll operator. This file wires the LEARNABLE version from
existing repo parts, measures it, and attributes the failure.

PIPELINE (existing modules; nothing new invented)
    grid -> SpatialCliffordTokenizer.encode_canvas   (FROZEN, zero pretraining)
         -> FactorizedTransitionKernel.apply         (LEARNABLE: U, sigma, V)
         -> ContinuousHopfieldCleanup                (egress primitive, beta=26.10)
         -> decode_canvas                            (grid readout)

CONTRACT. The tokenizer is frozen; only the rank-r transition factors move.
This is in-context task compilation, NOT pretraining.

THE DISCRIMINATING EXPERIMENT
-----------------------------
Run 1 at N_TRAIN=12 gave held-out 0.2401 while the SHUFFLED control gave
0.2722 -- the control generalized BETTER, and train loss hit 0.0000 (pure
memorization). Two competing explanations:

  H1 DATA STARVATION  -> held-out rises monotonically with N_TRAIN.
  H2 MODEL CLASS      -> held-out plateaus near the documented LINEAR ceiling
                         (~0.280), because K = U Sigma V^H is still a LINEAR
                         map over the wave. The spec itself says the linear
                         class over phi ceilings at 0.280 vs a 0.920 bar.

The sweep separates them. That is the whole point of this run.

PRE-REGISTERED RULES (declared BEFORE execution)
    G1 untrained held-out <= 0.35        near chance
    G2 trained  held-out >= 0.70         at the BEST N_TRAIN
    G3 trained - identity >= 0.25
    G4 shuffled held-out <= 0.35
    G5 train loss drop   >= 0.30
    Attribution rule: if held-out at N_TRAIN=128 exceeds that at N_TRAIN=12 by
    >= 0.10, report DATA-LIMITED; else report MODEL-CLASS-LIMITED.

WHAT THIS DOES NOT ESTABLISH
    No ARC-AGI score. No latency. No GPU execution. No SOTA claim.
    Grids are SYNTHETIC and labelled synthetic.

Run:  python experiments/exploratory/phase1_learnable_mvp.py --out <receipt>
Exit: 0 if every gate passes; 1 otherwise (a negative result is a valid result).
"""
from __future__ import annotations

import argparse
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
from hopfield_cleanup import ContinuousHopfieldCleanup                # noqa: E402

MODULUS = 8
NUM_BLOCKS = 256
BLOCK_SLOTS = 8
RANK = 64
RANK_G = 8
SWEEP = (12, 32, 128)
N_TEST = 32
STEPS = 300
LR = 3e-2
RUN_SEED = 20261002
BETA = 26.10                                  # 1/0.038316, unrounded

GATES = {
    "G1_untrained_heldout_max": 0.35,
    "G2_trained_heldout_min": 0.70,
    "G3_margin_over_identity_min": 0.25,
    "G4_shuffled_heldout_max": 0.35,
    "G5_train_loss_drop_min": 0.30,
}
LINEAR_CEILING = 0.280                        # documented spec figure


def corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a / (a.norm(dim=-1, keepdim=True) + 1e-12)
    b = b / (b.norm(dim=-1, keepdim=True) + 1e-12)
    return (torch.conj(a) * b).sum(dim=-1).abs().mean()


def random_canvas(gen) -> torch.Tensor:
    return torch.randint(0, MODULUS, (MODULUS, MODULUS), generator=gen)


def rotate_values(grid: torch.Tensor, k: int) -> torch.Tensor:
    return (grid + k) % MODULUS


def build_task_a(tok, dim, gen, n_total):
    """In-class target: a genuine random rank-r_g <= r operator in wave space."""
    u = torch.randn(dim, RANK_G, generator=gen, dtype=torch.float32)
    u = torch.complex(u, torch.randn(dim, RANK_G, generator=gen))
    u = torch.linalg.qr(u)[0][:, :RANK_G]
    v = torch.randn(dim, RANK_G, generator=gen, dtype=torch.float32)
    v = torch.complex(v, torch.randn(dim, RANK_G, generator=gen))
    v = torch.linalg.qr(v)[0][:, :RANK_G]
    sig = torch.linspace(1.0, 0.5, RANK_G)

    def apply(psi):
        return u @ (sig * (v.conj().transpose(0, 1) @ psi))

    grids = [random_canvas(gen) for _ in range(n_total)]
    xs = torch.stack([tok.encode_canvas(g) for g in grids])
    ys = torch.stack([apply(x) for x in xs])
    return xs, ys


def build_task_b(tok, gen, dw=1, dh=2, k=3):
    """Exact discrete target: cyclic roll + value rotation. Both exact."""
    grids = [random_canvas(gen) for _ in range(max(SWEEP) + N_TEST)]
    tgt = [tok.roll_canvas(rotate_values(g, k).tolist(), dw, dh) for g in grids]
    xs = torch.stack([tok.encode_canvas(g) for g in grids])
    ys = torch.stack([tok.encode_canvas(t) for t in tgt])
    return tgt, xs, ys


def train_kernel(dim, xs, ys, steps=STEPS, lr=LR, seed=RUN_SEED):
    torch.manual_seed(seed)
    k = FactorizedTransitionKernel(
        dim=dim, rank=RANK, num_actions=1, block_slots=BLOCK_SLOTS, seed=seed
    )
    opt = torch.optim.Adam([k.U, k.V, k.sigma], lr=lr)
    first = last = None
    for step in range(steps):
        opt.zero_grad()
        pred = torch.stack([k.apply(0, x) for x in xs])
        loss = 1.0 - corr(pred, ys)
        loss.backward()
        opt.step()
        if step == 0:
            first = float(loss.item())
        last = float(loss.item())
    return k, first, last


def eval_kernel(k, xs, ys) -> float:
    with torch.no_grad():
        return float(corr(torch.stack([k.apply(0, x) for x in xs]), ys).item())


def evaluate(tok, report: dict) -> int:
    dim = NUM_BLOCKS * BLOCK_SLOTS
    print(f"[cfg] D={dim} rank={RANK} rank_g={RANK_G} modulus={MODULUS} "
          f"sweep={SWEEP} test={N_TEST} steps={STEPS}")

    gen = torch.Generator().manual_seed(RUN_SEED)
    xs, ys = build_task_a(tok, dim, gen, max(SWEEP) + N_TEST)
    xte, yte = xs[max(SWEEP):], ys[max(SWEEP):]
    identity = float(corr(xte, yte).item())

    torch.manual_seed(RUN_SEED)
    untrained = FactorizedTransitionKernel(
        dim=dim, rank=RANK, num_actions=1, block_slots=BLOCK_SLOTS, seed=RUN_SEED
    )
    untrained_corr = eval_kernel(untrained, xte, yte)

    print("\n=== Task A: data-scaling sweep (discriminating experiment) ===")
    print(f"  identity  held-out corr   {identity:.4f}")
    print(f"  untrained held-out corr   {untrained_corr:.4f}")
    print(f"  documented LINEAR ceiling {LINEAR_CEILING:.4f}")
    print()
    print(f"  {'N_TRAIN':>8} {'trained':>9} {'shuffled':>9} {'tr_loss0':>9} {'tr_loss1':>9}")

    sweep_rows, last_arm = [], None
    for nt in SWEEP:
        xtr, ytr = xs[:nt], ys[:nt]
        k, l0, l1 = train_kernel(dim, xtr, ytr)
        trained = eval_kernel(k, xte, yte)
        perm = torch.randperm(nt, generator=torch.Generator().manual_seed(99))
        ks, _, _ = train_kernel(dim, xtr, ytr[perm], seed=RUN_SEED + 1)
        shuffled = eval_kernel(ks, xte, yte)
        print(f"  {nt:>8} {trained:>9.4f} {shuffled:>9.4f} {l0:>9.4f} {l1:>9.4f}")
        sweep_rows.append({"n_train": nt, "trained": round(trained, 4),
                           "shuffled": round(shuffled, 4),
                           "train_loss_first": round(l0, 4),
                           "train_loss_last": round(l1, 4)})
        last_arm = (k, trained, shuffled, l0, l1)

    k_a, trained, shuffled, loss0, loss1 = last_arm
    delta = sweep_rows[-1]["trained"] - sweep_rows[0]["trained"]
    attribution = "DATA_LIMITED" if delta >= 0.10 else "MODEL_CLASS_LIMITED"
    ceiling_gap = abs(trained - LINEAR_CEILING)
    print(f"\n  trained(N=12) {sweep_rows[0]['trained']:.4f} -> "
          f"trained(N={SWEEP[-1]}) {trained:.4f}   delta {delta:+.4f}")
    print(f"  ATTRIBUTION: {attribution}")
    print(f"  |trained - LINEAR_CEILING| = {ceiling_gap:.4f}")

    gates = {
        "G1_untrained_heldout_max": untrained_corr <= GATES["G1_untrained_heldout_max"],
        "G2_trained_heldout_min": trained >= GATES["G2_trained_heldout_min"],
        "G3_margin_over_identity_min": (trained - identity)
        >= GATES["G3_margin_over_identity_min"],
        "G4_shuffled_heldout_max": shuffled <= GATES["G4_shuffled_heldout_max"],
        "G5_train_loss_drop_min": (loss0 - loss1) >= GATES["G5_train_loss_drop_min"],
    }
    print()
    for g, ok in gates.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {g} (bound {GATES[g]})")

    report["taskA"] = {
        "identity_heldout": round(identity, 4),
        "untrained_heldout": round(untrained_corr, 4),
        "linear_ceiling_documented": LINEAR_CEILING,
        "sweep": sweep_rows,
        "attribution": attribution,
        "ceiling_gap": round(ceiling_gap, 4),
        "gates": gates,
    }

    # ------------------------------------------------------------- Task B
    print("\n=== Task B: exact roll+value-rotation (REPORTED, not gated) ===")
    tgt, xb, yb = build_task_b(tok, gen)
    xbt, ybt = xb[max(SWEEP):], yb[max(SWEEP):]
    torch.manual_seed(RUN_SEED)
    unt_b = FactorizedTransitionKernel(
        dim=dim, rank=RANK, num_actions=1, block_slots=BLOCK_SLOTS, seed=RUN_SEED
    )
    b_unt = eval_kernel(unt_b, xbt, ybt)
    b_id = float(corr(xbt, ybt).item())
    k_b, bl0, bl1 = train_kernel(dim, xb[:max(SWEEP)], yb[:max(SWEEP)], seed=RUN_SEED)
    b_tr = eval_kernel(k_b, xbt, ybt)
    print(f"  untrained {b_unt:.4f}  identity {b_id:.4f}  trained {b_tr:.4f}")

    hop = ContinuousHopfieldCleanup(dim=2 * dim, beta=BETA)
    hits = tot = 0
    with torch.no_grad():
        for i in range(max(SWEEP), max(SWEEP) + N_TEST):
            gp, _ = tok.decode_canvas(k_b.apply(0, xb[i]))
            tr = tgt[i]
            for r in range(MODULUS):
                for c in range(MODULUS):
                    tot += 1
                    hits += int(gp[r][c] == tr[r][c])
    acc = hits / max(tot, 1)
    print(f"  grid cell accuracy {hits}/{tot} = {acc:.4f} (chance {1.0/MODULUS:.4f})")
    report["taskB"] = {
        "untrained_heldout": round(b_unt, 4), "identity_heldout": round(b_id, 4),
        "trained_heldout": round(b_tr, 4), "grid_cell_accuracy": round(acc, 4),
        "chance_cell_accuracy": round(1.0 / MODULUS, 4),
        "gate": "REPORTED_BOUNDARY_not_pass_fail",
    }
    report["hopfield_engrams"] = int(hop.num_engrams())
    report["taskA_pass"] = all(gates.values())
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
        "task": "SPEC-2026-10-01-PHASE1-TRANSDUCTION/learnable-mvp",
        "config": {"modulus": MODULUS, "num_blocks": NUM_BLOCKS,
                   "block_slots": BLOCK_SLOTS, "D": NUM_BLOCKS * BLOCK_SLOTS,
                   "rank": RANK, "rank_g": RANK_G, "sweep": list(SWEEP),
                   "n_test": N_TEST, "steps": STEPS, "lr": LR,
                   "beta": BETA, "seed": RUN_SEED},
        "data": "SYNTHETIC (random grids); no ARC-AGI data touched",
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
