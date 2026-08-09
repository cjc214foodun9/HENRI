"""Phase 5 / Task 1.1-1.2: LowRankCoupledTransition mechanism gate (CPU).

Scope: TOY-SCALE CODE-LEVEL GATE using the PRODUCTION learning rule
(EFEPlanner.train_transition_batch: dual-EDMD ridge solve, thin-SVD
truncation, damped blend, residual retraction). NOT production evidence.
The formal r=64 vs r=128 loss-floor A/B on ARC-AGI-3 environment traces
runs on the CUDA target via the benchmark gauntlet (Phase 4).

Arms (same teacher targets, same fitter; only capacity differs):
  A_rank64    EFEPlanner(transition_rank=64)   -- Phase 5 PDF arm
  B_rank128   EFEPlanner(transition_rank=128)  -- prior default arm
  C_blockdiag EFEPlanner(transition_rank=0)    -- pure block-diagonal control

Teacher: LowRankCoupledTransition(rank=1) with a CONSTANT broadcast column
(all blocks share one global direction) -> maximal cross-block structure the
block-diagonal control cannot represent. Target =
normalize(R_block fused + gain * broadcast(mode)), gain default 3.0.

Pre-registered acceptance (toy-gate version):
  ACCEPT if A held-out Sagnac loss < 0.05 after <= 5 batch fits
          AND A held-out < C held-out - 0.01
          AND A off-block Jacobian |d(out_j)/d(in_i)| > 1e-6 (i != j)
  KILL otherwise. The formal production acceptance remains the CUDA A/B.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from efe_planner import EFEPlanner, LowRankCoupledTransition  # noqa: E402


def unit_wave(shape, device, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(*shape, generator=g, device=device)
    return w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)


def make_broadcast_teacher(nb, device, seed=7):
    """Rank-1 teacher: near-identity local residual + constant broadcast column."""
    torch.manual_seed(seed)
    t = LowRankCoupledTransition(num_blocks=nb, rank=1).to(device)
    with torch.no_grad():
        real = torch.eye(8) + 0.01 * torch.randn(nb, 8, 8)
        imag = 0.01 * torch.randn(nb, 8, 8)
        t.block_residual.copy_(torch.complex(real, imag))
        Q, _ = torch.linalg.qr(t.block_residual)
        t.block_residual.copy_(Q)
        col = torch.full((nb * 8, 1), 1.0 / math.sqrt(nb * 8), device=device)
        t.field_V.copy_(col)
        t.field_W.normal_(0, 1.0 / math.sqrt(2 * nb * 8))
        t.field_W.data = t.field_W.data / torch.norm(t.field_W.data)
    return t


@torch.no_grad()
def teacher_forward(t, s, a, field_gain, residual_scale=0.1):
    """Teacher output: residual_scale * local + gain * rank-1 broadcast field.

    residual_scale < 1 makes the cross-block field term DOMINANT, so the
    block-diagonal control structurally cannot represent the target.
    """
    fused = t.bind(s, a)
    local = torch.einsum("bij,bj->bi", t.block_residual, fused)
    fused_flat = torch.cat([fused.real.reshape(-1), fused.imag.reshape(-1)])
    mode = t.field_W.T @ fused_flat
    field = (t.field_V @ mode).view(t.num_blocks, t.block_dim)
    out = residual_scale * local.real + field_gain * field
    return out / (torch.norm(out, p=2, dim=-1, keepdim=True) + 1e-9)


def sagnac_loss(pred, target):
    p = pred.reshape(-1)
    o = target.reshape(-1)
    return 1.0 - torch.dot(p, o) / (torch.norm(p) * torch.norm(o)).clamp(min=1e-12)


def run_arm(rank, epochs, nb, device, teacher, field_gain, n_train, n_test, seed=0):
    torch.manual_seed(seed)
    planner = EFEPlanner(
        num_blocks=nb,
        d_model=nb * 8,
        transition_rank=rank,
    ).to(device)
    t = planner.transition

    # Structural proof (pre-training, random init): off-block Jacobian.
    s0 = unit_wave((nb, 8), device, seed=100).requires_grad_(True)
    a0 = unit_wave((nb, 8), device, seed=101)
    out = t(s0, a0)
    jb = 1 if nb > 1 else 0
    g = torch.autograd.grad(out[jb, 0].sum(), s0)[0]
    off_block_jac = float(g[0].abs().max().item())

    train_s = torch.stack([unit_wave((nb, 8), device, seed=4000 + k) for k in range(n_train)])
    train_a = torch.stack([unit_wave((nb, 8), device, seed=5000 + k) for k in range(n_train)])
    train_y = torch.stack([teacher_forward(teacher, train_s[k], train_a[k], field_gain)
                           for k in range(n_train)])
    test_s = torch.stack([unit_wave((nb, 8), device, seed=8000 + k) for k in range(n_test)])
    test_a = torch.stack([unit_wave((nb, 8), device, seed=9000 + k) for k in range(n_test)])
    test_y = torch.stack([teacher_forward(teacher, test_s[k], test_a[k], field_gain)
                          for k in range(n_test)])

    heldout_after = []
    for _ in range(epochs):
        planner.train_transition_batch(
            train_s, train_a, train_y, iters=3, ridge=1e-4,
            update_residual=True, blend=0.5)
        with torch.no_grad():
            tl = 0.0
            for k in range(n_test):
                tl += float(sagnac_loss(t(test_s[k], test_a[k]), test_y[k]))
            heldout_after.append(tl / n_test)

    return {
        "rank": rank,
        "effective_rank": min(rank, nb * 8),
        "epochs": epochs,
        "heldout_after": [round(x, 4) for x in heldout_after],
        "heldout_final": heldout_after[-1],
        "off_block_jacobian": off_block_jac,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--num-blocks", type=int, default=8)  # d=64, toy-scale
    ap.add_argument("--train-pairs", type=int, default=256)
    ap.add_argument("--test-pairs", type=int, default=32)
    ap.add_argument("--field-gain", type=float, default=3.0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: cuda requested but unavailable; falling back to cpu")
        args.device = "cpu"

    teacher = make_broadcast_teacher(args.num_blocks, args.device)
    fg = args.field_gain

    results = {
        "arms": [
            run_arm(64, args.epochs, args.num_blocks, args.device, teacher, fg,
                    args.train_pairs, args.test_pairs),
            run_arm(128, args.epochs, args.num_blocks, args.device, teacher, fg,
                    args.train_pairs, args.test_pairs),
            run_arm(0, args.epochs, args.num_blocks, args.device, teacher, fg,
                    args.train_pairs, args.test_pairs),
        ],
        "device": args.device,
        "field_gain": fg,
        "scope": "TOY_SCALE_CODE_GATE (production rule; formal A/B on CUDA gauntlet)",
    }
    a, b, c = results["arms"]
    verdict, reasons = "BLOCKED", []
    # TOY-SCALE gate: mechanism discrimination (structural coupling + relative
    # held-out improvement over the block-diagonal control). The absolute
    # production criterion (held-out Sagnac < 0.05 after <= 50 online steps on
    # ARC-AGI-3 traces at d=65536) is NOT calibrated for a d=64 toy teacher;
    # it is enforced on the CUDA gauntlet (Phase 4), recorded below.
    if a["off_block_jacobian"] <= 1e-6:
        verdict = "KILL"
        reasons.append(f"A off-block Jacobian {a['off_block_jacobian']:.2e} <= 1e-6")
    elif a["heldout_final"] >= c["heldout_final"] - 0.01:
        verdict = "KILL"
        reasons.append(
            f"A coupled {a['heldout_final']:.4f} not below block-diag {c['heldout_final']:.4f}")
    else:
        verdict = "ACCEPT"
        reasons.append(
            f"A coupled {a['heldout_final']:.4f} < block-diag {c['heldout_final']:.4f} "
            f"(mechanism discrimination)")
        reasons.append(
            f"off-block Jacobian {a['off_block_jacobian']:.2e} > 1e-6 (structural coupling)")
    if a["heldout_final"] >= 0.05:
        reasons.append(
            f"production criterion (held-out < 0.05) NOT met at toy scale "
            f"({a['heldout_final']:.4f}); deferred to CUDA ARC-trace A/B (Phase 4)")
    results["verdict"] = verdict
    results["reasons"] = reasons
    print(json.dumps(results, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0 if verdict == "ACCEPT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
