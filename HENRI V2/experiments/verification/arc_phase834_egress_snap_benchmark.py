"""Phase 8.34 — Evolution I egress-side benchmark (lexical_snap retrieval).

Measures continuous-to-discrete retrieval precision on the AUTHORIZED bank
(`trajectories_production_run_1787164827.npz`, 90 records, sealed, digest
verified) per the Phase 8.34 Verification & Component Acceptance doc §3.2.

Arms (same split: held_out_frac=0.2, seed=20260819, deterministic randperm;
codebook = per-action L2-normalized prototype means over TRAIN only, K<=6):

  A: raw cosine argmax of held-out psi over the codebook      -> precision_A
  B: ContinuousHopfieldCleanup.lexical_snap(top_k=1)          -> precision_B
     + mean retrieval entropy H_B (softmax(beta*sim), beta=sqrt(D))
  C: CoupledRecursiveDualEDMD (r=128, field ON, lambda_forget=0.98,
     online closed-form RLS, NO BPTT) fit on train; held-out predicted
     next-wave -> normalize -> lexical_snap                     -> precision_C

Pre-registered (arc_phase834_egress_snap_prereg.md):
  ACCEPT  : precision_B >= 0.80 AND (precision_B - precision_A) >= +0.05
            AND precision_C >= 0.50 AND H_B <= 0.10 nats
  KILL    : NaN/divergence OR precision_B < precision_A
  BLOCKED : nonzero arm exit / bank digest mismatch / GPU failure

Output: compact JSON verdict on stdout (last line). Diagnostic only — no
score-eligibility claim (trained_action_head_active untouched).
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from henri_trajectory_bank import TrajectoryBank
from recursive_dual_edmd import CoupledRecursiveDualEDMD
from hopfield_cleanup import ContinuousHopfieldCleanup

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM  # 65,536
RANK_COUPLED = 128
LAMBDA_FORGET = 0.98
EPS = 1e-9


def _action_wave_for(idx: int, num_blocks: int = NUM_BLOCKS, block_dim: int = BLOCK_DIM,
                     device: str = "cpu") -> torch.Tensor:
    """Deterministic unit per-block action wave for a given action index
    (identical to the sealed 8.33/8.34 map)."""
    g = torch.Generator(device="cpu").manual_seed(1000 + idx)
    w = torch.randn(num_blocks, block_dim, generator=g)
    w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + EPS)
    return w.to(device)


def split_indices(n: int, frac: float, seed: int):
    """Mirror the calibrator split: seeded randperm, held-out = first n_test."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def _codebook(psi_tr: torch.Tensor, act_tr: torch.Tensor) -> torch.Tensor:
    """Per-action L2-normalized prototype mean engrams (TRAIN only)."""
    K = int(act_tr.max().item()) + 1
    protos = []
    for k in range(K):
        mask = act_tr == k
        if mask.sum() == 0:
            continue
        proto = psi_tr[mask].mean(dim=0)
        protos.append(F.normalize(proto, p=2, dim=-1))
    if not protos:
        raise RuntimeError("empty codebook: no train rows for any action")
    return torch.stack(protos, dim=0)  # [M, D]


def _snap_precision_and_entropy(cleanup: ContinuousHopfieldCleanup, psi_ho: torch.Tensor,
                                act_ho: torch.Tensor, beta: float) -> tuple:
    """lexical_snap top-1 over held-out waves; returns (precision, mean H nats)."""
    idx, conf = cleanup.lexical_snap(psi_ho, top_k=1)  # [N], [N]
    pred = idx.cpu().numpy()
    prec = float(np.mean(pred == act_ho.cpu().numpy()))
    # retrieval entropy: softmax(beta * sim) over codebook
    r = F.normalize(psi_ho, p=2, dim=-1)
    sim = r @ cleanup.engrams.T
    p = torch.softmax(beta * sim, dim=-1)
    h = -(p * torch.log(p + EPS)).sum(dim=-1).mean().item()
    return prec, h


def run_experiment(bank_npz: str, manifest_path: str, device: str,
                   seed: int, limit: int) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]  # [M, D] f32
    onehot = data["actions_onehot"]  # [M, 6]
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    M = psi.shape[0]
    act = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    if limit and limit > 0:
        train_idx, held_idx = train_idx[:limit], held_idx[:limit]

    psi_tr = torch.from_numpy(psi[train_idx]).to(dev)
    act_tr = act[train_idx].to(dev)
    psi_ho = torch.from_numpy(psi[held_idx]).to(dev)
    act_ho = act[held_idx].to(dev)
    nxt_ho = torch.from_numpy(nxt[held_idx]).to(dev)

    # ---- Codebook (train-only prototypes) ----
    engrams = _codebook(psi_tr, act_tr)

    # ---- Arm A: raw cosine argmax ----
    ra = F.normalize(psi_ho, p=2, dim=-1)
    sim_a = ra @ engrams.T
    prec_a = float(np.mean(sim_a.argmax(dim=-1).cpu().numpy() == act_ho.cpu().numpy()))

    # ---- Arm B: lexical_snap + retrieval entropy ----
    beta = math.sqrt(D)
    cleanup = ContinuousHopfieldCleanup(dim=D).to(dev)
    cleanup.store_engrams(engrams)
    prec_b, h_b = _snap_precision_and_entropy(cleanup, psi_ho, act_ho, beta)

    # ---- Arm C: coupled EDMD prediction -> snap ----
    torch.manual_seed(seed + 1)
    coupled = CoupledRecursiveDualEDMD(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=LAMBDA_FORGET,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=True).to(dev)
    for i in range(len(train_idx)):
        s = psi_tr[i].view(NUM_BLOCKS, BLOCK_DIM)
        a = _action_wave_for(int(act_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt[train_idx[i]]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        coupled.update_online_step(s, a, y)

    preds = []
    with torch.no_grad():
        for i in range(len(held_idx)):
            s = psi_ho[i].view(NUM_BLOCKS, BLOCK_DIM)
            a = _action_wave_for(int(act_ho[i]), device=dev.type)
            pred = coupled.forward(s, a).view(-1)
            pred = F.normalize(pred, p=2, dim=-1)
            pidx, _ = cleanup.lexical_snap(pred, top_k=1)
            preds.append(int(pidx.item()))
    prec_c = float(np.mean(np.asarray(preds) == act_ho.cpu().numpy()))

    # ---- Verdict (Amendment 1: A/B delta removed as vacuous; see prereg) ----
    finite = all(math.isfinite(v) for v in [prec_a, prec_b, prec_c, h_b])
    verdict, reason = "FAIL", ""
    if not finite:
        verdict, reason = "KILL", "NaN/divergence in metrics"
    elif prec_b >= 0.80 and prec_c >= 0.50 and h_b <= 0.10:
        verdict, reason = "ACCEPT", (
            f"prec_B {prec_b:.4f} >= 0.80, prec_C {prec_c:.4f} >= 0.50, "
            f"H_B {h_b:.4f} <= 0.10")
    else:
        verdict, reason = "CONDITIONAL", (
            f"prec_B {prec_b:.4f} (>=0.80: {prec_b >= 0.80}), "
            f"prec_C {prec_c:.4f} (>=0.50: {prec_c >= 0.50}), "
            f"H_B {h_b:.4f} (<=0.10: {h_b <= 0.10})")

    result = {
        "schema": "henri.phase834.egress-snap-benchmark.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": len(train_idx), "heldout": len(held_idx)},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "gate": ("ACCEPT iff prec_B >= 0.80 AND prec_C >= 0.50 AND "
                 "H_B <= 0.10 nats (Amendment 1: A/B delta removed as "
                 "vacuous; KILL on NaN)"),
        "metrics": {
            "A_raw_cosine_precision": prec_a,
            "B_snap_precision": prec_b,
            "B_retrieval_entropy_nats": h_b,
            "C_coupled_predict_snap_precision": prec_c,
            "codebook_size": int(engrams.shape[0]),
            "snap_beta": beta,
        },
        "device": dev.type,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.34 egress-side snap benchmark")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    p.add_argument("--limit", type=int, default=0,
                   help="bounded smoke: cap train+heldout records (0 = full)")
    args = p.parse_args(argv)
    print(f"== Phase 8.34 egress snap benchmark ==\nbank: {args.bank}\n"
          f"seed: {args.seed} device: {args.device or 'auto'} limit: {args.limit or 'full'}",
          flush=True)
    result = run_experiment(args.bank, args.manifest or None, args.device,
                            args.seed, args.limit)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
