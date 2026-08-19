"""Phase 8.34 — Coupled EDMD + Lexical Snap benchmark (production D).

Arms on the AUTHORIZED bank (90 records, 72/18, seed 20260819):
  A: RecursiveDualEDMD r=16          (sealed 8.33 linear baseline)
  B: CoupledRecursiveDualEDMD r=128  (global field channel ON)
  C: CoupledRecursiveDualEDMD r=128  (field channel OFF = control)

Metric: held-out 1 - cos(pred, target); train final MSE;
field-attributable cross-block Jacobian delta (B full - B no-field).

Pre-registered gate (arc_phase834_coupled_transition_prereg.md):
  ACCEPT: B_holdout <= 0.90 AND (B_holdout - A_holdout) <= -0.05
          AND jac_delta > 1e-6
  FAIL   : otherwise (evidence-backed falsification)
  BLOCKED_INFRA: NaN / digest mismatch / GPU failure

Output: compact JSON verdict on stdout (last line).
"""

import argparse
import json
import math
import time

import numpy as np
import torch

from henri_trajectory_bank import TrajectoryBank
from recursive_dual_edmd import CoupledRecursiveDualEDMD, RecursiveDualEDMD

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM  # 65,536
RANK_COUPLED = 128
RANK_BASELINE = 16


def _action_wave_for(idx: int, num_blocks: int = NUM_BLOCKS, block_dim: int = BLOCK_DIM,
                     device: str = "cpu") -> torch.Tensor:
    """Deterministic unit per-block action wave (mirrors sealed 8.33)."""
    g = torch.Generator(device="cpu").manual_seed(1000 + idx)
    w = torch.randn(num_blocks, block_dim, generator=g)
    w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)
    return w.to(device)


def split_indices(n: int, frac: float, seed: int):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def _holdout_loss(predict, psi_ho, nxt_ho, y_ho, dev, num_blocks, block_dim):
    """1 - cos(pred, target) over held-out records."""
    total = 0.0
    for s, y, a in zip(psi_ho, nxt_ho, y_ho):
        s_w = torch.from_numpy(s).view(num_blocks, block_dim).to(dev)
        a_w = _action_wave_for(int(a), device=dev.type)
        y_w = torch.from_numpy(y).view(num_blocks, block_dim).to(dev)
        with torch.no_grad():
            pred = predict(s_w, a_w).view(-1)
            cos = torch.dot(torch.nn.functional.normalize(pred, p=2, dim=0),
                            torch.nn.functional.normalize(y_w.view(-1), p=2, dim=0))
            total += float(1.0 - cos.item())
    return total / len(psi_ho)


def run_experiment(bank_npz: str, manifest_path: str, device: str, seed: int) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]                    # [M, D] f32
    onehot = data["actions_onehot"]      # [M, 6]
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    if psi.shape[1] != D:
        raise RuntimeError(f"bank wave dim {psi.shape[1]} != {D} (production only)")
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    psi_tr = psi[train_idx]; nxt_tr = nxt[train_idx]
    psi_ho = psi[held_idx];  nxt_ho = nxt[held_idx]
    y_tr = action_idx[train_idx].numpy(); y_ho = action_idx[held_idx].numpy()

    # ---- Arm A: sealed r=16 linear baseline (exact 8.33 construction) ----
    torch.manual_seed(seed)
    edmd = RecursiveDualEDMD(d_model=D, r_rank=RANK_BASELINE, lambda_forget=0.98).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        edmd.update_online_step(s, a, y)

    # ---- Arm B: coupled r=128 (field ON) ----
    torch.manual_seed(seed + 1)
    coupled = CoupledRecursiveDualEDMD(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=True).to(dev)
    train_loss_b = 0.0
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        train_loss_b = coupled.update_online_step(s, a, y)

    # ---- Arm C: coupled r=128 (field OFF = control) ----
    torch.manual_seed(seed + 2)
    control = CoupledRecursiveDualEDMD(
        d_model=D, r_rank=RANK_COUPLED, lambda_forget=0.98,
        num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM, field_channel=False).to(dev)
    for i in range(len(train_idx)):
        s = torch.from_numpy(psi_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        a = _action_wave_for(int(y_tr[i]), device=dev.type)
        y = torch.from_numpy(nxt_tr[i]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
        control.update_online_step(s, a, y)

    # ---- Held-out metrics ----
    loss_a = _holdout_loss(edmd, psi_ho, nxt_ho, y_ho, dev, NUM_BLOCKS, BLOCK_DIM)
    loss_b = _holdout_loss(coupled, psi_ho, nxt_ho, y_ho, dev, NUM_BLOCKS, BLOCK_DIM)
    loss_c = _holdout_loss(control, psi_ho, nxt_ho, y_ho, dev, NUM_BLOCKS, BLOCK_DIM)

    # ---- Field-attributable cross-block Jacobian delta (B full - B no-field) ----
    s0 = torch.from_numpy(psi_ho[0]).view(NUM_BLOCKS, BLOCK_DIM).to(dev)
    a0 = _action_wave_for(int(y_ho[0]), device=dev.type)
    j_full = coupled.cross_block_jacobian(s0, a0, block_a=0, block_b=100, include_field=True)
    j_nofield = coupled.cross_block_jacobian(s0, a0, block_a=0, block_b=100, include_field=False)
    jac_delta = j_full - j_nofield

    delta_ba = loss_b - loss_a
    verdict, reason = "FAIL", ""
    if not all(math.isfinite(v) for v in [loss_a, loss_b, loss_c, jac_delta]):
        verdict, reason = "BLOCKED_INFRA", "NaN in metrics"
    elif loss_b <= 0.90 and delta_ba <= -0.05 and jac_delta > 1e-6:
        verdict = "ACCEPT"
        reason = (f"B {loss_b:.4f} <= 0.90, delta_BA {delta_ba:.4f} <= -0.05, "
                  f"jac_delta {jac_delta:.3e} > 1e-6")
    else:
        reason = (f"B {loss_b:.4f} (<=0.90: {loss_b <= 0.90}), "
                  f"delta_BA {delta_ba:.4f} (<=-0.05: {delta_ba <= -0.05}), "
                  f"jac_delta {jac_delta:.3e} (>1e-6: {jac_delta > 1e-6})")

    result = {
        "schema": "henri.phase834.coupled-transition-benchmark.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": len(train_idx), "heldout": len(held_idx)},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "gate": ("ACCEPT iff B <= 0.90 AND (B-A) <= -0.05 AND jac_delta > 1e-6; "
                 "else FAIL; BLOCKED_INFRA on NaN"),
        "metrics": {
            "A_linear_r16_holdout": loss_a,
            "B_coupled_r128_holdout": loss_b,
            "C_control_r128_holdout": loss_c,
            "delta_BA": delta_ba,
            "B_train_final_mse": train_loss_b,
            "jacobian_full": j_full,
            "jacobian_nofield": j_nofield,
            "jacobian_field_delta": jac_delta,
        },
        "device": dev.type,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.34 coupled transition benchmark")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    args = p.parse_args(argv)
    print(f"== Phase 8.34 coupled transition benchmark ==\nbank: {args.bank}\n"
          f"seed: {args.seed} device: {args.device or 'auto'}", flush=True)
    result = run_experiment(args.bank, args.manifest or None, args.device, args.seed)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
