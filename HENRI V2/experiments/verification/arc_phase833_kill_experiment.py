"""Phase 8.33 kill experiment: NonLinearWaveJEPA vs linear R-EDMD.

Compares held-out next-wave prediction on the AUTHORIZED trajectory bank
(`trajectories_production_run_1787164827.npz`, 90 records, sealed) between:

  A) linear baseline: `RecursiveDualEDMD` (r_rank=16, lambda_forget=0.98)
     fitted online over the 72 train records;
  B) treatment: `NonLinearWaveJEPA` (K=32 options, opt_dim=512, latent
     L=2048, GELU/LayerNorm phase-coupling core, JEPA loss + Sagnac stress)
     trained with Adam over the same 72 records.

Metrics on the SAME 18 held-out records (split mirrors the calibrator:
held_out_frac=0.2, seed=20260819, deterministic randperm):
  loss_ambient = 1 - cos(pred_full, target_full)            (verdict metric)
  sagnac_ambient = 1 - cos^2(pred_full, target_full)

RECORDED CONSTRAINT (bank schema, audited 2026-08-19): the trajectory bank
stores `actions_onehot` + `action_names` only — no action waves. Both arms
therefore condition on the action INDEX via the identical deterministic
per-action wave map (`_action_wave_for(idx)`), so the comparison is fair and
the JEPA option id is the onehot argmax (action index, K=32 >= 6 actions).

Pre-registered verdicts (see arc_phase833_nonlinear_macrooption_prereg.md):
  ACCEPT  : jepa_holdout < edmd_holdout - 0.05 AND jepa_train < 0.5
            AND jepa_sagnac_holdout < 0.5
  KILL    : jepa_holdout >= edmd_holdout (no gain) OR NaN/divergence
  INFRA   : bank corrupt / digest mismatch / GPU failure (no verdict)

Output: compact JSON verdict on stdout (last line), plus human lines.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

from henri_trajectory_bank import TrajectoryBank
from recursive_dual_edmd import RecursiveDualEDMD
from henri_nonlinear_wavejepa import NonLinearWaveJEPA

DEFAULT_SEED = 20260819
HELDOUT_FRAC = 0.2
NUM_BLOCKS = 8192
BLOCK_DIM = 8
D = NUM_BLOCKS * BLOCK_DIM  # 65,536


def _action_wave_for(idx: int, num_blocks: int = NUM_BLOCKS, block_dim: int = BLOCK_DIM,
                     device: str = "cpu") -> torch.Tensor:
    """Deterministic unit per-block action wave for a given action index."""
    g = torch.Generator(device="cpu").manual_seed(1000 + idx)
    w = torch.randn(num_blocks, block_dim, generator=g)
    w = w / (torch.norm(w, p=2, dim=-1, keepdim=True) + 1e-9)
    return w.to(device)


def split_indices(n: int, frac: float, seed: int):
    """Mirror the calibrator split: seeded randperm, held-out = first n_test."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_test = max(1, int(round(n * frac)))
    return perm[n_test:].tolist(), perm[:n_test].tolist()


def run_experiment(bank_npz: str, manifest_path: str, wave_dim: int,
                   epochs: int, seed: int, device: str) -> dict:
    dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    t0 = time.time()

    data = TrajectoryBank.load(bank_npz, manifest_path, verify_digest=True)
    psi = data["psi"]  # [M, D] f32
    onehot = data["actions_onehot"]  # [M, 6] f32
    nxt = data["next_wave"]
    if nxt is None or psi.shape[0] != nxt.shape[0]:
        raise RuntimeError("bank missing next_wave records")
    M = psi.shape[0]
    action_idx = torch.from_numpy(onehot).argmax(dim=-1).long()
    run_id = str(data["manifest"].get("run_id", "unknown"))

    train_idx, held_idx = split_indices(M, HELDOUT_FRAC, seed)
    n_train, n_held = len(train_idx), len(held_idx)

    # ---- Arm A: linear R-EDMD baseline (online fit over train) ----
    torch.manual_seed(seed)
    edmd = RecursiveDualEDMD(d_model=wave_dim, r_rank=16, lambda_forget=0.98).to(dev)
    for i in train_idx:
        s = torch.from_numpy(psi[i]).to(dev)
        a = _action_wave_for(int(action_idx[i]), device=dev.type)
        y = torch.from_numpy(nxt[i]).to(dev)
        edmd.update_online_step(s.view(NUM_BLOCKS, BLOCK_DIM), a, y.view(NUM_BLOCKS, BLOCK_DIM))

    # ---- Arm B: NonLinearWaveJEPA (Adam over train) ----
    torch.manual_seed(seed + 1)
    jepa = NonLinearWaveJEPA(
        full_dim=wave_dim, compressed_dim=2048,
        num_options=32, opt_dim=512, sagnac_lambda=0.15,
        device=dev.type,
    ).to(dev)
    optm = torch.optim.Adam([p for p in jepa.parameters() if p.requires_grad], lr=1e-3)

    psi_tr = torch.from_numpy(psi[train_idx]).to(dev)
    nxt_tr = torch.from_numpy(nxt[train_idx]).to(dev)
    opt_tr = action_idx[train_idx].to(dev)

    jepa_train_loss = float("nan")
    for ep in range(epochs):
        optm.zero_grad()
        out = jepa(psi_tr, opt_tr, nxt_tr)
        loss = out["loss"]
        if not torch.isfinite(loss):
            jepa_train_loss = float("nan")
            break
        loss.backward()
        optm.step()
        jepa_train_loss = float(out["jepa_loss"])
        if ep % 100 == 0:
            print(f"  jepa epoch {ep}: loss={out['loss']:.4f} "
                  f"jepa={out['jepa_loss']:.4f} sagnac={out['sagnac_stress']:.4f}", flush=True)

    # ---- Held-out evaluation (ambient full-wave metric) ----
    edmd_losses, jepa_losses, jepa_sagnacs = [], [], []
    with torch.no_grad():
        for i in held_idx:
            s = torch.from_numpy(psi[i]).to(dev)
            a = _action_wave_for(int(action_idx[i]), device=dev.type)
            y = torch.from_numpy(nxt[i]).to(dev)

            pred_e = edmd(s.view(NUM_BLOCKS, BLOCK_DIM), a).view(-1)
            pred_j = jepa.predict_full_wave(s, action_idx[i].unsqueeze(0).to(dev),
                                            num_blocks=NUM_BLOCKS, block_dim=BLOCK_DIM)[0].view(-1)

            y_n = y / (torch.norm(y, p=2) + 1e-9)
            pe_n = pred_e / (torch.norm(pred_e, p=2) + 1e-9)
            pj_n = pred_j / (torch.norm(pred_j, p=2) + 1e-9)
            edmd_losses.append(1.0 - float(torch.dot(pe_n, y_n).item()))
            jepa_losses.append(1.0 - float(torch.dot(pj_n, y_n).item()))
            jepa_sagnacs.append(1.0 - float(torch.dot(pj_n, y_n).item() ** 2))

    edmd_holdout = float(np.mean(edmd_losses))
    jepa_holdout = float(np.mean(jepa_losses))
    jepa_sagnac_holdout = float(np.mean(jepa_sagnacs))

    # ---- Verdict ----
    delta = jepa_holdout - edmd_holdout
    nan_seen = not (math.isfinite(jepa_train_loss) and math.isfinite(jepa_holdout)
                    and math.isfinite(edmd_holdout))
    if nan_seen:
        verdict = "KILL"
        reason = "NaN/divergence in training or evaluation"
    elif jepa_holdout >= edmd_holdout:
        verdict = "KILL"
        reason = f"no held-out gain: jepa {jepa_holdout:.4f} >= edmd {edmd_holdout:.4f}"
    elif delta <= -0.05 and jepa_train_loss < 0.5 and jepa_sagnac_holdout < 0.5:
        verdict = "ACCEPT"
        reason = f"held-out delta {delta:.4f} <= -0.05, train {jepa_train_loss:.4f} < 0.5, sagnac {jepa_sagnac_holdout:.4f} < 0.5"
    else:
        verdict = "CONDITIONAL"
        reason = (f"partial gain (delta {delta:.4f}) but gate unmet: "
                  f"train {jepa_train_loss:.4f} (need < 0.5), "
                  f"sagnac {jepa_sagnac_holdout:.4f} (need < 0.5)")

    result = {
        "schema": "henri.phase833.kill-experiment.v1",
        "verdict": verdict,
        "reason": reason,
        "run_id": run_id,
        "bank_npz_sha256": data["manifest"].get("npz_sha256", ""),
        "records": {"total": M, "train": n_train, "heldout": n_held},
        "split": {"held_out_frac": HELDOUT_FRAC, "seed": seed},
        "metrics": {
            "edmd_holdout_loss": edmd_holdout,
            "jepa_holdout_loss": jepa_holdout,
            "delta_holdout": delta,
            "jepa_train_jepa_loss": jepa_train_loss,
            "jepa_sagnac_holdout": jepa_sagnac_holdout,
        },
        "arms": {"edmd": "RecursiveDualEDMD r=16", "jepa": "NonLinearWaveJEPA K=32 L=2048"},
        "device": dev.type,
        "epochs": epochs,
        "elapsed_s": round(time.time() - t0, 1),
    }
    return result


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase 8.33 kill experiment")
    p.add_argument("--bank", required=True, help="bank npz path")
    p.add_argument("--manifest", default="", help="bank manifest json path")
    p.add_argument("--wave-dim", type=int, default=D)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--device", default="")
    args = p.parse_args(argv)

    print(f"== Phase 8.33 kill experiment ==\nbank: {args.bank}\nseed: {args.seed} "
          f"epochs: {args.epochs}\ndevice: {args.device or 'auto'}", flush=True)
    result = run_experiment(args.bank, args.manifest or None, args.wave_dim,
                            args.epochs, args.seed, args.device)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
