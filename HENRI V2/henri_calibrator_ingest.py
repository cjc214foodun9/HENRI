# -*- coding: utf-8 -*-
"""
Phase 8.32 — Calibrator ingest: authorized trajectory bank -> sealed artifact.

Consumes a bank produced by henri_trajectory_bank.py (data_source
"authorized" by construction), filters records to the target action vocab,
runs ActionHeadCalibrator.calibrate_from_trajectories (SVD-form ridge +
Stiefel retraction, HELD-OUT qualification), and writes the sealed
henri.calibrated-action-head.v1 artifact JSON.

Gate discipline:
- The artifact alone NEVER enables production. production_activation_eligible
  plus live-path wiring + task validation decide activation (checked in
  production_arc_run.py). data_source="authorized" is required.
- A synthetic fixture labeled "authorized" is a mock loop: this module
  takes its data_source from the bank manifest, and the bank only ever
  records live public-arcade tuples. No label reconstruction.
- Default-OFF: no flag flips production behavior. The runner only writes
  an artifact file when invoked explicitly with a bank path.

Memory: waves are loaded float32 [M, D]; w_down [latent, wave_dim] is
536 MB at D=65,536 / latent=2048 -> prefer CUDA (32 GB host) for
production dims. CPU reduced-dims for tests.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import torch

from henri_calibrated_action_head import (
    ActionHeadCalibrator,
    CalibratedActionHeadError,
    SCHEMA_ID,
    StiefelActionProactor,
    production_activation_eligible,
)
from henri_trajectory_bank import TrajectoryBank, filter_onehot_to_vocab

DEFAULT_LATENT_DIM = 2048
DEFAULT_RIDGE_GAMMA = 1e-3
DEFAULT_HELDOUT_FRAC = 0.2
CANONICAL_ARC_ACTIONS = [f"ACTION{i}" for i in range(1, 7)]


class CalibratorIngestError(RuntimeError):
    """Typed failure for the ingest pipeline."""


def ingest_bank_to_artifact(
    npz_path: str,
    manifest_path: Optional[str],
    artifact_out_path: str,
    wave_dim: int,
    latent_dim: int = DEFAULT_LATENT_DIM,
    action_dim: int = 6,
    target_vocab: Optional[List[str]] = None,
    ridge_gamma: float = DEFAULT_RIDGE_GAMMA,
    held_out_frac: float = DEFAULT_HELDOUT_FRAC,
    seed: int = 20260819,
    device: Optional[torch.device] = None,
    max_records: Optional[int] = None,
) -> Dict[str, Any]:
    """Load bank -> filter to vocab -> calibrate -> seal artifact.

    Returns the artifact dict AND writes it to artifact_out_path.
    """
    if not os.path.isfile(npz_path):
        raise CalibratorIngestError(f"bank npz missing: {npz_path}")
    target_vocab = list(target_vocab or CANONICAL_ARC_ACTIONS)
    if len(target_vocab) != action_dim:
        raise CalibratorIngestError(
            f"target_vocab {len(target_vocab)} != action_dim {action_dim}")

    data = TrajectoryBank.load(npz_path, manifest_path, verify_digest=True)
    psi = data["psi"]
    onehot = data["actions_onehot"]
    bank_vocab = data["action_vocab"]

    try:
        onehot_sub, _kept = filter_onehot_to_vocab(
            onehot, bank_vocab, target_vocab)
    except Exception as exc:
        raise CalibratorIngestError(
            f"vocab filter failed (bank {bank_vocab}, target {target_vocab}): {exc}"
        ) from exc
    if onehot_sub.shape[1] != action_dim:
        raise CalibratorIngestError(
            f"filtered action dim {onehot_sub.shape[1]} != {action_dim}")

    psi_sub = psi[_kept]
    if max_records is not None and psi_sub.shape[0] > max_records:
        psi_sub = psi_sub[:max_records]
        onehot_sub = onehot_sub[:max_records]
    if psi_sub.shape[0] < 2:
        raise CalibratorIngestError(
            f"only {psi_sub.shape[0]} records after filter; need >= 2")

    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wave_dim != psi_sub.shape[1]:
        raise CalibratorIngestError(
            f"wave_dim {wave_dim} != bank dim {psi_sub.shape[1]}")

    proactor = StiefelActionProactor(
        wave_dim=wave_dim, latent_dim=latent_dim,
        action_dim=action_dim).to(dev)
    calibrator = ActionHeadCalibrator(
        proactor, ridge_gamma=ridge_gamma,
        held_out_frac=held_out_frac, seed=seed)

    psi_t = torch.from_numpy(psi_sub).to(dev)
    a_t = torch.from_numpy(onehot_sub).to(dev)

    split_identity = str(data["manifest"].get("run_id", "unknown"))
    art = calibrator.calibrate_from_trajectories(
        psi_t, a_t,
        data_source="authorized",
        split_identity=split_identity,
        no_eval_cache_provenance=(
            "no_arc_solutions_or_eval_caches_used; "
            "records from live public-arcade run "
            f"{data['manifest'].get('provenance', '')}"),
    )
    art["bank_npz_sha256"] = data["manifest"].get("npz_sha256", "")
    art["bank_dataset_digest"] = data["manifest"].get("dataset_digest", "")

    os.makedirs(os.path.dirname(os.path.abspath(artifact_out_path)),
                exist_ok=True)
    with open(artifact_out_path, "w", encoding="utf-8") as f:
        json.dump(art, f, indent=1, default=str)
    return art


def ingest_cli(argv: Optional[List[str]] = None) -> int:
    """CLI: python henri_calibrator_ingest.py --bank X.npz --manifest M.json
    --artifact out.json [--wave-dim D] [--latent-dim L] [--action-dim 6]
    [--max-records N]"""
    import argparse

    p = argparse.ArgumentParser(description="Calibrator ingest (bank -> artifact)")
    p.add_argument("--bank", required=True, help="bank npz path")
    p.add_argument("--manifest", default="", help="bank manifest json path")
    p.add_argument("--artifact", required=True, help="output artifact json path")
    p.add_argument("--wave-dim", type=int, required=True)
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--action-dim", type=int, default=6)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--ridge-gamma", type=float, default=DEFAULT_RIDGE_GAMMA)
    p.add_argument("--held-out-frac", type=float, default=DEFAULT_HELDOUT_FRAC)
    p.add_argument("--seed", type=int, default=20260819)
    args = p.parse_args(argv)

    art = ingest_bank_to_artifact(
        args.bank,
        args.manifest or None,
        args.artifact,
        wave_dim=args.wave_dim,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        ridge_gamma=args.ridge_gamma,
        held_out_frac=args.held_out_frac,
        seed=args.seed,
        max_records=args.max_records,
    )
    ok, reason = production_activation_eligible(art)
    print(f"artifact  -> {args.artifact}")
    print(f"status    : {art['status']}  qualified={art['is_qualified']}")
    print(f"mse_te    : {art['calibration_mse_heldout']:.4f} "
          f"(threshold {art.get('mse_threshold', 0.05)})")
    print(f"sagnac_pr : {art['sagnac_stress_proxy_action_l2']:.4f} "
          f"(threshold {art.get('sagnac_threshold', 0.2)})")
    print(f"records   : train={art['train_count']} heldout={art['held_out_count']}")
    print(f"weight    : {art['weight_sha256'][:16]}...")
    print(f"activation: {ok} ({reason})")
    return 0


if __name__ == "__main__":
    raise SystemExit(ingest_cli())
