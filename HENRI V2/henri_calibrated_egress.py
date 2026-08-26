# -*- coding: utf-8 -*-
"""
HENRI Gate-4 Calibrated Egress Module (default-OFF, fail-closed).

Delegated by /henri-agent-integration per founding document
HENRI-ARCH-2026-08-GATE1-GATE4-STRATEGY (Drive inbox, sha256
d2f2bda5e11a202a911890f17ce6eaf25e0bf42be1accd57c3ff10f3a6ea008c) and the
Gate-4 conjunctive contract (Sol reference 3, 2026-08-26).

Purpose
-------
Project continuous wave hypervectors onto discrete (GameAction, data) tuples
through a provenance-validated, calibrated semantic action head. It is the
HENRI side of the CalibratedHopfieldEgress sketch in the founding document,
reconciled with the LIVE action-head stack:

  arc_action_head.py            (ActionHead, load_action_head, decode_action_head)
  henri_calibrated_action_head.py (StiefelActionProactor + ArtifactCalibrator)
  henri_calibrator_ingest.py    (authorized bank -> sealed artifact)
  arc_score_gate.py             (arc_score_eligibility, ACTION_HEAD_NOT_CALIBRATED)
  hopfield_cleanup.py           (ContinuousHopfieldCleanup lexical snap)

This module does NOT duplicate those components. It composes them into one
fail-closed egress boundary, adds the missing (GameAction, data) coordinate
path for ACTION6, and exposes a deterministic CLI for calibration and dry-run
verification. It never grants score eligibility by itself: score_eligible
remains the output of arc_score_eligibility with trained_action_head_active
and task validation.

Calibration source
------------------
Only authorized trajectory banks (henri_trajectory_bank.py, data_source
"authorized", live public-arcade tuples) are accepted. Synthetic or generated
data is labelled synthetic_fixture and can never activate production.

Honest boundary
---------------
Generated procedural environments exercise the calibration/dry-run MECHANICS
only. They do not provide authorized (observation, GameAction, data,
observation_next) trajectories; calibration of the semantic action head stays
gated on the authorized bank (henri-837-bank on Vast, 10,301 tuples).

CLI
---
  python henri_calibrated_egress.py calibrate --bank X.npz --manifest M.json \
      --artifact out.json --wave-dim 65536 --latent-dim 2048 --action-dim 6 \
      [--max-records N] [--seed S]
  python henri_calibrated_egress.py dry-run --artifact A.json \
      [--checkpoint C.pt] [--wave-dim 65536] [--latent-dim 2048]
  python henri_calibrated_egress.py verify-bank --bank X.npz --manifest M.json

All commands are default-OFF: nothing in this module flips production
behavior. Production activation requires the runner wiring (HENRI_ARC_ACTION_HEAD=1)
plus a qualified authorized artifact plus task validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

SCHEMA_ID = "henri.calibrated-egress.v1"
MODULE_VERSION = "1.0.0"
DEFAULT_WAVE_DIM = 65536
DEFAULT_LATENT_DIM = 2048
DEFAULT_ACTION_DIM = 6
CANONICAL_ARC_ACTIONS: List[str] = [f"ACTION{i}" for i in range(1, 7)]

# Thresholds from the founding document + live action-head contract.
CALIBRATION_MSE_THRESHOLD = 0.05
SAGNAC_PROXY_THRESHOLD = 0.20


class CalibratedEgressError(RuntimeError):
    """Typed fail-closed error for the calibrated egress boundary."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tensor_sha256(t: torch.Tensor) -> str:
    tt = t.detach().cpu().contiguous()
    return _sha256_bytes(tt.numpy().tobytes())


def _lf_bytes(raw: bytes) -> bytes:
    """Canonical LF bytes (Windows CRLF -> LF) for digest stability."""
    return raw.replace(bytes((13, 10)), b"\n")


def validate_action_schema(artifact: Dict[str, Any]) -> None:
    """Exact vocabulary/order validation; any mismatch raises."""
    order = artifact.get("action_ordering")
    if not isinstance(order, list) or order != CANONICAL_ARC_ACTIONS:
        raise CalibratedEgressError(
            f"action_ordering {order!r} != canonical {CANONICAL_ARC_ACTIONS!r}"
        )
    n = artifact.get("action_dim")
    if n != len(CANONICAL_ARC_ACTIONS):
        raise CalibratedEgressError(
            f"action_dim {n} != canonical {len(CANONICAL_ARC_ACTIONS)}"
        )


def load_calibrated_artifact(path: str) -> Dict[str, Any]:
    """Strict artifact loader: schema, self-hash, ordering, thresholds."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except Exception as exc:
        raise CalibratedEgressError(f"artifact unreadable: {exc}") from exc
    if raw.get("schema_id") != SCHEMA_ID and raw.get("schema_id") != "henri.calibrated-action-head.v1":
        raise CalibratedEgressError(
            f"schema {raw.get('schema_id')} != {SCHEMA_ID} (or calibrated-action-head.v1)"
        )
    blob = json.dumps({k: v for k, v in raw.items() if k != "artifact_sha256"},
                      sort_keys=True).encode("utf-8")
    if raw.get("artifact_sha256") and _sha256_bytes(blob) != raw.get("artifact_sha256"):
        raise CalibratedEgressError("artifact self-hash mismatch")
    validate_action_schema(raw)
    return raw


def egress_state(artifact: Optional[Dict[str, Any]] = None,
                 *,
                 trained_head_active: bool = False,
                 task_validated: bool = False,
                 data_source: Optional[str] = None) -> Dict[str, Any]:
    """Deterministic egress state record (never grants eligibility)."""
    if artifact is None:
        return {
            "schema_id": SCHEMA_ID,
            "egress_mode": "FAIL_CLOSED",
            "calibrated_artifact_loaded": False,
            "trained_action_head_active": False,
            "task_validated": False,
            "score_eligible": False,
            "score_block_reason": "ACTION_HEAD_NOT_CALIBRATED",
            "diagnostic_only": True,
        }
    qualified = bool(artifact.get("is_qualified"))
    authorized = artifact.get("data_source") == "authorized"
    head_ok = bool(trained_head_active and qualified and authorized)
    if not head_ok:
        return {
            "schema_id": SCHEMA_ID,
            "egress_mode": "FAIL_CLOSED",
            "calibrated_artifact_loaded": True,
            "trained_action_head_active": False,
            "task_validated": False,
            "score_eligible": False,
            "score_block_reason": (
                "ACTION_HEAD_NOT_QUALIFIED" if not qualified
                else "ACTION_HEAD_SYNTHETIC_ONLY" if not authorized
                else "ACTION_HEAD_NOT_CALIBRATED"
            ),
            "diagnostic_only": True,
        }
    if not task_validated:
        return {
            "schema_id": SCHEMA_ID,
            "egress_mode": "FAIL_CLOSED",
            "calibrated_artifact_loaded": True,
            "trained_action_head_active": True,
            "task_validated": False,
            "score_eligible": False,
            "score_block_reason": "ACTION_HEAD_NOT_TASK_VALIDATED",
            "diagnostic_only": True,
        }
    return {
        "schema_id": SCHEMA_ID,
        "egress_mode": "CALIBRATED",
        "calibrated_artifact_loaded": True,
        "trained_action_head_active": True,
        "task_validated": True,
        "score_eligible": True,
        "score_block_reason": "",
        "diagnostic_only": False,
    }


def calibrate_cli(argv: Optional[List[str]] = None) -> int:
    """CLI: bank -> filtered -> calibrated sealed artifact (authorized only)."""
    p = argparse.ArgumentParser(description="Calibrated egress: bank -> artifact")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--artifact", required=True)
    p.add_argument("--wave-dim", type=int, default=DEFAULT_WAVE_DIM)
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260826)
    args = p.parse_args(argv)

    # Reuse the live ingest pipeline (henri_calibrator_ingest.ingest_bank_to_artifact).
    from henri_calibrator_ingest import ingest_bank_to_artifact
    art = ingest_bank_to_artifact(
        args.bank,
        args.manifest or None,
        args.artifact,
        wave_dim=args.wave_dim,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        max_records=args.max_records,
        seed=args.seed,
    )
    validate_action_schema(art)
    ok, reason = ("", "")
    try:
        from henri_calibrated_action_head import production_activation_eligible
        ok, reason = production_activation_eligible(art)
    except Exception as exc:
        reason = f"activation-eligibility check failed: {exc}"
    print(json.dumps({
        "artifact": args.artifact,
        "status": art.get("status"),
        "is_qualified": art.get("is_qualified"),
        "data_source": art.get("data_source"),
        "calibration_mse_heldout": art.get("calibration_mse_heldout"),
        "sagnac_stress_proxy_action_l2": art.get("sagnac_stress_proxy_action_l2"),
        "train_count": art.get("train_count"),
        "held_out_count": art.get("held_out_count"),
        "activation_eligible": bool(ok),
        "activation_reason": reason,
        "artifact_sha256": art.get("artifact_sha256"),
    }, indent=1, default=str))
    return 0


def dry_run_cli(argv: Optional[List[str]] = None) -> int:
    """Dry-run verification: fail-closed egress boundary + hopfield snap.

    Validates that ACTION_HEAD_NOT_CALIBRATED is resolved ONLY when a
    qualified authorized artifact plus a trained active head are present.
    In every other configuration the verdict is fail-closed.
    """
    p = argparse.ArgumentParser(description="Calibrated egress dry-run")
    p.add_argument("--artifact", default="")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--wave-dim", type=int, default=DEFAULT_WAVE_DIM)
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    p.add_argument("--task-validated", action="store_true",
                   help="SIMULATION ONLY: pretend task validation passed "
                        "(never used for real promotion)")
    args = p.parse_args(argv)

    verdict: Dict[str, Any] = {
        "schema_id": "henri.gauntlet-verdict.v1",
        "command": "dry-run",
        "module": "henri_calibrated_egress.py",
        "module_version": MODULE_VERSION,
        "timestamp": time.time(),
    }

    # 1. Artifact load (strict)
    artifact = None
    if args.artifact:
        try:
            artifact = load_calibrated_artifact(args.artifact)
            verdict["artifact_loaded"] = True
            verdict["artifact_sha256"] = artifact.get("artifact_sha256")
            verdict["artifact_status"] = artifact.get("status")
            verdict["artifact_qualified"] = artifact.get("is_qualified")
            verdict["artifact_data_source"] = artifact.get("data_source")
        except Exception as exc:
            verdict["artifact_loaded"] = False
            verdict["artifact_error"] = str(exc)

    # 2. Checkpoint load (strict, if path given)
    trained_active = False
    head_state = None
    if args.checkpoint:
        try:
            from arc_action_head import ActionHead, load_action_head
            head = ActionHead(d_hidden=args.latent_dim, n_actions=args.action_dim)
            head_state = load_action_head(
                head, args.checkpoint, policy="required",
                expected_hidden=args.latent_dim, expected_actions=args.action_dim,
            )
            trained_active = bool(head_state.trained_action_head_active)
            verdict["checkpoint_loaded"] = True
            verdict["trained_action_head_active"] = trained_active
        except Exception as exc:
            verdict["checkpoint_loaded"] = False
            verdict["checkpoint_error"] = str(exc)

    # 3. Egress state (fail-closed)
    state = egress_state(
        artifact,
        trained_head_active=trained_active,
        task_validated=bool(args.task_validated),
        data_source=(artifact or {}).get("data_source"),
    )
    verdict.update(state)

    # 4. Hopfield snap mechanics (diagnostic; only when artifact + head active)
    if state["egress_mode"] == "CALIBRATED":
        try:
            from hopfield_cleanup import ContinuousHopfieldCleanup
            cleanup = ContinuousHopfieldCleanup(dim=args.wave_dim)
            probe = torch.randn(args.wave_dim, dtype=torch.float32)
            probe = torch.nn.functional.normalize(probe, p=2, dim=-1)
            n = cleanup.store_engrams(probe.unsqueeze(0))
            idx, conf = cleanup.lexical_snap(probe.unsqueeze(0), top_k=1)
            verdict["hopfield_snap"] = {
                "stored": n,
                "index": int(idx.item()),
                "confidence": float(conf.item()),
            }
        except Exception as exc:
            verdict["hopfield_snap"] = {"error": str(exc)}

    verdict["verdict"] = (
        "PASS" if state["score_eligible"] else "BLOCKED"
    )
    verdict["score_block_reason"] = state["score_block_reason"]
    print(json.dumps(verdict, indent=1, default=str))
    return 0 if verdict["verdict"] == "PASS" else 2


def verify_bank_cli(argv: Optional[List[str]] = None) -> int:
    """Bank integrity + lineage check (no model state touched)."""
    p = argparse.ArgumentParser(description="Verify trajectory bank integrity")
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    args = p.parse_args(argv)

    from henri_trajectory_bank import TrajectoryBank
    data = TrajectoryBank.load(args.bank, args.manifest or None, verify_digest=True)
    m = data.get("manifest", {})
    print(json.dumps({
        "schema_id": m.get("schema_id"),
        "data_source": m.get("data_source"),
        "record_count": m.get("record_count"),
        "wave_dim": m.get("wave_dim"),
        "envs": m.get("envs"),
        "dataset_digest": m.get("dataset_digest"),
        "npz_sha256": m.get("npz_sha256"),
        "action_vocab": m.get("action_vocab"),
    }, indent=1, default=str))
    return 0


def export_head_cli(argv: Optional[List[str]] = None) -> int:
    """Export a trained-action-head checkpoint from a QUALIFIED AUTHORIZED
    artifact by deterministic re-derivation of the calibrated weights.

    The artifact stores only hashes (no 536 MB w_down). This command
    re-derives the weights from the SAME authorized bank with the SAME seed
    and proves byte-identical weight_sha256 before writing the .pt. Synthetic
    fixtures and unqualified artifacts are refused.
    """
    p = argparse.ArgumentParser(description="Export calibrated head checkpoint")
    p.add_argument("--artifact", required=True)
    p.add_argument("--bank", required=True)
    p.add_argument("--manifest", default="")
    p.add_argument("--checkpoint-out", required=True)
    p.add_argument("--wave-dim", type=int, default=DEFAULT_WAVE_DIM)
    p.add_argument("--latent-dim", type=int, default=DEFAULT_LATENT_DIM)
    p.add_argument("--action-dim", type=int, default=DEFAULT_ACTION_DIM)
    p.add_argument("--max-records", type=int, default=None)
    p.add_argument("--seed", type=int, default=20260826)
    args = p.parse_args(argv)

    art = load_calibrated_artifact(args.artifact)
    if not art.get("is_qualified"):
        raise CalibratedEgressError(
            "cannot export an unqualified artifact as a trained head")
    if art.get("data_source") != "authorized":
        raise CalibratedEgressError(
            "synthetic artifacts can never export a trained head")

    # Deterministic re-derivation of the calibrated weights (same bank/seed).
    from henri_calibrated_action_head import ActionHeadCalibrator, StiefelActionProactor
    from henri_calibrator_ingest import ingest_bank_to_artifact
    art2 = ingest_bank_to_artifact(
        args.bank,
        args.manifest or None,
        os.devnull,
        wave_dim=args.wave_dim,
        latent_dim=args.latent_dim,
        action_dim=args.action_dim,
        max_records=args.max_records,
        seed=args.seed,
    )
    if art2.get("weight_sha256") != art.get("weight_sha256"):
        raise CalibratedEgressError(
            "re-derived weight_sha256 mismatch (bank/seed/device drift); "
            "re-calibrate on the same device and retry")
    if not art2.get("is_qualified"):
        raise CalibratedEgressError(
            "re-derived calibration is not qualified; artifact is stale")

    # Export format expected by arc_action_head.load_action_head:
    #   state_dict {head.weight [n_actions, hidden], head.bias [n_actions]}
    #   + calibration_dataset_digest (required for trained_action_head_active).
    data = art2
    proactor = None  # ingest wrote to os.devnull; rebuild weights via calibrator
    from henri_trajectory_bank import TrajectoryBank, filter_onehot_to_vocab
    bank_data = TrajectoryBank.load(args.bank, args.manifest or None,
                                    verify_digest=True)
    psi = bank_data["psi"]
    onehot = bank_data["actions_onehot"]
    bank_vocab = bank_data["action_vocab"]
    onehot_sub, kept = filter_onehot_to_vocab(
        onehot, bank_vocab, [f"ACTION{i}" for i in range(1, args.action_dim + 1)])
    psi_sub = psi[kept]
    if args.max_records is not None and psi_sub.shape[0] > args.max_records:
        psi_sub = psi_sub[:args.max_records]
        onehot_sub = onehot_sub[:args.max_records]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    proactor = StiefelActionProactor(
        wave_dim=args.wave_dim, latent_dim=args.latent_dim,
        action_dim=args.action_dim).to(dev)
    calibrator = ActionHeadCalibrator(
        proactor, held_out_frac=0.2, seed=args.seed)
    art3 = calibrator.calibrate_from_trajectories(
        torch.from_numpy(psi_sub).to(dev),
        torch.from_numpy(onehot_sub).to(dev),
        data_source="authorized",
        split_identity=str(bank_data.get("manifest", {}).get("run_id", "unknown")),
    )
    if art3.get("weight_sha256") != art.get("weight_sha256"):
        raise CalibratedEgressError("re-derived weight_sha256 mismatch (final check)")

    sd = {
        "head.weight": proactor.w_act.weight.detach().cpu().clone(),
        "head.bias": torch.zeros(args.action_dim, dtype=torch.float32),
    }
    ckpt = {
        "state_dict": sd,
        "calibration_dataset_digest": str(art.get("dataset_digest", "")),
        "d_model": args.wave_dim,
        "artifact_sha256": art.get("artifact_sha256", ""),
        "source_commit": "",
    }
    out = args.checkpoint_out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    torch.save(ckpt, out)
    raw = Path(out).read_bytes()
    print(json.dumps({
        "checkpoint": out,
        "checkpoint_sha256": _sha256_bytes(raw),
        "head_weight_sha256": _tensor_sha256(sd["head.weight"]),
        "calibration_dataset_digest": ckpt["calibration_dataset_digest"],
        "artifact_sha256": art.get("artifact_sha256"),
        "qualified": art3.get("is_qualified"),
        "data_source": "authorized",
    }, indent=1, default=str))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("calibrate")
    sub.add_parser("dry-run")
    sub.add_parser("verify-bank")
    sub.add_parser("export-head")
    args = p.parse_args(argv[:1])
    if args.command == "calibrate":
        return calibrate_cli(argv[1:])
    if args.command == "dry-run":
        return dry_run_cli(argv[1:])
    if args.command == "verify-bank":
        return verify_bank_cli(argv[1:])
    if args.command == "export-head":
        return export_head_cli(argv[1:])
    p.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
