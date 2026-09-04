"""F3 grouped 4-fold cross-validation egress gates (remote CUDA).

Consumes the authorized F3 broad-bank artifacts (npz + jsonl + manifest) and
the SEALED fold manifest (f3_split_seal.json). Per SPEC-2026-08-29-F3-BROAD-BANK:

  - 4-fold grouped CV: fold = lexicographic env index mod 4 (sealed).
    For each fold: calibrate on train envs ONLY, snap on held-out envs ONLY.
  - G1: macro held-out P@1 >= 0.99 AND min-fold held-out P@1 >= 0.95.
  - G2: per-action coverage: every action with N_a(test) >= 10 in a held-out
        fold has per-action held-out P@1 >= 0.80; lower-support actions are
        reported coverage-limited, not scored.
  - G3: ACTION6 coordinate-payload format validity >= 0.99.
        DISCLOSURE: the trajectory-bank schema (henri.arc-trajectory-bank.v1)
        records action_name only — no (GameAction, data) payload. Reported
        BLOCKED_NO_PAYLOAD_IN_BANK until a bank schema v2 persists payloads.
  - G4: margin vs train-marginal predictor >= +0.05: the marginal predictor
        is argmax of the fold's training action marginal (the F2 collapse mode).
  - G5/G6 (code contracts, verified by contract tests): default-OFF
    differential + eligibility boundary.
  - Kills: K1 macro < 0.99 or min-fold < 0.95; K2 margin < +0.05;
    K3 no engagement telemetry; K4 dense [D,D] ban; K6 hyperparameter delta.
    beta = 8.0, ridge = 1e-3 FROZEN (no CLI override).

Usage (remote, repo root):
  env PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f3_egress_gates.py" \
      --npz telemetry/f3_bank_capture/trajectories_<run_id>.npz \
      --jsonl telemetry/f3_bank_capture/trajectories_<run_id>.jsonl \
      --manifest telemetry/f3_bank_capture/trajectories_<run_id>_manifest.json \
      --split-seal telemetry/f3_bank_capture/f3_split_seal.json \
      --out telemetry/f3_bank_capture/f3_gates_receipt.json

Local CPU runs are software sanity only; remote CUDA is the verification boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from typing import Dict, List

import numpy as np
import torch


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# FROZEN hyperparameters (SPEC-2026-08-29-F3-BROAD-BANK section 2 / K6)
FROZEN_BETA = 8.0
FROZEN_RIDGE = 1e-3
FROZEN_N_FOLDS = 4


def load_bank(npz_path: str, jsonl_path: str, manifest_path: str):
    npz_sha = _sha256(npz_path)
    jsonl_sha = _sha256(jsonl_path)
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["data_source"] == "authorized", "bank must be authorized capture"
    assert manifest["npz_sha256"] == npz_sha, "npz hash mismatch vs manifest"
    assert manifest["jsonl_sha256"] == jsonl_sha, "jsonl hash mismatch vs manifest"

    bank = np.load(npz_path)
    psi = bank["psi"].astype(np.float32)
    actions_onehot = bank["actions_onehot"].astype(np.float32)
    action_names = [str(a) for a in bank["action_names"]]
    meta = []
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            meta.append(json.loads(line))
    assert len(meta) == psi.shape[0], "jsonl/meta row mismatch"
    return psi, actions_onehot, action_names, meta, manifest


def load_sealed_folds(split_seal_path: str) -> Dict[str, Dict]:
    with open(split_seal_path, "r", encoding="utf-8") as fp:
        seal = json.load(fp)
    assert seal.get("schema_id") == "f3-split-seal.v1", "wrong seal schema"
    assert seal.get("single_use") is True, "sealed split must be single_use"
    folds = seal["folds"]
    # Re-derive and verify the fold manifest digest
    manifest = {
        "rule": "grouped_4fold_env_disjoint_lexicographic_mod",
        "n_folds": seal["n_folds"],
        "seed": seal.get("seed"),
        "env_order": seal["envs"],
        "folds": folds,
        "single_use": True,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert digest == seal["fold_manifest_sha256"], (
        "fold manifest digest mismatch vs sealed receipt"
    )
    return folds


def per_env_mask(envs: List[str], env_ids: List[str]) -> Dict[str, np.ndarray]:
    arr = np.array(envs)
    return {e: arr == e for e in sorted(set(envs))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-seal", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260829)

    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest
    )
    D = psi.shape[1]
    V = actions_onehot.shape[1]
    assert D == 65536, f"expected D=65536, got {D}"
    assert V >= 2, f"action vocab must be >= 2, got {V}"

    envs = [str(m.get("env", "?")) for m in meta]
    folds = load_sealed_folds(args.split_seal)
    masks = per_env_mask(envs, envs)

    from f2_egress_codebook import F2HopfieldEgressCodebook

    per_fold = {}
    p1_folds: List[float] = []
    margin_folds: List[float] = []
    engaged = False
    for fold_name in [f"fold{i}" for i in range(FROZEN_N_FOLDS)]:
        fold = folds[fold_name]
        train_envs = set(fold["train_envs"])
        heldout_envs = set(fold["heldout_envs"])
        train_mask = np.array([e in train_envs for e in envs])
        hold_mask = ~train_mask
        assert int(train_mask.sum()) == fold["n_train"], "train count mismatch vs seal"
        assert int(hold_mask.sum()) == fold["n_heldout"], "heldout count mismatch vs seal"

        X_cal = torch.from_numpy(psi[train_mask]).to(device)
        Y_cal = torch.from_numpy(actions_onehot[train_mask]).to(device)
        X_hold = torch.from_numpy(psi[hold_mask]).to(device)
        Y_hold = torch.from_numpy(actions_onehot[hold_mask]).to(device)

        cb = F2HopfieldEgressCodebook(
            d_model=D, vocab_size=V, beta=FROZEN_BETA, ridge_lambda=FROZEN_RIDGE
        )
        t0 = time.perf_counter()
        cb.calibrate(X_cal, Y_cal)
        t_cal = time.perf_counter() - t0
        torch.cuda.synchronize() if device == "cuda" else None

        t0 = time.perf_counter()
        z, logits = cb.snap(X_hold, return_logits=True)
        t_snap = time.perf_counter() - t0
        torch.cuda.synchronize() if device == "cuda" else None

        pred = logits.argmax(dim=-1)
        true = Y_hold.argmax(dim=-1)
        p1 = (pred == true).float().mean().item()
        p1_folds.append(p1)

        # G2: per-action coverage on held-out records
        per_action = {}
        for a in range(V):
            mask_a = true == a
            n_a = int(mask_a.sum())
            if n_a >= 10:
                per_action[action_names[a]] = {
                    "n_test": n_a,
                    "p1": round((pred[mask_a] == a).float().mean().item(), 4),
                }
            else:
                per_action[action_names[a]] = {
                    "n_test": n_a,
                    "p1": None,
                    "coverage_limited": True,
                }

        # G4: margin vs train-marginal predictor
        train_counts = actions_onehot[train_mask].sum(axis=0)
        marginal_pred = int(np.argmax(train_counts))
        held_true = actions_onehot[hold_mask].argmax(axis=1)
        marginal_p1 = float((held_true == marginal_pred).mean())
        margin = p1 - marginal_p1
        margin_folds.append(margin)

        per_fold[fold_name] = {
            "train_envs": sorted(train_envs),
            "heldout_envs": sorted(heldout_envs),
            "n_train": int(train_mask.sum()),
            "n_heldout": int(hold_mask.sum()),
            "p1": round(p1, 4),
            "per_action": per_action,
            "marginal_p1": round(marginal_p1, 4),
            "margin_vs_marginal": round(margin, 4),
            "time_calibrate_s": round(t_cal, 3),
            "time_snap_s": round(t_snap, 3),
            "codebook_bytes": cb.codebook_bytes(),
            "finite": bool(torch.isfinite(cb.M).all().item() and torch.isfinite(z).all().item()),
        }
        engaged = engaged or p1 > 0.0

    macro_p1 = float(np.mean(p1_folds))
    min_p1 = float(np.min(p1_folds))
    macro_margin = float(np.mean(margin_folds))

    # G2 aggregate: min over scored per-action P@1 (only where n_test >= 10)
    g2_min = None
    for f in per_fold.values():
        for a, v in f["per_action"].items():
            if v.get("p1") is not None:
                g2_min = v["p1"] if g2_min is None else min(g2_min, v["p1"])
    g2_pass = g2_min is not None and g2_min >= 0.80

    # G3: ACTION6 payload-format validity — bank schema v1 records action_name only
    g3_status = "BLOCKED_NO_PAYLOAD_IN_BANK"
    g3_note = (
        "Trajectory-bank schema henri.arc-trajectory-bank.v1 persists action_name "
        "only (no (GameAction, data) payload). Payload-format validity requires a "
        "bank schema v2 that persists the ACTION6 coordinate payload; not fabricable "
        "from current artifacts."
    )

    verdicts = []
    if macro_p1 < 0.99 or min_p1 < 0.95:
        verdicts.append("K1_KILLED")
    if macro_margin < 0.05:
        verdicts.append("K2_KILLED")
    if not engaged:
        verdicts.append("K3_KILLED_NO_ENGAGEMENT")
    if not g2_pass:
        verdicts.append("G2_FAILED")
    verdict = verdicts[0] if verdicts else "F3_GATES_ACCEPT"

    receipt = {
        "schema_id": "f3-egress-gates.v1",
        "device": device,
        "npz_sha256": _sha256(args.npz),
        "jsonl_sha256": _sha256(args.jsonl),
        "split_seal": args.split_seal,
        "record_count": int(len(meta)),
        "action_vocab": action_names,
        "frozen_beta": FROZEN_BETA,
        "frozen_ridge": FROZEN_RIDGE,
        "n_folds": FROZEN_N_FOLDS,
        "per_fold": per_fold,
        "g1_macro_p1": round(macro_p1, 4),
        "g1_min_fold_p1": round(min_p1, 4),
        "g1_pass": macro_p1 >= 0.99 and min_p1 >= 0.95,
        "g2_min_scored_p1": g2_min,
        "g2_pass": g2_pass,
        "g3_status": g3_status,
        "g3_note": g3_note,
        "g4_macro_margin_vs_marginal": round(macro_margin, 4),
        "g4_pass": macro_margin >= 0.05,
        "verdict": verdict,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print(f"F3_GATES_VERDICT={verdict}")


if __name__ == "__main__":
    main()
