"""F3 grouped 4-fold split sealer — GENERATION-ONLY process.

Per SPEC-2026-08-29-F3-BROAD-BANK section 4: 12 environments -> 4 folds x 3
held-out envs; fold = lexicographic env index mod 4. All episodes of an env
live in exactly one fold (never split). Fold assignment is NON-ADAPTIVE.

No checkpoint/model load. Receipt schema f3-split-seal.v1 pins per-fold env
lists + record counts, rule, seed, fold-manifest SHA-256, single_use=true.

Usage (remote, repo root, after capture + finalize):
  /venv/main/bin/python "HENRI V2/experiments/verification/f3_split_seal.py" \
      --npz telemetry/f3_bank_capture/trajectories_<run_id>.npz \
      --jsonl telemetry/f3_bank_capture/trajectories_<run_id>.jsonl \
      --manifest telemetry/f3_bank_capture/trajectories_<run_id>_manifest.json \
      --seed 20260829 \
      --out telemetry/f3_bank_capture/f3_split_seal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_assignment(env_ids: Sequence[str], n_folds: int = 4) -> Dict[str, int]:
    """Fold = lexicographic index mod n_folds (non-adaptive, deterministic)."""
    ordered = sorted(set(env_ids))
    return {e: i % n_folds for i, e in enumerate(ordered)}


def per_env_counts(meta: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in meta:
        e = str(m.get("env", "?"))
        out[e] = out.get(e, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seed", type=int, default=20260829)
    ap.add_argument("--n-folds", type=int, default=4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    npz_sha = _sha256(args.npz)
    jsonl_sha = _sha256(args.jsonl)
    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["data_source"] == "authorized", "bank must be authorized capture"
    assert manifest["npz_sha256"] == npz_sha, "npz hash mismatch vs manifest"
    assert manifest["jsonl_sha256"] == jsonl_sha, "jsonl hash mismatch vs manifest"

    bank = np.load(args.npz)
    meta = []
    with open(args.jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            meta.append(json.loads(line))
    assert len(meta) == bank["psi"].shape[0], "jsonl/meta row mismatch"

    env_counts = per_env_counts(meta)
    envs = sorted(env_counts)
    assert len(envs) % args.n_folds == 0, (
        f"env count {len(envs)} must be divisible by {args.n_folds}"
    )
    assert len(envs) >= 12, f"need >= 12 envs, got {len(envs)}"
    assert len(envs) // args.n_folds == 3, (
        f"spec requires exactly 3 held-out envs per fold, got {len(envs) // args.n_folds}"
    )

    assign = fold_assignment(envs, args.n_folds)
    folds: Dict[str, Dict] = {}
    for f in range(args.n_folds):
        heldout = sorted(e for e in envs if assign[e] == f)
        train = sorted(e for e in envs if assign[e] != f)
        folds[f"fold{f}"] = {
            "heldout_envs": heldout,
            "train_envs": train,
            "n_heldout": int(sum(env_counts[e] for e in heldout)),
            "n_train": int(sum(env_counts[e] for e in train)),
        }
    assert all(len(folds[f"fold{f}"]["heldout_envs"]) == len(envs) // args.n_folds
               for f in range(args.n_folds))

    fold_manifest = {
        "rule": "grouped_4fold_env_disjoint_lexicographic_mod",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "env_order": envs,
        "folds": folds,
        "single_use": True,
    }
    fold_sha = hashlib.sha256(
        json.dumps(fold_manifest, sort_keys=True).encode()
    ).hexdigest()

    receipt = {
        "schema_id": "f3-split-seal.v1",
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "manifest_run_id": manifest.get("run_id"),
        "record_count": int(len(meta)),
        "envs": envs,
        "per_env_counts": env_counts,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "folds": folds,
        "split_rule": "grouped_4fold_env_disjoint_lexicographic_mod",
        "fold_manifest_sha256": fold_sha,
        "single_use": True,
        "generator": f"f3_split_seal.py@{_sha256(str(Path(__file__).resolve()))[:16]}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("F3_SPLIT_SEAL_OK")


if __name__ == "__main__":
    main()
