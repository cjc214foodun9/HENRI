"""F5 grouped 4-fold split sealer — GENERATION-ONLY process.

Spec: HENRI-SPEC-2026-08-F5-STRUCTURED-CODEC section 4.2.

The F3 split (seed 20260829, digest 30504659...) and F4 split (seed 20260830,
digest 640763c6..., single_use=true) are CONSUMED and must not be reused.

F5 seals a NEW split with the same env-disjoint seeded-permutation rule and a
NEW seed (default 20260831): `grouped_4fold_env_disjoint_seeded_permutation_mod`
+ seed 20260831 -> a fold assignment DIFFERENT from both consumed splits.

Receipt schema f5-split-seal.v1: npz/jsonl hashes, per-env counts, rule, seed,
folds (heldout/train env lists + counts), fold-manifest SHA-256, single_use=true,
generator identity, UTC. The gates harness re-derives the manifest from the
receipt and verifies the digest (fail-closed), and REFUSES f3/f4 receipt
schemas (consumed-guard).

Usage (remote, repo root, after capture + finalize):
  env PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f5_split_seal.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --seed 20260831 \
      --out telemetry/f3_bank_capture_v2/f5_split_seal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fold_assignment(env_ids: Sequence[str], n_folds: int = 4,
                    seed: int = 20260831) -> Dict[str, int]:
    """Fold = seeded-permutation index mod n_folds (F4 rule, new seed)."""
    ordered = sorted(set(env_ids))
    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(len(ordered))
    return {e: int(perm[i] % n_folds) for i, e in enumerate(ordered)}


def per_env_counts(meta: list) -> Dict[str, int]:
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
    ap.add_argument("--seed", type=int, default=20260831)
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
        f"env count {len(envs)} must be divisible by {args.n_folds}")
    assert len(envs) >= 12, f"need >= 12 envs, got {len(envs)}"
    assert len(envs) // args.n_folds == 3, (
        f"spec requires exactly 3 held-out envs per fold, got {len(envs) // args.n_folds}")

    assign = fold_assignment(envs, args.n_folds, seed=args.seed)
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
        "rule": "grouped_4fold_env_disjoint_seeded_permutation_mod",
        "n_folds": args.n_folds,
        "seed": args.seed,
        "env_order": envs,
        "folds": folds,
        "single_use": True,
    }
    fold_sha = hashlib.sha256(
        json.dumps(fold_manifest, sort_keys=True).encode()).hexdigest()

    receipt = {
        "schema_id": "f5-split-seal.v1",
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "manifest_run_id": manifest.get("run_id"),
        "record_count": int(len(meta)),
        "envs": envs,
        "per_env_counts": env_counts,
        "n_folds": args.n_folds,
        "seed": args.seed,
        "folds": folds,
        "split_rule": "grouped_4fold_env_disjoint_seeded_permutation_mod",
        "fold_manifest_sha256": fold_sha,
        "single_use": True,
        "generator": f"f5_split_seal.py@{_sha256(str(Path(__file__).resolve()))[:16]}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("F5_SPLIT_SEAL_OK")


if __name__ == "__main__":
    main()
