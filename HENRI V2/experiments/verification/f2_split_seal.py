"""F2-M3 calibration/eval split sealer — GENERATION-ONLY process.

Consumes the flushed trajectory bank (npz+jsonl+manifest) and emits a sealed
episode-disjoint split receipt WITHOUT loading any checkpoint or model.

Discipline (heldout-sealing):
  - Episode-disjoint by ENV: calibration envs vs held-out env(s).
  - Sealed BEFORE any evaluation; the gates harness pins the receipt SHA.
  - single_use=true; the split is never re-generated after exposure.

Receipt: f2-split-seal.v1 with full SHA-256, env lists, counts, rule,
generator identity (this script's SHA), UTC timestamp, single_use flag.

Usage (remote, repo root):
  /venv/main/bin/python experiments/verification/f2_split_seal.py \
      --npz telemetry/f2_bank_capture/trajectories_<run_id>.npz \
      --jsonl telemetry/f2_bank_capture/trajectories_<run_id>.jsonl \
      --manifest telemetry/f2_bank_capture/trajectories_<run_id>_manifest.json \
      --heldout-envs wa30-ee6fef47 \
      --out telemetry/f2_bank_capture/split_seal.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--heldout-envs", required=True, help="comma-separated env IDs held out")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # provenance pin against the bank manifest
    npz_sha = _sha256(args.npz)
    jsonl_sha = _sha256(args.jsonl)
    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["data_source"] == "authorized", "bank must be authorized capture"
    assert manifest["npz_sha256"] == npz_sha, "npz hash mismatch vs manifest"
    assert manifest["jsonl_sha256"] == jsonl_sha, "jsonl hash mismatch vs manifest"

    bank = np.load(args.npz)
    psi = bank["psi"].astype(np.float32)
    onehot = bank["actions_onehot"].astype(np.uint8)
    meta = []
    with open(args.jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            meta.append(json.loads(line))
    assert len(meta) == psi.shape[0], "jsonl/meta row mismatch"

    envs = [str(m.get("env", "?")) for m in meta]
    heldout_set = {e.strip() for e in args.heldout_envs.split(",") if e.strip()}
    calib_mask = np.array([e not in heldout_set for e in envs])
    heldout_mask = ~calib_mask
    n_cal = int(calib_mask.sum())
    n_hold = int(heldout_mask.sum())
    assert n_cal > 0 and n_hold > 0, "need both calibration and held-out envs"
    assert set(np.unique(envs)) == set(envs), "env ids must be stable per row"

    calib_envs = sorted({e for e, m in zip(envs, calib_mask) if m})
    hold_envs = sorted(heldout_set)

    # split digests (index lists only — the split is defined by indices)
    idx_cal = np.where(calib_mask)[0].astype(np.int64)
    idx_hold = np.where(heldout_mask)[0].astype(np.int64)
    split_bytes = (
        idx_cal.tobytes()
        + idx_hold.tobytes()
        + json.dumps({"calib": calib_envs, "heldout": hold_envs}, sort_keys=True).encode()
    )
    split_sha = hashlib.sha256(split_bytes).hexdigest()

    # generator identity: SHA-256 of this script
    script_sha = _sha256(str(Path(__file__).resolve()))

    receipt = {
        "schema_id": "f2-split-seal.v1",
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "manifest_run_id": manifest.get("run_id"),
        "manifest_provenance": manifest.get("provenance"),
        "record_count": int(len(meta)),
        "envs": sorted(set(envs)),
        "calibration_envs": calib_envs,
        "heldout_envs": hold_envs,
        "n_calibration": int(n_cal),
        "n_heldout": int(n_hold),
        "split_rule": "episode_disjoint_by_env",
        "split_sha256": split_sha,
        "single_use": True,
        "generator": f"f2_split_seal.py@{script_sha[:16]}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("SPLIT_SEAL_OK")


if __name__ == "__main__":
    main()
