"""F3 capture finalizer — deterministic post-capture validation (CPU-only).

Reads the bank artifacts produced by the authorized runner capture
(production_arc_run.py with HENRI_ARC_TRAJECTORY_BANK=1) plus the run's
telemetry JSONL (per-env frame deltas) and emits the F3 capture receipt with
the entropy-gate verdict per SPEC-2026-08-29-F3-BROAD-BANK section 3:

  - bank entropy  H(A) = -sum_a p(a) ln p(a) >= 1.70 nats
  - per-action support N_a >= 30 for every action in the 7-action vocab
  - cross-env frame-delta variation CV_diff = std(mu_e)/mean(mu_e) > 0.20
  - per-env record floor >= 100 records/env (budget lower bound)

No model load, no GPU. Verdict:
  ENTROPY_GATE_PASS | BLOCKED_ENTROPY_GATE | BLOCKED_DIVERSITY | BLOCKED_RECORD_FLOOR

Usage (remote, repo root, after capture flush):
  /venv/main/bin/python "HENRI V2/experiments/verification/f3_capture_finalize.py" \
      --npz telemetry/f3_bank_capture/trajectories_<run_id>.npz \
      --jsonl telemetry/f3_bank_capture/trajectories_<run_id>.jsonl \
      --manifest telemetry/f3_bank_capture/trajectories_<run_id>_manifest.json \
      --telemetry telemetry/f3_bank_capture/production_run_<run_id>.jsonl \
      --out telemetry/f3_bank_capture/f3_capture_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
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


def compute_entropy_nats(counts: Sequence[int]) -> float:
    """Shannon entropy in nats over a histogram; 0 for an empty/single class."""
    total = sum(counts)
    if total <= 0:
        return 0.0
    h = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            h -= p * math.log(p)
    return h


def per_action_counts(actions_onehot: np.ndarray) -> Dict[str, int]:
    """Return {action_name: count} from the [N, A] uint8 one-hot bank."""
    sums = actions_onehot.sum(axis=0).astype(int)
    return {f"ACTION{i + 1}": int(sums[i]) for i in range(sums.shape[0])}


def per_env_counts(meta: List[Dict]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for m in meta:
        e = str(m.get("env", "?"))
        out[e] = out.get(e, 0) + 1
    return out


def per_env_frame_delta_mean(telemetry_rows: List[Dict]) -> Dict[str, float]:
    """Mean |Δgrid| per env from the run telemetry JSONL (grid_dist field)."""
    sums: Dict[str, float] = {}
    cnt: Dict[str, int] = {}
    for r in telemetry_rows:
        e = str(r.get("env", "?"))
        g = r.get("grid_dist")
        if g is None:
            continue
        sums[e] = sums.get(e, 0.0) + float(g)
        cnt[e] = cnt.get(e, 0) + 1
    return {e: sums[e] / cnt[e] for e in sums if cnt.get(e, 0) > 0}


def cv_diff(means: Dict[str, float]) -> float:
    if not means:
        return 0.0
    vals = list(means.values())
    mu = float(np.mean(vals))
    if mu <= 0:
        return 0.0
    return float(np.std(vals) / mu)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--telemetry", required=True)
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
    actions_onehot = bank["actions_onehot"].astype(np.uint8)
    action_names = [str(a) for a in bank["action_names"]]

    meta = []
    with open(args.jsonl, "r", encoding="utf-8") as fp:
        for line in fp:
            meta.append(json.loads(line))
    assert len(meta) == actions_onehot.shape[0], "jsonl/meta row mismatch"

    with open(args.telemetry, "r", encoding="utf-8") as fp:
        telemetry = [json.loads(l) for l in fp if l.strip()]

    env_counts = per_env_counts(meta)
    envs = sorted(env_counts)
    action_counts = per_action_counts(actions_onehot)
    n_actions = len(action_names)
    h_entropy = compute_entropy_nats(list(action_counts.values()))
    means = per_env_frame_delta_mean(telemetry)
    cv = cv_diff(means)

    verdicts = []
    if h_entropy < 1.70:
        verdicts.append("BLOCKED_ENTROPY_GATE")
    if any(action_counts.get(n, 0) < 30 for n in action_names):
        verdicts.append("BLOCKED_ENTROPY_GATE")
    if cv <= 0.20:
        verdicts.append("BLOCKED_DIVERSITY")
    if any(c < 100 for c in env_counts.values()):
        verdicts.append("BLOCKED_RECORD_FLOOR")
    verdict = verdicts[0] if verdicts else "ENTROPY_GATE_PASS"

    receipt = {
        "schema_id": "f3-capture-receipt.v1",
        "bank_manifest_schema": manifest.get("schema_id"),
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "manifest_run_id": manifest.get("run_id"),
        "record_count": int(len(meta)),
        "envs": envs,
        "per_env_counts": env_counts,
        "action_vocab": action_names,
        "per_action_counts": action_counts,
        "bank_entropy_nats": round(h_entropy, 4),
        "max_entropy_nats": round(math.log(n_actions), 4) if n_actions else 0.0,
        "per_env_frame_delta_mean": {k: round(v, 5) for k, v in means.items()},
        "cv_diff": round(cv, 4),
        "entropy_gate": {
            "h_min": 1.70,
            "n_a_min": 30,
            "cv_min": 0.20,
            "record_floor": 100,
        },
        "verdict": verdict,
        "generator": f"f3_capture_finalize.py@{_sha256(str(Path(__file__).resolve()))[:16]}",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print(f"CAPTURE_VERDICT={verdict}")


if __name__ == "__main__":
    main()
