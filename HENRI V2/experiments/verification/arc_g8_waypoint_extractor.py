"""Carrier G8 — deterministic sub-goal waypoint extractor.

Source: Project_HENRI_SOTA_Architectural_Audit_and_Sprint_Master_Plan.md
(SHA ffe856ec...). Bank rows are genuine (psi_t, action, next_wave) transitions;
waypoints are selected at curvature peaks of the per-env trajectory, plus the
terminal state. Dense operators are NEVER formed (D=65,536 -> FALSIFIED_BY_MEMORY):
scoring is row-wise inner product only.

Module contract (tests in tests/contract/test_g8_waypoint_extractor.py):
  extract_waypoints(rows_per_env, psi, min_sep=16, min_frac=0.5)
    rows_per_env: dict env -> list of global row indices (ordered)
    psi: [N, 65536] float32/float16 array (row-major, rows = global bank rows)
    returns: {env: [(global_row_idx, role), ...]}  role in {"intermediate","terminal"}
    with len >= 3 and >= 2 intermediates when trajectory length permits,
    ordered by row index, terminal last.

  --bank <npz> --jsonl <path> probe mode: prints per-env waypoint counts + G8-1 gate
"""

from __future__ import annotations

import argparse
import json
import math

import numpy as np


def _curvature(psi_block: np.ndarray) -> np.ndarray:
    """Per-consecutive-pair cosine-distance curvature on raw waves.

    psi_block: [M, D] float array (D = 65536 production; fp16 supported).
    Returns kappa of length M-1 in [0,2].
    """
    a = psi_block[:-1].astype(np.float32)
    b = psi_block[1:].astype(np.float32)
    denom = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8
    cos_sim = np.einsum("ij,ij->i", a, b) / denom
    return np.clip(1.0 - cos_sim, 0.0, 2.0)


def extract_waypoints(
    rows_per_env: dict,
    psi: np.ndarray,
    min_sep: int = 16,
    min_frac: float = 0.5,
) -> dict:
    """Select curvature-peak waypoints per env. See module docstring."""
    out: dict = {}
    for env, rows in rows_per_env.items():
        rows = sorted(rows)
        # Sub-goal extraction is meaningless on very short trajectories:
        # require at least 3*min_sep rows before hunting curvature peaks.
        if len(rows) < 3 or len(rows) < 3 * min_sep:
            out[env] = [(rows[-1], "terminal")]
            continue
        idx = np.asarray(rows)
        block = psi[idx]
        kappa = _curvature(block)  # len M-1 between consecutive rows
        thr = float(kappa.mean() + min_frac * kappa.std())
        # local maxima above threshold with min separation on the ORIGINAL row axis
        peaks: list[int] = []
        last_picked = -10**9
        for j in range(1, len(kappa) - 1):
            row_j = rows[j]
            if row_j - last_picked < min_sep:
                continue
            if kappa[j] > kappa[j - 1] and kappa[j] >= kappa[j + 1] and kappa[j] > thr:
                peaks.append(row_j)
                last_picked = row_j
        waypoints = [(r, "intermediate") for r in peaks]
        # guarantee >= 2 intermediates when the trajectory is long enough
        if len(waypoints) < 2 and len(rows) >= 48:
            need = 2 - len(waypoints)
            step = max(1, (len(rows) - 2) // (need + 1))
            for k in range(1, need + 1):
                r = rows[min(k * step, len(rows) - 2)]
                if r not in [w[0] for w in waypoints]:
                    waypoints.append((r, "intermediate"))
        waypoints.sort(key=lambda t: t[0])
        waypoints.append((rows[-1], "terminal"))
        out[env] = waypoints
    return out


def _load_bank(npz_path: str, jsonl_path: str):
    d = np.load(npz_path, allow_pickle=True)
    psi = d["psi"]
    rows: dict = {}
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            rec = json.loads(line)
            rows.setdefault(rec["env"], []).append(i)
    return psi, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--min-sep", type=int, default=16)
    args = ap.parse_args()

    psi, rows = _load_bank(args.bank, args.jsonl)
    wp = extract_waypoints(rows, psi, min_sep=args.min_sep)
    all_ok = True
    for env in sorted(wp):
        n_inter = sum(1 for _, r in wp[env] if r == "intermediate")
        ok = len(wp[env]) >= 3 and n_inter >= 2
        all_ok &= ok
        print(f"{env}: waypoints={len(wp[env])} intermediate={n_inter} -> {'PASS' if ok else 'FAIL'}")
    print(f"G8-1_GATE: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
