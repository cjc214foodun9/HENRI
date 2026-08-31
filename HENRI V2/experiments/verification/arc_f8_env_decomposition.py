"""arc_f8_env_decomposition.py — Post-F8 Step 1: per-environment decodability decomposition.

Reads the VERIFIED F3v2 trajectory bank (hashes 9e3c01b4/1ca089b2 — npz psi
REAL float16 [N,65536], actions_onehot uint8 [N,7], jsonl env meta in row
order) and evaluates action decodability UN-POOLED per environment:

  for each environment e (grouped by jsonl 'env' field):
    - n_rows, n_classes, action histogram, H(A) nats
    - if n_classes < 2  ->  probes_skipped (trivial single-action env);
                            acc/margin = None (F8 trivial-env disclosure)
    - else 5-fold STRATIFIED (by class) cross-validation, seed 20260904:
        Probe LS   : min-norm least squares via thin SVD (lam=1e-6)
        Probe kNN3 : k-nearest-neighbors k=3 (the F8 acc_max operator)
        Acc(e) = mean CV accuracy over folds (per operator, plus best)
        Majority(e) = majority-class baseline on e's rows
        Margin(e) = Acc(e) - Majority(e)

Cross-task interference is removed by construction: folds are drawn ONLY
from e's own rows; no row of another environment ever enters train or test.

Receipt: schema f8-env-decomposition.v1, per-env scorecard, aggregate
statistics, bank hashes, seed, folds, device, git SHA. No wall-clock
timestamps (byte-deterministic).

Default-OFF: requires HENRI_F8_DECOMP=1 (mirrors HENRI_F8_PROBE).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from arc_f8_decodability_probe import (  # noqa: E402  (sealed F8 machinery)
    N_CLASSES,
    knn_predict,
    load_bank,
    majority_baseline,
    predict_ls,
    stratified_folds,
)

try:
    from arc_f8_decodability_probe import fit_minnorm_ls as _fit_ls  # noqa: E402
except Exception:  # pragma: no cover - defensive
    _fit_ls = None


def require_decomp_enabled() -> None:
    """Fail closed unless the decomposition flag is set (default-OFF)."""
    if os.environ.get("HENRI_F8_DECOMP") != "1":
        raise RuntimeError(
            "HENRI_F8_DECOMP != 1: per-environment decomposition is default-OFF"
        )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def action_entropy_nats(y: np.ndarray) -> float:
    """H(A) over the env's action-label histogram, in nats."""
    counts = np.bincount(y, minlength=int(y.max()) + 1).astype(np.float64)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log(p)).sum())


def run_env_cv(psi_e: np.ndarray, y_e: np.ndarray, n_folds: int, seed: int) -> dict:
    """5-fold stratified CV for one environment. Returns per-operator acc."""
    if int(y_e.max()) + 1 < 2:
        return {"probes_skipped": True}
    folds = stratified_folds(y_e, n_folds=n_folds, seed=seed)
    acc_ls, acc_knn3 = [], []
    for tr_idx, te_idx in folds:
        Xtr, ytr = psi_e[tr_idx], y_e[tr_idx]
        Xte, yte = psi_e[te_idx], y_e[te_idx]
        # Probe LS (min-norm least squares, thin SVD) — sealed contract:
        # fit_minnorm_ls(X, Y_onehot float [N, K]); labels are int64 -> onehot.
        if _fit_ls is not None:
            Y_onehot = np.eye(N_CLASSES, dtype=np.float32)[ytr]
            W = _fit_ls(Xtr, Y_onehot)
            pred_ls = predict_ls(Xte, W)
            acc_ls.append(float((pred_ls == yte).mean()))
        # Probe kNN-3 (F8 acc_max operator)
        pred_k = knn_predict(Xtr, ytr, Xte, k=3)
        acc_knn3.append(float((pred_k == yte).mean()))
    out = {"probes_skipped": False}
    if acc_ls:
        out["acc_ls"] = float(np.mean(acc_ls))
    out["acc_knn3"] = float(np.mean(acc_knn3))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank-npz", required=True)
    parser.add_argument("--bank-jsonl", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--receipt-out", required=True)
    args = parser.parse_args()

    require_decomp_enabled()

    bank = load_bank(args.bank_npz, args.bank_jsonl)
    psi, y, env_ids, env_names = (
        bank["psi"],
        bank["y"],
        bank["env_ids"],
        bank["env_names"],
    )
    n_envs = len(env_names)
    print(f"[decomp] bank: {psi.shape} rows x {psi.shape[1]} dims, {n_envs} envs")

    environments = []
    for env_idx, env_name in enumerate(env_names):
        rows = np.where(env_ids == env_idx)[0]
        psi_e = psi[rows]
        y_e = y[rows]
        entry = {
            "env": env_name,
            "n_rows": int(len(rows)),
            "n_classes": int(y_e.max()) + 1,
            "h_action_nats": action_entropy_nats(y_e),
        }
        majority = majority_baseline(y_e)
        entry["majority"] = majority
        res = run_env_cv(psi_e, y_e, args.n_folds, seed=args.seed + env_idx)
        entry.update(res)
        if not res["probes_skipped"]:
            accs = [entry[k] for k in ("acc_ls", "acc_knn3") if k in entry]
            entry["acc_best"] = max(accs)
            entry["margin_ls"] = (
                entry["acc_ls"] - majority if "acc_ls" in entry else None
            )
            entry["margin_knn3"] = entry["acc_knn3"] - majority
            entry["margin_best"] = entry["acc_best"] - majority
        else:
            entry["acc"] = None
            entry["margin"] = None
        environments.append(entry)
        print(
            f"[decomp] {env_name:8s} n={entry['n_rows']:4d} "
            f"K={entry['n_classes']} H={entry['h_action_nats']:.3f} "
            f"{'SKIPPED' if res['probes_skipped'] else 'acc_ls=' + str(round(entry.get('acc_ls', float('nan')), 4)) + ' acc_knn3=' + str(round(entry['acc_knn3'], 4))}"
        )

    nontriv = [e for e in environments if not e["probes_skipped"]]
    margins = [e["margin_best"] for e in nontriv]
    aggregate = {
        "n_envs": n_envs,
        "n_trivial": n_envs - len(nontriv),
        "trivial_envs": [e["env"] for e in environments if e["probes_skipped"]],
        "mean_margin_non_trivial": (
            float(np.mean(margins)) if margins else None
        ),
        "best_env": (
            max(nontriv, key=lambda e: e["margin_best"])["env"] if nontriv else None
        ),
        "best_margin": max(margins) if margins else None,
        "worst_margin": min(margins) if margins else None,
    }

    receipt = {
        "schema": "f8-env-decomposition.v1",
        "git_sha": git_sha(),
        "bank_npz_sha256": sha256_file(args.bank_npz),
        "bank_jsonl_sha256": sha256_file(args.bank_jsonl),
        "seed": args.seed,
        "n_folds": args.n_folds,
        "device": args.device,
        "environments": environments,
        "aggregate": aggregate,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = Path(args.receipt_out)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"[decomp] receipt written: {receipt_path}")
    print(f"[decomp] aggregate: {json.dumps(aggregate)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
