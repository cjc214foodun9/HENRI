"""Carrier F8.1 — Action-Conditioned Transition Derivative Ingress probe.

Falsification probe on the F3 v2 trajectory bank: does the transition
derivative DeltaPsi_t = irfft(rfft(next_wave[t]) * conj(rfft(psi[t])))
contain a decodable action channel, after the static-wave family (F4-F8)
was measured as policy-inert (F8_INDETERMINATE / CASE C; per-env margins
mean -0.0114)? Four probe families (P1 min-norm LS, P2 L2-logistic, P3 MLP,
P4 cosine k-NN k=3), 10-fold stratified CV, seed 20260905, gates G1-G4
verbatim from the directive, ternary verdict. Default-OFF
(HENRI_F8_1_PROBE=1). Diagnostic only — no production path is trained or
modified. See docs/spec/f8_1_derivative_probe_preregistration.md.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

D = 65_536  # production wave dimension
N_CLASSES = 7  # action labels 1..7

# Reuse the sealed F8 probe machinery (arc_f8_decodability_probe.py at
# HENRI V2/ root; production runs set PYTHONPATH="HENRI V2").
try:
    from arc_f8_decodability_probe import (  # noqa: F401
        fit_logistic,
        fit_minnorm_ls,
        fit_mlp,
        knn_predict,
        majority_baseline,
        predict_logistic,
        predict_ls,
        predict_mlp,
        stratified_folds,
    )
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from arc_f8_decodability_probe import (  # noqa: F401
        fit_logistic,
        fit_minnorm_ls,
        fit_mlp,
        knn_predict,
        majority_baseline,
        predict_logistic,
        predict_ls,
        predict_mlp,
        stratified_folds,
    )


def require_f8_1_enabled() -> None:
    """Fail closed unless the F8.1 probe flag is set (default-OFF)."""
    if os.environ.get("HENRI_F8_1_PROBE") != "1":
        raise RuntimeError(
            "HENRI_F8_1_PROBE != 1: Carrier F8.1 probe is default-OFF"
        )


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(npz_path: str, jsonl_path: str) -> dict:
    """Load and validate the REAL bank schema (F8 amendment 85da8bf4).

    OBSERVED schema: psi/next_wave real float16 [N, 65536],
    actions_onehot uint8 [N, 7], jsonl rows with env/step/action_name.
    Valid transition pairs: row t has a successor within the same env
    (envs[t+1] == envs[t]); the last row of each env and the final row are
    excluded. The causally paired successor is the STORED next_wave[t],
    which equals literal row t+1 in only 1225/1524 pairs (OBSERVED 2026-08-31).
    """
    data = np.load(npz_path, allow_pickle=False)
    psi = data["psi"]
    nxt = data["next_wave"]
    actions_onehot = data["actions_onehot"]
    if psi.ndim != 2 or nxt.ndim != 2:
        raise ValueError(f"psi/next_wave must be 2-D [N, D], got {psi.shape}/{nxt.shape}")
    if psi.shape != nxt.shape:
        raise ValueError(f"psi/next_wave shape mismatch: {psi.shape} != {nxt.shape}")
    if np.iscomplexobj(psi) or np.iscomplexobj(nxt):
        raise ValueError("psi/next_wave must be REAL (bank is float16 real-domain)")
    if actions_onehot.shape != (len(psi), N_CLASSES):
        raise ValueError(
            f"actions_onehot must be [N, {N_CLASSES}], got {actions_onehot.shape}"
        )
    y = np.argmax(actions_onehot.astype(np.float32), axis=1).astype(np.int64)
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        meta = [json.loads(line) for line in fp]
    if len(meta) != len(psi):
        raise ValueError(f"jsonl/meta row mismatch: {len(meta)} != {len(psi)}")
    envs = [str(m["env"]) for m in meta]
    steps = [m.get("step") for m in meta]

    n = len(psi)
    valid = np.zeros(n, dtype=bool)
    for t in range(n - 1):
        if envs[t + 1] == envs[t]:
            valid[t] = True
    valid_idx = np.where(valid)[0].astype(np.int64)

    # diagnostics
    boundaries = sum(1 for t in range(n - 1) if envs[t + 1] != envs[t])
    no_change = int(
        sum(1 for t in valid_idx if np.array_equal(nxt[t], psi[t]))
    )
    diff_pairs = int(
        sum(1 for t in valid_idx if not np.array_equal(nxt[t], psi[t + 1]))
    )
    step_gaps = int(
        sum(
            1
            for t in valid_idx
            if steps[t + 1] is not None
            and steps[t] is not None
            and steps[t + 1] != steps[t] + 1
        )
    )
    return {
        "psi": psi.astype(np.float32),
        "next": nxt.astype(np.float32),
        "y": y,
        "envs": envs,
        "steps": steps,
        "valid_idx": valid_idx,
        "n_total": n,
        "n_valid": int(valid_idx.shape[0]),
        "n_excluded": n - int(valid_idx.shape[0]),
        "boundaries": boundaries,
        "no_change": no_change,
        "diff_pairs": diff_pairs,
        "step_gaps": step_gaps,
        "npz_sha256": sha256_file(npz_path),
        "jsonl_sha256": sha256_file(jsonl_path),
    }


def transition_derivative(
    next_wave: np.ndarray,
    psi: np.ndarray,
    chunk: int = 256,
    device: str = "auto",
) -> np.ndarray:
    """DeltaPsi = irfft(rfft(next) * conj(rfft(psi)), n=D), real domain.

    Accepts [B, D] float32 numpy arrays; returns [B, D] float32 numpy.
    Chunked along rows to bound device memory. Circular cross-correlation
    in the real domain (no identity cancellation assumed — bank is real).
    """
    dev = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    nxt = torch.from_numpy(np.ascontiguousarray(next_wave, dtype=np.float32))
    psi_t = torch.from_numpy(np.ascontiguousarray(psi, dtype=np.float32))
    if dev == "cuda":
        nxt, psi_t = nxt.cuda(), psi_t.cuda()
    outs = []
    with torch.no_grad():
        for i in range(0, len(nxt), chunk):
            b_nxt = nxt[i : i + chunk]
            b_psi = psi_t[i : i + chunk]
            d = torch.fft.irfft(
                torch.fft.rfft(b_nxt, dim=-1)
                * torch.conj(torch.fft.rfft(b_psi, dim=-1)),
                n=b_psi.shape[-1],
                dim=-1,
            )
            outs.append(d.cpu().numpy())
    return np.concatenate(outs, axis=0)


def action_entropy_stats(y: np.ndarray) -> tuple[float, int, float]:
    """Natural-log entropy H(A), min per-class count, majority fraction."""
    vals, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    H = float(-(probs * np.log(probs)).sum())
    return H, int(counts.min()), float(counts.max() / len(y))


def _onehot(y: np.ndarray) -> np.ndarray:
    return np.eye(int(y.max()) + 1, dtype=np.float32)[y]


def _cv_probe(
    X: np.ndarray,
    y: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    probe: str,
    device: str,
    seed: int,
) -> float:
    accs = []
    for tr, te in folds:
        if probe == "ls":
            W = fit_minnorm_ls(X[tr], _onehot(y[tr]))
            pred = predict_ls(X[te], W)
        elif probe == "logistic":
            Wb = fit_logistic(X[tr], y[tr], device=device, seed=seed)
            pred = predict_logistic(X[te], Wb)
        elif probe == "mlp":
            model = fit_mlp(X[tr], y[tr], device=device, seed=seed)
            pred = predict_mlp(model, X[te])
        elif probe == "knn3":
            pred = knn_predict(X[tr], y[tr], X[te], k=3)
        else:
            raise ValueError(f"unknown probe {probe}")
        accs.append(float((pred == y[te]).mean()))
    return float(np.mean(accs))


def _train_acc(X: np.ndarray, y: np.ndarray, probe: str, device: str, seed: int) -> float:
    if probe == "ls":
        W = fit_minnorm_ls(X, _onehot(y))
        pred = predict_ls(X, W)
    elif probe == "logistic":
        Wb = fit_logistic(X, y, device=device, seed=seed)
        pred = predict_logistic(X, Wb)
    elif probe == "mlp":
        model = fit_mlp(X, y, device=device, seed=seed)
        pred = predict_mlp(model, X)
    else:
        return float("nan")  # kNN train acc excluded from G1 by spec
    return float((pred == y).mean())


def build_receipt(**kw) -> dict:
    """Assemble the f8-1-derivative-probe.v1 receipt from gauntlet results."""
    receipt = {
        "schema": "f8-1-derivative-probe.v1",
        "git_sha": kw["git_sha"],
        "bank": {"npz_sha256": kw["npz_sha256"], "jsonl_sha256": kw["jsonl_sha256"]},
        "n_total": kw["n_total"],
        "n_valid": kw["n_valid"],
        "n_excluded": kw["n_excluded"],
        "boundaries": kw["boundaries"],
        "no_change": kw["no_change"],
        "diff_pairs": kw["diff_pairs"],
        "step_gaps": kw["step_gaps"],
        "H_nats": kw["H_nats"],
        "min_Na": kw["min_Na"],
        "majority": kw["majority"],
        "seed": kw["seed"],
        "n_folds": kw["n_folds"],
        "device": kw["device"],
        "probes": kw["probes"],
        "static": kw["static"],
        "gates": kw["gates"],
        "verdict": kw["verdict"],
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    return receipt


def run_gauntlet(
    bank_npz: str,
    bank_jsonl: str,
    device: str = "auto",
    n_folds: int = 10,
    seed: int = 20260905,
    git_sha: str = "unknown",
    receipt_out: str | None = None,
) -> dict:
    """Full F8.1 gauntlet: derivative, folds, probes, gates, verdict."""
    t0 = time.time()
    require_f8_1_enabled()
    data = load_bank(bank_npz, bank_jsonl)
    vi = data["valid_idx"]
    X_delta = transition_derivative(data["next"][vi], data["psi"][vi], device=device)
    X_static = data["psi"][vi]
    y = data["y"][vi]

    H, min_na, majority = action_entropy_stats(y)
    folds = stratified_folds(y, n_folds=n_folds, seed=seed)

    probes = {}
    static = {}
    for p in ("ls", "logistic", "mlp", "knn3"):
        probes[p] = {"cv": _cv_probe(X_delta, y, folds, p, device, seed)}
        probes[p]["train"] = _train_acc(X_delta, y, p, device, seed)
        static[p] = {"cv": _cv_probe(X_static, y, folds, p, device, seed)}

    best_cv = max(v["cv"] for v in probes.values())
    best_static = max(v["cv"] for v in static.values())
    best_train = max(v["train"] for v in probes.values() if not np.isnan(v["train"]))

    gates = {
        "G1": {
            "criterion": "train acc >= 0.9000",
            "value": best_train,
            "pass": best_train >= 0.9000,
            "kill": "K1",
        },
        "G2": {
            "criterion": "10-fold CV >= 0.6500",
            "value": best_cv,
            "pass": best_cv >= 0.6500,
            "kill": "K2",
        },
        "G3": {
            "criterion": "CV(dPsi) - CV(static) >= +0.2000",
            "value": best_cv - best_static,
            "pass": best_cv - best_static >= 0.2000,
            "kill": "K3",
        },
        "G4": {
            "criterion": "H(A) >= 1.70 nats, min Na >= 30",
            "H_nats": H,
            "min_Na": min_na,
            "pass": H >= 1.70 and min_na >= 30,
            "kill": "K4",
        },
    }

    g1, g2, g3, g4 = gates["G1"]["pass"], gates["G2"]["pass"], gates["G3"]["pass"], gates["G4"]["pass"]
    if g1 and g2 and g3 and g4:
        verdict = "F8.1_TRANSITION_DERIVATIVE_VERIFIED"
    elif g4 and (not g1 or not g3):
        verdict = "F8.1_REPRESENTATION_FAMILY_FALSIFIED"
    else:
        verdict = "F8.1_INDETERMINATE"

    receipt = build_receipt(
        git_sha=git_sha,
        npz_sha256=data["npz_sha256"],
        jsonl_sha256=data["jsonl_sha256"],
        n_total=data["n_total"],
        n_valid=data["n_valid"],
        n_excluded=data["n_excluded"],
        boundaries=data["boundaries"],
        no_change=data["no_change"],
        diff_pairs=data["diff_pairs"],
        step_gaps=data["step_gaps"],
        H_nats=H,
        min_Na=min_na,
        majority=majority,
        seed=seed,
        n_folds=n_folds,
        device=device,
        probes=probes,
        static=static,
        gates=gates,
        verdict=verdict,
    )
    receipt["elapsed_s"] = round(time.time() - t0, 2)
    if receipt_out:
        Path(receipt_out).parent.mkdir(parents=True, exist_ok=True)
        with open(receipt_out, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
    return receipt


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Carrier F8.1 transition-derivative gauntlet")
    ap.add_argument("--bank-npz", required=True)
    ap.add_argument("--bank-jsonl", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out-dir", default="/tmp/henri_f8_1_derivative/")
    ap.add_argument("--receipt-out", default="/tmp/henri_f8_1_derivative/f8_1_gates_receipt.json")
    args = ap.parse_args()

    import subprocess

    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30
            ).stdout.strip()
        )
    except Exception:
        sha = "unknown"
    receipt = run_gauntlet(
        bank_npz=args.bank_npz,
        bank_jsonl=args.bank_jsonl,
        device=args.device,
        n_folds=args.n_folds,
        seed=args.seed,
        git_sha=sha,
        receipt_out=args.receipt_out,
    )
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
