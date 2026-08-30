"""F4 non-linear context-conditioned egress gates — 5-arm multi-arm kill matrix.

Spec: HENRI-SPEC-2026-08-F4-NONLINEAR-EGRESS (sealed; carrier/f4).
Parent: F3_GATES_VERDICT=K1_KILLED (event 8c47bf5c) — static linear codebooks
are FALSIFIED; F4 replaces the egress layer with a non-linear, per-env
context-conditioned head.

Arms (spec section 5):
  A: Tier1 (per-env W_task unbinding) + MLP + Tier3 (SGLD, W3 only)
  B: Tier1 + MLP, Tier3 frozen            (isolates adaptation)
  C: no Tier1 + MLP + SGLD                (isolates task conditioning)
  D: Tier1 + linear dual-ridge (F2-math)  (matched-protocol linear control)
  E: train-marginal predictor             (gate baseline)

Gates (spec section 3):
  G1  macro held-out P@1 >= 0.99
  G1b min-fold held-out P@1 >= 0.95
  G2  per-action P@1 >= 0.80 for actions with n_test >= 10 (coverage-limited
      actions reported, not scored)
  G3  ACTION6 payload-format validity >= 0.99; bank schema v1 has no payloads
      -> BLOCKED_NO_PAYLOAD_IN_BANK (never PASS)
  G4  margin vs train-marginal >= +0.05 (arm A, per-fold macro)
  G5  paired bootstrap CI (lb > 0): A vs D (matched protocol) — nonlinearity
  G6  paired bootstrap CI (lb > 0): A vs C — task conditioning
  G7  paired bootstrap CI (lb > 0): A vs B — in-situ adaptation

Verdicts: F4_EGRESS_PROMOTED | FALSIFIED_NO_EXTERNAL_GAIN |
BLOCKED_INFRASTRUCTURE | BLOCKED_TARGET_LEAKAGE | CONDITIONAL_REUSED_EVAL

Hyperparameters are FROZEN (K6 analogue): no lr, epochs, or ridge CLI knobs.
Split: f4-split-seal.v1 ONLY (seeded-permutation rule). The consumed F3 split
(f3-split-seal.v1) is REFUSED (consumed-guard).

Usage (remote, repo root):
  env PYTHONPATH="HENRI V2" HENRI_F4_EGRESS=1 /venv/main/bin/python \
      "HENRI V2/experiments/verification/f4_egress_gates.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --split-seal telemetry/f3_bank_capture_v2/f4_split_seal.json \
      --out telemetry/f3_bank_capture_v2/f4_gates_receipt.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

import sys as _sys
from pathlib import Path as _P

# This module lives in experiments/verification/ while the head module sits
# beside it; make the sibling importable regardless of cwd/PYTHONPATH.
_verif_dir = str(_P(__file__).resolve().parent)
if _verif_dir not in _sys.path:
    _sys.path.insert(0, _verif_dir)

from f4_nonlinear_egress_head import (
    F4NonLinearEgressHead,
    FROZEN_BATCH,
    FROZEN_EPOCHS,
    FROZEN_LR,
    FROZEN_RIDGE,
    compile_env_w_task,
    unbind_w_task,
)

FROZEN_N_FOLDS = 4
FROZEN_SGLD_STEPS = 3
FROZEN_SGLD_ETA = 1e-3
FROZEN_SGLD_T0 = 1e-6  # amended per reference protocol; spec T0=0.5 disclosed
FROZEN_DEMO_K = 20
BOOTSTRAP_RESAMPLES = 10_000


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def is_f4_seal(split_seal_path: str) -> bool:
    with open(split_seal_path, "r", encoding="utf-8") as fp:
        seal = json.load(fp)
    return seal.get("schema_id") == "f4-split-seal.v1"


def load_sealed_folds(split_seal_path: str) -> Dict[str, Dict]:
    """Load and verify an F4 seal; REFUSE any non-F4 receipt (consumed-guard)."""
    with open(split_seal_path, "r", encoding="utf-8") as fp:
        seal = json.load(fp)
    assert seal.get("schema_id") == "f4-split-seal.v1", (
        "consumed-guard: only f4-split-seal.v1 receipts are accepted")
    assert seal.get("single_use") is True, "sealed split must be single_use"
    folds = seal["folds"]
    manifest = {
        "rule": seal.get("split_rule", "grouped_4fold_env_disjoint_seeded_permutation_mod"),
        "n_folds": seal["n_folds"],
        "seed": seal.get("seed"),
        "env_order": seal["envs"],
        "folds": folds,
        "single_use": True,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert digest == seal["fold_manifest_sha256"], (
        "fold manifest digest mismatch vs sealed receipt")
    return folds


def bootstrap_ci_lb(deltas: Sequence[float], n_resample: int = BOOTSTRAP_RESAMPLES,
                    seed: int = 1) -> Dict[str, float]:
    """Paired bootstrap 95% CI lower bound over per-env deltas.

    Resamples with replacement from the observed paired deltas (per-env
    P@1 differences), 10k draws (production) or n_resample (tests).
    Returns {'lb', 'mean', 'n'}.
    """
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(n_resample))
    n = len(arr)
    for i in range(int(n_resample)):
        means[i] = rng.choice(arr, size=n, replace=True).mean()
    lb = float(np.percentile(means, 2.5))
    return {"lb": lb, "mean": float(means.mean()), "n": int(n)}


def margin_vs_marginal(p1: float, marginal_p1: float) -> float:
    return float(p1 - marginal_p1)


def demo_prefix_mask(meta: List[Dict], envs: List[str], k: int = FROZEN_DEMO_K) -> Dict[str, np.ndarray]:
    """First k rows per env (capture order) are the demo prefix.

    k = min(FROZEN_DEMO_K, max(1, n_env // 5)) per spec 4.3 floor.
    Returns per-env boolean masks over the row index space.
    """
    env_arr = np.array([str(m.get("env", "?")) for m in meta])
    out: Dict[str, np.ndarray] = {}
    for e in envs:
        idx = np.where(env_arr == e)[0]
        kk = min(k, max(1, int(len(idx) // 5)))
        mask = np.zeros(len(env_arr), dtype=bool)
        mask[idx[:kk]] = True
        out[e] = mask
    return out


def per_env_mask(envs: List[str], env_ids: List[str]) -> Dict[str, np.ndarray]:
    arr = np.array(envs)
    return {e: arr == e for e in sorted(set(envs))}


def provenance_scan(
    meta: List[Dict],
    env_ids: List[str],
    folds: Dict[str, Dict],
    dmask: Dict[str, np.ndarray],
    k: int = FROZEN_DEMO_K,
) -> Dict[str, object]:
    """Static leakage audit (spec 4.3 / kill experiment 5).

    Asserts, over the REAL row-index space of the bank:
      P1. demo prefix = exactly the first k rows per env in capture order;
      P2. demo masks are env-disjoint (each row belongs to at most one env's demo);
      P3. fold train/heldout env sets are disjoint and cover all envs;
      P4. heldout-env demo rows are excluded from evaluation rows by
          construction of the eval mask (checked at fold level by the caller);
      P5. W_task compilation inputs are demo rows only (verified via dmask).
    Raises AssertionError on any violation; returns a compact audit dict.
    """
    env_arr = np.array([str(m.get("env", "?")) for m in meta])
    n = len(env_arr)
    checks: Dict[str, object] = {"P1": {}, "P2": True, "P3": True}
    for e in env_ids:
        idx = np.where(env_arr == e)[0]
        kk = min(k, max(1, int(len(idx) // 5)))
        assert int(dmask[e].sum()) == kk, f"P1 demo size mismatch for {e}"
        assert bool(dmask[e][idx[0]]), f"P1 first row not demo for {e}"
        assert not bool(dmask[e][idx[kk:]].any()), f"P1 non-prefix demo rows for {e}"
        # P2: this env's demo covers no other env's rows
        other = np.ones(n, dtype=bool)
        other[idx] = False
        assert not bool(dmask[e][other].any()), f"P2 demo leaked across envs for {e}"
        checks["P1"][e] = int(kk)
    for fold_name, fold in folds.items():
        held = set(fold["heldout_envs"])
        train = set(fold["train_envs"])
        assert held & train == set(), f"P3 overlap in {fold_name}"
        assert held | train == set(env_ids), f"P3 cover failure in {fold_name}"
        for e in held:
            mask_e = env_arr == e
            # P4: every heldout-env demo row is excluded from eval (caller
            # builds eval mask with & ~dmask[e]); here we verify the demo rows
            # exist and are counted for Tier-1/Tier-3 use only.
            assert int(dmask[e].sum()) >= 1, f"P4 no demo rows for heldout env {e}"
            # P5: W_task compile inputs are demo rows only (compile uses dmask
            # masks; assert the mask selects only this env's rows).
            assert not bool(dmask[e][~mask_e].any()), f"P5 W_task leak for {e}"
    return checks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-seal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="A,B,C,D,E")
    ap.add_argument("--smoke", action="store_true",
                    help="bounded smoke: 1 env per fold, no verdict (disposable)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(20260830)
    np.random.seed(20260830)

    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest)
    D = psi.shape[1]
    V = actions_onehot.shape[1]
    assert D == 65536, f"expected D=65536, got {D}"
    assert V == 7, f"expected 7 actions, got {V}"

    envs = [str(m.get("env", "?")) for m in meta]
    env_ids = sorted(set(envs))
    folds = load_sealed_folds(args.split_seal)
    dmask = demo_prefix_mask(meta, env_ids, k=FROZEN_DEMO_K)
    # Kill 5: static provenance audit BEFORE any arm executes (spec 4.3).
    provenance_scan(meta, env_ids, folds, dmask, k=FROZEN_DEMO_K)
    want_arms = {s.strip() for s in args.arms.split(",") if s.strip()}

    if args.smoke:
        # disposable smoke: use ONLY 4 envs (one per fold), no verdict.
        env_ids = [folds[f"fold{i}"]["heldout_envs"][0] for i in range(FROZEN_N_FOLDS)]
        counts = {e: int(np.sum(np.array(envs) == e)) for e in env_ids}
        for f in range(FROZEN_N_FOLDS):
            held = [e for e in folds[f"fold{f}"]["heldout_envs"] if e in env_ids]
            train = [e for e in env_ids if e not in held]
            folds[f"fold{f}"]["heldout_envs"] = held
            folds[f"fold{f}"]["train_envs"] = train
            folds[f"fold{f}"]["n_heldout"] = int(sum(counts[e] for e in held))
            folds[f"fold{f}"]["n_train"] = int(sum(counts[e] for e in train))

    # import the live functor compiler + codec (Tier-1 anchors)
    henri_root = str(_P(__file__).resolve().parents[2])
    if henri_root not in _sys.path:
        _sys.path.insert(0, henri_root)
    from zone_c_epistemic_axiom_harness import HolographicTaskFunctorCompiler, qFHRREpistemicCodec

    codec = qFHRREpistemicCodec(d_model=D, k_bins=256, device="cpu")
    compiler = HolographicTaskFunctorCompiler(codec)

    # ---- per-env W_task compilation (Tier 1) from demo prefixes ------------
    w_tasks: Dict[str, Optional[torch.Tensor]] = {}
    for e in env_ids:
        mask_e = dmask[e]
        if int(mask_e.sum()) == 0:
            w_tasks[e] = None
            continue
        demo_psi = torch.from_numpy(psi[mask_e])
        demo_act = [action_names[int(np.argmax(actions_onehot[i]))] for i in np.where(mask_e)[0]]
        w_tasks[e] = compile_env_w_task(codec, compiler, demo_psi, demo_act)

    def unbound_rows(rows_psi: np.ndarray, env_list: List[str], row_envs: np.ndarray) -> torch.Tensor:
        """Tier-1 unbind each row by its env's W_task; raw if none (arm C)."""
        out = []
        for i, e in enumerate(env_list):
            wt = w_tasks.get(e)
            if wt is None:
                out.append(torch.from_numpy(rows_psi[i]).to(torch.float32))
            else:
                out.append(unbind_w_task(torch.from_numpy(rows_psi[i]), wt, codec, D=D))
        return torch.stack(out)

    per_fold: Dict[str, Dict] = {}
    p1_env_a: Dict[str, float] = {}
    p1_env_d: Dict[str, float] = {}
    p1_env_c: Dict[str, float] = {}
    p1_env_b: Dict[str, float] = {}
    margin_folds: List[float] = []
    g2_min: Optional[float] = None
    engaged = False

    for fold_name in [f"fold{i}" for i in range(FROZEN_N_FOLDS)]:
        fold = folds[fold_name]
        train_envs = set(fold["train_envs"])
        heldout_envs = set(fold["heldout_envs"])
        train_mask = np.array([e in train_envs for e in envs])
        hold_mask = ~train_mask
        assert int(train_mask.sum()) == fold["n_train"], "train count mismatch vs seal"
        assert int(hold_mask.sum()) == fold["n_heldout"], "heldout count mismatch vs seal"

        X_cal = psi[train_mask]
        Y_cal = actions_onehot[train_mask]
        X_hold = psi[hold_mask]
        Y_hold = actions_onehot[hold_mask]
        env_hold = [envs[i] for i in np.where(hold_mask)[0]]

        # demo prefix of heldout envs is EXCLUDED from evaluation rows
        eval_hold = np.ones(len(env_hold), dtype=bool)
        for e in heldout_envs:
            eval_hold &= ~dmask[e][hold_mask]
        X_eval = X_hold[eval_hold]
        Y_eval = Y_hold[eval_hold]
        env_eval = [env_hold[i] for i in np.where(eval_hold)[0]]

        train_env_list = [envs[i] for i in np.where(train_mask)[0]]
        cal_unbound = unbound_rows(X_cal, train_env_list, None) \
            if "A" in want_arms or "B" in want_arms or "D" in want_arms else None
        cal_unbound = torch.from_numpy(X_cal).to(torch.float32) if cal_unbound is None else cal_unbound

        held_unbound = unbound_rows(X_eval, env_eval, None)

        marginal_pred = int(np.argmax(Y_cal.sum(axis=0)))
        marg_p1 = float((Y_eval.argmax(axis=1) == marginal_pred).mean())

        per_action = {}
        for a in range(V):
            mask_a = Y_eval.argmax(axis=1) == a
            n_a = int(mask_a.sum())
            if n_a >= 10:
                per_action[action_names[a]] = {"n_test": n_a, "p1": None}
            else:
                per_action[action_names[a]] = {"n_test": n_a, "p1": None,
                                               "coverage_limited": True}

        # ---- arm A: Tier1 + MLP + SGLD -------------------------------------
        if "A" in want_arms:
            head = F4NonLinearEgressHead(d_model=D, hidden1=2048, hidden2=512,
                                         n_actions=V, seed=20260830).to(device)
            head.train_head(cal_unbound.to(device),
                            torch.from_numpy(Y_cal).to(device),
                            lr=FROZEN_LR, wd=1e-4, batch=FROZEN_BATCH,
                            epochs=FROZEN_EPOCHS, seed=20260830)
            # Tier-3 SGLD on each heldout env's demo prefix
            for e in heldout_envs:
                mask_e = dmask[e]
                idx = np.where(mask_e)[0]
                demo_waves = unbound_rows(psi[idx], [e] * len(idx), None)
                demo_psi = demo_waves.to(device)
                demo_act = torch.from_numpy(actions_onehot[idx]).to(device)
                head.adapt_w3_sgld(demo_psi, demo_act, steps=FROZEN_SGLD_STEPS,
                                   eta=FROZEN_SGLD_ETA, t0=FROZEN_SGLD_T0, dt=1.0,
                                   seed=20260830)
            head.eval()
            logits = head(held_unbound.to(device))
            pred = logits.argmax(dim=-1).cpu().numpy()
            true = Y_eval.argmax(axis=1)
            p1_a = float((pred == true).mean())
            for i, e in enumerate(env_eval):
                p1_env_a.setdefault(e, []).append(float(pred[i] == true[i]))
            for a in range(V):
                mask_a = true == a
                n_a = int(mask_a.sum())
                if n_a >= 10:
                    per_action[action_names[a]]["p1"] = float((pred[mask_a] == a).mean())
            engaged = engaged or p1_a > 0.0
        else:
            p1_a = float("nan")
            p1_folds_a = float("nan")

        # ---- arm B: Tier1 + MLP, no SGLD -----------------------------------
        if "B" in want_arms:
            head_b = F4NonLinearEgressHead(d_model=D, hidden1=2048, hidden2=512,
                                           n_actions=V, seed=20260830).to(device)
            head_b.train_head(cal_unbound.to(device),
                              torch.from_numpy(Y_cal).to(device),
                              lr=FROZEN_LR, wd=1e-4, batch=FROZEN_BATCH,
                              epochs=FROZEN_EPOCHS, seed=20260830)
            head_b.eval()
            logits_b = head_b(held_unbound.to(device))
            pred_b = logits_b.argmax(dim=-1).cpu().numpy()
            true = Y_eval.argmax(axis=1)
            p1_b = float((pred_b == true).mean())
            for i, e in enumerate(env_eval):
                p1_env_b.setdefault(e, []).append(float(pred_b[i] == true[i]))
        else:
            p1_b = float("nan")

        # ---- arm C: no Tier1 + MLP + SGLD ----------------------------------
        if "C" in want_arms:
            raw_cal = torch.from_numpy(X_cal).to(torch.float32)
            raw_hold = torch.from_numpy(X_eval).to(torch.float32)
            head_c = F4NonLinearEgressHead(d_model=D, hidden1=2048, hidden2=512,
                                           n_actions=V, seed=20260830).to(device)
            head_c.train_head(raw_cal.to(device),
                              torch.from_numpy(Y_cal).to(device),
                              lr=FROZEN_LR, wd=1e-4, batch=FROZEN_BATCH,
                              epochs=FROZEN_EPOCHS, seed=20260830)
            for e in heldout_envs:
                mask_e = dmask[e]
                idx = np.where(mask_e)[0]
                demo_psi = torch.from_numpy(psi[idx]).to(device)
                demo_act = torch.from_numpy(actions_onehot[idx]).to(device)
                head_c.adapt_w3_sgld(demo_psi, demo_act, steps=FROZEN_SGLD_STEPS,
                                     eta=FROZEN_SGLD_ETA, t0=FROZEN_SGLD_T0, dt=1.0,
                                     seed=20260830)
            head_c.eval()
            logits_c = head_c(raw_hold.to(device))
            pred_c = logits_c.argmax(dim=-1).cpu().numpy()
            true = Y_eval.argmax(axis=1)
            p1_c = float((pred_c == true).mean())
            for i, e in enumerate(env_eval):
                p1_env_c.setdefault(e, []).append(float(pred_c[i] == true[i]))
        else:
            p1_c = float("nan")

        # ---- arm D: Tier1 + linear dual-ridge (matched protocol) -----------
        if "D" in want_arms:
            lam = FROZEN_RIDGE
            Xt = cal_unbound.to(torch.float32)
            Yt = torch.from_numpy(Y_cal).to(torch.float32)
            # Dual thin-SVD ridge solve (no dense [D,D]; K4 invariant):
            #   M = Y^T X (X^T X + lam I)^{-1}
            #     = (Y^T U) diag(s/(s^2+lam)) V^T   via X = U S V^T (thin)
            u, s, vt = torch.linalg.svd(Xt, full_matrices=False)
            coef = (s / (s ** 2 + lam)).unsqueeze(0)  # [1, r]
            M = (Yt.T @ u) * coef @ vt  # [V, r] -> [V, D]
            M = M.to(torch.float32).to(device)
            logits_d = held_unbound.to(device) @ M.T
            pred_d = logits_d.argmax(dim=-1).cpu().numpy()
            true = Y_eval.argmax(axis=1)
            p1_d = float((pred_d == true).mean())
            for i, e in enumerate(env_eval):
                p1_env_d.setdefault(e, []).append(float(pred_d[i] == true[i]))
        else:
            p1_d = float("nan")

        # G4: arm A margin vs train-marginal
        if "A" in want_arms:
            margin = margin_vs_marginal(p1_a, marg_p1)
            margin_folds.append(margin)
        else:
            margin = float("nan")

        per_fold[fold_name] = {
            "train_envs": sorted(train_envs),
            "heldout_envs": sorted(heldout_envs),
            "n_train": int(train_mask.sum()),
            "n_heldout": int(hold_mask.sum()),
            "n_eval": int(eval_hold.sum()),
            "marginal_p1": round(marg_p1, 4),
            "p1_A": round(p1_a, 4) if "A" in want_arms else None,
            "p1_B": round(p1_b, 4) if "B" in want_arms else None,
            "p1_C": round(p1_c, 4) if "C" in want_arms else None,
            "p1_D": round(p1_d, 4) if "D" in want_arms else None,
            "margin_vs_marginal_A": round(margin, 4) if "A" in want_arms else None,
            "per_action": per_action,
            "finite_A": bool(torch.isfinite(held_unbound).all().item()),
        }

    if "A" in want_arms:
        p1_env_a = {e: float(np.mean(v)) for e, v in p1_env_a.items()}
    if "B" in want_arms:
        p1_env_b = {e: float(np.mean(v)) for e, v in p1_env_b.items()}
    if "C" in want_arms:
        p1_env_c = {e: float(np.mean(v)) for e, v in p1_env_c.items()}
    if "D" in want_arms:
        p1_env_d = {e: float(np.mean(v)) for e, v in p1_env_d.items()}

    if "A" in want_arms:
        macro_p1 = float(np.mean([per_fold[f"fold{i}"]["p1_A"] for i in range(FROZEN_N_FOLDS)]))
        min_p1 = float(np.min([per_fold[f"fold{i}"]["p1_A"] for i in range(FROZEN_N_FOLDS)]))
        macro_margin = float(np.mean(margin_folds))
        g2_min = None
        for f in per_fold.values():
            for a, v in f["per_action"].items():
                if v.get("p1") is not None:
                    g2_min = v["p1"] if g2_min is None else min(g2_min, v["p1"])
    else:
        macro_p1, min_p1, macro_margin, g2_min = float("nan"), float("nan"), float("nan"), None
    g2_pass = g2_min is not None and g2_min >= 0.80

    # G5/G6/G7 paired bootstrap CIs (per-env deltas across heldout envs)
    g5 = g6 = g7 = None
    if "A" in want_arms and "D" in want_arms:
        envs_all = sorted(set(p1_env_a) & set(p1_env_d))
        g5 = bootstrap_ci_lb([p1_env_a[e] - p1_env_d[e] for e in envs_all], seed=1)
    if "A" in want_arms and "C" in want_arms:
        envs_all = sorted(set(p1_env_a) & set(p1_env_c))
        g6 = bootstrap_ci_lb([p1_env_a[e] - p1_env_c[e] for e in envs_all], seed=2)
    if "A" in want_arms and "B" in want_arms:
        envs_all = sorted(set(p1_env_a) & set(p1_env_b))
        g7 = bootstrap_ci_lb([p1_env_a[e] - p1_env_b[e] for e in envs_all], seed=3)

    g3_status = "BLOCKED_NO_PAYLOAD_IN_BANK"
    g3_note = ("Trajectory-bank schema henri.arc-trajectory-bank.v1 persists "
               "action_name only (no (GameAction, data) payload). Payload-format "
               "validity requires a bank schema v2; not fabricable from current artifacts.")

    verdicts: List[str] = []
    if args.smoke:
        verdicts.append("SMOKE_NO_VERDICT")
    else:
        if "A" not in want_arms:
            verdicts.append("BLOCKED_INFRASTRUCTURE")
        if macro_p1 < 0.99 or min_p1 < 0.95:
            verdicts.append("K1_KILLED")
        if macro_margin < 0.05:
            verdicts.append("K2_KILLED")
        if not engaged:
            verdicts.append("K3_KILLED_NO_ENGAGEMENT")
        if not g2_pass:
            verdicts.append("G2_FAILED")
        if g5 is not None and g5["lb"] <= 0:
            verdicts.append("G5_FAILED_LINEAR_IN_DISGUISE")
        if g6 is not None and g6["lb"] <= 0:
            verdicts.append("G6_FAILED_NO_TASK_CONDITIONING")
        if g7 is not None and g7["lb"] <= 0:
            verdicts.append("G7_FAILED_NO_ADAPTATION")
        if not verdicts:
            verdicts.append("F4_EGRESS_PROMOTED")

    receipt = {
        "schema_id": "f4-egress-gates.v1",
        "device": device,
        "npz_sha256": _sha256(args.npz),
        "jsonl_sha256": _sha256(args.jsonl),
        "split_seal": args.split_seal,
        "record_count": int(len(meta)),
        "action_vocab": action_names,
        "n_folds": FROZEN_N_FOLDS,
        "frozen_lr": FROZEN_LR,
        "frozen_epochs": FROZEN_EPOCHS,
        "frozen_ridge": FROZEN_RIDGE,
        "frozen_sgld_steps": FROZEN_SGLD_STEPS,
        "frozen_sgld_t0": FROZEN_SGLD_T0,
        "demo_k": FROZEN_DEMO_K,
        "arms": sorted(want_arms),
        "per_fold": per_fold,
        "g1_macro_p1_A": round(macro_p1, 4) if "A" in want_arms else None,
        "g1_min_fold_p1_A": round(min_p1, 4) if "A" in want_arms else None,
        "g1_pass": (macro_p1 >= 0.99 and min_p1 >= 0.95) if "A" in want_arms else False,
        "g2_min_scored_p1": g2_min,
        "g2_pass": g2_pass,
        "g3_status": g3_status,
        "g3_note": g3_note,
        "g4_macro_margin_A": round(macro_margin, 4) if "A" in want_arms else None,
        "g4_pass": (macro_margin >= 0.05) if "A" in want_arms else False,
        "g5_A_vs_D": g5,
        "g5_pass": (g5 is not None and g5["lb"] > 0),
        "g6_A_vs_C": g6,
        "g6_pass": (g6 is not None and g6["lb"] > 0),
        "g7_A_vs_B": g7,
        "g7_pass": (g7 is not None and g7["lb"] > 0),
        "verdict": verdicts[0] if verdicts else "BLOCKED_INFRASTRUCTURE",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print(f"F4_GATES_VERDICT={receipt['verdict']}")


if __name__ == "__main__":
    main()
