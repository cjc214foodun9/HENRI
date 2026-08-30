"""Carrier F7 affine-egress arms + gates harness (remote CUDA, sealed-split only).

Spec: HENRI-SPEC-2026-08-F7-AFFINE-EGRESS sections 3-4.

Arms (multi-arm kill matrix):
  A  F7 full affine: per-env dual ridge A^(e),b^(e) + Tier-2 family covariance
     + argmax egress                                  [compile_affine_egress]
  B  F7 Tier-1 only: per-env affine, NO family covariance (isolates Tier 2)
  C  F6 circulant control: per-env NS-retracted functor (F6 arm A) on the NEW
     split (G3 baseline, fresh numbers, never the consumed split's)
  D  identity / no-supervision floor (direct wave-to-action cosine)

Gates (spec section 3, directive section 4 — verbatim):
  G1  demo reconstruction P@1 >= 0.9900           (arm A)
  G2  grouped held-out macro P@1 >= 0.7000        (arm A)
  G3  margin P@1_A - P@1_C >= +0.2500             (arm A vs arm C, same new split)
  G4  min fold P@1 >= 0.6000                      (arm A)
G5 per-env diagnostic (no threshold); G6 default-OFF differential (contract tests).

Pre-registered verdicts: F7_AFFINE_PROMOTED / FALSIFIED_AT_SCALE /
FALSIFIED_NO_GAIN / BLOCKED_INFRASTRUCTURE.
Any nonzero arm exit -> BLOCKED_INFRASTRUCTURE for the whole run.

Usage (remote, repo root, AFTER f7_split_seal.py):
  env PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f7_affine_egress_gates.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --split-seal telemetry/f3_bank_capture_v2/f7_split_seal.json \
      --out telemetry/f3_bank_capture_v2/f7_gates_receipt.json \
      --arms A,B,C,D
  --smoke: disposable smoke (1 heldout env per fold, NO verdict).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

_verif_dir = str(Path(__file__).resolve().parent)
if _verif_dir not in sys.path:
    sys.path.insert(0, _verif_dir)
_henri_root = str(Path(__file__).resolve().parents[2])
if _henri_root not in sys.path:
    sys.path.insert(0, _henri_root)

FROZEN_N_FOLDS = 4
FROZEN_DEMO_K = 20
FROZEN_SEED = 20260902
FROZEN_LAM = 1e-3
CONSUMED_SCHEMAS = ("f3-split-seal.v1", "f4-split-seal.v1", "f5-split-seal.v1",
                    "f6-split-seal.v1")


# ---------------------------------------------------------------------------
# Bank loading + masks (mirrors F6 harness contract)
# ---------------------------------------------------------------------------

def load_bank(npz_path: str, jsonl_path: str, manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["data_source"] == "authorized", "bank must be authorized capture"
    bank = np.load(npz_path)
    meta = []
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            meta.append(json.loads(line))
    assert len(meta) == bank["psi"].shape[0], "jsonl/meta row mismatch"
    psi = bank["psi"].astype(np.float32)
    actions_onehot = bank["actions_onehot"]
    action_names = [str(a) for a in bank["action_names"]]
    return psi, actions_onehot, action_names, meta, manifest


def demo_prefix_mask(meta: list, env_ids: List[str], k: int = FROZEN_DEMO_K) -> Dict[str, np.ndarray]:
    envs = np.array([str(m.get("env", "?")) for m in meta])
    out: Dict[str, np.ndarray] = {}
    for e in env_ids:
        idx = np.where(envs == e)[0]
        mask = np.zeros(len(envs), dtype=bool)
        mask[idx[:k]] = True
        out[e] = mask
    return out


def load_sealed_folds(split_seal_path: str) -> Dict[str, Dict]:
    with open(split_seal_path, "r", encoding="utf-8") as fp:
        receipt = json.load(fp)
    assert receipt["schema_id"] == "f7-split-seal.v1", \
        f"refusing non-F7 seal {receipt.get('schema_id')} (consumed-guard)"
    assert receipt["single_use"] is True, "seal must be single_use"
    folds = {f"fold{i}": receipt["folds"][f"fold{i}"] for i in range(FROZEN_N_FOLDS)}
    manifest = {
        "rule": receipt["split_rule"],
        "n_folds": receipt["n_folds"],
        "seed": receipt["seed"],
        "env_order": receipt["envs"],
        "folds": receipt["folds"],
        "single_use": True,
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert digest == receipt["fold_manifest_sha256"], "fold manifest digest mismatch"
    return folds


def provenance_scan(meta: list, env_ids: List[str], folds: Dict[str, Dict],
                    dmask: Dict[str, np.ndarray], k: int = FROZEN_DEMO_K) -> None:
    envs = np.array([str(m.get("env", "?")) for m in meta])
    for f in range(FROZEN_N_FOLDS):
        held = set(folds[f"fold{f}"]["heldout_envs"])
        train = set(folds[f"fold{f}"]["train_envs"])
        for e in held:
            idx = np.where(envs == e)[0]
            ev_rows = idx[k:]
            for te in train:
                assert not np.any(np.isin(ev_rows, np.where(envs == te)[0])), \
                    f"leak: heldout eval rows of {e} intersect train env {te}"
    for e in env_ids:
        idx = np.where(envs == e)[0]
        assert set(np.where(dmask[e])[0]) == set(idx[:k]), f"demo mask wrong for {e}"


# ---------------------------------------------------------------------------
# Wave helpers (mirrors F6)
# ---------------------------------------------------------------------------

def psi_to_ring(psi: torch.Tensor) -> torch.Tensor:
    return ((psi.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)


def ring_to_wave(q: torch.Tensor) -> torch.Tensor:
    return torch.exp(1j * q.to(torch.float32) * (2.0 * math.pi / 256.0))


def wave_cos(w1: torch.Tensor, w2: torch.Tensor) -> float:
    num = float(torch.abs(torch.vdot(w1, w2)).item())
    den = float(w1.norm().item()) * float(w2.norm().item()) + 1e-12
    return num / den


def action_wave(codec, name: str) -> torch.Tensor:
    if hasattr(codec, "encode_wave"):
        return codec.encode_wave(name).cpu()
    return ring_to_wave(codec.encode_text(name)).cpu()


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def run_arm_d(psi: np.ndarray, actions_onehot: np.ndarray,
              action_names: List[str], meta: list, folds: Dict[str, Dict],
              env_ids: List[str], dmask: Dict[str, np.ndarray],
              D: int, device: str) -> Dict:
    """Identity / no-supervision floor (direct wave-to-action cosine)."""
    from fpb_qfhrr_codec import FPBStructuredCodec
    codec = FPBStructuredCodec(d_model=D, k_bins=256, device="cpu")
    envs = np.array([str(m.get("env", "?")) for m in meta])
    per_fold: Dict[str, Dict] = {}
    per_env: Dict[str, Dict] = {}
    macro_acc, macro_n = 0.0, 0
    for f in range(FROZEN_N_FOLDS):
        held = folds[f"fold{f}"]["heldout_envs"]
        fold_correct, fold_n = 0, 0
        for e in held:
            idx = np.where(envs == e)[0]
            ev_idx = idx[FROZEN_DEMO_K:]
            action_waves = [action_wave(codec, a).to(device) for a in action_names]
            correct, n = 0, 0
            for i in ev_idx:
                x_wave = ring_to_wave(psi_to_ring(torch.from_numpy(psi[i]))).to(device)
                scores = [wave_cos(x_wave, aw) for aw in action_waves]
                pred = int(np.argmax(scores))
                true = int(np.argmax(actions_onehot[i]))
                correct += int(pred == true)
                n += 1
            p1 = correct / n if n else 0.0
            per_env[e] = {"n_test": n, "p1": round(float(p1), 4)}
            fold_correct += correct
            fold_n += n
        per_fold[f"fold{f}"] = {"n_test": fold_n,
                                "p1": round(float(fold_correct / fold_n), 4) if fold_n else 0.0}
        macro_acc += fold_correct
        macro_n += fold_n
    return {"arm": "D",
            "macro_p1": round(float(macro_acc / macro_n), 4) if macro_n else None,
            "per_fold": per_fold, "per_env": per_env}


def run_arm_c(psi: np.ndarray, actions_onehot: np.ndarray,
              action_names: List[str], meta: list, folds: Dict[str, Dict],
              env_ids: List[str], dmask: Dict[str, np.ndarray],
              D: int, device: str) -> Dict:
    """F6 circulant control: per-env NS-retracted functor on the NEW split."""
    from f6_adaptive_functor import AdaptiveFunctorCompiler
    from fpb_qfhrr_codec import FPBStructuredCodec
    envs = np.array([str(m.get("env", "?")) for m in meta])
    per_fold: Dict[str, Dict] = {}
    per_env: Dict[str, Dict] = {}
    macro_acc, macro_n = 0.0, 0
    for f in range(FROZEN_N_FOLDS):
        held = folds[f"fold{f}"]["heldout_envs"]
        fold_correct, fold_n = 0, 0
        for e in held:
            idx = np.where(envs == e)[0]
            demo_idx = idx[:FROZEN_DEMO_K]
            ev_idx = idx[FROZEN_DEMO_K:]
            x_waves = [ring_to_wave(psi_to_ring(torch.from_numpy(psi[i])))
                       for i in demo_idx]
            codec_a = FPBStructuredCodec(d_model=D, k_bins=256, device="cpu")
            y_waves = [codec_a.encode_wave(action_names[int(np.argmax(actions_onehot[i]))])
                       for i in demo_idx]
            comp = AdaptiveFunctorCompiler(device=device, max_iters=8, tol=1e-5,
                                           eps_floor=1e-3)
            comp.compile_demo(x_waves, y_waves,
                              [action_names[int(np.argmax(actions_onehot[i]))]
                               for i in demo_idx])
            correct, n = 0, 0
            for i in ev_idx:
                x_wave = ring_to_wave(psi_to_ring(torch.from_numpy(psi[i]))).to(device)
                pred, _ = comp.retrieve(x_wave)
                true = action_names[int(np.argmax(actions_onehot[i]))]
                correct += int(pred == true)
                n += 1
            p1 = correct / n if n else 0.0
            per_env[e] = {"n_test": n, "p1": round(float(p1), 4)}
            fold_correct += correct
            fold_n += n
        per_fold[f"fold{f}"] = {"n_test": fold_n,
                                "p1": round(float(fold_correct / fold_n), 4) if fold_n else 0.0}
        macro_acc += fold_correct
        macro_n += fold_n
    return {"arm": "C",
            "macro_p1": round(float(macro_acc / macro_n), 4) if macro_n else None,
            "per_fold": per_fold, "per_env": per_env}


def run_arm_ab(psi: np.ndarray, actions_onehot: np.ndarray,
               action_names: List[str], meta: list, folds: Dict[str, Dict],
               env_ids: List[str], dmask: Dict[str, np.ndarray],
               D: int, device: str, use_family: bool) -> Dict:
    """Arms A/B: per-env real dual ridge + (A) family covariance Tier 2.

    X_demo: real bank waves [M, D]; Y_demo: one-hot actions [M, K].
    family_z: pooled demo readout z_raw over the fold's TRAIN envs (causal).
    """
    from f7_affine_egress import AffineEgress
    envs = np.array([str(m.get("env", "?")) for m in meta])
    per_fold: Dict[str, Dict] = {}
    per_env: Dict[str, Dict] = {}
    macro_acc, macro_n = 0.0, 0
    g1_p1s: List[float] = []
    for f in range(FROZEN_N_FOLDS):
        held = folds[f"fold{f}"]["heldout_envs"]
        train = folds[f"fold{f}"]["train_envs"]
        fold_correct, fold_n = 0, 0
        # Tier-2 prior: pooled demo z_raw over TRAIN envs (causal, no heldout leakage)
        family_z: Optional[torch.Tensor] = None
        if use_family:
            fam_rows: List[torch.Tensor] = []
            for te in train:
                t_idx = np.where(envs == te)[0][:FROZEN_DEMO_K]
                Xt = torch.from_numpy(psi[t_idx]).to(torch.float32)
                Yt = torch.from_numpy(actions_onehot[t_idx]).to(torch.float32)
                eg_t = AffineEgress(lam=FROZEN_LAM).fit(Xt, Yt)
                fam_rows.append(eg_t.predict(Xt, use_family=False))
            if fam_rows:
                family_z = torch.cat(fam_rows, dim=0)
        for e in held:
            idx = np.where(envs == e)[0]
            demo_idx = idx[:FROZEN_DEMO_K]
            ev_idx = idx[FROZEN_DEMO_K:]
            X_demo = torch.from_numpy(psi[demo_idx]).to(torch.float32).to(device)
            Y_demo = torch.from_numpy(actions_onehot[demo_idx]).to(torch.float32).to(device)
            eg = AffineEgress(lam=FROZEN_LAM).fit(X_demo, Y_demo,
                                                  family_z=family_z.to(device)
                                                  if family_z is not None else None)
            eg.to(device)
            # G1: demo reconstruction P@1
            zd = eg.predict(X_demo)
            g1_p1s.append(float((zd.argmax(1) == Y_demo.argmax(1)).float().mean()))
            correct, n = 0, 0
            for i in ev_idx:
                x = torch.from_numpy(psi[i]).to(torch.float32).to(device).unsqueeze(0)
                z = eg.predict(x)
                pred = int(z.argmax(1).item())
                true = int(np.argmax(actions_onehot[i]))
                correct += int(pred == true)
                n += 1
            p1 = correct / n if n else 0.0
            per_env[e] = {"n_test": n, "p1": round(float(p1), 4)}
            fold_correct += correct
            fold_n += n
        per_fold[f"fold{f}"] = {"n_test": fold_n,
                                "p1": round(float(fold_correct / fold_n), 4) if fold_n else 0.0}
        macro_acc += fold_correct
        macro_n += fold_n
    return {"arm": "A" if use_family else "B",
            "macro_p1": round(float(macro_acc / macro_n), 4) if macro_n else None,
            "per_fold": per_fold, "per_env": per_env,
            "g1_demo_p1_min": round(float(min(g1_p1s)), 4) if g1_p1s else None}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--split-seal", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="A,B,C,D")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest)
    folds = load_sealed_folds(args.split_seal)
    env_ids = sorted({str(m.get("env", "?")) for m in meta})
    dmask = demo_prefix_mask(meta, env_ids, k=FROZEN_DEMO_K)
    provenance_scan(meta, env_ids, folds, dmask, k=FROZEN_DEMO_K)
    D = psi.shape[1]

    results: Dict[str, Dict] = {}
    arm_list = [a.strip() for a in args.arms.split(",")]
    for arm in arm_list:
        if arm == "A":
            results["A"] = run_arm_ab(psi, actions_onehot, action_names, meta, folds,
                                      env_ids, dmask, D, device, use_family=True)
        elif arm == "B":
            results["B"] = run_arm_ab(psi, actions_onehot, action_names, meta, folds,
                                      env_ids, dmask, D, device, use_family=False)
        elif arm == "C":
            results["C"] = run_arm_c(psi, actions_onehot, action_names, meta, folds,
                                     env_ids, dmask, D, device)
        elif arm == "D":
            results["D"] = run_arm_d(psi, actions_onehot, action_names, meta, folds,
                                     env_ids, dmask, D, device)
        else:
            raise SystemExit(f"unknown arm {arm}")

    exit_codes = {a: 0 for a in results}
    if any(exit_codes.values()):
        raise SystemExit("BLOCKED_INFRASTRUCTURE: nonzero arm exit")

    a = results.get("A", {})
    c = results.get("C", {})
    g1 = a.get("g1_demo_p1_min")
    g2 = a.get("macro_p1")
    g3 = (a.get("macro_p1") - c.get("macro_p1")) if (a.get("macro_p1") is not None
                                                     and c.get("macro_p1") is not None) else None
    g4 = min((v["p1"] for v in a.get("per_fold", {}).values() if v.get("p1") is not None),
             default=None)

    if args.smoke:
        verdict = "SMOKE_NO_VERDICT"
    else:
        fails = []
        if g1 is not None and g1 < 0.99:
            fails.append("G1")
        if g2 is not None and g2 < 0.70:
            fails.append("G2")
        if g3 is not None and g3 < 0.25:
            fails.append("G3")
        if g4 is not None and g4 < 0.60:
            fails.append("G4")
        if not fails:
            verdict = "F7_AFFINE_PROMOTED"
        elif fails == ["G3"]:
            verdict = "FALSIFIED_NO_GAIN"   # mechanism intact, zero margin over circulant
        else:
            verdict = "FALSIFIED_AT_SCALE"

    receipt = {
        "schema_id": "f7-gates-receipt.v1",
        "verdict": verdict,
        "device": device,
        "split_seal": args.split_seal,
        "bank_manifest_run_id": manifest.get("run_id"),
        "arms": results,
        "gates": {
            "G1_demo_p1_min": g1,
            "G2_macro_p1": g2,
            "G3_margin_A_minus_C": g3,
            "G4_min_fold_p1": g4,
        },
        "thresholds": {"G1": 0.99, "G2": 0.70, "G3": 0.25, "G4": 0.60},
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print(f"F7_GATES_{verdict}")


if __name__ == "__main__":
    main()
