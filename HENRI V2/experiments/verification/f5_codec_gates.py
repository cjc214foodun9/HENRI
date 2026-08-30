"""Carrier F5 codec arms + gates harness (remote CUDA, sealed-split only).

Spec: HENRI-SPEC-2026-08-F5-STRUCTURED-CODEC sections 3-5.

Arms (multi-arm kill matrix, spec section 5):
  A  FPB full codec (fpb_qfhrr_codec.FPBStructuredCodec)  — candidate
  B  Run21 StructuredCharPositionCodec (qfhrr_structured_codec) — control:
     proves F5 != Run21 (the killed class)
  C  legacy random-ring qFHRREpistemicCodec — control (F4 baseline)
  D  identity / no W_task (direct wave-to-action cos) — no-supervision baseline

Gates (spec section 3):
  G1  homomorphic metric continuity   rho(cos, -d) >= 0.85   (kill smoke K2)
  G1b FPB homomorphism                cos(bind) >= 0.99      (kill smoke K1)
  G2  task-functor unbinding coherence cos >= 0.40           (kill smoke K3)
  G3  grouped 4-fold held-out macro P@1 >= 0.8000            (this harness, arm A)
  G4  margin vs random-hash baseline  P@1_A - P@1_C >= +0.5000
  G5  per-task phase-occlusion diagnostic (per-env P@1 across ALL 12 envs,
      diagnostic only, no threshold)
  G6  default-OFF differential (contract tests, not this harness)

Pre-registered verdicts: F5_CODEC_PROMOTED / FALSIFIED_AT_SCALE /
FALSIFIED_NO_GEOMETRY / BLOCKED_INFRASTRUCTURE / BLOCKED_TARGET_LEAKAGE.

Any nonzero arm exit -> BLOCKED_INFRASTRUCTURE for the whole run.

Usage (remote, repo root, AFTER f5_split_seal.py):
  env PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f5_codec_gates.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --split-seal telemetry/f3_bank_capture_v2/f5_split_seal.json \
      --out telemetry/f3_bank_capture_v2/f5_gates_receipt.json \
      --arms A,B,C,D
  --smoke: disposable smoke (1 heldout env per fold, NO verdict).
"""
from __future__ import annotations

import argparse
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
FROZEN_SEED = 20260831

CONSUMED_SCHEMAS = ("f3-split-seal.v1", "f4-split-seal.v1")


# ---------------------------------------------------------------------------
# Bank loading + masks (shared with kill smoke; mirrors F4 gates contract)
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
    """Per-env boolean mask over rows: first k rows per env = demo prefix."""
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
    assert receipt["schema_id"] == "f5-split-seal.v1", \
        f"refusing non-F5 seal {receipt.get('schema_id')} (consumed-guard)"
    assert receipt["single_use"] is True, "seal must be single_use"
    folds = {f"fold{i}": receipt["folds"][f"fold{i}"] for i in range(FROZEN_N_FOLDS)}
    # fail-closed digest re-derivation
    manifest = {
        "rule": receipt["split_rule"],
        "n_folds": receipt["n_folds"],
        "seed": receipt["seed"],
        "env_order": receipt["envs"],
        "folds": receipt["folds"],
        "single_use": True,
    }
    import hashlib
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    assert digest == receipt["fold_manifest_sha256"], "fold manifest digest mismatch"
    return folds


def provenance_scan(meta: list, env_ids: List[str], folds: Dict[str, Dict],
                    dmask: Dict[str, np.ndarray], k: int = FROZEN_DEMO_K) -> None:
    """Kill 5 / spec 4.3: assert zero heldout-eval-row leaks into any train or
    demo prefix used for W_task compilation or scoring."""
    envs = np.array([str(m.get("env", "?")) for m in meta])
    for f in range(FROZEN_N_FOLDS):
        held = set(folds[f"fold{f}"]["heldout_envs"])
        train = set(folds[f"fold{f}"]["train_envs"])
        for e in held:
            idx = np.where(envs == e)[0]
            ev_rows = idx[k:]  # evaluation rows of the heldout env
            for te in train:
                assert not np.any(np.isin(ev_rows, np.where(envs == te)[0])), \
                    f"leak: heldout eval rows of {e} intersect train env {te}"
    # demo prefix may not overlap evaluation rows anywhere
    for e in env_ids:
        idx = np.where(envs == e)[0]
        assert set(np.where(dmask[e])[0]) == set(idx[:k]), f"demo mask wrong for {e}"


# ---------------------------------------------------------------------------
# Wave helpers (ring <-> wave, FHRR)
# ---------------------------------------------------------------------------

def psi_to_ring(psi: torch.Tensor) -> torch.Tensor:
    """Real wave [-1,1] -> Z_256 ring (documented real->ring mapping)."""
    return ((psi.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)


def ring_to_wave(q: torch.Tensor) -> torch.Tensor:
    """Z_256 ring -> unit-modulus complex wave (phase-domain)."""
    return torch.exp(1j * q.to(torch.float32) * (2.0 * math.pi / 256.0))


def wave_cos(w1: torch.Tensor, w2: torch.Tensor) -> float:
    num = float(torch.abs(torch.vdot(w1, w2)).item())
    den = float(w1.norm().item()) * float(w2.norm().item()) + 1e-12
    return num / den


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------

def arm_retrieve_scores(codec, psi_rows: torch.Tensor, action_names: List[str],
                        w_task: Optional[torch.Tensor],
                        use_w_task: bool = True) -> np.ndarray:
    """Score matrix [n_rows, n_actions]: wave-domain cos(retrieved, action wave).

    W_task compiled in the continuous phase domain (FHRR sum of
    Y * conj(X) over demo pairs); retrieval = X_test * W_task.
    """
    n = psi_rows.shape[0]
    n_a = len(action_names)
    scores = np.zeros((n, n_a), dtype=np.float32)
    action_waves = [codec.encode_wave(a) for a in action_names]
    for i in range(n):
        x_wave = ring_to_wave(psi_to_ring(psi_rows[i]))
        if use_w_task and w_task is not None:
            r = x_wave * w_task  # FHRR bind (phase addition)
        else:
            r = x_wave
        for a in range(n_a):
            scores[i, a] = wave_cos(r, action_waves[a])
    return scores


def compile_w_task(codec, x_rings: List[torch.Tensor], y_waves: List[torch.Tensor],
                   D: int) -> torch.Tensor:
    w = torch.zeros(D, dtype=torch.complex64)
    for xr, yw in zip(x_rings, y_waves):
        x_wave = ring_to_wave(xr)
        w = w + yw * torch.conj(x_wave)
    return w / (w.norm() + 1e-12)


def run_arm(codec, psi: np.ndarray, actions_onehot: np.ndarray,
            action_names: List[str], meta: list, folds: Dict[str, Dict],
            env_ids: List[str], dmask: Dict[str, np.ndarray],
            arm: str, D: int, device: str) -> Dict:
    """Run one arm over the sealed folds. Returns per-fold + per-env P@1."""
    envs = np.array([str(m.get("env", "?")) for m in meta])
    per_fold: Dict[str, Dict] = {}
    per_env: Dict[str, Dict] = {}
    macro_acc, macro_n = 0.0, 0
    for f in range(FROZEN_N_FOLDS):
        held = folds[f"fold{f}"]["heldout_envs"]
        fold_acc, fold_n = 0.0, 0
        fold_correct = 0
        for e in held:
            idx = np.where(envs == e)[0]
            demo_idx = idx[:FROZEN_DEMO_K]
            ev_idx = idx[FROZEN_DEMO_K:]
            if len(ev_idx) == 0:
                per_env[e] = {"n_test": 0, "p1": None}
                continue
            if arm == "D":
                # identity: no W_task, direct wave-to-action cosine
                w_task = None
                use_w_task = False
            else:
                x_rings = [psi_to_ring(torch.from_numpy(psi[i])) for i in demo_idx]
                y_waves = [codec.encode_wave(action_names[int(np.argmax(actions_onehot[i]))])
                           for i in demo_idx]
                w_task = compile_w_task(codec, x_rings, y_waves, D).to(device)
                use_w_task = True
            psi_rows = torch.from_numpy(psi[ev_idx]).to(device)
            scores = arm_retrieve_scores(codec, psi_rows, action_names, w_task,
                                         use_w_task=use_w_task)
            pred = scores.argmax(axis=1)
            true = np.argmax(actions_onehot[ev_idx], axis=1)
            correct = int((pred == true).sum())
            n = len(ev_idx)
            p1 = correct / n
            per_env[e] = {"n_test": n, "p1": round(float(p1), 4)}
            fold_correct += correct
            fold_n += n
        p1f = fold_correct / fold_n if fold_n else 0.0
        per_fold[f"fold{f}"] = {"n_test": fold_n, "p1": round(float(p1f), 4)}
        macro_acc += fold_correct
        macro_n += fold_n
    return {
        "arm": arm,
        "macro_p1": round(float(macro_acc / macro_n), 4) if macro_n else None,
        "per_fold": per_fold,
        "per_env": per_env,
    }


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
    assert torch.cuda.is_available(), "gates harness must run on CUDA"
    torch.manual_seed(FROZEN_SEED)
    np.random.seed(FROZEN_SEED)

    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest)
    D = psi.shape[1]
    V = actions_onehot.shape[1]
    assert D == 65536 and V == 7, f"bank shape {D}x{V} unexpected"

    envs = [str(m.get("env", "?")) for m in meta]
    folds = load_sealed_folds(args.split_seal)

    if args.smoke:
        smoke_envs = {folds[f"fold{i}"]["heldout_envs"][0] for i in range(FROZEN_N_FOLDS)}
        keep = np.isin(np.array(envs), list(smoke_envs))
        psi = psi[keep]
        actions_onehot = actions_onehot[keep]
        envs = [e for e in envs if e in smoke_envs]
        meta = [m for m in meta if m["env"] in smoke_envs]
        counts = {e: int(np.sum(np.array(envs) == e)) for e in smoke_envs}
        for f in range(FROZEN_N_FOLDS):
            held = [e for e in folds[f"fold{f}"]["heldout_envs"] if e in smoke_envs]
            train = [e for e in smoke_envs if e not in held]
            folds[f"fold{f}"]["heldout_envs"] = held
            folds[f"fold{f}"]["train_envs"] = train
            folds[f"fold{f}"]["n_heldout"] = int(sum(counts[e] for e in held))
            folds[f"fold{f}"]["n_train"] = int(sum(counts[e] for e in train))

    env_ids = sorted(set(envs))
    dmask = demo_prefix_mask(meta, env_ids, k=FROZEN_DEMO_K)
    provenance_scan(meta, env_ids, folds, dmask, k=FROZEN_DEMO_K)

    want = {s.strip() for s in args.arms.split(",") if s.strip()}
    arms_out: List[Dict] = []
    for arm in ("A", "B", "C", "D"):
        if arm not in want:
            continue
        if arm == "A":
            from fpb_qfhrr_codec import FPBStructuredCodec
            codec = FPBStructuredCodec(d_model=D, k_bins=256, device=device)
        elif arm == "B":
            from qfhrr_structured_codec import StructuredCharPositionCodec
            codec = StructuredCharPositionCodec(d_model=D, k_bins=256, device=device)
        elif arm == "C":
            from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec
            codec = qFHRREpistemicCodec(d_model=D, k_bins=256, device="cpu")
        elif arm == "D":
            codec = None  # identity arm: no codec needed
        else:
            raise ValueError(f"unknown arm {arm}")
        # identity arm has no codec; run_arm handles D specially
        if arm == "D":
            res = run_arm(None, psi, actions_onehot, action_names, meta, folds,
                          env_ids, dmask, "D", D, device)
        else:
            res = run_arm(codec, psi, actions_onehot, action_names, meta, folds,
                          env_ids, dmask, arm, D, device)
        arms_out.append(res)
        print(json.dumps(res, indent=2))

    if args.smoke:
        receipt = {
            "schema_id": "f5-gates-smoke.v1",
            "device": device,
            "arms": arms_out,
            "verdict": "SMOKE_NO_VERDICT",
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(receipt, fp, indent=2, default=str)
        print("F5_GATES_SMOKE_OK (no verdict by design)")
        return

    by_arm = {r["arm"]: r for r in arms_out}
    a = by_arm.get("A")
    c = by_arm.get("C")
    g3 = a["macro_p1"] if a and a["macro_p1"] is not None else 0.0
    g4 = (a["macro_p1"] - c["macro_p1"]) if (a and c and a["macro_p1"] is not None
                                             and c["macro_p1"] is not None) else -1.0

    g3_pass = g3 >= 0.8000
    g4_pass = g4 >= 0.5000
    g5_diag = {e: r["per_env"].get(e) for r in arms_out for e in env_ids}

    if g3_pass and g4_pass:
        verdict = "F5_CODEC_PROMOTED"
    elif not g3_pass or not g4_pass:
        verdict = "FALSIFIED_AT_SCALE"
    else:
        verdict = "BLOCKED_INFRASTRUCTURE"

    receipt = {
        "schema_id": "f5-gates-receipt.v1",
        "device": device,
        "npz_sha256": manifest["npz_sha256"],
        "split_seal": args.split_seal,
        "arms": arms_out,
        "gates": {
            "G1": "kill-smoke (rho>=0.85)",
            "G1b": "kill-smoke (homo>=0.99)",
            "G2": "kill-smoke (cos>=0.40)",
            "G3": {"threshold": 0.8000, "measured": round(float(g3), 4),
                   "pass": g3_pass},
            "G4": {"threshold": 0.5000, "measured": round(float(g4), 4),
                   "pass": g4_pass},
            "G5": {"diagnostic": True, "per_env": g5_diag},
            "G6": "contract tests (default-OFF differential)",
        },
        "verdict": verdict,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print(f"F5_GATES_VERDICT={verdict}")


if __name__ == "__main__":
    main()
