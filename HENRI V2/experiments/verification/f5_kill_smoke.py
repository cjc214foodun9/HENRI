"""Carrier F5 kill experiments 1-3 — CUDA production-scale, bounded, disposable.

Spec: HENRI-SPEC-2026-08-F5-STRUCTURED-CODEC section 6.
Runs BEFORE the fresh split seal and BEFORE the sealed arms. Reads the real
bank READ-ONLY; never loads a split seal; never trains on evaluation rows.

K1 (FPB homomorphism): bind(Psi(2), Psi(3)) == Psi(5), cos >= 0.99 (G1b).
K2 (metric continuity): Spearman rho(sim(orbit), -|dx|) >= 0.85 (G1).
K3 (G2 unbinding coherence on real bank rows): per-env W_task from the demo
   prefix (psi rows + FPB-encoded action names), wave-domain retrieval on
   heldout rows, mean cos(retrieved, true action wave) >= 0.40.
K4 (default-OFF differential): local contract test (test_codec_not_auto_imported
   + test_fpb_kernels_additive_only) — not re-run here.
K5 (leakage scan): provenance_scan runs inside the gates harness pre-arms.

Exit code 0 + F5_KILL_SMOKE_OK only when all kills pass; any failure raises
AssertionError (nonzero exit, fail-loud).

Usage (remote, repo root):
  env PYTHONPATH="HENRI V2" /venv/main/bin/python \
      "HENRI V2/experiments/verification/f5_kill_smoke.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --out telemetry/f3_bank_capture_v2/f5_kill_smoke_receipt.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_verif_dir = str(Path(__file__).resolve().parent)
if _verif_dir not in sys.path:
    sys.path.insert(0, _verif_dir)
_henri_root = str(Path(__file__).resolve().parents[2])
if _henri_root not in sys.path:
    sys.path.insert(0, _henri_root)

from f5_codec_gates import load_bank  # noqa: E402
from fpb_qfhrr_codec import FPBStructuredCodec  # noqa: E402
from qfhrr_kernels import fhrr_bind, fpb_power_wave, make_fpb_base_ring  # noqa: E402


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def _wave_cos(w1: torch.Tensor, w2: torch.Tensor) -> float:
    num = float(torch.abs(torch.vdot(w1, w2)).item())
    den = float(w1.norm().item()) * float(w2.norm().item()) + 1e-12
    return num / den


def psi_to_ring(psi: torch.Tensor) -> torch.Tensor:
    """Real wave [-1,1] -> Z_256 ring (documented real->ring mapping)."""
    return ((psi.clamp(-1.0, 1.0) + 1.0) / 2.0 * 255.0).round().to(torch.uint8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert torch.cuda.is_available(), "kill smoke must run on CUDA"
    torch.manual_seed(20260831)
    np.random.seed(20260831)

    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest)
    D = psi.shape[1]
    V = actions_onehot.shape[1]
    assert D == 65536 and V == 7, f"bank shape {D}x{V} unexpected"

    # ---- K1: FPB homomorphism (production D, CUDA) ------------------------
    base = make_fpb_base_ring(d_model=D, k_bins=256, seed=20260831,
                              amplitude=0.6).to(device)
    w2 = fpb_power_wave(base, 2.0)
    w3 = fpb_power_wave(base, 3.0)
    w5 = fpb_power_wave(base, 5.0)
    homo = _wave_cos(fhrr_bind(w2, w3), w5)
    assert homo >= 0.99, f"K1 dead: FPB homomorphism cos={homo:.6f}"

    # ---- K2: metric continuity (orbit) -------------------------------------
    xs = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    sims, dists = [], []
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            if i < j:
                sims.append(_wave_cos(fpb_power_wave(base, x),
                                      fpb_power_wave(base, y)))
                dists.append(abs(x - y))
    rho = _spearman(np.array(sims), -np.array(dists))
    assert rho >= 0.85, f"K2 dead: metric continuity rho={rho:.4f}"

    # ---- K3: G2 unbinding coherence on real bank rows ----------------------
    codec = FPBStructuredCodec(d_model=D, k_bins=256, device="cpu")
    envs = [str(m.get("env", "?")) for m in meta]
    env_ids = sorted(set(envs))

    # per-env W_task from demo prefix (psi rows + FPB-encoded action names)
    cos_sum, n_rows = 0.0, 0
    per_env = {}
    for e in env_ids:
        idx = np.where(np.array(envs) == e)[0]
        demo_idx = idx[:20]
        ev_idx = idx[20:60]
        if len(ev_idx) == 0:
            continue
        x_rings = psi_to_ring(torch.from_numpy(psi[demo_idx].astype(np.float32)))
        y_waves = []
        for i in demo_idx:
            a = action_names[int(np.argmax(actions_onehot[i]))]
            y_waves.append(codec.encode_wave(a))
        # wave-domain W_task: W = sum Y * conj(X_wave)
        w_task = torch.zeros(D, dtype=torch.complex64, device="cpu")
        for j, xr in enumerate(x_rings):
            x_wave = torch.exp(1j * xr.to(torch.float32) * (2.0 * math.pi / 256.0))
            w_task = w_task + fhrr_bind(y_waves[j], torch.conj(x_wave))
        w_task = w_task / (w_task.norm() + 1e-12)
        # retrieve on heldout rows, score vs the TRUE action wave
        acc = 0.0
        for i in ev_idx:
            xr = psi_to_ring(torch.from_numpy(psi[i].astype(np.float32)))
            x_wave = torch.exp(1j * xr.to(torch.float32) * (2.0 * math.pi / 256.0))
            r = fhrr_bind(x_wave, w_task)
            a = action_names[int(np.argmax(actions_onehot[i]))]
            y_true = codec.encode_wave(a)
            acc += _wave_cos(r, y_true)
        mean_c = acc / len(ev_idx)
        per_env[e] = round(mean_c, 4)
        cos_sum += acc
        n_rows += len(ev_idx)
    g2_mean = cos_sum / n_rows
    assert g2_mean >= 0.40, f"K3 dead: G2 unbinding cos={g2_mean:.4f} < 0.40"

    receipt = {
        "schema_id": "f5-kill-smoke.v1",
        "device": device,
        "npz_sha256": manifest["npz_sha256"],
        "k1_fpb_homomorphism_cos": round(homo, 6),
        "k2_metric_continuity_rho": round(rho, 4),
        "k3_g2_unbinding_mean_cos": round(g2_mean, 4),
        "k3_per_env": per_env,
        "verdict": "F5_KILL_SMOKE_OK",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("F5_KILL_SMOKE_OK")


if __name__ == "__main__":
    main()
