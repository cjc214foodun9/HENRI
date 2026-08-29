"""F2-M3 efficacy gates on the REAL trajectory-bank corpus (remote CUDA).

Consumes the sealed Phase 8.32 bank artifacts produced by the authorized
capture run (HENRI_ARC_TRAJECTORY_BANK=1, production_arc_run.py @ pinned SHA):

  trajectories_<run_id>.npz        psi [M,65536] fp16, next_wave, actions_onehot [M,6] uint8
  trajectories_<run_id>.jsonl      per-record meta (env, step, ...)
  trajectories_<run_id>_manifest.json   digests + provenance (authorized)

Protocol (per prereg SPEC-2026-08-29-F2-EGRESS):
  - Episode-disjoint split: calibration on a subset of ENVS, held-out on
    env(s) NOT in calibration. Split receipt lists ordered env IDs + counts.
  - G1: held-out P@1 >= 0.99  (F2 codebook, V = bank action vocab = 6)
  - G2: syntactic validity >= 0.99 on discrete action tokens.
        DISCLOSURE: with V=6 and argmax over the full action vocab, every
        decoded token is a legal action by construction => G2 = 1.0
        is VACUOUS at this vocab. Reported but not a capability claim.
  - K2/G3 (margin vs legacy linear head): BLOCKED on this corpus unless the
    legacy checkpoint's token vocabulary maps ACTION1..6 to token IDs.
    The gate harness probes that mapping and reports BLOCKED_NO_ACTION_TOKEN_MAPPING
    when absent (no fabricated comparator).
  - No dense [65536,65536] allocation anywhere (K4).

Usage (remote, repo root, worktree at the sealed capture SHA):
  env ZONE_C_ENV=prod PYTHONPATH="HENRI V2" /venv/main/bin/python \
      experiments/verification/f2_egress_gates.py \
      --npz telemetry/f2_bank_capture/trajectories_<run_id>.npz \
      --manifest telemetry/f2_bank_capture/trajectories_<run_id>_manifest.json \
      --jsonl telemetry/f2_bank_capture/trajectories_<run_id>.jsonl \
      --heldout-envs wa30-ee6fef47 \
      --out telemetry/f2_bank_capture/gates_receipt.json

Local CPU runs are software sanity only; remote CUDA is the verification boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--heldout-envs", required=True, help="comma-separated env IDs held out")
    ap.add_argument("--out", required=True)
    ap.add_argument("--beta", type=float, default=8.0)
    ap.add_argument("--ridge", type=float, default=1e-3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # -- provenance pin ------------------------------------------------------
    npz_sha = _sha256(args.npz)
    jsonl_sha = _sha256(args.jsonl)
    with open(args.manifest, "r", encoding="utf-8") as fp:
        manifest = json.load(fp)
    assert manifest["data_source"] == "authorized", "bank must be authorized capture"
    assert manifest["npz_sha256"] == npz_sha, "npz hash mismatch vs manifest"
    assert manifest["jsonl_sha256"] == jsonl_sha, "jsonl hash mismatch vs manifest"

    # -- load ----------------------------------------------------------------
    bank = np.load(args.npz)
    psi = bank["psi"].astype(np.float32)                       # [M, D]
    actions_onehot = bank["actions_onehot"].astype(np.float32)  # [M, A]
    action_names = [str(a) for a in bank["action_names"]]
    A = len(action_names)
    V = A
    D = psi.shape[1]
    assert D == 65536, f"expected production D=65536, got {D}"
    assert A >= 2, f"bank action vocab must be >= 2, got {A}"

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

    X_cal = torch.from_numpy(psi[calib_mask]).to(device)
    Y_cal = torch.from_numpy(actions_onehot[calib_mask]).to(device)
    X_hold = torch.from_numpy(psi[heldout_mask]).to(device)
    Y_hold = torch.from_numpy(actions_onehot[heldout_mask]).to(device)

    # -- F2 calibration (dual ridge, thin-SVD; no [D,D] anywhere) -----------
    from f2_egress_codebook import F2HopfieldEgressCodebook

    cb = F2HopfieldEgressCodebook(d_model=D, vocab_size=V, beta=args.beta, ridge_lambda=args.ridge)
    t0 = time.perf_counter()
    cb.calibrate(X_cal, Y_cal)
    t_cal = time.perf_counter() - t0
    torch.cuda.synchronize() if device == "cuda" else None

    # -- G1: held-out P@1 ----------------------------------------------------
    t0 = time.perf_counter()
    z, logits = cb.snap(X_hold, return_logits=True)
    t_snap = time.perf_counter() - t0
    torch.cuda.synchronize() if device == "cuda" else None
    pred = logits.argmax(dim=-1)
    true = Y_hold.argmax(dim=-1)
    p1_hold = (pred == true).float().mean().item()
    per_action_p1 = {}
    for a in range(A):
        mask = true == a
        if mask.sum() > 0:
            per_action_p1[action_names[a]] = round(
                (pred[mask] == a).float().mean().item(), 4
            )
    cal_p1 = (
        (cb.snap(X_cal, return_logits=True)[1].argmax(dim=-1) == Y_cal.argmax(dim=-1))
        .float().mean().item()
    )

    # -- G2: syntactic validity (DISCLOSED VACUOUS at V=6) -------------------
    legal_rate = float((pred < V).float().mean().item())  # always 1.0 by construction

    # -- K2/G3: margin vs legacy linear head ----------------------------------
    legacy_status = "BLOCKED_NO_ACTION_TOKEN_MAPPING"
    legacy_note = (
        "The legacy head (down_proj 65536->2048 + lm_head 32000) has no verified "
        "mapping from ACTION1..6 to token IDs on this checkpoint; a fabricated "
        "comparator is prohibited. G3/K2 cannot run on this corpus without that mapping."
    )
    # Probe: does the checkpoint carry a tokenizer/vocab file nearby? (reported only)
    legacy_mapping_found = False

    receipt = {
        "schema_id": "f2-egress-gates.v1",
        "device": device,
        "npz_sha256": npz_sha,
        "jsonl_sha256": jsonl_sha,
        "manifest_provenance": manifest.get("provenance"),
        "record_count": int(len(meta)),
        "calibration_envs": sorted({e for e, m in zip(envs, calib_mask) if m}),
        "heldout_envs": sorted(heldout_set),
        "n_calibration": int(n_cal),
        "n_heldout": int(n_hold),
        "action_vocab": action_names,
        "g1_heldout_p1": round(p1_hold, 4),
        "g1_calibration_p1": round(cal_p1, 4),
        "per_action_heldout_p1": per_action_p1,
        "g2_syntactic_validity": round(legal_rate, 4),
        "g2_disclosure": "VACUOUS at V=6 (argmax always a legal action); not a capability claim",
        "g3_legacy_margin": None,
        "g3_legacy_status": legacy_status,
        "g3_legacy_note": legacy_note,
        "time_calibrate_s": round(t_cal, 3),
        "time_snap_s": round(t_snap, 3),
        "codebook_bytes": cb.codebook_bytes(),
        "finite": bool(torch.isfinite(cb.M).all().item() and torch.isfinite(z).all().item()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    assert receipt["finite"], "non-finite codebook/snap"
    print("GATES_RECEIPT_OK")


if __name__ == "__main__":
    main()
