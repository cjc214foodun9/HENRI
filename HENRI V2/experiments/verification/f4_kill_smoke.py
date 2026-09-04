"""F4 kill experiments 1-3 — CUDA production-scale, bounded, disposable.

Spec: HENRI-SPEC-2026-08-F4-NONLINEAR-EGRESS section 6 (kill experiments).
Runs BEFORE the fresh split seal and BEFORE the sealed arms. Reads the real
bank READ-ONLY; never loads a split seal; never trains on evaluation rows.

K1 (production-shape sanity): train the head on a bounded real-bank subset;
   loss descends and argmax stays in [0, 7). The XOR nonlinearity-sanity
   contract itself lives in the CPU contract suite (kill 1, CPU edition).
K2 (Tier-1 engagement): per-env W_task compile from the demo prefix + unbind
   of real rows -> cos(raw, unbound) < 0.99 and unbound on S^{D-1}.
K3 (Tier-3 engagement): 3 SGLD steps on 20 demo rows -> ||dW3|| > 1e-6,
   CE descends, W1/W2 byte-unchanged.

Exit code 0 + F4_KILL_SMOKE_OK only when all kills pass; any failure raises
AssertionError (nonzero exit, fail-loud).

Usage (remote, repo root):
  env PYTHONPATH="HENRI V2" HENRI_F4_EGRESS=1 /venv/main/bin/python \
      "HENRI V2/experiments/verification/f4_kill_smoke.py" \
      --npz telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.npz \
      --jsonl telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2.jsonl \
      --manifest telemetry/f3_bank_capture_v2/trajectories_production_run_f3v2_manifest.json \
      --out telemetry/f3_bank_capture_v2/f4_kill_smoke_receipt.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_verif_dir = str(Path(__file__).resolve().parent)
if _verif_dir not in sys.path:
    sys.path.insert(0, _verif_dir)
_henri_root = str(Path(__file__).resolve().parents[2])
if _henri_root not in sys.path:
    sys.path.insert(0, _henri_root)

from f4_egress_gates import load_bank, demo_prefix_mask  # noqa: E402
from f4_nonlinear_egress_head import (  # noqa: E402
    F4NonLinearEgressHead,
    compile_env_w_task,
    unbind_w_task,
)
from zone_c_epistemic_axiom_harness import (  # noqa: E402
    HolographicTaskFunctorCompiler,
    qFHRREpistemicCodec,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert torch.cuda.is_available(), "kill smoke must run on CUDA"
    torch.manual_seed(20260830)
    np.random.seed(20260830)

    psi, actions_onehot, action_names, meta, manifest = load_bank(
        args.npz, args.jsonl, args.manifest)
    D = psi.shape[1]
    V = actions_onehot.shape[1]
    assert D == 65536 and V == 7, f"bank shape {D}x{V} unexpected"

    envs = [str(m.get("env", "?")) for m in meta]
    env_ids = sorted(set(envs))
    dmask = demo_prefix_mask(meta, env_ids, k=20)
    env0 = env_ids[0]
    idx0 = np.where(np.array(envs) == env0)[0]

    # ---- K1: production-shape head trains on a bounded subset ------------
    head = F4NonLinearEgressHead(d_model=D, hidden1=2048, hidden2=512,
                                 n_actions=V, seed=20260830).to(device)
    x1 = torch.from_numpy(psi[:200]).to(device)
    y1 = torch.from_numpy(actions_onehot[:200]).to(device)
    tel1 = head.train_head(x1, y1, lr=1e-3, wd=1e-4, batch=64, epochs=3, seed=20260830)
    assert tel1["loss_first"] is not None and tel1["loss_last"] < tel1["loss_first"], \
        "K1 dead: loss did not descend"
    with torch.no_grad():
        logits = head(x1)
    assert torch.isfinite(logits).all(), "K1 dead: non-finite logits"
    assert int(logits.argmax(dim=-1).min()) >= 0 and int(logits.argmax(dim=-1).max()) < V

    # ---- K2: Tier-1 engagement on real rows ------------------------------
    codec = qFHRREpistemicCodec(d_model=D, k_bins=256, device="cpu")
    compiler = HolographicTaskFunctorCompiler(codec)
    demo_idx = idx0[:20]
    demo_psi = torch.from_numpy(psi[demo_idx])
    demo_act = [action_names[int(np.argmax(actions_onehot[i]))] for i in demo_idx]
    w_task = compile_env_w_task(codec, compiler, demo_psi, demo_act)
    assert w_task.dtype == torch.uint8 and w_task.shape == (D,)
    ev_idx = idx0[20:40]
    cos_sum = 0.0
    for i in ev_idx:
        raw = torch.from_numpy(psi[i])
        unb = unbind_w_task(raw, w_task, codec, D=D)
        assert abs(float(unb.norm().item()) - 1.0) < 1e-3, "K2 unbound not on S^{D-1}"
        cos_sum += float(F.cosine_similarity(raw.to(torch.float32).unsqueeze(0),
                                             unb.unsqueeze(0)).item())
    mean_cos = cos_sum / len(ev_idx)
    assert mean_cos < 0.99, f"K2 inert: unbind cos {mean_cos:.4f} >= 0.99"

    # ---- K3: Tier-3 engagement (W3 only) ---------------------------------
    demo_unbound = torch.stack([
        unbind_w_task(torch.from_numpy(psi[i]), w_task, codec, D=D) for i in demo_idx
    ])
    demo_labels = torch.from_numpy(actions_onehot[demo_idx])
    w1_before = head.W1.detach().clone()
    w2_before = head.W2.detach().clone()
    tel3 = head.adapt_w3_sgld(demo_unbound.to(device), demo_labels.to(device),
                              steps=3, eta=1e-3, t0=1e-6, dt=1.0, seed=20260830)
    assert tel3["delta_w3_fro"] > 1e-6, "K3 dead: W3 did not move"
    assert tel3["loss_last"] < tel3["loss_first"], "K3 dead: CE did not descend"
    assert torch.equal(head.W1, w1_before) and torch.equal(head.W2, w2_before), \
        "K3 violated: SGLD touched W1/W2"
    assert torch.isfinite(head.W3).all(), "K3 dead: non-finite W3"

    receipt = {
        "schema_id": "f4-kill-smoke.v1",
        "device": device,
        "npz_sha256": manifest["npz_sha256"],
        "env0": env0,
        "k1": {"rows": 200, "epochs": 3,
               "loss_first": round(tel1["loss_first"], 6),
               "loss_last": round(tel1["loss_last"], 6)},
        "k2": {"n_demo": 20, "n_eval": len(ev_idx),
               "mean_cos_raw_unbound": round(float(mean_cos), 6)},
        "k3": {"steps": 3, "delta_w3_fro": round(tel3["delta_w3_fro"], 8),
               "loss_first": round(tel3["loss_first"], 6),
               "loss_last": round(tel3["loss_last"], 6)},
        "verdict": "F4_KILL_SMOKE_OK",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(receipt, fp, indent=2, default=str)
    print(json.dumps(receipt, indent=2))
    print("F4_KILL_SMOKE_OK")


if __name__ == "__main__":
    main()
