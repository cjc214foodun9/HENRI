"""Phase 10.0 full-D local probe: beta=0 byte-identity + gradient reachability
against the REAL production decoder checkpoint (D=65536, CPU-forced).

Report Step 2 pre-flight items:
  1. gradient reachability on all trainable deep parameters;
  2. zero dense [65536,65536] allocation;
  3. beta=0 byte-identity against the linear decoder checkpoint.

Emits a JSON receipt. Run from the repo root with the isolated interpreter:
    env HENRI_DEEP_EGRESS=1 -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME \\
      PYTHONPATH="HENRI V2" /c/Python314/python.exe \\
      "HENRI V2/experiments/verification/probe_deep_egress_identity.py"
"""
import hashlib
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

import torch

from henri_deep_egress import DeepEgressProposalHead
from henri_decoder import HENRIUnifiedEgressTransducer

D, NB, BD, PD, DH, V = 65536, 8192, 8, 2, 2048, 32000
CKPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "models", "henri_decoder_checkpoint.pt")


def main() -> None:
    t0 = time.time()
    ckpt_sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()

    # 1. Load the REAL linear decoder (checkpoint_policy=required).
    transducer = HENRIUnifiedEgressTransducer(
        d_model=D, hidden_dim=DH, vocab_size=V,
        device="cpu", checkpoint_path=CKPT, checkpoint_policy="required",
    )
    assert transducer.checkpoint_load_status == "LOADED", transducer.checkpoint_load_status
    unbinder = transducer.unbinder

    # Deterministic unit wave on S^(D-1).
    g = torch.Generator().manual_seed(20260826)
    wave = torch.randn(1, D, generator=g)
    unit_wave = wave / (wave.norm(dim=-1, keepdim=True) + 1e-8)

    # 2. Linear baseline logits (real checkpoint forward).
    with torch.no_grad():
        linear_logits = unbinder(unit_wave)

    # 3. Deep head sharing the SAME vocab head (checkpoint lm_head).
    head = DeepEgressProposalHead(
        D, NB, BD, PD, DH, V, beta=0.0, lm_head=unbinder.lm_head)

    # 3a. beta=0 byte-identity against the real checkpoint baseline.
    with torch.no_grad():
        out0 = head(unit_wave, linear_logits)
    byte_identical = torch.equal(out0, linear_logits)
    max_abs_diff_0 = (out0 - linear_logits).abs().max().item()

    # 3b. beta=0.5: deep path executes, blends, gradients reach deep params.
    out_half = head(unit_wave, linear_logits, beta=0.5)
    blend_ok = torch.allclose(
        out_half, 0.5 * linear_logits + 0.5 * head.deep_logits(unit_wave), atol=1e-6)
    out_half.sum().backward()
    grad_report = {}
    for name, p in {
        "block_proj.weight": head.block_proj.weight,
        "deep_down.weight": head.deep_down.weight,
        "layer_norm.weight": head.layer_norm.weight,
        "layer_norm.bias": head.layer_norm.bias,
    }.items():
        grad_report[name] = {
            "grad_present": p.grad is not None,
            "abs_sum": float(p.grad.abs().sum().item()) if p.grad is not None else 0.0,
        }
    all_grads = all(v["grad_present"] and v["abs_sum"] > 0.0 for v in grad_report.values())

    # 4. Dense-allocation audit + activation-budget accounting.
    dense_found = []
    for name, p in head.named_parameters():
        if len(p.shape) == 2 and p.shape[0] == D and p.shape[1] == D:
            dense_found.append(name)
    peak_act_bytes = NB * PD * 4 + NB * PD * 4 + DH * 4 + V * 4

    receipt = {
        "probe": "phase10.0_deep_egress_identity",
        "device": "cpu",
        "checkpoint_sha256": ckpt_sha,
        "checkpoint_load_status": transducer.checkpoint_load_status,
        "checkpoint_state_dict_sha256": transducer.checkpoint_state_dict_sha256,
        "d_model": D, "num_blocks": NB, "block_dim": BD, "proj_dim": PD,
        "d_hidden": DH, "vocab_size": V,
        "beta0_byte_identical": bool(byte_identical),
        "beta0_max_abs_diff": float(max_abs_diff_0),
        "beta_half_blend_ok": bool(blend_ok),
        "gradient_reachability": grad_report,
        "all_deep_params_grad_nonzero": bool(all_grads),
        "dense_d_d_params": dense_found,
        "peak_activation_bytes_est": peak_act_bytes,
        "activation_budget_ok": peak_act_bytes < int(1.5e9),
        "params_millions": {
            "deep_total": round(sum(p.numel() for p in head.parameters()) / 1e6, 3),
            "block_proj": round(head.block_proj.weight.numel() / 1e6, 3),
            "deep_down": round(head.deep_down.weight.numel() / 1e6, 3),
        },
        "elapsed_s": round(time.time() - t0, 2),
    }
    ok = (byte_identical and max_abs_diff_0 == 0.0 and blend_ok and all_grads
          and not dense_found and peak_act_bytes < int(1.5e9))
    receipt["verdict"] = "PASS" if ok else "FAIL"
    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "phase10.0_deep_egress_probe_receipt.json")
    with open(out_path, "w") as f:
        json.dump(receipt, f, indent=2, sort_keys=True)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"RECEIPT: {out_path}")
    print(f"VERDICT: {receipt['verdict']}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
