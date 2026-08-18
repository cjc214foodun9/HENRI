"""K2 probe: Sagnac veto false-negative / false-positive rates.

Two layers, both pre-registered (design doc a2886ec):
  L1 REPL layer (CPU-safe): 200 known-valid + 200 known-invalid commands
      through DualChannelREPLVeto.evaluate_execution. ACCEPT: 0 FN, <=1% FP.
  L2 wave layer (CUDA, D=65536): 200 axiom-consistent (valid) + 200
      orthogonal (invalid) candidate waves through
      SagnacMCTSPlanner.dual_channel_sagnac_veto. ACCEPT: 0 FN, <=1% FP.

Scheduling: L2 must run ALONE on the GPU (no concurrent production runs).

Usage: python k2_veto_probe.py [--wave-only]
Output: JSON telemetry to stdout + file.
"""

import argparse
import json

import numpy as np
import torch


def _valid_commands() -> list[str]:
    return [f"print({i})" for i in range(200)]


def _invalid_commands() -> list[str]:
    return [f"raise ValueError({i})" for i in range(200)]


def l1_repl_layer() -> dict:
    from henri_universal_repl import DualChannelREPLVeto

    veto = DualChannelREPLVeto(tau_veto=0.35)
    fp = fn = 0
    for cmd in _valid_commands():
        vetoed, _ = veto.evaluate_execution(cmd, 0, "0", "", 0.1)
        fp += int(vetoed)  # false positive: valid command vetoed
    for cmd in _invalid_commands():
        # failing code -> returncode != 0 -> hard veto
        res = veto.evaluate_execution(cmd, 1, "", "Error", 0.1)
        vetoed, _ = res
        fn += int(not vetoed)  # false negative: invalid command not vetoed
    n_valid, n_invalid = 200, 200
    return {
        "layer": "L1_repl",
        "valid": n_valid,
        "invalid": n_invalid,
        "false_positives": fp,
        "false_negatives": fn,
        "fp_rate": round(fp / n_valid, 4),
        "fn_rate": round(fn / n_invalid, 4),
        "accept": fn == 0 and fp <= 2,
    }


def l2_wave_layer(d: int = 65536, blocks: int = 8192) -> dict:
    from sagnac_mcts_planner import SagnacMCTSPlanner

    assert torch.cuda.is_available(), "L2 requires CUDA"
    dev = "cuda"
    plan = SagnacMCTSPlanner(d_model=d, k_blocks=blocks, tau_veto=0.35, device=dev)

    rng = np.random.default_rng(7)
    valid, invalid = [], []
    for i in range(200):
        base = torch.randn(blocks, 8, device=dev)
        base = torch.nn.functional.normalize(base, p=2, dim=-1)
        valid.append(base)  # axiom-consistent: identical wave
        noise = torch.randn(blocks, 8, device=dev)
        noise = torch.nn.functional.normalize(noise, p=2, dim=-1)
        # orthogonal candidate: ~1/sqrt(D) inner product -> delta ~ 1
        ortho = (base - (base * noise).sum(dim=-1, keepdim=True) * noise)
        ortho = torch.nn.functional.normalize(ortho, p=2, dim=-1)
        invalid.append(ortho)

    fp = fn = 0
    for wv in valid:
        _, _, hard = plan.dual_channel_sagnac_veto(wv, wv, wv)  # axiom == cand == world
        fp += int(hard)
    for wv in invalid:
        _, _, hard = plan.dual_channel_sagnac_veto(wv, valid[0], valid[0])
        fn += int(not hard)
    return {
        "layer": "L2_wave",
        "valid": 200,
        "invalid": 200,
        "false_positives": fp,
        "false_negatives": fn,
        "fp_rate": round(fp / 200, 4),
        "fn_rate": round(fn / 200, 4),
        "accept": fn == 0 and fp <= 2,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--l1-only", action="store_true")
    ap.add_argument("--wave-only", action="store_true")
    ap.add_argument("--out", type=str, default="/tmp/k2_veto.json")
    args = ap.parse_args()

    results = []
    if not args.wave_only:
        results.append(l1_repl_layer())
    if not args.l1_only:
        results.append(l2_wave_layer())

    accept = all(r["accept"] for r in results)
    telemetry = {"probe": "K2", "layers": results, "accept": bool(accept)}
    print(json.dumps(telemetry, indent=2))
    with open(args.out, "w") as f:
        json.dump(telemetry, f, indent=2)
    return 0 if accept else 1


if __name__ == "__main__":
    raise SystemExit(main())
