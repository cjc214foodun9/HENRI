"""K1/K2 corpus runner: build fresh corpus from T0+K0, audit, fit.

Usage (Vast CUDA target):
    HENRI_TEMPORAL_LEDGER=1 HENRI_LEDGER_PAYLOADS=1 \\
    HENRI_KOOPMAN_IDENTIFIABILITY=1 HENRI_KOOPMAN_FIT=1 \\
    /venv/main/bin/python koopman_corpus_runner.py \\
        --ledger /workspace/.../temporal_ledger.jsonl \\
        --payload-root /workspace/.../payloads \\
        --out /workspace/.../koopman.json

Steps:
  1. load_corpus from T0 rows + K0 payload sidecars (dedupe continuity)
  2. split BY EPISODE (calibration/evaluation, deterministic seed)
  3. audit identifiability (rank gates, per-action support, overlap)
  4. IF PASS: evaluate K2 arms (persistence, agnostic, shuffled, conditioned)
  5. emit one JSON artifact (verdict + telemetry)
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from koopman_identifiability import (  # noqa: E402
    audit,
    load_corpus,
    split_episodes,
)
from koopman_fit import evaluate  # noqa: E402
from ledger_payload_store import LedgerPayloadStore  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--payload-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--eval-frac", type=float, default=0.3)
    ap.add_argument("--candidate-ranks", default="2,4,8,16")
    ap.add_argument("--ridge", type=float, default=1e-4)
    ap.add_argument("--num-blocks", type=int, default=8)
    args = ap.parse_args()

    if os.environ.get("HENRI_KOOPMAN_IDENTIFIABILITY", "0") != "1" or \
            os.environ.get("HENRI_KOOPMAN_FIT", "0") != "1":
        print(json.dumps({"verdict": "BLOCKED_FLAG_OFF",
                          "reason": "HENRI_KOOPMAN_IDENTIFIABILITY=1 and "
                                    "HENRI_KOOPMAN_FIT=1 required"}))
        return 2

    store = LedgerPayloadStore(args.payload_root,
                               flag="HENRI_LEDGER_PAYLOADS")
    # Production lift: HENRIVisionEncoder encode_spatial_grid -> [blocks, 8]
    import torch
    from arc_spatial_basis import resolve_spatial_basis
    from henri_vision_encoder import HENRIVisionEncoder
    basis_kind, bg_mask = resolve_spatial_basis()
    tokenizer = HENRIVisionEncoder(
        d_model=65536, k_blocks=8192, device="cpu",
        spatial_basis_kind=basis_kind, bg_mask=bg_mask)
    # NOTE: production replay uses the live device; the corpus runner pins
    # CPU + the production basis resolution so the audit is reproducible.
    def lift(grid):
        w = tokenizer.encode_spatial_grid(grid).squeeze(0)
        return w.detach().cpu().to(torch.float32)

    # Action-wave map: on the production surface the orchestrator owns the
    # decoder engram waves. Runner accepts a JSON map {action_name: path}
    # or builds placeholder rings (CONTROL-ONLY — never claims production
    # action geometry without the orchestrator waves).
    action_wave_map = {}
    if os.path.exists("action_waves.json"):
        import numpy as np
        amap = json.load(open("action_waves.json"))
        for k, path in amap.items():
            w = np.load(path)
            action_wave_map[k] = torch.tensor(w, dtype=torch.float32)
    else:
        for i in range(16):
            rng = torch.Generator().manual_seed(1000 + i)
            w = torch.randn(args.num_blocks, 8, generator=rng)
            action_wave_map[f"a{i}"] = w / w.norm(dim=-1, keepdim=True)

    records, stats = load_corpus(args.ledger, store, lift,
                                 action_wave_map)
    if not records:
        print(json.dumps({"verdict": "BLOCKED_EMPTY_CORPUS",
                          "stats": stats}))
        return 2
    cal, evl, cal_ids, eval_ids = split_episodes(
        records, seed=args.seed, eval_frac=args.eval_frac)
    ranks = [int(x) for x in args.candidate_ranks.split(",") if x]
    audit_out = audit(cal, evl, ranks)
    if audit_out["verdict"].startswith("IDENTIFIABILITY_BLOCKED"):
        result = {"verdict": "IDENTIFIABILITY_BLOCKED",
                  "audit": audit_out, "stats": stats,
                  "cal_episodes": cal_ids, "eval_episodes": eval_ids,
                  "n_records": len(records), "n_cal": len(cal),
                  "n_eval": len(evl)}
        Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True),
                                  encoding="utf-8")
        print(json.dumps({"verdict": "IDENTIFIABILITY_BLOCKED",
                          "reason": audit_out.get("blocked_reasons")}))
        return 1
    r = audit_out["recommended_rank"]
    fit_out = evaluate(cal, evl, _dictionary_fn, ridge=args.ridge,
                       rank=r, num_blocks=args.num_blocks)
    result = {"verdict": fit_out["verdict"], "audit": audit_out,
              "fit": fit_out, "stats": stats,
              "cal_episodes": cal_ids, "eval_episodes": eval_ids,
              "n_records": len(records), "n_cal": len(cal),
              "n_eval": len(evl)}
    Path(args.out).write_text(json.dumps(result, indent=2, sort_keys=True),
                              encoding="utf-8")
    print(json.dumps({"verdict": fit_out["verdict"],
                      "rank": r,
                      "one_step": fit_out["one_step"],
                      "rollouts": fit_out["rollouts"]}))
    return 0


def _dictionary_fn(state_wave, action_wave):
    """Fused-intent dictionary: real||imag of the circular-convolution
    binding (production LowRankCoupledTransition.bind geometry), on CPU."""
    import torch
    s = torch.complex(state_wave[..., :4], state_wave[..., 4:])
    a = torch.complex(action_wave[..., :4], action_wave[..., 4:])
    bound = torch.fft.ifft(torch.fft.fft(s, dim=-1) * torch.fft.fft(a, dim=-1),
                           dim=-1)
    return torch.cat([bound.real.reshape(-1), bound.imag.reshape(-1)])


if __name__ == "__main__":
    sys.exit(main())
