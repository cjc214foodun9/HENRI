#!/usr/bin/env python3
"""Phase 2 CUDA gate for the live batched AST scorer.

This gate compares the exact production symbol with HENRI_SAGNAC_CUDA off/on.
It is not a benchmark score and does not exercise ARC task selection.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
ACTIVE = ROOT / "HENRI V2"
if str(ACTIVE) not in sys.path:
    sys.path.insert(0, str(ACTIVE))

from qfhrr_ast_discriminative_kernel import batched_mean_phase_cosine  # noqa: E402


def timed(fn, *, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(reps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", type=int, default=172)
    ap.add_argument("--codebook", type=int, default=100)
    ap.add_argument("--dimension", type=int, default=65536)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("BLOCKED_CUDA_NOT_EXECUTED")
        return 2
    device = torch.device("cuda")
    torch.manual_seed(20260821)
    candidates = torch.randint(
        0, 256, (args.candidates, args.dimension), dtype=torch.uint8, device=device
    ).contiguous()
    codebook = torch.randint(
        0, 256, (args.codebook, args.dimension), dtype=torch.uint8, device=device
    ).contiguous()

    os.environ["HENRI_SAGNAC_CUDA"] = "0"
    reference = batched_mean_phase_cosine(candidates, codebook)
    torch.cuda.synchronize()
    ref_times = timed(
        lambda: batched_mean_phase_cosine(candidates, codebook),
        warmup=args.warmup,
        reps=args.reps,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    os.environ["HENRI_SAGNAC_CUDA"] = "1"
    try:
        fused = batched_mean_phase_cosine(candidates, codebook)
        torch.cuda.synchronize()
        fused_times = timed(
            lambda: batched_mean_phase_cosine(candidates, codebook),
            warmup=args.warmup,
            reps=args.reps,
        )
    except Exception as exc:  # noqa: BLE001 - receipt must classify the boundary
        receipt = {
            "schema": "henri.phase2-cuda-gate.v1",
            "status": "BLOCKED_INFRASTRUCTURE",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "candidate_shape": list(candidates.shape),
            "codebook_shape": list(codebook.shape),
        }
        print(json.dumps(receipt, sort_keys=True))
        return 2

    # Contract: the op must launch on the caller's current stream. Queue a
    # dependent reduction on a non-default stream before synchronization. A
    # default-stream launch is not ordered with this consumer on all builds.
    non_default_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(non_default_stream):
        streamed = batched_mean_phase_cosine(candidates, codebook)
        streamed_dependency = streamed.sum()
    non_default_stream.synchronize()
    stream_max_abs_error = float((reference - streamed).abs().max().item())
    stream_dependency = float(streamed_dependency.item())

    max_abs_error = float((reference - fused).abs().max().item())
    ref_p50 = statistics.median(ref_times)
    fused_p50 = statistics.median(fused_times)
    peak_bytes = int(torch.cuda.max_memory_allocated(device))
    result = {
        "schema": "henri.phase2-cuda-gate.v1",
        "status": "PASS" if max_abs_error <= 1e-3 and stream_max_abs_error <= 1e-3 and fused_p50 < ref_p50 and peak_bytes < 512 * 2**20 else "FALSIFIED",
        "implementation": "torch.ops.henri.sagnac_mcts",
        "production_symbol": "qfhrr_ast_discriminative_kernel.batched_mean_phase_cosine",
        "flag": "HENRI_SAGNAC_CUDA=1",
        "device": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "candidate_shape": list(candidates.shape),
        "codebook_shape": list(codebook.shape),
        "max_abs_error": max_abs_error,
        "non_default_stream_max_abs_error": stream_max_abs_error,
        "non_default_stream_dependency": stream_dependency,
        "reference_p50_ms": ref_p50,
        "fused_p50_ms": fused_p50,
        "speedup": ref_p50 / max(fused_p50, 1e-9),
        "peak_memory_allocated_bytes": peak_bytes,
        "dense_intermediate_bytes_rejected": args.candidates * args.codebook * args.dimension * 4,
        "synchronized_timing": True,
        "task_score_claim": False,
    }
    print(json.dumps(result, sort_keys=True))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
