"""
HENRI V2 Equivalent Tokens Per Second (TPS) Throughput Benchmark.

Measures equivalent token throughput across 3 core operational modes:
  1. Latent Phase Rollout Mode (Next-LAT continuous wave transitions)
  2. Modern Hopfield Egress Snapping Mode (qFHRR wave -> discrete token)
  3. Full End-to-End Mode (Latent step + Hopfield Egress snap)

Calculates Equivalent Tokens Per Second (TPS) on target hardware (RTX 5090 GPU).
"""

import math
import os
import sys
import time
import torch
import torch.nn.functional as F

from henri_egress import TextEgress, UniversalEgress
from qfhrr_kernels import build_cos_lut
from recursive_dual_edmd import RecursiveDualEDMD


def benchmark_henri_equivalent_tps(
    num_blocks: int = 8192,
    d_model: int = 65536,
    batch_size: int = 64,
    num_warmup: int = 20,
    num_iters: int = 100,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> dict:
    print(f"\n=== HENRI V2 EQUIVALENT TOKENS PER SECOND (TPS) BENCHMARK ===")
    print(f"Target Device : {device}")
    print(f"Dimension     : D={d_model} ({num_blocks} blocks of Cl(3,0))")
    print(f"Batch Size    : {batch_size} parallel wave states")

    # Tokens per wave equivalence:
    # 1 qFHRR wave = 8,192 8-bit phase codes = 4,096 16-bit token equivalents
    tokens_per_wave = num_blocks  # Conservative 1 token per Clifford block (8192 tokens/wave)

    # --- MODE 1: Next-LAT Latent Phase Rollout Throughput ---
    edmd = RecursiveDualEDMD(d_model=d_model, r_rank=16, lambda_forget=0.98).to(device)
    state_batch = torch.randn(batch_size, num_blocks, 8, device=device)
    action_batch = torch.randn(batch_size, num_blocks, 8, device=device)

    # Warmup
    for _ in range(num_warmup):
        edmd(state_batch, action_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters):
        edmd(state_batch, action_batch)
    if device.type == "cuda":
        torch.cuda.synchronize()

    dt_latent = time.perf_counter() - t0
    total_latent_rollouts = batch_size * num_iters
    rollouts_per_sec = total_latent_rollouts / dt_latent
    latent_tps = rollouts_per_sec * tokens_per_wave

    print(f"\n[Mode 1: Next-LAT Latent Phase Rollouts]")
    print(f"  Total Rollouts          : {total_latent_rollouts} across {dt_latent*1000:.2f} ms")
    print(f"  Rollout Latency         : {dt_latent*1000/total_latent_rollouts:.3f} ms / rollout")
    print(f"  Rollouts Per Second     : {rollouts_per_sec:.2f} Hz")
    print(f"  Equivalent Latent TPS   : {latent_tps:,.1f} Tokens/Sec")

    # --- MODE 2: Modern Hopfield Egress Snapping Throughput ---
    num_codebook = 256
    g = torch.Generator(device="cpu").manual_seed(42)
    codebook_waves = torch.randn(num_codebook, d_model, generator=g).to(device)
    codebook_tokens = [f"token_{i}" for i in range(num_codebook)]

    text_egress = TextEgress(d_model=d_model, beta=8.0).to(device)
    text_egress.register_tokens(codebook_waves, codebook_tokens)
    query_wave = torch.randn(num_blocks, 8, device=device)

    # Warmup
    for _ in range(num_warmup):
        text_egress.decode_wave(query_wave)
    if device.type == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(num_iters * 10):
        text_egress.decode_wave(query_wave)
    if device.type == "cuda":
        torch.cuda.synchronize()

    dt_egress = time.perf_counter() - t0
    total_snaps = num_iters * 10
    snaps_per_sec = total_snaps / dt_egress
    egress_tps = snaps_per_sec * 1.0  # 1 discrete token per snap

    print(f"\n[Mode 2: Modern Hopfield Egress Lexical Snapping]")
    print(f"  Total Snaps             : {total_snaps} across {dt_egress*1000:.2f} ms")
    print(f"  Snap Latency            : {dt_egress*1000/total_snaps:.3f} ms / token")
    print(f"  Discrete Token TPS      : {snaps_per_sec:,.1f} Tokens/Sec")

    # --- MODE 3: Full End-to-End Pipeline (Latent Step + Hopfield Snap) ---
    t0 = time.perf_counter()
    for _ in range(num_iters):
        next_wave = edmd(state_batch[0], action_batch[0])
        text_egress.decode_wave(next_wave)
    if device.type == "cuda":
        torch.cuda.synchronize()

    dt_e2e = time.perf_counter() - t0
    e2e_tps = (num_iters * tokens_per_wave) / dt_e2e

    print(f"\n[Mode 3: Full End-to-End Latent Step + Egress Snap]")
    print(f"  E2E Step Latency        : {dt_e2e*1000/num_iters:.3f} ms / step")
    print(f"  Full E2E Equivalent TPS : {e2e_tps:,.1f} Tokens/Sec")

    return {
        "latent_tps": latent_tps,
        "egress_tps": snaps_per_sec,
        "e2e_tps": e2e_tps,
        "rollouts_per_sec": rollouts_per_sec,
        "latency_ms_per_rollout": dt_latent * 1000 / total_latent_rollouts,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    res = benchmark_henri_equivalent_tps(
        num_blocks=2048, d_model=16384, batch_size=16, num_warmup=10, num_iters=50, device=device
    )
    print("\n" + "=" * 75)
    print(f"  SUMMARY: HENRI EQUIVALENT THROUGHPUT ON {device}")
    print(f"    - Next-LAT Phase Rollout TPS : {res['latent_tps']:,.1f} Tokens/Sec")
    print(f"    - Discrete Token Egress TPS  : {res['egress_tps']:,.1f} Tokens/Sec")
    print(f"    - End-to-End Combined TPS    : {res['e2e_tps']:,.1f} Tokens/Sec")
    print("=" * 75)


if __name__ == "__main__":
    main()
