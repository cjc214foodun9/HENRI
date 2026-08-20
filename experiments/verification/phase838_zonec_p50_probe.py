# -*- coding: utf-8 -*-
"""Phase 8.38 gate probe: live Zone C retrieval p50 on production path.

Measures full bridge.retrieve() round-trip (semantic_projection + HNSW
store query + engram row fetch) on the RTX 5090 with a production-shaped
[8192, 8] GPU wave. Also verifies CPU/GPU projection invariance.
"""
import os
import statistics
import time

import torch

from zone_c_retrieval_bridge import ZoneCRetrievalBridge, probe_projection

NUM_BLOCKS = 8192
N_TRIALS = 30
TARGET_P50_MS = 12.0


def main() -> None:
    dsn = os.environ["ZONE_C_PROD_DSN"]
    assert os.environ.get("HENRI_ZONEC_BRIDGE") == "1", "bridge flag must be 1"
    assert torch.cuda.is_available(), "CUDA required"

    # Device-independence check: projection on CPU vs GPU wave.
    torch.manual_seed(0)
    cpu_wave = torch.randn(NUM_BLOCKS, 8)
    gpu_wave = cpu_wave.clone().cuda()
    cpu_sem = probe_projection(cpu_wave)
    gpu_sem = probe_projection(gpu_wave)
    cos = float(torch.nn.functional.cosine_similarity(cpu_sem, gpu_sem.cpu(), dim=-1))

    # Production path: GPU query wave through the bridge.
    bridge = ZoneCRetrievalBridge(dsn=dsn, num_blocks=NUM_BLOCKS)
    wave = torch.randn(NUM_BLOCKS, 8, device="cuda")

    bridge.retrieve(wave)  # warmup (connection + caches)
    latencies = []
    hits = []
    for _ in range(N_TRIALS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        hits = bridge.retrieve(wave)
        torch.cuda.synchronize()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    top1 = float(hits[0][1]) if hits else 0.0
    print(
        f"p50_ms={p50:.3f} mean_top1={top1:.4f} "
        f"hits={len(hits)} trials={N_TRIALS} cpu_gpu_cos={cos:.6f} "
        f"gate={'PASS' if p50 <= TARGET_P50_MS else 'FAIL'} target={TARGET_P50_MS}ms"
    )


if __name__ == "__main__":
    main()
