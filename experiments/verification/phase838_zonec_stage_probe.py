# -*- coding: utf-8 -*-
"""Phase 8.38 stage-timing probe: locate the residual retrieval latency.

Breaks bridge.retrieve() into: projection, q-list formatting, connection,
SQL execute, bytea transfer, decode. Prints per-stage p50 over N trials.
"""
import os
import statistics
import time

import torch

from zone_c_segment_cache import semantic_projection, bytes_to_wave, TimescaleZoneCStore

NUM_BLOCKS = 8192
N_TRIALS = 20


def main() -> None:
    dsn = os.environ["ZONE_C_PROD_DSN"]
    wave = torch.randn(NUM_BLOCKS, 8, device="cuda")
    store = TimescaleZoneCStore(dsn=dsn, num_blocks=NUM_BLOCKS)

    stages = {k: [] for k in
              ["projection", "qlist", "connect", "execute", "transfer", "decode"]}

    # warmup: projection matrix materialization + connection
    q = semantic_projection(wave)
    _ = q.tolist()
    with store._connect() as conn:
        conn.cursor().execute("SELECT 1")

    for _ in range(N_TRIALS):
        t0 = time.perf_counter()
        q = semantic_projection(wave)
        torch.cuda.synchronize()
        stages["projection"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        q_list = "[" + ",".join(f"{v:.6f}" for v in q.tolist()) + "]"
        stages["qlist"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        conn = store._connect()
        stages["connect"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT engram_wave_bytes,
                       1 - (semantic_index <=> %s::vector) AS similarity,
                       EXTRACT(EPOCH FROM (now() - timestamp)) / 3600.0 AS age_hours
                FROM phylogenetic_engrams_65536
                WHERE timestamp > now() - (%s || ' hours')::interval
                ORDER BY semantic_index <=> %s::vector
                LIMIT %s
                """,
                (q_list, 8760.0, q_list, 15),
            )
            rows = cur.fetchall()
        stages["execute"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        total_bytes = sum(len(r[0]) for r in rows)
        stages["transfer"].append((time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        kept = [r for r in rows if len(r[0]) == NUM_BLOCKS * 8 * 4][:5]
        decoded = [bytes_to_wave(r[0], NUM_BLOCKS) for r in kept]
        stages["decode"].append((time.perf_counter() - t0) * 1000)
        conn.close()

    def p50(v):
        v = sorted(v)
        return v[len(v) // 2]

    print("stage_p50_ms=" + " ".join(
        f"{k}={p50(v):.3f}" for k, v in stages.items()))
    print(f"rows_fetched={len(rows)} bytes_fetched={total_bytes} "
          f"kept={len(kept)} top1={float(decoded and 1.0)}")


if __name__ == "__main__":
    main()
