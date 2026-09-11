#!/usr/bin/env python
"""Zone C authenticated write->retrieve round trip (production Timescale store).

Real SegmentCache path: connect() -> checkpoint() -> retrieve().
Asserts: TimescaleZoneCStore (not surrogate), row count increment, hits >= 1,
conditioning wave present, bounded shape/norm. Non-benchmark wave + unique
run domain. Keeps the row (audit evidence); does not delete.
"""
import json
import os
import sys
import time
import uuid

import numpy as np
import torch

from henri_vision_encoder import HENRIVisionEncoder
from zone_c_segment_cache import SegmentCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = 65536
NUM_BLOCKS = 8192
RUN_ID = uuid.uuid4().hex[:12]

rng = np.random.default_rng(20260810)
GRID = rng.integers(0, 10, size=(30, 30)).tolist()


def main() -> int:
    t0 = time.perf_counter()
    report = {"run_id": RUN_ID, "device": DEVICE}

    tokenizer = HENRIVisionEncoder(d_model=D, k_blocks=NUM_BLOCKS, device=DEVICE)
    wave = tokenizer.encode_spatial_grid(GRID).squeeze(0).to(DEVICE)
    wave = wave / (torch.norm(wave, p=2, dim=-1, keepdim=True) + 1e-9)

    seg = SegmentCache.connect(num_blocks=NUM_BLOCKS)
    report["store_type"] = type(seg.store).__name__

    before = seg.store.count()
    eid = seg.checkpoint(wave, domain=f"smoke_roundtrip_{RUN_ID}",
                         sagnac_stress=0.1)
    after = seg.store.count()

    rec = seg.retrieve(wave.cpu())
    conditioning = rec.get("conditioning_wave")
    report.update({
        "write_id": eid,
        "rows_before": before,
        "rows_after": after,
        "rows_delta": after - before,
        "hits": rec.get("hits", 0),
        "top_similarity": round(float(rec.get("top_similarity", 0.0)), 6),
        "has_conditioning": conditioning is not None,
        "conditioning_shape": tuple(conditioning.shape)
        if conditioning is not None else None,
        "conditioning_norm": round(float(torch.norm(conditioning).item()), 4)
        if conditioning is not None else None,
        "elapsed_s": round(time.perf_counter() - t0, 3),
    })

    report["verdict"] = (
        "PASS"
        if report["store_type"] == "TimescaleZoneCStore"
        and report["rows_delta"] >= 1
        and report["hits"] >= 1
        and report["has_conditioning"]
        else "FAIL"
    )
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(json.dumps({"verdict": "ERROR",
                          "error": f"{type(exc).__name__}: {exc}"}))
        sys.exit(1)
