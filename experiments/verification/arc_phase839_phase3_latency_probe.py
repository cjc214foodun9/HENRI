#!/usr/bin/env python
"""Phase 3 profiler-first probe: operator-level latency breakdown on the
HumanEval score-bearing path at D=65,536 (RTX 5090).

Pre-registered verdict (HENRI-PHASE3-LATENCY-PROBE-2026):
- If the wave encode+rank phase is < 10% of total item latency AND
  the sandbox-execute phase is >= 80% of total item latency, then the
  roadmap's Phase 3 targets (ASTqFHRREncoder / SagnacMCTSPlanner kernel
  fusion) are OFF the score-path bottleneck -> seal BLOCKED_OFF_PATH
  (latency target <= 2.0 ms is a TARGET_GOAL on a non-bottleneck path).
- If the wave phase is >= 10% of item latency, record the breakdown and
  propose a bounded Triton port of the measured hot operator instead.

Reuses the exact production chain from humaneval_wave_ast_runner.py:
WaveASTDecoder._instantiate (candidate pool), ASTDiscriminativeEncoder
(IDF codebook N=100 from MBPP sha ccf64cea...), SecurePythonSandbox.
"""
import argparse
import ast
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "HENRI V2"))

from mbpp_secure_executor import SecurePythonSandbox  # noqa: E402
from qfhrr_ast_discriminative_kernel import (  # noqa: E402
    ASTDiscriminativeEncoder,
    build_idf_frequencies,
)
from wave_ast_decoder import WaveASTDecoder  # noqa: E402


def sha_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", default="/root/henri-839-wt")
    ap.add_argument("--items", type=int, default=3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--attempts", type=int, default=8)
    args = ap.parse_args()

    repo = Path(args.repo_path)
    print(f"[P3] repo={repo} d={args.d_model} device={args.device} items={args.items}",
          flush=True)
    print(f"[P3] torch={torch.__version__} cuda={torch.cuda.is_available()}",
          flush=True)
    torch.manual_seed(20260820)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(f"[P3] gpu={torch.cuda.get_device_name(0)} "
              f"mem={torch.cuda.get_device_properties(0).total_memory / 2**30:.1f}GiB",
              flush=True)

    # --- IDF encoder + codebook (production chain) ---
    mbpp_path = repo / "HENRI V2" / "data" / "mbpp.jsonl"
    mbpp_codes = []
    with open(mbpp_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                mbpp_codes.append(json.loads(line).get("code", ""))
    freqs, corpus_size = build_idf_frequencies(mbpp_codes)
    encoder = ASTDiscriminativeEncoder(d_model=args.d_model, device=device,
                                       idf_weighting=True,
                                       node_frequencies=freqs,
                                       corpus_size=corpus_size)
    cb_vecs = [encoder.encode_code_string(c) for c in mbpp_codes[:100]]
    cb_vecs = [v for v in cb_vecs if v is not None]
    print(f"[P3] codebook={len(cb_vecs)} mbpp_sha={sha_prefix(mbpp_path)}",
          flush=True)

    # --- HumanEval items ---
    he_path = repo / "HENRI V2" / "data" / "HumanEval.jsonl.gz"
    items = []
    with gzip.open(he_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"[P3] humaneval_total={len(items)} sha={sha_prefix(he_path)}", flush=True)

    decoder = WaveASTDecoder(d_model=args.d_model, device=device)

    SHIM = ("import math\nimport re\nimport sys\nimport json\n"
            "from typing import *\n\n")

    rows = []
    for idx, item in enumerate(items[: args.items]):
        prompt, entry, tests = item["prompt"], item["entry_point"], item["test"]
        args_list = re.findall(r"def \w+\((.*?)\):", prompt)
        arg_names = [a.strip() for a in args_list[0].split(",")] if args_list else []
        t0 = time.perf_counter()
        bodies = decoder._instantiate({"prompt": prompt, "entry_point": entry}, None)
        t_gen = (time.perf_counter() - t0) * 1000.0
        pool = []
        for body in bodies:
            src = f"def {entry}({', '.join(arg_names)}):\n{body}"
            try:
                ast.parse(src)
            except SyntaxError:
                continue
            pool.append((src, body))
        # encode + rank phase (IDF)
        t0 = time.perf_counter()
        sims = []
        for src, body in pool:
            v = encoder.encode_code_string(src)
            if v is None:
                sims.append(-1e9)
                continue
            s = 0.0
            for cb in cb_vecs:
                s += encoder.compute_cosine_similarity(v, cb)
            sims.append(s / max(1, len(cb_vecs)))
        ordered = [p for _, p in sorted(zip(sims, pool), key=lambda t: t[0], reverse=True)]
        t_rank = (time.perf_counter() - t0) * 1000.0
        # sandbox phase (first attempt only, bounded)
        sandbox = SecurePythonSandbox(timeout_sec=8.0, mode="container-rlimit")
        body = ordered[0][1] if ordered else "return None"
        full = SHIM + prompt.rstrip() + "\n" + body + "\n" + tests + f"\ncheck({entry})"
        t0 = time.perf_counter()
        res = sandbox.execute(full)
        t_sand = (time.perf_counter() - t0) * 1000.0
        total = t_gen + t_rank + t_sand
        row = {"item": idx, "task_id": item.get("task_id", entry),
               "candidates": len(pool), "gen_ms": round(t_gen, 1),
               "rank_ms": round(t_rank, 1), "sandbox_ms": round(t_sand, 1),
               "total_ms": round(total, 1),
               "rank_frac": round(t_rank / total, 3), "sandbox_frac": round(t_sand / total, 3),
               "sandbox_status": res.status}
        rows.append(row)
        print(json.dumps(row), flush=True)

    tot = {k: sum(r[k] for r in rows) for k in ("gen_ms", "rank_ms", "sandbox_ms", "total_ms")}
    summary = {"items": len(rows),
               "wave_phase_ms": round(tot["gen_ms"] + tot["rank_ms"], 1),
               "wave_phase_frac": round((tot["gen_ms"] + tot["rank_ms"]) / tot["total_ms"], 3),
               "sandbox_phase_ms": round(tot["sandbox_ms"], 1),
               "sandbox_phase_frac": round(tot["sandbox_ms"] / tot["total_ms"], 3),
               "avg_item_ms": round(tot["total_ms"] / len(rows), 1)}
    print("SUMMARY " + json.dumps(summary), flush=True)
    print("GATE: " + ("BLOCKED_OFF_PATH" if summary["wave_phase_frac"] < 0.10
                      and summary["sandbox_phase_frac"] >= 0.80 else "MEASURE_BOUNDED_PORT"),
          flush=True)


if __name__ == "__main__":
    main()
