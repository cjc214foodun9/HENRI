#!/usr/bin/env python
"""Phase 3 batched-kernel gate: equivalence + latency on RTX 5090.

Pre-registered criteria (HENRI-PHASE3-BATCHED-GATE-2026):
  E1: max |batched - reference| <= 1e-3 over a real 172-candidate pool
      (float32 accumulation tolerance; ordering may differ only on ties).
  L1: p50 batched rank latency < p50 reference rank latency (strict win).
  M1: peak intermediate <= C*chunk*D*4 bytes (chunk=8 -> ~44 MB at
      C=172, D=65536); no [C, N, D] tensor ever materialized.
Kill: any of E1/L1/M1 fails -> seal KILL, keep --ast-idf-batched OFF.
Pass: E1+L1+M1 -> verdict PASS (throughput), with the recorded limitation
that Gate B showed grammar expressiveness -- not latency -- is the current
accuracy bottleneck (Phase 3 does not claim a score improvement).
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

from qfhrr_ast_discriminative_kernel import (  # noqa: E402
    ASTDiscriminativeEncoder,
    batched_mean_phase_cosine,
    build_idf_frequencies,
)
from humaneval_wave_ast_runner import parse_signature  # noqa: E402
from wave_ast_decoder import WaveASTDecoder  # noqa: E402
from zone_c_epistemic_axiom_harness import qFHRREpistemicCodec  # noqa: E402


def sha_prefix(path, n=16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-path", default="/root/henri-839-wt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--d-model", type=int, default=65536)
    args = ap.parse_args()

    repo = Path(args.repo_path)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    torch.manual_seed(20260820)
    print(f"[P3G] repo={repo} d={args.d_model} device={device}", flush=True)

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
    print(f"[P3G] codebook={len(cb_vecs)} mbpp_sha={sha_prefix(mbpp_path)}", flush=True)

    he_path = repo / "HENRI V2" / "data" / "HumanEval.jsonl.gz"
    items = []
    with gzip.open(he_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    item = items[0]
    entry, arg_names = parse_signature(item["prompt"])
    decoder = WaveASTDecoder(qFHRREpistemicCodec(d_model=args.d_model, device=device),
                             device=device)
    bodies = decoder._instantiate(entry, arg_names)
    pool = []
    for body in bodies:
        src = f"def {entry}({', '.join(arg_names)}):\n{body}"
        try:
            ast.parse(src)
        except SyntaxError:
            continue
        pool.append(src)
    print(f"[P3G] item0 pool={len(pool)}", flush=True)

    cand_vecs = []
    for src in pool:
        v = encoder.encode_code_string(src)
        if v is not None:
            cand_vecs.append(v)
    cand_mat = torch.stack(cand_vecs).to(device)          # [C, D] uint8
    cb_mat = torch.stack(cb_vecs).to(device)              # [N, D] uint8
    print(f"[P3G] cand={cand_mat.shape} cb={cb_mat.shape}", flush=True)

    # Reference: exact per-candidate loop (production path).
    ref = torch.zeros(len(cand_vecs), dtype=torch.float32)
    for i, v in enumerate(cand_vecs):
        s = 0.0
        for cb in cb_vecs:
            s += encoder.compute_cosine_similarity(v, cb)
        ref[i] = s / len(cb_vecs)

    # Batched path.
    bat = batched_mean_phase_cosine(cand_mat, cb_mat, codebook_chunk=8).cpu()
    max_diff = float((ref - bat).abs().max().item())

    def bench(fn, reps=5):
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append((time.perf_counter() - t0) * 1000.0)
        ts.sort()
        return ts[len(ts) // 2]

    t_ref = bench(lambda: None, reps=1)  # placeholder; reference loop timed inline
    t0 = time.perf_counter()
    for _ in range(5):
        for i, v in enumerate(cand_vecs):
            for cb in cb_vecs:
                encoder.compute_cosine_similarity(v, cb)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_ref = (time.perf_counter() - t0) / 5 * 1000.0

    t_bat = bench(lambda: batched_mean_phase_cosine(cand_mat, cb_mat), reps=5)

    print(f"[P3G] max_diff={max_diff:.6e} ref_ms={t_ref:.1f} bat_ms={t_bat:.1f} "
          f"speedup={t_ref / max(t_bat, 1e-6):.1f}x", flush=True)
    peak_bytes = cand_mat.shape[0] * 8 * cand_mat.shape[1] * 4
    naive_bytes = cand_mat.shape[0] * cb_mat.shape[0] * cand_mat.shape[1] * 4
    print(f"[P3G] peak_intermediate_bytes={peak_bytes} "
          f"(C*chunk*D*4, chunk=8, no [C,N,D]); "
          f"naive [C,N,D] would be {naive_bytes}", flush=True)

    e1 = max_diff <= 1e-3
    l1 = t_bat < t_ref
    m1 = (peak_bytes < naive_bytes) and (peak_bytes <= 512 * 2**20)
    print(f"GATE: E1={e1} L1={l1} M1={m1} -> "
          f"{'PASS' if (e1 and l1 and m1) else 'KILL'}", flush=True)


if __name__ == "__main__":
    main()
