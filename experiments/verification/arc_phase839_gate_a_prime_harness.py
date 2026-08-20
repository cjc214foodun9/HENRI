"""Phase 8.39 Gate A' — test execution harness (CPU proxy, d=2048).

Spec: HENRI-SPEC-GATE-A-PRIME-IDF-2026 (arc_phase839_gate_a_prime_preregistration.md)
Arm: IDF-only (Lever 3.2-A enabled, Lever 3.1-A disabled) under --ast-idf-only.
Default-OFF: without --ast-idf-only the harness runs the Class 2.0 control arm
and applies NO gate verdict (exit 0, verdict=control).

Gate A' rules:
  M2: HumanEval/23 rank <= 5/71 AND HumanEval/35 rank <= 5/71  -> PASS
  Kill: either rank > 5/71 -> FALSIFIED (exit 1), Gate B skipped.
  M1: E[cos] recorded only, NOT a kill gate.

Substrate: CPU d=2048; production 71-candidate grammar pool via
WaveASTDecoder._instantiate; MBPP codebook N=100 (canonical mbpp.jsonl).

Usage (repo root, isolated interpreter):
  env -u VIRTUAL_ENV -u PYTHONPATH -u PYTHONHOME PYTHONPATH="HENRI V2" \
    /c/Python314/python.exe experiments/verification/arc_phase839_gate_a_prime_harness.py \
    --ast-idf-only [--d-model 2048] [--codebook-n 100] [--output-dir <dir>]

Exit: 0 = PASS (Gate B launchable), 1 = FALSIFIED (kill), 2 = harness error.
"""

import argparse
import ast
import gzip
import json
import math
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HENRI = os.path.join(REPO, "HENRI V2")
for p in (HENRI, REPO):
    if p not in sys.path:
        sys.path.insert(0, p)

from qfhrr_ast_discriminative_kernel import (  # noqa: E402
    ASTDiscriminativeEncoder,
    build_idf_frequencies,
)
from wave_ast_decoder import WaveASTDecoder  # noqa: E402

CORRECT_BODIES = {
    "HumanEval/23": "    return len(string)",
    "HumanEval/35": "    return max(l)",
}


def load_mbpp(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_humaneval(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_candidate_pool(decoder, entry, args):
    bodies = decoder._instantiate(entry, args)
    srcs = []
    for body in bodies:
        src = f"def {entry}({', '.join(args)}):\n{body}"
        try:
            ast.parse(src)
        except SyntaxError:
            continue
        srcs.append((src, body))
    return srcs


def mean_pairwise_cosine(vectors):
    if len(vectors) < 2:
        return 0.0
    stacked = torch.stack([v.to(torch.float32) for v in vectors])
    theta = 2.0 * math.pi / 256.0
    phase_diff = (stacked.unsqueeze(0) - stacked.unsqueeze(1)) * theta
    sims = torch.cos(phase_diff).mean(dim=-1)
    n = len(vectors)
    tri = torch.triu(torch.ones(n, n), diagonal=1).bool()
    return float(sims[tri].mean().item())


def rank_correct_body(encoder, pool, codebook_vecs, correct_body):
    sims = []
    for src, body in pool:
        v = encoder.encode_code_string(src)
        if v is None:
            sims.append((body, float("-inf")))
            continue
        s = 0.0
        for cb in codebook_vecs:
            s += encoder.compute_cosine_similarity(v, cb)
        sims.append((body, s / max(1, len(codebook_vecs))))
    sims.sort(key=lambda t: t[1], reverse=True)
    for i, (body, s) in enumerate(sims):
        if body == correct_body:
            return i + 1, s
    return len(sims) + 1, None


def git_head():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def sha_prefix(path, n=16):
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def main():
    ap = argparse.ArgumentParser(description="Gate A' IDF-only harness")
    ap.add_argument("--ast-idf-only", action="store_true",
                    help="Enable IDF-only treatment arm (default: control, no verdict)")
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--codebook-n", type=int, default=100)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    t0 = time.time()
    mbpp_path = os.path.join(HENRI, "data", "mbpp.jsonl")
    he_path = os.path.join(HENRI, "data", "HumanEval.jsonl.gz")
    mbpp_sha = sha_prefix(mbpp_path)

    mbpp = load_mbpp(mbpp_path)
    mbpp_codes = []
    for item in mbpp:
        code = item.get("code", item.get("solution", ""))
        if isinstance(code, str) and code.strip():
            mbpp_codes.append(code)
    print(f"[gate-a-prime] MBPP records={len(mbpp)} usable={len(mbpp_codes)} sha={mbpp_sha}")

    freqs, corpus_size = build_idf_frequencies(mbpp_codes)
    top = sorted(freqs.items(), key=lambda kv: -kv[1])[:6]
    print(f"[gate-a-prime] corpus_size={corpus_size} node_types={len(freqs)}")
    print("[gate-a-prime] top nodes:", ", ".join(f"{k}:{v}" for k, v in top))

    he = {it["task_id"]: it for it in load_humaneval(he_path)}
    decoder = WaveASTDecoder(codec=None, device="cpu")
    pools = {}
    import re

    for tid in ("HumanEval/23", "HumanEval/35"):
        prompt = he[tid]["prompt"]
        m = re.search(r"def\s+(\w+)\s*\(([^)]*)\)", prompt)
        entry, argstr = m.group(1), m.group(2)
        argnames = [a.split(":")[0].strip() for a in argstr.split(",") if a.strip()]
        pools[tid] = build_candidate_pool(decoder, entry, argnames)
        print(f"[gate-a-prime] {tid}: entry={entry} args={argnames} pool={len(pools[tid])}")

    idf_on = args.ast_idf_only
    enc = ASTDiscriminativeEncoder(
        d_model=args.d_model, device="cpu",
        idf_weighting=idf_on, carrier_subtract=False,
        node_frequencies=freqs if idf_on else None,
        corpus_size=corpus_size if idf_on else None,
    )
    codebook_codes = mbpp_codes[: args.codebook_n]
    cb_vecs = [enc.encode_code_string(c) for c in codebook_codes]
    cb_vecs = [v for v in cb_vecs if v is not None]
    print(f"[gate-a-prime] codebook kept={len(cb_vecs)}/{len(codebook_codes)}")

    d0 = enc.encode_code_string(pools["HumanEval/23"][0][0])
    d1 = enc.encode_code_string(pools["HumanEval/23"][0][0])
    deterministic = bool(
        d0 is not None and d1 is not None and torch.equal(d0, d1)
    )
    e_cos = mean_pairwise_cosine(
        [enc.encode_code_string(s) for s, _ in pools["HumanEval/23"]]
    )
    ranks = {}
    for tid in ("HumanEval/23", "HumanEval/35"):
        ranks[tid] = rank_correct_body(enc, pools[tid], cb_vecs, CORRECT_BODIES[tid])[0]
    print(f"[gate-a-prime] arm={'idf-only' if idf_on else 'control'} "
          f"E[cos]={e_cos:.4f} ranks /23={ranks['HumanEval/23']} "
          f"/35={ranks['HumanEval/35']} deterministic={deterministic}")

    if not idf_on:
        print("GATE_A_PRIME=CONTROL (no verdict; --ast-idf-only required for treatment)")
        sys.exit(0)

    passed = ranks["HumanEval/23"] <= 5 and ranks["HumanEval/35"] <= 5
    verdict = "PASS" if passed else "FALSIFIED"
    print(f"[gate-a-prime] M2 /23<=5: {ranks['HumanEval/23'] <= 5} | "
          f"/35<=5: {ranks['HumanEval/35'] <= 5} -> {verdict}")
    print(f"GATE_A_PRIME={verdict}")

    receipt = {
        "spec_id": "HENRI-SPEC-GATE-A-PRIME-IDF-2026",
        "arm": "idf-only",
        "flag": "--ast-idf-only",
        "d_model": args.d_model,
        "codebook_n": args.codebook_n,
        "codebook_kept": len(cb_vecs),
        "mbpp_sha": mbpp_sha,
        "ranks": ranks,
        "e_cos": round(e_cos, 6),
        "deterministic": deterministic,
        "verdict": verdict,
        "commit": git_head(),
        "command": " ".join(sys.argv),
        "wall_s": round(time.time() - t0, 1),
        "gate_b_condition": verdict == "PASS",
    }
    out_dir = args.output_dir or os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)
    receipt_path = os.path.join(out_dir, "arc_phase839_gate_a_prime_receipt.json")
    with open(receipt_path, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(f"[gate-a-prime] receipt -> {receipt_path}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
