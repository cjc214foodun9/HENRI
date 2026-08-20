"""Phase 8.39 Class 3.0 — pre-registered Proxy Gate A probe (CPU, d=2048).

Spec: HENRI_Class_3.0_Discriminative_Phase_Representation_Plan.md
Test population: the runner's real 71-candidate grammar pool (WaveASTDecoder
._instantiate, same grammar tables) evaluated against the real MBPP codebook
(N=100 waves from canonical mbpp.jsonl sha ccf64ceae9c5403b).

Configurations measured (all from ASTDiscriminativeEncoder):
  none    = Class 2.0 behavior (control; expect E[cos] ~ 0.59)
  idf     = Lever 3.2 only (--ast-idf-weighting)
  carrier = Lever 3.1 only (--codec-carrier-subtract)
  both    = Lever 3.1 + 3.2  <-- the gated Class 3.0 treatment

Gate A (pre-registered, treatment arm):
  M1: E[cos] across the 71 candidates <= 0.10
  M2: HumanEval/23 (return len(string)) rank <= 5/71 AND
      HumanEval/35 (return max(l))     rank <= 5/71
  Kill: either fails -> FALSIFIED; Gate B skipped.

Ranking mechanism (spec Lever 2.2/Class 2.0 precedent): mean raw phase-cosine
vs the MBPP codebook attractor bank, candidates AND codebook encoded in the
same configuration.
"""

import ast
import json
import math
import os
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
    compile_carrier_vector,
)
from wave_ast_decoder import WaveASTDecoder  # noqa: E402

D_MODEL = 2048  # cheapest-kill proxy dim (Class 2.0 precedent)
N_CODEBOOK = 100
MBPP_PATH = os.path.join(HENRI, "data", "mbpp.jsonl")
HUMANEVAL_PATH = os.path.join(HENRI, "data", "HumanEval.jsonl.gz")

CORRECT_BODIES = {
    "HumanEval/23": "    return len(string)",
    "HumanEval/35": "    return max(l)",
}


def load_mbpp():
    with open(MBPP_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_humaneval():
    import gzip

    with gzip.open(HUMANEVAL_PATH, "rt", encoding="utf-8") as f:
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


def encode_all(encoder, srcs):
    return [encoder.encode_code_string(s) for s in srcs]


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


def main():
    t0 = time.time()
    mbpp = load_mbpp()
    print(f"[class3] MBPP records: {len(mbpp)}")

    mbpp_codes = []
    for item in mbpp:
        code = item.get("code", item.get("solution", ""))
        if isinstance(code, str) and code.strip():
            mbpp_codes.append(code)
    print(f"[class3] MBPP usable codes: {len(mbpp_codes)}")

    # Phase 1: node histogram + global carrier (across all usable MBPP ASTs).
    freqs, corpus_size = build_idf_frequencies(mbpp_codes)
    print(f"[class3] corpus_size={corpus_size} node_types={len(freqs)}")
    top = sorted(freqs.items(), key=lambda kv: -kv[1])[:8]
    print("[class3] top node types:", ", ".join(f"{k}:{v}" for k, v in top))

    carrier = compile_carrier_vector(
        mbpp_codes, d_model=D_MODEL, device="cpu", node_frequencies=freqs,
        corpus_size=corpus_size,
    )
    print(f"[class3] carrier compiled (uint8, shape {tuple(carrier.shape)})")

    # HumanEval items + production grammar pools.
    he = {it["task_id"]: it for it in load_humaneval()}
    decoder = WaveASTDecoder(codec=None, device="cpu")
    pools = {}
    for tid in ("HumanEval/23", "HumanEval/35"):
        prompt = he[tid]["prompt"]
        m = __import__("re").search(r"def\s+(\w+)\s*\(([^)]*)\)", prompt)
        entry, args = m.group(1), [a.split(":")[0].strip() for a in m.group(2).split(",") if a.strip()]
        pools[tid] = build_candidate_pool(decoder, entry, args)
        print(f"[class3] {tid}: entry={entry} args={args} pool={len(pools[tid])}")

    codebook_codes = mbpp_codes[:N_CODEBOOK]
    print(f"[class3] codebook N={len(codebook_codes)}")

    configs = [
        ("none", False, False),
        ("idf", True, False),
        ("carrier", False, True),
        ("both", True, True),
    ]

    results = {}
    for name, idf, carr in configs:
        enc = ASTDiscriminativeEncoder(
            d_model=D_MODEL, device="cpu", idf_weighting=idf,
            carrier_subtract=carr, carrier_vector=carrier if carr else None,
            node_frequencies=freqs if idf else None, corpus_size=corpus_size,
        )
        # Codebook in the same configuration.
        cb_vecs = [enc.encode_code_string(c) for c in codebook_codes]
        cb_vecs = [v for v in cb_vecs if v is not None]
        # Determinism check on one candidate.
        d0 = enc.encode_code_string(pools["HumanEval/23"][0][0])
        d1 = enc.encode_code_string(pools["HumanEval/23"][0][0])
        det = bool(torch.equal(d0, d1)) if (d0 is not None and d1 is not None) else False

        cos_baseline = mean_pairwise_cosine(
            [enc.encode_code_string(s) for s, _ in pools["HumanEval/23"]]
        )
        ranks = {}
        for tid, cb in (("HumanEval/23", cb_vecs), ("HumanEval/35", cb_vecs)):
            ranks[tid] = rank_correct_body(
                enc, pools[tid], cb, CORRECT_BODIES[tid]
            )[0]
        results[name] = {
            "cos": cos_baseline, "deterministic": det, "ranks": ranks,
            "codebook_kept": len(cb_vecs),
        }
        print(
            f"[class3] {name:8s} E[cos]={cos_baseline:.4f} "
            f"ranks /23={ranks['HumanEval/23']} /35={ranks['HumanEval/35']} "
            f"det={det}"
        )

    # Gate A verdict on the treatment arm (both levers).
    tr = results["both"]
    m1 = tr["cos"] <= 0.10
    m2 = tr["ranks"]["HumanEval/23"] <= 5 and tr["ranks"]["HumanEval/35"] <= 5
    verdict = "PASS" if (m1 and m2) else "FALSIFIED"
    print(f"\n[class3] GATE A treatment (both): M1(cos<=0.10)={m1} "
          f"M2(ranks<=5)={m2} -> {verdict}")
    print(f"[class3] total wall: {time.time() - t0:.1f}s")
    sys.exit(0 if verdict == "PASS" else 1)


if __name__ == "__main__":
    main()
