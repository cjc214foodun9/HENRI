"""E4c PROBE 2 — construct viability measured with the REAL model logits.

WHY
---
Probe 1 established (OBSERVED):
  * readable argmax is CORRECT  ('...France is' -> ' Paris'; '...hydrogen and' ->
    ' oxygen'; '...falls over the' -> ' lazy')  -> model + tokenizer are fine
  * C1 (sentence-window) ORACLE P@1 = 0.2690 via the model's OWN o.logits
  * C1 context-free marginal baseline = 0.440
  => the oracle is BELOW the trivial baseline. If the best possible linear readout
     on these features cannot beat "always emit ' .'", the C1 construct is
     DEGENERATE and unwinnable. No readout reform can fix it.
  * my hand-rolled path (hidden_states[-1] then model.norm) differs from
     o.logits by max|diff|=5.81 -> the extraction script is double-normalizing.

THIS SCRIPT
  A. determine the correct hidden-state path EMPIRICALLY (A = no extra norm,
     B = extra norm) by comparing lm_head(h) to o.logits
  B. C2 (token-stream) oracle on a FRESH stream region, window 128
  C. C2 baselines on the SAME region: context-free unigram + last-token
  D. C1 oracle reconfirm on model logits
  E. tokenizer SHA (the E4a baselines were computed with a tokenizer copy that
     was never hashed -> pin it here)

Zero training. Read-only. Fail-closed on shard/corpus SHA.

Usage:
  HENRI_BACKBONE=1 python e4c_probe2.py --model-dir /root/e4c/qwen05b \
      --corpus /root/e2-corpus/wikitext2_train.parquet --out /root/e4c/probe2.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

import torch

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
EXPECT_CORPUS_SHA_PREFIX = "e83889ba"
MAX_WORDS = 24
FRESH_C1_EVAL = (21000, 22000)
C2_BASE_TOKENS = 200_000          # calibration region for baselines
C2_FRESH_START = 500_000          # eval region (fresh, disjoint from E4a)
C2_N = 1000
CTX = 128
FLAG = "HENRI_BACKBONE"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def sentence_split(t: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", t.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get(FLAG) != "1":
        print("PROBE2_VERDICT=BLOCKED_INFRA reason=FLAG_OFF")
        raise SystemExit(2)

    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    shard = args.model_dir / "model.safetensors"
    if (sha256_file(shard) != EXPECT_SHARD_SHA
            or shard.stat().st_size != EXPECT_SHARD_BYTES):
        print("PROBE2_VERDICT=BLOCKED_INFRA reason=SHARD")
        raise SystemExit(2)
    csha = sha256_file(args.corpus)
    if not csha.startswith(EXPECT_CORPUS_SHA_PREFIX):
        print("PROBE2_VERDICT=BLOCKED_INFRA reason=CORPUS")
        raise SystemExit(2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    tok_sha = sha256_file(args.model_dir / "tokenizer.json")
    print(f"[p2] model loaded device={dev} tokenizer_sha={tok_sha[:16]}", flush=True)

    @torch.no_grad()
    def run(ids_list: list[list[int]]):
        """Returns (logits_model, hA, hB) all [B, ...] at each sequence's last pos."""
        B = len(ids_list)
        L = max(len(x) for x in ids_list)
        inp = torch.zeros((B, L), dtype=torch.long, device=dev)
        for j, x in enumerate(ids_list):
            inp[j, :len(x)] = torch.tensor(x, device=dev)
        o = model(input_ids=inp, output_hidden_states=True)
        ar = torch.arange(B, device=dev)
        last = torch.tensor([len(x) - 1 for x in ids_list], device=dev)
        lg_model = o.logits[ar, last, :].float()
        hs = o.hidden_states[-1][ar, last, :]
        hA = hs.float()                                   # candidate A: as-is
        hB = model.model.norm(hs.float())                 # candidate B: extra norm
        lgA = model.lm_head(hA.to(torch.bfloat16)).float()
        lgB = model.lm_head(hB.to(torch.bfloat16)).float()
        return lg_model, hA, hB, lgA, lgB

    report: dict = {"device": dev, "tokenizer_sha256": tok_sha,
                    "corpus_sha256": csha}

    # ---------------- A. correct hidden-state path ------------------------
    prompts = [
        "The capital of France is",
        "In 1865 the Union general Gordon Granger arrived in Galveston ,",
        "The church was dedicated in April 1860 to",
        "Water is composed of hydrogen and",
        "The quick brown fox jumps over the",
        "The population of Texas in 1860 was",
    ]
    lg, hA, hB, lgA, lgB = run([tok.encode(t).ids for t in prompts])
    dA = float((lg - lgA).abs().max())
    dB = float((lg - lgB).abs().max())
    agreeA = int((lg.argmax(-1) == lgA.argmax(-1)).sum())
    agreeB = int((lg.argmax(-1) == lgB.argmax(-1)).sum())
    correct = "A_raw" if dA <= dB else "B_extra_norm"
    print(f"[A] max|diff| raw-path={dA:.4f} (argmax {agreeA}/{len(prompts)})   "
          f"extra-norm-path={dB:.4f} (argmax {agreeB}/{len(prompts)})", flush=True)
    print(f"[A] CORRECT hidden-state path = {correct}", flush=True)
    report["A_hidden_path"] = {
        "max_diff_raw": round(dA, 4), "argmax_raw": f"{agreeA}/{len(prompts)}",
        "max_diff_extra_norm": round(dB, 4), "argmax_extra_norm": f"{agreeB}/{len(prompts)}",
        "correct_path": correct,
        "implication": ("e4c_extract_features.py applied model.model.norm to "
                        "hidden_states[-1]; if correct_path == A_raw that script "
                        "double-normalized and its features.pt is INVALID"),
    }
    for t, i in zip(prompts, lg.argmax(-1).tolist()):
        print(f"[B] {t!r:62} -> {tok.decode([i])!r}", flush=True)

    # ---------------- build the token stream (bounded) ---------------------
    import pyarrow.parquet as pq
    rows = pq.read_table(str(args.corpus)).column("text").to_pylist()
    need = C2_FRESH_START + C2_N + CTX + 16
    stream: list[int] = []
    used_rows = 0
    for r in rows:
        stream.extend(tok.encode(r).ids)
        used_rows += 1
        if len(stream) >= need:
            break
    print(f"[p2] stream_tokens={len(stream)} from {used_rows} rows (need {need})",
          flush=True)
    if len(stream) < need:
        print("PROBE2_VERDICT=BLOCKED_CONSTRUCT reason=STREAM_TOO_SHORT")
        raise SystemExit(2)

    # ---------------- B/C. C2 oracle + baselines on the SAME fresh region --
    base_tokens = stream[:C2_BASE_TOKENS]
    uni = Counter(base_tokens)
    uni_top = [t for t, _ in uni.most_common(8)]
    m1: dict = {}
    for i in range(1, len(base_tokens)):
        m1.setdefault(base_tokens[i - 1], Counter())[base_tokens[i]] += 1

    idxs = list(range(C2_FRESH_START, C2_FRESH_START + C2_N))
    o1 = o5 = 0
    u1 = u5 = 0
    l1 = l5 = 0
    for i in range(0, len(idxs), 32):
        chunk = idxs[i:i + 32]
        inputs = [stream[j - CTX:j] for j in chunk]
        gold = torch.tensor([stream[j] for j in chunk], device=dev)
        lgc, _, _, _, _ = run(inputs)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        o1 += int((t5[:, 0] == gold).sum())
        o5 += int((t5 == gold.unsqueeze(1)).any(1).sum())
        for k, j in enumerate(chunk):
            g = stream[j]
            u1 += int(uni_top[0] == g)
            u5 += int(g in uni_top[:5])
            c = m1.get(stream[j - 1])
            lst = ([t for t, _ in c.most_common(8)] if c else []) + uni_top
            if lst and lst[0] == g:
                l1 += 1
            if g in lst[:5]:
                l5 += 1
    n = len(idxs)
    print(f"[C] C2 ORACLE      P@1={o1/n:.4f} P@5={o5/n:.4f}  (n={n})", flush=True)
    print(f"[C] C2 unigram-bse P@1={u1/n:.4f} P@5={u5/n:.4f}", flush=True)
    print(f"[C] C2 lasttok-bse P@1={l1/n:.4f} P@5={l5/n:.4f}", flush=True)
    report["B_C2_token_stream"] = {
        "n": n, "ctx_window": CTX, "stream_tokens": len(stream),
        "eval_region": [C2_FRESH_START, C2_FRESH_START + C2_N],
        "base_region": [0, C2_BASE_TOKENS],
        "oracle_p1": round(o1 / n, 4), "oracle_p5": round(o5 / n, 4),
        "unigram_baseline_p1": round(u1 / n, 4), "unigram_baseline_p5": round(u5 / n, 4),
        "last_token_baseline_p1": round(l1 / n, 4), "last_token_baseline_p5": round(l5 / n, 4),
    }

    # ---------------- D. C1 oracle reconfirm (model logits) ---------------
    sents: list[str] = []
    for r in rows:
        sents.extend(sentence_split(r))
        if len(sents) > 30000:
            break
    win = []
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = tok.encode(window).ids, tok.encode(prefix).ids
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            win.append((prefix, ia[len(ip)]))
    c1 = win[FRESH_C1_EVAL[0]:FRESH_C1_EVAL[1]]
    p1 = p5 = 0
    golds = Counter()
    for i in range(0, len(c1), 32):
        chunk = c1[i:i + 32]
        lgc, _, _, _, _ = run([tok.encode(p).ids for p, _ in chunk])
        g = torch.tensor([g for _, g in chunk], device=dev)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        p1 += int((t5[:, 0] == g).sum())
        p5 += int((t5 == g.unsqueeze(1)).any(1).sum())
        for _, gg in chunk:
            golds[gg] += 1
    n1 = len(c1)
    c1p1, c1p5 = p1 / n1, p5 / n1
    top3 = [(t, round(c / n1, 4), tok.decode([t])) for t, c in golds.most_common(3)]
    print(f"[D] C1 ORACLE P@1={c1p1:.4f} P@5={c1p5:.4f} n={n1}", flush=True)
    print(f"[D] C1 gold top3: {top3}", flush=True)
    report["D_C1_sentence_window"] = {
        "n": n1, "oracle_p1": round(c1p1, 4), "oracle_p5": round(c1p5, 4),
        "gold_top3": [{"id": t, "share": s, "tok": d} for t, s, d in top3],
        "e4a_marginal_baseline_p1": 0.440,
    }

    # ---------------- verdict -------------------------------------------
    c1_viable = c1p1 >= 0.440 + 0.05
    c2_viable = (o1 / n) >= (u1 / n) + 0.05
    if not c1_viable and c2_viable:
        verdict = "C1_CONSTRUCT_FALSIFIED__C2_VIABLE"
    elif c1_viable and c2_viable:
        verdict = "BOTH_VIABLE"
    elif not c1_viable and not c2_viable:
        verdict = "BOTH_CONSTRUCTS_FALSIFIED"
    else:
        verdict = "C1_VIABLE__C2_UNEXPECTED"
    report["verdict"] = verdict
    report["gates"] = {"correct_path": correct,
                       "C1_oracle_viable": bool(c1_viable),
                       "C2_oracle_viable": bool(c2_viable)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nPROBE2_VERDICT={verdict}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
