"""E4c ORACLE PROBE — validate the frozen-backbone feature/oracle extraction.

WHY THIS EXISTS
---------------
e4c_extract_features.py reported ORACLE P@1 = 0.1400 on the sentence-window
construct (C1), while the C1 context-free marginal baseline is 0.440. If the
frozen 0.5B LM's OWN lm_head cannot beat "always emit the most frequent gold
token", then either:

  (H1) the C1 construct is DEGENERATE (the gold ' .' mode is an artifact of
       window construction that the prefix carries no signal about), or
  (H2) my oracle extraction is WRONG (hand-rolled norm/lm_head path diverges
       from the model's own logits).

Running the 2x2 factorial on a mis-extracted feature artifact would produce
four meaningless arms. This probe decides H1 vs H2 FIRST, at scaffold cost.

MEASUREMENTS
  A. manual logits vs model(...).logits bitwise: max abs diff + argmax agreement
  B. readable argmax for natural prefixes (human sanity)
  C. oracle P@1/P@5 on C1 (sentence-window) eval slice
  D. oracle P@1/P@5 on C2 (token-stream) contiguous sample
  E. diagnostics: gold distribution of the probed slices

Zero training. Read-only. Fail-closed on SHA.

Usage:
  HENRI_BACKBONE=1 python e4c_oracle_probe.py \
      --model-dir /root/e4c/qwen05b --corpus /root/e2-corpus/wikitext2_train.parquet \
      --out /root/e4c/oracle_probe.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
EXPECT_CORPUS_SHA_PREFIX = "e83889ba"
MAX_WORDS = 24
FRESH_CALIB = (11000, 21000)
FRESH_EVAL = (21000, 22000)
C2_EVAL_START = 84000        # pair index in the contiguous stream (fresh region)
C2_N = 1000
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
        print(f"PROBE_VERDICT=BLOCKED_INFRA reason=FLAG_OFF")
        raise SystemExit(2)

    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    shard = args.model_dir / "model.safetensors"
    if sha256_file(shard) != EXPECT_SHARD_SHA or shard.stat().st_size != EXPECT_SHARD_BYTES:
        print("PROBE_VERDICT=BLOCKED_INFRA reason=SHARD")
        raise SystemExit(2)
    csha = sha256_file(args.corpus)
    if not csha.startswith(EXPECT_CORPUS_SHA_PREFIX):
        print("PROBE_VERDICT=BLOCKED_INFRA reason=CORPUS")
        raise SystemExit(2)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))
    print(f"[probe] model loaded device={dev}", flush=True)

    @torch.no_grad()
    def predict(ids_list: list[list[int]], batch: int = 32):
        """Return (logits_last [B,V] float32, manual [B,V] float32)."""
        L = max(len(x) for x in ids_list)
        B = len(ids_list)
        inp = torch.zeros((B, L), dtype=torch.long, device=dev)
        for j, x in enumerate(ids_list):
            inp[j, :len(x)] = torch.tensor(x, device=dev)
        o = model(input_ids=inp, output_hidden_states=True)   # right-padded
        last = torch.tensor([len(x) - 1 for x in ids_list], device=dev)
        logits = o.logits[torch.arange(B), last, :].float()
        # hand-rolled path (what the extractor used)
        h = o.hidden_states[-1][torch.arange(B), last, :]
        h2 = model.model.norm(h.float())
        manual = model.lm_head(h2.to(torch.bfloat16)).float()
        return logits, manual

    report: dict = {"model_dir": str(args.model_dir), "device": dev}

    # ---------------- A. manual vs model logits --------------------------
    probes_txt = [
        "The capital of France is",
        "In 1865 the Union general Gordon Granger arrived in Galveston ,",
        "The church was dedicated in April 1860 to",
        "Water is composed of hydrogen and",
        "The quick brown fox jumps over the",
        "The population of Texas in 1860 was",
    ]
    ids = [tok.encode(t).ids for t in probes_txt]
    lg, man = predict(ids)
    maxdiff = float((lg - man).abs().max())
    am_model = lg.argmax(-1)
    am_manual = man.argmax(-1)
    agree = int((am_model == am_manual).sum())
    print(f"[A] manual-vs-model max|diff|={maxdiff:.4f}  "
          f"argmax agree {agree}/{len(probes_txt)}", flush=True)
    report["A_manual_vs_model"] = {
        "max_abs_diff": round(maxdiff, 4),
        "argmax_agreement": f"{agree}/{len(probes_txt)}",
    }

    # ---------------- B. readable argmax --------------------------------
    readable = []
    for t, i in zip(probes_txt, am_model.tolist()):
        nxt = tok.decode([i])
        readable.append({"prompt": t, "next_token": nxt})
        print(f"[B] {t!r:60} -> {nxt!r}", flush=True)
    report["B_readable_argmax"] = readable

    # ---------------- C. oracle on C1 (sentence-window) ------------------
    import pyarrow.parquet as pq
    texts = pq.read_table(str(args.corpus)).column("text").to_pylist()
    sents: list[str] = []
    for r in texts:
        sents.extend(sentence_split(r))
    win: list[tuple[str, str, int]] = []
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = tok.encode(window).ids, tok.encode(prefix).ids
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            win.append((prefix, window, ia[len(ip)]))
    c1 = win[FRESH_EVAL[0]:FRESH_EVAL[1]]
    p1 = p5 = 0
    golds = Counter()
    for i in range(0, len(c1), 32):
        chunk = c1[i:i + 32]
        lgc, _ = predict([tok.encode(p).ids for p, _, _ in chunk])
        g = torch.tensor([g for _, _, g in chunk], device=dev)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        p1 += int((t5[:, 0] == g).sum())
        p5 += int((t5 == g.unsqueeze(1)).any(1).sum())
        for _, _, gg in chunk:
            golds[gg] += 1
    n = len(c1)
    c1_p1, c1_p5 = p1 / n, p5 / n
    c1_top = golds.most_common(3)
    print(f"[C] C1 oracle P@1={c1_p1:.4f} P@5={c1_p5:.4f} n={n}", flush=True)
    print(f"[C] C1 gold top3: {[(t, round(c/n,3), tok.decode([t])) for t, c in c1_top]}",
          flush=True)
    report["C_C1_sentence_window"] = {
        "n": n, "oracle_p1": round(c1_p1, 4), "oracle_p5": round(c1_p5, 4),
        "gold_top3": [{"id": t, "share": round(c / n, 4), "tok": tok.decode([t])}
                      for t, c in c1_top],
        "marginal_baseline_p1_from_e4a": 0.440,
    }

    # ---------------- D. oracle on C2 (token stream) ---------------------
    stream: list[int] = []
    for r in texts[:3000]:
        stream.extend(tok.encode(r).ids)
    pairs = [(stream[i - 1], stream[i]) for i in range(1, len(stream))]
    seg = pairs[4 * C2_EVAL_START:4 * C2_EVAL_START + C2_N]
    d1 = d5 = 0
    for i in range(0, len(seg), 32):
        chunk = seg[i:i + 32]
        lgc, _ = predict([[ctx] for ctx, _ in chunk])
        g = torch.tensor([g for _, g in chunk], device=dev)
        t5 = torch.topk(lgc, 5, dim=-1).indices
        d1 += int((t5[:, 0] == g).sum())
        d5 += int((t5 == g.unsqueeze(1)).any(1).sum())
    m = len(seg)
    c2_p1, c2_p5 = d1 / m, d5 / m
    print(f"[D] C2 oracle P@1={c2_p1:.4f} P@5={c2_p5:.4f} n={m}", flush=True)
    report["D_C2_token_stream"] = {
        "n": m, "oracle_p1": round(c2_p1, 4), "oracle_p5": round(c2_p5, 4),
        "marginal_baseline_p1_from_e4a": 0.117,
    }

    # ---------------- verdict -------------------------------------------
    extraction_ok = (agree == len(probes_txt)) and (maxdiff < 0.5)
    c1_viable = c1_p1 >= 0.440 + 0.05
    c2_viable = c2_p1 >= 0.117 + 0.05

    if not extraction_ok:
        verdict = "PROBE_EXTRACTION_BUG"
    elif not c1_viable and c2_viable:
        verdict = "C1_CONSTRUCT_FALSIFIED_C2_VIABLE"
    elif c1_viable and not c2_viable:
        verdict = "C1_VIABLE_C2_UNEXPECTED"
    elif c1_viable and c2_viable:
        verdict = "BOTH_CONSTRUCTS_VIABLE"
    else:
        verdict = "BOTH_CONSTRUCTS_FALSIFIED"

    report["verdict"] = verdict
    report["gates"] = {
        "extraction_ok": bool(extraction_ok),
        "C1_oracle_viable": bool(c1_viable),
        "C2_oracle_viable": bool(c2_viable),
    }
    report["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(f"\nPROBE_VERDICT={verdict}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
