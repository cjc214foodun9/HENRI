"""E4c step 1 — extract frozen-backbone features for the sealed fresh split.

Loads Qwen2.5-0.5B (revision-pinned, per-shard SHA-256 verified, eval-only,
zero trainable) and computes, for each window prefix:

    h = final_norm(backbone(input_ids = tok(prefix)).hidden_states[-1][:, -1, :])

`h` is exactly what the backbone's own lm_head consumes, so the linear-readout
comparison is fair. Also records the backbone's OWN argmax (oracle) per window.

Fail-closed: shard SHA, config, tokenizer all verified before load.
Nothing is trained here. Artifact is hashed.

Usage (remote):
  HENRI_BACKBONE=1 python e4c_extract_features.py \
      --model-dir /root/e4c/qwen05b --corpus /root/e2-corpus/wikitext2_train.parquet \
      --out /root/e4c/features.pt
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
EXPECT_CORPUS_SHA_PREFIX = "e83889ba"
VOCAB = 151936
HIDDEN = 896
MAX_WORDS = 24
FRESH_CALIB = (11000, 21000)
FRESH_EVAL = (21000, 22000)
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
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    # ---- default-OFF flag: this module is only importable/constructible when
    #      the backbone is explicitly enabled (CLASS51 pattern).
    if os.environ.get(FLAG) != "1":
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=FLAG_OFF ({FLAG}!=1)")
        raise SystemExit(2)

    import torch
    from tokenizers import Tokenizer
    from transformers import AutoModelForCausalLM

    t0 = time.time()
    shard = args.model_dir / "model.safetensors"
    for f in (shard, args.model_dir / "config.json",
              args.model_dir / "tokenizer.json", args.corpus):
        if not f.exists():
            print(f"E4C_VERDICT=BLOCKED_INFRA reason=ARTIFACT_MISSING {f}")
            raise SystemExit(2)
    if shard.stat().st_size != EXPECT_SHARD_BYTES:
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=SHARD_BYTES "
              f"{shard.stat().st_size}!={EXPECT_SHARD_BYTES}")
        raise SystemExit(2)
    shard_sha = sha256_file(shard)
    if shard_sha != EXPECT_SHARD_SHA:
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=SHARD_SHA {shard_sha[:16]}")
        raise SystemExit(2)
    corpus_sha = sha256_file(args.corpus)
    if not corpus_sha.startswith(EXPECT_CORPUS_SHA_PREFIX):
        print(f"E4C_VERDICT=BLOCKED_INFRA reason=CORPUS_SHA {corpus_sha[:16]}")
        raise SystemExit(2)
    print(f"[e4c] shard sha OK {shard_sha[:16]}  corpus {corpus_sha[:16]}", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        attn_implementation="eager", low_cpu_mem_usage=True).to(dev).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_backbone = sum(p.numel() for p in model.parameters())
    print(f"[e4c] backbone loaded params={n_backbone} device={dev} "
          f"wall={time.time()-t0:.1f}s", flush=True)

    tok = Tokenizer.from_file(str(args.model_dir / "tokenizer.json"))

    import pyarrow.parquet as pq
    texts = pq.read_table(str(args.corpus)).column("text").to_pylist()
    sents: list[str] = []
    for r in texts:
        sents.extend(sentence_split(r))
    print(f"[e4c] sentences {len(sents)}", flush=True)

    win: list[tuple[str, str, int]] = []
    dropped = 0
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = tok.encode(window).ids, tok.encode(prefix).ids
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            win.append((prefix, window, ia[len(ip)]))
        else:
            dropped += 1
    print(f"[e4c] windows {len(win)} prefix_unstable_dropped {dropped}", flush=True)
    need = FRESH_EVAL[1]
    if len(win) < need:
        print(f"E4C_VERDICT=BLOCKED_CONSTRUCT reason=INSUFFICIENT_WINDOWS {len(win)}")
        raise SystemExit(2)

    def slice_(rng):
        return win[rng[0]:rng[1]]

    out: dict = {"calib": {}, "eval": {}}
    with torch.no_grad():
        for name, rng in (("calib", FRESH_CALIB), ("eval", FRESH_EVAL)):
            part = slice_(rng)
            H, G, ORACLE, TXT = [], [], [], []
            for i in range(0, len(part), args.batch):
                chunk = part[i:i + args.batch]
                ids = [tok.encode(p).ids for p, _, _ in chunk]
                L = max(len(x) for x in ids)
                pad = 0
                inp = torch.full((len(ids), L), pad, dtype=torch.long, device=dev)
                mask = torch.zeros((len(ids), L), dtype=torch.long, device=dev)
                for j, x in enumerate(ids):
                    inp[j, :len(x)] = torch.tensor(x, device=dev)
                    mask[j, :len(x)] = 1
                o = model(input_ids=inp, attention_mask=mask,
                          output_hidden_states=True)
                last = torch.tensor([len(x) - 1 for x in ids], device=dev)
                h = o.hidden_states[-1][torch.arange(len(ids)), last, :]  # pre-norm
                h = model.model.norm(h.float())                            # post-norm
                oracle = model.lm_head(h.to(torch.bfloat16)).float().argmax(-1)
                H.append(h.cpu())
                G.extend(int(g) for _, _, g in chunk)
                ORACLE.extend(int(x) for x in oracle.cpu())
                TXT.extend(p for p, _, _ in chunk)
            Hs = torch.cat(H, 0)
            out[name] = {
                "h": Hs, "gold": torch.tensor(G, dtype=torch.long),
                "oracle": torch.tensor(ORACLE, dtype=torch.long), "names": TXT,
            }
            print(f"[e4c] {name}: h {tuple(Hs.shape)}", flush=True)

    # oracle sanity on eval
    ev = out["eval"]
    p1 = float((ev["oracle"] == ev["gold"]).float().mean())
    print(f"[e4c] ORACLE P@1 on eval = {p1:.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"features": out, "meta": {
        "shard_sha256": shard_sha, "corpus_sha256": corpus_sha,
        "n_backbone_params": n_backbone, "hidden": HIDDEN, "vocab": VOCAB,
        "fresh_calib": list(FRESH_CALIB), "fresh_eval": list(FRESH_EVAL),
        "oracle_p1_eval": p1,
    }}, str(args.out))
    a = sha256_file(args.out)
    print(f"[e4c] wrote {args.out} sha256 {a}")
    print(f"[e4c] wall {time.time()-t0:.1f}s")
    print("E4C_EXTRACT_DONE")


if __name__ == "__main__":
    sys.exit(main())
