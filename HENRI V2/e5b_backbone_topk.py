"""E5b backbone arm: frozen Qwen2.5-0.5B top-K coverage on the exported C2 contexts.

Zero trainable. Eval-only. Fail-closed on shard SHA.

Coverage computed simply and correctly:
    match[i, j] = (topk_ids[i, j] == golds[i])        # [N, TOPK] bool
    coverage@k  = match[:, :k].any(dim=1).mean()
    oracle_p1   = match[:, 0].mean()
(An earlier draft used rank-argmax on all-False rows, which reports rank 0 for a
miss; that is wrong and is replaced here.)

Usage (remote):
  HENRI_BACKBONE=1 python e5b_backbone_topk.py \
      --contexts /root/e5b/e5b_c2_eval_contexts.pt \
      --model-dir /root/e4c/qwen05b \
      --out /root/e5b/e5b_backbone_topk.pt --json /root/e5b/e5b_backbone_topk.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import torch

EXPECT_SHARD_SHA = "88c142557820ccad55bb59756bfcfcf891de9cc6202816bd346445188a0ed342"
EXPECT_SHARD_BYTES = 988097824
FLAG = "HENRI_BACKBONE"
TOPK = 1024
K_LIST = [1, 5, 16, 64, 256, 1024]


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", type=Path, required=True)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    if os.environ.get(FLAG) != "1":
        print("E5B_BACKBONE_VERDICT=BLOCKED_INFRA reason=FLAG_OFF")
        raise SystemExit(2)

    shard = args.model_dir / "model.safetensors"
    if not shard.exists():
        print("E5B_BACKBONE_VERDICT=BLOCKED_INFRA reason=SHARD_MISSING")
        raise SystemExit(2)
    if shard.stat().st_size != EXPECT_SHARD_BYTES:
        print("E5B_BACKBONE_VERDICT=BLOCKED_INFRA reason=SHARD_BYTES")
        raise SystemExit(2)
    got = sha(shard)
    if got != EXPECT_SHARD_SHA:
        print("E5B_BACKBONE_VERDICT=BLOCKED_INFRA reason=SHARD_SHA " + got[:16])
        raise SystemExit(2)
    print("[bb] shard sha OK " + got[:16], flush=True)

    from transformers import AutoModelForCausalLM

    pay = torch.load(str(args.contexts), map_location="cpu", weights_only=True)
    ctx = pay["contexts"]
    golds = pay["golds"]
    meta = pay.get("meta", {})
    N, L = ctx.shape
    print("[bb] contexts " + str(tuple(ctx.shape)) + " golds " + str(tuple(golds.shape))
          + " meta_ctx=" + str(meta.get("ctx")), flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        str(args.model_dir), torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True).to(dev).eval()
    for p_ in model.parameters():
        p_.requires_grad_(False)
    print("[bb] backbone loaded device=" + dev, flush=True)

    topk_ids = torch.zeros(N, TOPK, dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, N, args.batch):
            xb = ctx[i:i + args.batch].to(dev)
            o = model(input_ids=xb)
            logits = o.logits[:, -1, :].float()
            topk_ids[i:i + xb.shape[0]] = torch.topk(logits, TOPK, dim=-1).indices.cpu()
    wall = round(time.time() - t0, 2)
    print("[bb] wall " + str(wall) + "s", flush=True)

    match = (topk_ids == golds.unsqueeze(1))          # [N, TOPK] bool
    cov = {str(k): round(float(match[:, :k].any(dim=1).float().mean()), 6)
           for k in K_LIST}
    oracle_p1 = cov["1"]
    for k in K_LIST:
        print("[bb] k=" + str(k) + " coverage=" + str(cov[str(k)]), flush=True)
    print("[bb] oracle P@1 = " + str(oracle_p1), flush=True)

    rec = {"carrier": "E5b-backbone", "topk": TOPK, "n": int(N), "ctx": int(L),
           "coverage": cov, "oracle_p1": oracle_p1, "wall_s": wall,
           "shard_sha256": got, "contexts_meta": meta,
           "gold_in_top1024": cov["1024"]}
    torch.save({"topk_ids": topk_ids, "golds": golds, "coverage": cov,
                "oracle_p1": oracle_p1, "wall_s": wall, "meta": rec},
               str(args.out))
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(rec, indent=2))
    print("[bb] wrote " + str(args.json), flush=True)
    print("E5B_BACKBONE_DONE")


if __name__ == "__main__":
    main()
