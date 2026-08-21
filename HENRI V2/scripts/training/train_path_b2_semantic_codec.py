"""Train the Path B2 hard-negative discriminative codec (Class 4.4, default-OFF).

Ingests the MBPP code corpus (data/mbpp.jsonl, 974 items, sha256 ccf64ce...),
builds a bounded token+AST-node vocabulary with qFHRR-IDF document frequencies,
trains PathB2DiscriminativeCodec with hard-negative InfoNCE (tau=0.07) and
Cholesky retractions, and writes models/path_b2_codec.pt with provenance.

Checkpoint contract (consumed by humaneval_wave_ast_runner.py --path-b2-codec):
{
  "state_dict", "vocab", "df", "n_docs", "d_model", "d_latent",
  "val_contrastive_acc", "gram_max", "dataset_sha256", "split",
  "commit", "created_utc", "schema": "henri.path-b2-codec.v1"
}

Usage (local CPU smoke):
  python scripts/training/train_path_b2_semantic_codec.py --steps 32 --d-model 2048 --d-latent 64 --smoke
Usage (remote CUDA, production dims):
  python scripts/training/train_path_b2_semantic_codec.py --steps 1500 --d-model 65536 --d-latent 512
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
for p in (HERE.parents[2],):  # HENRI V2 root
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from path_b2_semantic_codec import (  # noqa: E402
    PathB2DiscriminativeCodec,
    KEYWORDS,
    ast_node_types,
    lex_code,
)

MBPP_PATH = "HENRI V2/data/mbpp.jsonl"
MAX_VOCAB = 4096


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load_mbpp(path: str) -> tuple[list[dict], str]:
    raw = open(path, "rb").read()
    digest = sha256_bytes(raw)
    items = []
    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(json.loads(line))
    return items, digest


def split_items(items: list[dict], val_frac: float = 0.1, seed: int = 7):
    import random

    rng = random.Random(seed)
    order = sorted(items, key=lambda d: str(d.get("task_id", "")))
    rng.shuffle(order)
    n_val = max(1, int(len(order) * val_frac))
    val, train = order[:n_val], order[n_val:]
    return train, val


def build_vocab_and_df(train: list[dict], val: list[dict], max_vocab: int = MAX_VOCAB):
    counts: dict[str, int] = {}
    df: dict[str, int] = {}
    for d in [*train, *val]:
        code = d.get("code", "")
        toks = lex_code(code) + ast_node_types(code)
        for tok in toks:
            counts[tok] = counts.get(tok, 0) + 1
        for tok in set(toks):
            df[tok] = df.get(tok, 0) + 1
    top = sorted(counts, key=lambda t: (-counts[t], t))[: max_vocab - 1]
    return top, df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--tau", type=float, default=0.07)
    ap.add_argument("--n-hard", type=int, default=8)
    ap.add_argument("--d-model", type=int, default=65536)
    ap.add_argument("--d-latent", type=int, default=512)
    ap.add_argument("--out", default="HENRI V2/models/path_b2_codec.pt",
                    help="checkpoint path relative to repo root")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--smoke", action="store_true", help="print receipts only")
    args = ap.parse_args()

    repo_root = Path.cwd()
    mbpp_path = repo_root / MBPP_PATH
    if not mbpp_path.exists():
        print("BLOCKED: HENRI V2/data/mbpp.jsonl missing (copy the reconciled overlay)")
        sys.exit(2)
    items, dataset_sha = load_mbpp(str(mbpp_path))
    print(f"[MBPP] items={len(items)} sha256={dataset_sha[:16]}")

    train, val = split_items(items, seed=args.seed)
    print(f"[SPLIT] train={len(train)} val={len(val)}")
    train_pairs = [(d.get("task_id", ""), d.get("code", "")) for d in train]
    val_pairs = [(d.get("task_id", ""), d.get("code", "")) for d in val]

    vocab, df = build_vocab_and_df(train, val)
    print(f"[VOCAB] {len(vocab)} tokens | [DF] {len(df)} doc-freq entries")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[DEVICE] {device}")
    codec = PathB2DiscriminativeCodec(
        d_model=args.d_model, d_latent=args.d_latent, vocab=vocab, df=df,
        n_docs=len(train) + len(val), device=device, seed=args.seed)
    codec = codec.to(device)

    t0 = time.perf_counter()
    metrics = codec.train_contrastive(
        train_pairs, val_pairs,
        steps=args.steps, batch_size=args.batch_size,
        lr=args.lr, tau=args.tau, n_hard=args.n_hard, seed=args.seed)
    print(f"[TRAIN] {metrics} wall={time.perf_counter() - t0:.1f}s")

    if args.smoke:
        print("[SMOKE] no checkpoint written")
        return

    out_path = repo_root / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import subprocess
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        commit = "unknown"
    ckpt = {
        "state_dict": {k: v.cpu() for k, v in codec.state_dict().items()},
        "vocab": vocab,
        "df": df,
        "n_docs": len(train) + len(val),
        "d_model": args.d_model,
        "d_latent": args.d_latent,
        "val_contrastive_acc": metrics["val_contrastive_acc"],
        "train_loss_mean": metrics["train_loss_mean"],
        "gram_max": metrics["gram_max"],
        "dataset_sha256": dataset_sha,
        "dataset": "MBPP",
        "split": {"train": len(train), "val": len(val)},
        "commit": commit,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "schema": "henri.path-b2-codec.v1",
    }
    torch.save(ckpt, str(out_path))
    raw = out_path.read_bytes()
    print(f"[CKPT] {out_path} sha256={sha256_bytes(raw)[:16]} bytes={len(raw)}")


if __name__ == "__main__":
    main()
