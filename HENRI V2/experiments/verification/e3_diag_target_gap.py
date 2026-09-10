"""E3 diagnostic (read-only, no training) — target-definition gap.

Question: does the frozen E2 probe centroid satisfy E2's OWN objective (align
to the window teacher centroid) while failing the TOKEN-level objective (align
to the gold next-token embedding)? If yes, the Gate-3.1 "readout construct
mismatch" diagnosis is falsified: the head is doing exactly what it was trained
to do, and the real gap is the TARGET DEFINITION (window centroid vs token).

Measures, over the first N windows of the SEALED E2 evaluation slice:
  1. mean cos(z_hat, window_centroid_target)   -- E2's training objective
  2. mean cos(z_hat, gold_token_embedding)     -- what token-level P@1 needs
  3. gold-token rank of z_hat over the full table; top-16 probe coverage
  4. P@1 reproduced from z_hat argmax (sanity check vs the sealed run)

No weights are modified. No training. Read-only telemetry.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("/root/e3/diag_target_gap.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    e2_dir = Path(os.environ.get("E2_DIR", "/root/e2-calib/out_full"))
    t_dir = Path(os.environ.get("TEACHER_DIR", "/root/e1-calib"))
    corpus = Path(os.environ.get("CORPUS", "/root/e2-corpus/wikitext2_train.parquet"))

    from e2_calibrate import (  # noqa: E402
        load_corpus, sentence_split, make_tokenizer, build_pairs_ordered)
    from e1_egress_calibration import (  # noqa: E402
        E1Config, E1EgressHead, load_teacher_embeddings)
    from e3_calibrate import gold_next_token  # noqa: E402
    from e3_egress_reform import probe_token_distribution  # noqa: E402

    N_CALIB, N_EVAL = 10000, 1000
    dev = args.device
    sents = []
    for r in load_corpus(corpus):
        sents.extend(sentence_split(r))
    emb = load_teacher_embeddings(t_dir / "teacher_embeddings.pt")
    tok = make_tokenizer(t_dir)
    pairs = build_pairs_ordered(sents, E1Config(device="cpu"), emb, tok,
                                N_CALIB + N_EVAL)
    ev = pairs[N_CALIB:N_CALIB + args.limit]
    if not ev:
        raise SystemExit("E3DIAG_FATAL: empty eval slice")

    payload = torch.load(str(e2_dir / "e2_egress_production.pt"),
                         map_location="cpu", weights_only=True)
    cfg = payload["config"]
    head = E1EgressHead(E1Config(
        d_model=int(cfg.get("d_model", 65536)),
        num_blocks=int(cfg.get("num_blocks", 8192)),
        block_dim=int(cfg.get("block_dim", 8)),
        d_bottleneck=int(cfg.get("d_bottleneck", 256)),
        d_target=int(cfg.get("d_target", 896)),
        vocab_size=int(cfg.get("vocab_size", 151936)),
        seed=int(cfg.get("seed", 20260908)), device="cpu"))
    head.load_state_dict(payload["model_state"])
    head.to(dev).eval()

    E = emb.to(device=dev, dtype=torch.float32)
    En = F.normalize(E, p=2, dim=-1)

    cos_centroid, cos_gold, ranks, cover, top1_ok = [], [], [], 0, 0
    with torch.no_grad():
        for p in ev:
            gold = gold_next_token(p.text, tok)
            w = p.wave.to(dev).unsqueeze(0)
            feats, _ = head(w)
            scores, z_hat = probe_token_distribution(feats, E, k=16, tau=0.07)
            z = F.normalize(z_hat, p=2, dim=-1)
            tgt = F.normalize(p.target.to(dev).unsqueeze(0), p=2, dim=-1)
            g_emb = En[gold].unsqueeze(0)
            cos_centroid.append(float((z * tgt).sum()))
            cos_gold.append(float((z * g_emb).sum()))
            if int(scores.argmax(-1).item()) == gold:
                top1_ok += 1
            # rank of the gold token under z_hat over the full table
            gsim = float(scores[0, gold])
            ranks.append(int((scores[0] > gsim).sum().item()) + 1)
            # probe coverage: is gold among the k-NN teacher rows?
            sims = z @ En.t()
            idx = torch.topk(sims, 16, dim=-1).indices[0].tolist()
            if gold in idx:
                cover += 1

    n = len(ev)
    ranks_t = torch.tensor(ranks, dtype=torch.float32)
    rec = {
        "diagnostic": "E3_TARGET_DEFINITION_GAP",
        "label": "CONDITIONAL_SAME_CORPUS_HELDOUT",
        "n": n,
        "mean_cos_to_window_centroid_target": round(sum(cos_centroid) / n, 6),
        "mean_cos_to_gold_token_embedding": round(sum(cos_gold) / n, 6),
        "median_gold_rank": int(ranks_t.median().item()),
        "mean_gold_rank": round(float(ranks_t.mean()), 1),
        "gold_top1_rate": round(top1_ok / n, 6),
        "gold_in_probe_top16_rate": round(cover / n, 6),
        "vocab": int(En.shape[0]),
        "reading": ("E2 objective satisfied / token objective not"
                    if (sum(cos_centroid) / n) > (sum(cos_gold) / n) else
                    "token objective not worse than centroid"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec, indent=2))
    print("E3DIAG_DONE")


if __name__ == "__main__":
    main()
