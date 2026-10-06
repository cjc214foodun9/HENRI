"""Slot-count sweep at FIXED aggregate rank (d_model=1024, dim=65536).

Answers A2 completely. resolve_n_mem returns n_mem = d_model // d_k, so
n_mem * d_k == d_model for EVERY arm. Aggregate key rank is therefore constant
and the metric G-U4 (a linear readout R^2 of the aggregate structure) cannot see
the difference. What varies is the SPLIT: how many memory slots, and how wide
each slot's key is.

Sweep (d_model=1024, aggregate rank 1024 in every row):
    dk_target=4  -> n_mem=256, d_k=4    (the committed old rule)
    dk_target=8  -> n_mem=128, d_k=8
    dk_target=16 -> n_mem=64,  d_k=16
    dk_target=32 -> n_mem=32,  d_k=32   (Spec B)
    dk_target=64 -> n_mem=16,  d_k=64

Measured per arm: pairwise |cos| collapse of the pooled features, KSG MI to the
4-slot energy target, the shuffled-target MI control, and the v1/v2 G-U4 value.
If collapse tracks d_k (not n_mem), per-slot WIDTH is the operative variable.

CPU-only. One seed, one scale. Compact JSON.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core import substrate as sub                        # noqa: E402
from henri_core.exp_perslot_replication import measure         # noqa: E402
from henri_core.m4_generative import build_corpus              # noqa: E402
from henri_core.model3_decoder import DecoderConfig            # noqa: E402
from henri_core.system import TriModelSystem                   # noqa: E402
from henri_core.tokenizer import ByteBPE                       # noqa: E402

DIM, DMODEL = 65536, 1024


def build(tok, dk_target, seed):
    cfg = DecoderConfig(dim=DIM, d_model=DMODEL, n_layers=24, n_heads=16,
                        n_kv_heads=4, d_ffn=2816, n_macro=256, n_invariants=256,
                        vocab=tok.vocab_size, dk_target=dk_target)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        system = TriModelSystem(vocab=tok.vocab_size, dim=DIM, small=False,
                                decoder_cfg=cfg, dk_target=dk_target,
                                ingress_seed=seed)
    system.eval()
    return system


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dk-targets", default="4,8,16,32,64")
    ap.add_argument("--n-texts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)

    rows = []
    for dkt in [int(x) for x in a.dk_targets.split(",")]:
        try:
            system = build(tok, dkt, a.seed)
            pool = system.decoder.pooling
            r = measure(system, tok, a.n_texts, 32, a.seed)
            r.update({"dk_target": dkt, "n_mem": int(pool.n_mem),
                      "d_k": int(pool.d_k), "d_model": int(system.decoder.cfg.d_model),
                      "aggregate_rank": int(pool.n_mem) * int(pool.d_k)})
            del system
            gc.collect()
        except Exception as e:                                # noqa: BLE001
            r = {"dk_target": dkt, "status": "BLOCKED", "error": f"{type(e).__name__}: {e}"}
        rows.append(r)
        print(json.dumps(r), flush=True)

    good = [r for r in rows if "collapse" in r]
    out = {
        "schema": "henri.slot_sweep.v1", "pin": 20261005,
        "question": "at fixed aggregate rank, is the operative variable slot COUNT or slot WIDTH?",
        "invariant": "n_mem * d_k == d_model for every arm (resolve_n_mem)",
        "rank_constant": all(r["aggregate_rank"] == DMODEL for r in good),
        "rows": rows,
        "trend": {
            "d_k": [r["d_k"] for r in good],
            "n_mem": [r["n_mem"] for r in good],
            "collapse": [r["collapse"] for r in good],
            "mi_real": [r["mi_real"] for r in good],
            "mi_shuffled": [r["mi_shuffled"] for r in good],
        },
        "n_texts": a.n_texts, "seed": a.seed,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out["trend"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
