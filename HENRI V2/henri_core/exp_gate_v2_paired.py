"""Paired per-slot discrimination: old rule vs Spec B at fixed aggregate rank.

Answers two approved questions with ONE experiment:
  A1  a discriminating metric where G-U4 saturates at d_model=1024
  A2  why the old rule passes at 1024 (n_mem=256 -> d_k=4)

Mechanism (confirmed by arithmetic, not assumed): resolve_n_mem returns
n_mem = d_model // d_k, so n_mem * d_k == d_model in BOTH rules:

    d_model  old(n_mem*d_k)      SpecB(n_mem*d_k)
    128      32 * 4  = 128       4  * 32 = 128
    512      128 * 4 = 512       16 * 32 = 512
    1024     256 * 4 = 1024      32 * 32 = 1024

Aggregate key rank is IDENTICAL. Only the slot-count x per-slot-width split
differs. G-U4 scores a linear readout of the AGGREGATE pooled structure, so it
tracks aggregate rank and cannot separate the arms at full scale. The per-slot
statistics (pairwise |cos| collapse, and KSG MI to the 4-slot energy target)
target the split that actually changed.

Both arms are required. No verdict is taken from a single arm.

CPU-only. No GPU, no Vast spend. The pooling path needs no transformer layers,
but n_layers is kept at the table value for faithfulness.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from henri_core.gates import gate_u4_retention                    # noqa: E402
from henri_core.gates_v2 import (gate_u4_retention_v2,            # noqa: E402
                                 gate_u5_perslot_mi)
from henri_core.m4_generative import build_corpus                 # noqa: E402
from henri_core.model3_decoder import DecoderConfig               # noqa: E402
from henri_core.system import TriModelSystem                      # noqa: E402
from henri_core.tokenizer import ByteBPE                          # noqa: E402

# name: (dim, d_model, n_layers, n_heads, n_kv_heads, d_ffn, n_macro, n_inv, small)
SCALES = {
    "full_1024": (65536, 1024, 24, 16, 4, 2816, 256, 256, False),
    "mid_512": (65536, 512, 4, 8, 2, 1024, 64, 64, False),
    "small_128": (4096, 128, 2, 4, 1, 256, 16, 32, True),
}


def build(tok, scale, dk_target, seed):
    dim, dm, nl, nh, nkv, dff, nmac, ninv, is_small = SCALES[scale]
    cfg = DecoderConfig(dim=dim, d_model=dm, n_layers=nl, n_heads=nh,
                        n_kv_heads=nkv, d_ffn=dff, n_macro=nmac,
                        n_invariants=ninv, vocab=tok.vocab_size,
                        dk_target=dk_target)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        system = TriModelSystem(vocab=tok.vocab_size, dim=dim, small=is_small,
                                decoder_cfg=None if is_small else cfg,
                                dk_target=dk_target, ingress_seed=seed)
    system.eval()
    return system


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scales", default="full_1024,small_128")
    ap.add_argument("--n-texts", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261004)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    t0 = time.time()
    corpus = build_corpus(max_len=3, holdout_len=3)
    tok = ByteBPE().train(corpus.corpus_texts, vocab_size=512)
    nt = a.n_texts

    rows = []
    for scale in a.scales.split(","):
        for label, dkt in (("old_rule", 0), ("spec_b", 32)):
            t1 = time.time()
            try:
                system = build(tok, scale, dkt, a.seed)
                pool = system.decoder.pooling
                u5 = gate_u5_perslot_mi(system, tok, n_texts=nt)
                u4v2 = gate_u4_retention_v2(system, tok, n_texts=nt)
                u4v1 = gate_u4_retention(system, tok, n_texts=nt)
                row = {"scale": scale, "arm": label,
                       "d_model": system.decoder.cfg.d_model,
                       "n_mem": int(pool.n_mem), "d_k": int(pool.d_k),
                       "collapse": u5["collapse_mean_pairwise_abs_cos"],
                       "ksg_mi": u5["ksg_mi"], "u5_status": u5["status"],
                       "u5_sane": u5["estimator_selftest"]["estimator_sane"],
                       "u4_v2": u4v2["value"], "u4_v2_status": u4v2["status"],
                       "u4_v2_sane": u4v2["estimator_sane"],
                       "u4_v1": u4v1["value"],
                       "sec": round(time.time() - t1, 2)}
                del system
                gc.collect()
            except Exception as e:                        # noqa: BLE001
                row = {"scale": scale, "arm": label, "status": "BLOCKED",
                       "error": f"{type(e).__name__}: {e}",
                       "sec": round(time.time() - t1, 2)}
            rows.append(row)
            print(f"[{scale} {label}] {json.dumps(row)}", flush=True)

    pair = {}
    for scale in a.scales.split(","):
        ar = {r["arm"]: r for r in rows if r["scale"] == scale}
        if "old_rule" in ar and "spec_b" in ar and "collapse" in ar["old_rule"] \
                and "collapse" in ar["spec_b"]:
            o, s = ar["old_rule"], ar["spec_b"]
            dcol = abs(o["collapse"] - s["collapse"])
            finite = (o["ksg_mi"] == o["ksg_mi"]) and (s["ksg_mi"] == s["ksg_mi"])
            pair[scale] = {
                "n_mem_old": o["n_mem"], "d_k_old": o["d_k"],
                "n_mem_spec_b": s["n_mem"], "d_k_spec_b": s["d_k"],
                "collapse_old": o["collapse"], "collapse_spec_b": s["collapse"],
                "d_collapse": dcol,
                "mi_old": o["ksg_mi"], "mi_spec_b": s["ksg_mi"],
                "d_mi": (s["ksg_mi"] - o["ksg_mi"]) if finite else float("nan"),
                "discriminated_collapse": bool(dcol >= 0.20),
                "u4_v1_old": o["u4_v1"], "u4_v1_spec_b": s["u4_v1"],
                "u4_v1_gap": abs(o["u4_v1"] - s["u4_v1"]),
            }
        else:
            pair[scale] = {"status": "BLOCKED"}

    disc = [v for v in pair.values() if isinstance(v, dict) and v.get("discriminated_collapse")]
    verdict = "DISCRIMINATED" if disc else "NOT_DISCRIMINATING"
    out = {
        "schema": "henri.gate_v2_paired.v1", "pin": 20261005,
        "question": "does a per-slot metric separate old_rule from spec_b where G-U4 cannot?",
        "mechanism": "n_mem*d_k == d_model in both rules -> identical aggregate rank",
        "rule": "arms DISCRIMINATED iff |d_collapse| >= 0.20 (prereg K4)",
        "rows": rows, "pair": pair, "verdict": verdict,
        "n_texts": nt, "seed": a.seed, "seconds": round(time.time() - t0, 2),
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({"verdict": verdict, "pair": pair}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
