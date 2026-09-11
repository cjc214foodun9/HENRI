"""E5b — C2 token-stream coverage gate (CPU arms + backbone context export).

Implements the sealed prereg (experiments/verification/e5b_c2_coverage_prereg.md).

CONSTRUCT: next-token on the raw token stream. Context = CTX tokens ending at
position i; gold = stream[i]. Fresh regions, disjoint from every prior carrier:
  calibration tokens 800,000..819,999   evaluation tokens 1,000,000..1,000,999

ARMS (zero trainable):
  C_const  top-k most frequent calib gold tokens
  C_ctx    top-k gold tokens observed after the context's last WORD (calib only)
  C_wave   the same library keyed on the terminal word DECODED from the context
           by the sealed E4b position channel (zero trainable)

BUG-CLASS GUARDS (each one bit me earlier this session):
  * decode reference = the CONTEXT's own last word (detokenized), never the gold.
  * library uses context-word -> observed GOLD TOKEN IDS from calib only; it is
    never built from within-context bigrams (those never observe a continuation).
  * library entries are token INTS, used directly; no word-expansion helper.
  * decode is a vectorized argmax over a precomputed e:-cell matrix, one op/sample.

Usage: python e5b_coverage_c2.py [--out PATH] [--export PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(HV2))

from e4b_position_codec import PositionBoundCodec, _pos_cells  # noqa: E402
from g7_highorder_codec import build_vocab  # noqa: E402

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_coverage_c2.json")
CTX_EXPORT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_c2_eval_contexts.pt")

CTX = 128
CALIB = (800_000, 820_000)
EVAL = (1_000_000, 1_001_000)
K_LIST = [1, 5, 16, 64, 256, 1024]
GAMMA = 0.90
NONINF = 0.02
E4A_C2_MARGINAL = 0.117


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--export", type=Path, default=CTX_EXPORT)
    args = ap.parse_args()
    t0 = time.time()
    rec: dict = {"carrier": "E5b", "construct": "C2_token_stream", "ctx": CTX,
                 "trainable_params": 0,
                 "label": "CONDITIONAL_FRESH_SPLIT_SAME_CORPUS",
                 "prereg": "experiments/verification/e5b_c2_coverage_prereg.md",
                 "regions": {"calib": list(CALIB), "eval": list(EVAL)},
                 "prereg_gates": {"gamma": GAMMA, "k_list": K_LIST,
                                  "noninf_tol": NONINF,
                                  "e4a_c2_marginal_crosscheck": E4A_C2_MARGINAL}}

    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer

    rec["corpus_sha256"] = sha(CORPUS)
    rec["tokenizer_sha256"] = sha(TOKJ)
    tok = Tokenizer.from_file(str(TOKJ))
    enc = lambda s: tok.encode(s).ids
    dec = lambda ids: tok.decode(ids)

    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream: list[int] = []
    need = EVAL[1] + CTX + 8
    for r in rows:
        stream.extend(enc(r))
        if len(stream) >= need:
            break
    rec["stream_tokens"] = len(stream)
    rec["stream_need"] = need
    print("[0] stream " + str(len(stream)) + " tokens (need " + str(need) + ")",
          flush=True)
    if len(stream) < need:
        rec["VERDICT"] = "BLOCKED_INFRA"
        rec["reason"] = "STREAM_TOO_SHORT"
        args.out.write_text(json.dumps(rec, indent=2))
        print("BLOCKED_INFRA: stream too short -> adjust regions")
        return

    # ---- libraries from the FRESH calib region ONLY -----------------------
    gold_freq = Counter(stream[i] for i in range(*CALIB))
    const_order = [t for t, _ in gold_freq.most_common(K_LIST[-1])]
    succ: dict[str, Counter] = defaultdict(Counter)
    for i in range(CALIB[0] + CTX, CALIB[1]):
        txt = dec(stream[i - CTX:i])
        words = txt.split()
        if words:
            succ[words[-1]][stream[i]] += 1
    RANK = {w: [t for t, _ in c.most_common(K_LIST[-1])] for w, c in succ.items()}
    rec["library"] = {"kind": "context_last_word -> ranked GOLD TOKEN ids (calib only)",
                      "const_distinct_golds": len(gold_freq),
                      "ctx_word_contexts": len(succ)}
    print("[1] const golds " + str(len(gold_freq))
          + " ctx word contexts " + str(len(succ)), flush=True)

    # ---- wave codec (sealed E4b; zero trainable) --------------------------
    vcb = build_vocab([dec(stream[i - CTX:i]) for i in range(
        CALIB[0] + CTX, min(CALIB[0] + CTX + 6000, CALIB[1]))], max_words=20000)
    codec = PositionBoundCodec(vcb)
    E_CELLS = np.stack([_pos_cells("e:" + w) for w in vcb]).astype(np.int64)
    E_SIGNS = np.where((E_CELLS % 2) == 0, 1.0, -1.0).astype(np.float32)
    rec["codec"] = {"class": "PositionBoundCodec (E4b)", "vocab": len(vcb),
                    "trainable_params": 0}
    print("[2] codec vocab " + str(len(vcb)), flush=True)

    def decode_terminal(txt: str):
        wb, _ = codec.encode(txt)
        flat = np.frombuffer(wb, dtype=np.float32).ravel()
        sc = ((flat[E_CELLS] * E_SIGNS) > 0.0).sum(axis=1)      # [V]
        i = int(np.argmax(sc))
        return vcb[i], float(sc[i]) / 16.0

    # ---- measure CPU arms -------------------------------------------------
    hits = {a: {k: 0 for k in K_LIST} for a in ("const", "ctx", "wave")}
    n = 0
    dec_ok = 0
    ex = []
    ctx_tensors = []
    golds_list = []
    for i in range(EVAL[0], EVAL[1]):
        ctx_toks = stream[i - CTX:i]
        gold = stream[i]
        txt = dec(ctx_toks)
        words = txt.split()
        n += 1
        for k in K_LIST:
            if gold in const_order[:k]:
                hits["const"][k] += 1
        surface = words[-1] if words else None
        if surface is not None:
            st = RANK.get(surface, [])
            for k in K_LIST:
                if gold in st[:k]:
                    hits["ctx"][k] += 1
        dw, dev = decode_terminal(txt)
        if surface is not None and dw == surface:      # CORRECT reference
            dec_ok += 1
        wt = RANK.get(dw, []) if dw else []
        for k in K_LIST:
            if gold in wt[:k]:
                hits["wave"][k] += 1
        ctx_tensors.append(ctx_toks)
        golds_list.append(gold)
        if len(ex) < 4:
            ex.append({"surface": surface, "decoded": dw, "ev": round(dev, 3),
                       "match": surface == dw, "gold": gold})

    cov = {a: {str(k): round(hits[a][k] / n, 6) for k in K_LIST} for a in hits}
    dec_acc = dec_ok / n
    rec.update({"n_eval": n, "coverage": cov,
                "wave_decode_accuracy": round(dec_acc, 6), "examples": ex})
    print("[3] const " + json.dumps(cov["const"]), flush=True)
    print("[3] ctx   " + json.dumps(cov["ctx"]), flush=True)
    print("[3] wave  " + json.dumps(cov["wave"]), flush=True)
    print("[3] decode acc " + str(round(dec_acc, 4)), flush=True)

    # ---- export contexts for the backbone arm (Vast) ----------------------
    torch.save({"contexts": torch.tensor(ctx_tensors, dtype=torch.long),
                "golds": torch.tensor(golds_list, dtype=torch.long),
                "meta": {"ctx": CTX, "calib": list(CALIB), "eval": list(EVAL),
                         "corpus_sha256": rec["corpus_sha256"],
                         "tokenizer_sha256": rec["tokenizer_sha256"]}},
               str(args.export))
    rec["contexts_export"] = {"path": str(args.export),
                              "sha256": sha(args.export)[:16], "n": len(ctx_tensors)}
    print("[4] exported " + str(len(ctx_tensors)) + " contexts -> "
          + str(args.export), flush=True)

    rec["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(args.out) + " sha256=" + sha(args.out)[:16])
    print("NOTE: G-C2-A/C need the backbone arm; gates evaluated after it returns.")


if __name__ == "__main__":
    main()
