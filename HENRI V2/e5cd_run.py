"""E5c / E5d combined runner — full arm set on a prover-proven fresh region.

WHY THE FULL ARM SET
  E5b v2's wave arm decoded a WORD from the word-keyed codec, then looked the
  library up through a TOKEN ID found by scanning for a token whose piece equals
  that word (`next((t for t in gold_freq if dec([t]) == dw), -1)`). That is a
  TYPE MISMATCH between the decoded key and the library key, and it is why v2's
  wave arm collapsed toward `const` (wave@1 0.041 vs const@1 0.036).

  So "was the deficit region-specific?" cannot be answered by re-running a
  DIFFERENT implementation on a fresh region. This runner therefore reports:

    tok1        token-keyed, GT key        surface floor (token keying ceiling)
    tok2        token bigram + backoff     stronger surface floor
    ctxword_gt  word-keyed, GT key         word-keying CEILING
    ctxpiece_gt piece-keyed, GT key        piece-keying CEILING
    wave_word   word-keyed library, DECODED key       <- corrected word mechanism
    wave_piece  piece-keyed library, DECODED key      <- E5d's mechanism
    wave_v2     token-keyed library, word-decoded key <- EXACT v2 replication
    const, rand

  E5c (fresh R1) answers region-specificity by comparing wave_v2@64 vs tok1@64
  against E5b's recorded -0.060. E5d (fresh R2) tests the piece repair via
  wave_piece vs wave_word and ctxpiece_gt vs ctxword_gt.

FAITHFUL-PORT DEVIATION (disclosed): E4b's encode() takes `tokenize(text)[:24]`,
i.e. the FIRST 24 units. Its inputs were short sentences, so truncation rarely
bound. On a 128-token C2 context the first 24 units are nowhere near the
terminal, so this runner takes the LAST 24 units for both segmentations. The
terminal feature ("e:{terminal}") is the only feature the decode probe reads.

Zero trainable. Fail-closed on the prover receipt.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(HV2))

from e4b_position_codec import _pos_cells, _signs  # noqa: E402
from g7_highorder_codec import (  # noqa: E402
    BLOCK_DIM, NUM_BLOCKS, WAVE_DIM, build_vocab,
)

PROVER = Path(r"C:\Users\chan\henri-telemetry\e3\e5_fresh_regions.json")
CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
E3 = Path(r"C:\Users\chan\henri-telemetry\e3")

CTX, SUFFIX = 128, 24
K_LIST = [1, 5, 16, 64, 256, 1024]
ARMS = ("const", "tok1", "tok2", "ctxword_gt", "wave_word",
        "ctxpiece_gt", "wave_piece", "wave_v2", "rand")
VOCAB_CAP, VOCAB_BUILD_N = 20_000, 6_000


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def wave_of(units: list[str]) -> np.ndarray:
    """Position channel only (the terminal probe reads only `e:`/`d:` features)."""
    u = units[-SUFFIX:]
    acc = np.zeros(WAVE_DIM, dtype=np.float32)
    if not u:
        return acc.reshape(NUM_BLOCKS, BLOCK_DIM)
    feats = [f"e:{u[-1]}"]
    for i in range(1, min(3, len(u) - 1) + 1):
        feats.append(f"d{i}:{u[-1 - i]}")
    for f in feats:
        c = _pos_cells(f)
        np.add.at(acc, c, _signs(c))
    return acc.reshape(NUM_BLOCKS, BLOCK_DIM)


def probe_matrix(vocab: list[str]) -> tuple[np.ndarray, np.ndarray]:
    cells = np.stack([_pos_cells("e:" + v) for v in vocab]).astype(np.int64)
    signs = np.where((cells % 2) == 0, 1.0, -1.0).astype(np.float32)
    return cells, signs


def decode(flat: np.ndarray, cells: np.ndarray, signs: np.ndarray,
           vocab: list[str]) -> tuple[str, float]:
    sc = ((flat[cells] * signs) > 0.0).sum(axis=1)
    i = int(np.argmax(sc))
    return vocab[i], float(sc[i]) / 16.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--carrier", choices=["e5c", "e5d"], required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--ctx", type=Path)
    args = ap.parse_args()

    # ---------------- fail-closed prover gate ------------------------------
    if not PROVER.exists():
        print("VERDICT=BLOCKED_INFRA reason=PROVER_RECEIPT_MISSING")
        return
    prov = json.loads(PROVER.read_text())
    if prov.get("VERDICT") != "FRESH_REGIONS_PROVEN":
        print("VERDICT=BLOCKED_INFRA reason=PROVER_NOT_PROVEN")
        return
    which = "R1" if args.carrier == "e5c" else "R2"
    reg = prov["chosen"][which]
    CALIB, EVAL = tuple(reg["calib"]), tuple(reg["eval"])
    out = args.out or (E3 / ("e5c_fresh_rerun.json" if args.carrier == "e5c"
                             else "e5d_tail_keyed.json"))
    ctxout = args.ctx or (E3 / ("e5c_contexts.pt" if args.carrier == "e5c"
                                else "e5d_contexts.pt"))
    print(f"[0] {args.carrier} {which} calib={CALIB} eval={EVAL}", flush=True)

    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKJ))
    dec = lambda ids: tok.decode(ids)
    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream: list[int] = []
    for r in rows:
        stream.extend(tok.encode(r).ids)
        if len(stream) >= EVAL[1] + 8:
            break
    print(f"[1] stream {len(stream)}", flush=True)

    # ---------------- libraries (calib ONLY) -------------------------------
    gold_freq = Counter(stream[i] for i in range(*CALIB))
    const_order = [t for t, _ in gold_freq.most_common(K_LIST[-1])]
    lib_tok1: dict[int, Counter] = defaultdict(Counter)
    lib_tok2: dict[tuple, Counter] = defaultdict(Counter)
    lib_word: dict[str, Counter] = defaultdict(Counter)
    lib_piece: dict[str, Counter] = defaultdict(Counter)
    for i in range(CALIB[0] + 4, CALIB[1]):
        lib_tok1[stream[i - 1]][stream[i]] += 1
        lib_tok2[(stream[i - 2], stream[i - 1])][stream[i]] += 1
        lib_piece[dec([stream[i - 1]])][stream[i]] += 1
        w = dec(stream[i - CTX:i]).split()
        if w:
            lib_word[w[-1]][stream[i]] += 1
    R = lambda d: {k: [t for t, _ in c.most_common(K_LIST[-1])] for k, c in d.items()}
    L_tok1, L_tok2, L_word, L_piece = (R(lib_tok1), R(lib_tok2),
                                       R(lib_word), R(lib_piece))
    print(f"[2] libs tok1={len(L_tok1)} tok2={len(L_tok2)} "
          f"word={len(L_word)} piece={len(L_piece)} const={len(gold_freq)}", flush=True)

    # ---------------- vocabs (same context set -> one lever) ---------------
    build_ctx = [dec(stream[i - CTX:i]) for i in
                 range(CALIB[0] + CTX, min(CALIB[0] + CTX + VOCAB_BUILD_N, CALIB[1]))]
    word_vocab = build_vocab(build_ctx, max_words=VOCAB_CAP)
    pc: Counter = Counter()
    for i in range(CALIB[0] + CTX, min(CALIB[0] + CTX + VOCAB_BUILD_N, CALIB[1])):
        pc.update(dec([t]) for t in stream[i - CTX:i])
    piece_vocab = [p for p, _ in pc.most_common(VOCAB_CAP)]
    W_CELLS, W_SIGNS = probe_matrix(word_vocab)
    P_CELLS, P_SIGNS = probe_matrix(piece_vocab)
    print(f"[3] vocab words={len(word_vocab)} pieces={len(piece_vocab)}", flush=True)

    # ---------------- eval -------------------------------------------------
    rng = random.Random(20260914 if args.carrier == "e5c" else 20260915)
    universe = list(gold_freq)
    hits = {a: {k: 0 for k in K_LIST} for a in ARMS}
    n = 0
    dec_word = dec_piece = 0
    in_vocab_word = in_vocab_piece = 0
    ctx_t, gold_t = [], []
    ctx_word = seg_piece = None
    for i in range(*EVAL):
        ctx = stream[i - CTX:i]
        gold = stream[i]
        txt = dec(ctx)
        words = txt.split()
        pieces = [dec([t]) for t in ctx]
        gt_word = words[-1] if words else None
        gt_piece = dec([stream[i - 1]])
        n += 1

        for k in K_LIST:
            if gold in const_order[:k]:
                hits["const"][k] += 1
        s1 = L_tok1.get(stream[i - 1], const_order)
        for k in K_LIST:
            if gold in s1[:k]:
                hits["tok1"][k] += 1
        s2 = L_tok2.get((stream[i - 2], stream[i - 1]), s1)
        for k in K_LIST:
            if gold in s2[:k]:
                hits["tok2"][k] += 1

        # GT floors
        if gt_word is not None:
            if gt_word in L_word:
                in_vocab_word += 1
            sw = L_word.get(gt_word, const_order)
            for k in K_LIST:
                if gold in sw[:k]:
                    hits["ctxword_gt"][k] += 1
        if gt_piece in L_piece:
            in_vocab_piece += 1
        sp = L_piece.get(gt_piece, const_order)
        for k in K_LIST:
            if gold in sp[:k]:
                hits["ctxpiece_gt"][k] += 1

        # decoded-key mechanisms
        fw = np.frombuffer(wave_of(words).tobytes(), dtype=np.float32).ravel() \
            if words else np.zeros(WAVE_DIM, dtype=np.float32)
        dw, _ = decode(fw, W_CELLS, W_SIGNS, word_vocab) if words else (None, 0.0)
        if gt_word is not None and dw == gt_word:
            dec_word += 1
        swd = L_word.get(dw, const_order) if dw else const_order
        for k in K_LIST:
            if gold in swd[:k]:
                hits["wave_word"][k] += 1

        fp = np.frombuffer(wave_of(pieces).tobytes(), dtype=np.float32).ravel()
        dp, _ = decode(fp, P_CELLS, P_SIGNS, piece_vocab)
        if dp == gt_piece:
            dec_piece += 1
        spd = L_piece.get(dp, const_order)
        for k in K_LIST:
            if gold in spd[:k]:
                hits["wave_piece"][k] += 1

        # EXACT v2 replication: word-decoded key -> token-keyed library
        key_tok = next((t for t in gold_freq if dec([t]) == dw), -1) if dw else -1
        sv2 = L_tok1.get(key_tok, const_order)
        for k in K_LIST:
            if gold in sv2[:k]:
                hits["wave_v2"][k] += 1

        for k in K_LIST:
            if gold in rng.sample(universe, min(k, len(universe))):
                hits["rand"][k] += 1

        ctx_t.append(ctx)
        gold_t.append(gold)

    cov = {a: {str(k): round(hits[a][k] / n, 6) for k in K_LIST} for a in ARMS}
    rec = {
        "carrier": args.carrier.upper(), "region": reg, "which": which,
        "n_eval": n, "trainable_params": 0, "ctx": CTX, "suffix_units": SUFFIX,
        "coverage": cov,
        "decode_accuracy": {"word": round(dec_word / n, 6),
                            "piece": round(dec_piece / n, 6)},
        "gt_key_in_library": {"word": round(in_vocab_word / n, 6),
                              "piece": round(in_vocab_piece / n, 6)},
        "prover_sha256": sha(PROVER),
        "corpus_sha256": sha(CORPUS), "tokenizer_sha256": sha(TOKJ),
        "vocab_sizes": {"words": len(word_vocab), "pieces": len(piece_vocab)},
        "faithful_port_deviation": ("E4b used tokenize(text)[:24] (FIRST 24); "
                                   "C2 needs the LAST 24 to preserve the terminal"),
        "v2_replication_note": ("wave_v2 reproduces E5b v2's type-mismatched "
                                "lookup (word-decoded key -> token-keyed library) "
                                "for region comparability"),
        "wall_s": None,
    }
    for a in ARMS:
        print(f"[4] {a.ljust(12)} {json.dumps(cov[a])}", flush=True)
    print(f"[4] decode word={rec['decode_accuracy']['word']} "
          f"piece={rec['decode_accuracy']['piece']} | "
          f"gt_key_in_lib word={rec['gt_key_in_library']['word']} "
          f"piece={rec['gt_key_in_library']['piece']}", flush=True)

    torch.save({"contexts": torch.tensor(ctx_t, dtype=torch.long),
                "golds": torch.tensor(gold_t, dtype=torch.long),
                "meta": {"ctx": CTX, "region": reg, "carrier": rec["carrier"],
                         "corpus_sha256": rec["corpus_sha256"],
                         "tokenizer_sha256": rec["tokenizer_sha256"]}},
               str(ctxout))
    rec["contexts_export"] = {"path": str(ctxout), "sha256": sha(ctxout),
                              "n": len(ctx_t)}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=2))
    print(f"[5] exported {len(ctx_t)} -> {ctxout}", flush=True)
    print("VERDICT=" + rec["carrier"] + "_ARMS_COMPLETE")
    print("WROTE " + str(out) + " sha256=" + sha(out)[:16])


if __name__ == "__main__":
    main()
