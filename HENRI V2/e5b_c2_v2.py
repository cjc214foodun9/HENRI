"""E5b v2 — C2 token-stream coverage: E4a reconciliation + corrected construct.

TWO THINGS, both required before any C2 gate can be evaluated.

(1) RECONCILE E4a's authoritative C2 marginal.
    The summary quotes marginal 0.117 / backbone oracle 0.433. The authoritative
    receipt is henri-telemetry/e3/e4a_construct_audit.json (sha 0a1b86fe42d83136):
        C2_token_stream.marginal_baseline = {p1: 0.117, p5: 0.23}
        fresh_split.c2_pair_range_calib = [0, 10000]
        fresh_split.c2_pair_range_eval  = [84000, 85000]   (on PAIRS (t, t+1))
    This script RECOMPUTES that marginal from the raw stream under that exact
    rule. It does not assume the quoted number is right.

(2) RE-MEASURE coverage with the CORRECT conditioning variable.
    v1 keyed the successor library on the context's last WHITESPACE WORD.
    Measured defect (e5b_marginal_diag.json sha 569573b3): the 128-token window
    ends MID-WORD, so 19% of golds are subword CONTINUATIONS of the tail; a
    word-keyed library cannot represent them (decode accuracy fell 0.839->0.363).
    On a TOKEN STREAM the natural conditioning variable is the last T TOKENS.
    v2 keys on the token n-gram with pure backoff (2-gram -> 1-gram -> const-free)
    and reports the word-keyed arm alongside for comparison.

SPLIT (fresh, disclosed):
    calib tokens [400_000, 420_000)   eval tokens [600_000, 601_000)
    CONSUMED (do not reuse): [1_000_000, 1_001_000) -- evaluated by v1.

CONTROLS: a deterministic RANDOM candidate set, which must score ~k/|gold universe|.
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

from e4b_position_codec import PositionBoundCodec, _pos_cells  # noqa: E402
from g7_highorder_codec import build_vocab  # noqa: E402

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
E4A_RECEIPT = Path(r"C:\Users\chan\henri-telemetry\e3\e4a_construct_audit.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_c2_v2.json")
CTX_EXPORT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_c2_v2_contexts.pt")

CTX = 128
K_LIST = [1, 5, 16, 64, 256, 1024]
GAMMA = 0.90
CALIB = (400_000, 420_000)
EVAL = (600_000, 601_000)
E4A_CALIB_PAIRS = (0, 10_000)
E4A_EVAL_PAIRS = (84_000, 85_000)
E4A_P1, E4A_P5 = 0.117, 0.23
STREAM_NEED = EVAL[1] + 2


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--export", type=Path, default=CTX_EXPORT)
    args = ap.parse_args()
    t0 = time.time()
    rec: dict = {"carrier": "E5b", "version": "v2", "construct": "C2_token_stream",
                 "trainable_params": 0, "ctx": CTX,
                 "fresh_split": {"calib_tokens": list(CALIB), "eval_tokens": list(EVAL)},
                 "consumed_split_disclosed": {"region": [1_000_000, 1_001_000],
                                              "by": "v1 (e5b_coverage_c2.json sha 41baffac)"}}

    import pyarrow.parquet as pq
    import torch
    from tokenizers import Tokenizer

    rec["corpus_sha256"] = sha(CORPUS)
    rec["tokenizer_sha256"] = sha(TOKJ)
    rec["e4a_receipt_sha256"] = sha(E4A_RECEIPT)
    tok = Tokenizer.from_file(str(TOKJ))
    enc = lambda s: tok.encode(s).ids
    dec = lambda ids: tok.decode(ids)

    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream: list[int] = []
    for r in rows:
        stream.extend(enc(r))
        if len(stream) >= STREAM_NEED:
            break
    rec["stream_tokens"] = len(stream)
    print("[0] stream " + str(len(stream)) + " need " + str(STREAM_NEED), flush=True)
    if len(stream) < STREAM_NEED:
        rec["VERDICT"] = "BLOCKED_INFRA"
        rec["reason"] = "STREAM_TOO_SHORT"
        args.out.write_text(json.dumps(rec, indent=2))
        print("BLOCKED_INFRA STREAM_TOO_SHORT")
        return

    # ================= (1) E4a RECONCILIATION =============================
    e_cal = [stream[t + 1] for t in range(*E4A_CALIB_PAIRS)]
    e_ev = [stream[t + 1] for t in range(*E4A_EVAL_PAIRS)]
    e_uni = Counter(e_cal)
    e_top = [t for t, _ in e_uni.most_common(5)]
    p1 = sum(1 for g in e_ev if g == e_top[0]) / len(e_ev)
    p5 = sum(1 for g in e_ev if g in e_top[:5]) / len(e_ev)
    rec["e4a_reconciliation"] = {
        "rule": "calib golds=stream[t+1] for t in [0,10000); eval golds=stream[t+1] for t in [84000,85000)",
        "n_calib": len(e_cal), "n_eval": len(e_ev),
        "top5": [(int(t), int(c), dec([t])) for t, c in e_uni.most_common(5)],
        "distinct_golds": len(e_uni),
        "recomputed_p1": round(p1, 6), "recomputed_p5": round(p5, 6),
        "receipt_p1": E4A_P1, "receipt_p5": E4A_P5,
        "p1_delta": round(p1 - E4A_P1, 6), "p5_delta": round(p5 - E4A_P5, 6),
        "reproduces_receipt_within_002": bool(abs(p1 - E4A_P1) <= 0.02),
    }
    R = rec["e4a_reconciliation"]
    print("[1] E4a reconcile: recomputed p1=" + str(R["recomputed_p1"])
          + " p5=" + str(R["recomputed_p5"]) + " | receipt p1=" + str(E4A_P1)
          + " p5=" + str(E4A_P5) + " | delta=" + str(R["p1_delta"]), flush=True)

    # ================= (2) CORRECTED COVERAGE =============================
    gold_freq = Counter(stream[i] for i in range(*CALIB))
    const_order = [t for t, _ in gold_freq.most_common(K_LIST[-1])]
    lib1: dict[int, Counter] = defaultdict(Counter)
    lib2: dict[tuple, Counter] = defaultdict(Counter)
    for i in range(CALIB[0] + 4, CALIB[1]):
        lib1[stream[i - 1]][stream[i]] += 1
        lib2[(stream[i - 2], stream[i - 1])][stream[i]] += 1
    R1 = {k: [t for t, _ in c.most_common(K_LIST[-1])] for k, c in lib1.items()}
    R2 = {k: [t for t, _ in c.most_common(K_LIST[-1])] for k, c in lib2.items()}
    rec["library"] = {"kind": "token n-gram -> ranked observed GOLD TOKEN ids (calib only)",
                      "const_distinct_golds": len(gold_freq),
                      "n_1gram_keys": len(R1), "n_2gram_keys": len(R2)}
    print("[2] lib: const_golds " + str(len(gold_freq)) + " 1g " + str(len(R1))
          + " 2g " + str(len(R2)), flush=True)

    # wave codec (sealed E4b; zero trainable)
    vcb = build_vocab([dec(stream[i - CTX:i]) for i in range(
        CALIB[0] + CTX, min(CALIB[0] + CTX + 6000, CALIB[1]))], max_words=20000)
    codec = PositionBoundCodec(vcb)
    E_CELLS = np.stack([_pos_cells("e:" + w) for w in vcb]).astype(np.int64)
    E_SIGNS = np.where((E_CELLS % 2) == 0, 1.0, -1.0).astype(np.float32)
    rec["codec"] = {"class": "PositionBoundCodec (E4b)", "vocab": len(vcb),
                    "trainable_params": 0}
    print("[3] codec vocab " + str(len(vcb)), flush=True)

    def decode_terminal(txt: str):
        wb, _ = codec.encode(txt)
        flat = np.frombuffer(wb, dtype=np.float32).ravel()
        sc = ((flat[E_CELLS] * E_SIGNS) > 0.0).sum(axis=1)
        i = int(np.argmax(sc))
        return vcb[i], float(sc[i]) / 16.0

    rng = random.Random(20260911)
    gold_universe = list(gold_freq)
    arms = ("const", "tok1", "tok2", "wave", "rand")
    hits = {a: {k: 0 for k in K_LIST} for a in arms}
    n = 0
    dec_ok = 0
    ctx_t, gold_t, ex = [], [], []
    for i in range(*EVAL):
        ctx = stream[i - CTX:i]
        gold = stream[i]
        txt = dec(ctx)
        words = txt.split()
        n += 1
        # const
        for k in K_LIST:
            if gold in const_order[:k]:
                hits["const"][k] += 1
        # tok1 (pure backoff)
        s1 = R1.get(stream[i - 1], const_order)
        for k in K_LIST:
            if gold in s1[:k]:
                hits["tok1"][k] += 1
        # tok2 (pure backoff to tok1 to const)
        s2 = R2.get((stream[i - 2], stream[i - 1]), s1)
        for k in K_LIST:
            if gold in s2[:k]:
                hits["tok2"][k] += 1
        # wave (word-keyed, comparability with v1)
        dw, dv = decode_terminal(txt)
        if words and dw == words[-1]:
            dec_ok += 1
        sw = R1.get(next((t for t in gold_freq if dec([t]) == dw), -1), const_order)
        for k in K_LIST:
            if gold in sw[:k]:
                hits["wave"][k] += 1
        # rand control
        for k in K_LIST:
            if gold in rng.sample(gold_universe, min(k, len(gold_universe))):
                hits["rand"][k] += 1
        ctx_t.append(ctx)
        gold_t.append(gold)
        if len(ex) < 4:
            ex.append({"decoded": dw, "ev": round(dv, 3),
                       "match": bool(words and dw == words[-1]), "gold_piece": dec([gold])})

    cov = {a: {str(k): round(hits[a][k] / n, 6) for k in K_LIST} for a in arms}
    rec.update({"n_eval": n, "coverage": cov,
                "wave_decode_accuracy": round(dec_ok / n, 6),
                "rand_control_expected": round(1.0 / len(gold_universe), 8),
                "examples": ex})
    for a in arms:
        print("[4] " + a.ljust(6) + " " + json.dumps(cov[a]), flush=True)
    print("[4] decode acc " + str(round(dec_ok / n, 4))
          + " | rand len " + str(len(gold_universe)), flush=True)

    torch.save({"contexts": torch.tensor(ctx_t, dtype=torch.long),
                "golds": torch.tensor(gold_t, dtype=torch.long),
                "meta": {"ctx": CTX, "calib": list(CALIB), "eval": list(EVAL),
                         "corpus_sha256": rec["corpus_sha256"],
                         "tokenizer_sha256": rec["tokenizer_sha256"]}},
               str(args.export))
    rec["contexts_export"] = {"path": str(args.export), "sha256": sha(args.export),
                              "n": len(ctx_t)}
    print("[5] exported " + str(len(ctx_t)) + " -> " + str(args.export), flush=True)

    best = max(("tok2", "tok1", "const", "wave"), key=lambda a: cov[a]["1024"])
    rec["G_C2_A_wave_visible"] = {
        "best_zerotrainable_arm_at_k1024": best,
        "best_at_k1024": cov[best]["1024"],
        "wave_at_k1024": cov["wave"]["1024"],
        "note": "gate finalised only after the backbone oracle arrives",
    }
    rec["G_C2_D_status"] = {
        "finding": ("the reference marginal 0.117 is REGION/SPLIT-specific (E4a "
                    "pairs [0,10000]->[84000,85000]). A cross-region comparison is "
                    "mis-specified. The correct floor check is that the const arm's "
                    "k=1 reproduces the CALIB top-1 frequency ON THE SAME REGION. "
                    "Pre-seal prereg correction, disclosed with both doc hashes."),
        "e4a_reconciles": R["reproduces_receipt_within_002"],
    }
    rec["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(args.out) + " sha256=" + sha(args.out)[:16])
    print("NOTE: G-C2-A/C finalise only after the backbone oracle arm returns.")


if __name__ == "__main__":
    main()
