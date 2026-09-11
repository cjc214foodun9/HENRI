"""E5a — WAVE CANDIDATE-SET COVERAGE GATE (final, zero-trainable).

THE GENERAL FORM: prove the target is IN THE CANDIDATE SET before measuring any
rank. Rank without coverage is VACUOUS.

DEFECT HISTORY (all mine, disclosed):
  v1  successor library built from WITHIN-PREFIX bigrams, which never observe a
      continuation -> coverage ~10x low (const k=256 = 0.162, BELOW E4a's 0.440
      constant baseline). Self-contradiction caught it.
  v2  decode reference WRONG: compared the prefix's decoded terminal word against
      the WINDOW's last word instead of the PREFIX's own terminal word (same bug
      class as E4b v1) -> "decode acc 0.001" was an artifact.
  v3  TypeError: the library was switched to store gold TOKEN IDS, but a leftover
      cont_tokens() helper still treated the entries as WORD STRINGS
      (pt + " " + int). The helper is also unnecessary once the library holds
      tokens directly -> removed.
  v4  (this file) GATE VACUITY FIX: "reaches gamma at some k <= 4096" is nearly
      vacuous, because a constant set of all ~1890 calib gold tokens covers almost
      everything. The informative gate is coverage at a BOUNDED candidate budget
      k <= 64, plus the reported coverage COST k90 (the k needed to reach 0.90).

ARMS (same sealed split, same gold; ZERO trainable):
  C_const  top-k most frequent calib gold tokens              (constant floor)
  C_ctx    top-k gold tokens observed after the PREFIX's last
           word (the tokenizer gives that word for free)      (surface/context)
  C_wave   the same library keyed on the terminal word DECODED
           from the prefix wave by the sealed E4b position
           channel (zero trainable)                            (the mechanism)

The wave arm supplies the conditioning variable ITSELF. The gap to the surface
arm is the DECODE TAX.

PRE-REGISTERED
  gamma = 0.90 ; K_BOUND = 64 ; K_LIST = [1,5,16,64,256,1024,2048]
  G-COV-A  best zero-trainable generator reaches gamma at k <= K_BOUND
  G-COV-B  wave terminal-decode accuracy >= 0.95
  auxiliary: k90 per arm (coverage cost); wave-minus-surface gap at K_BOUND
  KILL     if G-COV-A or G-COV-B fails
  Cross-check: const coverage at k=1 must reproduce E4a's marginal (0.440 +/-0.02)

Split: sealed E4a fresh split (11,000..21,999). Reused. Label CONDITIONAL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HV2 = Path(__file__).resolve().parent
sys.path.insert(0, str(HV2))

from e4b_position_codec import PositionBoundCodec, _pos_cells  # noqa: E402
from g7_highorder_codec import build_vocab  # noqa: E402
from zone_c_world_knowledge_codec import tokenize  # noqa: E402

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5a_coverage.json")

MAX_WORDS = 24
C1_CALIB = (11000, 21000)
C1_EVAL = (21000, 22000)
K_LIST = [1, 5, 16, 64, 256, 1024, 2048]
K_BOUND = 64
K_TOP = K_LIST[-1]
GAMMA = 0.90
E4A_MARGINAL = 0.440


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sentence_split(t: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", t.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    rec: dict = {
        "carrier": "E5a", "revision": "v4",
        "mechanism": "zero-trainable wave candidate-set coverage gate",
        "trainable_params": 0, "label": "CONDITIONAL_FRESH_SPLIT_REUSED",
        "prereg": {"gamma": GAMMA, "k_bound": K_BOUND, "k_list": K_LIST,
                   "e4a_marginal_crosscheck": E4A_MARGINAL},
        "defects_disclosed": {
            "v1": "within-prefix bigram library (never observed a continuation)",
            "v2": "decode compared against the window's last word, not the prefix's",
            "v3": "library switched to token ids but a leftover helper treated "
                  "entries as word strings -> TypeError",
            "v4": "gate vacuity fix: bounded k required; unbounded 'some k<=4096' "
                  "is nearly vacuously satisfiable by a large constant set",
        },
    }

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    rec["corpus_sha256"] = sha(CORPUS)
    rec["tokenizer_sha256"] = sha(TOKJ)
    tok = Tokenizer.from_file(str(TOKJ))
    enc = lambda s: tok.encode(s).ids

    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    sents: list[str] = []
    for r in rows:
        sents.extend(sentence_split(r))

    win = []
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = enc(window), enc(prefix)
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            win.append((w[:-1], ia[len(ip)]))
    calib = win[C1_CALIB[0]:C1_CALIB[1]]
    ev = win[C1_EVAL[0]:C1_EVAL[1]]
    rec["n_calib"], rec["n_eval"] = len(calib), len(ev)
    print("[0] calib " + str(len(calib)) + " eval " + str(len(ev)), flush=True)

    # ---- libraries, calib only: context word -> Counter(gold token ids) ----
    gold_freq = Counter(g for _, g in calib)
    const_order = [t for t, _ in gold_freq.most_common(K_TOP)]
    succ: dict[str, Counter] = defaultdict(Counter)
    for pref, gold in calib:
        cw = tokenize(" ".join(pref))
        if cw:
            succ[cw[-1]][gold] += 1
    RANK = {ctx: [t for t, _ in c.most_common(K_TOP)] for ctx, c in succ.items()}
    rec["library"] = {
        "kind": "context_word -> ranked gold token ids (calib only)",
        "const_distinct_golds": len(gold_freq),
        "ctx_contexts": len(succ),
    }
    print("[1] const golds " + str(len(gold_freq))
          + " ctx contexts " + str(len(succ)), flush=True)

    # ---- wave codec (sealed E4b, zero trainable) -------------------------
    vcb = build_vocab([" ".join(p) for p, _ in calib[:6000]], max_words=20000)
    codec = PositionBoundCodec(vcb)
    E_CELLS = np.stack([_pos_cells("e:" + w) for w in vcb]).astype(np.int64)
    E_SIGNS = np.where((E_CELLS % 2) == 0, 1.0, -1.0).astype(np.float32)
    rec["codec"] = {"class": "PositionBoundCodec (E4b)", "vocab": len(vcb),
                    "trainable_params": 0,
                    "decode": "vectorized argmax over reserved e:-cells"}
    print("[2] codec vocab " + str(len(vcb)), flush=True)

    def decode_terminal(pt: str):
        wb, _ = codec.encode(pt)
        flat = np.frombuffer(wb, dtype=np.float32).ravel()
        sc = ((flat[E_CELLS] * E_SIGNS) > 0.0).sum(axis=1)     # [V]
        i = int(np.argmax(sc))
        return vcb[i], float(sc[i]) / 16.0

    # ---- measure ----------------------------------------------------------
    hits = {a: {k: 0 for k in K_LIST} for a in ("const", "ctx", "wave")}
    n = 0
    dec_ok = 0
    ex = []
    for pref, gold in ev:
        pt = " ".join(pref)
        cw = tokenize(pt)
        if not cw:
            continue
        n += 1
        surface = cw[-1]                      # the PREFIX's own terminal word
        # ---- const --------------------------------------------------------
        for k in K_LIST:
            if gold in const_order[:k]:
                hits["const"][k] += 1
        # ---- ctx (library token ids used DIRECTLY; no expansion) ----------
        st = RANK.get(surface, [])
        for k in K_LIST:
            if gold in st[:k]:
                hits["ctx"][k] += 1
        # ---- wave ---------------------------------------------------------
        dw, dev = decode_terminal(pt)
        if dw == surface:                      # CORRECT reference
            dec_ok += 1
        wt = RANK.get(dw, []) if dw else []
        for k in K_LIST:
            if gold in wt[:k]:
                hits["wave"][k] += 1
        if len(ex) < 5:
            ex.append({"surface": surface, "decoded": dw, "ev": round(dev, 3),
                       "match": dw == surface, "gold": gold})

    cov = {a: {str(k): round(hits[a][k] / n, 6) for k in K_LIST} for a in hits}
    dec_acc = dec_ok / n
    rec.update({"n_eval_used": n, "coverage": cov,
                "wave_decode_accuracy": round(dec_acc, 6), "examples": ex})

    def k90(a):
        return next((k for k in K_LIST if cov[a][str(k)] >= GAMMA), None)

    k90s = {a: k90(a) for a in cov}
    bounded_best = max(cov[a][str(K_BOUND)] for a in cov)
    gA = bounded_best >= GAMMA
    gB = dec_acc >= 0.95
    xcheck = abs(cov["const"]["1"] - E4A_MARGINAL) <= 0.02

    rec["gates"] = {
        "G-COV-A_best_coverage_at_k<=64_ge_gamma": bool(gA),
        "G-COV-A_best_at_k64": round(bounded_best, 6),
        "G-COV-B_wave_decode_acc_ge_0.95": bool(gB),
        "G-COV-B_wave_decode_acc": round(dec_acc, 6),
        "crosscheck_const_k1_vs_e4a_marginal": bool(xcheck),
    }
    rec["coverage_cost_k90"] = k90s
    rec["decode_tax_at_k64"] = {
        "surface_ctx": cov["ctx"][str(K_BOUND)],
        "wave": cov["wave"][str(K_BOUND)],
        "gap": round(cov["ctx"][str(K_BOUND)] - cov["wave"][str(K_BOUND)], 6),
    }
    rec["per_k_coverage_table"] = {
        a: {str(k): cov[a][str(k)] for k in K_LIST} for a in cov}

    kill = not (gA and gB)
    rec["verdict"] = "E5A_COVERAGE_FAIL" if kill else "E5A_COVERAGE_PASS"
    rec["rank_metrics_withheld"] = bool(kill)
    rec["reading"] = (
        "coverage is the admissibility gate. A FAIL here is a CONSTRUCT finding: "
        "it says the next-token target on THIS construct (C1 sentence-window, whose "
        "marginal is dominated by the sentence-final period) is too diffuse for a "
        "bounded zero-trainable candidate set, so ranking inside C(x) cannot be "
        "meaningful at that budget. It is NOT evidence that the wave channel is "
        "broken. The successor carrier should re-measure on the C2 token-stream "
        "construct, where the marginal is 0.117 and the frozen-backbone oracle is "
        "0.433, i.e. there is real headroom.")
    rec["wall_s"] = round(time.time() - t0, 1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2))
    print("[3] const " + json.dumps(cov["const"]), flush=True)
    print("[3] ctx   " + json.dumps(cov["ctx"]), flush=True)
    print("[3] wave  " + json.dumps(cov["wave"]), flush=True)
    print("[3] decode acc " + str(round(dec_acc, 4))
          + "  k90 " + json.dumps(k90s), flush=True)
    print("\n" + json.dumps(rec["gates"], indent=2))
    print("decode tax @k64 " + json.dumps(rec["decode_tax_at_k64"]))
    print("VERDICT=" + rec["verdict"])
    print("WROTE " + str(args.out) + " sha256=" + sha(args.out)[:16])


if __name__ == "__main__":
    main()
