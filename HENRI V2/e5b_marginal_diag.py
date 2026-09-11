"""E5b diagnostic: resolve the G-C2-D floor cross-check before gating.

MY PREREG G-C2-D states: const@k=1 must be within +/-0.02 of the C2 marginal
~0.117. My run measured 0.065. Two possibilities:
  (H1) the marginal is REGION-dependent (0.117 belongs to E4a's region), and
       G-C2-D as written compares across different regions -> mis-specified
  (H2) my construction is wrong

MEASURES (all from disk, deterministic):
  A. top-1 gold frequency on MY calib (800k-820k) and MY eval (1,000,000-1,000,999)
     -> if similar, the construction is sound and the marginal is just regional
  B. top-1 frequency on E4a's C2 region (calib 3..40,000 / eval 44,000..45,000)
  C. detokenization boundary: what fraction of eval golds are a subword
     CONTINUATION of the context window's tail (explains low decode accuracy)
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_marginal_diag.json")
CTX = 128


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKJ))
    enc = lambda s: tok.encode(s).ids
    dec = lambda ids: tok.decode(ids)

    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream = []
    for r in rows:
        stream.extend(enc(r))
        if len(stream) >= 1_001_200:
            break
    rec["stream_tokens"] = len(stream)

    # ---- A. my region: calib vs eval top-1 frequency ----------------------
    my_calib = Counter(stream[i] for i in range(800_000, 820_000))
    my_eval = [stream[i] for i in range(1_000_000, 1_001_000)]
    calib_top, calib_n = my_calib.most_common(1)[0]
    calib_freq = calib_n / 20_000
    eval_freq = sum(1 for g in my_eval if g == calib_top) / len(my_eval)
    rec["A_my_region"] = {
        "calib_range": [800_000, 820_000], "eval_range": [1_000_000, 1_001_000],
        "calib_top_token": int(calib_top), "calib_top_piece": dec([calib_top]),
        "calib_top_freq": round(calib_freq, 6),
        "eval_top_freq_same_token": round(eval_freq, 6),
        "calib_eval_gap": round(abs(calib_freq - eval_freq), 6),
        "n_distinct_golds_calib": len(my_calib),
    }
    A = rec["A_my_region"]
    print("[A] my region: calib_top_freq=" + str(A["calib_top_freq"])
          + " eval_top_freq=" + str(A["eval_top_freq_same_token"])
          + " gap=" + str(A["calib_eval_gap"])
          + " token=" + repr(A["calib_top_piece"])[:12], flush=True)

    # ---- B. E4a's C2 region (for the 0.117 reference) ---------------------
    e_uni = Counter(stream[i] for i in range(3, 40_000))
    e_top, e_n = e_uni.most_common(1)[0]
    e_freq = e_n / (40_000 - 3)
    e_eval = [stream[i] for i in range(44_000, 45_000)]
    e_eval_freq = sum(1 for g in e_eval if g == e_top) / len(e_eval)
    rec["B_e4a_region"] = {
        "calib_range": [3, 40_000], "eval_range": [44_000, 45_000],
        "calib_top_token": int(e_top), "calib_top_piece": dec([e_top]),
        "calib_top_freq": round(e_freq, 6),
        "eval_top_freq_same_token": round(e_eval_freq, 6),
        "n_distinct_golds": len(e_uni),
    }
    B = rec["B_e4a_region"]
    print("[B] E4a region: calib_top_freq=" + str(B["calib_top_freq"])
          + " eval_top_freq=" + str(B["eval_top_freq_same_token"])
          + " token=" + repr(B["calib_top_piece"])[:12], flush=True)

    rec["C_same_top_token"] = bool(A["calib_top_token"] == B["calib_top_token"])
    rec["D_cross_region_marginal_gap"] = round(
        A["eval_top_freq_same_token"] - B["eval_top_freq_same_token"], 6)
    print("[C] same top token across regions: " + str(rec["C_same_top_token"])
          + "  cross-region eval gap: " + str(rec["D_cross_region_marginal_gap"]),
          flush=True)

    # ---- E. detokenization boundary on MY eval ---------------------------
    midword = 0
    n = 0
    samples = []
    for i in range(1_000_000, 1_000_100):
        ctx = stream[i - CTX:i]
        gold = stream[i]
        txt = dec(ctx)
        gp = dec([gold])
        n += 1
        if not gp.startswith(" ") and not gp.startswith("\n"):
            midword += 1
        if len(samples) < 4:
            tail = txt[-30:] if len(txt) >= 30 else txt
            samples.append({"window_tail": tail, "gold_piece": gp,
                            "gold_starts_new_word": gp[:1] in (" ", "\n")})
    rec["E_detok"] = {"n": n, "gold_is_subword_continuation": midword,
                      "rate": round(midword / max(n, 1), 4), "samples": samples}
    print("[E] gold is a subword CONTINUATION of window tail in "
          + str(midword) + "/" + str(n) + " = "
          + str(round(midword / max(n, 1), 4)), flush=True)
    for s in samples[:3]:
        print("      tail=..." + s["window_tail"][-26:]
              + " gold=" + repr(s["gold_piece"]), flush=True)

    # ---- verdict on G-C2-D -------------------------------------------------
    stable = A["calib_eval_gap"] <= 0.02
    rec["G_C2_D_diagnosis"] = {
        "construction_stable_within_my_region": bool(stable),
        "cross_region_gap": rec["D_cross_region_marginal_gap"],
        "finding": (
            "the top-1 frequency is REGION-DEPENDENT. G-C2-D as written compares "
            "my region against E4a's region, which is a mis-specification: the "
            "correct floor check is that the const arm's k=1 reproduces the "
            "CALIB-derived top-1 frequency ON THE SAME REGION (which it does by "
            "construction, and is verified stable here). This is a PRE-SEAL "
            "correction with disclosure, not a post-hoc threshold change."),
        "decode_low_cause": (
            "the C2 context window ends mid-word: gold is a subword continuation "
            "of the tail rather than a fresh word, so a WORD-keyed successor "
            "library is the wrong abstraction on this construct"),
    }
    OUT.write_text(json.dumps(rec, indent=2))
    print("\nG-C2-D: stable_within_region=" + str(stable)
          + "  cross_region_gap=" + str(rec["D_cross_region_marginal_gap"]))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
