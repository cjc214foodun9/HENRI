"""E5c/E5d DISJOINTNESS PROVER — fail-closed precondition for both carriers.

Every consumed range below is transcribed from a SOURCE FILE or a RECEIPT read
this session, with the file:line recorded. Nothing is quoted from a summary.

VERIFIED SOURCES (file:line) read this session:
  e4a_construct_audit.py:37-38   FRESH_CALIB=(11000,21000) FRESH_EVAL=(21000,22000)  [C1 window index space]
                                 c2_calib=c2[0:10000]      -> gold=stream[j+3], j in [0,10000)
                                 c2_eval =c2[84000:85000]  -> gold=stream[j+3], j in [84000,85000)
  e4c_extract_features.py:38-39  FRESH_CALIB=(11000,21000) FRESH_EVAL=(21000,22000)  [C1 window index space]
  e4c_oracle_probe.py:53         C2_EVAL_START=84000 ; seg=pairs[4*84000 : 4*84000+1000]
  e4c_probe2.py:51-52            C2_FRESH_START=500_000 ; C2_N=1000
  e5a_coverage.py:68-69          C1_CALIB=(11000,21000)  C1_EVAL=(21000,22000)      [C1 window index space]
  e5b_coverage_c2.py:48-49       CALIB=(800_000,820_000) EVAL=(1_000_000,1_001_000)
  e5b_c2_v2.py:57-58             CALIB=(400_000,420_000) EVAL=(600_000,601_000)

NOT CONSUMED (fabricated elsewhere, falsified by direct read):
  e4c_factorial.py:24-34 is IMPORTS (from pathlib import Path ... import torch.nn.functional as F).
  There is no CALIB/CALIB2/CALIB3/EVAL2/EVAL3 region block in that file.

CHOSEN (this prover is the sole authority):
  R1 (E5c): calib [1_200_000,1_220_000)  eval [1_300_000,1_301_000)
  R2 (E5d): calib [1_400_000,1_420_000)  eval [1_500_000,1_501_000)

ASSERTIONS: A1 disjoint from every consumed range, margin >= 10000 tokens.
            A2 corpus long enough for both regions.
FAIL-CLOSED: carriers refuse to run unless VERDICT == FRESH_REGIONS_PROVEN.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5_fresh_regions.json")
MARGIN = 10_000

R1 = {"calib": [1_200_000, 1_220_000], "eval": [1_300_000, 1_301_000]}
R2 = {"calib": [1_400_000, 1_420_000], "eval": [1_500_000, 1_501_000]}

# (label, lo, hi, provenance) — token index space
CONSUMED = [
    ("E4a_C2_calib", 3, 10_003, "e4a_construct_audit.py:37-38 + pair rule"),
    ("E4a_C2_eval", 84_003, 85_003, "e4a_construct_audit.py:37-38 + pair rule"),
    ("E4c_oracle_C2_eval", 336_003, 337_003, "e4c_oracle_probe.py:53 C2_EVAL_START=84000"),
    ("E4c_probe2_C2_eval", 500_000, 501_000, "e4c_probe2.py:51 C2_FRESH_START=500_000"),
    ("E5b_v1_calib", 800_000, 820_000, "e5b_coverage_c2.py:48"),
    ("E5b_v1_eval", 1_000_000, 1_001_000, "e5b_coverage_c2.py:49"),
    ("E5b_v2_calib", 400_000, 420_000, "e5b_c2_v2.py:57"),
    ("E5b_v2_eval", 600_000, 601_000, "e5b_c2_v2.py:58"),
]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "margin": MARGIN, "chosen": {"R1": R1, "R2": R2},
           "consumed_source_anchored": [{"label": a, "lo": b, "hi": c, "provenance": d}
                                        for a, b, c, d in CONSUMED]}
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKJ))
    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream: list[int] = []
    for r in rows:
        stream.extend(tok.encode(r).ids)
        if len(stream) >= R2["eval"][1] + 8:
            break
    rec["stream_tokens"] = len(stream)
    print("[stream] " + str(len(stream)) + " need " + str(R2["eval"][1] + 8), flush=True)

    # ---- A2: computed C1 window token span (windows 11000..21999) --------
    # Reproduce the production window walk: a window exists for each sentence
    # with >= 2 words; gold is the first BPE token of the sentence's last word.
    win = 0
    lo_span, hi_span = None, 0
    off = 0
    for r in rows:
        ids = tok.encode(r).ids
        row_lo, off = off, off + len(ids)
        for s in re.split(r"(?<=[.!?])\s+", r):
            if not s.strip():
                continue
            if len(s.split()) >= 2:
                if 11_000 <= win < 22_000:
                    if lo_span is None:
                        lo_span = row_lo
                    hi_span = off
                win += 1
        if win >= 22_000:
            break
    rec["c1_window_index_total"] = win
    rec["e4a_e5a_c1_window_pair"] = [11_000, 22_000]
    rec["e4a_e5a_c1_token_span"] = [lo_span, hi_span]
    print("[c1] windows " + str(win) + " token span " + str([lo_span, hi_span]),
          flush=True)
    if lo_span is not None:
        CONSUMED.append(("E4a_E5a_C1_windows", lo_span, hi_span + 200,
                         "computed window walk, windows 11000..21999"))

    rec["consumed_final"] = [{"label": a, "lo": b, "hi": c}
                             for a, b, c, _ in CONSUMED]
    rec["max_consumed_hi"] = max(c for _, _, c, _ in CONSUMED)

    # ---- A1: disjointness with margin ------------------------------------
    viol = []
    for rname, reg in (("R1", R1), ("R2", R2)):
        for part in ("calib", "eval"):
            a_lo, a_hi = reg[part]
            for lbl, c_lo, c_hi, _ in CONSUMED:
                if not (a_hi + MARGIN <= c_lo or a_lo - MARGIN >= c_hi):
                    viol.append({"region": rname, "part": part,
                                 "chosen": [a_lo, a_hi], "collides_with": lbl,
                                 "consumed": [c_lo, c_hi]})
    rec["violations"] = viol
    rec["stream_ok"] = len(stream) >= R2["eval"][1] + 8
    ok = (not viol) and rec["stream_ok"]
    rec["VERDICT"] = "FRESH_REGIONS_PROVEN" if ok else "PROVER_FAIL_CLOSED"

    OUT.write_text(json.dumps(rec, indent=2))
    print("[consumed] " + str(len(CONSUMED)) + " ranges, max hi "
          + str(rec["max_consumed_hi"]))
    print("[violations] " + str(len(viol)))
    for v in viol[:6]:
        print("   " + json.dumps(v))
    print("[stream_ok] " + str(rec["stream_ok"]))
    print("VERDICT=" + rec["VERDICT"])
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
