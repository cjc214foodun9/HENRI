"""Exact index-mapping test for the E4a C2 marginal, + final state.

E4a's generator (e4a_construct_audit.py:108-118):
    c2 = [((stream[i-3], stream[i-2], stream[i-1]), stream[i]) for i in range(3, len(stream))]
    c2_calib = c2[:FRESH_CALIB[1]-FRESH_CALIB[0]]            # c2[0:10000]
    c2_eval  = c2[4*FRESH_CALIB[1] : 4*FRESH_CALIB[1]+1000]  # c2[84000:85000]
So c2[j] has gold = stream[j+3].  My earlier recompute used gold = stream[t+1]
for t in the SAME index ranges -> an off-by-two mapping. Test every candidate
mapping here and report which (if any) reproduces the receipt's p1 = 0.117.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_c2_index_mapping.json")
LED = Path(r"C:\Users\chan\AppData\Local\hermes\audit\henri_audit_chain.jsonl")
E4A_P1, E4A_P5 = 0.117, 0.23


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(TOKJ))
    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    stream = []
    for r in rows:
        stream.extend(tok.encode(r).ids)
        if len(stream) >= 90000:
            break
    rec["stream_tokens"] = len(stream)

    def marg(c_lo, c_hi, e_lo, e_hi, shift):
        """gold(k) = stream[k + shift]; ranges are c2-pair indices."""
        cal = [stream[k + shift] for k in range(c_lo, c_hi)]
        ev = [stream[k + shift] for k in range(e_lo, e_hi)]
        uni = Counter(cal)
        top = [t for t, _ in uni.most_common(5)]
        p1 = sum(1 for g in ev if g == top[0]) / len(ev)
        p5 = sum(1 for g in ev if g in top[:5]) / len(ev)
        return {"p1": round(p1, 6), "p5": round(p5, 6), "distinct": len(uni),
                "top1": tok.decode([top[0]])}

    # candidate mappings: gold = stream[j + shift]
    cases = {
        "shift3_c2_index_EXACT_from_script": (0, 10000, 84000, 85000, 3),
        "shift1_my_earlier_recompute":       (0, 10000, 84000, 85000, 1),
        "shift3_alt_eval_44k":               (0, 10000, 44000, 45000, 3),
        "shift1_alt_eval_44k":               (0, 10000, 44000, 45000, 1),
        "shift3_calib_receipt_range_11k_21k": (11000, 21000, 84000, 85000, 3),
    }
    rec["cases"] = {}
    print("receipt: p1=" + str(E4A_P1) + " p5=" + str(E4A_P5))
    best, bd = None, 9.0
    for name, (cl, ch, el, eh, sh) in cases.items():
        try:
            m = marg(cl, ch, el, eh, sh)
            rec["cases"][name] = m
            d = abs(m["p1"] - E4A_P1)
            flag = "  <== CLOSEST" if d < bd else ""
            print("  " + name + " -> p1=" + str(m["p1"]) + " p5=" + str(m["p5"])
                  + " distinct=" + str(m["distinct"]) + " top1=" + repr(m["top1"])
                  + flag)
            if d < bd:
                best, bd = name, d
        except Exception as e:
            rec["cases"][name] = {"error": str(e)[:120]}
            print("  " + name + " -> ERROR " + str(e)[:120])

    rec["closest"] = {"case": best, "abs_delta_p1": round(bd, 6),
                      "reproduces_within_002": bool(bd <= 0.02)}
    print("\nCLOSEST=" + str(best) + " delta=" + str(round(bd, 6))
          + " reproduces_within_002=" + str(bd <= 0.02))
    rec["conclusion"] = (
        "EXPLAINED_INDEX_MAPPING" if bd <= 0.02 else
        "STILL_UNREPRODUCIBLE_under_every_tested_mapping")

    # ---- final chain state -------------------------------------------------
    rowsj = [json.loads(l) for l in LED.read_text(encoding="utf-8").splitlines() if l.strip()]
    prev, ok = "0" * 64, True
    for i, r in enumerate(rowsj):
        b = (f"{r['idx']}|{r['ts']}|{r['actor']}|{r['action']}|"
             f"{json.dumps(r['payload'], sort_keys=True)}|{r['prev_hash']}")
        if (r["idx"] != i or r["prev_hash"] != prev
                or hashlib.sha256(b.encode()).hexdigest() != r["hash"]):
            ok = False
            break
        prev = r["hash"]
    rec["chain"] = {"records": len(rowsj), "intact": ok, "head": rowsj[-1]["hash"][:16],
                    "e5b_c2_gates": sum(1 for r in rowsj
                                        if r["action"] == "HENRI_E5B_C2_GATES")}
    print("[chain] records=" + str(len(rowsj)) + " intact=" + str(ok)
          + " head=" + rec["chain"]["head"][:16])
    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
