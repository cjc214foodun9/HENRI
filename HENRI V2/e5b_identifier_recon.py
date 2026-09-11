"""Identifier reconciliation: is the E4a C2 marginal even comparable to mine?

THE QUESTION: my recompute of E4a's C2 rule gave p1=0.033 while the E4a receipt
says 0.117. Two hypotheses:
  (H1) IDENTIFIER DIFFERENCE -- a different tokenizer or corpus byte stream, in
       which case my numbers are NOT comparable to E4a's and the cross-check is
       meaningless rather than merely mis-specified.
  (H2) REGION DIFFERENCE -- same identifiers, different text.

Discipline: compare the PINNED identifiers first. E4a's receipt records its own
corpus and tokenizer sha256. Compare them with the ones MY carrier loaded.
Then recompute the marginal under E4a's exact pair rule AND under a couple of
plausible rule variants, to see which reproduces 0.117.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path

CORPUS = Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet")
TOKJ = Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json")
E4A = Path(r"C:\Users\chan\henri-telemetry\e3\e4a_construct_audit.json")
V2 = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_c2_v2.json")
OUT = Path(r"C:\Users\chan\henri-telemetry\e3\e5b_identifier_recon.json")

E4A_CORPUS = "e83889baabc497075506f91975be5fac0d45c5290b6b20582c8cd1e853d0c9f7"
E4A_TOK = "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539"


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> None:
    rec = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    e4a = json.loads(E4A.read_text())
    v2 = json.loads(V2.read_text())

    my_corpus = sha(CORPUS)
    my_tok = sha(TOKJ)
    rec["pins"] = {
        "e4a_receipt_corpus": e4a.get("corpus_sha256"),
        "e4a_receipt_tokenizer": e4a.get("tokenizer_sha256"),
        "my_corpus": my_corpus, "my_tokenizer": my_tok,
        "corpus_match": my_corpus == e4a.get("corpus_sha256"),
        "tokenizer_match": my_tok == e4a.get("tokenizer_sha256"),
        "v2_recorded_corpus": v2.get("corpus_sha256"),
        "v2_recorded_tokenizer": v2.get("tokenizer_sha256"),
        "v2_corpus_match": v2.get("corpus_sha256") == e4a.get("corpus_sha256"),
        "v2_tokenizer_match": v2.get("tokenizer_sha256") == e4a.get("tokenizer_sha256"),
    }
    P = rec["pins"]
    print("[pins] corpus_match=" + str(P["corpus_match"])
          + " tokenizer_match=" + str(P["tokenizer_match"]), flush=True)
    print("       e4a_tok=" + str(P["e4a_receipt_tokenizer"])[:16]
          + " my_tok=" + str(my_tok)[:16], flush=True)
    print("       e4a_corpus=" + str(P["e4a_receipt_corpus"])[:16]
          + " my_corpus=" + str(my_corpus)[:16], flush=True)

    # ---- recompute the E4a C2 marginal under several rule variants --------
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(TOKJ))
    enc = lambda s: tok.encode(s).ids
    dec = lambda ids: tok.decode(ids)

    rows = pq.read_table(str(CORPUS)).column("text").to_pylist()
    # variant A: tokens joined with NO separator (my v2 behaviour)
    stream_a: list[int] = []
    for r in rows:
        stream_a.extend(enc(r))
        if len(stream_a) >= 700_000:
            break
    # variant B: tokens joined with a newline between rows
    stream_b: list[int] = []
    for r in rows:
        stream_b.extend(enc(r))
        stream_b.extend(enc("\n"))
        if len(stream_b) >= 700_000:
            break

    def marg(stream, cal, ev):
        e_cal = [stream[t + 1] for t in range(*cal)]
        e_ev = [stream[t + 1] for t in range(*ev)]
        uni = Counter(e_cal)
        top = [t for t, _ in uni.most_common(5)]
        p1 = sum(1 for g in e_ev if g == top[0]) / len(e_ev)
        p5 = sum(1 for g in e_ev if g in top[:5]) / len(e_ev)
        return {"p1": round(p1, 6), "p5": round(p5, 6),
                "top1": int(top[0]), "top1_piece": dec([top[0]]),
                "distinct": len(uni), "n_eval": len(e_ev)}

    variants = {
        "A_nosep_e4a_rule":[0, 10000], "A_nosep_early": [0, 20000],
        "A_nosep_wide_eval": [0, 10000],
    }
    rules = {
        "A_e4a_pairs_0_10k__84k_85k": (stream_a, (0, 10_000), (84_000, 85_000)),
        "A_e4a_pairs_0_10k__44k_45k": (stream_a, (0, 10_000), (44_000, 45_000)),
        "B_nl_e4a_pairs_0_10k__84k_85k": (stream_b, (0, 10_000), (84_000, 85_000)),
        "A_v2_region_400k__600k": (stream_a, (400_000, 420_000), (600_000, 601_000)),
    }
    rec["recomputed"] = {}
    for name, (st, cal, ev) in rules.items():
        try:
            rec["recomputed"][name] = marg(st, cal, ev)
            m = rec["recomputed"][name]
            print("[" + name + "] p1=" + str(m["p1"]) + " p5=" + str(m["p5"])
                  + " top1=" + repr(m["top1_piece"]) + " distinct=" + str(m["distinct"]),
                  flush=True)
        except Exception as e:
            rec["recomputed"][name] = {"error": type(e).__name__ + ": " + str(e)[:120]}
            print("[" + name + "] ERROR " + str(e)[:120], flush=True)

    rec["e4a_receipt_c2"] = e4a["constructs"]["C2_token_stream"]
    rec["e4a_receipt_c2_split"] = {
        "calib": e4a["fresh_split"]["c2_pair_range_calib"],
        "eval": e4a["fresh_split"]["c2_pair_range_eval"],
    }
    best_name, best_delta = None, 9.0
    for name, m in rec["recomputed"].items():
        if "p1" in m:
            d = abs(m["p1"] - e4a["constructs"]["C2_token_stream"]["marginal_baseline"]["p1"])
            if d < best_delta:
                best_name, best_delta = name, d
    rec["reconciliation"] = {
        "closest_rule": best_name,
        "closest_abs_delta_p1": round(best_delta, 6),
        "reproduces_within_002": bool(best_delta <= 0.02),
        "identifier_aligned": bool(P["corpus_match"] and P["tokenizer_match"]),
        "verdict": (
            "IDENTIFIER_ALIGNED_MARGINAL_NOT_REPRODUCED" if (P["corpus_match"]
            and P["tokenizer_match"] and best_delta > 0.02)
            else ("IDENTIFIER_MISMATCH" if not (P["corpus_match"] and P["tokenizer_match"])
                  else "MARGINAL_REPRODUCED")),
    }
    R = rec["reconciliation"]
    print("\n[recon] identifiers_aligned=" + str(R["identifier_aligned"])
          + " closest=" + str(R["closest_rule"])
          + " delta=" + str(R["closest_abs_delta_p1"]))
    print("[recon] VERDICT=" + R["verdict"])
    OUT.write_text(json.dumps(rec, indent=2))
    print("WROTE " + str(OUT) + " sha256=" + sha(OUT)[:16])


if __name__ == "__main__":
    main()
