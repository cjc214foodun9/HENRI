"""A-K4 EGRESS GATE. Does retrieved context change a generated answer for the better?

CONTEXT (why this test, why now)
    SpecContract A was APPROVED by the operator (audit event #10b2d1f45cd8626c,
    chain 1816 -> 1817). A-K4 is the binding kill: retrieved Zone C context must
    improve a GENERATED answer, not merely occupy a store.

    Prior state: `top1_token_unique = 1` across 16 distinct waves on the ARM_U
    unbinder path (down_proj [2048,65536] -> lm_head [32000,2048]).

WHAT THE M1 GATE CHANGED (run 2026-10-03, pre-registered, first ever execution)
    m1_open_answer_gate.py: verdict VACUOUS_DISTINCT_COUNT_NOT_INFORMATIVE for
    BOTH position_binding arms. The RANDOM-wave arm scored distinct_ratio 0.59 /
    0.70 while the treatment scored 0.23 / 0.12. So "count of distinct top-1
    tokens" is NOT evidence of semantic content -- the exact `argmax is
    beta-invariant` lesson. THIS SCRIPT THEREFORE DOES NOT SCORE ON ARGMAX
    UNIQUENESS. It scores on required-content presence in generated text.

DESIGN (frozen before the run; no post-hoc edit of thresholds)
    Arms, per question:
      A0  NO evidence                    -> should ABSTAIN or miss
      A1  CORRECT retrieved evidence     -> should contain the required content
      A2  MISMATCHED evidence (other fact) -> must NOT reproduce the A1 gain
    Positive control P0: A1 answers must be non-empty and not all "ABSTAIN".

    Pre-registered kills:
      K4a  hit(A1) - hit(A0) >= +0.50
      K4b  hit(A1) - hit(A2) >= +0.25
      K4c  hit(A0) <= 0.25 and hit(A2) <= 0.25   (negative controls stay low)
      K4d  determinism: same prompt twice -> identical answer for 100%
      VACUITY: if retrieval returns zero chunks for any A1 item -> NO VERDICT.

    Scoring is CONTENT-based on the answer string (case-insensitive), NOT argmax.

HONESTY BOUNDARY
    A pass establishes that retrieved context improves GENERATED answers on a
    6-fact synthetic corpus. That is mechanism evidence, not end-task capability.
    Small N (12 items) is declared. CPU only. $0. Latency is NOT measured.
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

_HENRI_V2 = str(Path(__file__).resolve().parents[2])
sys.path.insert(0, _HENRI_V2)

TMP = os.environ.get("TEMP") or "/tmp"
CORPUS = r"C:/Users/chan/AppData/Local/Temp/ak4_corpus"
DSN_ENV = os.path.join(CORPUS, ".dsn.env")
MODEL_DIR = os.environ.get(
    "HENRI_BACKBONE_MODEL_DIR",
    r"C:/Users/chan/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/"
    r"snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306")
MAX_NEW = int(os.environ.get("AK4_MAX_NEW", "16"))
_AK4_LIMIT = int(os.environ.get("AK4_LIMIT", "0"))  # 0 = all; smoke uses small N

# ---------------------------------------------------------------- pre-registration
K4A_MIN = 0.50
K4B_MIN = 0.25
K4C_MAX = 0.25
DOMAIN = "computing"

# Two phrasings per fact = 12 items. `accept` lists content that MUST appear
# in a correct answer. Queries never contain the accepted string.
ITEMS = [
    ("fact_01.txt", "What is the ambient dimension of a HENRI wave state?",
     ["65536", "65,536"]),
    ("fact_01.txt", "How many dimensions does a single HENRI wave have?",
     ["65536", "65,536"]),
    ("fact_02.txt", "How many blocks organize a HENRI wave?",
     ["8192", "8,192"]),
    ("fact_02.txt", "What is the block count of a full HENRI wave?",
     ["8192", "8,192"]),
    ("fact_03.txt", "Which Clifford algebra does each HENRI block carry?",
     ["cl(3,0)", "clifford", "3,0"]),
    ("fact_03.txt", "What algebra type is used per HENRI block?",
     ["cl(3,0)", "clifford"]),
    ("fact_04.txt", "How many slots does each HENRI block contain?",
     ["8 slots", "eight slots"]),
    ("fact_04.txt", "What is the slot count per HENRI block?",
     ["8 slots", "eight slots", "8)"]),
    ("fact_05.txt", "Which database stores Zone C engrams?",
     ["timescaledb", "timescale"]),
    ("fact_05.txt", "What persistent store holds Zone C engrams?",
     ["timescaledb", "timescale"]),
    ("fact_06.txt", "Which ANN index type does Zone C retrieval use?",
     ["hnsw"]),
    ("fact_06.txt", "What index type is used for Zone C retrieval?",
     ["hnsw"]),
]

PROMPT_TMPL = (
    "Answer the question using ONLY the provided evidence.\n"
    "If the evidence is insufficient, answer: ABSTAIN.\n\n"
    f"EVIDENCE:\n{{ctx}}\n\nQUESTION: {{q}}\n\nANSWER:"
)


def hit(answer: str, accept) -> bool:
    a = (answer or "").lower()
    return any(tok.lower() in a for tok in accept)


def main() -> int:
    os.environ["HENRI_BACKBONE"] = "1"
    os.environ["HENRI_BACKBONE_MODEL_DIR"] = MODEL_DIR
    items = ITEMS[:_AK4_LIMIT] if _AK4_LIMIT > 0 else ITEMS

    import zone_c_world_knowledge_harness as wkh
    from henri_backbone_adapter import QwenBackboneAdapter

    t_total = time.time()
    report = {"preregistration": {
        "domain": DOMAIN, "n_items": len(ITEMS), "max_new_tokens": MAX_NEW,
        "K4a_min": K4A_MIN, "K4b_min": K4B_MIN, "K4c_max": K4C_MAX,
        "scoring": "content_presence_not_argmax",
        "vacuity_rule": "zero retrieved chunks for any A1 item -> NO VERDICT",
    }}

    # ---- retrieval ---------------------------------------------------------
    print("== retrieval ==", flush=True)
    ctx_of, src_of = {}, {}
    for src, q, _acc in ITEMS:
        hits = wkh.query_corpus(DOMAIN, q, 3, dsn_env=DSN_ENV)
        text, used = wkh.build_context(hits, Path(CORPUS), max_chars=3000)
        ctx_of[q] = text
        src_of[q] = (used[0]["source_id"] if used else None)
    zero = [q for q in ctx_of if src_of[q] is None]
    report["retrieval"] = {"items": len(ctx_of),
                           "matched_source": {q: src_of[q] for q in ctx_of},
                           "zero_context_items": zero}
    print("   zero_context_items =", zero, flush=True)

    # ---- model (ONE instance for all generations) --------------------------
    print("== adapter load ==", flush=True)
    adapt = QwenBackboneAdapter(model_dir=MODEL_DIR)
    adapt.load()
    tel = adapt.telemetry
    report["model"] = {
        "architecture": getattr(tel, "architecture", None),
        "total_params": getattr(tel, "total_params", None),
        "unexpected_key_count": getattr(tel, "unexpected_key_count", None),
        "loaded_from": getattr(tel, "loaded_from", None),
        "load_status": getattr(tel, "checkpoint_load_status", None),
    }
    print("   %s" % report["model"], flush=True)
    if (report["model"]["unexpected_key_count"] or 0) != 0:
        raise SystemExit("BLOCKED: checkpoint did not load cleanly")

    def answer(q, ctx):
        txt, _meta = adapt.generate_text(PROMPT_TMPL.format(ctx=ctx, q=q))
        return (txt or "").strip()

    # ---- arms --------------------------------------------------------------
    print("== arms ==", flush=True)
    # CONTAMINATION DEFECT FIXED (2026-10-03, smoke run 12-item evidence):
    # The first version reused another item's context directly. With 6 chunks
    # and k=3, every context already contains 3 of 6 facts, so A2 shared ~50%
    # of its evidence with A1 and scored hit_A2 = 0.500 exactly -- a
    # CONTAMINATED negative control, not a mechanism failure. A2 now EXCLUDES
    # every chunk whose source_id is the target source, and the retained
    # source list is recorded as evidence.
    mism_of, mism_src = {}, {}
    for src, q, _acc in items:
        m, ms = "", None
        for s2, q2, _a in items:
            if s2 == src:
                continue
            h2 = wkh.query_corpus(DOMAIN, q2, 3, dsn_env=DSN_ENV)
            h2 = [h for h in h2 if h["source_id"] != src]
            if not h2:
                continue
            t2, u2 = wkh.build_context(h2, Path(CORPUS), max_chars=3000)
            if t2 and u2 and all(u["source_id"] != src for u in u2):
                m, ms = t2, [u["source_id"] for u in u2]
                break
        mism_of[q], mism_src[q] = m, ms
    print("   A2 non-contaminating source set: %s" % mism_src, flush=True)
    report["A2_sources"] = mism_src

    rows = []
    for src, q, acc in ITEMS:
        correct = ctx_of[q]
        mismatch = mism_of[q]
        a0 = answer(q, "")            # A0 null: no evidence at all
        a1 = answer(q, correct)
        a2 = answer(q, mismatch) if mismatch else ""
        # determinism: repeat A1
        a1b = answer(q, correct)
        rows.append({
            "source": src, "q": q, "accept": acc,
            "A0": a0, "A1": a1, "A2": a2,
            "A1_repeat": a1b,
            "h0": hit(a0, acc), "h1": hit(a1, acc), "h2": hit(a2, acc),
            "deterministic": a1 == a1b,
            "A2_sources": mism_src[q],
        })
        print("   %-14s h0=%-5s h1=%-5s h2=%-5s det=%-5s A1=%r"
              % (src, rows[-1]["h0"], rows[-1]["h1"], rows[-1]["h2"],
                 rows[-1]["deterministic"], a1[:60]), flush=True)

    n = len(rows)
    r1 = sum(r["h1"] for r in rows) / n
    r0 = sum(r["h0"] for r in rows) / n
    r2 = sum(r["h2"] for r in rows) / n
    det = sum(r["deterministic"] for r in rows) / n
    nonempty = sum(1 for r in rows if len(r["A1"]) > 0) / n
    abstain = sum(1 for r in rows if "abstain" in r["A1"].lower()) / n

    summary = {
        "n": n, "hit_A0": r0, "hit_A1": r1, "hit_A2": r2,
        "k4a_gain_A1_over_A0": r1 - r0, "k4b_gain_A1_over_A2": r1 - r2,
        "determinism": det, "A1_nonempty_ratio": nonempty,
        "A1_abstain_ratio": abstain,
    }
    summary["K4a"] = (r1 - r0) >= K4A_MIN
    summary["K4b"] = (r1 - r2) >= K4B_MIN
    summary["K4c"] = (r0 <= K4C_MAX) and (r2 <= K4C_MAX)
    summary["K4d"] = det >= 1.0
    summary["P0_positive_control"] = (nonempty >= 0.90) and (r1 >= K4A_MIN)

    if zero:
        summary["verdict"] = "VACUOUS_NO_RETRIEVAL"
    elif summary["K4a"] and summary["K4b"] and summary["K4c"] and summary["K4d"]:
        summary["verdict"] = "AK4_PASS_CONTEXT_IMPROVES_GENERATED_ANSWER"
    else:
        summary["verdict"] = "AK4_FAIL:" + ",".join(
            k for k in ("K4a", "K4b", "K4c", "K4d") if not summary[k])

    report["items"] = rows
    report["summary"] = summary
    report["elapsed_s"] = round(time.time() - t_total, 2)
    report["evidence_class"] = "OBSERVED"
    report["scope_limit"] = (
        "6-fact synthetic corpus, 12 items, CPU, frozen 1.5B. Mechanism evidence, "
        "not end-task capability. Latency NOT measured.")

    body = json.dumps(report, sort_keys=True, indent=2)
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    outp = os.path.join(TMP, "ak4_egress_receipt.json")
    Path(outp).write_text(body, encoding="utf-8")

    print("\n== SUMMARY ==")
    for k in ("n", "hit_A0", "hit_A1", "hit_A2", "k4a_gain_A1_over_A0",
              "k4b_gain_A1_over_A2", "determinism", "A1_nonempty_ratio",
              "A1_abstain_ratio", "K4a", "K4b", "K4c", "K4d",
              "P0_positive_control", "verdict"):
        print("  %-22s = %s" % (k, summary[k]))
    print("  receipt      = %s" % outp)
    print("  receipt_sha256 = %s" % sha)
    return 0 if summary["verdict"].startswith("AK4_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
