"""E4a — Construct Reform and Metric Re-Registration (zero-trainable, model-free).

Re-derives evaluator bounds for TWO constructs on a FRESH, never-measured split:

  C1 sentence-window : prefix = words[:-1], gold = first BPE token of last word
                       (the E2/E3/Gate-3.1 construct — kept for continuity)
  C2 token-stream    : prefix = token stream, gold = next token
                       (the honest next-token task)

For each construct it measures, on the FRESH split only:
  - the context-free marginal baseline           (trivial floor)
  - the best context-conditioned trivial baseline (prefix-only backoff)
  - the empirical entropy floor H(gold) and H(gold|context)
  - gold-in-top-k coverage at the scored object
and then REGISTERS bounds strictly above the strongest trivial baseline.

Kill criterion (E4a): any registered bound <= its trivial baseline => TERMINATE.

Split discipline: the E2/E3/Gate-3.1 split (windows 0..10,999) has been measured
at least three times. This script uses windows 11,000..20,999 (calib) and
21,000..21,999 (eval) — never previously measured. Fresh-split receipt written.

Usage: python e4a_construct_audit.py [--corpus PATH] [--tokenizer PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

MAX_WORDS = 24
VOCAB = 151936
FRESH_CALIB = (11000, 21000)
FRESH_EVAL = (21000, 22000)


def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " ").replace("  ", " "))
    return [p.strip() for p in parts if len(p.strip().split()) >= 3]


def entropy(counts: Counter) -> float:
    n = sum(counts.values())
    if n == 0:
        return 0.0
    return -sum((c / n) * math.log(c / n) for c in counts.values() if c > 0)


def cond_entropy(ctx: dict, key) -> float:
    tot = sum(sum(c.values()) for c in ctx.values())
    if tot == 0:
        return 0.0
    acc = 0.0
    for c in ctx.values():
        n = sum(c.values())
        acc += (n / tot) * entropy(c)
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path,
                    default=Path(r"C:\Users\chan\henri-telemetry\e2\wikitext2_train.parquet"))
    ap.add_argument("--tokenizer", type=Path,
                    default=Path(r"C:\Users\chan\AppData\Local\Temp\henri-e3-audit\tokenizer.json"))
    ap.add_argument("--out", type=Path,
                    default=Path(r"C:\Users\chan\henri-telemetry\e3\e4a_construct_audit.json"))
    args = ap.parse_args()

    import pyarrow.parquet as pq
    from tokenizers import Tokenizer

    corpus_sha = hashlib.sha256(args.corpus.read_bytes()).hexdigest()
    tok_sha = hashlib.sha256(args.tokenizer.read_bytes()).hexdigest()
    print(f"corpus_sha256 {corpus_sha[:16]} (expect e83889ba)")
    print(f"tokenizer_sha256 {tok_sha[:16]}")
    assert corpus_sha[:8] == "e83889ba", "CORPUS_SHA_MISMATCH"

    tok = Tokenizer.from_file(str(args.tokenizer))
    enc = lambda t: tok.encode(t).ids  # noqa: E731
    rows = pq.read_table(str(args.corpus)).column("text").to_pylist()
    print(f"corpus_rows {len(rows)}")

    # ---------------- C1: sentence-window construct ------------------------
    sents: list[str] = []
    for r in rows:
        sents.extend(sentence_split(r))
    c1: list[tuple[list[str], int]] = []
    for s in sents:
        w = s.split()[:MAX_WORDS]
        if len(w) < 2:
            continue
        prefix, window = " ".join(w[:-1]), " ".join(w)
        ia, ip = enc(window), enc(prefix)
        if ia[:len(ip)] == ip and len(ia) > len(ip):
            c1.append((w[:-1], ia[len(ip)]))
    print(f"C1_windows_total {len(c1)}")

    # ---------------- C2: token-stream construct ---------------------------
    stream: list[int] = []
    for r in rows:
        stream.extend(enc(r))
    c2: list[tuple[tuple, int]] = []
    for i in range(3, len(stream)):
        c2.append(((stream[i - 3], stream[i - 2], stream[i - 1]), stream[i]))
    print(f"C2_stream_pairs_total {len(c2)}")

    # ---------------- fresh split ------------------------------------------
    assert len(c1) >= FRESH_EVAL[1], "INSUFFICIENT_C1_WINDOWS"
    assert len(c2) > 4 * FRESH_EVAL[1], "INSUFFICIENT_C2_PAIRS"

    c1_calib, c1_eval = c1[FRESH_CALIB[0]:FRESH_CALIB[1]], c1[FRESH_EVAL[0]:FRESH_EVAL[1]]
    c2_calib, c2_eval = c2[:FRESH_CALIB[1] - FRESH_CALIB[0]], \
        c2[4 * FRESH_CALIB[1]:4 * FRESH_CALIB[1] + FRESH_EVAL[1] - FRESH_EVAL[0]]
    print(f"fresh C1 calib={len(c1_calib)} eval={len(c1_eval)}")
    print(f"fresh C2 calib={len(c2_calib)} eval={len(c2_eval)}")

    report: dict = {
        "carrier": "E4a",
        "construct_reform": True,
        "constructs": {},
        "fresh_split": {
            "c1_index_range_calib": list(FRESH_CALIB),
            "c1_index_range_eval": list(FRESH_EVAL),
            "c2_pair_range_calib": [0, len(c2_calib)],
            "c2_pair_range_eval": [4 * FRESH_CALIB[1], 4 * FRESH_CALIB[1] + len(c2_eval)],
            "previously_measured_splits": ["0-10999 (E2/E3/Gate3.1)"],
            "single_use": True,
        },
    }

    # ============ C1 measurements =========================================
    c1_uni = Counter(g for _, g in c1_calib)
    c1_uni_top = [t for t, _ in c1_uni.most_common(VOCAB)]
    c1_m1: dict = defaultdict(Counter)
    c1_m3: dict = defaultdict(Counter)
    for p, g in c1_calib:
        if p:
            c1_m1[p[-1]][g] += 1
            if len(p) >= 3:
                c1_m3[tuple(p[-3:])][g] += 1

    def c1_cands(p, order):
        out, seen = [], set()
        for c, on in ((c1_m3.get(tuple(p[-3:])) if len(p) >= 3 else None, order >= 3),
                      (c1_m1.get(p[-1]) if p else None, order >= 1)):
            if on and c:
                for t, _ in c.most_common(8):
                    if t not in seen:
                        seen.add(t)
                        out.append(t)
        for t in c1_uni_top[:8]:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def c1_score(order):
        p1 = p5 = 0
        for p, g in c1_eval:
            cs = c1_uni_top[:8] if order == 0 else c1_cands(p, order)
            if cs and cs[0] == g:
                p1 += 1
            if g in cs[:5]:
                p5 += 1
        n = len(c1_eval)
        return round(p1 / n, 4), round(p5 / n, 4)

    m_p1, m_p5 = c1_score(0)
    b_p1, b_p5 = c1_score(3)
    # coverage: gold in the candidate set actually scored (full V for a full softmax)
    cov_full = 1.0  # gold is always in a full-V softmax denominator
    cov_top16 = sum(1 for p, g in c1_eval if g in c1_cands(p, 3)[:16]) / len(c1_eval)

    h_c1 = entropy(c1_uni)
    h_c1_cond = cond_entropy(c1_m3, None)
    h_c1_eval = entropy(Counter(g for _, g in c1_eval))
    c1_dot = c1_uni.get(659, 0) / len(c1_calib)

    report["constructs"]["C1_sentence_window"] = {
        "description": "prefix=words[:-1]; gold=first BPE token of last word",
        "n_calib": len(c1_calib), "n_eval": len(c1_eval),
        "marginal_baseline": {"p1": m_p1, "p5": m_p5},
        "best_trivial_baseline": {"kind": "prefix_last3_backoff", "p1": b_p1, "p5": b_p5},
        "ce_floor_uniform": round(math.log(VOCAB), 6),
        "ce_floor_marginal_H_calib": round(h_c1, 6),
        "ce_floor_marginal_H_eval": round(h_c1_eval, 6),
        "ce_floor_conditional_in_sample_DEGENERATE": {
            "value": round(h_c1_cond, 6),
            "caveat": ("in-sample conditional entropy over sparse trigram keys "
                       "(most keys are singletons -> H=0 by construction). "
                       "NOT a usable floor. Reported as diagnostic only."),
        },
        "distinct_golds": len(c1_uni),
        "dot_token_rate_calib": round(c1_dot, 4),
        "coverage_gold_in_full_scoring_set": cov_full,
        "coverage_gold_in_top16_candidates": round(cov_top16, 4),
    }

    # ============ C2 measurements =========================================
    c2_uni = Counter(g for _, g in c2_calib)
    c2_uni_top = [t for t, _ in c2_uni.most_common(8)]
    c2_m1: dict = defaultdict(Counter)
    c2_m3: dict = defaultdict(Counter)
    for ctx, g in c2_calib:
        c2_m1[ctx[-1]][g] += 1
        c2_m3[ctx][g] += 1

    def c2_cands(ctx, order):
        out, seen = [], set()
        if order >= 3 and c2_m3.get(ctx):
            for t, _ in c2_m3[ctx].most_common(8):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        if order >= 1 and c2_m1.get(ctx[-1]):
            for t, _ in c2_m1[ctx[-1]].most_common(8):
                if t not in seen:
                    seen.add(t)
                    out.append(t)
        for t in c2_uni_top:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    def c2_score(order):
        p1 = p5 = 0
        for ctx, g in c2_eval:
            cs = c2_cands(ctx, order)
            if cs and cs[0] == g:
                p1 += 1
            if g in cs[:5]:
                p5 += 1
        n = len(c2_eval)
        return round(p1 / n, 4), round(p5 / n, 4)

    c2_m_p1, c2_m_p5 = c2_score(1)
    c2_b_p1, c2_b_p5 = c2_score(3)
    h_c2 = entropy(c2_uni)
    h_c2_cond = cond_entropy(c2_m3, None)

    report["constructs"]["C2_token_stream"] = {
        "description": "prefix=token stream; gold=next token (honest next-token task)",
        "n_calib": len(c2_calib), "n_eval": len(c2_eval),
        "marginal_baseline": {"p1": c2_m_p1, "p5": c2_m_p5},
        "best_trivial_baseline": {"kind": "bpe_trigram_backoff", "p1": c2_b_p1, "p5": c2_b_p5},
        "ce_floor_uniform": round(math.log(VOCAB), 6),
        "ce_floor_marginal_H_calib": round(h_c2, 6),
        "ce_floor_conditional_in_sample_DEGENERATE": {
            "value": round(h_c2_cond, 6),
            "caveat": ("in-sample conditional entropy over sparse trigram keys; "
                       "NOT a usable floor. Diagnostic only."),
        },
        "distinct_golds": len(c2_uni),
    }

    # ============ bound registration + kill check ==========================
    # CE bound must be a HELD-OUT cross-entropy the strongest trivial model can
    # actually achieve. In-sample conditional entropy is degenerate (sparse
    # singleton keys -> H=0 by construction) and produced a negative bound.
    def proper_prob(tables, ctx_key, g, lam):
        """Backoff mixture: lambda * key-distribution + (1-lambda) * unigram."""
        uni, m1, m3, n_calib = tables
        p_uni = uni.get(g, 0) / n_calib
        if len(ctx_key) >= 3:
            c = m3.get(ctx_key)
            if c:
                n = sum(c.values())
                return lam * (c.get(g, 0) / n) + (1 - lam) * p_uni
        c = m1.get(ctx_key[-1])
        if c:
            n = sum(c.values())
            return lam * (c.get(g, 0) / n) + (1 - lam) * p_uni
        return p_uni

    def heldout_ce(items, tables, to_key, lam=0.7):
        tot = 0.0
        for ctx, g in items:
            p = proper_prob(tables, to_key(ctx), g, lam)
            tot += -math.log(max(p, 1e-12))
        return tot / len(items)

    c1_tables = (c1_uni, c1_m1, c1_m3, len(c1_calib))
    c2_tables = (c2_uni, c2_m1, c2_m3, len(c2_calib))

    def best_trivial_ce(items, tables, to_key):
        """Lowest HELD-OUT CE over the trivial family: {uniform, unigram,
        lambda-interpolated backoff}. lam=0 reduces to the unigram model."""
        uni, n = tables[0], tables[3]
        uni_ce = -sum(math.log(max(uni.get(g, 0) / n, 1e-12))
                      for _, g in items) / len(items)
        cands = {"uniform": math.log(VOCAB), "unigram": uni_ce}
        best_ce, best_lam = uni_ce, 0.0
        for lam in (0.2, 0.4, 0.6, 0.8):
            ce = heldout_ce(items, tables, to_key, lam)
            cands[f"backoff_lam{lam}"] = round(ce, 4)
            if ce < best_ce:
                best_ce, best_lam = ce, lam
        cands["best_backoff"] = round(best_ce, 4)
        return min(cands["uniform"], cands["best_backoff"]), best_lam, cands

    ce_max_c1, lam_c1, cand_c1 = best_trivial_ce(c1_eval, c1_tables,
                                                 lambda p: tuple(p[-3:]))
    ce_max_c2, lam_c2, cand_c2 = best_trivial_ce(c2_eval, c2_tables, lambda c: tuple(c))
    # Marginal-only CE (trivial model that ignores context entirely)
    ce_marg_c1 = heldout_ce(c1_eval, c1_tables, lambda p: tuple(), lam=0.0) \
        if False else -sum(math.log(max(c1_uni.get(g, 0) / len(c1_calib), 1e-12))
                           for _, g in c1_eval) / len(c1_eval)
    ce_marg_c2 = -sum(math.log(max(c2_uni.get(g, 0) / len(c2_calib), 1e-12))
                      for _, g in c2_eval) / len(c2_eval)

    ce_margin = 0.10  # nats; bound must beat the trivial model by this much
    p_margin = 0.05

    def reg(base_p1, base_p5, base_ce, tag):
        p1 = round(base_p1 + p_margin, 3)
        p5 = round(base_p5 + p_margin, 3)
        ce = round(base_ce - ce_margin, 3)
        return {"p1": p1, "p5": p5, "ce_max": ce, "baseline_kind": tag}

    c1_bounds = reg(b_p1, b_p5, ce_max_c1, "prefix_last3_backoff")
    c2_bounds = reg(max(c2_m_p1, c2_b_p1), max(c2_m_p5, c2_b_p5), ce_max_c2,
                    "max(marginal,bpe_trigram_backoff)")

    assert 0 < c1_bounds["ce_max"] < ce_max_c1, "C1_CE_BOUND_NOT_STRICTLY_BETTER"
    assert 0 < c2_bounds["ce_max"] < ce_max_c2, "C2_CE_BOUND_NOT_STRICTLY_BETTER"

    checks = {
        "C1_p1_above_baseline": c1_bounds["p1"] > b_p1,
        "C1_p5_above_baseline": c1_bounds["p5"] > b_p5,
        "C2_p1_above_baseline": c2_bounds["p1"] > c2_b_p1,
        "C2_p5_above_baseline": c2_bounds["p5"] > c2_b_p5,
    }
    terminate = not all(checks.values())

    report["registered_bounds"] = {
        "C1_sentence_window": c1_bounds,
        "C2_token_stream_primary": c2_bounds,
        "p_margin": p_margin,
        "ce_margin_nats": ce_margin,
        "rule": ("P@k bound must exceed the STRONGEST trivial baseline "
                 "(not the marginal); CE bound must beat the strongest trivial "
                 "model's HELD-OUT CE by ce_margin nats"),
        "trivial_baseline_heldout_CE": {
            "C1_prefix_last3_backoff": round(ce_max_c1, 4),
            "C1_marginal_only": round(ce_marg_c1, 4),
            "C2_trigram_backoff": round(ce_max_c2, 4),
            "C2_marginal_only": round(ce_marg_c2, 4),
        },
    }
    report["kill_check"] = checks
    report["verdict"] = "E4A_TERMINATE" if terminate else "E4A_BOUNDS_REGISTERED"
    report["retired_bounds"] = {
        "p1": 0.285, "p5": 0.640,
        "reason": "registered BELOW the context-free marginal baseline; vacuous",
        "supersedes": ["Gate 3.1", "E3"],
    }
    report["E3_cross_check"] = {
        "authoritative": "henri-telemetry/e3/e3_construct_audit.json",
        "e3_sha256_prefix": "4f08ef31",
    }
    report["corpus_sha256"] = corpus_sha
    report["tokenizer_sha256"] = tok_sha

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    out_sha = hashlib.sha256(args.out.read_bytes()).hexdigest()

    print()
    print("=== C1 sentence-window ===")
    print(f"  marginal baseline   P@1={m_p1} P@5={m_p5}")
    print(f"  prefix-backoff base P@1={b_p1} P@5={b_p5}")
    print(f"  H(gold)={h_c1:.3f}  H(gold|last3)={h_c1_cond:.3f}  uniform={math.log(VOCAB):.3f}")
    print(f"  dot_rate={c1_dot:.3f}  distinct_golds={len(c1_uni)}")
    print("=== C2 token-stream ===")
    print(f"  marginal baseline   P@1={c2_m_p1} P@5={c2_m_p5}")
    print(f"  trigram-backoff     P@1={c2_b_p1} P@5={c2_b_p5}")
    print(f"  H(gold)={h_c2:.3f}  H(gold|trigram)={h_c2_cond:.3f}")
    print()
    print(f"registered C1 bounds {c1_bounds}")
    print(f"registered C2 bounds {c2_bounds}")
    print(f"kill checks {checks}")
    print(f"E4A_VERDICT={report['verdict']}")
    print(f"wrote {args.out}")
    print(f"e4a_audit_sha256 {out_sha}")
    raise SystemExit(1 if terminate else 0)


if __name__ == "__main__":
    main()
