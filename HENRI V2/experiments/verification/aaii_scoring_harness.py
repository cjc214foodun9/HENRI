"""AAII v4.3 scoring harness. Scoring, not staging.

REMEDIES THE RECORDED GAP
    "g3_aaii_scaffold.py is staging: <=2 items/constituent, diagnostic-only."
    That module stages sources. This module scores items end to end.

WHAT THIS IS NOT
    It does NOT emit an "AAII v4.3 score". Per
    henri-research/references/aaii-v43-composite-and-exposure-audit.md:
      * only 25% of index weight is locally reproducible
        (SciCode, Terminal-Bench 4.0, AutomationBench-AA)
      * 75% is externally graded or private (Elo panels, HLE, LCR,
        AA-Omniscience, CritPt, GDP.pdf)
    Any number from a bounded subset is DERIVED, never the composite.
    External members record BLOCKED_PRIVATE_GRADER with the reason.

CONTRACT
    * per-item JSONL is written BEFORE aggregation (earned lesson)
    * per-run output paths are keyed to commit SHA (earned lesson)
    * contamination gate runs before scoring; a firing gate voids the run
    * constituent weights and category totals are read from the pinned
      audit, never hand-summed

Every emitted verdict carries an explicit evidence class. Extends the
existing henri.run-evidence.v1 schema rather than inventing a parallel one.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

# Pinned from aaii-v43-composite-and-exposure-audit.md (2026-09-16).
# weights sum to 1.0000; categories 30/20/30/20.
CONSTITUENTS = {
    "AA-Briefcase":        {"cat": "Agents",           "w": 0.15, "tools": True,  "local": False, "why": "rubric Elo, judge panel"},
    "GDPval-AA v2":        {"cat": "Agents",           "w": 0.10, "tools": True,  "local": False, "why": "pairwise Elo, human anchor 1000"},
    "AutomationBench-AA":  {"cat": "Agents",           "w": 0.05, "tools": True,  "local": True,  "why": "objective completion, no LLM judge"},
    "Terminal-Bench 4.0":  {"cat": "Coding",           "w": 0.10, "tools": True,  "local": True,  "why": "test-suite pass@1"},
    "SciCode":             {"cat": "Coding",           "w": 0.10, "tools": False, "local": True,  "why": "code exec pass@1"},
    "AA-Omniscience":      {"cat": "General",          "w": 0.15, "tools": False, "local": False, "why": "non-public"},
    "GDP.pdf":             {"cat": "General",          "w": 0.10, "tools": False, "local": False, "why": "all-pass headline"},
    "AA-LCR v1.1":         {"cat": "General",          "w": 0.05, "tools": False, "local": False, "why": "Equality Checker LLM"},
    "HLE":                 {"cat": "SciReasoning",     "w": 0.10, "tools": False, "local": False, "why": "gated"},
    "CritPt":              {"cat": "SciReasoning",     "w": 0.10, "tools": False, "local": False, "why": "official grading server"},
}

SCHEMA_ID = "henri.aaii-scoring-run.v1"


def check_weights() -> None:
    total = sum(c["w"] for c in CONSTITUENTS.values())
    assert abs(total - 1.0) < 1e-9, f"weights sum {total}, expected 1.0"
    cats = {}
    for c in CONSTITUENTS.values():
        cats[c["cat"]] = cats.get(c["cat"], 0.0) + c["w"]
    for cat, exp in (("Agents", 0.30), ("Coding", 0.20),
                     ("General", 0.30), ("SciReasoning", 0.20)):
        assert abs(cats[cat] - exp) < 1e-9, f"{cat}={cats[cat]} expected {exp}"


def eight_grams(text: str) -> set:
    t = text.split()
    return {" ".join(t[i:i + 8]) for i in range(max(0, len(t) - 7))}


def contamination_gate(items: list, corpus_ngrams: set, threshold: float = 0.5):
    """Exclude items whose 8-gram overlap with the contamination corpus
    exceeds threshold. Returns (kept, excluded_ids)."""
    kept, excluded = [], []
    for it in items:
        grams = eight_grams(it["prompt"])
        if not grams:
            kept.append(it)
            continue
        overlap = len(grams & corpus_ngrams) / len(grams)
        (excluded if overlap > threshold else kept).append(it["item_id"])
    return [i for i in items if i["item_id"] not in set(excluded)], excluded


def score_local(item: dict, response: str) -> float:
    """Deterministic exact-match on the declared gold field.

    This is the LOCAL scorer for locally-reproducible constituents only.
    External constituents are never scored here; they record a block.
    """
    gold = item.get("gold")
    if gold is None:
        return 0.0
    return 1.0 if response.strip() == str(gold).strip() else 0.0


def run(items: list, model_fn, out_dir: str, commit: str, seed: int,
        corpus_ngrams: set | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    corpus_ngrams = corpus_ngrams or set()

    kept, excluded = contamination_gate(items, corpus_ngrams)
    if excluded:
        verdict_prefix = "CONTAMINATION_BLOCKED"
    else:
        verdict_prefix = None

    # per-item JSONL persisted BEFORE aggregation
    jsonl_path = os.path.join(out_dir, f"items_{commit[:8]}.jsonl")
    t0 = time.time()
    with open(jsonl_path, "w", encoding="utf-8") as fh:
        for it in kept:
            const = it["constituent"]
            meta = CONSTITUENTS.get(const)
            if meta is None:
                rec = {"item_id": it["item_id"], "constituent": const,
                       "status": "UNKNOWN_CONSTITUENT", "score": None}
            elif not meta["local"]:
                rec = {"item_id": it["item_id"], "constituent": const,
                       "status": "BLOCKED_PRIVATE_GRADER", "why": meta["why"],
                       "score": None}
            else:
                try:
                    resp = model_fn(it["prompt"])
                    rec = {
                        "item_id": it["item_id"], "constituent": const,
                        "status": "SCORED",
                        "prompt_sha256": hashlib.sha256(
                            it["prompt"].encode()).hexdigest(),
                        "response_sha256": hashlib.sha256(
                            resp.encode()).hexdigest(),
                        "response_chars": len(resp),
                        "score": score_local(it, resp),
                    }
                except Exception as exc:
                    rec = {"item_id": it["item_id"], "constituent": const,
                           "status": "BLOCKED_INFRA", "error": repr(exc)[:120],
                           "score": None}
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
            fh.flush()

    rows = [json.loads(l) for l in open(jsonl_path, encoding="utf-8")]
    scored = [r for r in rows if r["status"] == "SCORED"]

    # per-constituent means over locally scored items only
    per_const = {}
    for name in CONSTITUENTS:
        vals = [r["score"] for r in scored if r["constituent"] == name]
        per_const[name] = (sum(vals) / len(vals), len(vals)) if vals else (None, 0)

    # local-only weighted sub-index, with the covered weight declared
    covered, acc = 0.0, 0.0
    for name, meta in CONSTITUENTS.items():
        if meta["local"] and per_const[name][0] is not None:
            covered += meta["w"]
            acc += meta["w"] * per_const[name][0]
    local_subindex = (acc / covered) if covered > 0 else None

    out = {
        "schema_id": SCHEMA_ID,
        "commit": commit,
        "seed": seed,
        "n_items_input": len(items),
        "n_items_kept": len(kept),
        "n_items_scored": len(scored),
        "excluded_contamination": excluded,
        "per_constituent": {k: {"mean": v[0], "n": v[1]}
                            for k, v in per_const.items()},
        "local_subindex_mean": local_subindex,
        "local_weight_covered": covered,
        "external_weight_blocked": 1.0 - covered,
        "verdict": verdict_prefix or "DIAGNOSTIC_LOCAL_ONLY",
        "evidence_class": "DERIVED",
        "notice": ("This is NOT an AAII v4.3 score. 75% of index weight is "
                   "externally graded or private and is reported as blocked. "
                   "The local sub-index covers only locally reproducible "
                   "constituents and is diagnostic."),
        "elapsed_s": round(time.time() - t0, 3),
        "items_jsonl": jsonl_path,
    }
    body = json.dumps(out, sort_keys=True).encode()
    out["receipt_sha256"] = hashlib.sha256(body).hexdigest()
    summary_path = os.path.join(out_dir, f"summary_{commit[:8]}.json")
    json.dump(out, open(summary_path, "w"), indent=2)
    return out, summary_path


def main():
    check_weights()
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSONL: item_id,constituent,prompt[,gold]")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model-dir", default=None,
                    help="pinned checkpoint dir; omit for a stub responder")
    ap.add_argument("--contamination-corpus", default=None,
                    help="text file whose 8-grams are checked against items")
    ap.add_argument("--seed", type=int, default=20261002)
    a = ap.parse_args()

    commit = subprocess.check_output(
        ["git", "-C", os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
         "rev-parse", "HEAD"], text=True).strip()

    items = [json.loads(l) for l in open(a.items, encoding="utf-8")]
    grams = set()
    if a.contamination_corpus and os.path.exists(a.contamination_corpus):
        grams = eight_grams(open(a.contamination_corpus, encoding="utf-8").read())

    if a.model_dir:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        os.environ.setdefault("HENRI_BACKBONE", "1")
        from henri_backbone_adapter import QwenBackboneAdapter
        ad = QwenBackboneAdapter(model_dir=a.model_dir)
        ad.load()
        model_fn = lambda p: ad.generate_text(p)[0]
    else:
        model_fn = lambda p: "ABSTAIN"

    out, path = run(items, model_fn, a.out_dir, commit, a.seed, grams)
    print(json.dumps({k: out[k] for k in (
        "n_items_input", "n_items_scored", "local_subindex_mean",
        "local_weight_covered", "external_weight_blocked", "verdict",
        "evidence_class")}, indent=2))
    print("SUMMARY=" + path)
    print("RECEIPT_SHA256=" + out["receipt_sha256"][:24])


if __name__ == "__main__":
    main()
