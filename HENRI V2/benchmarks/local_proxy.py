"""Local proxy scorers for pinned AAII datasets. Deterministic. Judge-free.

WHAT THIS IS -- READ FIRST
    A SCORER VALIDATION, not a model evaluation. It proves the metric
    implementation discriminates, by scoring deterministic control arms whose
    expected values are known before the run:

        oracle_self : predict the pinned gold answer      -> must be 1.0
        abstain     : predict the empty string           -> must be 0.0
        rotate_1    : predict the NEXT item's gold answer-> must be ~0

    A scorer that cannot be driven to 1.0, or cannot be driven to 0.0, is a
    broken instrument and every downstream number is worthless. These three
    arms bracket it.

WHAT THIS IS NOT
    NOT an AAII v4.3.2 score. AAII grades HLE and AA-Omniscience with an
    external equality-checker / grading model (GPT-5.6 Luna medium per the
    operator PDFs). This module uses normalized exact string match, which is a
    STRICT LOWER BOUND on that judge: any model score produced here
    understates the official metric and must never be reported as it.

WHY NORMALIZED MATCH IS THE RIGHT PROXY
    Both pinned datasets ship a short, closed-form `answer` field (e.g.
    "ASC 606-10-25-15"). Short factual answers are exactly the case where
    deterministic matching agrees with a judge, so the proxy is meaningful
    here in a way it would not be for HLE's proof-style questions. That
    asymmetry is recorded in the receipt, not hidden.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")

# ------------------------------------------------------------------ normalize
_LEAD = re.compile(r'^\s*(?:the\s+answer\s+is|final\s+answer|answer|ans)\s*[:\-]\s*', re.I)
_WS = re.compile(r'[\s\u00a0]+')
_TRIM = re.compile(r'^[\s"\'`$\\{}\[\]()]+|[\s"\'`$\\{}\[\]()]+$')
_TAIL = re.compile(r'\s*[,;:.!?]+\s*$')


def norm(s) -> str:
    """Aggressive normalization for short-answer equality comparison."""
    if s is None:
        return ""
    t = str(s).strip()
    t = _LEAD.sub("", t)
    t = t.strip("$").strip()
    t = _TRIM.sub("", t)
    t = _WS.sub(" ", t).strip().lower()
    t = _TAIL.sub("", t)
    return t


# ------------------------------------------------------------------ loaders
def load(eval_id: str) -> list[dict]:
    if eval_id == "hle":
        return _load_hle()
    if eval_id == "aa-omniscience":
        return _load_omni()
    raise KeyError(eval_id)


def _load_hle() -> list[dict]:
    import pyarrow.parquet as pq
    paths = glob.glob(os.path.join(DATA, "hle/**/*.parquet"), recursive=True)
    if not paths:
        raise FileNotFoundError("no HLE parquet pinned")
    tbl = pq.read_table(paths[0]).to_pydict()
    cols = set(tbl)
    qk = "question"
    ak = "answer" if "answer" in cols else ("canonical_answer" if "canonical_answer" in cols else None)
    imk = "image" if "image" in cols else None
    n = len(tbl[qk])
    rows = []
    for i in range(n):
        img = None
        if imk:
            v = tbl[imk][i]
            if v not in (None, b"", "", [], {}):
                img = "present"
        rows.append({"q": tbl[qk][i], "a": (tbl[ak][i] if ak else None), "image": img})
    return rows


def _load_omni() -> list[dict]:
    paths = glob.glob(os.path.join(DATA, "aa-omniscience/*.csv"))
    if not paths:
        raise FileNotFoundError("no AA-Omniscience csv pinned")
    rows = []
    with open(paths[0], encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            rows.append({"q": r.get("question"), "a": r.get("answer"),
                         "domain": r.get("domain"), "image": None})
    return rows


# ------------------------------------------------------------------ scoring
def score_arm(rows: list[dict], arm: str) -> tuple[float, dict]:
    """Return (accuracy, detail). Deterministic for every arm."""
    elig = [r for r in rows if norm(r.get("a")) != ""]
    n = len(elig)
    if n == 0:
        return float("nan"), {"n_eligible": 0, "reason": "no non-empty gold answers"}
    hit = 0
    for i, r in enumerate(elig):
        gold = norm(r["a"])
        if arm == "oracle_self":
            pred = gold
        elif arm == "abstain":
            pred = ""
        elif arm == "rotate_1":
            pred = norm(elig[(i + 1) % n]["a"])
        else:
            raise KeyError(arm)
        if pred == gold:
            hit += 1
    return hit / n, {"n_eligible": n, "n_excluded_empty_gold": len(rows) - n,
                     "n_hit": hit}


# ------------------------------------------------------------------ gates
GATES = [
    dict(id="G-PROXY-ORACLE",  fail_if="oracle_self accuracy < 1.0",
         why="a scorer that cannot be driven to 1.0 cannot detect a correct answer"),
    dict(id="G-PROXY-ABSTAIN", fail_if="abstain accuracy > 0.0",
         why="a scorer that credits an empty prediction is not measuring knowledge"),
    dict(id="G-PROXY-CONTROL", fail_if="rotate_1 accuracy > 0.05",
         why="a scorer near chance on shifted labels is not measuring anything specific"),
]


def run(eval_id: str) -> dict:
    rows = load(eval_id)
    arms = {}
    for arm in ("oracle_self", "abstain", "rotate_1"):
        acc, det = score_arm(rows, arm)
        arms[arm] = {"accuracy": acc, **det}

    checks = [
        dict(gate="G-PROXY-ORACLE",  observed=arms["oracle_self"]["accuracy"],
             passed=arms["oracle_self"]["accuracy"] == 1.0),
        dict(gate="G-PROXY-ABSTAIN", observed=arms["abstain"]["accuracy"],
             passed=arms["abstain"]["accuracy"] == 0.0),
        dict(gate="G-PROXY-CONTROL", observed=arms["rotate_1"]["accuracy"],
             passed=arms["rotate_1"]["accuracy"] <= 0.05),
    ]
    n_img = sum(1 for r in rows if r["image"] == "present")
    return {
        "schema": "henri.bench.local-proxy.v1",
        "written_utc": "2026-10-04",
        "eval_id": eval_id,
        "scorer": "normalized_exact_match",
        "scorer_is_lower_bound_of": (
            "AAII external equality-checker / grading model (GPT-5.6 Luna medium). "
            "Any model score computed with this scorer UNDERSTATES the official metric."),
        "n_items_total": len(rows),
        "n_items_with_image": n_img,
        "n_items_text_only": len(rows) - n_img,
        "arms": arms,
        "gates": checks,
        "gates_passed": all(c["passed"] for c in checks),
        "gates_failed": [c["gate"] for c in checks if not c["passed"]],
        "not_claimed": [
            "NOT an AAII v4.3.2 score.",
            "No model was run. Arms are deterministic controls, not a model.",
            "Normalized exact match is a STRICT LOWER BOUND on the official judge.",
        ],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True,
                    choices=["hle", "aa-omniscience", "both"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    targets = ["hle", "aa-omniscience"] if a.eval == "both" else [a.eval]
    results, ok = {}, True
    for e in targets:
        r = run(e)
        results[e] = r
        ok &= r["gates_passed"]
        print(f"=== {e} ===")
        print(f"  items: {r['n_items_total']} (text-only {r['n_items_text_only']}, "
              f"with image {r['n_items_with_image']})")
        for arm, d in r["arms"].items():
            print(f"  {arm:<12} acc={d['accuracy']:.4f}  n={d.get('n_eligible')} "
                  f"hits={d.get('n_hit')}")
        for c in r["gates"]:
            print(f"    [{'PASS' if c['passed'] else 'FAIL'}] {c['gate']} "
                  f"observed={c['observed']:.4f}")
        print()

    if a.out:
        payload = results if a.eval == "both" else results[targets[0]]
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(payload, fh, indent=2)
        print("wrote", a.out)
    print("SCORER VALIDATION:", "ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
