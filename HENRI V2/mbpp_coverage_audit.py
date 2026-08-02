"""Coverage audit for the 485 CEGIS_MISS items (run17 verdict follow-up).

Verdict routing (pre-registered, Egress and ast dev tree.pdf): when
COVERAGE_MISS dominates, isolate the failure to the discrete egress
boundary and justify a CONTROLLED grammar expansion. This audit makes
the expansion quantitative: for every miss, classify the canonical
solution's structural features (arity, call names, control flow,
comprehensions) and test in-grammar expressibility against the live
enumerator. Output: feature histograms + coverage counts. CPU-only,
no wave encoding, no CUDA.
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from mbpp_cegis_synthesizer import parse_entry_from_tests, parse_entry_signature
from mbpp_heldout_pilot import load_items, render_prompt
from mbpp_rank_probe import canonical_key, canonical_signature
from mbpp_wave_ast_decoder import WaveASTDecoder


def feature_set(code: str) -> dict[str, Any]:
    """Structural feature summary of the canonical solution."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {"arity": -1, "calls": [], "control": [], "comprehensions": 0,
                "lambda": 0, "binops": [], "strings": []}
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
    calls: list[str] = []
    control: list[str] = []
    comps = 0
    lams = 0
    binops: list[str] = []
    str_methods: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name):
                calls.append(f.id)
            elif isinstance(f, ast.Attribute):
                calls.append(f.attr)
        elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try, ast.Return)):
            control.append(type(node).__name__)
        elif isinstance(node, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
            comps += 1
        elif isinstance(node, ast.Lambda):
            lams += 1
        elif isinstance(node, ast.BinOp):
            binops.append(type(node.op).__name__)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            str_methods.append(node.attr)
    return {"arity": len(fn.args.args), "calls": calls, "control": control,
            "comprehensions": comps, "lambda": lams, "binops": binops,
            "str_methods": str_methods}


def main() -> int:
    failures_path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if failures_path is None:
        print("usage: mbpp_coverage_audit.py <item_results.jsonl>")
        return 2
    rows = []
    with open(failures_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    misses = [r for r in rows if str(r.get("failure_reason", "")).startswith("CEGIS")]
    items = {int(i["task_id"]): i for i in load_items()}
    decoder = WaveASTDecoder(None)  # _instantiate needs no codec

    n_total = n_in_grammar = 0
    arity = Counter()
    calls = Counter()
    control = Counter()
    comps_used = 0
    lam_used = 0
    top_arity_miss = Counter()
    top_call_miss = Counter()
    in_grammar_ids: list[int] = []

    for r in misses:
        tid = int(r["task_id"])
        it = items.get(tid)
        if it is None:
            continue
        code = it.get("code", "")
        if not code:
            continue
        n_total += 1
        feat = feature_set(code)
        arity[feat["arity"]] += 1
        for c in feat["calls"]:
            calls[c] += 1
        for c in feat["control"]:
            control[c] += 1
        comps_used += 1 if feat["comprehensions"] else 0
        lam_used += 1 if feat["lambda"] else 0

        sig = parse_entry_signature(render_prompt(it)) or parse_entry_from_tests(
            it.get("test_list") or [])
        if sig is None:
            continue
        entry, args = sig
        key = canonical_key(code, entry)
        if key is None:
            continue
        bodies = decoder._instantiate(entry, args)
        hit = False
        for b in bodies:
            src = f"def {entry}({', '.join(args)}):\n{b}"
            cand_key = canonical_key(src, entry)
            if cand_key is not None and cand_key == key:
                hit = True
                break
        if hit:
            n_in_grammar += 1
            in_grammar_ids.append(tid)
        else:
            top_arity_miss[feat["arity"]] += 1
            for c in feat["calls"][:3]:
                top_call_miss[c] += 1

    print(f"misses_total={n_total} in_grammar={n_in_grammar} coverage_gap={n_total - n_in_grammar}")
    print("in_grammar_ids=", in_grammar_ids[:30])
    print("arity_hist=", dict(arity.most_common()))
    print("control_hist=", dict(control.most_common()))
    print("comprehensions_used=", comps_used, "lambda_used=", lam_used)
    print("top_calls=", dict(calls.most_common(15)))
    print("arity_gap=", dict(top_arity_miss.most_common()))
    print("call_gap=", dict(top_call_miss.most_common(15)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
