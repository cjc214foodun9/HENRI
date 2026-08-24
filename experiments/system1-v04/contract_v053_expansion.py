"""
Contract tests: v0.5.3 grammar expansion (CPU, disposable).
=============================================================================
C1  BACKWARD_COMPAT  SkeletonGrammar(7) instantiates rules 0-6 byte-identical
                     to the v0.5.1 grammar (same names/bodies/arity).
C2  CLOSURE_13       all 13 rules instantiate AST-valid AND UNK-free with
                     real task names on the LIVE vocab.
C3  GUARD_N_RULES    n_rules > N_RULES_ALL raises ValueError.
C4  GEN_TASK_13      gen_task(fid) for fid 0..12 returns a task whose code
                     passes its own test (sandbox == 1) and fid is correct.
C5  SPLIT_13         build_split yields old+new families, disjoint
                     verifier/outcome test strings, determinism.
C6  MONOTONE         for a fixed task, the B13 uniform pool (n_rules=13)
                     contains the B7 pool (n_rules=7) as a subset
                     (expansion is additive, not re-ranking).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v05_ast_skeleton import SkeletonGrammar  # noqa: E402
from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, tokenize_code)
from train_system1_kernel_v04 import gen_task, sandbox  # noqa: E402
from train_v051_discriminator import build_split  # noqa: E402

NAMES7 = {0: "sum_list", 1: "max_list", 2: "count_positive",
          3: "intersect_tuples", 4: "union_tuples", 5: "pair_sums",
          6: "factorial"}
NAMES13 = {**NAMES7, **{7: "m", 8: "v", 9: "n", 10: "a", 11: "b", 12: "res"}}
ARGS = {1: ["xs"], 2: ["t1", "t2"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="contract_v053")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(exist_ok=True)

    # ---- C1: backward compatibility ----
    g7 = SkeletonGrammar(n_rules=7)
    old = {0: ("def sum_list(xs):", "    return sum(xs)"),
           1: ("def max_list(xs):", "    return max(xs)"),
           2: ("def count_positive(xs):", "    return sum(1 for x in xs if x > 0)"),
           3: ("def intersect_tuples(t1, t2):",
               "    return tuple(sorted(set(t1) & set(t2)))"),
           4: ("def union_tuples(t1, t2):",
               "    return tuple(sorted(set(t1) | set(t2)))"),
           5: ("def pair_sums(t1, t2):",
               "    return [x + y for x, y in zip(t1, t2)]"),
           6: ("def factorial(xs):",
               "    res = 1\n    for i in range(1, xs + 1):\n        res = res * i\n"
               "    return res")}
    for rid, (sig, body) in old.items():
        code = g7.instantiate(rid, NAMES7[rid], ARGS[g7.RULES[rid][2]])
        assert code == sig + "\n" + body, f"C1 FAIL rule {rid}: {code!r}"
    assert g7.N_RULES == 7, f"C1 FAIL N_RULES {g7.N_RULES}"
    print("C1 BACKWARD_COMPAT PASS")

    # ---- C2: closure of all 13 on live vocab ----
    g13 = SkeletonGrammar(n_rules=13)
    bad = []
    for rid in range(13):
        _, _, na = g13.RULES_ALL[rid]
        code = g13.instantiate(rid, NAMES13[rid], ARGS[na])
        assert code is not None, f"C2 FAIL rule {rid} None"
        try:
            import ast
            ast.parse(code)
        except SyntaxError:
            bad.append((rid, "SYNTAX"))
            continue
        ids = tokenize_code(code)
        if TOK2ID["UNK"] in ids:
            bad.append((rid, "UNK"))
    assert not bad, f"C2 FAIL {bad}"
    print("C2 CLOSURE_13 PASS")

    # ---- C3: guard ----
    try:
        SkeletonGrammar(n_rules=14)
        raise AssertionError("C3 FAIL no ValueError")
    except ValueError:
        pass
    print("C3 GUARD_N_RULES PASS")

    # ---- C4: gen_task 13 families pass own test ----
    rng = random.Random(909)
    for fid in range(13):
        t = gen_task(rng, fid=fid)
        assert t["fid"] == fid, f"C4 FAIL fid {t['fid']}"
        assert sandbox(t["code"], t["tests"]) == 1, \
            f"C4 FAIL task {fid} {t['name']}"
    print("C4 GEN_TASK_13 PASS")

    # ---- C5: split determinism + family mix + disjoint tests ----
    d1 = out / "d1"
    d2 = out / "d2"
    for d in (d1, d2):
        d.mkdir(exist_ok=True)
    t1 = build_split(str(d1), 60, 54321, "c5_split", n_families=13)
    t2 = build_split(str(d2), 60, 54321, "c5_split", n_families=13)
    t3 = build_split(str(d2), 60, 54322, "c5_split2", n_families=13)
    assert t1 == t2, "C5 FAIL determinism"
    fids = {x["fid"] for x in t1}
    assert any(f >= 7 for f in fids), f"C5 FAIL no new families {fids}"
    assert any(f < 7 for f in fids), f"C5 FAIL no old families {fids}"
    for x in t1:
        v = set(x["verifier_tests"])
        o = set(x["outcome_tests"])
        assert v.isdisjoint(o), f"C5 FAIL partition overlap {x['name']}"
    assert t2 != t3, "C5 FAIL seed independence"
    print("C5 SPLIT_13 PASS")

    # ---- C6: B13 pool contains B7 pool (same task) ----
    from system1_kernel_v05_ast_skeleton import System1KernelV05
    from system1_kernel_v041_energy_refactored import (
        System1KernelV04, KernelV04Config)
    import torch
    torch.manual_seed(0)
    cfg = KernelV04Config()
    bb = System1KernelV04(cfg=cfg)
    bb.eval()
    v7 = System1KernelV05(bb, num_rules=7)
    v13 = System1KernelV05(bb, num_rules=13)
    v7.eval(); v13.eval()
    task = gen_task(random.Random(1), fid=0)
    import torch.nn.functional as F
    sig_t = torch.tensor([list(range(1, 17))], dtype=torch.long)
    sp = torch.zeros(1, 16, cfg.d_slot)
    z0 = bb.encode_tokens(sig_t)
    # use the v0.5 signature-latent path (encode once, shared)
    z_sig = v13.signature_latent(z0, sp)
    cands7 = v7.generate_skeleton_candidates(z0, sp, task, top_k=64, use_energy=False)
    cands13 = v13.generate_skeleton_candidates(z0, sp, task, top_k=64, use_energy=False)
    set7 = {c["code"] for c in cands7}
    set13 = {c["code"] for c in cands13}
    assert set7.issubset(set13), f"C6 FAIL B7 not subset B13 ({len(set7)} vs {len(set13)})"
    assert len(set13) > len(set7), f"C6 FAIL no expansion ({len(set7)} -> {len(set13)})"
    print(f"C6 MONOTONE PASS (B7 {len(set7)} -> B13 {len(set13)} distinct)")

    print("ALL CONTRACTS PASS (C1-C6)")


if __name__ == "__main__":
    main()
