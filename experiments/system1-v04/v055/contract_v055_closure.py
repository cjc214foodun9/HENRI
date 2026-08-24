"""
System-1 v0.5.5 SEMANTIC CLOSURE CONTRACT (run BEFORE sealing anything).
=============================================================================
Reference 3 mandate: grammar closure must be SEMANTIC, not merely syntactic.
For EVERY family (13): the canonical candidate (real generator-produced task
name + kernel arg-name scheme) must pass that family's verifier fixtures AND
disjoint outcome fixtures, with AST + FSA closure, non-empty pool.

Gates (all must pass for C1..C9):
  C1 pool non-empty for every rule
  C2 AST parse succeeds for every canonical candidate
  C3 tokenizer/FSA closure (no UNK) for every canonical candidate
  C4 canonical candidate passes VERIFIER fixtures (per family)
  C5 canonical candidate passes DISJOINT OUTCOME fixtures (per family)
  C6 minimum per-family support == 1.0 (all 13)
  C7 aggregate support == 1.0
  C8 rule 10 emits 'sum(range(len(' in its body
  C9 old 12-family behavior preserved: instantiation of rules 0-9,11,12
     is byte-identical between the v0.5.4 frozen kernel and the v0.5.5 carrier
=============================================================================
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import TOK2ID, tokenize_code  # noqa: E402
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    SkeletonGrammar as GrammarFrozen)
from system1_kernel_v055_ast_skeleton import (  # noqa: E402
    SkeletonGrammar as GrammarV055)
from train_system1_kernel_v04 import gen_task, sandbox  # noqa: E402
from train_v051_discriminator import (  # noqa: E402
    N_VERIFIER, N_OUTCOME, _rand_args, _expected, _args_key)

FAILS: list[str] = []


def _build_tests(rng: random.Random, fid: int, name: str,
                 nargs: int) -> dict:
    """Construct DISJOINT verifier/outcome fixtures exactly like build_split:
    4 verifier + 4 outcome tests, cross-boundary input uniqueness."""
    verifier_args, outcome_args = [], []
    seen_inputs: set = set()
    for _ in range(N_VERIFIER):
        a = _rand_args(rng, fid)
        while _args_key(a) in seen_inputs:
            a = _rand_args(rng, fid)
        seen_inputs.add(_args_key(a))
        verifier_args.append(a)
    for _ in range(N_OUTCOME):
        a = _rand_args(rng, fid)
        while _args_key(a) in seen_inputs:
            a = _rand_args(rng, fid)
        seen_inputs.add(_args_key(a))
        outcome_args.append(a)
    tests = []
    for args_list in verifier_args + outcome_args:
        exp = _expected(fid, args_list)
        if nargs == 1:
            tests.append(f"assert {name}({args_list[0]}) == {exp}")
        else:
            tests.append(
                f"assert {name}({tuple(args_list[0])}, "
                f"{tuple(args_list[1])})"
                f" == {tuple(exp) if isinstance(exp, tuple) else exp}")
    return {"verifier_tests": tests[:N_VERIFIER],
            "outcome_tests": tests[N_VERIFIER:]}


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{name}: {'PASS' if ok else 'FAIL'}"
          + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILS.append(name)


def main() -> None:
    g_frozen = GrammarFrozen(n_rules=13)
    g_v055 = GrammarV055(n_rules=13)
    rng = random.Random(20260824)

    fam_support = {}
    for fid in range(13):
        t = gen_task(rng, fid=fid)
        name, nargs = t["name"], t["nargs"]
        fixtures = _build_tests(rng, fid, name, nargs)
        # kernel arg-name scheme (identical to System1KernelV05 generator)
        arg_names = ["xs", "t1", "t2"][:nargs] if nargs <= 2 else \
            ["xs", "ys", "zs"][:nargs]
        code_frozen = g_frozen.instantiate(fid, name, arg_names)
        code_v055 = g_v055.instantiate(fid, name, arg_names)

        # C9: old 12-family byte-identity
        if fid == 10:
            check(f"C9_fid{fid}_diff", code_frozen != code_v055,
                  "rule-10 bodies MUST differ")
        else:
            check(f"C9_fid{fid}_same", code_frozen == code_v055,
                  "other 12 bodies MUST be byte-identical")

        # C1: non-empty
        check(f"C1_fid{fid}_pool", code_v055 is not None, name)

        # C2: AST
        ok_ast = False
        try:
            import ast
            ast.parse(code_v055 or "")
            ok_ast = True
        except Exception:
            ok_ast = False
        check(f"C2_fid{fid}_ast", ok_ast, name)

        # C3: FSA closure
        ids = tokenize_code(code_v055 or "")
        ok_fsa = TOK2ID["UNK"] not in ids
        check(f"C3_fid{fid}_fsa", ok_fsa, name)

        # C4: verifier fixtures (disjoint from outcome)
        ok_ver = sandbox(code_v055 or "",
                         fixtures["verifier_tests"]) == 1
        check(f"C4_fid{fid}_verifier", ok_ver, name)

        # C5: outcome fixtures (disjoint)
        ok_out = sandbox(code_v055 or "",
                         fixtures["outcome_tests"]) == 1
        check(f"C5_fid{fid}_outcome", ok_out, name)

        fam_support[fid] = 1 if (ok_ver and ok_out) else 0

    # C8: rule-10 body
    code10 = g_v055.instantiate(10, "range_sum", ["xs"])
    check("C8_rule10_body", "sum(range(len(" in (code10 or ""),
          repr(code10))

    # C6/C7: per-family + aggregate support
    check("C6_min_family_support_1.0", all(v == 1 for v in fam_support.values()),
          f"support={fam_support}")
    check("C7_aggregate_support_1.0",
          sum(fam_support.values()) == 13,
          f"{sum(fam_support.values())}/13")

    if FAILS:
        print(f"\nCLOSURE_CONTRACT: {len(FAILS)} FAIL(S): {FAILS}")
        raise SystemExit(1)
    print("\nCLOSURE_CONTRACT: ALL PASS (13/13 semantic closure)")


if __name__ == "__main__":
    main()
