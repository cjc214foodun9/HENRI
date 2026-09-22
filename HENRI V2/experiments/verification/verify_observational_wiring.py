#!/usr/bin/env python3
"""Verify the observational wiring: names bound, default-off byte-identical, A/B.

THE MISSING-CHECK LESSON
    An earlier patch referenced `os.environ` with no `import os`, and BOTH
    `py_compile` and `ast.parse` passed, because neither resolves names. So this
    script must (a) import the module, (b) touch every new name, and (c) actually
    CALL the new path. Parse-time checks are reported but never trusted alone.

GATES
  W1 names bound        module imports; observational_tau and _demo_observational_stress
                        exist and are callable; `os` is bound
  W2 no target leak     search() still has no target_grid; the new method never reads
                        a held-out answer (AST scan of the scoring region)
  W3 DEFAULT OFF        HENRI_SAGNAC_OBSERVATIONAL_VETO unset -> search() returns the
                        SAME op and delta as before the wiring
  W4 FLAG ON            flag set -> the observational channel runs and does not raise
  W5 tau per lattice    observational_tau differs by lattice size and is NOT 0.35
  W6 fail-open          no demo_pairs -> channel reports valid=False; nothing vetoed
  W7 shape mismatch     a program changing the lattice size scores stress 1.0
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

OUT = os.path.join(HERE, "observational_wiring_verified.json")


def main() -> int:
    from sagnac_mcts_planner import SagnacMCTSPlanner, SpelkeDSLNode

    res: dict = {}
    fails: list = []

    # ------------------------------------------------------------------ W1
    import sagnac_mcts_planner as mod
    res["W1_os_bound"] = hasattr(mod, "os")
    res["W1_has_observational_tau"] = hasattr(
        SagnacMCTSPlanner, "observational_tau")
    res["W1_has_demo_stress"] = hasattr(
        SagnacMCTSPlanner, "_demo_observational_stress")
    for k, v in (("W1_os_bound", res["W1_os_bound"]),
                 ("W1_has_observational_tau", res["W1_has_observational_tau"]),
                 ("W1_has_demo_stress", res["W1_has_demo_stress"])):
        if not v:
            fails.append(f"{k} is False; a referenced name is not bound")

    # ------------------------------------------------------------------ W5
    p = SagnacMCTSPlanner(d_model=1024, k_blocks=128, tau_veto=0.35, device="cpu")
    taus = {n: p.observational_tau(n) for n in (16, 64, 256, 900)}
    res["W5_tau_by_lattice"] = taus
    res["W5_tau_not_0p35"] = all(abs(v - 0.35) > 0.1 for v in taus.values())
    res["W5_tau_increases_with_size"] = (taus[16] < taus[64] < taus[256] < taus[900])
    if not res["W5_tau_not_0p35"]:
        fails.append(f"observational_tau returned 0.35 for some lattice: {taus}")
    if not res["W5_tau_increases_with_size"]:
        fails.append(f"tau did not increase with lattice size: {taus}")

    # ------------------------------------------------------------------ W2
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    methods = {}
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for fn in [n for n in cls.body if isinstance(n, ast.FunctionDef)]:
            methods.setdefault(fn.name, []).append(fn)
    search_args = ([a.arg for a in methods["search"][0].args.args]
                   + [a.arg for a in methods["search"][0].args.kwonlyargs])
    res["W2_search_has_no_target_grid"] = "target_grid" not in search_args
    if not res["W2_search_has_no_target_grid"]:
        fails.append(f"search() regained target_grid: {search_args}")
    # The NEW method must not reference the held-out output under any name.
    demo_fn = methods["_demo_observational_stress"][0]
    demo_names = {n.id for n in ast.walk(demo_fn) if isinstance(n, ast.Name)}
    res["W2_demo_method_names"] = sorted(demo_names)
    leaked = [n for n in demo_names if "target" in n.lower()]
    res["W2_no_target_in_demo_method"] = not leaked
    if leaked:
        fails.append(f"the observational method references {leaked}; it must use "
                     f"demonstrations only")

    # ------------------------------------------------------------------ W3/W4
    grid = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    demos = [(np.array([[1, 2], [3, 4]]), np.array([[3, 1], [4, 2]]))]

    def run(flag: str | None) -> dict:
        env = dict(os.environ)
        env.pop("HENRI_SAGNAC_OBSERVATIONAL_VETO", None)
        if flag is not None:
            env["HENRI_SAGNAC_OBSERVATIONAL_VETO"] = flag
        code = (
            "import sys, json, numpy as np;"
            f"sys.path.insert(0, r'{ROOT}');"
            "from sagnac_mcts_planner import SagnacMCTSPlanner;"
            "p=SagnacMCTSPlanner(d_model=1024,k_blocks=128,tau_veto=0.35,device='cpu');"
            "g=np.array([[1,2,3],[4,5,6],[7,8,9]]);"
            "d=[(np.array([[1,2],[3,4]]),np.array([[3,1],[4,2]]))];"
            "prog,delta=p.search(g,num_simulations=4,demo_pairs=d);"
            "print(json.dumps({'op':getattr(prog,'op_name',str(prog)),"
            "'delta':float(delta)}))"
        )
        cp = subprocess.run([sys.executable, "-c", code], capture_output=True,
                            text=True, env=env, timeout=900)
        line = [l for l in cp.stdout.splitlines() if l.strip().startswith("{")]
        if not line:
            return {"error": cp.stderr[-500:]}
        return json.loads(line[-1])

    base = run(None)
    off = run("0")
    on = run("1")
    res["W3_default_unset"] = base
    res["W3_flag_0"] = off
    res["W4_flag_1"] = on
    if "error" in base:
        fails.append(f"W3: default run raised: {base['error']}")
    if "error" in on:
        fails.append(f"W4: flag-on run raised: {on['error']}")
    res["W3_default_matches_flag_0"] = (base == off)
    if "error" not in base and "error" not in off and base != off:
        fails.append(f"W3: default {base} != flag-0 {off}; the default path changed")
    if "error" not in on and on.get("op") is None:
        fails.append("W4: flag-on produced no op")

    # ------------------------------------------------------------------ W6
    stress, tau, valid = p._demo_observational_stress(
        SpelkeDSLNode(op_name="Identity"), None, grid)
    res["W6_no_demos_valid"] = bool(valid)
    res["W6_no_demos_stress"] = stress
    if valid:
        fails.append("W6: the channel reported valid=True with no demonstration pairs; "
                     "fail-open is broken")

    # ------------------------------------------------------------------ W7
    prog = SpelkeDSLNode(op_name="Rotate90")
    s1, t1, v1 = p._demo_observational_stress(prog, demos, grid)
    res["W7_shape_ok_stress"] = s1
    res["W7_shape_ok_valid"] = bool(v1)
    # A program that reshapes the lattice must score a total mismatch, not raise.
    class _Reshaper(SpelkeDSLNode):
        def execute(self, g):
            return np.zeros((2, 3), dtype=np.int64)

    s2, t2, v2 = p._demo_observational_stress(_Reshaper(op_name="Identity"), demos, grid)
    res["W7_shape_mismatch_stress"] = s2
    res["W7_shape_mismatch_valid"] = bool(v2)
    if v2 and s2 != 1.0:
        fails.append(f"W7: lattice-changing program scored {s2}, expected 1.0")

    # ------------------------------------------------------------------ W8
    # IS THE CHANNEL ACTUALLY LIVE? W4 was a weak gate: flag=1 returned the SAME
    # program as flag=0, which is consistent with the channel working AND with the
    # channel being dead code. A flag that changes nothing observable is exactly the
    # "phantom flag" failure this project audits for, so the channel must be shown to
    # make real decisions: it must VETO some programs and ADMIT others.
    demo = (np.array([[1, 2], [3, 4]]), np.array([[3, 1], [4, 2]]))
    decisions = []
    for op in p.primitive_ops:
        st, tau, ok = p._demo_observational_stress(
            SpelkeDSLNode(op_name=op), [demo], grid)
        decisions.append({
            "op": op,
            "stress": None if st is None else round(st, 4),
            "tau": None if tau is None else round(tau, 4),
            "valid": bool(ok),
            "vetoed": bool(ok and st > tau),
        })
    res["W8_channel_decisions"] = decisions
    n_valid = sum(1 for d in decisions if d["valid"])
    n_vetoed = sum(1 for d in decisions if d["vetoed"])
    n_admitted = n_valid - n_vetoed
    res["W8_n_valid"] = n_valid
    res["W8_n_vetoed"] = n_vetoed
    res["W8_n_admitted"] = n_admitted
    res["W8_channel_is_discriminative"] = n_vetoed > 0 and n_admitted > 0
    if not res["W8_channel_is_discriminative"]:
        fails.append(
            f"W8: the observational channel vetoed {n_vetoed} and admitted "
            f"{n_admitted} programs; it must do BOTH or it cannot be a gate. "
            f"Decisions: {decisions}")
    # The program that EXACTLY reproduces the demo must be admitted.
    rot = next((d for d in decisions if d["op"] == "Rotate90"), None)
    res["W8_rotate90_admitted"] = bool(rot and rot["valid"] and not rot["vetoed"])
    if not res["W8_rotate90_admitted"]:
        fails.append(f"W8: Rotate90 reproduces the demo exactly but was not admitted: {rot}")

    verdict = "PASS" if not fails else "FAIL"
    out = {"module": "observational_wiring", "evidence_class": "OBSERVED",
           "results": res, "gate_failures": fails, "verdict": verdict,
           "note": ("Names were verified by IMPORTING and CALLING, not by parsing: "
                    "py_compile and ast.parse both pass on a module with unbound "
                    "globals, which is how a missing `import os` survived earlier.")}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)

    print("=" * 80)
    print("OBSERVATIONAL WIRING VERIFICATION")
    print("=" * 80)
    print(f"  W1 names bound: os={res['W1_os_bound']} tau={res['W1_has_observational_tau']} "
          f"stress={res['W1_has_demo_stress']}")
    print(f"  W2 search has no target_grid = {res['W2_search_has_no_target_grid']}")
    print(f"     demo method names = {res['W2_demo_method_names']}")
    print(f"  W5 tau by lattice = {taus}")
    print(f"  W3 default (unset) = {base}")
    print(f"     flag=0          = {off}")
    print(f"     default == flag0: {res['W3_default_matches_flag_0']}")
    print(f"  W4 flag=1          = {on}")
    print(f"  W6 no demos -> valid={valid} (must be False)")
    print(f"  W7 demo stress (Rotate90) = {s1} valid={v1}")
    print(f"     lattice-changing stress = {s2} valid={v2}")
    print()
    if fails:
        print("GATE FAILURES:")
        for f_ in fails:
            print(f"  - {f_}")
    print(f"VERDICT: {verdict}")
    print(f"wrote {OUT}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
