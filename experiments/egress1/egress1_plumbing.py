"""
Egress-1 disposable plumbing: semantic closure + sandbox + tokenizer closure.
==============================================================================
Disposable tasks ONLY (tag smoke_egress1_*, seed 90001, 1 per family = 13 tasks).
The sealed heldout54_egress1 split is NEVER loaded here.
Checks:
  1. Nonempty pools: build_split_stratified returns 13 tasks, 1 per family.
  2. Tokenizer closure: canonical candidates contain no UNK tokens.
  3. Semantic closure: each family's CANONICAL candidate passes its own
     verifier tests (4) AND outcome tests (4) in the sandbox.
  4. Sandbox: container-rlimit mode executes and returns 1 for passing code.
Reference 3: grammar closure must be SEMANTIC, not merely syntactic.
"""
from __future__ import annotations

import json, pathlib, random, sys

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from eval_v055_heldout import build_split_stratified
from train_system1_kernel_v04 import gen_task, sandbox
from system1_kernel_v041_energy_refactored import TOK2ID, tokenize_code

OUT = _HERE / "egress1_plumbing"
OUT.mkdir(exist_ok=True)

def main() -> int:
    # 1. disposable split (13 tasks, 1/family) — NEVER the sealed split
    tasks = build_split_stratified(str(OUT), 13, 90001, "smoke_egress1_disposable", n_families=13)
    fams = sorted({t["fid"] for t in tasks})
    print("POOL_NONEMPTY", len(tasks) == 13, "FAMS", len(fams) == 13)

    # 2+3+4. per-family semantic closure with canonical candidates
    results = {}
    all_pass = True
    for fid in range(13):
        t = gen_task(random.Random(fid), fid=fid)  # canonical task for family fid
        code = t["code"]
        # tokenizer closure
        unk = "UNK" in tokenize_code(code) or TOK2ID["UNK"] in tokenize_code(code)
        # use the split task's fixtures for THIS family
        st = next(x for x in tasks if x["fid"] == fid)
        vt, ot = st["verifier_tests"], st["outcome_tests"]
        v_pass = sandbox(code, vt) == 1
        o_pass = sandbox(code, ot) == 1
        ok = (not unk) and v_pass and o_pass
        all_pass = all_pass and ok
        results[fid] = {"unk": bool(unk), "verifier": bool(v_pass), "outcome": bool(o_pass), "ok": bool(ok)}
        print(f"F{fid} unk={unk} verifier={v_pass} outcome={o_pass} ok={ok}")

    with open(OUT / "plumbing_results.json", "w") as f:
        json.dump({"all_pass": all_pass, "per_family": results}, f, indent=1)
    print("ALL_PASS", all_pass)
    return 0 if all_pass else 1

if __name__ == "__main__":
    raise SystemExit(main())
