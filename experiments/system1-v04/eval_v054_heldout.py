"""
System-1 v0.5.4 heldout verification: Arm A (token beam) vs Arm B13
(13-rule uniform CEGIS-first) on a FRESH single-use heldout split.
=============================================================================
Pre-registered 2026-08-24 (Option B authorized by user; references
system1-structural-egress-cegis.md; heldout-sealing + grammar-expansion
sections of henri-research).

Decision: HELDOUT_13_RULE_CARRIER_PROMOTED requires ALL of
  (1) efficacy: B13 > A with McNemar p < 0.05, delta >= +0.10,
      task-blocked CI lb > 0 on DISJOINT outcome tests;
  (2) validity: admitted-program AST validity >= 0.95 BOTH arms
      (admitted programs as denominator; non-admission is NOT invalidity);
  (3) integrity: fresh seal (single_use=true), pinned sha, no leakage.
Else: delta > 0 -> CONDITIONAL_VERIFIER_ASSISTED_IMPROVEMENT,
      delta <= 0 -> HELDOUT_PROMOTION_NOT_ESTABLISHED,
      guard/leak  -> INVALID_VERIFIER_REPLAY (raised at startup).

Attribution boundary (pre-registered):
- This run promotes the 13-RULE CARRIER (A vs B13).
- B7 is a DIAGNOSTIC arm only. B13-vs-B7 incremental attribution is NOT
  claimed from this run (it remains dev-supported by v0.5.3 dev3_v053).
- A vs B13 cannot isolate grammar-expansion attribution alone; old/new
  family rates are reported as stratified diagnostics, not proof.

Frozen implementation set (sha256, recorded BEFORE seal):
- system1_kernel_v05_ast_skeleton.py     a237d4239f256c34  (13-rule grammar)
- system1_kernel_v041_energy_refactored.py 754cbe4910e35c8f (backbone)
- system1_kernel_v042_cegis_beam.py      057e0ec1f39c58cf  (token beam)
- train_system1_kernel_v04.py            35025cfa5d40819d  (gen_task/sandbox)
- train_v051_discriminator.py            dd596e93327fb994  (split builder)
- v041_energy_checkpoint.pt              11d56121e4b091e2  (frozen ckpt)

Integrity contract:
- CONSUMED_DIGESTS includes every smoke/contract/dev/train/heldout digest
  (incl. 9a17af61 dev3_v053, 635c2aaa smoke53, 888809df/8ea34261 c5
  contract splits). Any match is refused as INVALID_VERIFIER_REPLAY.
- The heldout split is sealed ONCE (--seal-only, no checkpoint allowed),
  stratified: exactly 4 tasks per family x 13 families = 52 tasks.
- CUDA run loads the sealed split and verifies the pinned sha.
- No CPU smoke or preflight may load the heldout split; smoke uses a
  disposable tag/seed only.
- Arms use DISJOINT verifier/outcome tests (4+4 per task, input-uniqueness
  partition guaranteed by build_split stratify path).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import random
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04, detokenize, KernelV04Config, tokenize_code)
from system1_kernel_v042_cegis_beam import (  # noqa: E402
    CEGISBeamPriorityDecoder)
from system1_kernel_v05_ast_skeleton import System1KernelV05  # noqa: E402
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from train_v051_discriminator import (  # noqa: E402
    build_split, sha256_file, N_VERIFIER, N_OUTCOME, _rand_args, _expected,
    _args_key)


def build_split_stratified(out_dir: str, n_tasks: int, seed: int, tag: str,
                           n_families: int = 13) -> list[dict]:
    """Deterministic multi-test split with EXACT family stratification.

    n_tasks must be divisible by n_families; every family gets exactly
    n_tasks // n_families tasks. Family order is deterministically shuffled
    by the seeded rng. Test construction is IDENTICAL to
    train_v051_discriminator.build_split (same _rand_args/_expected/
    _args_key), so partition semantics (4 verifier + 4 outcome tests,
    cross-boundary input uniqueness) are preserved exactly.
    """
    per_family = n_tasks // n_families
    if per_family * n_families != n_tasks:
        raise SystemExit(
            f"STRATIFY: n_tasks {n_tasks} not divisible by "
            f"n_families {n_families}")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{tag}.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    rng = random.Random(seed)
    fam_slots = [f for f in range(n_families) for _ in range(per_family)]
    rng.shuffle(fam_slots)
    tasks = []
    for i in range(n_tasks):
        fid = fam_slots[i]
        t = gen_task(rng, fid=fid)
        name, nargs = t["name"], t["nargs"]
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
        t["tests"] = tests          # 8 tests: [0:4]=verifier, [4:8]=outcome
        t["verifier_tests"] = tests[:N_VERIFIER]
        t["outcome_tests"] = tests[N_VERIFIER:]
        t["verifier_args"] = verifier_args
        t["outcome_args"] = outcome_args
        tasks.append(t)
    with open(p, "w") as f:
        json.dump(tasks, f, indent=1)
    return tasks

CONSUMED_DIGESTS = [
    "887d0d6c",  # smoke40_v04 (v0.4 era; quarantined)
    "23b36795",  # heldout40_v04 (consumed v0.4 heldout)
    "82c97532",  # dev42_v04
    "db027f9c",  # dev43_v04
    "9d4c29ad",  # dev50_v05
    "1f81e4d0",  # dev2_v051 (trained + evaluated)
    "181cc59b",  # train_v051 split
    "092cb0c1",  # dev41_v04 (plumb)
    "2eb8d29b",  # dev41_v04 (telemetry)
    "78b4cfb4",  # dev42_v04 (plumb)
    "7a8c1e7b",  # dev50_smoke
    "306ab62d",  # dev51_smoke
    "6b5bb1b4",  # train51_smoke
    "0ec0528d",  # dev43_smoke
    "0535a2dc",  # contract c1 split
    "35d15aae",  # contract c1_split2
    "5e5f4a00",  # heldout51_v052 (consumed v0.5.2 heldout)
    "ce2a76fb",  # smoke52_disposable
    "9a17af61",  # dev3_v053 (v0.5.3 grammar-expansion dev split)
    "635c2aaa",  # smoke53_disposable
    "888809df",  # contract c5_split (v0.5.3)
    "8ea34261",  # contract c5_split2 (v0.5.3)
]


def _mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return 2 * min(
        sum(math.comb(n, k) * 0.5 ** n for k in range(0, min(b, c) + 1)),
        sum(math.comb(n, k) * 0.5 ** n for k in range(max(b, c), n + 1)))


def _task_bootstrap_cis(vals: list[float], n_rep: int = 2000,
                        seed: int = 0, alpha: float = 0.1) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(vals)
    means = []
    for _ in range(n_rep):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_rep)]
    hi = means[int((1 - alpha / 2) * n_rep) - 1]
    return lo, hi


def _cegis_first(pool: list[tuple], verifier_tests: list[str],
                 budget: int = 64) -> tuple[int, int]:
    """Scan pool in order, run verifier tests only. Return (admit_index or -1,
    calls_used). sandbox returns 1 iff ALL given tests pass."""
    for i, (code, _order) in enumerate(pool[:budget]):
        if sandbox(code, verifier_tests) == 1:
            return i, i + 1
    return -1, len(pool[:budget])


def _ast_ok(code: str) -> int:
    return 1 if TOK2ID["UNK"] not in tokenize_code(code) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=52)
    ap.add_argument("--dev-seed", type=int, default=99991)
    ap.add_argument("--tag", default="heldout52_v054")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--seal-only", action="store_true",
                    help="generate + seal the split and exit (no eval)")
    ap.add_argument("--expect-sha", default="",
                    help="require the loaded split to match this sha256 "
                         "prefix (integrity pin for the CUDA run)")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    # ---- split: load sealed file if present, else generate (stratified) ----
    split_p = out / f"{args.tag}.json"
    if split_p.exists():
        sd = sha256_file(split_p)
        tasks = json.loads(split_p.read_text())
    else:
        tasks = build_split_stratified(
            args.out, args.dev_n, args.dev_seed, args.tag, n_families=13)
        sd = sha256_file(split_p)

    if any(sd.startswith(d) for d in CONSUMED_DIGESTS):
        raise SystemExit(
            f"INVALID_VERIFIER_REPLAY: split {args.tag} matches a consumed "
            f"digest {sd[:12]}. REFUSED.")
    if args.expect_sha and not sd.startswith(args.expect_sha):
        raise SystemExit(
            f"SPLIT_MISMATCH: loaded split sha {sd[:16]} does not match "
            f"pinned {args.expect_sha}. REFUSED.")

    if args.seal_only:
        if args.ckpt:
            raise SystemExit("SEAL_ONLY: --ckpt not allowed")
        fam_counts: dict[int, int] = {}
        for t in tasks:
            fam_counts[t["fid"]] = fam_counts.get(t["fid"], 0) + 1
        receipt = {
            "tag": args.tag, "n": len(tasks), "seed": args.dev_seed,
            "sha256": sd, "utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "phase": "heldout", "single_use": True,
            "stratified_families": fam_counts,
            "verifier_per_task": N_VERIFIER, "outcome_per_task": N_OUTCOME,
            "generator": "eval_v054_heldout.build_split_stratified "
                         "(13 families x 4 = 52, seeded family shuffle)",
            "evaluator_sha256": sha256_file(
                str(_HERE / "eval_v054_heldout.py")),
        }
        rec_p = out / f"seal_{args.tag}.json"
        rec_p.write_text(json.dumps(receipt, indent=2))
        print(f"SEALED {args.tag} n={len(tasks)} seed={args.dev_seed} "
              f"sha={sd}", flush=True)
        print("SEAL_RECEIPT " + str(rec_p), flush=True)
        return

    # ---- full evaluation path ----
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05_13.eval()
    v05_7 = System1KernelV05(backbone, num_rules=7).to(dev)
    v05_7.eval()
    dec = CEGISBeamPriorityDecoder(backbone)

    trainable_backbone = [n for n, p in backbone.named_parameters()
                          if p.requires_grad]
    if trainable_backbone:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable_backbone}")

    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)
    print(f"HELDOUT {args.tag} n={len(tasks)} seed={args.dev_seed} "
          f"sha={sd[:16]}", flush=True)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    calls = {"A": [], "B13": [], "B7": []}
    outcome_pass = {"A": [], "B13": [], "B7": []}
    ast_valid = {"A": [], "B13": [], "B7": []}
    oracle_support = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t), "fid": t["fid"]}
        ver_tests = t["verifier_tests"]
        out_tests = t["outcome_tests"]

        # ---- ARM A: token beam + CEGIS-first (verifier tests) ----
        _, rec_a = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=args.beam_width,
            beta_priority=0.0, return_all_finals=True)
        pool_a = []
        seen = set()
        for s, sc, _e in rec_a["final_candidates"]:
            code = detokenize(s)
            if code in seen:
                continue
            seen.add(code)
            pool_a.append((code, sc))
        idx_a, call_a = _cegis_first(pool_a, ver_tests, budget=args.budget)
        calls["A"].append(call_a)
        if idx_a >= 0:
            code_a = pool_a[idx_a][0]
            outcome_pass["A"].append(sandbox(code_a, out_tests))
            ast_valid["A"].append(_ast_ok(code_a))
        else:
            outcome_pass["A"].append(0)
            ast_valid["A"].append(0)

        # ---- ARM B13: 13-rule skeleton pool, uniform generator order ----
        cands13 = v05_13.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget, use_energy=False)
        pool13 = [(c["code"], c["rule_id"]) for c in cands13]
        if pool13:
            idx13, call13 = _cegis_first(pool13, ver_tests,
                                         budget=args.budget)
        else:
            idx13, call13 = -1, 0
        calls["B13"].append(call13)
        if idx13 >= 0:
            code13 = pool13[idx13][0]
            outcome_pass["B13"].append(sandbox(code13, out_tests))
            ast_valid["B13"].append(_ast_ok(code13))
        else:
            outcome_pass["B13"].append(0)
            ast_valid["B13"].append(0)

        # ---- ARM B7 (DIAGNOSTIC ONLY): 7-rule skeleton pool ----
        cands7 = v05_7.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget, use_energy=False)
        pool7 = [(c["code"], c["rule_id"]) for c in cands7]
        if pool7:
            idx7, call7 = _cegis_first(pool7, ver_tests, budget=args.budget)
        else:
            idx7, call7 = -1, 0
        calls["B7"].append(call7)
        if idx7 >= 0:
            code7 = pool7[idx7][0]
            outcome_pass["B7"].append(sandbox(code7, out_tests))
            ast_valid["B7"].append(_ast_ok(code7))
        else:
            outcome_pass["B7"].append(0)
            ast_valid["B7"].append(0)

        sup = any(sandbox(code, out_tests) for code, _r in pool13)
        oracle_support.append(1 if sup else 0)

        row["A"] = {"admit": idx_a, "calls": call_a,
                    "outcome": outcome_pass["A"][-1],
                    "ast_valid": ast_valid["A"][-1], "pool": len(pool_a)}
        row["B13"] = {"admit": idx13, "calls": call13,
                      "outcome": outcome_pass["B13"][-1],
                      "ast_valid": ast_valid["B13"][-1], "pool": len(pool13)}
        row["B7"] = {"admit": idx7, "calls": call7,
                     "outcome": outcome_pass["B7"][-1],
                     "ast_valid": ast_valid["B7"][-1], "pool": len(pool7)}
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{len(tasks)}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)

    def rate(arm: str) -> float:
        return sum(outcome_pass[arm]) / n

    def ci(arm: str) -> tuple[float, float]:
        return _task_bootstrap_cis([float(v) for v in outcome_pass[arm]],
                                   seed=args.dev_seed + 7)

    def call_stats(arm: str) -> dict:
        v = sorted(calls[arm])
        m = len(v)
        return {"mean": sum(v) / m,
                "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0,
                "max": v[-1] if m else 0}

    def budget_success(arm: str, K: int) -> float:
        ok = 0
        for i, r in enumerate(per_task):
            if r[arm]["admit"] >= 0 and r[arm]["admit"] < K \
                    and r[arm]["outcome"] == 1:
                ok += 1
        return ok / n

    def ast_rate(arm: str) -> float:
        """AST validity over ADMITTED programs only. Non-admitted tasks are
        excluded (no program was produced); they must NOT count as invalid."""
        vals = [r[arm]["ast_valid"] for r in per_task
                if r[arm]["admit"] >= 0]
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def family_rates(arm: str) -> dict:
        fam: dict[int, list[int]] = {}
        for r in per_task:
            fam.setdefault(r["fid"], []).append(r[arm]["outcome"])
        return {str(f): sum(v) / len(v) for f, v in sorted(fam.items())}

    def paired(arm_x: str, arm_y: str) -> dict:
        both = x_only = y_only = neither = 0
        for i in range(n):
            xv = outcome_pass[arm_x][i]
            yv = outcome_pass[arm_y][i]
            if xv and yv:
                both += 1
            elif xv:
                x_only += 1
            elif yv:
                y_only += 1
            else:
                neither += 1
        return {"both": both, f"{arm_x}_only": x_only,
                f"{arm_y}_only": y_only, "neither": neither,
                "mcnemar_p": _mcnemar_two_sided(x_only, y_only)}

    # ---- heldout verdict (pre-registered) ----
    paired_BA = paired("B13", "A")
    delta_BA = rate("B13") - rate("A")
    delta_ci_BA = _task_bootstrap_cis(
        [b13 - a for b13, a in zip(outcome_pass["B13"], outcome_pass["A"])],
        seed=args.dev_seed + 17)
    promo = (paired_BA["mcnemar_p"] < 0.05 and delta_BA >= 0.10
             and delta_ci_BA[0] > 0)
    min_ast = min(ast_rate("A"), ast_rate("B13"))
    valid_preserved = min_ast >= 0.95

    if promo and valid_preserved:
        verdict = "HELDOUT_13_RULE_CARRIER_PROMOTED"
    elif delta_BA > 0:
        verdict = "CONDITIONAL_VERIFIER_ASSISTED_IMPROVEMENT"
    else:
        verdict = "HELDOUT_PROMOTION_NOT_ESTABLISHED"

    result = {
        "run": "system1_v054_heldout_ab13",
        "ckpt": args.ckpt,
        "heldout_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                          "sha256": sd},
        "budget": args.budget,
        "beam_width": args.beam_width,
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a),
                     "ast_valid_rate": ast_rate(a),
                     "family_rates": family_rates(a),
                     "budget_success": {str(K): budget_success(a, K)
                                        for K in (1, 2, 4, 8)}}
                 for a in ("A", "B13", "B7")},
        "oracle_outcome_support": sum(oracle_support) / n,
        "paired": {"B13_vs_A": paired_BA, "B7_vs_A": paired("B7", "A"),
                   "B13_vs_B7_diagnostic_only": paired("B13", "B7")},
        "delta_B13_minus_A": {"mean": delta_BA, "ci90": delta_ci_BA},
        "gates": {"promotion": bool(promo),
                  "validity_preserved": bool(valid_preserved),
                  "min_ast_valid": min_ast},
        "verdict": verdict,
        "attribution_boundary": "A vs B13 promotes the 13-rule carrier; "
                                "B7 is diagnostic only; B13-vs-B7 incremental "
                                "attribution NOT claimed from this run",
        "heldout_guard": {"consumed_digests": CONSUMED_DIGESTS,
                          "single_use": True, "quarantine": True},
    }
    with open(out / "eval_v054_heldout.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "eval_v054_results.json", "w") as f:
        json.dump({"per_task": per_task}, f)
    print(json.dumps(result, indent=2), flush=True)
    print("RECEIPT " + str(out / "eval_v054_heldout.json"), flush=True)


if __name__ == "__main__":
    main()
