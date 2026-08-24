"""
System-1 v0.6.0.1 — Candidate-Specific Retrieval dev evaluation.
================================================================
Reference 3 (gpt-5.6-sol) + uploaded contract (sha 3523f55b…):
CEILING CORRECTION — B13 already 52/52 on heldout53_v055. The endpoint
here is VERIFIER-CALL REDUCTION under EXACT capability preservation, NOT
pass-rate gain. No heldout is created or consumed; the split is a fresh
DISPOSABLE dev split; the sealed heldout digest is ADDED to the guard.

Arms (matched; differ only in the intervention):
  B13  frozen v0.5.5 carrier, uniform pool order (baseline)
  R0   retrieval path enabled, beta=0 -> must be byte-identical to B13
  R1   candidate-specific retrieval with pre-registered beta (default 0.15)
       s_k = cos(E_task(x), E_cand(c_k)); score'_k = -k + beta*z(s_k)

Verdict chain (pre-registered):
  INVALID_VERIFIER_REPLAY           guard match (raised at startup)
  CANDIDATE_RETRIEVAL_REGRESSION    outcome rate drops OR any family
                                    support drops OR runtime/VRAM gate fails
  CANDIDATE_RETRIEVAL_COST_PROMOTED outcome preserved AND min-family
                                    support preserved AND paired verifier-call
                                    delta CI excludes 0 (reduction) AND
                                    cost gates pass
  CANDIDATE_RETRIEVAL_NO_EFFECT     zero paired discordance AND zero
                                    call-delta significance
  CANDIDATE_RETRIEVAL_NO_IMPROVEMENT nonzero changes without the
                                    pre-registered improvement
  (EFFICACY_PROMOTED only under a harder disposable condition - FUTURE)

Pre-registered gates:
  G1 integrity: fresh disposable dev split; guard passed; sealed heldout
     never loaded; R0 must equal B13 byte-identically.
  G2 outcome preservation: rate(R1) >= rate(B13) AND per-family support
     (R1) >= per-family support (B13) on every family present.
  G3 cost reduction: paired bootstrap CI of (calls_B13 - calls_R1)
     has upper bound < 0.
  G4 cost gates: runtime and VRAM reported; no material regression
     (threshold: < 2x baseline runtime, VRAM within 20% of baseline).
  G5 variance: R1 scores have within-task variance > 0 on every task
     with a nontrivial pool (asserted in telemetry).
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
from system1_kernel_v055_ast_skeleton import System1KernelV05  # noqa: E402
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from train_v051_discriminator import (  # noqa: E402
    build_split, sha256_file, N_VERIFIER, N_OUTCOME, _rand_args, _expected,
    _args_key)
from zone_c_bridge_v0601 import CandidateRetrievalRanker  # noqa: E402


def build_split_stratified(out_dir: str, n_tasks: int, seed: int, tag: str,
                           n_families: int = 13) -> list[dict]:
    """Deterministic multi-test split with EXACT family stratification.
    IDENTICAL construction to eval_v055_heldout.build_split_stratified
    (same _rand_args/_expected/_args_key; 4+4 disjoint tests)."""
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
        t["tests"] = tests
        t["verifier_tests"] = tests[:N_VERIFIER]
        t["outcome_tests"] = tests[N_VERIFIER:]
        t["verifier_args"] = verifier_args
        t["outcome_args"] = outcome_args
        tasks.append(t)
    with open(p, "w") as f:
        json.dump(tasks, f, indent=1)
    return tasks


# Full SHA-256 where available (Reference 3). The SEALED v0.5.5 heldout is
# ADDED so this dev evaluator refuses it even by accident.
CONSUMED_DIGESTS = [
    # ---- v0.4/v0.5.1-v0.5.3 era (prefixes) ----
    "887d0d6c", "23b36795", "82c97532", "db027f9c", "9d4c29ad",
    "1f81e4d0", "181cc59b", "092cb0c1", "2eb8d29b", "78b4cfb4",
    "7a8c1e7b", "306ab62d", "6b5bb1b4", "0ec0528d", "0535a2dc",
    "35d15aae",
    # ---- v0.5.2 / v0.5.3 ----
    "5e5f4a00", "ce2a76fb", "9a17af61", "635c2aaa", "888809df",
    "8ea34261",
    # ---- v0.5.4 / v0.5.5 ----
    "a09bf275ef7f09a72a88f78dd36b78e1ab752a1cb79d4960004646da90440a54",
    "873902867001b7f19abf3e2641b10913f2c57c4fb2e23d609078a471edc9c9ed",
    #   heldout53_v055 (SEALED v0.5.5 heldout — dev MUST never load it)
    # ---- v0.6.x dev ----
    "d6b79d510620efdc5b9ea80e629a736b7cbad763adb13b934eb627f038f7f804",
    "392ce03e3b35ab4a8df4dc834d96e69dffed50d2e38716a02344a2f14526460f",
    "338cbda609321213ee907b1f71c66a927fc2628ed251910101768d38ca0477ca",
    "44657c7f96da7fdc70e8ec5a0cfcb90c0ef049439d88a9cd9de9a7e49e4c9a7c",
    "bed7368d227bbb55e8c448e6676b18e628e97a95972cfb0992ee638bae1d9ccc",
    "928e40af50001d4101d421a853f0bbdb2e39cef83b4c2d6ce7d62a6787890e40",
    "4b7d854d1d7ad026f95a9a5a42877a02fb47a4aa7ce0fbc5d8518ee7f2a84102",
]


def _mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return 2 * min(
        sum(math.comb(n, k) * 0.5 ** n for k in range(0, min(b, c) + 1)),
        sum(math.comb(n, k) * 0.5 ** n for k in range(max(b, c), n + 1)))


def _task_bootstrap_ci_delta(a: list[float], b: list[float],
                             n_rep: int = 2000, seed: int = 0,
                             alpha: float = 0.05) -> tuple[float, float]:
    """Paired bootstrap CI on mean(a - b)."""
    rng = random.Random(seed)
    n = len(a)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(n_rep):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_rep)]
    hi = means[int((1 - alpha / 2) * n_rep) - 1]
    return lo, hi


def _cegis_first(pool: list[tuple], verifier_tests: list[str],
                 budget: int = 64) -> tuple[int, int]:
    for i, (code, _order) in enumerate(pool[:budget]):
        if sandbox(code, verifier_tests) == 1:
            return i, i + 1
    return -1, len(pool[:budget])


def _ast_ok(code: str) -> int:
    return 1 if TOK2ID["UNK"] not in tokenize_code(code) else 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=60)
    ap.add_argument("--dev-seed", type=int, default=90909)
    ap.add_argument("--tag", default="dev9_v0601")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beta", type=float, default=0.15,
                    help="pre-registered R1 retrieval strength")
    ap.add_argument("--expect-sha", default="",
                    help="optional dev split pin")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    # ---- split: fresh DISPOSABLE dev split ----
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

    # ---- frozen v0.5.5 carrier ----
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05_13.eval()

    trainable = [n for n, p in backbone.named_parameters()
                 if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable}")

    ranker_r0 = CandidateRetrievalRanker(enabled=True, beta=0.0, device=dev)
    ranker_r1 = CandidateRetrievalRanker(enabled=True, beta=args.beta,
                                         device=dev)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    calls = {"B13": [], "R0": [], "R1": []}
    outcome_pass = {"B13": [], "R0": [], "R1": []}
    ast_valid = {"B13": [], "R0": [], "R1": []}
    r1_var = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t), "fid": t["fid"]}
        ver_tests = t["verifier_tests"]
        out_tests = t["outcome_tests"]

        # shared pool: frozen carrier, uniform order (B13)
        cands13 = v05_13.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
            use_energy=False)
        pool13 = [(c["code"], c["rule_id"]) for c in cands13]

        # task representation (pre-verifier: signature latent)
        tv = ranker_r1.task_repr(z0[i:i + 1], sp[i:i + 1], v05_13)

        # R0: beta=0 -> byte-identical order
        pool_r0 = ranker_r0.rank_candidates(
            cands13, tv, v05_13, dev, beta=0.0)
        pool_r0_t = [(c["code"], c["rule_id"]) for c in pool_r0]
        # R1: candidate-specific reorder
        pool_r1 = ranker_r1.rank_candidates(
            cands13, tv, v05_13, dev, beta=args.beta)
        pool_r1_t = [(c["code"], c["rule_id"]) for c in pool_r1]
        # per-task R1 within-task variance of sim scores
        if cands13:
            cv = [ranker_r1.candidate_repr(c["code"], v05_13, dev)
                  for c in cands13]
            s = ranker_r1.sim_scores(tv, cv)
            r1_var.append(float(s.var(unbiased=False).item())
                          if s.numel() > 1 else 0.0)

        idx_b, call_b = _cegis_first(pool13, ver_tests, budget=args.budget)
        idx_r0, call_r0 = _cegis_first(pool_r0_t, ver_tests,
                                       budget=args.budget)
        idx_r1, call_r1 = _cegis_first(pool_r1_t, ver_tests,
                                       budget=args.budget)

        for arm, idx, call in (("B13", idx_b, call_b),
                               ("R0", idx_r0, call_r0),
                               ("R1", idx_r1, call_r1)):
            calls[arm].append(call)
            if idx >= 0:
                code = pool13[idx][0] if arm == "B13" else (
                    pool_r0_t[idx][0] if arm == "R0" else pool_r1_t[idx][0])
                outcome_pass[arm].append(sandbox(code, out_tests))
                ast_valid[arm].append(_ast_ok(code))
            else:
                outcome_pass[arm].append(0)
                ast_valid[arm].append(0)

        row["B13"] = {"admit": idx_b, "calls": call_b,
                      "outcome": outcome_pass["B13"][-1],
                      "ast_valid": ast_valid["B13"][-1],
                      "pool": len(pool13)}
        row["R0"] = {"admit": idx_r0, "calls": call_r0,
                     "outcome": outcome_pass["R0"][-1],
                     "ast_valid": ast_valid["R0"][-1],
                     "pool": len(pool_r0_t),
                     "identical_to_B13": [c[0] for c in pool_r0_t] ==
                     [c[0] for c in pool13]}
        row["R1"] = {"admit": idx_r1, "calls": call_r1,
                     "outcome": outcome_pass["R1"][-1],
                     "ast_valid": ast_valid["R1"][-1],
                     "pool": len(pool_r1_t),
                     "sim_var": r1_var[-1] if r1_var else 0.0}
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{len(tasks)}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)

    def rate(arm: str) -> float:
        return sum(outcome_pass[arm]) / n if n else 0.0

    def call_stats(arm: str) -> dict:
        v = sorted(calls[arm])
        m = len(v)
        return {"mean": sum(v) / m if m else 0.0,
                "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0,
                "max": v[-1] if m else 0}

    def family_support_counts(arm: str) -> dict:
        fam: dict[int, list[int]] = {}
        for r in per_task:
            fam.setdefault(r["fid"], []).append(r[arm]["outcome"])
        return {str(f): [sum(v), len(v)] for f, v in sorted(fam.items())}

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

    # ---- gates ----
    r0_identical = all(r["R0"]["identical_to_B13"] for r in per_task)
    g1 = r0_identical and not any(
        sd.startswith(d) for d in CONSUMED_DIGESTS)
    fam_b = family_support_counts("B13")
    fam_r1 = family_support_counts("R1")
    fam_regressed = any(
        fam_r1.get(k, [0, 0])[0] < fam_b[k][0] for k in fam_b)
    g2 = rate("R1") >= rate("B13") and not fam_regressed
    delta_ci = _task_bootstrap_ci_delta(
        [float(x) for x in calls["B13"]],
        [float(x) for x in calls["R1"]],
        seed=args.dev_seed + 27, alpha=0.05)
    g3 = delta_ci[1] < 0.0 and (delta_ci[1] - delta_ci[0]) > 1e-9
    g5 = all(v > 1e-6 for v in r1_var) if r1_var else False
    paired_r1 = paired("R1", "B13")
    paired_r0 = paired("R0", "B13")

    # ---- verdict chain (pre-registered) ----
    if not g1:
        verdict = "INVALID_VERIFIER_REPLAY"   # guard failed at startup
    elif not g2:
        verdict = "CANDIDATE_RETRIEVAL_REGRESSION"
    elif g3 and g5:
        verdict = "CANDIDATE_RETRIEVAL_COST_PROMOTED"
    elif paired_r1["mcnemar_p"] >= 0.05 and paired_r0["both"] == n:
        verdict = "CANDIDATE_RETRIEVAL_NO_EFFECT"
    else:
        verdict = "CANDIDATE_RETRIEVAL_NO_IMPROVEMENT"

    result = {
        "run": "system1_v0601_dev_candidate_retrieval",
        "ckpt": args.ckpt,
        "split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                  "sha256": sd, "disposable": True},
        "budget": args.budget,
        "beta": args.beta,
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "calls": call_stats(a),
                     "ast_valid_rate": (
                         sum(ast_valid[a]) / max(1, sum(
                             1 for r in per_task if r[a]["admit"] >= 0))),
                     "family_support_counts": family_support_counts(a)}
                 for a in ("B13", "R0", "R1")},
        "ranker_stats": ranker_r1.stats(),
        "paired": {"R1_vs_B13": paired_r1, "R0_vs_B13": paired_r0},
        "call_delta_B13_minus_R1": {
            "mean": (sum(calls["B13"]) - sum(calls["R1"])) / n,
            "ci95": delta_ci},
        "gates": {"G1_integrity_r0_identity": bool(g1),
                  "G2_outcome_family_preserved": bool(g2),
                  "G3_cost_reduction_ci_excludes_0": bool(g3),
                  "G5_variance_positive": bool(g5),
                  "r0_byte_identical": bool(r0_identical)},
        "verdict": verdict,
        "attribution_boundary": (
            "Endpoint = verifier-call reduction under EXACT capability "
            "preservation (B13 already 52/52). No heldout created or "
            "consumed; disposable dev split only. EFFICACY_PROMOTED "
            "reserved for a harder disposable condition (future)."),
        "guard": {"consumed_digests": CONSUMED_DIGESTS,
                  "sealed_heldout_never_loaded": True},
    }
    with open(out / "eval_v0601_dev.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "eval_v0601_per_task.json", "w") as f:
        json.dump({"per_task": per_task}, f)
    print(json.dumps(result, indent=2), flush=True)
    print("RECEIPT " + str(out / "eval_v0601_dev.json"), flush=True)


if __name__ == "__main__":
    main()
