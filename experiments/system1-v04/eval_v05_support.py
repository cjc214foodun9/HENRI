"""
System-1 v0.5 paired structural-egress support evaluator (dev; heldout guard).
=============================================================================
Pre-registered 2026-08-24 (v0.5 upload 3dfd53f0... audited; faithful kernel
system1_kernel_v05_ast_skeleton.py; references system1-decoder-support-audit
+ system1-calibrated-probe-search-integration + system1-outcome-energy-
calibration):

ARMS (matched candidate budget 64, same tasks, same sandbox, same dedupe):
  A  v0.4.3 token beam (beta=0.0), width 64, return_all_finals
  B  v0.5 structural skeleton, UNIFORM selection (no energy)
  C  v0.5 structural skeleton + calibrated-energy filter
Every DISTINCT final candidate is sandboxed (oracle labels).

METRICS per arm: pass@1, any_pass@K (K=2,4,8,16,32,64), S = any_pass@64 -
pass@1, mean distinct finals/task, mean distinct rules/task, CEGIS admit
rate (first sandbox-passer in the pool), task-blocked bootstrap 90% CIs on
pass@1 and any_pass@64. Paired transitions A/B/C with McNemar. Energy/pass
Spearman on the EXACT v0.5 candidate states (arm C) — CONDITIONAL, clustered.

DECISION RULE (pre-registered, support on arm B — the structural claim):
  S_B >= +0.15 with task-blocked 90% CI lower bound of any_pass@64_B -
  pass@1_B > 0        -> SUPPORT_RESTORED (correct programs enter the pool)
  S_B <= +0.05        -> SUPPORT_NOT_RESTORED (grammar still misses)
  else                -> AMBIGUOUS
  CAPABILITY    cegis_pass_B > pass1_A, paired, McNemar p < 0.05,
                delta >= 0.10
  ENERGY_FILTER cegis_pass_C >= cegis_pass_B (filter does not damage)
  PROMOTION     SUPPORT_RESTORED AND CAPABILITY, validity preserved
Syntactic diversity alone is NOT the support gate.

GUARD: refuses the consumed heldout digest. Fresh disposable split only.
"""
from __future__ import annotations

import argparse
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
    KernelV04Config, System1KernelV04, detokenize)
from system1_kernel_v042_cegis_beam import (  # noqa: E402
    CEGISBeamPriorityDecoder)
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    System1KernelV05)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens,
    load_split, sha256_file)

HELDOUT_DIGEST = "887d0d6c"           # consumed heldout40_v04 (sealed)
WIDTH = 64                             # matched candidate budget per arm
BETA_A = 0.0


def _ast_ok(code: str) -> bool:
    try:
        import ast
        ast.parse(code)
        return True
    except Exception:
        return False


def _mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for k in range(min(b, c) + 1):
        p += math.comb(n, k) / (2.0 ** n)
    return 2.0 * min(p, 0.5)


def _task_bootstrap_cis(passers: list[int], n_rep: int = 2000,
                        alpha: float = 0.10) -> tuple[float, float]:
    n = len(passers)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(20260824)
    rates = []
    for _ in range(n_rep):
        s = 0
        for _ in range(n):
            s += passers[rng.randrange(n)]
        rates.append(s / n)
    rates.sort()
    return rates[int(n_rep * alpha / 2)], rates[int(n_rep * (1 - alpha / 2)) - 1]


def _dedupe_ranked(items: list[tuple]) -> tuple[list[tuple], list[int]]:
    """Dedupe by complete program tuple, preserving order; return (items,
    sandbox-order labels per ORIGINAL index mapping)."""
    seen: set[tuple] = set()
    out: list[tuple] = []
    for it in items:
        key = tuple(it[0]) if isinstance(it[0], list) else it[0]
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=55123)
    ap.add_argument("--tag", default="dev50_v05")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    split_p = pathlib.Path(args.out) / f"{args.tag}.json"
    if split_p.exists():
        d = sha256_file(split_p)
        if d.startswith(HELDOUT_DIGEST):
            raise SystemExit("REFUSED: consumed heldout.")
    tasks = load_split(args.out, args.dev_n, args.dev_seed, args.tag)
    split_digest = sha256_file(split_p)
    if split_digest.startswith(HELDOUT_DIGEST):
        raise SystemExit("REFUSED: generated split matches heldout digest.")

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    dec = CEGISBeamPriorityDecoder(backbone)
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()
    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)
    print(f"DEV_SPLIT {args.tag} n={args.dev_n} seed={args.dev_seed} "
          f"sha={split_digest[:16]}", flush=True)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    e_all: list[float] = []
    y_all: list[int] = []
    gen_calls = {"A": 0, "B": 0, "C": 0}
    sandbox_calls = 0
    t0 = time.time()
    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t)}
        # ---- ARM A: token beam (v0.4.3 baseline) ----
        seq_a, rec_a = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=WIDTH, beta_priority=BETA_A,
            return_all_finals=True)
        items_a = [(s, sc, e) for s, sc, e in rec_a["final_candidates"]]
        items_a = _dedupe_ranked(items_a)
        labs_a = []
        for s, sc, e in items_a:
            labs_a.append(sandbox(detokenize(s), t["tests"]))
            sandbox_calls += 1
        pass1_a = labs_a[0] if labs_a else 0
        any_a: dict[int, bool] = {}
        hit = False
        for K in (1, 2, 4, 8, 16, 32, 64):
            for j in range(min(K, len(labs_a))):
                hit = hit or labs_a[j]
            any_a[K] = hit
        cegis_a = next((l for l in labs_a if l), 0)
        row["A"] = {"pass1": pass1_a, "any_pass": {str(K): any_a[K]
                    for K in any_a}, "cegis_admit": cegis_a,
                    "distinct": len(items_a)}
        # ---- ARM B: skeleton uniform ----
        cands_b = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=WIDTH, use_energy=False)
        gen_calls["B"] += 1
        labs_b = [sandbox(c["code"], t["tests"]) for c in cands_b]
        sandbox_calls += len(labs_b)
        pass1_b = labs_b[0] if labs_b else 0
        any_b = {}
        hit = False
        for K in (1, 2, 4, 8, 16, 32, 64):
            for j in range(min(K, len(labs_b))):
                hit = hit or labs_b[j]
            any_b[K] = hit
        cegis_b = next((l for l in labs_b if l), 0)
        row["B"] = {"pass1": pass1_b, "any_pass": {str(K): any_b[K]
                    for K in any_b}, "cegis_admit": cegis_b,
                    "distinct": len(set(c["code"] for c in cands_b)),
                    "rules": len(set(c["rule_id"] for c in cands_b))}
        # ---- ARM C: skeleton + energy filter ----
        cands_c = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=WIDTH, use_energy=True)
        gen_calls["C"] += 1
        labs_c = [sandbox(c["code"], t["tests"]) for c in cands_c]
        sandbox_calls += len(labs_c)
        pass1_c = labs_c[0] if labs_c else 0
        any_c = {}
        hit = False
        for K in (1, 2, 4, 8, 16, 32, 64):
            for j in range(min(K, len(labs_c))):
                hit = hit or labs_c[j]
            any_c[K] = hit
        cegis_c = next((l for l in labs_c if l), 0)
        row["C"] = {"pass1": pass1_c, "any_pass": {str(K): any_c[K]
                    for K in any_c}, "cegis_admit": cegis_c,
                    "distinct": len(set(c["code"] for c in cands_c)),
                    "rules": len(set(c["rule_id"] for c in cands_c))}
        # energy/pass pairs on the EXACT v0.5 candidate states (arm C)
        for c, l in zip(cands_c, labs_c):
            e_all.append(c["energy"])
            y_all.append(l)
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{args.dev_n}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)

    def rate(arm: str, key: str) -> float:
        vals = []
        for r in per_task:
            v = r[arm]
            for part in key.split("."):
                v = v[part]
            vals.append(v)
        return sum(1 for v in vals if v) / n

    agg: dict = {"run": "system1_v05_structural_support",
                 "ckpt": args.ckpt,
                 "ckpt_sha": sha256_file(pathlib.Path(args.ckpt)),
                 "dev_split": {"tag": args.tag, "n": args.dev_n,
                               "seed": args.dev_seed, "sha256": split_digest},
                 "budget": WIDTH, "beta_A": BETA_A,
                 "n_tasks": n, "runtime_s": round(time.time() - t0, 1),
                 "compute": {"gen_calls": gen_calls,
                             "sandbox_executions": sandbox_calls}}
    for arm in ("A", "B", "C"):
        p1 = rate(arm, "pass1")
        any64 = rate(arm, "any_pass.64")
        lo1, hi1 = _task_bootstrap_cis(
            [1 if r[arm]["pass1"] else 0 for r in per_task])
        lo64, hi64 = _task_bootstrap_cis(
            [1 if r[arm]["any_pass"]["64"] else 0 for r in per_task])
        agg[arm] = {
            "pass1": round(p1, 4),
            "pass1_ci90": [round(lo1, 4), round(hi1, 4)],
            "any_pass": {str(K): round(rate(arm, f"any_pass.{K}"), 4)
                         for K in (1, 2, 4, 8, 16, 32, 64)},
            "any_pass64_ci90": [round(lo64, 4), round(hi64, 4)],
            "S": round(any64 - p1, 4),
            "cegis_admit": round(rate(arm, "cegis_admit"), 4),
            "mean_distinct_finals": round(
                sum(r[arm]["distinct"] for r in per_task) / n, 2),
            "mean_distinct_rules": round(
                sum(r[arm].get("rules", 0) for r in per_task) / n, 2),
        }

    # ---- paired transitions ----
    def transitions(arm_x: str, arm_y: str) -> dict:
        both = sum(1 for r in per_task
                   if r[arm_x]["pass1"] and r[arm_y]["pass1"])
        x_only = sum(1 for r in per_task
                     if r[arm_x]["pass1"] and not r[arm_y]["pass1"])
        y_only = sum(1 for r in per_task
                     if r[arm_y]["pass1"] and not r[arm_x]["pass1"])
        neither = n - both - x_only - y_only
        return {"both": both, f"{arm_x}_only": x_only,
                f"{arm_y}_only": y_only, "neither": neither,
                "mcnemar_p": round(_mcnemar_two_sided(y_only, x_only), 4)}

    agg["transitions"] = {
        "A_vs_B": transitions("A", "B"),
        "A_vs_C": transitions("A", "C"),
        "B_vs_C": transitions("B", "C"),
    }

    # ---- energy/pass association on v0.5 states (CONDITIONAL) ----
    if e_all and len(e_all) >= 3:
        rx = {v: i for i, v in enumerate(sorted(set(e_all)))}
        ry = {v: i for i, v in enumerate(sorted(set(y_all)))}
        a = [rx[v] for v in e_all]
        b = [ry[v] for v in y_all]
        m = len(a)
        mx, my = sum(a) / m, sum(b) / m
        num = sum((u - mx) * (v - my) for u, v in zip(a, b))
        dx = sum((u - mx) ** 2 for u in a) ** 0.5
        dy = sum((v - my) ** 2 for v in b) ** 0.5
        rho = 0.0 if dx == 0 or dy == 0 else num / (dx * dy)
        agg["energy_assoc_v05"] = {
            "n_pairs": len(e_all), "spearman_raw": round(rho, 4),
            "note": "exact v0.5 candidate states (arm C); task-pooled, "
                    "clustered -> anti-conservative; CONDITIONAL, not "
                    "inherited from v0.4.1"}

    # ---- decision (pre-registered; support on arm B) ----
    s_b = agg["B"]["S"]
    lo_s = agg["B"]["any_pass64_ci90"][0] - agg["B"]["pass1_ci90"][1]
    if s_b >= 0.15 and lo_s > 0.0:
        support = "SUPPORT_RESTORED"
    elif s_b <= 0.05:
        support = "SUPPORT_NOT_RESTORED"
    else:
        support = "AMBIGUOUS"
    t_ab = agg["transitions"]["A_vs_B"]
    cap = (support == "SUPPORT_RESTORED"
           and agg["B"]["cegis_admit"] > agg["A"]["cegis_admit"]
           and t_ab["mcnemar_p"] < 0.05
           and (agg["B"]["cegis_admit"] - agg["A"]["cegis_admit"]) >= 0.10)
    energy_ok = agg["C"]["cegis_admit"] >= agg["B"]["cegis_admit"]
    promotion = support == "SUPPORT_RESTORED" and cap and energy_ok
    agg["decision"] = {
        "rule": "S_B = any_pass@64_B - pass1_B; >=0.15 with CI>0 -> "
                "SUPPORT_RESTORED; <=0.05 -> NOT_RESTORED",
        "S_B": round(s_b, 4), "ci90_delta_lb": round(lo_s, 4),
        "support": support,
        "capability": bool(cap),
        "energy_filter_ok": bool(energy_ok),
        "verdict": ("PROMOTED" if promotion else
                    ("SUPPORT_RESTORED_NOT_PROMOTED"
                     if support == "SUPPORT_RESTORED"
                     else "NOT_PROMOTED")),
    }
    agg["heldout_guard"] = {"consumed_heldout_digest": HELDOUT_DIGEST,
                            "quarantine": True}

    print("EVAL:", json.dumps(agg, indent=2), flush=True)
    pathlib.Path(args.out + "/eval_v05_support.json").write_text(
        json.dumps(agg, indent=2))
    pathlib.Path(args.out + "/eval_v05_results.json").write_text(
        json.dumps(per_task, indent=2))
    print(f"RECEIPT {args.out}/eval_v05_support.json", flush=True)


if __name__ == "__main__":
    main()
