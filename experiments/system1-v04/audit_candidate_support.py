"""
System-1 v0.4.3 oracle candidate-support audit (decoder capacity diagnosis).
===========================================================================
Pre-registered 2026-08-23 (roadmap: decoder/capacity work; reference
system1-calibrated-probe-search-integration.md; corpus INFERRED ca4bb787):

QUESTION: at the frozen-decoder capability ceiling, does the CORRECT program
exist inside a widened beam candidate set (pruning/selection failure) or does
it never enter the set (representational support failure)?

DESIGN (matched, disposable):
  - ARM A  standard beam (beta=0.00), width 64, return_all_finals
  - ARM B  CEGIS priority (beta=0.40), width 64, return_all_finals
  - Same checkpoint (v0.4.1 calibrated, frozen), same tasks, same FSA/EOS,
    same expansions. Only beta differs.
  - Sandbox EVERY final candidate (64/task/arm) -> oracle pass labels.

METRICS:
  pass@1 (selected program), any_pass@K for K in {2,4,8,16,32,64}
  (any of the top-K DISTINCT finals by cumulative score passes the sandbox),
  unique valid programs per task, energy rank of passers (min rank by energy
  among passers; 1 = best energy), task-pooled energy Spearman vs pass,
  task-blocked bootstrap 90% CIs on pass@1 and any_pass@64.

DECISION RULE (pre-registered, ARM A):
  let S = any_pass@64 - pass@1.
  S >= +0.15 and task-blocked 90% CI lower bound > 0  -> SELECTION FAILURE
      (correct programs exist in the candidate set; ranking/pruning is the
       bottleneck) -> energy-guided re-ranking / bounded outcome-supervised
       unfreeze is justified.
  S <= +0.05  -> SUPPORT FAILURE (correct programs never enter the set)
      -> structural AST egress is the grounded next mechanism; unfreezing
         the decoder alone is poorly grounded.
  else -> AMBIGUOUS; widen further or re-test.

GUARD: refuses the consumed heldout digest. Fresh disposable split only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
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
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens,
    load_split, sha256_file)

HELDOUT_DIGEST = "887d0d6c"           # consumed heldout40_v04 (sealed)
BETA = 0.40                            # pre-registered priority (v0.4.2)


def _ast_ok(code: str) -> bool:
    try:
        import ast
        ast.parse(code)
        return True
    except Exception:
        return False


def _task_bootstrap_cis(passers: list[int], n_rep: int = 2000,
                        alpha: float = 0.10) -> tuple[float, float]:
    """Task-blocked bootstrap 90% CI for a per-task Bernoulli rate."""
    n = len(passers)
    if n == 0:
        return 0.0, 0.0
    rng = random.Random(20260823)
    rates = []
    for _ in range(n_rep):
        s = 0
        for _ in range(n):
            s += passers[rng.randrange(n)]
        rates.append(s / n)
    rates.sort()
    lo = rates[int(n_rep * alpha / 2)]
    hi = rates[int(n_rep * (1 - alpha / 2)) - 1]
    return lo, hi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=77031)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--tag", default="dev43_v04")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    # ---- heldout guard (code-level) ----
    split_p = pathlib.Path(args.out) / f"{args.tag}.json"
    if split_p.exists():
        d = sha256_file(split_p)
        if d.startswith(HELDOUT_DIGEST):
            raise SystemExit(
                f"REFUSED: {split_p} matches consumed heldout {HELDOUT_DIGEST}...")
    tasks = load_split(args.out, args.dev_n, args.dev_seed, args.tag)
    split_digest = sha256_file(split_p)
    if split_digest.startswith(HELDOUT_DIGEST):
        raise SystemExit("REFUSED: generated split matches heldout digest.")

    cfg = KernelV04Config()
    model = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    model.load_state_dict(st["model"])
    model.eval()
    dec = CEGISBeamPriorityDecoder(model)
    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)
    print(f"DEV_SPLIT {args.tag} n={args.dev_n} seed={args.dev_seed} "
          f"sha={split_digest[:16]}", flush=True)

    z0 = model.encode_tokens(pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(model, tasks, 16, dev)

    per_task: list[dict] = []
    e_all: list[float] = []
    y_all: list[int] = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t)}
        for arm, beta in (("A", 0.0), ("B", BETA)):
            seq_best, rec = dec.decode_cegis_beam(
                z0[i:i + 1], sp[i:i + 1], beam_width=args.width,
                beta_priority=beta, return_all_finals=True)
            # dedupe finals preserving score order
            seen: set[tuple[int, ...]] = set()
            ranked: list[tuple[list[int], float, float | None]] = []
            for s, sc, e in rec["final_candidates"]:
                k = tuple(s)
                if k not in seen:
                    seen.add(k)
                    ranked.append((s, sc, e))
            labels = [sandbox(detokenize(s), t["tests"]) for s, _, _ in ranked]
            pass1 = labels[0] if ranked else False
            # any_pass@K over DISTINCT ranked finals
            any_at: dict[int, bool] = {}
            hit = False
            for K in (1, 2, 4, 8, 16, 32, 64):
                for j in range(min(K, len(labels))):
                    hit = hit or labels[j]
                any_at[K] = hit
            uniq_valid = sum(1 for s, _, _ in ranked if _ast_ok(detokenize(s)))
            # energy rank of passers (1 = best energy; None if none pass)
            e_rank_sorted = sorted(
                [(e, labels[j]) for j, (_, _, e) in enumerate(ranked)
                 if e is not None], key=lambda x: -x[0])
            e_rank_passers = [j + 1 for j, (_, p) in enumerate(e_rank_sorted)
                              if p]
            row[arm] = {
                "best_seq": seq_best, "pass1": pass1,
                "any_pass": {str(K): any_at[K] for K in any_at},
                "n_distinct": len(ranked),
                "unique_valid": uniq_valid,
                "min_energy_rank_of_passer": min(e_rank_passers)
                if e_rank_passers else None,
            }
            for s, _, e in ranked:
                if e is not None:
                    e_all.append(e)
                    y_all.append(labels[ranked.index((s, _, e))])
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{args.dev_n}] t={time.time() - t0:.0f}s",
                  flush=True)

    # ---- aggregates (ARM A primary per decision rule) ----
    def rate(arm: str, key: str) -> float:
        vals = []
        for r in per_task:
            v = r[arm]
            for part in key.split("."):
                v = v[part]
            vals.append(v)
        return sum(1 for v in vals if v) / len(vals)

    n = len(per_task)
    agg: dict = {"run": "system1_v043_oracle_support_audit",
                 "ckpt": args.ckpt,
                 "ckpt_sha": sha256_file(pathlib.Path(args.ckpt)),
                 "dev_split": {"tag": args.tag, "n": args.dev_n,
                               "seed": args.dev_seed, "sha256": split_digest},
                 "width": args.width, "beta_B": BETA,
                 "n_tasks": n, "runtime_s": round(time.time() - t0, 1)}
    for arm in ("A", "B"):
        p1 = rate(arm, "pass1")
        any64 = rate(arm, "any_pass.64")
        lo1, hi1 = _task_bootstrap_cis(
            [1 if r[arm]["pass1"] else 0 for r in per_task])
        lo64, hi64 = _task_bootstrap_cis(
            [1 if r[arm]["any_pass"]["64"] else 0 for r in per_task])
        nuniq = sum(r[arm]["n_distinct"] for r in per_task) / n
        nvalid = sum(r[arm]["unique_valid"] for r in per_task) / n
        eranks = [r[arm]["min_energy_rank_of_passer"] for r in per_task
                  if r[arm]["min_energy_rank_of_passer"] is not None]
        agg[arm] = {
            "pass1": round(p1, 4),
            "pass1_ci90": [round(lo1, 4), round(hi1, 4)],
            "any_pass": {str(K): round(rate(arm, f"any_pass.{K}"), 4)
                         for K in (1, 2, 4, 8, 16, 32, 64)},
            "any_pass64_ci90": [round(lo64, 4), round(hi64, 4)],
            "mean_distinct_finals": round(nuniq, 2),
            "mean_unique_valid": round(nvalid, 2),
            "passer_energy_ranks": {"n_tasks_with_passer": len(eranks),
                                    "mean_min_rank": round(
                                        sum(eranks) / len(eranks), 2)
                                    if eranks else None},
        }

    # candidate-level energy/pass association (task-pooled; clustered note)
    if e_all and len(e_all) >= 3:
        rx = {v: i for i, v in enumerate(sorted(set(e_all)))}
        ry = {v: i for i, v in enumerate(sorted(set(y_all)))}
        a = [rx[v] for v in e_all]
        b = [ry[v] for v in y_all]
        m = len(a)
        mx, my = sum(a) / m, sum(b) / m
        num = sum((u - mx) * (v - my) for u, v in zip(a, b))
        dx = (sum((u - mx) ** 2 for u in a) ** 0.5)
        dy = (sum((v - my) ** 2 for v in b) ** 0.5)
        rho = 0.0 if dx == 0 or dy == 0 else num / (dx * dy)
        agg["energy_assoc"] = {"n_pairs": len(e_all),
                               "spearman_raw_pooled": round(rho, 4),
                               "note": "task-pooled, clustered (anti-"
                                       "conservative); diagnostic only"}

    # ---- decision (pre-registered; ARM A) ----
    s = agg["A"]["any_pass"]["64"] - agg["A"]["pass1"]
    lo_delta = agg["A"]["any_pass64_ci90"][0] - agg["A"]["pass1_ci90"][1]
    if s >= 0.15 and lo_delta > 0.0:
        decision = "SELECTION_FAILURE"
    elif s <= 0.05:
        decision = "SUPPORT_FAILURE"
    else:
        decision = "AMBIGUOUS"
    agg["decision"] = {
        "rule": "S = any_pass@64 - pass1; S>=0.15 with CI>0 -> SELECTION; "
                "S<=0.05 -> SUPPORT",
        "S": round(s, 4), "ci90_delta_lb": round(lo_delta, 4),
        "verdict": decision,
        "next": {"SELECTION_FAILURE":
                 "energy-guided re-ranking / bounded outcome-supervised "
                 "unfreeze justified",
                 "SUPPORT_FAILURE":
                 "structural AST egress is the grounded next mechanism; "
                 "unfreeze alone poorly grounded",
                 "AMBIGUOUS": "widen beam further or re-test"}[decision],
    }
    agg["heldout_guard"] = {"consumed_heldout_digest": HELDOUT_DIGEST,
                            "quarantine": True}

    print("AUDIT:", json.dumps(agg, indent=2), flush=True)
    pathlib.Path(args.out + "/audit_support.json").write_text(
        json.dumps(agg, indent=2))
    pathlib.Path(args.out + "/audit_support_results.json").write_text(
        json.dumps(per_task, indent=2))
    print(f"RECEIPT {args.out}/audit_support.json", flush=True)


if __name__ == "__main__":
    main()
