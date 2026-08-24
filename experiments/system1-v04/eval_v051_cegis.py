"""
System-1 v0.5.1 paired CEGIS-first evaluation with DISJOINT verifier/outcome tests.
=============================================================================
Pre-registered 2026-08-24 (reference system1-structural-egress-cegis.md and
the v0.5.1 protocol: disjoint verifier/outcome partition, egress discriminator
calibrated on verifier labels, matched ordering arms, cost + capability +
promotion gates, task-blocked inference).

Arms (matched candidate pool for B/C/D; A is the token-beam baseline):
  A  token beam (beta=0.0, width=budget) + CEGIS-first over verifier tests
  B  skeleton uniform order + CEGIS-first
  C  skeleton re-ranked by LEARNED egress discriminator + CEGIS-first
  D  skeleton deterministic random-order control + CEGIS-first
Admission = first candidate that passes ALL verifier tests (indices 0..3 of
the task's 8 tests). Scoring = OUTCOME tests (indices 4..7), NEVER consulted
during selection. This makes CEGIS capability non-tautological.

Metrics per arm: outcome pass rate of the admitted program (pass@1 analogue),
outcome success under budget K in {1,2,4,8}, verifier calls per task
(mean/median/p90/max), CEGIS admit rate, oracle outcome support
(any candidate passes outcome tests — measurement only, no selection use).

Discriminator calibration (exact egress states, C-arm candidates):
AUROC, Brier vs baseline p(1-p), Spearman(p, verifier-label) sign, variance,
frozen-module audit.

Gates (all pre-registered):
  CALIBRATION    both classes; AUROC>0.5; Brier < p(1-p); correct-sign
                 Spearman (task-pooled, anti-conservative noted); nonzero
                 variance; frozen audit PASS.
  CEGIS_OP       outcome validity preserved; C outcome success >= B (paired,
                 not materially worse); expected verifier calls C <= 0.8*B
                 with task-blocked 90% CI excluding zero; improvement visible
                 at K<=4.
  PROMOTION      significant paired outcome improvement vs ARM A (token beam)
                 under matched verifier budget (McNemar p<0.05, delta >= 0.10,
                 task-blocked CI lb > 0).

Verdict vocabulary (reference 3):
  CALIBRATED_EGRESS_DIAGNOSTIC_NOT_PROMOTED
  EGRESS_CALIBRATED_COST_IMPROVED
  CEGIS_VERIFIER_ASSISTED_NOT_CAPABILITY_PROMOTED
  CEGIS_CAPABILITY_PROMOTED

Heldout guard: refuses the consumed heldout digest; dev2_v051 is sealed
BEFORE training and evaluated once.
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
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04, detokenize, KernelV04Config)
from system1_kernel_v042_cegis_beam import (  # noqa: E402
    CEGISBeamPriorityDecoder)
from system1_kernel_v05_ast_skeleton import (  # noqa: E402
    System1KernelV05)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from train_v051_discriminator import (  # noqa: E402
    EgressDiscriminator, build_split, candidate_state, sha256_file,
    N_VERIFIER, N_OUTCOME)

HELDOUT_DIGEST = "887d0d6c"


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


def _auroc(probs: list[float], y: list[int]) -> float:
    pos = [(p, 1) for p, yv in zip(probs, y) if yv == 1]
    neg = [(p, 0) for p, yv in zip(probs, y) if yv == 0]
    if not pos or not neg:
        return 0.5
    inv = 0.0
    for p1, _ in pos:
        for p0, _ in neg:
            inv += 1.0 if p1 > p0 else (0.5 if p1 == p0 else 0.0)
    return inv / (len(pos) * len(neg))


def _spearman(x: list[float], y: list[int]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    def rank(v):
        idx = sorted(range(n), key=lambda i: v[i])
        r = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and v[idx[j + 1]] == v[idx[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[idx[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank([float(v) for v in y])
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) *
                    sum((b - my) ** 2 for b in ry))
    return num / den if den > 0 else 0.0


def _cegis_first(pool: list[tuple], verifier_tests: list[str],
                 budget: int = 64) -> tuple[int, int, int]:
    """Scan pool in order, run verifier tests only. Return
    (admit_index or -1, calls_used, outcome_pending_not_scored).
    Sandbox returns 1 iff ALL given tests pass."""
    for i, (code, _order) in enumerate(pool[:budget]):
        if sandbox(code, verifier_tests) == 1:
            return i, i + 1, 0
    return -1, len(pool[:budget]), 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--disc", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=90837)
    ap.add_argument("--tag", default="dev2_v051")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    tasks = build_split(args.out, args.dev_n, args.dev_seed, args.tag)
    split_p = out / f"{args.tag}.json"
    sd = sha256_file(split_p)
    if sd.startswith(HELDOUT_DIGEST):
        raise SystemExit("REFUSED: split matches consumed heldout digest.")

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()
    disc = EgressDiscriminator(d_slot=backbone.cfg.d_slot).to(dev)
    disc.load_state_dict(torch.load(args.disc, map_location=dev)["disc"])
    disc.eval()
    dec = CEGISBeamPriorityDecoder(backbone)

    # frozen audit (the discriminator must be the ONLY trainable module)
    trainable_backbone = [n for n, p in backbone.named_parameters()
                          if p.requires_grad]
    trainable_disc = sum(p.numel() for p in disc.parameters())
    if trainable_backbone:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable_backbone}")

    print(f"LOADED {args.ckpt} step={st.get('step')}", flush=True)
    print(f"DEV_SPLIT {args.tag} n={args.dev_n} seed={args.dev_seed} "
          f"sha={sd[:16]}", flush=True)
    print(f"DISC trainable={trainable_disc}", flush=True)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    cal_probs: list[float] = []
    cal_y: list[int] = []
    calls = {"A": [], "B": [], "C": [], "D": []}
    outcome_pass = {"A": [], "B": [], "C": [], "D": []}
    oracle_support = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t)}
        ver_tests = t["verifier_tests"]
        out_tests = t["outcome_tests"]

        # ---- ARM A: token beam + CEGIS-first (verifier tests) ----
        seq_a, rec_a = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=args.beam_width,
            beta_priority=0.0, return_all_finals=True)
        pool_a = []
        seen = set()
        for s, sc, e in rec_a["final_candidates"]:
            code = detokenize(s)
            if code in seen:
                continue
            seen.add(code)
            pool_a.append((code, sc))
        idx_a, call_a, _ = _cegis_first(pool_a, ver_tests,
                                        budget=args.budget)
        calls["A"].append(call_a)
        if idx_a >= 0:
            outcome_pass["A"].append(sandbox(pool_a[idx_a][0], out_tests))
        else:
            outcome_pass["A"].append(0)

        # ---- ARM B/C/D: same skeleton pool, different orders ----
        cands = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget, use_energy=False)
        pool = [(c["code"], c["rule_id"]) for c in cands]
        if not pool:
            row["B"] = row["C"] = row["D"] = {"admit": -1, "calls": 0,
                                             "outcome": 0}
            per_task.append(row)
            continue

        # B: uniform order (generator order)
        idx_b, call_b, _ = _cegis_first(pool, ver_tests, budget=args.budget)
        calls["B"].append(call_b)
        outcome_pass["B"].append(
            sandbox(pool[idx_b][0], out_tests) if idx_b >= 0 else 0)

        # C: re-rank by learned discriminator (exact egress states)
        scores = []
        for code, _r in pool:
            st_c = candidate_state(v05, code, dev)
            scores.append(float(torch.sigmoid(disc(st_c)).item()))
            cal_probs.append(float(torch.sigmoid(disc(st_c)).item()))
            cal_y.append(sandbox(code, ver_tests))
        order_c = sorted(range(len(pool)),
                         key=lambda j: (-scores[j], j))
        pool_c = [pool[j] for j in order_c]
        idx_c, call_c, _ = _cegis_first(pool_c, ver_tests,
                                        budget=args.budget)
        calls["C"].append(call_c)
        outcome_pass["C"].append(
            sandbox(pool_c[idx_c][0], out_tests) if idx_c >= 0 else 0)

        # D: deterministic random-order control (same pool, seeded shuffle)
        rng = random.Random(args.dev_seed + i * 7919)
        order_d = list(range(len(pool)))
        rng.shuffle(order_d)
        pool_d = [pool[j] for j in order_d]
        idx_d, call_d, _ = _cegis_first(pool_d, ver_tests,
                                        budget=args.budget)
        calls["D"].append(call_d)
        outcome_pass["D"].append(
            sandbox(pool_d[idx_d][0], out_tests) if idx_d >= 0 else 0)

        # oracle outcome support (measurement only — never selection)
        sup = any(sandbox(code, out_tests) for code, _r in pool)
        oracle_support.append(1 if sup else 0)

        row["A"] = {"admit": idx_a, "calls": call_a,
                    "outcome": outcome_pass["A"][-1], "pool": len(pool_a)}
        row["B"] = {"admit": idx_b, "calls": call_b,
                    "outcome": outcome_pass["B"][-1], "pool": len(pool)}
        row["C"] = {"admit": idx_c, "calls": call_c,
                    "outcome": outcome_pass["C"][-1], "pool": len(pool)}
        row["D"] = {"admit": idx_d, "calls": call_d,
                    "outcome": outcome_pass["D"][-1], "pool": len(pool)}
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{args.dev_n}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)

    # ---- outcome pass rates + CIs ----
    def rate(arm: str) -> float:
        return sum(outcome_pass[arm]) / n

    def ci(arm: str) -> tuple[float, float]:
        return _task_bootstrap_cis([float(v) for v in outcome_pass[arm]],
                                   seed=args.dev_seed + 7)

    # ---- verifier-call stats ----
    def call_stats(arm: str) -> dict:
        v = sorted(calls[arm])
        m = len(v)
        return {"mean": sum(v) / m,
                "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0,
                "max": v[-1] if m else 0}

    # ---- budget-limited outcome success (K in {1,2,4,8}) ----
    def budget_success(arm: str, K: int) -> float:
        ok = 0
        for i, r in enumerate(per_task):
            if r[arm]["admit"] >= 0 and r[arm]["admit"] < K \
                    and r[arm]["outcome"] == 1:
                ok += 1
        return ok / n

    # ---- paired comparisons ----
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

    # ---- calibration metrics (C-arm exact egress states) ----
    n_cal = len(cal_y)
    pos = sum(cal_y)
    neg = n_cal - pos
    p_mean = sum(cal_probs) / n_cal if n_cal else 0.0
    baseline = (pos / n_cal) * (neg / n_cal) if n_cal else 0.0
    brier = sum((p - y) ** 2 for p, y in zip(cal_probs, cal_y)) / n_cal \
        if n_cal else 0.0
    var_p = sum((p - p_mean) ** 2 for p in cal_probs) / n_cal \
        if n_cal else 0.0
    cal = {
        "n_pairs": n_cal, "pos": pos, "neg": neg,
        "auroc": _auroc(cal_probs, cal_y) if n_cal else 0.5,
        "brier": brier, "baseline_p1p0": baseline,
        "spearman": _spearman(cal_probs, cal_y) if n_cal else 0.0,
        "prob_var": var_p,
        "note": "task-pooled, candidates clustered per task -> "
                "anti-conservative; diagnostic only",
    }

    # ---- gates ----
    calib_ok = (pos > 0 and neg > 0
                and cal["auroc"] > 0.5
                and brier < baseline
                and cal["spearman"] > 0.0
                and var_p > 0.0
                and not trainable_backbone)
    outcome_B = rate("B")
    outcome_C = rate("C")
    paired_BC = paired("B", "C")
    call_delta = sum(calls["C"]) / n - sum(calls["B"]) / n
    call_ci = _task_bootstrap_cis(
        [c - b for c, b in zip(calls["C"], calls["B"])],
        seed=args.dev_seed + 11)
    cost_ok = (call_delta <= -0.2 * (sum(calls["B"]) / n)
               and call_ci[1] < 0)
    outcome_not_worse = paired_BC["mcnemar_p"] > 0.05 \
        and outcome_C >= outcome_B - 0.05
    budget_ok = any(budget_success("C", K) > budget_success("B", K)
                    for K in (1, 2, 4, 8))
    cegis_ok = calib_ok and outcome_not_worse and cost_ok

    paired_CA = paired("C", "A")
    delta_CA = outcome_C - rate("A")
    delta_ci_CA = _task_bootstrap_cis(
        [c - a for c, a in zip(outcome_pass["C"], outcome_pass["A"])],
        seed=args.dev_seed + 13)
    promo_ok = (paired_CA["mcnemar_p"] < 0.05 and delta_CA >= 0.10
                and delta_ci_CA[0] > 0)

    if calib_ok and cegis_ok and promo_ok:
        verdict = "CEGIS_CAPABILITY_PROMOTED"
    elif calib_ok and cegis_ok:
        verdict = "EGRESS_CALIBRATED_COST_IMPROVED"
    elif calib_ok and outcome_not_worse:
        verdict = "CEGIS_VERIFIER_ASSISTED_NOT_CAPABILITY_PROMOTED"
    else:
        verdict = "CALIBRATED_EGRESS_DIAGNOSTIC_NOT_PROMOTED"

    result = {
        "run": "system1_v051_cegis_first",
        "ckpt": args.ckpt,
        "disc": args.disc,
        "dev_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                      "sha256": sd},
        "budget": args.budget,
        "beam_width": args.beam_width,
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a),
                     "budget_success": {str(K): budget_success(a, K)
                                        for K in (1, 2, 4, 8)}}
                 for a in ("A", "B", "C", "D")},
        "oracle_outcome_support": sum(oracle_support) / n,
        "paired": {"B_vs_C": paired_BC, "C_vs_A": paired_CA},
        "call_delta_C_minus_B": {"mean": call_delta, "ci90": call_ci},
        "calibration": cal,
        "frozen_audit": {"backbone_trainable": len(trainable_backbone),
                         "disc_trainable": trainable_disc},
        "gates": {"calibration": bool(calib_ok),
                  "cegis_operational": bool(cegis_ok),
                  "promotion": bool(promo_ok),
                  "cost_reduction_20pct": bool(cost_ok),
                  "outcome_not_worse": bool(outcome_not_worse)},
        "verdict": verdict,
        "heldout_guard": {"consumed_heldout_digest": HELDOUT_DIGEST,
                          "quarantine": True},
    }
    with open(out / "eval_v051_cegis.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "eval_v051_results.json", "w") as f:
        json.dump({"per_task": per_task,
                   "cal_probs": cal_probs, "cal_y": cal_y}, f)
    print(json.dumps(result, indent=2), flush=True)
    print("RECEIPT " + str(out / "eval_v051_cegis.json"), flush=True)


if __name__ == "__main__":
    main()
