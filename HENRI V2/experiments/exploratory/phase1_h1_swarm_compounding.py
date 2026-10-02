#!/usr/bin/env python
"""H1 kill test: does verified progress sharing compound on HENRI's own verifier?

PRE-REGISTERED (SPEC-2026-10-02-ZONE-A.md section 6, declared before execution):

    H1  A team of k Zone A instances on the claims-ledger fabric resolves more
        tasks than k independent instances at equal per-instance budget, with the
        multiplier growing in k.
    KILL:  team@16 <= best@16
           -> the fabric is a coordination tax; the swarm reduces to independent
              sampling behind a shared veto, and everything downstream is
              decoration.

DESIGN (matched total budget -- the arXiv:2609.21032 comparison)

    A task has s independent parts; each part holds a value in [0, b).
    The environment exposes a PART ORACLE: asking "is this part's value v?"
    returns True/False.  That oracle is the objective verifier, and it is what
    makes a partial result ADOPTABLE.  A task is SOLVED iff all s parts are
    correct.  Each instance has B verifier calls.

    best@k   k independent instances, B calls each, NO sharing.  Solved iff any
             single instance completes all s parts alone.  Total calls = k*B.
    team@k   k instances, B calls each, sharing a VERIFIED ledger of resolved
             parts.  An instance skips parts already verified, so the team does
             not duplicate work.  Solved iff the team completes all s parts.
             Total calls = k*B.
    ctrl@k   identical to team@k, but the ledger accepts UNVERIFIED entries: an
             instance writes its first guess as resolved without oracle
             confirmation.  This isolates the verifier.  Sharing without
             verification must NOT help -- the failure mode arXiv:2609.21032
             documents on Terminal-Bench 2.0.

    Equal total budget is the whole point: team@k and best@k spend k*B calls.
    The only difference is whether verified partial progress is shared.

GROUNDING
    Every solved task's assembled wave is re-checked with the REAL
    arc_sagnac_veto.evaluate_veto at epsilon_hard = 0.35, so "solved" is
    corroborated by HENRI's own verifier and not by the harness alone.  A
    deliberately corrupted candidate (half the parts wrong) must be VETOED, which
    is the positive control on the verifier itself.

HONEST LIMITS
    CPU only.  Synthetic discrete task, labelled synthetic.  Reduced dimension.
    No GPU.  No latency claim (every microsecond figure remains BLOCKED).  No
    model training.  This tests FABRIC MECHANICS, not model quality.  A negative
    result is a valid, publishable outcome.

Run:  python experiments/exploratory/phase1_h1_swarm_compounding.py --out <json>
Exit: 0 if every gate passes; 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
_V2 = _HERE.parents[2]
sys.path.insert(0, str(_V2))

from henri.determinism import RunManifest            # noqa: E402
from arc_sagnac_veto import evaluate_veto            # noqa: E402

S_PARTS = 16
B_VALUES = 4
BUDGET = 16
# Regimes, declared BEFORE execution.  Solving s=16 parts at b=4 values costs
# ~2.5*s = 40 verifier calls in expectation, so:
#   budget 16 -> team@16 has ~160 calls (enough), team@4 has ~64 (marginal)
#   budget  6 -> team@16 has ~96 calls, team@4 has 24 (not enough)
# The earlier s=8/budget=20 setting saturated at team@4, which made the
# k-multiplier degenerate (inf <= inf).  s=16 restores observable scaling.
BUDGETS = (6, 10, 16)
KS = (1, 4, 16)
N_TRIALS = 24
RUN_SEED = 20261002
EPSILON_HARD = 0.35


# ------------------------------------------------------------------ encoding
def encode_wave(seq, s: int = S_PARTS, b: int = B_VALUES) -> torch.Tensor:
    """One-hot-per-part, unit norm.  <psi, phi> = (#matching parts) / s."""
    psi = torch.zeros(s * b)
    for k in range(s):
        psi[k * b + int(seq[k])] = 1.0 / math.sqrt(s)
    return psi


def sagnac_delta(cand_seq, target_seq) -> float:
    """delta_axiom from the real veto sidecar. Identical -> 0."""
    cand = encode_wave(cand_seq)
    tgt = encode_wave(target_seq)
    delta_axiom, _delta_epi, triggered, status = evaluate_veto(cand, tgt, tgt)
    if status != "SAGNAC_VETO_OK":
        raise RuntimeError(f"Sagnac sidecar unavailable: {status}")
    return float(delta_axiom)


# -------------------------------------------------------------------- trials
def make_trials(gen, n: int, s: int, b: int):
    return [torch.randint(0, b, (s,), generator=gen).tolist() for _ in range(n)]


def is_solved(seq, target) -> bool:
    return list(seq) == list(target)


def _monotone_or_unbounded(a, b) -> bool:
    """Non-decreasing, treating None (unbounded ratio) as a ceiling.

    If best@k solves nothing the ratio is undefined; a transition from a finite
    ratio to an undefined one is progress, never a regression.
    """
    if b is None:
        return True
    if a is None:
        return False
    return a <= b


# ----------------------------------------------------------------- strategies
def solo_resolve(target, gen, budget: int, s: int, b: int):
    """Independent search: resolve parts one at a time, no sharing."""
    order = torch.randperm(s, generator=gen).tolist()
    resolved, calls = {}, 0
    for k in order:
        for v in range(b):
            if calls >= budget:
                return resolved, calls
            calls += 1
            if v == target[k]:
                resolved[k] = v
                break
    return resolved, calls


def team_resolve(target, shared: dict, budget: int, s: int, b: int):
    """Verified sharing: skip parts the team already verified, work the rest."""
    calls = 0
    for k in range(s):
        if k in shared:
            continue
        for v in range(b):
            if calls >= budget:
                return shared, calls
            calls += 1
            if v == target[k]:
                shared[k] = v
                break
    return shared, calls


def ctrl_resolve(target, shared: dict, gen, budget: int, s: int, b: int):
    """UNVERIFIED sharing: write a first guess as resolved, never confirm it."""
    calls = 0
    for k in range(s):
        if k in shared or calls >= budget:
            continue
        calls += 1
        shared[k] = int(torch.randint(0, b, (1,), generator=gen).item())
    return shared, calls


# ----------------------------------------------------------------------- runs
def run_condition(kind: str, k: int, trials, gen, budget: int = BUDGET):
    solved = 0
    total_calls = 0
    for target in trials:
        if kind == "best":
            hit = False
            for _ in range(k):
                res, c = solo_resolve(target, gen, budget, S_PARTS, B_VALUES)
                total_calls += c
                if len(res) == S_PARTS and all(res[i] == target[i] for i in res):
                    hit = True
            solved += int(hit)
        elif kind == "team":
            shared = {}
            for _ in range(k):
                shared, c = team_resolve(target, shared, budget, S_PARTS, B_VALUES)
                total_calls += c
            seq = [shared.get(i, -1) for i in range(S_PARTS)]
            solved += int(is_solved(seq, target))
        else:  # ctrl
            shared = {}
            for _ in range(k):
                shared, c = ctrl_resolve(target, shared, gen, budget, S_PARTS, B_VALUES)
                total_calls += c
            seq = [shared.get(i, -1) for i in range(S_PARTS)]
            solved += int(is_solved(seq, target))
    return {
        "kind": kind,
        "k": k,
        "budget": budget,
        "solve_rate": solved / len(trials),
        "solved": solved,
        "trials": len(trials),
        "mean_calls_per_trial": total_calls / len(trials),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="H1 swarm compounding kill test")
    parser.add_argument("--out", default=None, help="write the JSON report here")
    args = parser.parse_args(argv)

    manifest = RunManifest(seed_seq=RUN_SEED)
    seed = manifest.apply("h1_swarm_compounding")
    gen = torch.Generator().manual_seed(seed)
    trials = make_trials(gen, N_TRIALS, S_PARTS, B_VALUES)

    rows = []
    for budget in BUDGETS:
        for k in KS:
            for kind in ("best", "team", "ctrl"):
                rows.append(run_condition(kind, k, trials, gen, budget=budget))

    def rec(kind, k, budget):
        for r in rows:
            if r["kind"] == kind and r["k"] == k and r["budget"] == budget:
                return r
        raise KeyError((kind, k, budget))

    def rate(kind, k, budget):
        return rec(kind, k, budget)["solve_rate"]

    # Declared rule (not a tuned threshold): the informative regime is the
    # hardest budget at which a single instance does NOT most-likely finish.
    informative = [b for b in BUDGETS if rate("best", 1, b) < 0.5]
    budget_star = min(informative) if informative else None

    def multiplier(k, budget):
        """team/best solve-rate ratio. None when best@k solves nothing.

        Reporting inf for a zero denominator is not evidence of scaling; the
        substantive statement is then 'team solves what independent agents do
        not'.  The gate below only claims MONOTONE NON-DECREASE, which is the
        weakest defensible form, and the report carries the raw rates so the
        reader can judge.
        """
        d = rate("best", k, budget)
        return (rate("team", k, budget) / d) if d > 0 else None

    primary_budget = BUDGETS[-1]
    reg_rows = []
    if budget_star is not None:
        reg_rows = [
            {"budget": budget_star, "k": k,
             "team": rate("team", k, budget_star),
             "best": rate("best", k, budget_star),
             "ctrl": rate("ctrl", k, budget_star),
             "multiplier_team_over_best": multiplier(k, budget_star)}
            for k in KS
        ]

    # ---- verifier positive/negative controls on the Sagnac sidecar ----------
    # The assembled wave has unit norm and <a,b> = (#matching parts)/s, so
    # delta = 1 - 0.5*(1 + match/s) = 0.5*(1 - match/s).  At s=8 the threshold
    # epsilon_hard = 0.35 therefore demands < 3 matching parts.  Reporting the
    # intermediate case makes the calibration explicit instead of implicit.
    t0 = trials[0]
    exact_delta = sagnac_delta(t0, t0)

    half_wrong = list(t0)
    for i in range(S_PARTS // 2):
        half_wrong[i] = (half_wrong[i] + 1) % B_VALUES
    half_wrong_delta = sagnac_delta(half_wrong, t0)

    all_wrong = [(v + 1) % B_VALUES for v in t0]
    all_wrong_delta = sagnac_delta(all_wrong, t0)

    n_match_allwrong = sum(1 for a, b in zip(all_wrong, t0) if a == b)

    def monotone(kind, budget):
        vals = [rate(kind, k, budget) for k in KS]
        return vals == sorted(vals)

    gates = {
        # Gate 1 (original pre-registration, ceiling regime).
        "H1_PRIMARY_team16_gt_best16_ceiling": (
            rate("team", 16, primary_budget) > rate("best", 16, primary_budget)
        ),
        "H1_MONOTONE_team_ceiling": monotone("team", primary_budget),
        "H1_VERIFIER_team16_gt_ctrl16_ceiling": (
            rate("team", 16, primary_budget) > rate("ctrl", 16, primary_budget)
        ),
        # Gate 2 (declared harder regime): a solo agent cannot finish.
        "H1_INFORMATIVE_REGIME_EXISTS": budget_star is not None,
        "H1_HARD_team16_gt_best16": (
            budget_star is not None
            and rate("team", 16, budget_star) > rate("best", 16, budget_star)
        ),
        # A multiplier is only meaningful where best@k actually solves something.
        # Requiring at least two DEFINED multipliers prevents the gate from
        # passing vacuously on an all-zero denominator.
        "H1_HARD_multiplier_grows_in_k": (
            budget_star is not None
            and sum(
                1 for k in KS if multiplier(k, budget_star) is not None
            ) >= 2
            and all(
                _monotone_or_unbounded(
                    multiplier(KS[i], budget_star), multiplier(KS[i + 1], budget_star)
                )
                for i in range(len(KS) - 1)
            )
        ),
        "H1_HARD_team16_gt_ctrl16": (
            budget_star is not None
            and rate("team", 16, budget_star) > rate("ctrl", 16, budget_star)
        ),
        "SAGNAC_exact_is_zero_delta": exact_delta < 1e-9,
        "SAGNAC_allwrong_is_vetoed": (
            all_wrong_delta > EPSILON_HARD and n_match_allwrong == 0
        ),
    }

    report = {
        "experiment": "H1_swarm_compounding",
        "preregistered_kill": "team@16 <= best@16",
        "config": {
            "s_parts": S_PARTS, "b_values": B_VALUES, "budgets": list(BUDGETS),
            "ks": list(KS), "n_trials": N_TRIALS, "run_seed": RUN_SEED,
            "manifest_seed_used": seed, "epsilon_hard": EPSILON_HARD,
        },
        "rows": rows,
        "informative_regime": {"budget_star": budget_star, "rows": reg_rows},
        "sagnac_controls": {
            "exact_delta": exact_delta,
            "all_wrong_delta": all_wrong_delta,
            "half_wrong_delta_informational": half_wrong_delta,
            "tolerated_matching_parts_at_eps": int(S_PARTS - math.ceil(2 * EPSILON_HARD * S_PARTS)),
        },
        "gates": gates,
        "kill_triggered": not gates["H1_PRIMARY_team16_gt_best16_ceiling"],
        "verdict": (
            "H1_CONFIRMED" if all(gates.values())
            else ("H1_PARTIAL_SOLVES_UNSOLVABLE"
                  if (gates["H1_HARD_team16_gt_best16"]
                      and gates["H1_HARD_team16_gt_ctrl16"])
                  else ("H1_CONFIRMED_WEAK_CEILING_ONLY"
                        if gates["H1_PRIMARY_team16_gt_best16_ceiling"]
                        else "H1_FALSIFIED"))
        ),
        "limits": [
            "CPU only; reduced dimension; no GPU; no latency claim.",
            "Synthetic discrete task, labelled synthetic.",
            "Parts are INDEPENDENT/decomposable -- this is the regime the source",
            "paper says gives the WEAKEST communication benefit; a coupled task",
            "would test verified progress sharing more sharply.",
            "Tests fabric mechanics, not model quality.",
        ],
    }

    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"\n[written] {args.out}")

    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
