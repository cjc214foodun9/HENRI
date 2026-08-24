"""
System-1 v0.6.1-dev: fast-weight epistemic memory efficacy (dev only).
=============================================================================
Pre-registered 2026-08-24 (supplied-artifact feature; reference
system1-structural-egress-cegis.md; staged isolation per premise audit:
v0.6.0 retrieval DONE (NO_EFFECT) -> v0.6.1 fast weights -> v0.6.2 partition).

Question: does failure-downweighting via factorized fast-weight memory
(FastWeightRuleMemory, U in R^{r x N}, lambda=0.95) change candidate order
during admission and raise outcome pass vs the uniform CEGIS-first baseline?

Mechanism (honest):
  base prior per candidate = skeleton-head prob (use_energy=False ->
  energy_score = 0.5 * prob). Admission loop scans pool in order; each
  verifier failure records one rank-1 slot (U <- lam*U + eta*e_onehot);
  remaining unseen candidates re-sorted by adjusted_probs(prior) after each
  failure. reset_each_task=True (no cross-task leakage). eta=0/disabled ->
  order byte-identical to baseline (identity gate, contract C8).

Integrity:
- Fresh DISPOSABLE split dev7_v061 (seed 71727, never used). dev6_v060
  (d6b79d51) and ALL consumed digests refused.
- Frozen backbone (v0.4.1 ckpt), grammar 13-rule as frozen (rule-10 defect
  structural; family 10 reported separately).
- Disjoint verifier/outcome 4+4. Matched budget 64 / beam 64.
- Same verifier-call count as baseline (reordering adds no sandbox calls).

Arms:
  A    token beam + CEGIS-first                     [baseline]
  B13  skeleton uniform CEGIS-first                 [control]
  D    skeleton + fast-weight failure-downweighted admission  [intervention]

Gates:
  G0 IDENTITY        D with eta=0 == B13 order on first 5 tasks.
  G1 EFFICACY        D pass >= B13 pass; promoted only if McNemar p<0.05.
  G2 NO_REGRESSION   D old-family pass >= B13 old-family pass - 0.10.
  G3 COST            D mean calls <= B13 mean calls * 1.5.
  G4 DIVERSITY       D distinct >= B13 distinct - 0.5.
  Kill: G2/G3 violated -> FASTWEIGHT_REGRESSION (do not promote).

Verdicts (pre-registered):
  FASTWEIGHT_EFFICACY_PROMOTED   G1+G2+G3+G4, p<0.05, D better
  FASTWEIGHT_NO_EFFECT           zero discordance
  FASTWEIGHT_NO_IMPROVEMENT      otherwise
  FASTWEIGHT_REGRESSION          G2/G3 violated

DEV ONLY. No heldout claim.
"""
from __future__ import annotations

import argparse
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
    System1KernelV04, detokenize, KernelV04Config)
from system1_kernel_v042_cegis_beam import CEGISBeamPriorityDecoder  # noqa: E402
from system1_kernel_v05_ast_skeleton import System1KernelV05  # noqa: E402
from train_system1_kernel_v04 import (  # noqa: E402
    sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from train_v051_discriminator import build_split, sha256_file  # noqa: E402
from zone_c_bridge_v060 import FastWeightRuleMemory  # noqa: E402

CONSUMED_DIGESTS = [
    "887d0d6c", "23b36795", "82c97532", "db027f9c", "9d4c29ad",
    "1f81e4d0", "181cc59b", "092cb0c1", "2eb8d29b", "78b4cfb4",
    "7a8c1e7b", "306ab62d", "6b5bb1b4", "0ec0528d", "0535a2dc",
    "35d15aae", "5e5f4a00", "ce2a76fb", "9a17af61", "635c2aaa",
    "a09bf275", "d6b79d51",
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
    for i, (code, _o) in enumerate(pool[:budget]):
        if sandbox(code, verifier_tests) == 1:
            return i, i + 1
    return -1, len(pool[:budget])


def _cegis_fastweight(pool: list[dict], verifier_tests: list[str],
                      budget: int = 64, rank: int = 8, eta: float = 0.5,
                      lam: float = 0.95) -> tuple[int, int]:
    """Admission with failure-downweighted reordering.

    pool: list of dicts {code, rule_id, prob}. Each failure updates one
    rank-1 slot; unseen candidates re-sorted by adjusted_probs after each
    failure. Returns (original_pool_index_of_first_pass, calls).
    """
    fw = FastWeightRuleMemory(num_rules=13, rank=rank, eta=eta, lam=lam)
    order = list(range(len(pool)))
    remaining = list(order)
    calls = 0
    for _ in range(budget):
        if not remaining:
            break
        i = remaining[0]
        calls += 1
        if sandbox(pool[i]["code"], verifier_tests) == 1:
            return i, calls
        fw.record_failure(pool[i]["rule_id"])
        rest = remaining[1:]
        if rest:
            probs = torch.tensor([pool[j]["prob"] for j in rest])
            rids = [pool[j]["rule_id"] for j in rest]
            adj = fw.adjusted_probs(probs, rule_ids=rids)
            rest = [j for _, j in sorted(
                zip([-float(a) for a in adj], rest))]
        remaining = rest
    return -1, calls


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=60)
    ap.add_argument("--dev-seed", type=int, default=71727)
    ap.add_argument("--tag", default="dev7_v061")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.95)
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    tasks = build_split(args.out, args.dev_n, args.dev_seed, args.tag,
                        n_families=13)
    split_p = out / f"{args.tag}.json"
    sd = sha256_file(split_p)
    if any(sd.startswith(d) for d in CONSUMED_DIGESTS):
        raise SystemExit(f"REFUSED: split {args.tag} consumed digest {sd[:12]}")
    fids = [t["fid"] for t in tasks]
    print(f"SPLIT {args.tag} n={len(tasks)} sha={sd[:12]} "
          f"families={sorted(set(fids))}", flush=True)

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05.eval()
    dec = CEGISBeamPriorityDecoder(backbone)

    trainable = [n for n, p in backbone.named_parameters()
                 if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable}")

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    calls = {"A": [], "B13": [], "D": []}
    outcome = {"A": [], "B13": [], "D": []}
    distinct = {"B13": [], "D": []}
    per_task: list[dict] = []
    identity_ok = True
    t0 = time.time()

    for i, t in enumerate(tasks):
        row = {"task": t["name"], "fid": t["fid"], "fp": fp_of(t)}
        ver = t["verifier_tests"]
        out_t = t["outcome_tests"]

        _, rec_a = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=args.beam_width,
            beta_priority=0.0, return_all_finals=True)
        pool_a, seen = [], set()
        for s, sc, _e in rec_a["final_candidates"]:
            code = detokenize(s)
            if code in seen:
                continue
            seen.add(code)
            pool_a.append((code, sc))
        idx_a, call_a = _cegis_first(pool_a, ver, budget=args.budget)
        calls["A"].append(call_a)
        outcome["A"].append(sandbox(pool_a[idx_a][0], out_t)
                            if idx_a >= 0 else 0)

        cands = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
            use_energy=False)
        pool_b = [(c["code"], c["rule_id"]) for c in cands]
        distinct["B13"].append(len({c["code"] for c in cands}))
        idx_b, call_b = _cegis_first(pool_b, ver, budget=args.budget)
        calls["B13"].append(call_b)
        outcome["B13"].append(sandbox(pool_b[idx_b][0], out_t)
                              if idx_b >= 0 else 0)

        # D: fast-weight pool with recoverable head probs
        pool_d = [{"code": c["code"], "rule_id": c["rule_id"],
                   "prob": c["energy_score"] * 2.0} for c in cands]
        distinct["D"].append(len({c["code"] for c in cands}))
        if i < 5:
            idx_d0, call_d0 = _cegis_fastweight(
                pool_d, ver, budget=args.budget, eta=0.0)
            if idx_d0 != idx_b:
                identity_ok = False
        idx_d, call_d = _cegis_fastweight(
            pool_d, ver, budget=args.budget, rank=args.rank,
            eta=args.eta, lam=args.lam)
        calls["D"].append(call_d)
        outcome["D"].append(sandbox(pool_d[idx_d]["code"], out_t)
                            if idx_d >= 0 else 0)

        row["B13"] = {"admit": idx_b, "calls": call_b,
                      "outcome": outcome["B13"][-1],
                      "distinct": distinct["B13"][-1]}
        row["D"] = {"admit": idx_d, "calls": call_d,
                    "outcome": outcome["D"][-1],
                    "distinct": distinct["D"][-1]}
        row["A"] = {"admit": idx_a, "calls": call_a,
                    "outcome": outcome["A"][-1]}
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{len(tasks)}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)

    def rate(arm):
        return sum(outcome[arm]) / n

    def rate_family(arm, fid_cond):
        vals = [outcome[arm][i] for i, t in enumerate(tasks)
                if fid_cond(t["fid"])]
        return sum(vals) / len(vals) if vals else 0.0

    def ci(arm, vals=None):
        return _task_bootstrap_cis(
            [float(v) for v in (vals or outcome[arm])], seed=args.dev_seed + 7)

    def call_stats(arm):
        v = sorted(calls[arm])
        m = len(v)
        return {"mean": sum(v) / m, "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0,
                "max": v[-1] if m else 0}

    def paired(x, y):
        both = xo = yo = neither = 0
        for i in range(n):
            xv, yv = outcome[x][i], outcome[y][i]
            both += xv and yv
            xo += xv and not yv
            yo += yv and not xv
            neither += (not xv) and (not yv)
        return {"both": both, f"{x}_only": xo, f"{y}_only": yo,
                "neither": neither,
                "mcnemar_p": _mcnemar_two_sided(xo, yo)}

    old_mask = lambda f: f < 7
    b13_old = rate_family("B13", old_mask)
    d_old = rate_family("D", old_mask)

    g0 = identity_ok
    g1_pass = rate("D") >= rate("B13")
    g2 = d_old >= b13_old - 0.10
    g3 = call_stats("D")["mean"] <= call_stats("B13")["mean"] * 1.5
    g4 = (sum(distinct["D"]) / n) >= (sum(distinct["B13"]) / n) - 0.5
    mc = paired("D", "B13")

    if g1_pass and g2 and g3 and g4 and mc["mcnemar_p"] < 0.05:
        verdict = "FASTWEIGHT_EFFICACY_PROMOTED"
    elif not (g2 and g3):
        verdict = "FASTWEIGHT_REGRESSION"
    elif mc["D_only"] == 0 and mc["B13_only"] == 0:
        verdict = "FASTWEIGHT_NO_EFFECT"
    else:
        verdict = "FASTWEIGHT_NO_IMPROVEMENT"

    family_support = {}
    for fid in sorted(set(fids)):
        family_support[fid] = {
            "B13": rate_family("B13", lambda f, _f=fid: f == _f),
            "D": rate_family("D", lambda f, _f=fid: f == _f)}

    result = {
        "run": "system1_v061_fastweight_dev",
        "ckpt": args.ckpt,
        "dev_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                      "sha256": sd},
        "budget": args.budget, "beam_width": args.beam_width,
        "fw": {"rank": args.rank, "eta": args.eta, "lam": args.lam,
               "reset_each_task": True},
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a)} for a in outcome},
        "distinct_mean": {a: sum(distinct[a]) / n for a in distinct},
        "family_support": family_support,
        "gates": {"G0_identity": g0, "G1_pass": g1_pass,
                  "G2_no_regression": g2, "G3_cost": g3,
                  "G4_diversity": g4},
        "paired_D_vs_B13": mc,
        "verdict": verdict,
        "note": ("DEV ONLY. Family 10 structurally unsupported (frozen "
                 "grammar rule-10 TypeError); reported for diagnosis only."),
    }
    (out / "eval_v061_results.json").write_text(json.dumps(result, indent=2))
    (out / "eval_v061_per_task.json").write_text(json.dumps(per_task))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
