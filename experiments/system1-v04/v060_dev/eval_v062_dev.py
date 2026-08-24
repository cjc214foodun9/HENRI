"""
System-1 v0.6.2-dev: heterogeneous particle sub-swarm partition (dev only).
=============================================================================
Pre-registered 2026-08-24 (supplied-artifact feature; reference
system1-structural-egress-cegis.md; staged isolation: v0.6.0 retrieval DONE
(NO_EFFECT), v0.6.1 fast weights, v0.6.2 partition).

Question: does sub-swarm partitioning (PartitionOrder: arity sub-swarms +
argument-name rotation across sweeps) raise DISTINCT PROGRAMS per task at
matched budget, and does it change outcome pass vs the uniform CEGIS-first
baseline?

Honest mechanism:
  - Sub-swarms = arity groups (structural premise; not a family oracle).
  - Within each sweep, rules ordered by the skeleton head's learned probs
    (per-arity), so the learned prior is respected.
  - Argument-name rotation (in-vocab names only; contract C10) produces
    DISTINCT CODE STRINGS with identical semantics (positional call
    convention) -> distinct-programs metric rises; semantic pool unchanged.
  - The upload's "Distinct 3.63 -> 15.0+" is a HYPOTHESIS on a different
    substrate; here B13 baseline distinct is ~7.25, and the honest claim
    under test is E distinct >= 2x B13 distinct.

Integrity:
- Fresh DISPOSABLE split dev8_v062 (seed 81828, never used). dev6_v060
  (d6b79d51), dev7_v061 and ALL consumed digests refused.
- Frozen backbone, grammar 13-rule as frozen (family 10 structural).
- Disjoint verifier/outcome 4+4. Matched budget 64 / beam 64.

Arms:
  A    token beam + CEGIS-first                     [baseline]
  B13  skeleton uniform CEGIS-first                 [control]
  E    skeleton + partition rotation (sweeps x arg rotation)  [intervention]

Gates:
  G0 IDENTITY        E with sweeps=0 (baseline arg names) == B13 pool.
  G1 EFFICACY        E pass >= B13 pass; promoted only if McNemar p<0.05.
  G2 NO_REGRESSION   E old-family pass >= B13 old-family pass - 0.10.
  G3 COST            E mean calls <= B13 mean calls * 1.5.
  G4 DIVERSITY       E distinct-mean >= 2x B13 distinct-mean (claim test).
  Kill: G2/G3 violated -> PARTITION_REGRESSION.

Verdicts (pre-registered):
  PARTITION_EFFICACY_PROMOTED   G1+G2+G3+G4, p<0.05, E better
  PARTITION_DIVERSITY_ONLY      G4 TRUE but no efficacy (p>=0.05 or
                                zero discordance): diversity claim verified,
                                efficacy not established
  PARTITION_NO_EFFECT           identical outcomes, diversity unmet
  PARTITION_NO_IMPROVEMENT      otherwise
  PARTITION_REGRESSION          G2/G3 violated

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
from zone_c_bridge_v060 import PartitionOrder  # noqa: E402

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


def build_partition_pool(v05, z0, sp, task, budget: int = 64,
                         sweeps: int = 3) -> list[tuple]:
    """E-arm pool: sweep-major rotation of in-vocab arg names.

    sweeps=0 -> baseline arg names (identity gate).
    Within each sweep, rules are arity-filtered and ordered by the head's
    learned probs (from generate_skeleton_candidates), then rotated.
    """
    cands = v05.generate_skeleton_candidates(
        z0, sp, task, top_k=budget, use_energy=False)
    probs = {c["rule_id"]: c["energy_score"] for c in cands}
    g = v05.grammar
    po = PartitionOrder(num_rules=g.N_RULES, p=max(1, sweeps))
    fname = task["name"]
    nargs = task["nargs"]
    rids = [rid for rid in range(g.N_RULES) if g.RULES[rid][2] == nargs]
    rids.sort(key=lambda r: -probs.get(r, 0.0))  # learned-prior order
    seen: set[str] = set()
    pool: list[tuple] = []
    if sweeps == 0:
        for rid in rids:
            names = ["xs", "t1"][:nargs] if nargs <= 2 else \
                ["xs", "ys", "zs"][:nargs]
            code = g.instantiate(rid, fname, names)
            if code is None or code in seen:
                continue
            seen.add(code)
            pool.append((code, rid))
    else:
        for s in range(sweeps):
            for rid in rids:
                names = po.arg_rotation(rid, nargs, s)
                code = g.instantiate(rid, fname, names)
                if code is None or code in seen:
                    continue
                seen.add(code)
                pool.append((code, rid))
    return pool


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=60)
    ap.add_argument("--dev-seed", type=int, default=81828)
    ap.add_argument("--tag", default="dev8_v062")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--sweeps", type=int, default=3)
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

    calls = {"A": [], "B13": [], "E": []}
    outcome = {"A": [], "B13": [], "E": []}
    distinct = {"B13": [], "E": []}
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

        # E: partition pool
        pool_e = build_partition_pool(v05, z0[i:i + 1], sp[i:i + 1], t,
                                      budget=args.budget,
                                      sweeps=args.sweeps)
        distinct["E"].append(len({c for c, _r in pool_e}))
        if i < 5 and args.sweeps != 0:
            pool_e0 = build_partition_pool(
                v05, z0[i:i + 1], sp[i:i + 1], t, budget=args.budget,
                sweeps=0)
            if [c for c, _r in pool_e0] != [c for c, _r in pool_b]:
                identity_ok = False
        idx_e, call_e = _cegis_first(pool_e, ver, budget=args.budget)
        calls["E"].append(call_e)
        outcome["E"].append(sandbox(pool_e[idx_e][0], out_t)
                            if idx_e >= 0 else 0)

        row["B13"] = {"admit": idx_b, "calls": call_b,
                      "outcome": outcome["B13"][-1],
                      "distinct": distinct["B13"][-1]}
        row["E"] = {"admit": idx_e, "calls": call_e,
                    "outcome": outcome["E"][-1],
                    "distinct": distinct["E"][-1]}
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
    e_old = rate_family("E", old_mask)
    d_mean = {"B13": sum(distinct["B13"]) / n,
              "E": sum(distinct["E"]) / n}

    g0 = identity_ok
    g1_pass = rate("E") >= rate("B13")
    g2 = e_old >= b13_old - 0.10
    g3 = call_stats("E")["mean"] <= call_stats("B13")["mean"] * 1.5
    g4 = d_mean["E"] >= 2.0 * d_mean["B13"]
    mc = paired("E", "B13")

    if g1_pass and g2 and g3 and g4 and mc["mcnemar_p"] < 0.05:
        verdict = "PARTITION_EFFICACY_PROMOTED"
    elif not (g2 and g3):
        verdict = "PARTITION_REGRESSION"
    elif g4 and (mc["E_only"] == 0 and mc["B13_only"] == 0):
        verdict = "PARTITION_DIVERSITY_ONLY"
    elif mc["E_only"] == 0 and mc["B13_only"] == 0:
        verdict = "PARTITION_NO_EFFECT"
    else:
        verdict = "PARTITION_NO_IMPROVEMENT"

    family_support = {}
    for fid in sorted(set(fids)):
        family_support[fid] = {
            "B13": rate_family("B13", lambda f, _f=fid: f == _f),
            "E": rate_family("E", lambda f, _f=fid: f == _f)}

    result = {
        "run": "system1_v062_partition_dev",
        "ckpt": args.ckpt,
        "dev_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                      "sha256": sd},
        "budget": args.budget, "beam_width": args.beam_width,
        "sweeps": args.sweeps,
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a)} for a in outcome},
        "distinct_mean": d_mean,
        "family_support": family_support,
        "gates": {"G0_identity": g0, "G1_pass": g1_pass,
                  "G2_no_regression": g2, "G3_cost": g3,
                  "G4_diversity_2x": g4},
        "paired_E_vs_B13": mc,
        "verdict": verdict,
        "note": ("DEV ONLY. Family 10 structurally unsupported (frozen "
                 "grammar rule-10 TypeError); reported for diagnosis only. "
                 "Rotation changes code strings with identical semantics "
                 "(positional calls) -> diversity metric, not semantics."),
    }
    (out / "eval_v062_results.json").write_text(json.dumps(result, indent=2))
    (out / "eval_v062_per_task.json").write_text(json.dumps(per_task))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
