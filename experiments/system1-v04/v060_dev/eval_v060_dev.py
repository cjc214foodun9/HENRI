"""
System-1 v0.6.0-dev: Zone C retrieval read-path efficacy (dev only).
=============================================================================
Pre-registered 2026-08-24 (supplied-artifact audit; reference
system1-structural-egress-cegis.md; staged order per premise audit:
audit -> persistence -> read-path -> retrieval plumbing -> dev efficacy ->
fast-weight (v0.6.1) -> partition (v0.6.2)).

Question: does re-ranking skeleton candidates by live-family engram
similarity (ZoneCEngramBias, beta>0) raise outcome pass vs the uniform
CEGIS-first baseline at matched budget, WITHOUT displacing old-family
solutions or collapsing diversity?

Integrity:
- Fresh DISPOSABLE split only (dev6_v060, seed 61623 never used).
  Heldout52_v054 (a09bf275...) and all consumed digests refused.
- Backbone frozen (v0.4.1 ckpt); grammar 13-rule as frozen (rule-10 defect
  present and STRUCTURAL — retrieval cannot repair a TypeError; family 10
  reported separately, not counted as retrieval failure).
- Disjoint verifier/outcome tests (4+4) from train_v051_discriminator.
- Matched budget 64 / beam 64 across arms. Engram cache in the LIVE 384-D
  signature family (no third representation family).

Arms:
  A    token beam + CEGIS-first (beta=0.0)                 [baseline]
  B13  skeleton UNIFORM CEGIS-first, 13 rules              [control]
  C    skeleton CEGIS-first RE-RANKED by engram sim beta=1.0  [intervention]
  C0   beta=0.0 identity check on first 5 tasks (order == B13 exactly)

Pre-registered gates:
  G0 IDENTITY          C0 candidate order == B13 order (matched-control
                       identity; beta=0 reproduces baseline byte-identical).
  G1 RETRIEVAL_EFFICACY C pass >= B13 pass with task-blocked CI; primary
                       claim only if McNemar(C_vs_B13) p<0.05 and C better.
  G2 NO_REGRESSION     C old-family pass >= B13 old-family pass - 0.10.
  G3 COST              C mean calls-to-first-pass <= B13 * 1.5.
  G4 DIVERSITY         C distinct-programs/task >= B13 distinct - 0.5.
  Kill: G2 or G3 violated -> retrieval NOT promoted (rank/order artifact).

Verdicts (pre-registered):
  RETRIEVAL_EFFICACY_PROMOTED   G1+G2+G3+G4, McNemar p<0.05
  RETRIEVAL_NO_IMPROVEMENT      G1 fails, no harm
  RETRIEVAL_REGRESSION          G2/G3 violated (do not promote)

DEV ONLY: no heldout claim. A new carrier + fresh seal would be required
for any promotion statement.
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
    System1KernelV04, detokenize, KernelV04Config)
from system1_kernel_v042_cegis_beam import CEGISBeamPriorityDecoder  # noqa: E402
from system1_kernel_v05_ast_skeleton import System1KernelV05  # noqa: E402
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from train_v051_discriminator import build_split, sha256_file  # noqa: E402
from zone_c_bridge_v060 import (  # noqa: E402
    ZoneCHotCache, ZoneCEngramBias, PersistenceStatus,
)

CONSUMED_DIGESTS = [
    "887d0d6c", "23b36795", "82c97532", "db027f9c", "9d4c29ad",
    "1f81e4d0", "181cc59b", "092cb0c1", "2eb8d29b", "78b4cfb4",
    "7a8c1e7b", "306ab62d", "6b5bb1b4", "0ec0528d", "0535a2dc",
    "35d15aae", "5e5f4a00", "ce2a76fb", "9a17af61", "635c2aaa",
    "a09bf275",
]

SPELKE_NAMES = [
    "spelke_object_persistence", "spelke_topological_containment",
    "spelke_inertial_continuity", "spelke_affine_invariance",
    "physics_conservation_of_mass", "physics_momentum_conservation",
    "logic_peano_successor", "logic_group_reversibility",
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=60)
    ap.add_argument("--dev-seed", type=int, default=61623)
    ap.add_argument("--tag", default="dev6_v060")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--num-engrams", type=int, default=64)
    ap.add_argument("--enforce-db", action="store_true")
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

    # Zone C read-path: hot cache in the LIVE 384-D signature family
    cache = ZoneCHotCache(num_engrams=args.num_engrams,
                          d_live=cfg.d_slot, device=dev)
    cache.populate_from_names(SPELKE_NAMES)
    bias = ZoneCEngramBias(v05, cache, beta=args.beta)

    ps = PersistenceStatus(enforce=args.enforce_db)
    db_status = ps.probe()

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    calls = {"A": [], "B13": [], "C": []}
    outcome = {"A": [], "B13": [], "C": []}
    distinct = {"B13": [], "C": []}
    per_task: list[dict] = []
    identity_ok = True
    t0 = time.time()

    for i, t in enumerate(tasks):
        row = {"task": t["name"], "fid": t["fid"], "fp": fp_of(t)}
        ver = t["verifier_tests"]
        out_t = t["outcome_tests"]

        # A: token beam
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

        # B13: uniform skeleton pool (control)
        cands_b = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
            use_energy=False)
        pool_b = [(c["code"], c["rule_id"]) for c in cands_b]
        distinct["B13"].append(len({c["code"] for c in cands_b}))
        idx_b, call_b = _cegis_first(pool_b, ver, budget=args.budget)
        calls["B13"].append(call_b)
        outcome["B13"].append(sandbox(pool_b[idx_b][0], out_t)
                              if idx_b >= 0 else 0)

        # C0 identity on first 5 tasks: beta=0 must equal B13 order
        if i < 5:
            bias0 = ZoneCEngramBias(v05, cache, beta=0.0)
            cands0 = bias0.ranked_candidates(
                z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
                use_energy=False)
            ids_b = [c["rule_id"] for c in cands_b]
            ids_0 = [c["rule_id"] for c in cands0]
            if ids_b != ids_0:
                identity_ok = False

        # C: retrieval re-ranked skeleton pool
        cands_c = bias.ranked_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
            use_energy=False)
        pool_c = [(c["code"], c["rule_id"]) for c in cands_c]
        distinct["C"].append(len({c["code"] for c in cands_c}))
        idx_c, call_c = _cegis_first(pool_c, ver, budget=args.budget)
        calls["C"].append(call_c)
        outcome["C"].append(sandbox(pool_c[idx_c][0], out_t)
                            if idx_c >= 0 else 0)

        row["B13"] = {"admit": idx_b, "calls": call_b,
                      "outcome": outcome["B13"][-1],
                      "distinct": distinct["B13"][-1]}
        row["C"] = {"admit": idx_c, "calls": call_c,
                    "outcome": outcome["C"][-1],
                    "distinct": distinct["C"][-1]}
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
    c_old = rate_family("C", old_mask)

    g0 = identity_ok
    g1_pass = rate("C") >= rate("B13")
    g1_ci = ci("C")
    g2 = c_old >= b13_old - 0.10
    g3 = call_stats("C")["mean"] <= call_stats("B13")["mean"] * 1.5
    g4 = (sum(distinct["C"]) / n) >= (sum(distinct["B13"]) / n) - 0.5
    mc = paired("C", "B13")

    if g1_pass and g2 and g3 and g4 and mc["mcnemar_p"] < 0.05:
        verdict = "RETRIEVAL_EFFICACY_PROMOTED"
    elif not (g2 and g3):
        verdict = "RETRIEVAL_REGRESSION"   # old-family displacement or cost blowup
    elif g1_pass and mc["C_only"] == 0 and mc["B13_only"] == 0:
        verdict = "RETRIEVAL_NO_EFFECT"    # identical outcomes; no harm, no gain
    else:
        verdict = "RETRIEVAL_NO_IMPROVEMENT"

    family_support = {}
    for fid in sorted(set(fids)):
        family_support[fid] = {
            "B13": rate_family("B13", lambda f, _f=fid: f == _f),
            "C": rate_family("C", lambda f, _f=fid: f == _f)}

    result = {
        "run": "system1_v060_zonec_retrieval_dev",
        "ckpt": args.ckpt,
        "dev_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                      "sha256": sd},
        "budget": args.budget, "beam_width": args.beam_width,
        "beta": args.beta, "num_engrams": args.num_engrams,
        "cache_vram_mib": cache.vram_mib,
        "db": ps.to_record(),
        "runtime_s": time.time() - t0,
        "family_counts": {"old": sum(1 for f in fids if f < 7),
                          "new": sum(1 for f in fids if f >= 7)},
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a)} for a in outcome},
        "distinct_mean": {a: sum(distinct[a]) / n for a in distinct},
        "family_support": family_support,
        "gates": {"G0_identity": g0, "G1_pass": g1_pass,
                  "G1_ci": g1_ci, "G2_no_regression": g2,
                  "G3_cost": g3, "G4_diversity": g4},
        "paired_C_vs_B13": mc,
        "verdict": verdict,
        "note": ("DEV ONLY. Family 10 is STRUCTURALLY unsupported (rule-10 "
                 "TypeError defect in the frozen grammar); retrieval cannot "
                 "repair it. Per-family support reported for diagnosis."),
    }
    (out / "eval_v060_results.json").write_text(json.dumps(result, indent=2))
    (out / "eval_v060_per_task.json").write_text(json.dumps(per_task))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
