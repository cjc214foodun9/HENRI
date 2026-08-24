"""
System-1 v0.5.3 grammar-manifold expansion evaluation (dev only).
=============================================================================
Pre-registered 2026-08-24 (post-HELDOUT cycle; reference
system1-structural-egress-cegis.md; run14 rank-dilution warning).

Question: does expanding the skeleton grammar from 7 to 13 production rules
BROADEN generative support Supp(P_gen) to the 6 new DSL families WITHOUT
displacing simple correct programs on the original 7 (run14 dilution)?

Integrity:
- Fresh DISPOSABLE split only (dev3_v053, seed never used). Heldout51_v052
  (5e5f4a00...) is consumed and refused.
- Backbone frozen (v0.4.1 ckpt); only num_rules differs between arms.
- Disjoint verifier/outcome tests (4+4) from train_v051_discriminator.

Arms (matched budget 64, beam 64):
  A   token beam + CEGIS-first (beta=0.0)                      [baseline]
  B7  skeleton UNIFORM CEGIS-first, grammar n_rules=7          [old]
  B13 skeleton UNIFORM CEGIS-first, grammar n_rules=13         [expanded]

Pre-registered gates:
  G1 SUPPORT_NEW   B13 outcome pass on NEW-family tasks (fid>=7) > 0,
                   task-blocked CI lb > 0 (support extended).
  G2 NO_DILUTION   B13 old-family pass >= B7 old-family pass - 0.10
                   (expansion does not displace old solutions).
  G3 OVERALL       B13 >= B7 overall; report paired McNemar B13_vs_B7 and
                   B13_vs_A (delta + CI).
  G4 COST          mean calls-to-first-pass B13 <= B7 * 1.5 (not explosive).
  Kill: G2 violated -> rank dilution confirmed -> do NOT promote 13-rule
        grammar as default; keep n_rules=7.
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

CONSUMED_DIGESTS = [
    "887d0d6c", "23b36795", "82c97532", "db027f9c", "9d4c29ad",
    "1f81e4d0", "181cc59b", "092cb0c1", "2eb8d29b", "78b4cfb4",
    "7a8c1e7b", "306ab62d", "6b5bb1b4", "0ec0528d", "0535a2dc",
    "35d15aae", "5e5f4a00", "ce2a76fb",
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
    ap.add_argument("--dev-seed", type=int, default=37123)
    ap.add_argument("--tag", default="dev3_v053")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--beam-width", type=int, default=64)
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

    # family composition check
    fids = [t["fid"] for t in tasks]
    new_fids = sorted({f for f in fids if f >= 7})
    old_fids = sorted({f for f in fids if f < 7})
    print(f"SPLIT {args.tag} n={len(tasks)} sha={sd[:12]} "
          f"old_families={old_fids} new_families={new_fids}", flush=True)

    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05_7 = System1KernelV05(backbone, num_rules=7).to(dev)
    v05_7.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05_13.eval()
    dec = CEGISBeamPriorityDecoder(backbone)

    trainable_backbone = [n for n, p in backbone.named_parameters()
                          if p.requires_grad]
    if trainable_backbone:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable_backbone}")

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    calls = {"A": [], "B7": [], "B13": []}
    outcome = {"A": [], "B7": [], "B13": []}
    distinct = {"B7": [], "B13": []}
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
        outcome["A"].append(sandbox(pool_a[idx_a][0], out_t) if idx_a >= 0 else 0)

        # B7 / B13: skeleton pools
        for arm, v05 in (("B7", v05_7), ("B13", v05_13)):
            cands = v05.generate_skeleton_candidates(
                z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
                use_energy=False)
            pool = [(c["code"], c["rule_id"]) for c in cands]
            uniq = {c["code"] for c in cands}
            distinct[arm].append(len(uniq))
            idx, call_n = _cegis_first(pool, ver, budget=args.budget)
            calls[arm].append(call_n)
            outcome[arm].append(sandbox(pool[idx][0], out_t) if idx >= 0 else 0)
            row[arm] = {"admit": idx, "calls": call_n,
                        "outcome": outcome[arm][-1],
                        "distinct": len(uniq), "pool": len(pool)}
        row["A"] = {"admit": idx_a, "calls": call_a,
                    "outcome": outcome["A"][-1], "pool": len(pool_a)}
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
                "p90": v[int(0.9 * (m - 1))] if m else 0.0, "max": v[-1] if m else 0}

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

    new_mask = lambda f: f >= 7
    old_mask = lambda f: f < 7
    b13_new = rate_family("B13", new_mask)
    b13_new_ci = ci("B13", [outcome["B13"][i] for i, t in enumerate(tasks) if new_mask(t["fid"])])
    b7_old = rate_family("B7", old_mask)
    b13_old = rate_family("B13", old_mask)

    g1 = b13_new > 0 and b13_new_ci[0] > 0
    g2 = b13_old >= b7_old - 0.10
    g3_overall = rate("B13") >= rate("B7")
    g4 = call_stats("B13")["mean"] <= call_stats("B7")["mean"] * 1.5

    result = {
        "run": "system1_v053_grammar_expansion",
        "ckpt": args.ckpt,
        "dev_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                      "sha256": sd},
        "budget": args.budget, "beam_width": args.beam_width,
        "runtime_s": time.time() - t0,
        "family_counts": {"old": sum(1 for f in fids if f < 7),
                          "new": sum(1 for f in fids if f >= 7)},
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a),
                     "mean_distinct": (sum(distinct[a]) / len(distinct[a])
                                       if a in distinct else None)}
                 for a in ("A", "B7", "B13")},
        "family_rates": {"B7_old": b7_old, "B13_old": b13_old,
                         "B13_new": b13_new, "B13_new_ci90": list(b13_new_ci)},
        "paired": {"B13_vs_B7": paired("B13", "B7"),
                   "B13_vs_A": paired("B13", "A")},
        "gates": {"support_new_families": bool(g1),
                  "no_dilution_old_families": bool(g2),
                  "overall_B13_ge_B7": bool(g3_overall),
                  "cost_not_explosive": bool(g4)},
        "kill": "DILUTION_CONFIRMED" if not g2 else
                ("SUPPORT_NOT_EXTENDED" if not g1 else None),
        "heldout_guard": {"consumed_digests": CONSUMED_DIGESTS,
                          "quarantine": True},
    }
    with open(out / "eval_v053_expansion.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "eval_v053_results.json", "w") as f:
        json.dump({"per_task": per_task}, f)
    print(json.dumps(result, indent=2), flush=True)
    print("RECEIPT " + str(out / "eval_v053_expansion.json"), flush=True)


if __name__ == "__main__":
    main()
