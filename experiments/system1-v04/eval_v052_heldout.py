"""
System-1 v0.5.2 heldout verification: Arm B (uniform CEGIS-first) vs Arm A
(token beam) on a FRESH single-use heldout split.
=============================================================================
Pre-registered 2026-08-24 (reference system1-structural-egress-cegis.md,
v0.5.1 verdict dca9e90).

Integrity contract:
- 887d0d6c... (smoke40_v04) is CONSUMED and refused by guard. It is NOT a
  fresh holdout. Running it as heldout = replay = INVALID_VERIFIER_REPLAY.
- CONSUMED_DIGESTS lists every split digest already used by any smoke,
  contract, dev, train, or heldout run. Any match is refused.
- The heldout split is sealed ONCE (--seal-only) with a new seed never used
  anywhere; the seal receipt records sha256 + seed + tag + UTC BEFORE any
  evaluation. The CUDA run loads the sealed split file and verifies the sha.
- No CPU smoke or preflight may load the heldout split. Smoke uses a
  disposable tag/seed only.
- Arms A and B use DISJOINT verifier/outcome tests (4+4 per task, input-
  uniqueness-guaranteed partition from train_v051_discriminator.build_split).
- Discriminator is NOT part of the promotion comparison (learned C ordering
  showed zero outcome change in v0.5.1; capability carrier is uniform
  CEGIS-first over the skeleton pool).

Pre-registered verdicts (heldout phase):
- B > A significant (McNemar p<0.05), delta >= +0.10, task-blocked CI lb > 0,
  and admitted-program validity >= 0.95:
      CEGIS_VERIFIER_ASSISTED_CAPABILITY_PROMOTED
- delta > 0 but not significant:
      CONDITIONAL_VERIFIER_ASSISTED_IMPROVEMENT
- delta <= 0:
      HELDOUT_PROMOTION_NOT_ESTABLISHED
- guard refusal / leakage:
      INVALID_VERIFIER_REPLAY (raised at startup)
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
    build_split, sha256_file, N_VERIFIER, N_OUTCOME)

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
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=271828)
    ap.add_argument("--tag", default="heldout51_v052")
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

    # ---- split: load sealed file if present, else generate ----
    split_p = out / f"{args.tag}.json"
    if split_p.exists():
        sd = sha256_file(split_p)
        tasks = json.loads(split_p.read_text())
    else:
        tasks = build_split(args.out, args.dev_n, args.dev_seed, args.tag)
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
        receipt = {
            "tag": args.tag, "n": len(tasks), "seed": args.dev_seed,
            "sha256": sd, "utc": datetime.datetime.now(
                datetime.timezone.utc).isoformat(),
            "phase": "heldout", "single_use": True,
            "verifier_per_task": N_VERIFIER, "outcome_per_task": N_OUTCOME,
            "generator": "train_v051_discriminator.build_split",
        }
        rec_p = out / f"seal_{args.tag}.json"
        rec_p.write_text(json.dumps(receipt, indent=2))
        print(f"SEALED {args.tag} n={len(tasks)} seed={args.dev_seed} "
              f"sha={sd}")
        print("SEAL_RECEIPT " + str(rec_p))
        return

    # ---- full evaluation path ----
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone).to(dev)
    v05.eval()
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
    calls = {"A": [], "B": []}
    outcome_pass = {"A": [], "B": []}
    ast_valid = {"A": [], "B": []}
    oracle_support = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        row: dict = {"task": t["name"], "fp": fp_of(t)}
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

        # ---- ARM B: skeleton pool, uniform generator order ----
        cands = v05.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget, use_energy=False)
        pool = [(c["code"], c["rule_id"]) for c in cands]
        if pool:
            idx_b, call_b = _cegis_first(pool, ver_tests, budget=args.budget)
        else:
            idx_b, call_b = -1, 0
        calls["B"].append(call_b)
        if idx_b >= 0:
            code_b = pool[idx_b][0]
            outcome_pass["B"].append(sandbox(code_b, out_tests))
            ast_valid["B"].append(_ast_ok(code_b))
        else:
            outcome_pass["B"].append(0)
            ast_valid["B"].append(0)

        sup = any(sandbox(code, out_tests) for code, _r in pool)
        oracle_support.append(1 if sup else 0)

        row["A"] = {"admit": idx_a, "calls": call_a,
                    "outcome": outcome_pass["A"][-1],
                    "ast_valid": ast_valid["A"][-1], "pool": len(pool_a)}
        row["B"] = {"admit": idx_b, "calls": call_b,
                    "outcome": outcome_pass["B"][-1],
                    "ast_valid": ast_valid["B"][-1], "pool": len(pool)}
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
        return sum(ast_valid[arm]) / n

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
    paired_BA = paired("B", "A")
    delta_BA = rate("B") - rate("A")
    delta_ci_BA = _task_bootstrap_cis(
        [b - a for b, a in zip(outcome_pass["B"], outcome_pass["A"])],
        seed=args.dev_seed + 17)
    promo = (paired_BA["mcnemar_p"] < 0.05 and delta_BA >= 0.10
             and delta_ci_BA[0] > 0)
    min_ast = min(ast_rate("A"), ast_rate("B"))
    valid_preserved = min_ast >= 0.95

    if promo and valid_preserved:
        verdict = "CEGIS_VERIFIER_ASSISTED_CAPABILITY_PROMOTED"
    elif delta_BA > 0:
        verdict = "CONDITIONAL_VERIFIER_ASSISTED_IMPROVEMENT"
    else:
        verdict = "HELDOUT_PROMOTION_NOT_ESTABLISHED"

    result = {
        "run": "system1_v052_heldout_ab",
        "ckpt": args.ckpt,
        "heldout_split": {"tag": args.tag, "n": n, "seed": args.dev_seed,
                          "sha256": sd},
        "budget": args.budget,
        "beam_width": args.beam_width,
        "runtime_s": time.time() - t0,
        "arms": {a: {"outcome_pass_rate": rate(a), "ci90": ci(a),
                     "calls": call_stats(a),
                     "ast_valid_rate": ast_rate(a),
                     "budget_success": {str(K): budget_success(a, K)
                                        for K in (1, 2, 4, 8)}}
                 for a in ("A", "B")},
        "oracle_outcome_support": sum(oracle_support) / n,
        "paired": {"B_vs_A": paired_BA},
        "delta_B_minus_A": {"mean": delta_BA, "ci90": delta_ci_BA},
        "gates": {"promotion": bool(promo),
                  "validity_preserved": bool(valid_preserved),
                  "min_ast_valid": min_ast},
        "verdict": verdict,
        "heldout_guard": {"consumed_digests": CONSUMED_DIGESTS,
                          "single_use": True, "quarantine": True},
    }
    with open(out / "eval_v052_heldout.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(out / "eval_v052_results.json", "w") as f:
        json.dump({"per_task": per_task}, f)
    print(json.dumps(result, indent=2), flush=True)
    print("RECEIPT " + str(out / "eval_v052_heldout.json"), flush=True)


if __name__ == "__main__":
    main()
