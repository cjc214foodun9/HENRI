"""
System-1 v0.4.2 paired CEGIS evaluator (dev-split; heldout quarantined).
===========================================================================
Pre-registered 2026-08-23 (roadmap Option 1 approved + upload cadf9788...):

  ARM A  beta_priority = 0.00  standard beam search (baseline)
  ARM B  beta_priority = 0.40  CEGIS Brier-prioritized beam search
  MATCH  same checkpoint, same tasks/order, same width, same expansions,
         same FSA/EOS/dead-row, same tie-breaking. Only beta differs.

Gates (pre-registered; no post-hoc beta tuning):
  ENGAGEMENT  fraction of tasks whose beam trajectory changed (best seq
              or best score) > 0 under beta=0.40.
  VALIDITY    ast_valid_rate >= 0.9 on both arms AND candidate
              energy/outcome Spearman > 0 (head stays calibrated).
  EFFICACY    beta=0.40 wins more paired tasks than it loses.
  PROMOTION   efficacy AND McNemar p < 0.05 AND delta >= 0.10 (with n=40,
              explicitly underpowered; reported, not decisive).
  If engagement > 0 but pass delta == 0: ENGAGED_DIAGNOSTIC_NOT_PROMOTED.

GUARD: refuses to run on any split whose fingerprint matches the consumed
heldout40_v04 digest (887d0d6c...). The old heldout is NEVER re-run.
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
    TOK2ID, KernelV04Config, System1KernelV04, detokenize)
from system1_kernel_v042_cegis_beam import (  # noqa: E402
    CEGISBeamPriorityDecoder)
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens,
    load_split, sha256_file)

HELDOUT_DIGEST = "887d0d6c"           # consumed heldout40_v04 (sealed)
BETA = 0.40                            # pre-registered hypothesis value


def _mcnemar_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    p = 0.0
    for k in range(min(b, c) + 1):
        p += math.comb(n, k) / (2.0 ** n)
    return 2.0 * min(p, 0.5)


def _spearman_raw(xs: list[float], ys: list[int]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    rx = {v: i for i, v in enumerate(sorted(set(xs)))}
    ry = {v: i for i, v in enumerate(sorted(set(ys)))}
    a = [rx[v] for v in xs]
    b = [ry[v] for v in ys]
    mx, my = sum(a) / n, sum(b) / n
    num = sum((u - mx) * (v - my) for u, v in zip(a, b))
    dx = math.sqrt(sum((u - mx) ** 2 for u in a))
    dy = math.sqrt(sum((v - my) ** 2 for v in b))
    return 0.0 if dx == 0 or dy == 0 else num / (dx * dy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=40)
    ap.add_argument("--dev-seed", type=int, default=42 + 88831)
    ap.add_argument("--width", type=int, default=16)
    ap.add_argument("--tag", default="dev42_v04")
    args = ap.parse_args()

    pathlib.Path(args.out).mkdir(parents=True, exist_ok=True)
    dev = args.device
    torch.manual_seed(args.dev_seed)

    # ---- heldout guard (code-level) ----
    # load_split() creates the tagged split; if the caller passes the
    # consumed heldout tag/path, its digest will match and we refuse.
    held_p = pathlib.Path(args.out) / f"{args.tag}.json"
    if held_p.exists():
        d = sha256_file(held_p)
        if d.startswith(HELDOUT_DIGEST):
            raise SystemExit(
                f"REFUSED: {held_p} matches consumed heldout {HELDOUT_DIGEST}... "
                "heldout40_v04 is quarantined forever.")
    tasks = load_split(args.out, args.dev_n, args.dev_seed, args.tag)
    split_p = pathlib.Path(args.out) / f"{args.tag}.json"
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

    results: list[dict] = []
    e_all: list[float] = []
    y_all: list[int] = []
    t0 = time.time()
    for i, t in enumerate(tasks):
        seq0, r0 = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=args.width, beta_priority=0.0)
        seq1, r1 = dec.decode_cegis_beam(
            z0[i:i + 1], sp[i:i + 1], beam_width=args.width, beta_priority=BETA)
        p0 = sandbox(detokenize(seq0), t["tests"])
        p1 = sandbox(detokenize(seq1), t["tests"])
        # candidate-level energy/outcome association (both arms, final top-4)
        for seq_c, e_c in r0["final_candidates"] + r1["final_candidates"]:
            if e_c is not None and seq_c:
                e_all.append(e_c)
                y_all.append(sandbox(detokenize(seq_c), t["tests"]))
        results.append({
            "task": t["name"], "fp": fp_of(t),
            "beta0": {"seq": seq0, "pass": p0, "score": r0["best_score"],
                      "energy": r0["best_energy"]},
            "beta1": {"seq": seq1, "pass": p1, "score": r1["best_score"],
                      "energy": r1["best_energy"]},
            "traj_changed": seq0 != seq1 or r0["best_score"] != r1["best_score"],
            "first_divergence": next((j for j, (a, b) in
                                      enumerate(zip(seq0, seq1)) if a != b),
                                     min(len(seq0), len(seq1))),
        })
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{args.dev_n}] t={time.time() - t0:.0f}s",
                  flush=True)

    # ---- aggregates ----
    n = len(results)
    a_pass = sum(r["beta0"]["pass"] for r in results)
    b_pass = sum(r["beta1"]["pass"] for r in results)
    eng = sum(r["traj_changed"] for r in results) / n
    both = sum(1 for r in results if r["beta0"]["pass"] and r["beta1"]["pass"])
    a_only = sum(1 for r in results if r["beta0"]["pass"] and not r["beta1"]["pass"])
    b_only = sum(1 for r in results if r["beta1"]["pass"] and not r["beta0"]["pass"])
    neither = n - both - a_only - b_only
    mcnemar = _mcnemar_two_sided(b_only, a_only)
    delta = (b_pass - a_pass) / n

    ast0 = sum(1 for r in results
               if _ast_ok(detokenize(r["beta0"]["seq"]))) / n
    ast1 = sum(1 for r in results
               if _ast_ok(detokenize(r["beta1"]["seq"]))) / n

    rho = _spearman_raw(e_all, y_all) if e_all else 0.0

    report = {
        "run": "system1_v042_cegis_beam",
        "ckpt": args.ckpt, "ckpt_sha": sha256_file(pathlib.Path(args.ckpt)),
        "dev_split": {"tag": args.tag, "n": args.dev_n, "seed": args.dev_seed,
                      "sha256": split_digest},
        "beta": BETA, "width": args.width,
        "n_tasks": n,
        "pass": {"beta0": a_pass, "beta1": b_pass,
                 "delta": round(delta, 4)},
        "transitions": {"both": both, "beta0_only": a_only,
                        "beta1_only": b_only, "neither": neither},
        "mcnemar_p": round(mcnemar, 4),
        "engagement": {"traj_changed_frac": round(eng, 4),
                       "mean_first_divergence": round(
                           sum(r["first_divergence"] for r in results) / n, 2)},
        "ast_valid_rate": {"beta0": round(ast0, 4), "beta1": round(ast1, 4)},
        "energy_assoc": {"n_pairs": len(e_all),
                         "spearman_raw": round(rho, 4)},
        "runtime_s": round(time.time() - t0, 1),
    }

    # ---- gates ----
    engagement_gate = eng > 0.0
    validity_gate = (ast0 >= 0.9 and ast1 >= 0.9 and rho > 0.0
                     and len(e_all) >= 20)
    efficacy_gate = b_only > a_only
    promotion_gate = (efficacy_gate and mcnemar < 0.05
                      and delta >= 0.10 and validity_gate)
    report["gates"] = {
        "engagement_gate_pass": bool(engagement_gate),
        "validity_gate_pass": bool(validity_gate),
        "efficacy_gate_pass": bool(efficacy_gate),
        "promotion_gate_pass": bool(promotion_gate),
        "classification": ("PROMOTED" if promotion_gate
                           else ("ENGAGED_DIAGNOSTIC_NOT_PROMOTED"
                                 if engagement_gate else "NOT_ENGAGED")),
        "power_note": "n=40: underpowered; McNemar can only reach "
                      "p<0.05 with >=5 discordant pairs.",
    }
    report["heldout_guard"] = {
        "consumed_heldout_digest": HELDOUT_DIGEST,
        "quarantine": True,
    }

    print("EVAL:", json.dumps(report, indent=2), flush=True)
    pathlib.Path(args.out + "/eval_cegis.json").write_text(
        json.dumps(report, indent=2))
    pathlib.Path(args.out + "/eval_cegis_results.json").write_text(
        json.dumps(results, indent=2))
    print(f"RECEIPT {args.out}/eval_cegis.json", flush=True)


def _ast_ok(code: str) -> bool:
    try:
        import ast
        ast.parse(code)
        return True
    except Exception:
        return False


if __name__ == "__main__":
    main()
