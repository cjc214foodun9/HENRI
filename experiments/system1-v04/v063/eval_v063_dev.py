"""
System-1 v0.6.3a — Pre-Reasoning Entropy INSTRUMENTATION evaluator (real machinery).
====================================================================================
NO behavioral change. NO gating. NO learning. Default OFF for gating.

Records per task, pre-verifier: Shannon entropy H and H/logK of the REAL
candidate cosine-sim distribution (bridge sims over the frozen carrier's
candidate pool), first-passing rank, verifier calls, outcome, family.
Imports the REAL split builder, verifier (_cegis_first), and AST check
from eval_v0601_dev (same machinery, same semantics).

Answer: does pre-verifier entropy predict first-pass rank or outcome?
Report Spearman rho + bootstrap CI. Verdicts pre-registered:
ENTROPY_PREDICTIVE_ONLY / COST_EFFECTIVE_PRESERVED / NO_EFFECT /
UNSAFE_BYPASS / REGRESSION.

Fresh disposable split only (dev10_v063, seed 70707, n=65, 13x5).
Consumed digests (incl. dev9_v0601 a8a2d7a7...) refused.
"""

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Explicitly enable entropy INSTRUMENTATION for the measurement run.
# This is the v0.6.3a recording path — NOT gating, NOT learning. The
# carrier remains OFF by default (contract C1); this evaluator is the
# explicit opt-in that makes the telemetry real instead of NaN.
os.environ["HENRI_V063A_ENABLE"] = "1"

import torch

# ---- REAL production machinery (identical semantics to v0.6.0.1) ----
from eval_v0601_dev import (  # noqa: E402
    build_split_stratified, _cegis_first, _ast_ok, CONSUMED_DIGESTS)
# Extend the imported guard with v0.6.0.1 consumed digests (dev9_v0601 set).
CONSUMED = list(CONSUMED_DIGESTS) + [
    "a8a2d7a7",  # dev9_v0601 split (v0.6.0.1) — consumed, never replay
    "a303ebd4",  # eval_v0601_dev.json (same run)
    "ba1cc963",  # dev9_v0601 seal bytes
]
from system1_kernel_v041_energy_refactored import (  # noqa: E402
    System1KernelV04, KernelV04Config, TOK2ID, tokenize_code)
from system1_kernel_v055_ast_skeleton import System1KernelV05  # noqa: E402
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens)
from zone_c_bridge_v0601 import CandidateRetrievalRanker  # noqa: E402
from v063_entropy_gate_carrier import (  # noqa: E402
    candidate_score_distribution, shannon_entropy_nats, normalized_entropy)

VERDICTS = [
    "ENTROPY_PREDICTIVE_ONLY",
    "COST_EFFECTIVE_PRESERVED",
    "NO_EFFECT",
    "UNSAFE_BYPASS",
    "REGRESSION",
]


def _bootstrap_ci_rho(xs, ys, seed=1234, n_boot=200):
    """Bootstrap 95% CI for Spearman rho."""
    import scipy.stats as st
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        idx = [rng.randrange(len(xs)) for _ in range(len(xs))]
        rh, _ = st.spearmanr([xs[i] for i in idx], [ys[i] for i in idx])
        if rh == rh:
            bs.append(rh)
    if not bs:
        return None, None
    bs.sort()
    lo = bs[max(0, int(0.025 * len(bs)) - 1)]
    hi = bs[min(len(bs) - 1, int(0.975 * len(bs)) - 1)]
    return round(lo, 3), round(hi, 3)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dev-n", type=int, default=65)
    ap.add_argument("--dev-seed", type=int, default=70707)
    ap.add_argument("--tag", default="dev10_v063")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--expect-sha", default="")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.dev_seed)
    print(f"device={dev}")

    # ---- fresh disposable split (exact 13x5 stratification) ----
    split_p = out / f"{args.tag}.json"
    if split_p.exists():
        sd = hashlib.sha256(split_p.read_bytes()).hexdigest()
        tasks = json.loads(split_p.read_text())
    else:
        tasks = build_split_stratified(
            str(out), args.dev_n, args.dev_seed, args.tag, n_families=13)
        sd = hashlib.sha256(split_p.read_bytes()).hexdigest()

    if any(sd.startswith(d) for d in CONSUMED):
        raise SystemExit(
            f"INVALID_VERIFIER_REPLAY: split sha {sd[:12]} matches a "
            f"consumed digest. REFUSED.")
    if args.expect_sha and not sd.startswith(args.expect_sha):
        raise SystemExit(
            f"SPLIT_MISMATCH: {sd[:16]} != pinned {args.expect_sha}. REFUSED.")
    print(f"split {args.tag} sha {sd[:16]} n={len(tasks)}")

    # ---- frozen carrier ----
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05_13.eval()
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable}")

    ranker = CandidateRetrievalRanker(enabled=True, beta=0.0, device=dev)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    per_task: list[dict] = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        ver_tests = t["verifier_tests"]
        out_tests = t["outcome_tests"]

        # shared pool: frozen carrier, uniform order (B13)
        cands13 = v05_13.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget,
            use_energy=False)
        pool13 = [(c["code"], c["rule_id"]) for c in cands13]

        # ---- pre-verifier entropy signal (REAL cosine sims) ----
        tv = ranker.task_repr(z0[i:i + 1], sp[i:i + 1], v05_13)
        cv = [ranker.candidate_repr(c["code"], v05_13, dev)
              for c in cands13]
        sims = ranker.sim_scores(tv, cv) if cv else torch.tensor([])
        probs, h, h_norm, k = candidate_score_distribution(sims)

        # ---- REAL verifier scan (first-pass rank = calls) ----
        idx, call = _cegis_first(pool13, ver_tests, budget=args.budget)

        outcome = 0
        if idx >= 0:
            code = pool13[idx][0]
            outcome = sandbox(code, out_tests)

        row = {
            "task": t["name"], "fp": fp_of(t), "fid": t["fid"],
            "pool_len": len(pool13),
            "entropy": {
                "H_nats": h, "H_norm": h_norm, "K": k,
                "p1": probs[0] if probs else None,
                "sims": [round(float(x), 6) for x in sims.tolist()],
            },
            "calls": call,
            "first_pass_rank": (idx + 1) if idx >= 0 else None,
            "admit": idx,
            "outcome": outcome,
            "ast_valid": _ast_ok(code) if idx >= 0 else 0,
        }
        per_task.append(row)
        if (i + 1) % 5 == 0:
            print(f"[{i + 1}/{len(tasks)}] t={time.time() - t0:.0f}s",
                  flush=True)

    n = len(per_task)
    outcome_pass = sum(r["outcome"] for r in per_task)
    calls = [r["calls"] for r in per_task]
    fpr = [r["first_pass_rank"] for r in per_task
           if r["first_pass_rank"] is not None]
    fam = Counter()
    for r in per_task:
        fam[r["fid"]] += r["outcome"]
    min_fam = min(fam.values()) if fam else 0

    # entropy predictiveness: H vs first-pass rank; H vs rank-1-pass
    # Strict alignment: only tasks with BOTH a valid entropy AND a
    # first-pass rank enter the correlation (a task with an empty pool has
    # no rank and must be excluded from BOTH arrays, not just one).
    paired = [(r["entropy"]["H_nats"], r["first_pass_rank"])
              for r in per_task
              if r["entropy"] and r["entropy"]["H_nats"] is not None
              and r["first_pass_rank"] is not None]
    hs = [x[0] for x in paired]
    ranks = [x[1] for x in paired]
    rank1 = [1 if x[1] == 1 else 0 for x in paired]
    hs_norm = [r["entropy"]["H_norm"] for r in per_task
               if r["entropy"] and r["entropy"]["H_norm"] is not None
               and r["first_pass_rank"] is not None]

    rho = None
    ci = None
    rho_rank1 = None
    ci_rank1 = None
    if len(hs) >= 8:
        import scipy.stats as st
        rho, _ = st.spearmanr(hs, ranks)
        ci = _bootstrap_ci_rho(hs, ranks)
        rho_rank1, _ = st.spearmanr(hs, rank1)
        ci_rank1 = _bootstrap_ci_rho(hs, rank1)
        rho = round(rho, 3)
        rho_rank1 = round(rho_rank1, 3)
    elif hs:
        # instrumentation-dead guard: entropy present but < 8 valid pairs
        raise SystemExit(
            "INSTRUMENTATION_DEAD: entropy not recorded for >= 8 tasks "
            "(check HENRI_V063A_ENABLE / carrier path). REFUSED to emit "
            "a verdict on empty signal.")

    verdict = "NO_EFFECT"
    if outcome_pass < n:
        verdict = "REGRESSION"
    elif ci is not None and ci[0] > 0.2:
        verdict = "ENTROPY_PREDICTIVE_ONLY"

    result = {
        "tag": args.tag,
        "n": n,
        "seed": args.dev_seed,
        "budget": args.budget,
        "device": dev,
        "split_sha": sd,
        "outcome_pass": round(outcome_pass / n, 4),
        "min_family_support": min_fam,
        "mean_calls": round(sum(calls) / n, 3),
        "call_dist": dict(sorted(Counter(calls).items())),
        "first_pass_rank_dist": dict(sorted(Counter(fpr).items())),
        "spearman_H_vs_rank": rho,
        "spearman_ci95": ci,
        "spearman_H_vs_rank1": rho_rank1,
        "spearman_ci95_rank1": ci_rank1,
        "entropy_H_range": [round(min(hs), 4), round(max(hs), 4)]
        if hs else None,
        "entropy_Hnorm_range": [round(min(hs_norm), 4),
                                round(max(hs_norm), 4)]
        if hs_norm else None,
        "verdict": verdict,
        "consumed_guard_count": len(CONSUMED),
    }
    (out / "eval_v063_dev.json").write_text(json.dumps(result, indent=2))
    (out / "eval_v063_per_task.json").write_text(
        json.dumps({"per_task": per_task}, indent=1))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
