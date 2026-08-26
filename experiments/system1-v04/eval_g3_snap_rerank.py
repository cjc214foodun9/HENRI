"""G3 snapped-rerank substrate evaluator (System-1 v0.5.x 13-family DSL).

Pre-registered 2026-08-25 (henri-g3-snap-rerank-prereg-20260825-001,
audit bdae5bda70e46379). Parent: G2 result 2cdd5e33b583a911.

Carrier: task prompt + signature (spec) -> Channel-T wave query;
candidate bodies (generated pre-verification) -> Channel-T wave keys;
relational per-block cosine scores -> arms A/B/C (baseline order /
continuous reorder / snapped routing) -> CEGIS-first admission on disjoint
verifier tests -> disjoint outcome tests.

Leakage guard: provenance_scan forbids code/tests/fid/outcome/answer fields
in the reranker input. ONLY prompt + signature + generated candidate bodies.

Endpoint (ceiling 1.0/1.0 DSL): outcome preservation + paired verifier-call
reduction (bootstrap CI lb > 0) + per-family preservation. Engagement
(reorder fraction, score spread, first-passing-rank movement) is a gate.

Modes:
  --mode bridge   disposable 1-task bridge kill test (no split)
  --mode dev      disposable dev11_g3 (n=52, seed 43126) — engagement gates
  --mode seal     generation-only heldout seal (no checkpoint)
  --mode heldout  single-use heldout56_g3 (n=520, seed 73126) with --expect-sha
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import pathlib
import sys
import time

import torch

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent.parent))  # repo root for universal_wave_harness

from system1_kernel_v041_energy_refactored import (  # noqa: E402
    TOK2ID, System1KernelV04, KernelV04Config, detokenize, tokenize_code)
try:  # v0.5.5 substrate (egress1-run) fallback
    from system1_kernel_v05_ast_skeleton import System1KernelV05  # noqa: E402
except ImportError:  # pragma: no cover
    from system1_kernel_v055_ast_skeleton import System1KernelV05  # type: ignore
from train_system1_kernel_v04 import (  # noqa: E402
    gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens, sha256_file)
from train_v051_discriminator import (  # noqa: E402
    N_VERIFIER, N_OUTCOME)
try:
    from eval_v054_heldout import (  # noqa: E402
        CONSUMED_DIGESTS, build_split_stratified, _cegis_first,
        _task_bootstrap_cis, _mcnemar_two_sided)
except ImportError:  # pragma: no cover - v0.5.5 substrate
    from eval_v055_heldout import (  # type: ignore
        CONSUMED_DIGESTS, build_split_stratified, _cegis_first,
        _task_bootstrap_cis, _mcnemar_two_sided)

from universal_wave_harness.snap_rerank import (  # noqa: E402
    TargetLeakageError, dead_keys, implementation_sha256, is_enabled,
    mismatched_keys, order_baseline, order_continuous, order_snapped,
    provenance_scan, write_item_row)
from universal_wave_harness.ingress.text import TextWaveAdapter  # noqa: E402
from universal_wave_harness.lexical_snap import (  # noqa: E402
    DEFAULT_TAU, pre_snap_stats, scores_for)

# ---- G3 guard: all consumed/claimed digests (extends eval_v054 list) ----
G3_CONSUMED = list(CONSUMED_DIGESTS) + [
    "5e5f4a00",  # heldout51_v052
    "a09bf275",  # heldout52_v054
    "529e5ddc",  # heldout54_egress1 (QUARANTINED, arms executed)
    "ec2e1cfd",  # heldout55_egress1 (consumed, verdict COST_EFFECTIVE)
    "59f2b851",  # heldout55_egress2 (sealed for egress2 carrier; refused)
    "4a6b9b94",  # egress2 smoke split (disposable, consumed)
    "fae844ec",  # dev11_g3 (disposable dev smoke, consumed 2026-08-26)
    "14d564ba",  # heldout56_g3 QUARANTINED (mis-index harness defect:
                 # outcome tested codes[idx] not codes[order[idx]]; arms
                 # executed; replaced by heldout57_g3 per protocol)
    "e3f2d131",  # dev12_g3 (disposable re-verify of fixed evaluator,
                 # consumed 2026-08-26)
]


def _task_spec_text(t: dict) -> str:
    """Leakage-safe query text: prompt + signature line ONLY (spec)."""
    sig = t["code"].splitlines()[0].strip()
    return (t.get("prompt", "") + "\n" + sig).strip()


def _candidate_pool(v05: System1KernelV05, z0: torch.Tensor,
                    sp: torch.Tensor, t: dict, budget: int) -> list[dict]:
    cands = v05.generate_skeleton_candidates(
        z0, sp, t, top_k=budget, use_energy=False)
    # dedupe by code (generator already dedupes; defensive)
    seen: set[str] = set()
    out = []
    for c in cands:
        if c["code"] in seen:
            continue
        seen.add(c["code"])
        out.append(c)
    return out


def _encode(adapter: TextWaveAdapter, texts: list[str],
            device: str) -> torch.Tensor:
    return torch.stack([adapter.encode(tx, source_uri="g3",
                                       item_id=str(i)).wave.to(device)
                        for i, tx in enumerate(texts)])


def run_bridge(adapter: TextWaveAdapter, device: str, seed: int) -> dict:
    """Disposable bridge kill test on one generated task."""
    rng = __import__("random").Random(seed)
    t = gen_task(rng, fid=0)
    q_text = _task_spec_text(t)
    # generate a pool via the skeleton grammar (deterministic)
    import torch as _t
    _t.manual_seed(seed)
    # minimal pool: enumerate grammar rules directly
    from system1_kernel_v05_ast_skeleton import SkeletonGrammar
    g = SkeletonGrammar(n_rules=13)
    codes = []
    for r in range(13):
        if g.RULES[r][2] != t["nargs"]:
            continue
        code = g.instantiate(r, t["name"], ["xs", "t1", "t2"][:t["nargs"]])
        if code and TOK2ID["UNK"] not in tokenize_code(code):
            codes.append(code)
    if len(codes) < 2:
        return {"verdict": "BLOCKED_MISSING_CANDIDATE_FRAME",
                "error": "pool < 2"}
    q = adapter.encode(q_text, source_uri="g3-bridge",
                       item_id="q").wave.to(device)
    keys = _encode(adapter, codes, device)
    scores = scores_for(q, keys)
    st = pre_snap_stats(scores, DEFAULT_TAU)
    order_a = order_baseline(len(codes))
    order_b = order_continuous(scores, order_a)
    order_c = order_snapped(scores, DEFAULT_TAU, order_a)
    kd = dead_keys(keys, device)
    st_d = pre_snap_stats(scores_for(q, kd), DEFAULT_TAU)
    km = mismatched_keys(keys, seed=seed, device=device)
    order_mm = order_continuous(scores_for(q, km), order_a)
    return {
        "verdict": "BRIDGE_OK" if (
            len(set(codes)) > 1
            and float(scores.std().item()) > 1e-9
            and (order_b != order_a or order_c != order_a)
            and not (st_d["p_top1"] > 0.99 or st_d["s_margin"] > 1e-6)
            and order_mm != order_a
        ) else "BRIDGE_FAIL",
        "n_candidates": len(codes),
        "unique": len(set(codes)),
        "scores_std": float(scores.std().item()),
        "reordered_b": order_b != order_a,
        "reordered_c": order_c != order_a,
        "dead_p_top1": st_d["p_top1"],
        "mismatched_reordered": order_mm != order_a,
        "pre_snap": st,
    }


def run_split(args, tasks, split_sha: str, adapter: TextWaveAdapter,
              dev: str) -> dict:
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(dev)
    st = torch.load(args.ckpt, map_location=dev)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05 = System1KernelV05(backbone, num_rules=13).to(dev)
    v05.eval()
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN AUDIT FAILED: {trainable}")

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(dev))
    sp = sig_matrix(backbone, tasks, 16, dev)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    items_p = out / "per_task.jsonl"
    if items_p.exists():
        items_p.unlink()

    n = len(tasks)
    arms = {"A": [], "B": [], "C": []}
    rows = []
    reordered_b = reordered_c = 0
    all_spread = True
    all_dead_safe = True
    t0 = time.time()

    for i, t in enumerate(tasks):
        # leakage scan on the input set (prompt + spec only)
        try:
            provenance_scan({"prompt": t.get("prompt", ""),
                             "signature": t["code"].splitlines()[0]})
        except TargetLeakageError as exc:
            raise SystemExit(str(exc))
        for c in _candidate_pool(v05, z0[i:i + 1], sp[i:i + 1], t,
                                 args.budget):
            provenance_scan({"candidate": c["code"]})

        pool = _candidate_pool(v05, z0[i:i + 1], sp[i:i + 1], t, args.budget)
        codes = [c["code"] for c in pool]
        if len(codes) < 2:
            rows.append({"task": t["name"], "fid": t["fid"],
                         "blocked": "EMPTY_POOL"})
            write_item_row(str(items_p), rows[-1])
            continue

        q_text = _task_spec_text(t)
        q = adapter.encode(q_text, source_uri="g3",
                           item_id="q").wave.to(dev)
        keys = _encode(adapter, codes, dev)
        scores = scores_for(q, keys)
        st = pre_snap_stats(scores, DEFAULT_TAU)
        base = order_baseline(len(codes))
        oA = base
        oB = order_continuous(scores, base)
        oC = order_snapped(scores, DEFAULT_TAU, base)
        if not is_enabled():
            # beta=0 identity gate (default-OFF): B/C reproduce A exactly.
            oB = oA
            oC = oA

        kd = dead_keys(keys, dev)
        st_d = pre_snap_stats(scores_for(q, kd), DEFAULT_TAU)
        km = mismatched_keys(keys, seed=args.seed + i, device=dev)
        order_mm = order_continuous(scores_for(q, km), base)

        if oB != oA:
            reordered_b += 1
        if oC != oA:
            reordered_c += 1
        if float(scores.std().item()) <= 1e-9:
            all_spread = False
        if st_d["p_top1"] > 0.99 or st_d["s_margin"] > 1e-6:
            all_dead_safe = False

        ver = t["verifier_tests"]
        out_t = t["outcome_tests"]
        row = {"task": t["name"], "fid": t["fid"]}
        for arm, order in (("A", oA), ("B", oB), ("C", oC)):
            pool_ordered = [(codes[j], j) for j in order]
            idx, calls = _cegis_first(pool_ordered, ver,
                                      budget=args.budget)
            # CRITICAL: idx is the position in the REORDERED pool; the
            # admitted candidate's ORIGINAL index is order[idx]. The
            # outcome/ast evaluation MUST use codes[order[idx]], not
            # codes[idx] (mis-index defect found 2026-08-26: heldout56_g3
            # REGRESSION verdict was a harness artifact; quarantined).
            admitted_code = codes[order[idx]] if idx >= 0 else None
            outcome = sandbox(admitted_code, out_t) if idx >= 0 else 0
            ast_ok = 1 if (idx >= 0 and admitted_code is not None
                           and TOK2ID["UNK"] not in
                           tokenize_code(admitted_code)) else 0
            row[arm] = {"admit": order[idx] if idx >= 0 else -1,
                        "calls": calls, "outcome": outcome,
                        "ast_valid": ast_ok,
                        "first_pass_rank": idx}
            arms[arm].append(outcome)
        row["scores"] = scores.tolist()
        row["scores_std"] = float(scores.std().item())
        row["order_A"] = oA
        row["order_B"] = oB
        row["order_C"] = oC
        row["order_mismatched"] = order_mm
        row["pre_snap"] = st
        row["dead_p_top1"] = st_d["p_top1"]
        row["dead_s_margin"] = st_d["s_margin"]
        row["split_sha256"] = split_sha
        row["impl_sha256"] = implementation_sha256()
        rows.append(row)
        write_item_row(str(items_p), row)
        if (i + 1) % 10 == 0:
            print(f"[{i + 1}/{n}] t={time.time() - t0:.0f}s", flush=True)

    def rate(a: str) -> float:
        return sum(arms[a]) / n if n else 0.0

    def call_stats(a: str) -> dict:
        v = sorted([r[a]["calls"] for r in rows if "calls" in r[a]])
        m = len(v)
        return {"mean": sum(v) / m if m else 0.0,
                "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0,
                "max": v[-1] if m else 0}

    def ci_reduction(a: str) -> tuple:
        """calls[A] - calls[arm]; positive = arm cheaper (reduction)."""
        return _task_bootstrap_cis(
            [float(r["A"]["calls"] - r[a]["calls"]) for r in rows
             if "calls" in r["A"] and "calls" in r[a]], seed=args.seed + 7)

    def family_min(a: str) -> dict:
        fam: dict[int, list] = {}
        for r in rows:
            if "calls" not in r[a]:
                continue
            fam.setdefault(r["fid"], []).append(r[a]["outcome"])
        return {str(f): sum(v) / len(v) for f, v in sorted(fam.items())}

    paired = {"B_vs_A": _mcnemar_two_sided(
        sum(1 for r in rows if "calls" in r["B"] and r["B"]["outcome"]
            and not r["A"]["outcome"]),
        sum(1 for r in rows if "calls" in r["B"] and r["A"]["outcome"]
            and not r["B"]["outcome"])),
        "C_vs_A": _mcnemar_two_sided(
        sum(1 for r in rows if "calls" in r["C"] and r["C"]["outcome"]
            and not r["A"]["outcome"]),
        sum(1 for r in rows if "calls" in r["C"] and r["A"]["outcome"]
            and not r["C"]["outcome"]))}

    result = {
        "mode": args.mode, "n": n, "split_sha256": split_sha,
        "arms": {a: {"outcome_pass_rate": rate(a), "calls": call_stats(a),
                     "family_min": family_min(a)} for a in ("A", "B", "C")},
        "engagement": {"reordered_fraction_B": reordered_b / n,
                       "reordered_fraction_C": reordered_c / n,
                       "all_score_spread": all_spread,
                       "all_dead_safe": all_dead_safe},
        "paired": paired,
        "call_reduction_CI": {"B_minus_A": ci_reduction("B"),
                              "C_minus_A": ci_reduction("C")},
        "runtime_s": time.time() - t0,
        "verdict": "pending",
    }

    if args.mode == "dev":
        # Engagement gates PLUS outcome preservation (this catches the
        # mis-index class of defect in dev, before any heldout exposure).
        out_ok = (abs(rate("B") - rate("A")) <= 0.02
                  and abs(rate("C") - rate("A")) <= 0.02
                  and paired["B_vs_A"] >= 0.05
                  and paired["C_vs_A"] >= 0.05)
        eng = (result["engagement"]["reordered_fraction_B"] > 0
               and result["engagement"]["all_score_spread"]
               and result["engagement"]["all_dead_safe"])
        result["verdict"] = ("DEV11_G3_ENGAGEMENT_PASS" if (eng and out_ok)
                             else "FALSIFIED_OUTCOME_NOT_PRESERVED"
                             if eng and not out_ok
                             else "FALSIFIED_NO_ENGAGEMENT")
    elif args.mode == "heldout":
        cb = result["call_reduction_CI"]["C_minus_A"]
        outcome_same = abs(rate("C") - rate("A")) <= 0.02 \
            and paired["C_vs_A"] >= 0.05
        fam_same = all(
            abs(family_min("C").get(k, 0) - family_min("A").get(k, 0)) <= 0.1
            for k in family_min("A"))
        if outcome_same and fam_same and cb[0] > 0:
            result["verdict"] = "COST_EFFECTIVE_REORDER"
        elif outcome_same and fam_same:
            result["verdict"] = "ENGAGED_NO_EFFICACY"
        else:
            result["verdict"] = "REGRESSION"
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["bridge", "dev", "seal", "heldout"],
                    required=True)
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=52)
    ap.add_argument("--seed", type=int, default=43126)
    ap.add_argument("--tag", default="dev11_g3")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--expect-sha", default="")
    ap.add_argument("--expect-flag", action="store_true")
    args = ap.parse_args()

    if args.expect_flag and not is_enabled():
        raise SystemExit("G3_FLAG_OFF: HENRI_G3_SNAP_RERANK=1 required")

    if args.mode == "seal":
        if args.ckpt:
            raise SystemExit("SEAL_ONLY: --ckpt not allowed")
        tasks = build_split_stratified(args.out, args.n, args.seed,
                                       args.tag, n_families=13)
        sd = sha256_file(pathlib.Path(args.out) / f"{args.tag}.json")
        if any(sd.startswith(d) for d in G3_CONSUMED):
            raise SystemExit(
                f"INVALID_VERIFIER_REPLAY: {sd[:12]} consumed. REFUSED.")
        fam: dict[int, int] = {}
        for t in tasks:
            fam[t["fid"]] = fam.get(t["fid"], 0) + 1
        rec = {"tag": args.tag, "n": len(tasks), "seed": args.seed,
               "sha256": sd, "utc": datetime.datetime.now(
                   datetime.timezone.utc).isoformat(),
               "single_use": True, "stratified_families": fam,
               "generator": "eval_g3_snap_rerank.build_split_stratified "
                            "(13 x per_family)",
               "checked_consumed": len(G3_CONSUMED)}
        (pathlib.Path(args.out) / f"seal_{args.tag}.json").write_text(
            json.dumps(rec, indent=2))
        print(f"SEALED {args.tag} n={len(tasks)} seed={args.seed} "
              f"sha={sd}", flush=True)
        return

    adapter = TextWaveAdapter(device=args.device)
    dev = args.device

    if args.mode == "bridge":
        res = run_bridge(adapter, dev, seed=args.seed)
        print(json.dumps(res, indent=2), flush=True)
        if res["verdict"] != "BRIDGE_OK":
            raise SystemExit("BRIDGE_KILL: " + res["verdict"])
        return

    split_p = pathlib.Path(args.out) / f"{args.tag}.json"
    if split_p.exists():
        sd = sha256_file(split_p)
        tasks = json.loads(split_p.read_text())
    else:
        tasks = build_split_stratified(args.out, args.n, args.seed,
                                       args.tag, n_families=13)
        sd = sha256_file(split_p)
    if any(sd.startswith(d) for d in G3_CONSUMED):
        raise SystemExit(f"INVALID_VERIFIER_REPLAY: {sd[:12]} REFUSED.")
    if args.expect_sha and not sd.startswith(args.expect_sha):
        raise SystemExit(f"SPLIT_MISMATCH: {sd[:16]} != {args.expect_sha}")

    res = run_split(args, tasks, sd, adapter, dev)
    (pathlib.Path(args.out) / "g3_results.json").write_text(
        json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    print("RECEIPT " + str(pathlib.Path(args.out) / "g3_results.json"),
          flush=True)


if __name__ == "__main__":
    main()
