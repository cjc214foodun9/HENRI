"""
Egress-1 A/B evaluator — Arm A (frozen 13-rule DSL baseline) vs Arm B (backbone-conditioned).
=============================================================================================
Contract: egress1_contract.md (sha 7fcc9361...); approval 2b30c69f; prereg 75e4f911.
Split: heldout54_egress1 (520 = 13x40, sha 529e5ddc...), single-use, pinned via --expect-sha.

Arm A: System1KernelV055 generate_skeleton_candidates (top_k=budget, use_energy=False),
       uniform rule order (run18 carrier). First verifier-pass admits (CEGIS-first).
Arm B: same generator; pool = base candidates REORDERED by family prior sim_f = e_task . P_f
       (top-2 families first, stable within family) + up to E extra in-grammar candidates
       from the same generator beyond budget for the top-2 families (structural expansion),
       capped at budget + 2E. beta=0 identity arm must reproduce Arm A byte-identically.

Verdicts (fixed before evaluation): BLOCKED / NO_EFFECT / COST_EFFECTIVE / SUPPORT_RESTORED /
CAPABILITY_PROMOTED / FALSIFIED_NO_EXTERNAL_GAIN / REGRESSION.
Boundaries: no backbone fine-tuning; no CartPole coupling; no AAII/VLA claim.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, math, pathlib, random, sys, time

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import torch

from system1_kernel_v041_energy_refactored import (
    TOK2ID, System1KernelV04, detokenize, KernelV04Config, tokenize_code)
from system1_kernel_v042_cegis_beam import CEGISBeamPriorityDecoder
from system1_kernel_v055_ast_skeleton import System1KernelV05
from train_system1_kernel_v04 import gen_task, sandbox, fp_of, sig_ids, sig_matrix, pad_tokens
from eval_v055_heldout import _cegis_first, _mcnemar_two_sided, _task_bootstrap_cis, sha256_file


def sha256_file(path: str) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _ast_ok(code: str) -> int:
    return 1 if TOK2ID["UNK"] not in tokenize_code(code) else 0


def _prototypes(backbone, fam_codes: list[str]) -> torch.Tensor:
    """P_f = L2norm(embed_text(canonical_body_f)) for f in 0..12."""
    protos = []
    for code in fam_codes:
        e, _t = backbone.embed_text(code)
        protos.append(e)
    return torch.stack(protos)  # [13, D]


def _family_prior(e_task: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
    return e_task @ P.T  # [13]


def _family_variants(fid: int, rng: random.Random, n: int) -> list[str]:
    """Explicit family-variant instantiation: gen_task bodies with varied args.

    The skeleton generator saturates at 4–9 unique candidates (OBSERVED
    2026-08-24); expansion needs structural support, so we build variants
    from the family's own _rand_args/_expected space. Each variant is a
    tokenizer-closed, AST-valid body for family fid.
    """
    out = []
    seen = set()
    for _ in range(4 * n):
        if len(out) >= n:
            break
        t = gen_task(rng, fid=fid)
        code = t["code"]
        if code in seen:
            continue
        seen.add(code)
        out.append(code)
    return out


def build_pool_b(base_pool: list, sim: torch.Tensor, budget: int, expand: int,
                 fid_top2: list[int], rng: random.Random) -> list:
    """Arm B pool: top-2 family reorder + family-variant structural expansion."""
    top2 = sorted(range(13), key=lambda f: -float(sim[f]))[:2]
    fam_map = [r for (_c, r) in base_pool]
    first = [i for i, r in enumerate(fam_map) if r in top2]
    rest = [i for i, r in enumerate(fam_map) if r not in top2]
    ordered = [base_pool[i] for i in first] + [base_pool[i] for i in rest]
    # expansion: explicit family variants for top-2 families (structural support)
    extra = []
    if expand > 0:
        seen = {c[0] for c in base_pool}
        for f in top2:
            for code in _family_variants(f, rng, expand):
                if len(extra) >= expand:
                    break
                if code not in seen:
                    seen.add(code)
                    extra.append((code, f))
    return (ordered + extra)[: budget + 2 * expand]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, type=pathlib.Path)
    ap.add_argument("--expect-sha", required=True)
    ap.add_argument("--expect-count", type=int, default=520)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--ckpt", required=True, type=pathlib.Path)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budget", type=int, default=64)
    ap.add_argument("--expand", type=int, default=0,
                    help="FINAL SEALED: 0 (reorder-only; grammar cardinality — gen_task bodies fixed per family, no expansion possible)")
    ap.add_argument("--backbone-model-dir", default="/root/models/qwen3vl-8b-0c351dd0")
    ap.add_argument("--backbone-manifest", default="/root/models/qwen3vl-8b-0c351dd0/qwen3vl8b_tree_manifest.json")
    ap.add_argument("--sandbox-mode", choices=["namespace", "container-rlimit"], default="container-rlimit")
    ap.add_argument("--arm", choices=["A", "B", "both"], default="both")
    args = ap.parse_args()

    # ---- split integrity ----
    sd = sha256_file(str(args.split))
    if not sd.startswith(args.expect_sha):
        raise SystemExit(f"SPLIT_MISMATCH {sd[:16]} != {args.expect_sha[:16]} REFUSED")
    tasks = json.loads(args.split.read_text())
    if len(tasks) != args.expect_count:
        raise SystemExit(f"SPLIT_COUNT {len(tasks)} != {args.expect_count} REFUSED")
    print(f"SPLIT_OK n={len(tasks)} sha={sd[:16]}", flush=True)

    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    # ---- kernel + checkpoint (frozen run18 carrier) ----
    torch.manual_seed(82026)
    cfg = KernelV04Config()
    backbone = System1KernelV04(cfg=cfg).to(args.device)
    st = torch.load(args.ckpt, map_location=args.device)
    backbone.load_state_dict(st["model"])
    backbone.eval()
    v05_13 = System1KernelV05(backbone, num_rules=13).to(args.device)
    v05_13.eval()
    dec = CEGISBeamPriorityDecoder(backbone)
    trainable = [n for n, p in backbone.named_parameters() if p.requires_grad]
    if trainable:
        raise SystemExit(f"FROZEN_AUDIT_FAILED {trainable}")
    print(f"KERNEL_LOADED ckpt={args.ckpt} step={st.get('step')}", flush=True)

    # ---- Qwen backbone (Arm B only) ----
    adapter = None
    P = None
    if args.arm in ("B", "both"):
        from henri_backbone_adapter import QwenBackboneAdapter
        adapter = QwenBackboneAdapter(
            model_dir=args.backbone_model_dir, manifest_path=args.backbone_manifest,
            verify_shards=True, device="cuda", dtype=torch.bfloat16)
        adapter.load()
        fam_codes = [gen_task(random.Random(f), f)["code"] for f in range(13)]
        P = _prototypes(adapter, fam_codes).to("cuda")
        print(f"BACKBONE_LOADED {adapter.telemetry.to_dict().get('checkpoint_load_status')}", flush=True)

    z0 = backbone.encode_tokens(
        pad_tokens([sig_ids(t) for t in tasks], 16).to(args.device))
    sp = sig_matrix(backbone, tasks, 16, args.device)

    per_task = []
    calls = {"A": [], "B": []}
    outcome_pass = {"A": [], "B": []}
    ast_valid = {"A": [], "B": []}
    first_rank = {"A": [], "B": []}
    pool_delta = {"added": 0, "reordered": 0, "identical": 0}
    sims = []
    t0 = time.time()

    for i, t in enumerate(tasks):
        row = {"task": t["name"], "fid": t["fid"]}
        ver, ote = t["verifier_tests"], t["outcome_tests"]

        # ---- base pool (shared construction; Arm A uses it directly) ----
        # Frozen run18 carrier: v05_13.generate_skeleton_candidates (rule_id
        # tagged, use_energy=False), deduped, capped at budget. OBSERVED
        # saturation: 9 unique (nargs-1) / 4 unique (nargs-2).
        cands = v05_13.generate_skeleton_candidates(
            z0[i:i + 1], sp[i:i + 1], t, top_k=args.budget, use_energy=False)
        seen = set()
        base = []
        for c in cands:
            if c["code"] in seen:
                continue
            seen.add(c["code"])
            base.append((c["code"], c["rule_id"]))
        base = base[: args.budget]

        if args.arm in ("A", "both"):
            idx_a, call_a = _cegis_first(base, ver, budget=args.budget)
            calls["A"].append(call_a)
            first_rank["A"].append(idx_a)
            if idx_a >= 0:
                code_a = base[idx_a][0]
                outcome_pass["A"].append(sandbox(code_a, ote))
                ast_valid["A"].append(_ast_ok(code_a))
            else:
                outcome_pass["A"].append(0)
                ast_valid["A"].append(0)
            row["A"] = {"admit": idx_a, "calls": call_a, "pool": len(base),
                        "ast_valid": ast_valid["A"][-1]}

        if args.arm in ("B", "both"):
            e_task, _tel = adapter.embed_text(t["prompt"])
            sim = _family_prior(e_task.to("cuda"), P)
            sims.append(sim.detach().cpu().numpy().tolist())
            base_order = [c[0] for c in base]
            rng_b = random.Random(82026 + i)  # deterministic per-task expansion
            pool_b = build_pool_b(base, sim, args.budget, args.expand, None, rng_b)
            b_order = [c[0] for c in pool_b]
            if b_order[: len(base)] == base_order:
                pool_delta["identical"] += 1
            elif b_order[: len(base)] == base_order and len(pool_b) == len(base):
                pool_delta["identical"] += 1
            else:
                pool_delta["reordered"] += 1
                pool_delta["added"] += max(0, len(pool_b) - len(base))
            idx_b, call_b = _cegis_first(pool_b, ver, budget=len(pool_b))
            calls["B"].append(call_b)
            first_rank["B"].append(idx_b)
            if idx_b >= 0:
                code_b = pool_b[idx_b][0]
                outcome_pass["B"].append(sandbox(code_b, ote))
                ast_valid["B"].append(_ast_ok(code_b))
            else:
                outcome_pass["B"].append(0)
                ast_valid["B"].append(0)
            row["B"] = {"admit": idx_b, "calls": call_b, "pool": len(pool_b),
                        "ast_valid": ast_valid["B"][-1]}

        per_task.append(row)
        # INCREMENTAL per-task persistence: an aggregation crash must never
        # lose task-level evidence (OBSERVED 2026-08-24: KeyError in ast_rate
        # lost a full 520-task run of split 529e5ddc; split quarantined).
        with open(out / "per_task.jsonl", "a", encoding="utf-8") as pf:
            pf.write(json.dumps(row) + "\n")
        if (i + 1) % 20 == 0:
            print(f"[{i + 1}/{len(tasks)}] t={time.time() - t0:.0f}s", flush=True)

    n = len(per_task)
    res = {"n": n, "split_sha": sd[:16], "per_task": per_task,
           "pool_delta": pool_delta, "sims": sims}

    def rate(arm):
        return sum(outcome_pass[arm]) / n

    def call_stats(arm):
        v = sorted(calls[arm]); m = len(v)
        if m == 0:
            return {"mean": 0.0, "median": 0.0, "p90": 0.0, "max": 0}
        return {"mean": sum(v) / m, "median": v[m // 2] if m else 0.0,
                "p90": v[int(0.9 * (m - 1))] if m else 0.0, "max": v[-1] if m else 0}

    def family_support(arm):
        fs = {}
        for i, r in enumerate(per_task):
            if arm not in r:
                continue
            fs.setdefault(r["fid"], []).append(outcome_pass[arm][i])
        return {f: (sum(v) / len(v) if v else 0.0) for f, v in fs.items()}

    def ast_rate(arm):
        vals = [r[arm]["ast_valid"] for r in per_task
                if arm in r and r[arm]["admit"] >= 0]
        return (sum(vals) / len(vals)) if vals else 0.0

    res["A"] = {"rate": rate("A"), "calls": call_stats("A"),
                "family_support": family_support("A"), "ast_valid": ast_rate("A"),
                "first_rank": first_rank["A"]}
    res["B"] = {"rate": rate("B"), "calls": call_stats("B"),
                "family_support": family_support("B"), "ast_valid": ast_rate("B"),
                "first_rank": first_rank["B"]}

    # ---- verdict chain (dual-arm only; single-arm = plumbing mode) ----
    if "A" not in res["per_task"][0] or "B" not in res["per_task"][0]:
        res["verdict"] = "BLOCKED"
        res["stats"] = {"reason": "single-arm plumbing run (no paired verdict)"}
        res["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        (out / "egress1_results.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"verdict": res["verdict"],
                          "delta": None, "p": None,
                          "calls_A": res["A"]["calls"], "calls_B": res["B"]["calls"],
                          "pool_delta": pool_delta}, indent=1))
        return 0

    b_c = sum(1 for i in range(n) if outcome_pass["B"][i] and not outcome_pass["A"][i])
    c_b = sum(1 for i in range(n) if outcome_pass["A"][i] and not outcome_pass["B"][i])
    p_mc = _mcnemar_two_sided(b_c, c_b)
    delta = res["B"]["rate"] - res["A"]["rate"]
    lo, hi = _task_bootstrap_cis([outcome_pass["B"][i] - outcome_pass["A"][i]
                                  for i in range(n)], seed=82033)
    call_delta = res["B"]["calls"]["mean"] - res["A"]["calls"]["mean"]
    fam_min = min(min(res["A"]["family_support"].values()), min(res["B"]["family_support"].values()))
    validity_ok = res["A"]["ast_valid"] == 1.0 and res["B"]["ast_valid"] == 1.0
    fam_ok = fam_min >= 0.8

    if not validity_ok or not fam_ok:
        verdict = "REGRESSION"
    elif delta <= 0 and b_c == 0 and c_b == 0 and call_delta == 0:
        verdict = "NO_EFFECT"
    elif b_c == 0 and c_b == 0 and call_delta < 0 and lo <= 0:
        verdict = "COST_EFFECTIVE"
    elif delta > 0 and p_mc < 0.05 and lo > 0 and call_delta <= 0.05 * res["A"]["calls"]["mean"]:
        verdict = "CAPABILITY_PROMOTED"
    elif (res["B"]["rate"] > res["A"]["rate"] or fam_min > min(res["A"]["family_support"].values())):
        verdict = "SUPPORT_RESTORED"
    else:
        verdict = "FALSIFIED_NO_EXTERNAL_GAIN"

    res["verdict"] = verdict
    res["stats"] = {"delta": delta, "b_c": b_c, "c_b": c_b, "mcnemar_p": p_mc,
                    "ci90": [lo, hi], "call_delta_mean": call_delta,
                    "family_min": fam_min, "validity_ok": validity_ok, "family_ok": fam_ok}
    res["utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    (out / "egress1_results.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({"verdict": verdict, "delta": delta, "p": p_mc,
                      "calls_A": res["A"]["calls"], "calls_B": res["B"]["calls"],
                      "pool_delta": pool_delta}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
